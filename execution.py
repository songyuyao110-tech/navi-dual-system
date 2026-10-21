from __future__ import annotations

from dataclasses import dataclass
import json
from threading import RLock
from typing import Callable

from jsonschema import Draft202012Validator

from registry import RegisteredTool, RegistrySnapshot
from model_clients import CONTROL_TOOLS, ModelError
from task_context import Task, TaskLedger


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


class Runtime:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = RLock()
        self.state: dict = {"route_active": False}
        self.handlers: dict[str, Callable[[RegisteredTool, dict], dict]] = {
            "mock": self._mock,
            "mock_error": self._mock_error,
        }

    def validate(
        self, snapshot: RegistrySnapshot, disclosed: frozenset[str], call: ToolCall
    ) -> RegisteredTool:
        if call.name not in disclosed:
            raise ExecutionError(f"Tool is not disclosed: {call.name}")
        if call.name not in snapshot.tools:
            raise ExecutionError(f"Tool is not registered: {call.name}")
        tool = snapshot.tools[call.name]
        errors = list(Draft202012Validator(tool.parameters).iter_errors(call.arguments))
        if errors:
            raise ExecutionError(f"Invalid arguments for {call.name}: {errors[0].message}")
        if tool.binding not in self.handlers:
            raise ExecutionError(f"Execution binding unavailable for {call.name}")
        return tool

    def execute(
        self, snapshot: RegistrySnapshot, disclosed: frozenset[str], call: ToolCall
    ) -> dict:
        with self.lock:
            tool = self.validate(snapshot, disclosed, call)
            result = self.handlers[tool.binding](tool, call.arguments)
            self.calls.append(
                {"version": snapshot.version, "tool": call.name, "arguments": call.arguments, "result": result}
            )
            return result

    def _mock(self, tool: RegisteredTool, arguments: dict) -> dict:
        if tool.name in {"navi_to_special_poi", "navi_start_route"}:
            self.state["route_active"] = True
            self.state["destination"] = arguments.get("end_name", arguments.get("poi_type"))
        elif tool.name == "navi_cancel_route":
            self.state["route_active"] = False
        return {
            "status": "success",
            "mock": True,
            "tool": tool.name,
            "facts": {"arguments": arguments, "navigation": dict(self.state)},
        }

    @staticmethod
    def _mock_error(tool: RegisteredTool, arguments: dict) -> dict:
        raise ExecutionError(f"Injected explicit failure: {tool.name}")


def run_fast(
    task: Task, ledger: TaskLedger, runtime: Runtime,
    planner: Callable[[dict, list[dict]], dict],
) -> None:
    with ledger.lock:
        revision = task.revision
        if task.status == "cancelled":
            return
        task.status = "planning"
        brief = task.brief()
        schemas = task.snapshot.schemas(task.disclosed)
    try:
        plan = planner(brief, schemas)
        with ledger.lock:
            if not ledger.current(task, revision):
                return
            task.plans.append(plan)
            if plan["question"] is not None:
                task.status = "waiting_input"
                task.question = plan["question"]
                task.asked_questions.append(task.question)
                return
        calls = [ToolCall(step["tool"], step["arguments"]) for step in plan["steps"]]
        if not calls:
            raise ModelError("Fast planner returned no steps")
        for call in calls:
            runtime.validate(task.snapshot, task.disclosed, call)
        for call in calls:
            with ledger.lock:
                if not ledger.current(task, revision):
                    return
                task.status = "running"
                task.inflight = True
            try:
                result = runtime.execute(task.snapshot, task.disclosed, call)
                with ledger.lock:
                    task.completed.append(
                        {"tool": call.name, "arguments": call.arguments, "result": result}
                    )
            finally:
                with ledger.lock:
                    task.inflight = False
                    ledger.condition.notify_all()
        with ledger.lock:
            if ledger.current(task, revision):
                task.status = "done"
    except (ExecutionError, ModelError, ValueError) as error:
        with ledger.lock:
            if ledger.current(task, revision):
                task.status = "needs_slow"
                task.route = "slow"
                task.error = str(error)


def run_slow(
    task: Task, ledger: TaskLedger, runtime: Runtime,
    model: Callable[[dict, list[dict], list[dict]], dict],
    disclose: Callable[[Task, str], frozenset[str]],
    max_rounds: int = 10,
) -> None:
    with ledger.condition:
        revision = task.revision
        while task.inflight and ledger.current(task, revision):
            ledger.condition.wait()
        if not ledger.current(task, revision):
            return
        task.status = "running"
        task.route = "slow"
    messages = [{"role": "user", "content": "继续完成任务；先阅读简报中的已完成事实和用户回填。"}]
    control = {tool["name"]: tool for tool in CONTROL_TOOLS}
    try:
        for _ in range(max_rounds):
            with ledger.lock:
                if not ledger.current(task, revision):
                    return
                brief = task.brief()
                schemas = task.snapshot.schemas(task.disclosed)
            response = model(brief, messages, schemas)
            with ledger.lock:
                if not ledger.current(task, revision):
                    return
                task.slow_rounds.append(
                    {"model": response.get("model"), "usage": response.get("usage"),
                     "content": response["content"]}
                )
            content = response["content"]
            calls = [item for item in content if item["type"] == "tool_use"]
            if not calls:
                raise ModelError("Slow model ended without an explicit task outcome")
            control_calls = [item for item in calls if item["name"] in control]
            if control_calls:
                if len(calls) != 1:
                    raise ModelError("Control actions cannot share a round with business actions")
                call = control_calls[0]
                name, arguments = call["name"], call["input"]
                errors = list(
                    Draft202012Validator(control[name]["input_schema"]).iter_errors(arguments)
                )
                if errors:
                    raise ModelError(f"Invalid control action: {errors[0].message}")
                with ledger.lock:
                    if not ledger.current(task, revision):
                        return
                    if name == "ask_user":
                        task.question = arguments["question"]
                        task.asked_questions.append(task.question)
                        task.status = "waiting_input"
                        return
                    if name == "finish":
                        if not task.completed:
                            raise ModelError("Cannot report execution success without tool facts")
                        task.status = "done"
                        task.error = None
                        return
                added = disclose(task, arguments["intent"])
                with ledger.lock:
                    if not ledger.current(task, revision):
                        return
                    task.snapshot.schemas(added)
                    task.disclosed |= added
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": call["id"],
                         "content": json.dumps({"disclosed_tools": sorted(task.disclosed)})}
                    ]},
                ])
                continue
            results = []
            failed = False
            for call in calls:
                with ledger.lock:
                    if not ledger.current(task, revision):
                        return
                    task.inflight = True
                name, arguments = call["name"], call["input"]
                try:
                    if failed:
                        raise ExecutionError("Skipped after preceding call failed")
                    runtime.validate(task.snapshot, task.disclosed, ToolCall(name, arguments))
                    duplicate = next(
                        (step for step in task.completed
                         if step["tool"] == name and step["arguments"] == arguments), None
                    )
                    if duplicate and task.snapshot.tools[name].side_effect != "idempotent":
                        result = duplicate["result"]
                    else:
                        result = runtime.execute(task.snapshot, task.disclosed, ToolCall(name, arguments))
                        with ledger.lock:
                            task.completed.append(
                                {"tool": name, "arguments": arguments, "result": result}
                            )
                    results.append(
                        {"type": "tool_result", "tool_use_id": call["id"],
                         "content": json.dumps(result, ensure_ascii=False)}
                    )
                except ExecutionError as error:
                    failed = True
                    results.append(
                        {"type": "tool_result", "tool_use_id": call["id"],
                         "is_error": True, "content": str(error)}
                    )
                finally:
                    with ledger.condition:
                        task.inflight = False
                        ledger.condition.notify_all()
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": results},
            ])
        raise ModelError("Slow chain reached its round limit; task remains incomplete")
    except (ModelError, ExecutionError, ValueError) as error:
        with ledger.lock:
            if ledger.current(task, revision):
                task.status = "failed"
                task.error = str(error)
