from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    SystemOneResponse,
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str


@dataclass(frozen=True)
class SlotResult:
    slot: str
    started_ms: float
    finished_ms: float
    response: SystemOneResponse


def a_questions(pending_tasks: Mapping[str, str]) -> dict[str, Question]:
    questions: dict[str, Question] = {
        "task_kind": Choice(
            instructions=(
                "判断当前话的台账任务性质。只有简报 pending_tasks 中未完成的任务可以修订或取消。"
                "已完成的工具调用不是可修订任务；车辆仍导航中不代表台账任务仍未完成。"
                "当前没有未完成任务时，导航去另一个目的地、取消当前导航等业务操作属于新建执行任务。"
                "只补充或改变 pending_tasks 中现有任务才属于修订。"
            ),
            criteria={
                "new": "新建执行任务",
                "revise": "补充或修订 pending_tasks 中的一个未完成任务",
                "cancel": "取消 pending_tasks 中的一个未完成任务",
                "non_instruction": "非执行指令",
            },
        ),
        "route": Choice(
            instructions="判断新建任务适合哪条执行链路。非新建时此题不使用。",
            criteria={
                "fast": "能够一次生成完整工具计划，执行中不再规划",
                "slow": "需要根据中间结果规划后续步骤或解决歧义",
            },
        ),
    }
    if pending_tasks:
        questions["target_task"] = Choice(
            instructions="选择当前话对应的已有任务，无法唯一确定或不适用时选 unknown。",
            criteria={**pending_tasks, "unknown": "无法唯一确定或不适用"},
        )
    return questions


def b_questions(tools: tuple[Tool, ...]) -> dict[str, Question]:
    if not tools:
        raise ValueError("A B domain must contain at least one tool")
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("Duplicate tool identifiers in B domain")
    return {
        tool.name: Noul(
            instructions=(
                "判断完成整个原始任务是否可能需要下面的工具，包括后续步骤。"
                "只判断相关性，不填写参数，不执行。\n"
                f"工具：{tool.name}\n能力：{tool.description}"
            )
        )
        for tool in tools
    }


def validate_response(
    questions: Mapping[str, Question], response: SystemOneResponse
) -> None:
    if set(response.answers) != set(questions):
        raise ValueError("JEV response question identifiers are incomplete or unexpected")
    for name, question in questions.items():
        answer = response.answers[name]
        if isinstance(question, Choice):
            if not isinstance(answer, ChoiceAnswer):
                raise ValueError(f"Expected Choice answer for {name}")
            if answer.choice not in question.criteria:
                raise ValueError(f"Invalid selected label for {name}")
            if set(answer.probabilities) != set(question.criteria):
                raise ValueError(f"Incomplete Choice probabilities for {name}")
            probabilities = [answer.confidence, *answer.probabilities.values()]
        else:
            if not isinstance(answer, NoulAnswer):
                raise ValueError(f"Expected Noul answer for {name}")
            probabilities = [answer.noul]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError(f"Invalid probability for {name}")


async def run_decisions(
    client: AsyncTypeSafeClient,
    state: str,
    domains: Mapping[str, tuple[Tool, ...]],
    pending_tasks: Mapping[str, str] | None = None,
    stop_after_a: Callable[[SlotResult], bool] | None = None,
) -> tuple[SlotResult, ...]:
    if not domains:
        raise ValueError("At least one registered domain is required")
    if "A" in domains:
        raise ValueError("Domain identifier A is reserved")
    groups = {"A": a_questions(pending_tasks or {})}
    groups.update({f"B:{domain}": b_questions(tools) for domain, tools in domains.items()})
    return await _run_groups(client, state, groups, stop_after_a)


async def run_b_decisions(
    client: AsyncTypeSafeClient, state: str, domains: Mapping[str, tuple[Tool, ...]]
) -> tuple[SlotResult, ...]:
    if not domains:
        raise ValueError("At least one registered domain is required")
    groups = {f"B:{domain}": b_questions(tools) for domain, tools in domains.items()}
    return await _run_groups(client, state, groups)


async def _run_groups(
    client: AsyncTypeSafeClient, state: str, groups: Mapping[str, Mapping[str, Question]],
    stop_after_a: Callable[[SlotResult], bool] | None = None,
) -> tuple[SlotResult, ...]:
    origin = time.perf_counter()

    async def call(slot: str, questions: Mapping[str, Question]) -> SlotResult:
        started = (time.perf_counter() - origin) * 1000
        response = await client.system_one(state=state, questions=questions)
        finished = (time.perf_counter() - origin) * 1000
        validate_response(questions, response)
        return SlotResult(slot, started, finished, response)

    tasks = [asyncio.create_task(call(slot, questions)) for slot, questions in groups.items()]
    try:
        if stop_after_a is not None:
            first = await tasks[0]
            if stop_after_a(first):
                return (first,)
        return tuple(await asyncio.gather(*tasks))
    finally:
        # A failed shard must not leave calls running after the HTTP client closes.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def mixed_questions(tools: tuple[Tool, ...], memories: tuple[dict, ...]) -> dict[str, Noul]:
    """Experimental flat heads; full legal catalog in, no domain/semantic prefilter."""
    ids = [tool.name for tool in tools] + [entry.get("id") for entry in memories]
    if any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None for name in ids):
        raise ValueError("Invalid mixed candidate identifier")
    if len({entry["id"] for entry in memories}) != len(memories):
        raise ValueError("Duplicate memory identifier")
    questions = {"tool:" + name: question for name, question in b_questions(tools).items()}
    for entry in memories:
        if not isinstance(entry.get("summary"), str) or not entry["summary"].strip():
            raise ValueError("Memory summary must be nonempty")
        questions["memory:" + entry["id"]] = Noul(instructions=(
            "判断该候选记忆是否为整个当前任务的理解或填参所需；不把偏好当授权。\n" + entry["summary"]))
    return questions


def select_mixed(questions: Mapping[str, Noul], probabilities: Mapping[str, float], *,
                 tool_threshold: float = 0.5, memory_threshold: float = 0.5,
                 max_selected: int | None = None) -> dict:
    """Threshold the complete response; malformed is never a valid empty set."""
    for threshold in (tool_threshold, memory_threshold):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be a finite probability")
    if max_selected is not None and (type(max_selected) is not int or max_selected < 0):
        raise ValueError("Capacity must be a nonnegative integer")
    if not questions or set(probabilities) != set(questions):
        raise ValueError("Mixed probability identifiers incomplete or unexpected")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in probabilities.values()):
        raise ValueError("Mixed probabilities must be finite numbers in [0,1]")
    if any(not isinstance(name, str) or not name.startswith(("tool:", "memory:")) for name in questions):
        raise ValueError("Unknown mixed head")
    tools = sorted(name[5:] for name in questions if name.startswith("tool:") and probabilities[name] >= tool_threshold)
    memories = sorted(name[7:] for name in questions if name.startswith("memory:") and probabilities[name] >= memory_threshold)
    count = len(tools) + len(memories)
    status = "ok" if count else "empty"
    if max_selected is not None and count > max_selected:
        status, tools, memories = "capacity_failure", [], []  # Never top-k/truncate or rescue.
    return {"status": status, "selected_tool_ids": tools, "selected_memory_ids": memories,
            "probabilities": dict(probabilities), "thresholds": {"tool": tool_threshold, "memory": memory_threshold},
            "max_selected": max_selected, "candidate_count": len(questions)}


def serialize_results(results: tuple[SlotResult, ...]) -> dict:
    return {
        "request_count": len(results),
        "decision_ms": max(result.finished_ms for result in results),
        "slots": [
            {
                "slot": result.slot,
                "started_ms": result.started_ms,
                "finished_ms": result.finished_ms,
                "latency_ms": result.finished_ms - result.started_ms,
                "model": result.response.model,
                "question_count": len(result.response.answers),
                "answers": {
                    name: answer.model_dump()
                    for name, answer in result.response.answers.items()
                },
                "usage": result.response.usage.model_dump(),
            }
            for result in results
        ],
    }
