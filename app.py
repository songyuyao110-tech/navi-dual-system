import argparse
import asyncio
import json
from pathlib import Path
import os
from threading import RLock
import time
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from execution import Runtime, run_slow
from gateway import additional_disclosure, judge
from jev_smoke import load_environment
from model_clients import ModelError, slow_turn
from registry import Registry
from task_context import TaskLedger

ROOT = Path(__file__).resolve().parent


def create_app() -> Flask:
    app = Flask(__name__)
    runtime = Runtime()
    ledger = TaskLedger()
    registry_store = Registry(
        Path(os.environ.get("TOOLS_FILE", ROOT / "tools.yaml")), frozenset(runtime.handlers)
    )
    app.extensions["runtime"] = runtime
    app.extensions["ledger"] = ledger
    app.extensions["registry"] = registry_store
    socketio = SocketIO(app, async_mode="threading")
    jobs: dict[str, dict] = {}
    lock = RLock()
    app.extensions["jobs"] = jobs

    def disclose(task, intent: str):
        async def query():
            async with AsyncTypeSafeClient(
                retry=RetryPolicy(max_retries=0), timeout=30.0
            ) as client:
                return await additional_disclosure(
                    client, task.snapshot, task.brief(), intent
                )
        try:
            return asyncio.run(query())
        except TypeSafeError as error:
            raise ModelError(f"Additional JEV disclosure failed: {type(error).__name__}") from None

    def run_job(
        job_id: str, utterance: str, context: dict, captured, execute: bool,
        submitted: float, task=None, submission_kind: str = "preview",
    ):
        async def decide():
            async with AsyncTypeSafeClient(
                retry=RetryPolicy(max_retries=0), timeout=30.0
            ) as client:
                return await judge(client, utterance, captured, context=context)

        worker_revision = task.revision if task is not None else None
        outcome = {"status": "error", "error": {"type": "InternalWorkerError"}}
        try:
            if submission_kind == "refill":
                result = {
                    "execution_policy": "b_only_slow", "route": "slow",
                    "registry_version": task.snapshot.version,
                    "disclosed_tools": sorted(task.disclosed),
                    "schemas": task.snapshot.schemas(task.disclosed),
                    "thresholds": {"B": 0.5},
                    "decisions": {"request_count": 0, "response_count": 0,
                                  "slots": [], "skipped": "refill_keeps_original_tools"},
                }
            else:
                result = asyncio.run(decide())
            result["submission_kind"] = submission_kind
            if execute:
                with ledger.lock:
                    task.disclosed = frozenset(result["disclosed_tools"])
                    task.status = "needs_slow"
                run_slow(task, ledger, runtime, slow_turn, disclose)
                with ledger.lock:
                    if task.question:
                        ledger.record("assistant", task.question)
                    elif task.status == "done":
                        ledger.record(
                            "assistant", "执行事实（mock）：" +
                            json.dumps(task.completed, ensure_ascii=False),
                        )
            result["round_ms"] = (time.perf_counter() - submitted) * 1000
            outcome = {"status": "finished" if execute else "decided", "result": result}
        except (TypeSafeError, ValueError, OSError) as error:
            diagnostic = {"type": type(error).__name__,
                          "http_status": getattr(error, "status_code", None)}
            app.logger.error("Task job failed: %s", diagnostic)
            outcome = {"status": "error", "error": diagnostic}
        except Exception as error:
            # Do not publish provider bodies, credentials, or an exception message.
            app.logger.error("Unexpected task worker failure: %s", type(error).__name__)
        finally:
            with ledger.lock:
                if task is not None:
                    if outcome["status"] == "error" and ledger.current(task, worker_revision):
                        task.status = "failed"
                        task.error = "Task worker failed; see job diagnostic type"
                    task.processing = False
                    if "result" in outcome:
                        outcome["result"]["task"] = task.public()
                    else:
                        outcome["task"] = task.public()
                with lock:
                    jobs[job_id] = outcome

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        current = registry_store.current
        return jsonify(
            status="ready",
            execution_mode="mock",
            execution_policy="b_only_slow",
            registry_version=current.version,
            tools=len(current.tools),
        )

    @app.get("/api/registry")
    def registry():
        current = registry_store.current
        return jsonify(
            version=current.version,
            tools=[
                {
                    "name": tool.name,
                    "domain": tool.domain,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "binding": tool.binding,
                    "side_effect": tool.side_effect,
                }
                for tool in current.tools.values()
            ],
        )

    @app.post("/api/registry/reload")
    def reload_registry():
        try:
            current = registry_store.reload()
        except (OSError, ValueError) as error:
            app.logger.error("Registry reload rejected: %s", type(error).__name__)
            return jsonify(error=str(error), active_version=registry_store.current.version), 400
        return jsonify(version=current.version, tools=len(current.tools))

    @app.get("/api/runtime")
    def calls():
        with runtime.lock:
            return jsonify(mode="mock", calls=runtime.calls, navigation=runtime.state)

    @app.get("/api/tasks")
    def tasks():
        return jsonify(tasks=ledger.public())

    @app.post("/api/decide")
    @app.post("/api/chat")
    def decide():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Expected a JSON object"), 400
        utterance = data.get("utterance")
        context = data.get("context", {})
        if not isinstance(utterance, str) or not utterance.strip() or not isinstance(context, dict):
            return jsonify(error="Nonempty utterance and object context required"), 400
        if not os.environ.get("TYPESAFE_API_KEY"):
            return jsonify(error="JEV credentials are not configured"), 503
        submitted = time.perf_counter()
        job_id = str(uuid4())
        execute = request.path == "/api/chat"
        captured = registry_store.current
        task = None
        submission_kind = "preview"
        with ledger.lock:
            context = ledger.brief_context(context)
            context["conversation"].append({"role": "user", "content": utterance})
            context["conversation"] = context["conversation"][-12:]
            with runtime.lock:
                context["navigation_state"] = dict(runtime.state)
            if execute:
                try:
                    task, submission_kind = ledger.accept_slow(utterance, context, captured)
                except ValueError as error:
                    return jsonify(error=str(error)), 409
                captured = task.snapshot
                ledger.record("user", utterance)
            with lock:
                jobs[job_id] = {"status": "judging", "registry_version": captured.version}
        try:
            socketio.start_background_task(
                run_job, job_id, utterance, context, captured, execute,
                submitted, task, submission_kind,
            )
        except Exception:
            with ledger.lock:
                if task is not None:
                    task.processing = False
                    task.status = "failed"
                    task.error = "Task worker could not start"
                with lock:
                    jobs[job_id] = {"status": "error", "error": {"type": "WorkerStartError"}}
            return jsonify(error="无法启动任务处理，请重试。"), 503
        return jsonify(job_id=job_id), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str):
        with lock:
            if job_id not in jobs:
                return jsonify(error="Unknown job"), 404
            return jsonify(jobs[job_id])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the independent dual-chain PoC")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        load_environment(args.env_file)
    application = create_app()
    application.extensions["socketio"].run(
        application,
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8260")),
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )
