import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx2

from execution import Runtime, run_fast
from model_clients import ModelError, fast_plan
from registry import load_registry
from task_context import TaskLedger


def one_step(brief, schemas):
    return {
        "question": None,
        "steps": [{"tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}}],
    }


class FastTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.snapshot = load_registry(
            Path(__file__).resolve().parents[1] / "tools.yaml", frozenset(self.runtime.handlers)
        )
        self.ledger = TaskLedger()
        self.task = self.ledger.create(
            "导航回家", {}, self.snapshot,
            {"route": "fast", "disclosed_tools": list(self.snapshot.tools)},
        )

    def test_one_plan_executes_shared_runtime(self):
        planner_calls = []

        def planner(brief, schemas):
            planner_calls.append(brief)
            return one_step(brief, schemas)

        run_fast(self.task, self.ledger, self.runtime, planner)
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(self.task.status, "done")
        self.assertEqual(len(self.runtime.calls), 1)

    def test_missing_input_and_refill_keep_id_and_registry(self):
        run_fast(
            self.task, self.ledger, self.runtime,
            lambda brief, schemas: {"question": "去哪个地址？", "steps": []},
        )
        self.assertEqual(self.task.status, "waiting_input")
        self.assertEqual(self.runtime.calls, [])
        identity = self.task.id
        revised = self.ledger.revise(identity, "去家里", {})
        run_fast(revised, self.ledger, self.runtime, one_step)
        self.assertEqual(revised.id, identity)
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.snapshot.version, self.snapshot.version)
        self.assertEqual(len(self.ledger.tasks), 1)
        self.assertEqual(revised.status, "done")

    def test_all_parameters_checked_before_any_action(self):
        run_fast(
            self.task, self.ledger, self.runtime,
            lambda brief, schemas: {
                "question": None,
                "steps": [
                    {"tool": "navi_get_status", "arguments": {}},
                    {"tool": "navi_to_special_poi", "arguments": {}},
                ],
            },
        )
        self.assertEqual(self.task.status, "needs_slow")
        self.assertEqual(self.runtime.calls, [])

    def test_failure_stops_remaining_and_keeps_completed_facts(self):
        original = self.runtime.handlers["mock"]

        def fail_second(tool, arguments):
            if tool.name == "navi_get_status":
                raise ModelError("explicit test failure")
            return original(tool, arguments)

        self.runtime.handlers["mock"] = fail_second
        run_fast(
            self.task, self.ledger, self.runtime,
            lambda brief, schemas: {
                "question": None,
                "steps": [
                    {"tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}},
                    {"tool": "navi_get_status", "arguments": {}},
                    {"tool": "navi_cancel_route", "arguments": {}},
                ],
            },
        )
        self.assertEqual(self.task.status, "needs_slow")
        self.assertEqual(len(self.task.completed), 1)
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertTrue(self.task.completed[0]["result"]["facts"]["navigation"]["route_active"])

    def test_cancel_during_plan_prevents_execution(self):
        def planner(brief, schemas):
            self.ledger.cancel(self.task.id)
            return one_step(brief, schemas)

        run_fast(self.task, self.ledger, self.runtime, planner)
        self.assertEqual(self.task.status, "cancelled")
        self.assertEqual(self.runtime.calls, [])

    def test_revision_invalidates_old_plan(self):
        def planner(brief, schemas):
            self.ledger.revise(self.task.id, "改去公司", {})
            return one_step(brief, schemas)

        run_fast(self.task, self.ledger, self.runtime, planner)
        self.assertEqual(self.task.status, "new")
        self.assertEqual(self.runtime.calls, [])

    def test_cancel_after_dispatched_step_retains_fact_stops_remainder(self):
        original = self.runtime.handlers["mock"]

        def cancel_after_call(tool, arguments):
            result = original(tool, arguments)
            self.ledger.cancel(self.task.id)
            return result

        self.runtime.handlers["mock"] = cancel_after_call
        run_fast(
            self.task, self.ledger, self.runtime,
            lambda brief, schemas: {
                "question": None,
                "steps": [
                    {"tool": "navi_get_status", "arguments": {}},
                    {"tool": "navi_cancel_route", "arguments": {}},
                ],
            },
        )
        self.assertEqual(self.task.status, "cancelled")
        self.assertEqual(len(self.task.completed), 1)
        self.assertEqual(len(self.runtime.calls), 1)

    def test_revision_during_dispatch_cannot_start_another_fast_plan(self):
        original = self.runtime.handlers["mock"]

        def revise_during_call(tool, arguments):
            self.ledger.revise(self.task.id, "改去公司", {})
            return original(tool, arguments)

        self.runtime.handlers["mock"] = revise_during_call
        run_fast(self.task, self.ledger, self.runtime, one_step)
        self.assertEqual(self.task.status, "needs_slow")
        self.assertEqual(self.task.route, "slow")
        self.assertEqual(len(self.task.completed), 1)

    def test_local_model_uses_structured_non_thinking_output(self):
        body = {}

        def handler(request):
            body.update(json.loads(request.content))
            return httpx2.Response(
                200, json={"model": "qwen3:4b", "message": {"content": json.dumps(one_step({}, []))}}
            )

        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        with patch("model_clients.httpx2.Client", return_value=client):
            result = fast_plan({}, [])
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])
        self.assertEqual(result["steps"][0]["arguments"], {"poi_type": "home"})
        self.assertIsNone(result["usage"]["input_tokens"])


if __name__ == "__main__":
    unittest.main()
