import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx2

from execution import Runtime, run_fast, run_slow
from model_clients import ModelError, review_task, slow_turn
from registry import load_registry
from task_context import TaskLedger


def reply(name, arguments=None, identifier="call-1"):
    return {"model": "slow-test", "usage": {"output_tokens": 1}, "content": [
        {"type": "tool_use", "id": identifier, "name": name, "input": arguments or {}}
    ]}


def completion(name="navi_get_status", arguments=None, identifier="call-1"):
    return {
        "model": "deepseek-test",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": identifier, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments or {})},
                }],
            },
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }


class SlowTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.snapshot = load_registry(
            Path(__file__).resolve().parents[1] / "tools.yaml", frozenset(self.runtime.handlers)
        )
        self.ledger = TaskLedger()
        self.task = self.ledger.create(
            "synthetic navigation task", {}, self.snapshot,
            {"route": "slow", "disclosed_tools": ["navi_to_special_poi", "navi_get_status"]},
        )

    def test_fast_fallback_never_repeats_successful_non_idempotent_action(self):
        run_fast(self.task, self.ledger, self.runtime, lambda brief, schemas: {
            "question": None, "steps": [
                {"tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}},
            ],
        })
        self.task.status = "needs_slow"
        replies = iter([
            reply("navi_to_special_poi", {"poi_type": "home"}),
            reply("navi_get_status"), reply("finish"),
        ])
        briefs = []

        def model(brief, messages, schemas):
            briefs.append(brief)
            return next(replies)

        run_slow(self.task, self.ledger, self.runtime, model, lambda task, intent: frozenset())
        self.assertEqual(self.task.status, "done")
        self.assertEqual([c["tool"] for c in self.runtime.calls],
                         ["navi_to_special_poi", "navi_get_status"])
        self.assertEqual(len(briefs[0]["completed_steps"]), 1)

    def test_additional_disclosure_uses_same_snapshot(self):
        self.task.disclosed = frozenset()
        replies = iter([reply("request_tools", {"intent": "读取当前导航状态"}),
                        reply("navi_get_status"), reply("finish")])
        snapshots = []

        def disclose(task, intent):
            snapshots.append(task.snapshot.version)
            return frozenset({"navi_get_status"})

        run_slow(
            self.task, self.ledger, self.runtime,
            lambda brief, messages, schemas: next(replies), disclose,
        )
        self.assertEqual(self.task.status, "done")
        self.assertEqual(snapshots, [self.snapshot.version])
        self.assertEqual(self.task.disclosed, frozenset({"navi_get_status"}))

    def test_clarification_and_refill_preserve_task_and_question(self):
        run_slow(
            self.task, self.ledger, self.runtime,
            lambda brief, messages, schemas: reply("ask_user", {"question": "家还是公司？"}),
            lambda task, intent: frozenset(),
        )
        identity = self.task.id
        self.assertEqual(self.task.status, "waiting_input")
        self.ledger.revise(identity, "公司", {})
        replies = iter([reply("navi_to_special_poi", {"poi_type": "company"}), reply("finish")])
        briefs = []

        def model(brief, messages, schemas):
            briefs.append(brief)
            return next(replies)

        run_slow(self.task, self.ledger, self.runtime, model, lambda task, intent: frozenset())
        self.assertEqual(self.task.status, "done")
        self.assertEqual(self.task.id, identity)
        self.assertEqual(briefs[0]["asked_questions"], ["家还是公司？"])
        self.assertEqual(len(self.ledger.tasks), 1)

    def test_round_limit_retains_facts_without_success(self):
        run_slow(
            self.task, self.ledger, self.runtime,
            lambda brief, messages, schemas: reply("navi_get_status"),
            lambda task, intent: frozenset(), max_rounds=1,
        )
        self.assertEqual(self.task.status, "failed")
        self.assertIn("round limit", self.task.error)
        self.assertEqual(len(self.task.completed), 1)

    def test_undisclosed_tool_cannot_execute_or_finish_successfully(self):
        self.task.disclosed = frozenset({"navi_get_status"})
        replies = iter([reply("navi_cancel_route"), reply("finish")])
        messages_seen = []

        def model(brief, messages, schemas):
            messages_seen.append(json.loads(json.dumps(messages)))
            return next(replies)

        run_slow(self.task, self.ledger, self.runtime, model, lambda task, intent: frozenset())
        self.assertEqual(self.task.status, "failed")
        self.assertEqual(self.runtime.calls, [])
        self.assertTrue(messages_seen[1][-1]["content"][0]["is_error"])

    def test_cancel_while_model_returns_does_not_execute(self):
        def model(brief, messages, schemas):
            self.ledger.cancel(self.task.id)
            return reply("navi_to_special_poi", {"poi_type": "home"})

        run_slow(self.task, self.ledger, self.runtime, model, lambda task, intent: frozenset())
        self.assertEqual(self.task.status, "cancelled")
        self.assertEqual(self.runtime.calls, [])

    def test_invalid_control_action_is_explicit_failure(self):
        run_slow(
            self.task, self.ledger, self.runtime,
            lambda brief, messages, schemas: reply("ask_user", {"question": None}),
            lambda task, intent: frozenset(),
        )
        self.assertEqual(self.task.status, "failed")
        self.assertIn("Invalid control", self.task.error)

    def test_client_sees_only_disclosed_business_schemas(self):
        body = {}

        def handler(request):
            body.update(json.loads(request.content))
            self.assertEqual(str(request.url), "https://litellm.invalid/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer unit-test-key")
            return httpx2.Response(200, json=completion())

        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        with patch.dict("os.environ", {
            "LITELLM_BASE_URL": "https://litellm.invalid",
            "LITELLM_API_KEY": "unit-test-key",
            "SLOW_MODEL": "deepseek-test",
        }), patch("model_clients.httpx2.Client", return_value=client):
            result = slow_turn(
                {}, [{"role": "user", "content": "synthetic"}],
                self.snapshot.schemas(frozenset({"navi_get_status"})),
            )
        names = {tool["function"]["name"] for tool in body["tools"]}
        self.assertEqual(names, {"navi_get_status", "ask_user", "request_tools", "finish"})
        self.assertNotIn("navi_cancel_route", names)
        self.assertEqual(body["model"], "deepseek-test")
        self.assertFalse(body["stream"])
        self.assertNotIn("anthropic_version", body)
        self.assertEqual(result["content"][0]["input"], {})
        self.assertEqual(result["usage"], {"input_tokens": 20, "output_tokens": 5})

    def test_litellm_two_rounds_preserve_results_and_reasoning(self):
        self.task.disclosed = frozenset({"navi_get_status"})
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                data = completion()
                data["choices"][0]["message"]["reasoning_content"] = "opaque test metadata"
                return httpx2.Response(200, json=data)
            return httpx2.Response(200, json=completion("finish", identifier="call-2"))

        client_class = httpx2.Client

        def client(*args, **kwargs):
            return client_class(transport=httpx2.MockTransport(handler))

        with patch.dict("os.environ", {
            "LITELLM_BASE_URL": "https://litellm.invalid/v1/",
            "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
        }), patch("model_clients.httpx2.Client", side_effect=client):
            run_slow(
                self.task, self.ledger, self.runtime, slow_turn,
                lambda task, intent: frozenset(),
            )
        self.assertEqual(self.task.status, "done")
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(len(bodies), 2)
        assistant = bodies[1]["messages"][-2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["reasoning_content"], "opaque test metadata")
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], "{}")
        tool_result = bodies[1]["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(tool_result["tool_call_id"], "call-1")
        self.assertTrue(json.loads(tool_result["content"])["mock"])

    def test_tool_errors_are_explicit_in_openai_messages(self):
        body = {}

        def handler(request):
            body.update(json.loads(request.content))
            return httpx2.Response(200, json=completion("ask_user", {"question": "目标是什么？"}))

        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        with patch.dict("os.environ", {
            "LITELLM_BASE_URL": "https://litellm.invalid/v1",
            "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
        }), patch("model_clients.httpx2.Client", return_value=client):
            slow_turn({}, [
                {"role": "assistant", "content": reply("navi_cancel_route")["content"]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "call-1",
                    "content": "Tool is not disclosed", "is_error": True,
                }]},
            ], [])
        error = json.loads(body["messages"][-1]["content"])
        self.assertTrue(error["is_error"])
        self.assertEqual(error["content"], "Tool is not disclosed")

    def test_litellm_additional_disclosure_and_review_use_same_adapter(self):
        self.task.disclosed = frozenset()
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                data = completion("request_tools", {"intent": "查询导航状态"}, "request-1")
            elif len(bodies) == 2:
                data = completion("navi_get_status", identifier="status-1")
            elif len(bodies) == 3:
                data = completion("finish", identifier="finish-1")
            else:
                data = completion("ask_user", {"question": "新建还是修订现有任务？"}, "review-1")
            return httpx2.Response(200, json=data)

        client_class = httpx2.Client

        def client(*args, **kwargs):
            return client_class(transport=httpx2.MockTransport(handler))

        with patch.dict("os.environ", {
            "LITELLM_BASE_URL": "https://litellm.invalid/v1",
            "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
        }), patch("model_clients.httpx2.Client", side_effect=client):
            run_slow(
                self.task, self.ledger, self.runtime, slow_turn,
                lambda task, intent: frozenset({"navi_get_status"}),
            )
            question = review_task(
                "查询状态", {}, {"task_kind": "new", "target_task": None, "route": "slow"}, {},
            )
        self.assertEqual(self.task.status, "done")
        self.assertEqual(self.task.snapshot.version, self.snapshot.version)
        before = {tool["function"]["name"] for tool in bodies[0]["tools"]}
        after = {tool["function"]["name"] for tool in bodies[1]["tools"]}
        self.assertNotIn("navi_get_status", before)
        self.assertEqual(after - before, {"navi_get_status"})
        added = bodies[1]["messages"][-1]
        self.assertEqual(added["tool_call_id"], "request-1")
        self.assertEqual(json.loads(added["content"])["disclosed_tools"], ["navi_get_status"])
        self.assertEqual(question, "新建还是修订现有任务？")
        self.assertEqual(
            {tool["function"]["name"] for tool in bodies[3]["tools"]},
            {"ask_user", "request_tools", "finish"},
        )

    def test_missing_litellm_settings_never_use_bedrock_credentials(self):
        with patch.dict("os.environ", {
            "ANTHROPIC_BEDROCK_BASE_URL": "http://127.0.0.1",
            "AWS_BEARER_TOKEN_BEDROCK": "old-unit-test-key",
        }, clear=True), patch("model_clients.httpx2.Client") as client:
            with self.assertRaisesRegex(ModelError, "LiteLLM configuration required"):
                slow_turn({}, [], [])
        client.assert_not_called()

    def test_remote_plain_http_litellm_is_rejected_before_transport(self):
        with patch.dict("os.environ", {
            "LITELLM_BASE_URL": "http://litellm.invalid/v1",
            "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
        }), patch("model_clients.httpx2.Client") as client:
            with self.assertRaisesRegex(ModelError, "require HTTPS"):
                slow_turn({}, [], [])
        client.assert_not_called()

    def test_invalid_completions_fail_explicitly(self):
        invalid = [
            ("choices", {}, "invalid choices"),
            ("truncation", {"choices": [{"finish_reason": "length"}]}, "truncated"),
            ("arguments", completion(), "invalid JSON tool arguments"),
            ("array-arguments", completion(), "must be an object"),
            ("non-finite", completion(), "invalid JSON tool arguments"),
            ("duplicate-id", completion(), "identifier"),
            ("wrong-type", completion(), "identifier"),
        ]
        invalid[2][1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{"
        invalid[3][1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "[]"
        invalid[4][1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"x":NaN}'
        duplicate = invalid[5][1]["choices"][0]["message"]["tool_calls"]
        duplicate.append(json.loads(json.dumps(duplicate[0])))
        invalid[6][1]["choices"][0]["message"]["tool_calls"][0]["type"] = "invalid"
        for name, data, error in invalid:
            with self.subTest(name=name):
                client = httpx2.Client(transport=httpx2.MockTransport(
                    lambda request: httpx2.Response(200, json=data)
                ))
                with patch.dict("os.environ", {
                    "LITELLM_BASE_URL": "https://litellm.invalid",
                    "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
                }), patch("model_clients.httpx2.Client", return_value=client):
                    with self.assertRaisesRegex(ModelError, error):
                        slow_turn({}, [], [])

    def test_http_and_json_errors_do_not_leak_credentials(self):
        for response in [
            httpx2.Response(401, text="private provider response"),
            httpx2.Response(200, text="not JSON"),
        ]:
            with self.subTest(status=response.status_code):
                client = httpx2.Client(transport=httpx2.MockTransport(lambda request: response))
                with patch.dict("os.environ", {
                    "LITELLM_BASE_URL": "https://litellm.invalid",
                    "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
                }), patch("model_clients.httpx2.Client", return_value=client):
                    with self.assertRaises(ModelError) as caught:
                        slow_turn({}, [], [])
                self.assertNotIn("unit-test-key", str(caught.exception))
                self.assertNotIn("private provider response", str(caught.exception))

    def test_task_review_preserves_known_preferences(self):
        context = {"company_address_configured": True}
        with patch("model_clients.slow_turn", return_value=reply(
            "ask_user", {"question": "这是一个新的导航任务吗？"}
        )) as model:
            question = review_task(
                "导航去公司", {}, {"task_kind": "revise", "target_task": None, "route": "slow"},
                context,
            )
        self.assertEqual(model.call_args.args[0]["context"], context)
        self.assertEqual(question, "这是一个新的导航任务吗？")


if __name__ == "__main__":
    unittest.main()
