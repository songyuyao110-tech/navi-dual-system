import json
import os
import unittest
from unittest.mock import patch

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app import create_app
from test_decisions import response_for
from test_slow import completion


class BOnlyHTTPProtocolTests(unittest.TestCase):
    def test_real_adapters_with_mock_transports_disclose_ask_refill_execute_finish(self):
        """No live providers: exercise actual SDK and LiteLLM normalization via HTTP."""
        app = create_app()
        client = app.test_client()
        jev_requests = []
        slow_requests = []

        async def jev_handler(request):
            body = json.loads(request.content)
            jev_requests.append(body)
            self.assertEqual(len(body["questions"]), 14)
            self.assertTrue(all(q["type"] == "noul" for q in body["questions"].values()))
            response = response_for(body["questions"])
            for name, answer in response["answers"].items():
                answer["noul"] = 0.9 if name == "navi_start_route" else 0.1
            return httpx2.Response(200, json=response)

        def jev_client(**kwargs):
            return AsyncTypeSafeClient(
                api_key="unit-test-key", retry=RetryPolicy(max_retries=0),
                transport=httpx2.MockTransport(jev_handler),
            )

        def slow_handler(request):
            body = json.loads(request.content)
            slow_requests.append(body)
            self.assertEqual(body["model"], "deepseek-test")
            self.assertEqual({tool["function"]["name"] for tool in body["tools"]},
                             {"navi_start_route", "ask_user", "request_tools", "finish"})
            if len(slow_requests) == 1:
                result = completion("ask_user", {"question": "去哪里？"}, "ask-1")
            elif len(slow_requests) == 2:
                result = completion("navi_start_route", {"end_name": "天安门"}, "start-1")
            else:
                result = completion("finish", identifier="finish-1")
            return httpx2.Response(200, json=result)

        sync_client = httpx2.Client

        def slow_client(*args, **kwargs):
            return sync_client(transport=httpx2.MockTransport(slow_handler))

        with patch.dict(os.environ, {
            "TYPESAFE_API_KEY": "unit-test-key",
            "LITELLM_BASE_URL": "https://litellm.invalid/v1",
            "LITELLM_API_KEY": "unit-test-key", "SLOW_MODEL": "deepseek-test",
        }), patch("app.AsyncTypeSafeClient", side_effect=jev_client), \
             patch("model_clients.httpx2.Client", side_effect=slow_client), \
             patch("model_clients.fast_plan", side_effect=AssertionError("Fast model called")), \
             patch("model_clients.review_task", side_effect=AssertionError("Task review called")), \
             patch("decisions.a_questions", side_effect=AssertionError("A questions called")), \
             patch.object(app.extensions["socketio"], "start_background_task",
                          side_effect=lambda target, *args: target(*args)):
            first = client.post("/api/chat", json={"utterance": "合成测试：请帮我导航"})
            self.assertEqual(first.status_code, 202)
            first_job = client.get("/api/jobs/" + first.get_json()["job_id"]).get_json()
            self.assertEqual(first_job["result"]["task"]["status"], "waiting_input")
            self.assertEqual(first_job["result"]["decisions"]["request_count"], 1)
            second = client.post("/api/chat", json={"utterance": "天安门"})
            self.assertEqual(second.status_code, 202)
            second_job = client.get("/api/jobs/" + second.get_json()["job_id"]).get_json()

        task = second_job["result"]["task"]
        self.assertEqual(task["status"], "done")
        self.assertEqual(task["route"], "slow")
        self.assertEqual(task["id"], first_job["result"]["task"]["id"])
        self.assertEqual(task["registry_version"], first_job["result"]["task"]["registry_version"])
        self.assertEqual(task["turns"], ["合成测试：请帮我导航", "天安门"])
        self.assertEqual(len(jev_requests), 1)
        self.assertEqual(len(slow_requests), 3)
        self.assertEqual(len(app.extensions["runtime"].calls), 1)
        self.assertEqual(app.extensions["runtime"].state["destination"], "天安门")
        brief = json.loads(slow_requests[1]["messages"][0]["content"].split("任务简报：", 1)[1])
        self.assertEqual(brief["latest_utterance"], "天安门")
        self.assertEqual(brief["asked_questions"], ["去哪里？"])
        tool_message = slow_requests[2]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertTrue(json.loads(tool_message["content"])["mock"])
        self.assertEqual(len(app.extensions["ledger"].tasks), 1)


if __name__ == "__main__":
    unittest.main()
