import asyncio
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from decisions import Tool, a_questions, run_b_decisions, run_decisions, serialize_results
from jev_smoke import load_environment


def response_for(questions):
    answers = {}
    for name, question in questions.items():
        if question["type"] == "choice":
            labels = list(question["criteria"])
            answers[name] = {
                "type": "choice",
                "choice": labels[0],
                "confidence": 0.95,
                "probabilities": {
                    label: 1.0 if index == 0 else 0.0
                    for index, label in enumerate(labels)
                },
            }
        else:
            answers[name] = {"type": "noul", "noul": 0.9}
    return {"model": "jev-test", "usage": {}, "answers": answers}


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_domains_are_concurrent_one_request_each(self):
        requests = []
        ready = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 11:
                ready.set()
            await asyncio.wait_for(ready.wait(), 2)
            return httpx2.Response(200, json=response_for(body["questions"]))

        domains = {
            f"domain-{domain}": tuple(
                Tool(f"tool-{domain}-{tool}", f"capability {tool}")
                for tool in range(20)
            )
            for domain in range(10)
        }
        async with AsyncTypeSafeClient(
            api_key="unit-test-key",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(handler),
        ) as client:
            results = await run_decisions(client, "synthetic task", domains)
        self.assertEqual(len(requests), 11)
        self.assertEqual(sorted(len(r["questions"]) for r in requests), [2] + [20] * 10)
        self.assertTrue(all(r["state"] == "synthetic task" for r in requests))
        report = serialize_results(results)
        self.assertEqual(report["request_count"], 11)
        self.assertEqual(len(report["slots"]), 11)
        self.assertTrue(all(math.isfinite(r["latency_ms"]) for r in report["slots"]))
        self.assertLess(max(r.started_ms for r in results), min(r.finished_ms for r in results))
        self.assertNotIn("confidence", results[1].response.answers["tool-0-0"].model_dump())

    async def test_b_only_ten_domains_twenty_tools_are_ten_concurrent_requests(self):
        requests = []
        ready = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 10:
                ready.set()
            await asyncio.wait_for(ready.wait(), 2)
            return httpx2.Response(200, json=response_for(body["questions"]))

        domains = {
            f"domain-{domain}": tuple(
                Tool(f"tool-{domain}-{tool}", f"capability {tool}") for tool in range(20)
            ) for domain in range(10)
        }
        async with AsyncTypeSafeClient(
            api_key="unit-test-key", retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(handler),
        ) as client:
            results = await run_b_decisions(client, "synthetic task", domains)
        self.assertEqual(len(requests), 10)
        self.assertEqual([len(r["questions"]) for r in requests], [20] * 10)
        self.assertTrue(all(q["type"] == "noul" for r in requests
                            for q in r["questions"].values()))
        self.assertTrue(all(result.slot.startswith("B:") for result in results))
        self.assertLess(max(r.started_ms for r in results), min(r.finished_ms for r in results))
        self.assertEqual(serialize_results(results)["request_count"], 10)

    async def test_missing_answer_is_not_empty_domain(self):
        async def handler(request):
            body = json.loads(request.content)
            response = response_for(body["questions"])
            if "tool" in response["answers"]:
                response["answers"] = {}
            return httpx2.Response(200, json=response)

        async with AsyncTypeSafeClient(
            api_key="unit-test-key",
            transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                await run_decisions(client, "synthetic", {"domain": (Tool("tool", "test"),)})

    async def test_out_of_range_probability_fails(self):
        async def handler(request):
            body = json.loads(request.content)
            response = response_for(body["questions"])
            if "tool" in response["answers"]:
                response["answers"]["tool"]["noul"] = 1.2
            return httpx2.Response(200, json=response)

        async with AsyncTypeSafeClient(
            api_key="unit-test-key",
            transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            with self.assertRaisesRegex(ValueError, "Invalid probability"):
                await run_decisions(client, "synthetic", {"domain": (Tool("tool", "test"),)})

    async def test_duplicate_tool_identifiers_fail_before_calls(self):
        async with AsyncTypeSafeClient(
            api_key="unit-test-key",
            transport=httpx2.MockTransport(lambda request: self.fail("Unexpected request")),
        ) as client:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                await run_decisions(
                    client, "synthetic", {"domain": (Tool("tool", "a"), Tool("tool", "b"))}
                )


class QuestionAndEnvironmentTests(unittest.TestCase):
    def test_a_has_no_domain_questions(self):
        self.assertEqual(set(a_questions({})), {"task_kind", "route"})
        self.assertEqual(
            set(a_questions({"pending-1": "waiting for destination"})),
            {"task_kind", "route", "target_task"},
        )

    def test_only_allowlisted_variables_load_and_existing_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.env"
            path.write_text(
                'TYPESAFE_API_KEY="file-key"\n'
                'export TYPESAFE_DEFAULT_MODEL=jev-test # comment\n'
                'UNRELATED_SECRET=must-not-load\n'
            )
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "process-key"}, clear=True):
                load_environment(path)
                self.assertEqual(os.environ["TYPESAFE_API_KEY"], "process-key")
                self.assertEqual(os.environ["TYPESAFE_DEFAULT_MODEL"], "jev-test")
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_malformed_secret_is_not_printed_in_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.env"
            path.write_text('TYPESAFE_API_KEY="sensitive-value\n')
            with self.assertRaisesRegex(ValueError, "Malformed") as raised:
                load_environment(path)
            self.assertNotIn("sensitive-value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
