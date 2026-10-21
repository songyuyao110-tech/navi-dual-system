import asyncio
from dataclasses import replace
import json
from pathlib import Path
import unittest

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from execution import Runtime
from gateway import additional_disclosure, judge
from registry import RegistrySnapshot, load_registry
from test_decisions import response_for


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        original = load_registry(
            Path(__file__).resolve().parents[1] / "tools.yaml", frozenset(Runtime().handlers)
        )
        self.snapshot = RegistrySnapshot(
            original.version,
            {name: replace(tool, domain=f"domain-{index % 2}")
             for index, (name, tool) in enumerate(original.tools.items())},
        )

    async def run_case(self, missing=False, probability=None):
        requests = []
        ready = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 2)
            response = response_for(body["questions"])
            for name, answer in response["answers"].items():
                answer["noul"] = (
                    probability if probability is not None
                    else 0.5 if name == "navi_get_status" else 0.1
                )
            if missing:
                response["answers"].pop(next(iter(response["answers"])))
            return httpx2.Response(200, json=response)

        async with AsyncTypeSafeClient(
            api_key="unit-test-key", transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            result = await judge(client, "synthetic task", self.snapshot,
                                 context={"synthetic": True})
        return result, requests

    async def test_union_threshold_and_same_brief_for_every_domain_without_a(self):
        result, requests = await self.run_case()
        self.assertEqual(result["disclosed_tools"], ["navi_get_status"])
        self.assertEqual(result["decisions"]["request_count"], 2)
        self.assertEqual(result["decisions"]["response_count"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(set(r["state"] for r in requests)), 1)
        self.assertTrue(all(q["type"] == "noul" for r in requests
                            for q in r["questions"].values()))
        self.assertEqual(result["schemas"][0]["name"], "navi_get_status")
        self.assertEqual(result["route"], "slow")
        self.assertEqual(result["execution_policy"], "b_only_slow")
        self.assertEqual(result["thresholds"], {"B": 0.5})
        self.assertNotIn("task_kind", result)
        self.assertNotIn("needs_task_review", result)
        self.assertTrue(all(s["slot"].startswith("B:") for s in result["decisions"]["slots"]))

    async def test_empty_disclosure_still_routes_slow(self):
        result, _ = await self.run_case(probability=0.1)
        self.assertEqual(result["disclosed_tools"], [])
        self.assertEqual(result["schemas"], [])
        self.assertEqual(result["route"], "slow")

    async def test_all_tools_are_union_not_one_selected_domain(self):
        result, _ = await self.run_case(probability=0.9)
        self.assertEqual(len(result["disclosed_tools"]), 14)
        self.assertEqual(len(result["schemas"]), 14)

    async def test_invalid_threshold_fails_before_calls(self):
        async with AsyncTypeSafeClient(
            api_key="unit-test-key",
            transport=httpx2.MockTransport(lambda request: self.fail("Unexpected request")),
        ) as client:
            for threshold in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                    await judge(client, "synthetic", self.snapshot, b_threshold=threshold)

    async def test_missing_domain_answer_is_error(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            await self.run_case(missing=True)

    async def test_additional_disclosure_runs_all_b_domains_without_a(self):
        requests = []
        ready = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 2)
            return httpx2.Response(200, json=response_for(body["questions"]))

        async with AsyncTypeSafeClient(
            api_key="unit-test-key", transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            added = await additional_disclosure(client, self.snapshot, {}, "synthetic intent")
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(added), 14)
        self.assertTrue(all("task_kind" not in r["questions"] for r in requests))
        self.assertEqual(len(set(r["state"] for r in requests)), 1)


if __name__ == "__main__":
    unittest.main()
