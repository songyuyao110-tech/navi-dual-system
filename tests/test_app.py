from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
import os
import unittest
from unittest.mock import AsyncMock, patch

from app import create_app
from test_slow import reply


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    @staticmethod
    def b_decision(names=None):
        return {
            "execution_policy": "b_only_slow", "route": "slow",
            "disclosed_tools": names if names is not None else ["navi_to_special_poi"],
            "thresholds": {"B": 0.5},
            "decisions": {"request_count": 1, "response_count": 1,
                          "slots": [{"slot": "B:navigation"}]},
        }

    def submit(self, turns, utterance="synthetic task", decision=None, endpoint="/api/chat"):
        socketio = self.app.extensions["socketio"]
        model = iter(turns)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch("app.AsyncTypeSafeClient") as jev_client, \
             patch("app.judge", new=AsyncMock(return_value=decision or self.b_decision())) as judge, \
             patch("app.slow_turn", side_effect=lambda *args: next(model)) as slow, \
             patch("model_clients.fast_plan", side_effect=AssertionError("Fast model called")), \
             patch("model_clients.review_task", side_effect=AssertionError("A review called")), \
             patch.object(socketio, "start_background_task",
                          side_effect=lambda target, *args: target(*args)):
            response = self.client.post(endpoint, json={"utterance": utterance})
        self.assertEqual(response.status_code, 202)
        job = self.client.get("/api/jobs/" + response.get_json()["job_id"]).get_json()
        return job, judge, slow, jev_client

    def home_turns(self):
        return [reply("navi_to_special_poi", {"poi_type": "home"}), reply("finish")]

    def test_registry_runtime_and_page(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("仅披露工具 B", page)
        self.assertNotIn("仅判断 A/B", page)
        health = self.client.get("/health").get_json()
        self.assertEqual(health["tools"], 14)
        self.assertEqual(health["execution_mode"], "mock")
        self.assertEqual(health["execution_policy"], "b_only_slow")
        self.assertEqual(len(self.client.get("/api/registry").get_json()["tools"]), 14)
        self.assertEqual(self.client.get("/api/runtime").get_json()["calls"], [])

    def test_full_http_b_then_slow_flow_never_calls_fast_or_a_review(self):
        job, judge, slow, _ = self.submit(self.home_turns())
        self.assertEqual(job["status"], "finished")
        self.assertEqual(job["result"]["submission_kind"], "new")
        task = job["result"]["task"]
        self.assertEqual(task["status"], "done")
        self.assertEqual(task["route"], "slow")
        self.assertFalse(task["processing"])
        self.assertEqual(task["plans"], [])
        self.assertEqual(len(task["slow_rounds"]), 2)
        self.assertEqual(judge.await_count, 1)
        self.assertEqual(slow.call_count, 2)
        self.assertEqual(len(self.client.get("/api/runtime").get_json()["calls"]), 1)

    def test_completed_command_allows_new_task_while_vehicle_is_navigating(self):
        first, _, _, _ = self.submit(self.home_turns())
        self.assertEqual(self.app.extensions["ledger"].pending(), {})
        self.assertTrue(self.app.extensions["runtime"].state["route_active"])
        second, _, _, _ = self.submit(self.home_turns())
        self.assertNotEqual(first["result"]["task"]["id"], second["result"]["task"]["id"])
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 2)

    def test_refill_keeps_identity_questions_snapshot_and_does_not_rerun_b(self):
        first, _, _, _ = self.submit([reply("ask_user", {"question": "家还是公司？"})])
        original = first["result"]["task"]
        store = self.app.extensions["registry"]
        old = store.current
        store._snapshot = replace(old, version="new-test-version")
        second, judge, slow, jev_client = self.submit([
            reply("navi_to_special_poi", {"poi_type": "company"}), reply("finish"),
        ], utterance="公司")
        task = second["result"]["task"]
        self.assertEqual(second["result"]["submission_kind"], "refill")
        self.assertEqual(task["id"], original["id"])
        self.assertEqual(task["revision"], 2)
        self.assertEqual(task["registry_version"], old.version)
        self.assertEqual(task["status"], "done")
        self.assertEqual(task["turns"], ["synthetic task", "公司"])
        brief = slow.call_args_list[0].args[0]
        self.assertEqual(brief["asked_questions"], ["家还是公司？"])
        self.assertIn("conversation", brief["context"])
        self.assertIn("navigation_state", brief["context"])
        self.assertEqual(second["result"]["decisions"]["request_count"], 0)
        self.assertNotIn("decision_ms", second["result"]["decisions"])
        judge.assert_not_awaited()
        jev_client.assert_not_called()
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 1)
        self.assertEqual(self.app.extensions["runtime"].calls[0]["version"], old.version)
        third, _, _, _ = self.submit(self.home_turns())
        self.assertEqual(third["result"]["task"]["registry_version"], "new-test-version")

    def test_refill_retains_completed_facts_without_replaying_action(self):
        first, _, _, _ = self.submit([
            reply("navi_to_special_poi", {"poi_type": "home"}),
            reply("ask_user", {"question": "是否还有其他要求？"}),
        ])
        second, _, slow, _ = self.submit(self.home_turns(), utterance="没有")
        self.assertEqual(first["result"]["task"]["id"], second["result"]["task"]["id"])
        self.assertEqual(len(slow.call_args_list[0].args[0]["completed_steps"]), 1)
        self.assertEqual(len(self.app.extensions["runtime"].calls), 1)
        self.assertEqual(second["result"]["task"]["status"], "done")

    def test_multiple_followups_still_use_one_task(self):
        first, _, _, _ = self.submit([reply("ask_user", {"question": "去哪里？"})])
        second, judge, _, _ = self.submit(
            [reply("ask_user", {"question": "请给具体地点"})], utterance="那里")
        judge.assert_not_awaited()
        third, judge, _, _ = self.submit(self.home_turns(), utterance="家")
        judge.assert_not_awaited()
        self.assertEqual(first["result"]["task"]["id"], third["result"]["task"]["id"])
        self.assertEqual(second["result"]["task"]["status"], "waiting_input")
        self.assertEqual(third["result"]["task"]["revision"], 3)
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 1)

    def test_running_and_disclosing_tasks_reject_another_submission(self):
        queue = []
        socketio = self.app.extensions["socketio"]
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch.object(socketio, "start_background_task",
                          side_effect=lambda target, *args: queue.append((target, args))):
            first = self.client.post("/api/chat", json={"utterance": "first synthetic"})
            self.assertEqual(first.status_code, 202)
            second = self.client.post("/api/chat", json={"utterance": "second synthetic"})
            self.assertEqual(second.status_code, 409)
            self.assertIn("等待", second.get_json()["error"])
            task = next(iter(self.app.extensions["ledger"].tasks.values()))
            self.assertEqual(task.status, "disclosing")
            task.status = "running"
            self.assertEqual(self.client.post("/api/chat", json={"utterance": "third"}).status_code, 409)
        self.assertEqual(len(queue), 1)
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 1)
        self.assertEqual(list(self.app.extensions["ledger"].conversation),
                         [{"role": "user", "content": "first synthetic"}])

    def test_refill_is_reserved_before_worker_starts(self):
        self.submit([reply("ask_user", {"question": "去哪里？"})])
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch.object(self.app.extensions["socketio"], "start_background_task") as worker:
            first = self.client.post("/api/chat", json={"utterance": "天安门"})
            second = self.client.post("/api/chat", json={"utterance": "故宫"})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 1)
        self.assertEqual(next(iter(self.app.extensions["ledger"].tasks.values())).revision, 2)

    def test_multiple_existing_pending_tasks_are_not_arbitrarily_refilled(self):
        ledger = self.app.extensions["ledger"]
        for text in ("one", "two"):
            task = ledger.create(text, {}, self.app.extensions["registry"].current, self.b_decision())
            task.status = "waiting_input"
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch.object(self.app.extensions["socketio"], "start_background_task") as worker:
            response = self.client.post("/api/chat", json={"utterance": "天安门"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("多个", response.get_json()["error"])
        worker.assert_not_called()
        self.assertTrue(all(t.revision == 1 for t in ledger.tasks.values()))

    def test_empty_initial_disclosure_can_expand_and_finish(self):
        with patch("app.additional_disclosure", new=AsyncMock(
            return_value=frozenset({"navi_to_special_poi"})
        )) as extra:
            job, _, _, _ = self.submit([
                reply("request_tools", {"intent": "导航回家"}), *self.home_turns(),
            ], decision=self.b_decision([]))
        self.assertEqual(job["result"]["task"]["status"], "done")
        self.assertEqual(extra.await_count, 1)
        self.assertEqual(extra.call_args.args[1].version,
                         job["result"]["task"]["registry_version"])

    def test_two_simultaneous_requests_reserve_only_one_task(self):
        barrier = Barrier(2)

        def post(text):
            with self.app.test_client() as client:
                barrier.wait(timeout=2)
                return client.post("/api/chat", json={"utterance": text}).status_code

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch.object(self.app.extensions["socketio"], "start_background_task") as worker, \
             ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(post, text) for text in ("one", "two")]
            statuses = [future.result(timeout=3) for future in futures]
        self.assertEqual(sorted(statuses), [202, 409])
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(len(self.app.extensions["ledger"].tasks), 1)
        self.assertEqual(len(self.app.extensions["ledger"].conversation), 1)

    def test_waiting_input_is_not_refillable_before_worker_publishes_outcome(self):
        ledger = self.app.extensions["ledger"]
        original_record = ledger.record
        attempts = []

        def record(role, text):
            if role == "assistant":
                attempts.append(self.client.post("/api/chat", json={"utterance": "early reply"}))
            original_record(role, text)

        with patch.object(ledger, "record", side_effect=record):
            first, _, _, _ = self.submit([reply("ask_user", {"question": "去哪里？"})])
        self.assertEqual(first["result"]["task"]["status"], "waiting_input")
        self.assertEqual(attempts[0].status_code, 409)
        second, _, _, _ = self.submit(self.home_turns(), utterance="家")
        self.assertEqual(second["result"]["task"]["id"], first["result"]["task"]["id"])
        self.assertEqual(second["result"]["task"]["status"], "done")

    def test_additional_b_failure_retains_facts_and_releases_task(self):
        with patch("app.additional_disclosure", new=AsyncMock(side_effect=ValueError("bad B"))):
            job, _, _, _ = self.submit([
                reply("navi_to_special_poi", {"poi_type": "home"}),
                reply("request_tools", {"intent": "查看路线"}),
            ])
        task = job["result"]["task"]
        self.assertEqual(task["status"], "failed")
        self.assertEqual(len(task["completed_steps"]), 1)
        self.assertFalse(task["processing"])
        self.assertEqual(len(self.app.extensions["runtime"].calls), 1)
        following, _, _, _ = self.submit(self.home_turns())
        self.assertEqual(following["result"]["task"]["status"], "done")
        self.assertNotEqual(task["id"], following["result"]["task"]["id"])

    def test_b_failure_releases_task_for_next_request(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch("app.AsyncTypeSafeClient"), \
             patch("app.judge", new=AsyncMock(side_effect=ValueError("bad response"))), \
             patch.object(self.app.extensions["socketio"], "start_background_task",
                          side_effect=lambda target, *args: target(*args)):
            response = self.client.post("/api/chat", json={"utterance": "synthetic"})
        job = self.client.get("/api/jobs/" + response.get_json()["job_id"]).get_json()
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["task"]["status"], "failed")
        self.assertFalse(job["task"]["processing"])
        self.assertEqual(self.app.extensions["ledger"].pending(), {})
        next_job, _, _, _ = self.submit(self.home_turns())
        self.assertEqual(next_job["result"]["task"]["status"], "done")

    def test_slow_failure_releases_task_and_never_calls_fast(self):
        job, _, _, _ = self.submit([reply("finish")])
        self.assertEqual(job["result"]["task"]["status"], "failed")
        self.assertFalse(job["result"]["task"]["processing"])
        self.assertEqual(self.app.extensions["ledger"].pending(), {})

    def test_worker_start_failure_releases_reservation(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "unit-test-key"}), \
             patch.object(self.app.extensions["socketio"], "start_background_task",
                          side_effect=RuntimeError("synthetic failure")):
            response = self.client.post("/api/chat", json={"utterance": "synthetic"})
        self.assertEqual(response.status_code, 503)
        task = next(iter(self.app.extensions["ledger"].tasks.values()))
        self.assertEqual(task.status, "failed")
        self.assertFalse(task.processing)

    def test_decision_preview_runs_b_without_creating_task_or_calling_model(self):
        job, judge, slow, _ = self.submit([], endpoint="/api/decide")
        self.assertEqual(job["status"], "decided")
        self.assertEqual(judge.await_count, 1)
        slow.assert_not_called()
        self.assertNotIn("task", job["result"])
        self.assertEqual(self.app.extensions["ledger"].tasks, {})
        self.assertEqual(list(self.app.extensions["ledger"].conversation), [])

    def test_invalid_input_and_missing_credentials_are_explicit(self):
        self.assertEqual(self.client.post("/api/chat", json={"utterance": ""}).status_code, 400)
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post("/api/chat", json={"utterance": "synthetic"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
