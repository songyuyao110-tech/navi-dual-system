"""Offline 5.3 absolute deadline/chunk/queue checks on actual core."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from registry import load_registry
from task_context import BoundedDispatchLedger, BoundedSendFixture

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.contract = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
        self.snapshot = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
        self.ledger = BoundedDispatchLedger(self.contract, clock=self.clock)
        self.task = self.ledger.create("synthetic", {}, self.snapshot,
                                      {"route": "slow", "disclosed_tools": ["navi_start_route"]})
        self.owner = self.ledger.capture_worker(self.task.id)
        self.identity = dict(actor="synthetic", session="synthetic", device="synthetic",
                             contract_version=self.contract["contract_version"], event_source="synthetic-fixture")
        self.calls = []

    def prepare(self, **changes):
        values = dict(task_deadline=10.0, send_not_after=8.0, **self.identity)
        values.update(changes)
        return self.ledger.prepare_operation(self.owner, "bounded-call", "navi_start_route",
                                             {"end_name": "synthetic"}, **values)

    def write(self, payload):
        self.calls.append(payload)
        return len(payload)

    def held(self, operation):
        self.assertEqual(self.ledger.resource_owner(operation.device, "navigation"), operation.operation_id)
        self.assertFalse(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_immutable_absolute_window_and_no_task_renewal(self):
        operation = self.prepare()
        window = self.ledger.dispatch_window(operation.operation_id)
        self.assertEqual((window.task_deadline, window.send_not_after), (10.0, 8.0))
        with self.assertRaises(FrozenInstanceError): window.send_not_after = 20
        self.assertEqual(self.ledger.capture_deadline(self.owner, 10), 10)
        for value in (9, 11):
            with self.assertRaises(ValueError): self.ledger.capture_deadline(self.owner, value)
        self.ledger.revise(self.task.id, "revised", {})
        with self.assertRaises(ValueError):
            self.ledger.capture_deadline(self.ledger.capture_worker(self.task.id), 20)

    def test_invalid_and_already_expired_prepare_never_reserves(self):
        for value in (True, float("nan"), float("inf"), -1, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError): self.prepare(task_deadline=value)
        with self.assertRaises(ValueError): self.prepare(send_not_after=11)
        with self.assertRaises(TimeoutError): self.prepare(send_not_after=0)
        self.clock.now = 10
        with self.assertRaises(TimeoutError): self.prepare()
        self.assertIsNone(self.ledger.resource_owner("synthetic", "navigation"))

    def test_outer_phase_rejects_late_result_without_library_timeout(self):
        self.ledger.capture_deadline(self.owner, 10)
        def slow(): self.clock.now = 10;return {"late": True}
        with self.assertRaises(TimeoutError): self.ledger.run_phase(self.owner, slow)
        with self.assertRaises(TimeoutError): self.ledger.run_phase(self.owner, lambda: self.fail("expired phase called"))
        self.assertEqual(self.task.plans, [])

    def test_outer_phase_rejects_rotated_owner_result(self):
        self.ledger.capture_deadline(self.owner, 10)
        def stale(): self.ledger.rotate_worker(self.task.id);return {"stale": True}
        with self.assertRaises(ValueError): self.ledger.run_phase(self.owner, stale)
        self.assertEqual(self.task.plans, [])

    def test_exact_send_boundary_is_expired_zero_write_positive_local_proof(self):
        operation = self.prepare()
        fixture = BoundedSendFixture(self.ledger, operation)
        self.clock.now = 8
        with self.assertRaises(TimeoutError): fixture.send_chunk(b"x", self.write)
        self.assertEqual(self.calls, [])
        self.held(operation)
        receipt = fixture.revoke_before_handoff()
        self.assertTrue(self.ledger.accept_not_sent(receipt))
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_continuous_small_chunks_do_not_reset_absolute_budget(self):
        operation = self.prepare()
        fixture = BoundedSendFixture(self.ledger, operation)
        def slow_chunk(payload):
            self.calls.append(payload);self.clock.now += 3;return len(payload)
        with self.assertRaises(TimeoutError): fixture.send_chunks([b"a", b"b", b"c", b"d"], slow_chunk)
        self.assertEqual(self.calls, [b"a", b"b", b"c"])
        with self.assertRaises(TimeoutError): fixture.send_chunk(b"never", self.write)
        self.assertEqual(len(self.calls), 3)
        self.held(operation)
        self.assertIsNone(fixture.atomic_reject_unforwardable())

    def test_write_crossing_deadline_is_unknown_not_sent_false(self):
        operation = self.prepare();fixture = BoundedSendFixture(self.ledger, operation)
        def crossing(payload): self.clock.now = 9;self.calls.append(payload);return len(payload)
        with self.assertRaises(TimeoutError): fixture.send_chunk(b"written", crossing)
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "unknown")
        self.held(operation)
        self.assertIsNone(fixture.atomic_reject_unforwardable())

    def test_queue_wait_expires_before_dequeue_atomic_rejection_proves_no_forward(self):
        operation = self.prepare();fixture = BoundedSendFixture(self.ledger, operation)
        fixture.enqueue();self.clock.now = 8
        with self.assertRaises(TimeoutError): fixture.dequeue_and_send(b"queued", self.write)
        self.assertEqual(self.calls, [])
        self.held(operation)
        receipt = fixture.atomic_reject_unforwardable()
        self.assertTrue(self.ledger.accept_not_sent(receipt))
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_queue_before_deadline_sends_and_uncertain_accounting_never_proves_no_send(self):
        operation = self.prepare();fixture = BoundedSendFixture(self.ledger, operation)
        fixture.enqueue();self.clock.now = 7
        self.assertEqual(fixture.dequeue_and_send(b"ok", self.write), 2)
        self.held(operation)
        self.assertIsNone(fixture.atomic_reject_unforwardable())

    def test_write_error_unknown_or_inflight_cannot_issue_unreachability_receipt(self):
        operation = self.prepare();fixture = BoundedSendFixture(self.ledger, operation)
        def uncertain(payload):
            self.assertIsNone(fixture.atomic_reject_unforwardable())
            self.clock.now = 11
            raise RuntimeError("synthetic write accounting failed")
        with self.assertRaises(RuntimeError): fixture.send_chunk(b"maybe", uncertain)
        self.held(operation)
        self.assertIsNone(fixture.atomic_reject_unforwardable())
        self.assertIsNone(fixture.revoke_before_handoff())

    def test_expired_or_cancelled_queued_effect_cannot_mutate_world(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                self.setUp();operation=self.prepare();fixture=BoundedSendFixture(self.ledger,operation)
                fixture.enqueue()
                if cancelled: self.ledger.cancel(self.task.id)
                else: self.clock.now=10
                with self.assertRaises((TimeoutError, ValueError)): fixture.deliver_verified_effect()
                self.assertNotIn(operation.device, self.ledger._fixture_world)
                self.held(operation)

    def _assert_late_unknown_preserves_terminal_A_and_B(self, terminal):
        a = self.prepare()
        fixture_a = BoundedSendFixture(self.ledger, a)
        fixture_a.enqueue()
        def fact(operation, event_id, status, verified=True, final=True):
            return self.ledger.append_operation_fact(operation.operation_id, call_id=operation.call_id,
                event_id=event_id, fact={"status": status, "verified": verified, "final": final}, **self.identity)
        fact(a, "terminal-A", terminal)
        self.assertTrue(self.ledger.release_resources(a.operation_id, a.cleanup_token))
        task_b = self.ledger.create("synthetic B", {}, self.snapshot,
                                   {"route": "slow", "disclosed_tools": ["navi_start_route"]})
        b = self.ledger.prepare_operation(self.ledger.capture_worker(task_b.id), "B-call", "navi_start_route",
            {"end_name": "synthetic B"}, task_deadline=10, send_not_after=8, **self.identity)
        BoundedSendFixture(self.ledger, b).deliver_verified_effect()
        before_a, before_b = self.task.public(), task_b.public()
        self.assertLess(self.clock(), self.ledger.dispatch_window(a.operation_id).send_not_after)
        fact(a, "late-unknown-A", "unknown", verified=False, final=False)
        self.assertEqual(self.ledger.operation_facts(a.operation_id)[-1]["status"], "unknown")
        self.assertEqual(self.ledger.operation_state(a.operation_id), terminal)
        with self.assertRaises(RuntimeError): fixture_a.deliver_verified_effect()
        self.assertEqual(self.ledger._fixture_world[a.device], b.operation_id)
        self.assertEqual(self.ledger.resource_owner(a.device, "navigation"), b.operation_id)
        self.assertTrue(self.ledger.release_resources(a.operation_id, a.cleanup_token))
        self.assertEqual(self.ledger.resource_owner(a.device, "navigation"), b.operation_id)
        self.assertEqual(self.task.public(), before_a)
        self.assertEqual(task_b.public(), before_b)

    def test_late_unknown_cannot_resurrect_verified_final_failed_A_over_B(self):
        self._assert_late_unknown_preserves_terminal_A_and_B("failed")

    def test_late_unknown_cannot_resurrect_verified_final_rejected_A_over_B(self):
        self._assert_late_unknown_preserves_terminal_A_and_B("rejected")

    def test_timely_synthetic_effect_still_valid_and_late_facts_do_not_revive_task(self):
        operation=self.prepare();fixture=BoundedSendFixture(self.ledger,operation)
        fixture.enqueue();self.clock.now=7
        fixture.deliver_verified_effect()
        self.assertEqual(self.ledger._fixture_world[operation.device],operation.operation_id)
        self.assertTrue(self.ledger.release_resources(operation.operation_id,operation.cleanup_token))
        self.ledger.cancel(self.task.id);self.clock.now=20;before=self.task.public()
        self.ledger.append_operation_fact(operation.operation_id,call_id=operation.call_id,event_id="late-accepted",
                                          fact={"status":"accepted"},**self.identity)
        self.assertEqual(self.ledger.operation_state(operation.operation_id),"effective")
        self.assertEqual(self.task.public(),before)


if __name__ == "__main__": unittest.main()
