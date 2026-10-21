"""Offline Step 5.2: actual core and synthetic fixture, no real sender."""
from dataclasses import FrozenInstanceError, asdict, replace
import inspect
import json
from pathlib import Path
from queue import Queue
from threading import Barrier, Thread
import textwrap
import unittest

from execution import Runtime
from registry import load_registry
import task_context
from task_context import NotSentReceipt, OperationLedger, SendEvidenceLedger, SyntheticSendFixture

ROOT = Path(__file__).resolve().parents[1]


class SendBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
        self.snapshot = load_registry(ROOT / "tools.yaml", frozenset(Runtime().handlers))
        self.ledger = SendEvidenceLedger(self.contract)
        self.task = self.create_task()
        self.owner = self.ledger.capture_worker(self.task.id)
        self.identity = dict(actor="synthetic-actor", session="synthetic-session", device="synthetic-device",
                             contract_version=self.contract["contract_version"], event_source="synthetic-fixture")
        self.sequence = 0

    def create_task(self):
        return self.ledger.create("synthetic task", {}, self.snapshot,
                                 {"route": "slow", "disclosed_tools": list(self.snapshot.tools)})

    def prepare(self, *, owner=None, call_id=None, tool="navi_start_route", arguments=None, **overrides):
        self.sequence += 1
        return self.ledger.prepare_operation(
            self.owner if owner is None else owner, call_id or f"call-{self.sequence}", tool,
            {"end_name": "synthetic"} if arguments is None else arguments, **{**self.identity, **overrides})

    def append(self, operation, status, *, verified=True, final=True, **overrides):
        self.sequence += 1
        identity = {name: getattr(operation, name) for name in self.identity}
        return self.ledger.append_operation_fact(
            operation.operation_id, event_id=f"event-{self.sequence}", call_id=operation.call_id,
            fact={"status": status, "verified": verified, "final": final}, **{**identity, **overrides})

    def assert_held(self, operation, state="unknown"):
        self.assertEqual(self.ledger.operation_state(operation.operation_id), state)
        self.assertFalse(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))
        for resource in operation.resources:
            self.assertEqual(self.ledger.resource_owner(operation.device, resource), operation.operation_id)

    def unknown_scenario(self, action, *, cancel=False):
        operation = self.prepare()
        fixture = SyntheticSendFixture(self.ledger, operation)
        before = self.task.public()
        if action == "write_then_accounting_error":
            with self.assertRaises(RuntimeError):
                fixture.write_then_accounting_error()
        else:
            getattr(fixture, action)()
        if cancel:
            self.ledger.cancel(self.task.id)
        else:
            self.assertEqual(self.task.public(), before)
        self.assert_held(operation)
        return operation, fixture

    def test_parameter_effects_readonly_ui_idempotent_and_unknown_resources(self):
        self.assertIsInstance(self.ledger, OperationLedger)
        cases = (
            ("navi_search_keyword", {"keyword": "synthetic", "show_in_map": False}, "read_only", ()),
            ("navi_search_keyword", {"keyword": "synthetic", "show_in_map": True}, "ui_write", ("navigation_ui",)),
            ("navi_search_keyword", {"keyword": "synthetic"}, "unknown_effect", ("navigation_ui",)),
            ("navi_open_address_page", {"type": "home"}, "ui_write", ("navigation_ui",)),
        )
        for index, (tool, arguments, kind, resources) in enumerate(cases):
            with self.subTest(tool=tool, arguments=arguments):
                operation = self.prepare(tool=tool, arguments=arguments, device=f"effect-{index}")
                self.assertEqual(operation.resources, resources)
                captured = self.ledger.operation_effect(operation.operation_id)
                self.assertEqual(captured, {"kind": kind, "resources": list(resources)})
                captured["resources"].append("caller-mutation")
                self.assertEqual(self.ledger.operation_effect(operation.operation_id),
                                 {"kind": kind, "resources": list(resources)})
                self.assertEqual(self.ledger.operation_state(operation.operation_id), "prepared")
                for resource in resources:
                    self.assertEqual(self.ledger.resource_owner(operation.device, resource), operation.operation_id)
        tool = next(t for t in self.contract["tools"] if t["name"] == "navi_open_address_page")
        self.assertEqual(tool["side_effect"], "idempotent")
        read = self.prepare(tool="navi_search_keyword", arguments={"keyword": "synthetic", "show_in_map": False},
                            device="effect-1")
        self.assertEqual(read.resources, ())  # Explicit no-UI search can run while UI write is held.
        for device in ("effect-1", "effect-2", "effect-3"):
            with self.subTest(held_device=device), self.assertRaises(ValueError):
                self.prepare(tool="navi_open_address_page", arguments={"type": "home"}, device=device)

    def test_captured_arguments_contract_and_no_arbitrary_resource_override(self):
        arguments = {"end_name": "synthetic", "via_points": [{"name": "A"}]}
        operation = self.prepare(arguments=arguments)
        digest = operation.parameters_sha256
        arguments["via_points"][0]["name"] = "changed"
        next(t for t in self.contract["tools"] if t["name"] == "navi_start_route")["effect"] = {"kind": "read_only", "resources": []}
        self.assertEqual(operation.parameters_sha256, digest)
        self.assertEqual(operation.resources, ("navigation",))
        with self.assertRaises((TypeError, ValueError)):
            self.ledger.register_operation(self.owner, "override", "navi_start_route", {"end_name": "B"},
                                           resources=(), **self.identity)
        with self.assertRaises(ValueError):
            self.prepare(call_id="blocked-after-contract-mutation")
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])

    def test_same_device_cross_task_actor_session_blocks_write_allows_readonly(self):
        operation = self.prepare()
        SyntheticSendFixture(self.ledger, operation).enqueue()
        owner_b = self.ledger.capture_worker(self.create_task().id)
        other = dict(actor="other-actor", session="other-session")
        with self.assertRaises(ValueError):
            self.prepare(owner=owner_b, call_id="B", **other)
        self.assert_held(operation)
        read = self.prepare(owner=owner_b, call_id="B", tool="navi_get_status", arguments={}, **other)
        self.assertEqual(read.resources, ())
        self.assertEqual(self.ledger.operation_facts(read.operation_id), [])
        independent = self.prepare(owner=owner_b, device="other-device", **other)
        self.assertEqual(self.ledger.resource_owner("other-device", "navigation"), independent.operation_id)

    def test_prepared_label_sent_false_without_receipt_never_free(self):
        operation = self.prepare()
        self.assert_held(operation, "prepared")
        with self.assertRaises(ValueError):
            self.ledger.accept_not_sent(dict(operation_id=operation.operation_id, cleanup_token=operation.cleanup_token,
                                            sent=False, status="prepared"))
        self.append(operation, "unknown", verified=False, final=False)
        self.assert_held(operation)

    def test_enqueue_then_task_cancel_stays_unknown(self):
        operation, fixture = self.unknown_scenario("enqueue", cancel=True)
        self.assertIsNone(fixture.revoke_before_handoff())
        self.assert_held(operation)

    def test_partial_write_stays_unknown(self):
        self.unknown_scenario("partial_write")

    def test_write_then_accounting_error_stays_unknown(self):
        self.unknown_scenario("write_then_accounting_error")

    def test_hidden_retry_stays_unknown(self):
        self.unknown_scenario("hidden_retry")

    def test_cancel_done_slot_cleanup_never_release_device_resource(self):
        for transition in ("cancel", "done"):
            with self.subTest(transition=transition):
                task = self.create_task()
                owner = self.ledger.capture_worker(task.id)
                operation = self.prepare(owner=owner, device=transition)
                self.assertTrue(self.ledger.claim_slot(transition, owner))
                SyntheticSendFixture(self.ledger, operation).enqueue()
                if transition == "cancel":
                    self.ledger.cancel(task.id)
                else:
                    with self.ledger.lock:
                        task.status = "done"
                before = task.public()
                self.assertTrue(self.ledger.release_slot(transition, owner, io_safe=True))
                self.assertIsNone(self.ledger.slot_owner(transition))
                self.assert_held(operation)
                self.assertEqual(task.public(), before)

    def test_foreign_cleanup_token_cannot_handoff_or_release(self):
        operation = self.prepare()
        with self.assertRaises(ValueError):
            self.ledger.mark_handoff(operation.operation_id, "foreign")
        self.assert_held(operation, "prepared")
        self.ledger.mark_handoff(operation.operation_id, operation.cleanup_token)
        self.assert_held(operation)
        self.append(operation, "effective")
        self.assertFalse(self.ledger.release_resources(operation.operation_id, "foreign"))
        self.assertEqual(self.ledger.resource_owner(operation.device, "navigation"), operation.operation_id)
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_revoke_before_handoff_not_sent_allows_B_and_old_cleanup_cannot_clear_B(self):
        operation = self.prepare()
        receipt = SyntheticSendFixture(self.ledger, operation).revoke_before_handoff()
        self.ledger.cancel(self.task.id)  # Cleanup authority survives task revision/token rotation.
        before = self.task.public()
        self.assertIsInstance(receipt, NotSentReceipt)
        self.assertEqual(receipt.operation_id, operation.operation_id)
        self.assertEqual(receipt.cleanup_token, operation.cleanup_token)
        self.assertTrue(self.ledger.accept_not_sent(receipt))
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
        self.assertEqual(self.task.public(), before)
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))
        self.assertIsNone(self.ledger.resource_owner(operation.device, "navigation"))
        operation_b = self.prepare(owner=self.ledger.capture_worker(self.create_task().id))
        self.ledger.release_resources(operation.operation_id, operation.cleanup_token)
        self.assertEqual(self.ledger.resource_owner(operation.device, "navigation"), operation_b.operation_id)

    def test_atomic_queue_zero_bytes_rejection_proves_not_sent(self):
        operation = self.prepare()
        fixture = SyntheticSendFixture(self.ledger, operation)
        fixture.enqueue()
        self.assert_held(operation)
        receipt = fixture.atomic_reject_unforwardable()
        self.assertIsInstance(receipt, NotSentReceipt)
        self.assertTrue(self.ledger.accept_not_sent(receipt))
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))
        self.assertIsNone(self.ledger.resource_owner(operation.device, "navigation"))

    def test_actual_bytes_and_forwarded_retry_preclude_proof(self):
        for action in ("partial_write", "hidden_retry", "write_then_accounting_error"):
            with self.subTest(action=action):
                operation = self.prepare(device=action)
                fixture = SyntheticSendFixture(self.ledger, operation)
                if action == "write_then_accounting_error":
                    with self.assertRaises(RuntimeError):
                        fixture.write_then_accounting_error()
                else:
                    getattr(fixture, action)()
                self.assertIsNone(fixture.revoke_before_handoff())
                self.assertIsNone(fixture.atomic_reject_unforwardable())
                self.assert_held(operation)

    def test_both_closed_fixture_paths_cannot_send_later(self):
        for closure in ("revoke_before_handoff", "atomic_reject_unforwardable"):
            with self.subTest(closure=closure):
                operation = self.prepare(device=closure)
                fixture = SyntheticSendFixture(self.ledger, operation)
                if closure == "atomic_reject_unforwardable":
                    fixture.enqueue()
                receipt = getattr(fixture, closure)()
                with self.assertRaises(ValueError):
                    SyntheticSendFixture(self.ledger, operation)  # No hidden second route can reopen the same operation.
                self.assertTrue(self.ledger.accept_not_sent(receipt))
                for action in ("enqueue", "partial_write", "hidden_retry", "write_then_accounting_error"):
                    with self.subTest(action=action), self.assertRaises(RuntimeError):
                        getattr(fixture, action)()
                self.assertEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
                self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_receipt_immutable_registered_not_dict_forged_crossop_or_replay(self):
        operation = self.prepare()
        receipt = SyntheticSendFixture(self.ledger, operation).revoke_before_handoff()
        with self.assertRaises(FrozenInstanceError):
            receipt.kind = "forged"
        for forged in (asdict(receipt), NotSentReceipt(**asdict(receipt)),
                       *(replace(receipt, **{field: "foreign"}) for field in
                         ("operation_id", "cleanup_token", "fixture_id", "receipt_id", "kind"))):
            with self.subTest(forged_type=type(forged).__name__), self.assertRaises(ValueError):
                self.ledger.accept_not_sent(forged)
        other = self.prepare(device="other-device")
        SyntheticSendFixture(self.ledger, other).partial_write()
        with self.assertRaises(ValueError):
            self.ledger.accept_not_sent(replace(receipt, operation_id=other.operation_id,
                                                cleanup_token=other.cleanup_token))
        self.assert_held(operation, "prepared")
        self.assert_held(other)
        self.assertTrue(self.ledger.accept_not_sent(receipt))
        self.assertFalse(self.ledger.accept_not_sent(receipt))
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
        self.assert_held(other)

    def test_receipt_cannot_override_terminal_correlated_effect(self):
        operation = self.prepare()
        receipt = SyntheticSendFixture(self.ledger, operation).revoke_before_handoff()
        self.append(operation, "effective")
        before = self.task.public()
        with self.assertRaises(ValueError):
            self.ledger.accept_not_sent(receipt)
        self.assertNotEqual(self.ledger.operation_state(operation.operation_id), "not_sent")
        self.assertEqual(self.task.public(), before)

    def test_late_verified_effective_resolves_unknown_old_accepted_cannot_downgrade(self):
        operation = self.prepare()
        fixture = SyntheticSendFixture(self.ledger, operation)
        fixture.enqueue()
        self.append(operation, "accepted")
        self.assert_held(operation, "accepted")
        self.ledger.revise(self.task.id, "synthetic revision", {})
        self.assertTrue(self.ledger.commit_plan(self.ledger.capture_worker(self.task.id), {"current": True}))
        self.ledger.cancel(self.task.id)
        before = self.task.public()
        fixture.deliver_verified_effect()
        self.append(operation, "accepted")
        self.assertEqual(self.ledger.operation_state(operation.operation_id), "effective")
        self.assertEqual(self.task.public(), before)
        self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))

    def test_late_accepted_cannot_resurrect_trusted_failed_or_rejected_A_over_B(self):
        for terminal in ("failed", "rejected"):
            with self.subTest(terminal=terminal):
                device = "resurrection-" + terminal
                a = self.prepare(device=device)
                sender_a = SyntheticSendFixture(self.ledger, a)
                sender_a.enqueue()
                self.append(a, terminal, verified=True, final=True)
                self.assertTrue(self.ledger.release_resources(a.operation_id, a.cleanup_token))
                task_b = self.create_task()
                owner_b = self.ledger.capture_worker(task_b.id)
                b = self.prepare(owner=owner_b, device=device, arguments={"end_name": "synthetic B"})
                SyntheticSendFixture(self.ledger, b).deliver_verified_effect()
                self.assertEqual(self.ledger._fixture_world[device], b.operation_id)
                before_a, before_b = self.task.public(), task_b.public()
                self.append(a, "accepted", verified=False, final=False)
                self.assertEqual(self.ledger.operation_facts(a.operation_id)[-1]["status"], "accepted")
                self.assertEqual(self.ledger.operation_state(a.operation_id), terminal)
                with self.assertRaises(RuntimeError):
                    sender_a.deliver_verified_effect()
                self.assertEqual(self.ledger._fixture_world[device], b.operation_id)
                for resource in b.resources:
                    self.assertEqual(self.ledger.resource_owner(device, resource), b.operation_id)
                self.assertTrue(self.ledger.release_resources(a.operation_id, a.cleanup_token))
                for resource in b.resources:
                    self.assertEqual(self.ledger.resource_owner(device, resource), b.operation_id)
                self.assertEqual(self.task.public(), before_a)
                self.assertEqual(task_b.public(), before_b)

    def test_unverified_nonfinal_facts_cannot_free_and_failed_effective_conflicts(self):
        for status in ("effective", "failed", "rejected"):
            for index, (verified, final) in enumerate(((False, True), (True, False), (None, None), (1, 1))):
                with self.subTest(status=status, verified=verified, final=final):
                    operation = self.prepare(device=f"{status}-{index}")
                    SyntheticSendFixture(self.ledger, operation).enqueue()
                    self.append(operation, status, verified=verified, final=final)
                    self.assertFalse(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))
                    self.assertEqual(self.ledger.resource_owner(operation.device, "navigation"), operation.operation_id)
        for status in ("failed", "rejected"):
            operation = self.prepare(device="final-" + status)
            self.append(operation, status)
            self.assertTrue(self.ledger.release_resources(operation.operation_id, operation.cleanup_token))
        operation = self.prepare(device="contradiction")
        self.append(operation, "failed")
        self.append(operation, "effective")
        self.assert_held(operation, "conflict")

    def test_invalid_metadata_fact_no_state_facts_resource_task_mutation(self):
        operation = self.prepare()
        SyntheticSendFixture(self.ledger, operation).enqueue()
        before = self.task.public()
        for field in ("actor", "session", "device", "contract_version", "event_source"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.append(operation, "effective", **{field: "foreign"})
        with self.assertRaises(ValueError):
            self.ledger.append_operation_fact(operation.operation_id, call_id="foreign", event_id="wrong-call",
                                              fact={"status": "effective", "verified": True, "final": True}, **self.identity)
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])
        self.assertEqual(self.task.public(), before)
        self.assert_held(operation)

    def test_two_worker_barrier_atomic_reservation_one_winner_no_orphan(self):
        owner_b = self.ledger.capture_worker(self.create_task().id)
        barrier = Barrier(2, timeout=2)
        arrivals, results, errors = Queue(), Queue(), Queue()

        def worker(index, owner):
            try:
                barrier.wait()
                arrivals.put(index)
                try:
                    operation = self.ledger.prepare_operation(owner, f"race-{index}", "navi_start_route",
                                                             {"end_name": "synthetic"}, **self.identity)
                    results.put((index, operation))
                except ValueError:
                    results.put((index, None))
            except BaseException as error:
                errors.put(error)

        workers = [Thread(target=worker, args=(index, owner), daemon=True)
                   for index, owner in enumerate((self.owner, owner_b))]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=3)
        self.assertFalse(any(thread.is_alive() for thread in workers), "worker leaked")
        self.assertTrue(errors.empty(), f"worker exception: {list(errors.queue)!r}")
        self.assertEqual(sorted(arrivals.queue), [0, 1], "both workers must reach barrier")
        outcomes = dict(results.queue)
        self.assertEqual(set(outcomes), {0, 1})
        winners = [(index, op) for index, op in outcomes.items() if op is not None]
        self.assertEqual(len(winners), 1)
        index, winner = winners[0]
        self.assertEqual(self.ledger.resource_owner(winner.device, "navigation"), winner.operation_id)
        self.assertEqual(self.ledger.operation_facts(winner.operation_id), [])
        read = self.ledger.prepare_operation((self.owner, owner_b)[1-index], f"race-{1-index}",
                                             "navi_get_status", {}, **self.identity)
        self.assertEqual(self.ledger.operation_facts(read.operation_id), [])
        self.assertEqual(len(self.ledger._operations), 2)
        self.assertEqual(set(self.ledger._facts), {winner.operation_id, read.operation_id})

    def test_unknown_release_negative_control_detects_real_source_mutant(self):
        assert_unknown_release_retained(SendEvidenceLedger)
        mutant = unknown_release_mutant_class()
        with self.assertRaises(AssertionError):
            assert_unknown_release_retained(mutant)


def assert_unknown_release_retained(ledger_class=SendEvidenceLedger):
    """Standalone oracle: mutant callers leave AssertionError uncaught for exit 1."""
    contract = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
    snapshot = load_registry(ROOT / "tools.yaml", frozenset(Runtime().handlers))
    ledger = ledger_class(contract)
    task = ledger.create("synthetic negative control", {}, snapshot,
                         {"route": "slow", "disclosed_tools": ["navi_start_route"]})
    operation = ledger.prepare_operation(
        ledger.capture_worker(task.id), "negative-control-call", "navi_start_route", {"end_name": "synthetic"},
        actor="synthetic-actor", session="synthetic-session", device="synthetic-device",
        contract_version=contract["contract_version"], event_source="synthetic-fixture")
    SyntheticSendFixture(ledger, operation).enqueue()
    assert ledger.operation_state(operation.operation_id) == "unknown", "control did not enter unknown"
    released = ledger.release_resources(operation.operation_id, operation.cleanup_token)
    assert released is False, "unknown operation resources were wrongly released"
    assert ledger.resource_owner(operation.device, "navigation") == operation.operation_id, "unknown reservation disappeared"


def unknown_release_mutant_class():
    """Compile actual release source in copied module globals; never mock gold."""
    source = textwrap.dedent(inspect.getsource(SendEvidenceLedger.release_resources))
    selector = 'state not in {"not_sent", "effective", "failed", "rejected"} or operation_id not in self._resolved'
    if source.count(selector) != 1:
        raise AssertionError("real release_resources selector must match exactly once")
    namespace = dict(vars(task_context))
    exec(compile(source.replace(selector, "False", 1), "<unknown-release-source-mutant>", "exec"), namespace)
    return type("UnknownReleaseSourceMutant", (SendEvidenceLedger,),
                {"release_resources": namespace["release_resources"]})


if __name__ == "__main__":
    unittest.main()
