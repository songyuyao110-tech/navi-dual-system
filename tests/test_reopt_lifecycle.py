"""Offline Step 5.1: operation facts, plan CAS, and local slot ownership only."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from queue import Queue
from threading import Barrier, Thread
import unittest

from execution import Runtime
from registry import load_registry
from task_context import OperationLedger, TaskLedger, WorkerOwner


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        runtime = Runtime()
        self.snapshot = load_registry(
            Path(__file__).resolve().parents[1] / "tools.yaml", frozenset(runtime.handlers)
        )
        self.ledger = OperationLedger()
        self.task = self.create_task()
        self.owner = self.ledger.capture_worker(self.task.id)
        self.identity = {
            "actor": "synthetic-actor", "session": "synthetic-session",
            "device": "synthetic-device", "contract_version": "synthetic-v1",
            "event_source": "synthetic-device-events",
        }

    def create_task(self):
        return self.ledger.create(
            "synthetic navigation task", {"locale": "zh-CN"}, self.snapshot,
            {"route": "slow", "disclosed_tools": ["navi_start_route", "navi_get_status"]},
        )

    def register(self, *, owner=None, call_id="call-1", arguments=None, **overrides):
        return self.ledger.register_operation(
            self.owner if owner is None else owner, call_id, "navi_start_route",
            {"end_name": "synthetic destination"} if arguments is None else arguments,
            **{**self.identity, **overrides},
        )

    def append(self, operation, *, event_id="event-1", fact=None, **overrides):
        return self.ledger.append_operation_fact(
            operation.operation_id, call_id=operation.call_id, event_id=event_id,
            **{**self.identity, **overrides},
            fact={"status": "effective"} if fact is None else fact,
        )

    def race(self, first, second):
        """Both workers must cross a real bounded barrier before invoking core APIs."""
        barrier = Barrier(2, timeout=2)
        reached, results, errors = Queue(), Queue(), Queue()

        def worker(index, action):
            try:
                barrier.wait()
                reached.put(index)
                results.put((index, action()))
            except BaseException as error:
                errors.put((index, error))

        workers = [Thread(target=worker, args=(index, action), daemon=True)
                   for index, action in enumerate((first, second))]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=3)
        self.assertFalse(any(thread.is_alive() for thread in workers), "worker leaked")
        failures = []
        while not errors.empty():
            failures.append(errors.get_nowait())
        self.assertEqual(failures, [], f"worker exceptions: {failures!r}")
        arrivals = []
        while not reached.empty():
            arrivals.append(reached.get_nowait())
        self.assertEqual(sorted(arrivals), [0, 1], "both workers must reach the barrier")
        outcomes = {}
        while not results.empty():
            index, outcome = results.get_nowait()
            outcomes[index] = outcome
        self.assertEqual(set(outcomes), {0, 1})
        return outcomes[0], outcomes[1]

    def test_capture_is_idempotent_frozen_and_rotation_invalidates_same_revision(self):
        self.assertIsInstance(self.ledger, TaskLedger)
        self.assertIsInstance(self.owner, WorkerOwner)
        self.assertEqual(self.ledger.capture_worker(self.task.id), self.owner)
        self.assertEqual({field.name for field in fields(self.owner)},
                         {"task_id", "revision", "token"})
        with self.assertRaises(FrozenInstanceError):
            self.owner.token = "replacement"
        current = self.ledger.rotate_worker(self.task.id)
        self.assertEqual(current.task_id, self.owner.task_id)
        self.assertEqual(current.revision, self.owner.revision)
        self.assertNotEqual(current.token, self.owner.token)
        self.assertEqual(self.ledger.capture_worker(self.task.id), current)
        self.assertFalse(self.ledger.commit_plan(self.owner, {"stale": True}))
        self.assertFalse(self.ledger.claim_slot("business", self.owner))
        unknown = replace(current, task_id="unregistered-task")
        self.assertFalse(self.ledger.commit_plan(unknown, {}))
        self.assertFalse(self.ledger.claim_slot("business", unknown))
        self.assertEqual(self.task.plans, [])

    def test_unregistered_or_invalid_worker_token_cannot_claim_commit_or_register(self):
        for token in (None, "", " ", True, 1, "unissued-token"):
            with self.subTest(token_type=type(token).__name__):
                task = self.create_task()  # Deliberately no capture_worker registration.
                owner = WorkerOwner(task.id, task.revision, token)
                self.assertFalse(self.ledger.commit_plan(owner, {"invalid-owner": True}))
                self.assertFalse(self.ledger.claim_slot("unissued", owner))
                with self.assertRaises(ValueError):
                    self.register(owner=owner, call_id="unissued-" + task.id)
                self.assertEqual(task.plans, [])
                self.assertIsNone(self.ledger.slot_owner("unissued"))
                issued = self.ledger.capture_worker(task.id)
                self.assertTrue(self.ledger.claim_slot("unissued", issued))
                self.assertTrue(self.ledger.release_slot("unissued", issued, io_safe=True))

    def test_revision_type_is_exact_for_owner_release_and_embedded_fact(self):
        operation = self.register()
        self.assertTrue(self.ledger.claim_slot("typed-revision", self.owner))
        for revision in (True, 1.0, "1", None):
            with self.subTest(revision_type=type(revision).__name__):
                malformed = replace(self.owner, revision=revision)
                self.assertFalse(self.ledger.commit_plan(malformed, {"wrong-revision": True}))
                self.assertFalse(self.ledger.claim_slot("wrong-type", malformed))
                self.assertFalse(self.ledger.release_slot("typed-revision", malformed, io_safe=True))
                with self.assertRaises(ValueError):
                    self.register(owner=malformed, call_id="wrong-type-" + type(revision).__name__)
                with self.assertRaises(ValueError):
                    self.append(operation, event_id="typed-" + type(revision).__name__,
                                fact={"status": "effective", "revision": revision})
        self.assertEqual(self.ledger.slot_owner("typed-revision"), self.owner)
        self.assertEqual(self.task.plans, [])
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])
        self.assertTrue(self.append(operation, fact={"status": "effective", "revision": 1}))
        self.assertTrue(self.ledger.release_slot("typed-revision", self.owner, io_safe=True))
        for revision in (True, 1.0):
            task = self.create_task()
            task.revision = revision
            with self.assertRaises(ValueError):
                self.ledger.capture_worker(task.id)

    def test_duplicate_nested_json_preserves_numeric_boolean_types_and_key_order(self):
        operation = self.register()
        original = {"status": "accepted", "details": {"sequence": [1], "nested": {"count": 1}}}
        self.assertTrue(self.append(operation, fact=original))
        reordered = {"details": {"nested": {"count": 1}, "sequence": [1]}, "status": "accepted"}
        self.assertFalse(self.append(operation, fact=reordered))
        for sequence, count in (([True], 1), ([1.0], 1), ([1], True), ([1], 1.0)):
            with self.subTest(sequence=sequence, count_type=type(count).__name__), self.assertRaises(ValueError):
                self.append(operation, fact={"status": "accepted", "details": {"sequence": sequence, "nested": {"count": count}}})
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [original])

    def test_plan_commit_only_appends_a_deepcopy(self):
        operation = self.register()
        self.append(operation)
        before = self.task.public()
        plan = {"steps": [{"tool": "navi_get_status", "arguments": {}}]}
        expected = deepcopy(plan)
        self.assertTrue(self.ledger.commit_plan(self.owner, plan))
        self.assertTrue(self.ledger.commit_plan(self.owner, {"steps": []}))
        plan["steps"][0]["arguments"]["mutated"] = True
        self.assertEqual(self.task.plans, [expected, {"steps": []}])
        self.assertEqual(self.task.completed, before["completed_steps"])
        self.assertEqual(self.task.status, before["status"])
        self.assertEqual(self.task.revision, before["revision"])
        self.assertEqual(len(self.ledger.operation_facts(operation.operation_id)), 1)

    def test_revise_rotates_owner_and_preserves_slots_and_operation_facts(self):
        operation = self.register()
        self.append(operation)
        self.assertTrue(self.ledger.claim_slot("business", self.owner))
        self.ledger.revise(self.task.id, "synthetic revised destination", {})
        current = self.ledger.capture_worker(self.task.id)
        self.assertEqual(current.revision, self.owner.revision + 1)
        self.assertNotEqual(current.token, self.owner.token)
        self.assertFalse(self.ledger.commit_plan(self.owner, {"stale": True}))
        self.assertFalse(self.ledger.claim_slot("other", self.owner))
        self.assertEqual(self.ledger.slot_owner("business"), self.owner)
        self.assertEqual(len(self.ledger.operation_facts(operation.operation_id)), 1)
        before = self.task.public()
        self.assertTrue(self.append(operation, event_id="late", fact={
            "status": "accepted", "task_id": self.owner.task_id,
            "revision": self.owner.revision, "worker_token": self.owner.token,
        }))
        self.assertEqual(self.task.public(), before)
        self.assertTrue(self.ledger.commit_plan(current, {"revised": True}))

    def test_terminal_tasks_reject_plans_but_accept_late_registered_facts(self):
        for terminal in ("cancelled", "done", "failed"):
            with self.subTest(terminal=terminal):
                task = self.create_task()
                owner = self.ledger.capture_worker(task.id)
                operation = self.register(owner=owner, call_id="call-" + terminal)
                self.assertTrue(self.append(operation, event_id="before-" + terminal,
                                            fact={"status": "accepted"}))
                self.assertTrue(self.ledger.claim_slot(terminal, owner))
                if terminal == "cancelled":
                    self.ledger.cancel(task.id)
                    self.assertEqual(task.revision, owner.revision + 1)
                else:
                    with self.ledger.lock:
                        task.status = terminal
                before = task.public()
                self.assertFalse(self.ledger.commit_plan(owner, {"late": True}))
                self.assertFalse(self.ledger.claim_slot("late-" + terminal, owner))
                with self.assertRaises(ValueError):
                    self.register(owner=owner, call_id="terminal-new-" + terminal)
                self.assertEqual(self.ledger.slot_owner(terminal), owner)
                self.assertTrue(self.append(operation, event_id="late-" + terminal))
                self.assertEqual(task.public(), before)
                self.assertEqual(len(self.ledger.operation_facts(operation.operation_id)), 2)

    def test_operation_metadata_and_schema_snapshot_are_immutable(self):
        arguments = {"end_name": "synthetic destination", "via_points": [{"name": "A"}]}
        resources = ["navigation"]
        operation = self.register(arguments=arguments, resources=resources)
        expected_fields = {
            "operation_id", "call_id", "task_id", "revision", "worker_token", "tool", "actor",
            "session", "device", "contract_version", "parameters_sha256", "resources",
            "cleanup_token", "event_source",
        }
        self.assertEqual({field.name for field in fields(operation)}, expected_fields)
        self.assertEqual(operation.tool, "navi_start_route")
        self.assertEqual((operation.task_id, operation.revision, operation.worker_token),
                         (self.owner.task_id, self.owner.revision, self.owner.token))
        for key, value in self.identity.items():
            self.assertEqual(getattr(operation, key), value)
        self.assertTrue(operation.operation_id)
        self.assertTrue(operation.cleanup_token)
        self.assertRegex(operation.parameters_sha256, r"^[0-9a-f]{64}$")
        digest = operation.parameters_sha256
        arguments["via_points"][0]["name"] = "mutated"
        resources.append("different-resource")
        self.assertEqual(operation.parameters_sha256, digest)
        self.assertEqual(operation.resources, ("navigation",))
        with self.assertRaises(FrozenInstanceError):
            operation.device = "different-device"
        second = self.register(call_id="same-arguments", arguments={
            "via_points": [{"name": "A"}], "end_name": "synthetic destination",
        })
        self.assertEqual(second.parameters_sha256, digest)
        self.assertNotEqual(second.cleanup_token, operation.cleanup_token)
        schema = self.snapshot.schemas(frozenset({"navi_start_route"}))
        schema[0]["parameters"]["required"].append("mutated")
        self.assertEqual(self.snapshot.tools["navi_start_route"].parameters["required"],
                         ["end_name"])
        self.assertIs(self.task.snapshot, self.snapshot)
        with self.assertRaises(TypeError):
            self.snapshot.tools["fake"] = self.snapshot.tools["navi_start_route"]

    def test_operation_registration_requires_current_registered_owner_and_unique_call(self):
        operation = self.register()
        with self.assertRaises(ValueError):
            self.register()
        for tool, arguments in (("unregistered_tool", {}), ("navi_cancel_route", {}),
                                ("navi_start_route", {"end_name": 7})):
            with self.subTest(tool=tool, arguments=arguments), self.assertRaises(ValueError):
                self.ledger.register_operation(
                    self.owner, "invalid-" + tool, tool, arguments, **self.identity,
                )
        self.ledger.rotate_worker(self.task.id)
        for owner in (self.owner, replace(self.owner, task_id="unregistered-task")):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                self.register(owner=owner, call_id="stale-" + owner.task_id)
        current = self.ledger.capture_worker(self.task.id)
        with self.assertRaises(ValueError):
            self.register(owner=current, device="different-device")
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])

    def test_fact_identity_must_match_every_registered_identity_field(self):
        operation = self.register()
        for field in ("call_id", "actor", "session", "device", "contract_version", "event_source"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                identity = {**self.identity, "call_id": operation.call_id}
                identity[field] = "wrong-" + field
                self.ledger.append_operation_fact(
                    operation.operation_id, event_id="bad-" + field,
                    fact={"status": "effective"}, **identity,
                )
        with self.assertRaises(ValueError):
            self.ledger.append_operation_fact(
                "unregistered-operation", call_id=operation.call_id, event_id="missing",
                fact={"status": "effective"}, **self.identity,
            )
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])
        self.assertTrue(self.append(operation))

    def test_fact_embedded_task_owner_identity_cannot_be_rebound(self):
        operation = self.register()
        for field, value in (("task_id", "other-task"), ("revision", self.owner.revision + 1),
                             ("worker_token", "other-token")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.append(operation, event_id="bad-" + field,
                            fact={"status": "effective", field: value})
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])

    def test_duplicate_event_is_idempotent_but_changed_event_is_rejected(self):
        operation = self.register()
        fact = {"status": "accepted", "details": {"sequence": [1]}}
        self.assertTrue(self.append(operation, fact=fact))
        self.assertFalse(self.append(operation, fact=deepcopy(fact)))
        with self.assertRaises(ValueError):
            self.append(operation, fact={"status": "effective"})
        with self.assertRaises(ValueError):
            self.append(operation, device="wrong-device", fact=fact)
        self.assertEqual(len(self.ledger.operation_facts(operation.operation_id)), 1)

    def test_fact_input_and_return_values_are_detached_deepcopies(self):
        operation = self.register()
        fact = {"status": "effective", "details": {"observations": ["original"]}}
        self.assertTrue(self.append(operation, fact=fact))
        stored = self.ledger.operation_facts(operation.operation_id)
        expected = deepcopy(stored)
        fact["details"]["observations"].append("caller-mutation")
        stored[0]["details"]["observations"].append("reader-mutation")
        stored.append({"status": "failed"})
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), expected)

    def test_allowed_fact_statuses_are_storage_only_without_task_aggregation(self):
        operation = self.register()
        before = self.task.public()
        statuses = ("effective", "accepted", "failed", "rejected", "unknown")
        for index, status in enumerate(statuses):
            self.assertTrue(self.append(operation, event_id=f"event-{index}",
                                        fact={"status": status}))
        self.assertEqual([fact["status"] for fact in self.ledger.operation_facts(operation.operation_id)],
                         list(statuses))
        self.assertEqual(self.task.public(), before)

    def test_invalid_fact_statuses_do_not_append_or_mutate_task(self):
        operation = self.register()
        before = self.task.public()
        for index, fact in enumerate(({}, {"status": "done"}, {"status": "not_sent"},
                                      {"status": None})):
            with self.subTest(fact=fact), self.assertRaises(ValueError):
                self.append(operation, event_id=f"invalid-{index}", fact=fact)
        self.assertEqual(self.ledger.operation_facts(operation.operation_id), [])
        self.assertEqual(self.task.public(), before)

    def test_old_finally_can_release_own_slot_after_revision_or_terminal_change(self):
        for transition in ("revise", "cancel", "done"):
            with self.subTest(transition=transition):
                task = self.create_task()
                owner = self.ledger.capture_worker(task.id)
                self.assertTrue(self.ledger.claim_slot(transition, owner))
                if transition == "revise":
                    self.ledger.revise(task.id, "synthetic revision", {})
                elif transition == "cancel":
                    self.ledger.cancel(task.id)
                else:
                    with self.ledger.lock:
                        task.status = "done"
                before = task.public()
                self.assertFalse(self.ledger.release_slot(transition, owner, io_safe=False))
                self.assertEqual(self.ledger.slot_owner(transition), owner)
                self.assertTrue(self.ledger.release_slot(transition, owner, io_safe=True))
                self.assertIsNone(self.ledger.slot_owner(transition))
                self.assertEqual(task.public(), before)
                self.assertFalse(self.ledger.release_slot(transition, owner, io_safe=True))

    def test_slot_claim_is_idempotent_and_release_requires_exact_owner_and_safe_io(self):
        self.assertIsNone(self.ledger.slot_owner("business"))
        self.assertTrue(self.ledger.claim_slot("business", self.owner))
        self.assertTrue(self.ledger.claim_slot("business", self.owner))
        for impostor in (replace(self.owner, token="wrong"),
                         replace(self.owner, revision=self.owner.revision + 1),
                         replace(self.owner, task_id="wrong")):
            self.assertFalse(self.ledger.release_slot("business", impostor, io_safe=True))
        for unsafe in (False, None, 1):
            with self.subTest(io_safe=unsafe):
                self.assertFalse(self.ledger.release_slot("business", self.owner, io_safe=unsafe))
                self.assertEqual(self.ledger.slot_owner("business"), self.owner)
        self.assertTrue(self.ledger.release_slot("business", self.owner, io_safe=True))

    def test_old_cleanup_racing_current_owner_cannot_clear_reused_slot(self):
        for new_task in (False, True):
            with self.subTest(new_task=new_task):
                old = self.ledger.capture_worker(self.task.id)
                self.assertTrue(self.ledger.claim_slot("business", old))
                self.ledger.rotate_worker(self.task.id)
                self.assertTrue(self.ledger.release_slot("business", old, io_safe=True))
                task = self.create_task() if new_task else self.task
                current = self.ledger.capture_worker(task.id)
                self.assertTrue(self.ledger.claim_slot("business", current))
                cleanup, claim = self.race(
                    lambda: self.ledger.release_slot("business", old, io_safe=True),
                    lambda: self.ledger.claim_slot("business", current),
                )
                self.assertFalse(cleanup)
                self.assertTrue(claim)
                self.assertEqual(self.ledger.slot_owner("business"), current)
                self.assertTrue(self.ledger.release_slot("business", current, io_safe=True))

    def test_barrier_plan_cas_only_current_token_or_revision_wins(self):
        for change in ("token", "revision"):
            with self.subTest(change=change):
                stale = self.ledger.capture_worker(self.task.id)
                if change == "token":
                    current = self.ledger.rotate_worker(self.task.id)
                else:
                    self.ledger.revise(self.task.id, "synthetic revision", {})
                    current = self.ledger.capture_worker(self.task.id)
                before = len(self.task.plans)
                stale_result, current_result = self.race(
                    lambda: self.ledger.commit_plan(stale, {"worker": "stale"}),
                    lambda: self.ledger.commit_plan(current, {"worker": "current"}),
                )
                self.assertEqual((stale_result, current_result), (False, True))
                self.assertEqual(self.task.plans[before:], [{"worker": "current"}])

    def test_barrier_slot_cas_has_exactly_one_owner(self):
        other = self.create_task()
        other_owner = self.ledger.capture_worker(other.id)
        first, second = self.race(
            lambda: self.ledger.claim_slot("business", self.owner),
            lambda: self.ledger.claim_slot("business", other_owner),
        )
        self.assertEqual(sum((first, second)), 1)
        winner, loser = (self.owner, other_owner) if first else (other_owner, self.owner)
        self.assertEqual(self.ledger.slot_owner("business"), winner)
        self.assertFalse(self.ledger.release_slot("business", loser, io_safe=True))
        self.assertTrue(self.ledger.release_slot("business", winner, io_safe=True))

    def test_barrier_duplicate_call_registration_has_one_winner(self):
        def register_once():
            try:
                return self.register()
            except ValueError as error:
                return error

        outcomes = self.race(register_once, register_once)
        errors = [outcome for outcome in outcomes if isinstance(outcome, ValueError)]
        operations = [outcome for outcome in outcomes if not isinstance(outcome, ValueError)]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].call_id, "call-1")
        self.assertTrue(self.append(operations[0]))
        self.assertEqual(len(self.ledger.operation_facts(operations[0].operation_id)), 1)


if __name__ == "__main__":
    unittest.main()
