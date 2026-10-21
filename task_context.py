from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from threading import Condition, RLock
from uuid import uuid4

from jsonschema import Draft202012Validator
from registry import RegistrySnapshot


@dataclass
class Task:
    id: str
    original: str
    context: dict
    snapshot: RegistrySnapshot
    disclosed: frozenset[str]
    route: str
    revision: int = 1
    status: str = "new"
    turns: list[str] = field(default_factory=list)
    completed: list[dict] = field(default_factory=list)
    question: str | None = None
    error: str | None = None
    plans: list[dict] = field(default_factory=list)
    inflight: bool = False
    asked_questions: list[str] = field(default_factory=list)
    slow_rounds: list[dict] = field(default_factory=list)
    processing: bool = False

    def brief(self) -> dict:
        return {
            "original_utterance": self.original,
            "latest_utterance": self.turns[-1],
            "turns": list(self.turns),
            "context": deepcopy(self.context),
            "completed_steps": deepcopy(self.completed),
            "question": self.question,
            "asked_questions": list(self.asked_questions),
        }

    def public(self) -> dict:
        return {
            "id": self.id, "original": self.original, "revision": self.revision,
            "status": self.status, "route": self.route,
            "registry_version": self.snapshot.version,
            "disclosed_tools": sorted(self.disclosed),
            "turns": list(self.turns), "question": self.question,
            "completed_steps": deepcopy(self.completed), "error": self.error,
            "plans": deepcopy(self.plans), "inflight": self.inflight,
            "processing": self.processing,
            "slow_rounds": deepcopy(self.slow_rounds),
        }


class TaskLedger:
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.lock = RLock()
        self.condition = Condition(self.lock)
        self.conversation: deque[dict] = deque(maxlen=12)

    def record(self, role: str, content: str) -> None:
        with self.lock:
            self.conversation.append({"role": role, "content": content})

    def brief_context(self, context: dict) -> dict:
        with self.lock:
            return {**deepcopy(context), "conversation": deepcopy(list(self.conversation))}

    def create(self, utterance: str, context: dict, snapshot: RegistrySnapshot, decision: dict) -> Task:
        with self.lock:
            task = Task(
                id="task-" + str(uuid4()), original=utterance, context=deepcopy(context),
                snapshot=snapshot, disclosed=frozenset(decision["disclosed_tools"]),
                route=decision["route"], turns=[utterance],
            )
            self.tasks[task.id] = task
            return task

    def accept_slow(
        self, utterance: str, context: dict, snapshot: RegistrySnapshot
    ) -> tuple[Task, str]:
        """Atomically reserve a new task or refill the sole waiting task."""
        with self.lock:
            active = [task for task in self.tasks.values()
                      if task.processing or task.status not in {"done", "failed", "cancelled"}]
            if len(active) > 1:
                raise ValueError("存在多个未完成任务，当前单任务模式无法自动选择回填目标。")
            if active:
                task = active[0]
                if task.processing or task.inflight or task.status != "waiting_input":
                    raise ValueError("当前任务正在处理，请等待完成或追问后再输入。")
                task = self.revise(task.id, utterance, context)
                task.route = "slow"
                task.status = "needs_slow"
                submission_kind = "refill"
            else:
                task = self.create(utterance, context, snapshot,
                                   {"route": "slow", "disclosed_tools": []})
                task.status = "disclosing"
                submission_kind = "new"
            task.processing = True
            return task, submission_kind

    def pending(self) -> dict[str, str]:
        with self.lock:
            return {
                task.id: f"{task.original}; 状态={task.status}; 缺项={task.question}"
                for task in self.tasks.values()
                if task.status not in {"done", "failed", "cancelled"}
            }

    def revise(self, task_id: str, utterance: str, context: dict) -> Task:
        with self.lock:
            task = self._active(task_id)
            task.revision += 1
            task.turns.append(utterance)
            task.context.update(deepcopy(context))
            task.question = None
            task.error = None
            task.status = "needs_slow" if task.completed or task.inflight else "new"
            if task.completed or task.inflight:
                task.route = "slow"
            return task

    def cancel(self, task_id: str) -> Task:
        with self.lock:
            task = self._active(task_id)
            task.revision += 1
            task.status = "cancelled"
            return task

    def _active(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise ValueError("Target task does not exist")
        task = self.tasks[task_id]
        if task.status in {"done", "failed", "cancelled"}:
            raise ValueError("Target task is already terminal")
        return task

    def current(self, task: Task, revision: int) -> bool:
        return task.revision == revision and task.status not in {"cancelled", "done", "failed"}

    def public(self) -> list[dict]:
        with self.lock:
            return [task.public() for task in self.tasks.values()]


@dataclass(frozen=True)
class WorkerOwner:
    task_id: str
    revision: int
    token: str


@dataclass(frozen=True)
class Operation:
    operation_id: str
    call_id: str
    task_id: str
    revision: int
    worker_token: str
    tool: str
    actor: str
    session: str
    device: str
    contract_version: str
    parameters_sha256: str
    resources: tuple[str, ...]
    cleanup_token: str
    event_source: str


class OperationLedger(TaskLedger):
    """Opt-in 5.1 in-memory ownership. Slots are local, not device reservations."""
    def __init__(self):
        super().__init__()
        self._workers: dict[str, str] = {}
        self._slots: dict[str, WorkerOwner] = {}
        self._operations: dict[str, Operation] = {}
        self._calls: dict[str, str] = {}
        self._facts: dict[str, list[dict]] = {}
        self._events: dict[tuple[str, str], str] = {}

    def capture_worker(self, task_id: str) -> WorkerOwner:
        with self.lock:
            task = self._active(task_id)
            if type(task.revision) is not int or task.revision < 1:
                raise ValueError("Task revision must be a positive integer")
            token = self._workers.setdefault(task_id, "worker-" + str(uuid4()))
            return WorkerOwner(task.id, task.revision, token)

    def rotate_worker(self, task_id: str) -> WorkerOwner:
        with self.lock:
            self._active(task_id)
            self._workers[task_id] = "worker-" + str(uuid4())
            return self.capture_worker(task_id)

    def revise(self, task_id: str, utterance: str, context: dict) -> Task:
        with self.lock:
            task = super().revise(task_id, utterance, context)
            self._workers[task_id] = "worker-" + str(uuid4())
            return task

    def cancel(self, task_id: str) -> Task:
        with self.lock:
            task = super().cancel(task_id)
            self._workers[task_id] = "worker-" + str(uuid4())
            return task

    @staticmethod
    def _valid_owner(owner: WorkerOwner) -> bool:
        return (type(owner) is WorkerOwner and type(owner.task_id) is str and bool(owner.task_id)
                and type(owner.revision) is int and owner.revision > 0
                and type(owner.token) is str and bool(owner.token.strip()))

    def _current_owner(self, owner: WorkerOwner) -> bool:
        if not self._valid_owner(owner):
            return False
        task = self.tasks.get(owner.task_id)
        registered = self._workers.get(owner.task_id)
        return bool(task is not None and type(task.revision) is int and self.current(task, owner.revision)
                    and type(registered) is str and registered.strip() and registered == owner.token)

    def commit_plan(self, owner: WorkerOwner, plan: dict) -> bool:
        if not isinstance(plan, dict):
            raise ValueError("Plan must be an object")
        captured = deepcopy(plan)
        with self.lock:
            if not self._current_owner(owner):
                return False
            self.tasks[owner.task_id].plans.append(captured)
            return True

    def claim_slot(self, slot: str, owner: WorkerOwner) -> bool:
        if not isinstance(slot, str) or not slot:
            raise ValueError("Local slot name required")
        with self.lock:
            if not self._current_owner(owner) or (slot in self._slots and self._slots[slot] != owner):
                return False
            self._slots[slot] = owner
            return True

    def release_slot(self, slot: str, owner: WorkerOwner, *, io_safe: bool) -> bool:
        with self.condition:
            if (io_safe is not True or not self._valid_owner(owner)
                    or slot not in self._slots or self._slots[slot] != owner):
                return False
            del self._slots[slot]  # Cleanup authority is captured slot owner, not current task CAS.
            self.condition.notify_all()
            return True

    def slot_owner(self, slot: str) -> WorkerOwner | None:
        with self.lock:
            return self._slots.get(slot)

    def register_operation(self, owner: WorkerOwner, call_id: str, tool: str, arguments: dict, *,
                           actor: str, session: str, device: str, contract_version: str,
                           event_source: str, resources=()) -> Operation:
        labels = (call_id, tool, actor, session, device, contract_version, event_source)
        if any(not isinstance(value, str) or not value.strip() for value in labels) or not isinstance(arguments, dict):
            raise ValueError("Operation identity and arguments required")
        resources = tuple(resources)
        if any(not isinstance(value, str) or not value for value in resources) or len(set(resources)) != len(resources):
            raise ValueError("Operation resource identity invalid")
        captured_arguments = deepcopy(arguments)
        digest = hashlib.sha256(json.dumps(captured_arguments, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        with self.lock:
            if not self._current_owner(owner) or call_id in self._calls:
                raise ValueError("Stale worker or duplicate registered call")
            task = self.tasks[owner.task_id]
            if tool not in task.disclosed or tool not in task.snapshot.tools:
                raise ValueError("Operation tool is not disclosed/registered")
            if not Draft202012Validator(task.snapshot.tools[tool].parameters).is_valid(captured_arguments):
                raise ValueError("Operation arguments violate the captured schema")
            operation = Operation("op-" + str(uuid4()), call_id, owner.task_id, owner.revision, owner.token,
                                  tool, actor, session, device, contract_version, digest, resources,
                                  "cleanup-" + str(uuid4()), event_source)
            self._operations[operation.operation_id] = operation
            self._calls[call_id] = operation.operation_id
            self._facts[operation.operation_id] = []
            return operation

    def append_operation_fact(self, operation_id: str, *, call_id: str, event_id: str, actor: str,
                              session: str, device: str, contract_version: str, event_source: str,
                              fact: dict) -> bool:
        if (not isinstance(event_id, str) or not event_id or not isinstance(fact, dict)
                or fact.get("status") not in {"effective", "accepted", "failed", "rejected", "unknown"}):
            raise ValueError("Invalid operation event")
        event_json = json.dumps(fact, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        captured = json.loads(event_json)
        with self.lock:
            operation = self._operations.get(operation_id)
            if operation is None or any(value != getattr(operation, name) for name, value in
                    (("call_id", call_id), ("actor", actor), ("session", session), ("device", device),
                     ("contract_version", contract_version), ("event_source", event_source))):
                raise ValueError("Operation event identity mismatch")
            if any(name in captured and (type(captured[name]) is not type(getattr(operation, name))
                                         or captured[name] != getattr(operation, name))
                   for name in ("operation_id", "call_id", "task_id", "revision", "worker_token", "tool",
                                "actor", "session", "device", "contract_version", "event_source")):
                raise ValueError("Operation fact does not match captured owner")
            key = (operation_id, event_id)
            if key in self._events:
                if self._events[key] != event_json:
                    raise ValueError("Conflicting duplicate event")
                return False
            self._events[key] = event_json
            self._facts[operation_id].append(captured)
            return True  # Late facts retained; never mutate task plan/status/completed or new CAS.

    def operation_facts(self, operation_id: str) -> list[dict]:
        with self.lock:
            if operation_id not in self._operations:
                raise ValueError("Unknown operation")
            return deepcopy(self._facts[operation_id])


@dataclass(frozen=True)
class NotSentReceipt:
    operation_id: str
    cleanup_token: str
    fixture_id: str
    receipt_id: str
    kind: str


class SendEvidenceLedger(OperationLedger):
    """5.2 synthetic resource/send evidence; no real bridge or deadline policy."""
    def __init__(self, contract: dict):
        super().__init__()
        from eval.validate_reopt import FROZEN_CONTRACT_VERSION
        captured = json.loads(json.dumps(contract, ensure_ascii=False, allow_nan=False))
        digest = hashlib.sha256(json.dumps({k: v for k, v in captured.items() if k != "contract_version"},
                                          sort_keys=True, ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")).encode()).hexdigest()
        if captured.get("contract_version") != digest or digest != FROZEN_CONTRACT_VERSION:
            raise ValueError("Resource/send contract differs from frozen composite")
        self._contract = captured
        self._tools = {tool["name"]: tool for tool in captured["tools"]}
        self._resource_owners: dict[tuple[str, str], str] = {}
        self._states: dict[str, str] = {}
        self._effects: dict[str, dict] = {}
        self._resolved: set[str] = set()
        self._fixtures: dict[str, SyntheticSendFixture] = {}
        self._proofs: dict[str, NotSentReceipt] = {}
        self._fixture_world: dict[str, str] = {}

    def prepare_operation(self, owner: WorkerOwner, call_id: str, tool: str, arguments: dict, **identity) -> Operation:
        captured = deepcopy(arguments)
        if not isinstance(captured, dict):
            raise ValueError("Prepared operation arguments must be an object")
        with self.lock:
            if not self._current_owner(owner) or tool not in self._tools:
                raise ValueError("Invalid prepared operation owner/tool")
            task, definition = self.tasks[owner.task_id], self._tools[tool]
            if (identity.get("contract_version") != self._contract["contract_version"]
                    or task.snapshot.version != self._contract["source_sha256"]["tools.yaml"][:16]
                    or tool not in task.snapshot.tools or task.snapshot.tools[tool].parameters != definition["parameters"]):
                raise ValueError("Prepared operation combines different schema/semantics")
            effect = definition["effect"]
            if "parameter" in effect:
                value = captured.get(effect["parameter"])
                mode = "true" if value is True else "false" if value is False else "unspecified"
                effect = effect["modes"][mode]
            resources = () if effect["kind"] == "read_only" else tuple(effect["resources"])
            if "resources" in identity:
                raise ValueError("Resources must come from the captured effect contract")
            device = identity.get("device")
            if any((device, resource) in self._resource_owners for resource in resources):
                raise ValueError("Conflicting device resource remains unresolved")
            operation = super().register_operation(owner, call_id, tool, captured, resources=resources, **identity)
            self._states[operation.operation_id] = "prepared"
            self._effects[operation.operation_id] = {"kind": effect["kind"], "resources": list(resources)}
            for resource in resources:
                self._resource_owners[(operation.device, resource)] = operation.operation_id
            return operation

    def register_operation(self, owner, call_id, tool, arguments, **identity):
        return self.prepare_operation(owner, call_id, tool, arguments, **identity)

    def operation_effect(self, operation_id: str) -> dict:
        with self.lock:
            return deepcopy(self._effects[operation_id])

    def operation_state(self, operation_id: str) -> str:
        with self.lock:
            return self._states[operation_id]

    def resource_owner(self, device: str, resource: str) -> str | None:
        with self.lock:
            return self._resource_owners.get((device, resource))

    def mark_handoff(self, operation_id: str, cleanup_token: str) -> bool:
        with self.lock:
            operation = self._operations[operation_id]
            fixture = self._fixtures.get(operation_id)
            if (cleanup_token != operation.cleanup_token or (fixture is not None and fixture._closed)
                    or self._states[operation_id] not in {"prepared", "unknown"}):
                raise ValueError("Invalid or terminal synthetic handoff")
            self._states[operation_id] = "unknown"  # Before queue, partial write or sent-accounting.
            return True

    def append_operation_fact(self, operation_id: str, **event) -> bool:
        with self.lock:
            added = super().append_operation_fact(operation_id, **event)
            if not added:
                return False
            fact = self._facts[operation_id][-1]
            previous, status = self._states[operation_id], fact["status"]
            if (operation_id in self._resolved and previous in {"effective", "failed", "rejected", "not_sent"}
                    and (status in {"accepted", "unknown"} or fact.get("verified") is not True or fact.get("final") is not True)):
                return True  # Retain trusted terminal authority; lower-stage facts remain recorded.
            if ((previous in {"failed", "rejected", "not_sent"} and status == "effective")
                    or (previous == "effective" and status in {"failed", "rejected", "unknown"})):
                self._states[operation_id] = "conflict"
                self._resolved.discard(operation_id)
            elif previous == "conflict" or (status == "accepted" and (
                    previous == "effective" or (previous in {"failed", "rejected"} and operation_id in self._resolved))):
                pass  # Retain trusted terminal state; late accepted is a fact, not send authority.
            elif status == "effective" and fact.get("verified") is not True:
                self._states[operation_id] = "unknown"
                self._resolved.discard(operation_id)
            else:
                self._states[operation_id] = status
                self._resolved.discard(operation_id)
                if status in {"effective", "failed", "rejected"} and fact.get("verified") is True and fact.get("final") is True:
                    self._resolved.add(operation_id)
            return True  # Operation facts cannot advance or revive its task.

    def release_resources(self, operation_id: str, cleanup_token: str) -> bool:
        with self.lock:
            operation = self._operations[operation_id]
            if cleanup_token != operation.cleanup_token:
                return False
            state = self._states[operation_id]
            if state not in {"not_sent", "effective", "failed", "rejected"} or operation_id not in self._resolved:
                return False
            for resource in operation.resources:
                key = (operation.device, resource)
                if self._resource_owners.get(key) == operation_id:
                    del self._resource_owners[key]
            return True

    def _issue_not_sent(self, fixture: SyntheticSendFixture, kind: str) -> NotSentReceipt:
        with self.lock:
            operation = fixture.operation
            if (self._fixtures.get(operation.operation_id) is not fixture or not fixture._closed
                    or fixture._queued or fixture._written or fixture._retry_pending
                    or self._states[operation.operation_id] not in {"prepared", "unknown"}):
                raise ValueError("Fixture has no positive unreachability evidence")
            receipt = NotSentReceipt(operation.operation_id, operation.cleanup_token, fixture.fixture_id,
                                     "receipt-" + str(uuid4()), kind)
            self._proofs[receipt.receipt_id] = receipt
            return receipt

    def accept_not_sent(self, receipt: NotSentReceipt) -> bool:
        with self.lock:
            if type(receipt) is not NotSentReceipt or self._proofs.get(receipt.receipt_id) is not receipt:
                raise ValueError("Unregistered or forged unreachability receipt")
            fixture = self._fixtures[receipt.operation_id]
            operation = self._operations[receipt.operation_id]
            if (receipt.cleanup_token != operation.cleanup_token or receipt.fixture_id != fixture.fixture_id
                    or not fixture._closed or fixture._queued or fixture._written or fixture._retry_pending):
                raise ValueError("Receipt does not prove all captured send paths closed")
            if self._states[receipt.operation_id] == "not_sent":
                return False
            if self._states[receipt.operation_id] not in {"prepared", "unknown"}:
                raise ValueError("Receipt contradicts an observed operation effect")
            self._states[receipt.operation_id] = "not_sent"
            self._resolved.add(receipt.operation_id)
            return True


class SyntheticSendFixture:
    """One synthetic channel with explicit queue/bytes/retry closure, not a real adapter."""
    def __init__(self, ledger: SendEvidenceLedger, operation: Operation):
        with ledger.lock:
            if ledger._operations.get(operation.operation_id) is not operation or operation.operation_id in ledger._fixtures:
                raise ValueError("Synthetic fixture requires its unique registered operation")
            self.ledger, self.operation = ledger, operation
            self.fixture_id = "fixture-" + str(uuid4())
            self._queued, self._written, self._retry_pending, self._closed = False, 0, False, False
            ledger._fixtures[operation.operation_id] = self

    def _handoff(self):
        if self._closed:
            raise RuntimeError("Synthetic channel was positively revoked")
        self.ledger.mark_handoff(self.operation.operation_id, self.operation.cleanup_token)

    def enqueue(self):
        with self.ledger.lock:
            self._handoff()
            self._queued = True

    def partial_write(self):
        with self.ledger.lock:
            self._handoff()
            self._written += 1

    def write_then_accounting_error(self):
        self.partial_write()
        raise RuntimeError("Synthetic bytes written before accounting failed")

    def hidden_retry(self):
        with self.ledger.lock:
            self._handoff()
            self._written += 1
            self._retry_pending = True

    def revoke_before_handoff(self) -> NotSentReceipt | None:
        with self.ledger.lock:
            if self.ledger.operation_state(self.operation.operation_id) != "prepared":
                return None
            self._closed = True
            return self.ledger._issue_not_sent(self, "all_local_paths_revoked_before_handoff")

    def atomic_reject_unforwardable(self) -> NotSentReceipt | None:
        with self.ledger.lock:
            if self._written or self._retry_pending:
                return None
            self._queued, self._closed = False, True
            return self.ledger._issue_not_sent(self, "atomic_channel_reject_no_forwarding_path")

    def deliver_verified_effect(self):
        with self.ledger.lock:
            if self._closed:
                raise RuntimeError("Revoked synthetic channel cannot apply")
            state = self.ledger.operation_state(self.operation.operation_id)
            if state == "prepared":
                self._handoff()
            elif state not in {"unknown", "accepted"}:
                raise RuntimeError("Synthetic terminal operation cannot apply again")
            self._written += 1
            self._queued, self._retry_pending, self._closed = False, False, True
            self.ledger._fixture_world[self.operation.device] = self.operation.operation_id
            op = self.operation
            self.ledger.append_operation_fact(op.operation_id, call_id=op.call_id, event_id="effect-" + self.fixture_id,
                actor=op.actor, session=op.session, device=op.device, contract_version=op.contract_version,
                event_source=op.event_source, fact={"status": "effective", "verified": True, "final": True})


@dataclass(frozen=True)
class DispatchWindow:
    task_deadline: float
    send_not_after: float


class BoundedDispatchLedger(SendEvidenceLedger):
    """5.3 absolute monotonic windows; no timeout resets or real transport."""
    def __init__(self, contract: dict, *, clock=time.monotonic):
        super().__init__(contract)
        self._clock = clock
        self._task_deadlines: dict[str, float] = {}
        self._windows: dict[str, DispatchWindow] = {}

    @staticmethod
    def _deadline(value) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Deadline must be a finite monotonic timestamp")
        return float(value)

    def capture_deadline(self, owner: WorkerOwner, task_deadline: float) -> float:
        deadline = self._deadline(task_deadline)
        with self.lock:
            if not self._current_owner(owner):
                raise ValueError("Stale deadline owner")
            previous = self._task_deadlines.get(owner.task_id)
            if previous is not None and previous != deadline:
                raise ValueError("Task deadline cannot be extended or replaced")
            if self._clock() >= deadline:
                raise TimeoutError("Outer task deadline expired")
            self._task_deadlines[owner.task_id] = deadline
            return deadline

    def run_phase(self, owner: WorkerOwner, callback):
        with self.lock:
            self._check_task_window(owner)
        result = callback()  # Synthetic external work outside the state lock.
        with self.lock:
            self._check_task_window(owner)  # Late results cannot resume the plan.
        return result

    def _check_task_window(self, owner: WorkerOwner):
        if not self._current_owner(owner) or owner.task_id not in self._task_deadlines:
            raise ValueError("Unregistered or stale bounded worker")
        if self._clock() >= self._task_deadlines[owner.task_id]:
            raise TimeoutError("Outer task deadline expired")

    def prepare_operation(self, owner, call_id, tool, arguments, *, task_deadline: float,
                          send_not_after: float, **identity) -> Operation:
        deadline, sending = self._deadline(task_deadline), self._deadline(send_not_after)
        if sending > deadline:
            raise ValueError("Send window cannot outlive task deadline")
        with self.lock:
            self.capture_deadline(owner, deadline)
            if self._clock() >= sending:
                raise TimeoutError("Send permission already expired")
            operation = super().prepare_operation(owner, call_id, tool, arguments, **identity)
            self._windows[operation.operation_id] = DispatchWindow(deadline, sending)
            return operation

    def dispatch_window(self, operation_id: str) -> DispatchWindow:
        with self.lock:
            return self._windows[operation_id]

    def check_send_window(self, operation_id: str):
        with self.lock:
            op, window = self._operations[operation_id], self._windows[operation_id]
            owner = WorkerOwner(op.task_id, op.revision, op.worker_token)
            self._check_task_window(owner)
            if self._clock() >= window.send_not_after:
                raise TimeoutError("Captured send_not_after expired")

    def mark_handoff(self, operation_id, cleanup_token):
        with self.lock:
            self.check_send_window(operation_id)
            return super().mark_handoff(operation_id, cleanup_token)

    def _issue_not_sent(self, fixture, kind):
        if getattr(fixture, "_write_inflight", False) or getattr(fixture, "_write_unknown", False):
            raise ValueError("In-flight or unaccounted write is not unreachability evidence")
        return super()._issue_not_sent(fixture, kind)


class BoundedSendFixture(SyntheticSendFixture):
    """Each synthetic chunk and queued dequeue rechecks the same captured window."""
    def __init__(self, ledger: BoundedDispatchLedger, operation: Operation):
        super().__init__(ledger, operation)
        self._write_inflight, self._write_unknown = False, False

    def send_chunk(self, payload: bytes, write):
        if type(payload) is not bytes or not payload:
            raise ValueError("Nonempty synthetic byte chunk required")
        with self.ledger.lock:
            if self._write_inflight or self._write_unknown:
                raise ValueError("Synthetic write remains unresolved")
            self._handoff()  # Window check precedes every possible write.
            self._write_inflight = True
        try:
            written = write(payload)
            if type(written) is not int or not 0 <= written <= len(payload):
                raise ValueError("Invalid synthetic write result")
        except BaseException:
            with self.ledger.lock:
                self._write_inflight, self._write_unknown = False, True
            raise  # Resource already unknown from handoff; never refund/retry.
        with self.ledger.lock:
            self._write_inflight = False
            self._written += written
            self.ledger.check_send_window(self.operation.operation_id)
        return written

    def send_chunks(self, chunks, write) -> int:
        total = 0
        for chunk in chunks:
            total += self.send_chunk(chunk, write)
        return total  # No per-chunk renewal of task/send deadlines.

    def dequeue_and_send(self, payload: bytes, write) -> int:
        with self.ledger.lock:
            self.ledger.check_send_window(self.operation.operation_id)
            if not self._queued:
                raise ValueError("No captured synthetic queue entry")
        return self.send_chunk(payload, write)

    def atomic_reject_unforwardable(self):
        with self.ledger.lock:
            if self._write_inflight or self._write_unknown:
                return None
            return super().atomic_reject_unforwardable()

    def deliver_verified_effect(self):
        with self.ledger.lock:
            self.ledger.check_send_window(self.operation.operation_id)
            return super().deliver_verified_effect()
