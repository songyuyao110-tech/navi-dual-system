from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from threading import Condition, RLock
from uuid import uuid4

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
