"""Streaming turn session: ASR updates + VAD → KEV EOT judging + tool shortlist → one decision.

All state transitions happen on the caller's thread (the driver that calls tick()); the injected
KEV/LLM callables run in worker threads and their results are applied only if text_version is
still the newest one, otherwise they are dropped as stale.

Speculative decision chain (tmp/poc-speculative-toolselect-spec.md A): every ASR update starts the
*same* chain the turn would run after "the user finished" - KEV tool scoring, the finite-fast gate
and its KEV parameter filling - so the select segment no longer sits on the critical path after the
turn is confirmed. Each speculative result is bound to the text it was computed from (normalized
text + sha256 + text_version). At confirm the newest result is reused only when that text equals the
final text; otherwise it is discarded and the chain is recomputed for the final text. Nothing is
executed or written back on a text other than the final one: `_apply_llm` re-checks the key against
`turn_text()` right before execution and drops the decision instead of using a stale one.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from core_pipeline_run import ROOT, decide, select_tools
from finite_fast_path import try_finite_fast_decide
from result_writeback import COMPLETED_STEPS, EVENT as WRITEBACK_EVENT, append_writeback, writeback_result
from turn_eot import score_eot

EVIDENCE_ROOT = ROOT / "eval/evidence/turn-session"
_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’()（）\[\]【】\-—…~]+")


def normalize(text: str) -> str:
    return _PUNCT.sub("", text or "")


def text_key(text: str) -> dict:
    """Identity of the input a speculative/decision result was computed from.

    `norm` is what the reuse comparison uses (ASR may re-send the same words with different
    punctuation); the sha256 is the immutable fingerprint logged with every result so a stale
    result can be spotted after the fact without recomputing the normalization rule.
    """
    norm = normalize(text)
    return {"norm": norm, "sha256": hashlib.sha256(norm.encode()).hexdigest(), "text": text or ""}


@dataclass
class TurnConfig:
    # M3 recommended defaults: margin 0.2 + a 700 ms confirm window is the only measured setting that
    # drives premature LLM starts to 0%; see tmp/poc-streaming-final-report.md for the tradeoff table.
    eot_policy: str = "margin"          # argmax|margin
    eot_margin: float = 0.2
    spec_margin: float = 0.3
    speculative_llm: bool = True     # intentional: start the decision early, cancel it on reversal
    eot_input: str = "last_segment"     # last_segment|turn_text
    confirm_silence_ms: int = 700
    cut_wait_ms: int = 1500
    think_wait_ms: int = 3000
    max_silence_ms: int = 2000
    eot_timeout_ms: int = 900
    eot_max_inflight: int = 3        # >1 removes single-flight starvation: every prefix gets scored
    # Must stay below the ASR partial interval, otherwise each pending call is replaced before it fires
    # and prefixes stop being judged at all (measured at 150 ms/prefix).
    debounce_ms: int = 80
    reuse_tools_if_text_unchanged: bool = True
    tool_set: str = "navigation"        # navigation|multidomain
    # Finite fast path (tmp/poc-finite-fast-spec.md): for allowlisted tools whose parameters are
    # all enum/boolean, KEV fills the arguments and a confident result skips the LLM decision.
    finite_fast_path: bool = True
    finite_fast_tool_tau: float = 0.70
    finite_fast_param_tau: float = 0.70
    finite_fast_require_top1: bool = True
    # Speculative chain (spec A): run tool scoring + gate + KEV parameter filling on ASR partials,
    # in parallel with EOT judging, and reuse the result at confirm when its text is the final text.
    # False restores the pre-change behaviour (shortlist only, gate/LLM after confirm) so the same
    # replay harness can measure before vs after.
    speculative_chain: bool = True

    @classmethod
    def from_dict(cls, data):
        known = {item.name for item in fields(cls)}
        unknown = set(data or {}) - known
        if unknown:
            raise ValueError("Unknown TurnConfig fields: " + ", ".join(sorted(unknown)))
        return cls(**data)


class TurnSession:
    def __init__(self, cfg: TurnConfig, on_event, *, clock=None, eot_fn=score_eot,
                 select_tools_fn=select_tools, decide_fn=decide, finite_fast_fn=try_finite_fast_decide,
                 session_id=None, evidence_root=EVIDENCE_ROOT):
        self.cfg = cfg
        self.on_event = on_event
        self.clock = clock or (lambda: time.monotonic() * 1000.0)
        self.eot_fn = eot_fn
        self.select_tools_fn = select_tools_fn
        self.decide_fn = decide_fn
        self.finite_fast_fn = finite_fast_fn
        self.session_id = session_id or ("s-" + str(int(time.time() * 1000)))
        self.evidence_root = Path(evidence_root) if evidence_root else None
        self._pool = ThreadPoolExecutor(max_workers=4 + max(0, cfg.eot_max_inflight - 1))
        # Speculation gets its own pool: a decision job may legitimately wait for an in-flight
        # speculative chain, and that wait must never starve EOT judging or tool scoring.
        self._spec_pool = ThreadPoolExecutor(max_workers=2)
        # Session state, not turn state: completed_steps is the Agent-visible record of finished
        # turns (see result_writeback) and therefore outlives _reset_turn().
        self.context = {"home_address_configured": True, "company_address_configured": True,
                        "synthetic": True, COMPLETED_STEPS: []}
        self.events = []
        self.turns = []
        self.state = "LISTENING"
        self.turn_start_ms = None
        self.segment = ""
        self.segments = []
        self.version = 0
        self.last_change_ms = None
        self.last_speech_end_ms = None
        self.speech_start_ms = None
        self.t_last_word = None
        self.t_confirmed = None
        self.t_llm_start = None
        self.t_decision = None
        self.timeline_path = None
        self.turn_done = False
        self.turn_index = 1
        self.segment_closed = False
        self._jobs = {"spec": None, "llm": None}
        self._eot_jobs = []
        self._eot_pending = None
        self._spec_pending = None
        self._spec_ready = None          # newest finished speculation (one slot, keyed by text)
        self._spec_orphans = []          # preempted chains still running; dropped when they land
        self._llm_prepared = None
        self.counters = {"eot_calls": 0, "eot_skipped": 0, "eot_stale": 0, "tool_calls": 0,
                         "llm_calls": 0, "llm_wasted": 0,
                         "finite_fast_hits": 0, "finite_fast_misses": 0, "finite_fast_errors": 0,
                         "spec_calls": 0, "spec_wasted": 0, "spec_reused_hit": 0,
                         "spec_reused_selection": 0, "spec_reused_gate": 0,
                         "spec_stale_results": 0,
                         "spec_stale_executions_blocked": 0, "spec_recomputes": 0,
                         "spec_waited": 0}
        self.eot_scores = None
        self.decision = None
        self.routed_by = None
        self.miss_reason = None

    # ---------- public API ----------

    def on_asr(self, t_ms: int, text: str, final: bool) -> None:
        if self.turn_done:
            if normalize(text) == normalize(self.turn_text()):
                # A repeated/duplicate final update after the decision is not a barge-in.
                self.counters["eot_skipped"] += 1
                self._emit("asr_update", t_ms, text=text, final=bool(final), segment=self.segment,
                           turn_text=self.turn_text(), changed=False)
                self._emit("eot_skipped", t_ms, reason="unchanged_after_turn_done", text=text)
                return
            self._emit("barge_in", t_ms, why="asr_after_action", text=text)
            self._reset_turn(t_ms)
        if self.turn_start_ms is None:
            self.turn_start_ms = t_ms
        if self.segment_closed:
            self.segments.append(self.segment)
            self.segment = ""
            self.segment_closed = False
        before = normalize(self.turn_text())
        self.segment = text
        changed = normalize(self.turn_text()) != before
        if changed:
            self.version += 1
        self._emit("asr_update", t_ms, text=text, final=bool(final), segment=self.segment,
                   turn_text=self.turn_text(), changed=changed)
        if not changed:
            # No new text: keep the current state and any in-flight judgement for the same version.
            self.counters["eot_skipped"] += 1
            self._emit("eot_skipped", t_ms, reason="unchanged", text=text)
            return
        if self.state != "LISTENING":
            self._to_listening(t_ms, "new_text")
        self.last_change_ms = t_ms
        self.t_last_word = t_ms
        self._cancel_inflight_llm(t_ms, "new_text")
        self._eot_pending = {"version": self.version, "text": self.eot_input(), "due_ms": t_ms + self.cfg.debounce_ms}
        self._spec_pending = {"version": self.version, "text": self.turn_text(),
                              "due_ms": t_ms + self.cfg.debounce_ms, "reason": "text_change"}
        if final and self.cfg.speculative_chain:
            # The last ASR update is the whole utterance: give that text the remaining confirm window
            # instead of waiting for the in-flight (now stale) chain to finish first.
            self._spec_pending["due_ms"] = t_ms
            self._spec_pending["reason"] = "final_update"
        self.eot_scores = None
        self.tick(t_ms)

    def on_vad(self, t_ms: int, kind: str) -> None:
        self._emit("vad", t_ms, kind=kind)
        if kind == "speech_start":
            if self.turn_done:
                self._reset_turn(t_ms)
            elif self.state != "LISTENING":
                self._to_listening(t_ms, "speech_start")
            if self.segment:
                self.segment_closed = True
            # The user is speaking again: the previous silence interval is over.
            self.speech_start_ms = t_ms
            self._cancel_inflight_llm(t_ms, "speech_start")
        elif kind == "speech_end":
            self.last_speech_end_ms = t_ms
            if self.cfg.speculative_chain and self.turn_start_ms is not None:
                # No new words since the last speculate: still the newest text, so let it preempt.
                self._maybe_speculate(t_ms, "speech_end")
        self.tick(t_ms)

    def tick(self, now_ms: int) -> None:
        if self.turn_start_ms is None:
            return
        self._poll(now_ms)
        silence = self._silence_ms(now_ms)
        if self.state == "THINK_WAIT":
            # A thinking pause is allowed to outlast max_silence_ms; only think_wait_ms confirms it.
            if silence >= self.cfg.think_wait_ms:
                self._confirm(now_ms, "think_wait")
        elif self.state == "MAYBE_DONE":
            if silence >= self.cfg.confirm_silence_ms:
                self._confirm(now_ms, "confirm_silence")
            elif silence >= self.cfg.max_silence_ms:
                self._confirm(now_ms, "max_silence")
        elif self.state == "CUT_OFF_WAIT":
            if silence >= self.cfg.cut_wait_ms:
                self._confirm(now_ms, "cut_wait")
            elif silence >= self.cfg.max_silence_ms:
                self._confirm(now_ms, "max_silence")
        elif self.state == "LISTENING" and silence >= self.cfg.max_silence_ms:
            self._confirm(now_ms, "max_silence")
        for job in self._eot_jobs:
            if not job.get("timed_out") and now_ms - job["started_ms"] > self.cfg.eot_timeout_ms:
                # Design doc: a KEV failure/timeout falls back to the max_silence rule below; the late
                # result is still scored and reported, it just no longer gates the state machine.
                job["timed_out"] = True
                self.counters["eot_skipped"] += 1
                self._emit("eot_skipped", now_ms, reason="timeout", text=job["text"])
        while len(self._eot_jobs) < max(1, self.cfg.eot_max_inflight) and self._eot_pending \
                and now_ms >= self._eot_pending["due_ms"]:
            pending, self._eot_pending = self._eot_pending, None
            self._fire_eot(now_ms, pending)
        if self._spec_pending and now_ms >= self._spec_pending["due_ms"]:
            pending, self._spec_pending = self._spec_pending, None
            self._maybe_speculate(now_ms, pending.get("reason", "text_change"))
        if self.state == "CONFIRMED":
            self._maybe_start_llm(now_ms)

    def wait_until(self, now_ms: int) -> None:
        self.tick(now_ms)

    def idle(self) -> bool:
        return (not any(self._jobs.values()) and not self._eot_jobs
                and not self._eot_pending and not self._spec_pending)

    def summary(self) -> dict:
        return {"session_id": self.session_id, "turns": self.turn_index,
                "t_last_word": self.t_last_word, "t_confirmed": self.t_confirmed,
                "t_llm_start": self.t_llm_start, "t_decision": self.t_decision,
                "routed_by": self.routed_by,
                # How many results this session has written back so far (one per finished turn).
                "writebacks": len(self.context.get(COMPLETED_STEPS) or []),
                "t_confirm_latency_ms": self._delta(self.t_confirmed, self.t_last_word),
                "t_llm_start_latency_ms": self._delta(self.t_llm_start, self.t_last_word),
                "t_decision_latency_ms": self._delta(self.t_decision, self.t_last_word),
                "decision_after_confirm_ms": self._delta(self.t_decision, self.t_confirmed),
                **self.counters}

    # ---------- text ----------

    def turn_text(self) -> str:
        return self._join(self.segments + ([self.segment] if self.segment else []))

    def eot_input(self) -> str:
        if self.cfg.eot_input == "turn_text":
            return self.turn_text()
        if len(normalize(self.segment)) >= 4 or not self.segments:
            return self.segment
        return self._join(self.segments[-1:] + [self.segment])

    @staticmethod
    def _join(parts):
        text = ""
        for part in parts:
            if not part:
                continue
            if text and text[-1].isascii() and part[0].isascii():
                text += " "
            text += part
        return text

    # ---------- workers ----------

    def _fire_eot(self, now_ms, pending):
        self.counters["eot_calls"] += 1
        self._emit("eot_start", now_ms, text=pending["text"], version=pending["version"])
        self._eot_jobs.append({"future": self._pool.submit(self.eot_fn, pending["text"]),
                               "version": pending["version"], "started_ms": now_ms, "text": pending["text"]})

    def _maybe_speculate(self, now_ms, reason):
        """Start (or restart) the speculative chain for the newest text.

        Policy: at most one chain in flight. A text change is served when the pool is free or when
        the update is `final`/`speech_end` (then the older chain is cancelled and its late result is
        dropped - counted as waste). A finished stale chain is immediately re-fired once for the
        newest text, so the speculation always converges on the text that confirm will see.
        """
        if self.turn_done or self.state == "CONFIRMED":
            return
        text = self.turn_text()
        if not normalize(text):
            return
        job = self._jobs["spec"]
        if job is not None and job["norm"] == normalize(text):
            return                      # already speculating exactly this text
        preempt = reason in {"final_update", "speech_end"}
        if job is not None and not job["canceled"] and not preempt:
            return                      # one chain at a time; the change is picked up when it ends
        if job is not None and not job["canceled"]:
            # Preempted: keep polling it so the late result is validated and explicitly dropped
            # (spec A.3), but it can never be reused for a different text.
            job["canceled"] = True
            self._spec_orphans.append(job)
            self.counters["spec_wasted"] += 1
            self._emit("spec_cancel", now_ms, reason=reason, text=job["text"])
        key = text_key(text)
        self.counters["spec_calls"] += 1
        self.counters["tool_calls"] += 1
        self._emit("spec_start", now_ms, text=text, version=self.version, reason=reason,
                   sha256=key["sha256"])
        self._jobs["spec"] = {
            "future": self._spec_pool.submit(self._speculate, text, self.version, key),
            "version": self.version, "started_ms": now_ms, "text": text,
            "norm": key["norm"], "sha256": key["sha256"], "canceled": False}

    def _speculate(self, text, version, key):
        """Worker: tool shortlist, then (optional) the finite-fast gate and its KEV parameter fill."""
        result = {"text": text, "version": version, **{name: key[name] for name in ("norm", "sha256")},
                  "selection": None, "fast": None, "record": {}, "gated": False}
        try:
            result["selection"] = self.select_tools_fn(text, self.cfg.tool_set, self.tool_context())
        except Exception as error:      # surfaced by the caller; never kills the turn
            result["error"] = f"{type(error).__name__}: {error}"
            return result
        if not (self.cfg.speculative_chain and self.cfg.finite_fast_path):
            return result
        result["gated"] = True
        record = {}
        selection = result["selection"] or {}
        try:
            fast = self.finite_fast_fn(text, selection.get("selected", []), selection.get("scores") or {},
                                       selection.get("schemas") or [], self.cfg, record=record)
        except Exception as error:      # a broken gate must degrade, never kill the turn
            record["finite_fast_error"] = f"{type(error).__name__}: {error}"
            record["finite_fast_reason"] = "gate_error"
            fast = None
        result["fast"], result["record"] = fast, record
        return result

    def tool_context(self):
        """What KEV tool scoring sees: the fixed public context, without cumulative writebacks.

        context[COMPLETED_STEPS] is Agent history for the decide() brief; feeding it into the
        shortlist would change KEV's selection prompt as a side effect of this PoC. Selection stays
        keyed on the utterance, the decision stage gets the full picture.
        """
        return {name: value for name, value in self.context.items() if name != COMPLETED_STEPS}

    def _start_llm(self, now_ms, speculative):
        # `llm_calls` now counts real decide() calls only (see _on_llm_result); the fast path
        # counts finite_fast_hits instead, so a HIT turn never looks like an LLM turn.
        self.t_llm_start = now_ms
        text = self.turn_text()
        key = text_key(text)
        handle = self._spec_handle(key)
        ready = (handle or {}).get("result") if handle and handle["kind"] == "done" else None
        result = (ready or {}).get("selection") or {}
        self._emit("llm_start", now_ms, speculative=bool(speculative), selected=result.get("selected", []),
                   finite_fast=bool(self.cfg.finite_fast_path),
                   spec_reused=handle is not None,
                   spec_waiting=bool(handle and handle["kind"] == "running"),
                   spec_gated=bool(ready and ready.get("gated")))
        if handle is None:
            # No speculative result for the final text: synchronous recompute (spec A.2).
            self._emit("spec_recompute", now_ms, reason="no_matching_spec", text=text)
            future = self._pool.submit(self._decide_after_select, text)
        elif handle["kind"] == "done":
            future = self._pool.submit(self._decide, text, ready["selection"],
                                       ({"fast": ready["fast"], "record": ready["record"]}
                                        if ready.get("gated") else None))
        else:
            # The chain for the final text is still running: join it in a worker, never on the
            # driver thread, so the state machine keeps ticking while it finishes.
            self.counters["spec_waited"] += 1
            future = self._pool.submit(self._wait_then_decide, text, handle["future"])
        self._jobs["llm"] = {"future": future, "version": self.version, "text": text,
                             "sha256": key["sha256"], "speculative": bool(speculative),
                             "canceled": False, "spec_reused": handle is not None,
                             "spec_gated": bool(handle and handle.get("kind") == "done"
                                                and (handle.get("result") or {}).get("gated"))}

    def _spec_handle(self, key):
        """Newest speculation for `key`: {"kind": "done"|"running"} or None when there is none.

        Never blocks: a running chain is handed to the decision worker, which joins it off the
        driver thread.
        """
        if not self.cfg.reuse_tools_if_text_unchanged:
            return None
        ready = self._spec_ready
        if ready is not None and ready["sha256"] == key["sha256"]:
            return {"kind": "done", "result": ready}
        job = self._jobs["spec"]
        if job is not None and not job["canceled"] and job["sha256"] == key["sha256"]:
            return {"kind": "running", "future": job["future"]}
        return None

    def _wait_then_decide(self, utterance, future):
        """Worker: join the speculative chain for this text, then decide with what it produced."""
        try:
            result = future.result()
        except Exception as error:
            self._emit("error", 0, stage="spec_join", error_type=type(error).__name__, error=str(error))
            result = None
        key = text_key(utterance)
        if not result or result.get("sha256") != key["sha256"] or result.get("selection") is None:
            self._emit("spec_recompute", 0, reason="spec_result_unusable", text=utterance)
            return self._decide_after_select(utterance)
        return self._decide(utterance, result["selection"],
                            ({"fast": result["fast"], "record": result["record"]}
                             if result.get("gated") else None))

    def _decide_after_select(self, utterance):
        """Worker: no usable speculation for this text, so select and gate here (the old path)."""
        try:
            tools = self.select_tools_fn(utterance, self.cfg.tool_set, self.tool_context())
        except Exception as error:
            raise RuntimeError("KEV tool scoring failed: " + str(error)) from error
        return self._decide(utterance, tools)

    def _decide(self, utterance, tools, precomputed=None):
        """Worker: finite fast path first, then the existing decide() (DeepSeek) when it MISSes."""
        record = {}
        selected, schemas = tools.get("selected", []), tools.get("schemas", [])
        gate_precomputed = precomputed is not None
        if precomputed is not None:
            # The gate already ran on this exact text inside the speculative chain.
            record = precomputed.get("record") or {}
            fast = precomputed.get("fast")
            if fast is not None:
                return {"finite_fast": True, "finite_fast_record": record,
                        "gate_precomputed": True, **fast}
        elif self.cfg.finite_fast_path:
            try:
                fast = self.finite_fast_fn(utterance, selected, tools.get("scores") or {}, schemas,
                                           self.cfg, record=record)
            except Exception as error:  # a broken gate must degrade, never kill the turn
                record["finite_fast_error"] = type(error).__name__ + ": " + str(error)
                record["finite_fast_reason"] = "gate_error"
                fast = None
            if fast is not None:
                return {"finite_fast": True, "finite_fast_record": record, **fast}
        outcome = self.decide_fn(utterance, selected, schemas, self.cfg.tool_set, self.context,
                                 native_schemas=tools.get("native_schemas"),
                                 available_domains=tools.get("domains"))
        return {"finite_fast": False, "finite_fast_record": record,
                "gate_precomputed": gate_precomputed, **outcome}

    def _poll(self, now_ms):
        for job in [job for job in self._eot_jobs if job["future"].done()]:
            self._eot_jobs.remove(job)
            try:
                result = job["future"].result()
            except Exception as error:  # PoC: surface, do not retry
                self._emit("error", now_ms, stage="eot", error_type=type(error).__name__, error=str(error))
                continue
            self._on_eot_result(now_ms, job, result)
        self._poll_spec_orphans(now_ms)
        for kind in ("spec", "llm"):
            job = self._jobs[kind]
            if job is None or not job["future"].done():
                continue
            self._jobs[kind] = None
            try:
                result = job["future"].result()
            except Exception as error:  # PoC: surface, do not retry
                self._emit("error", now_ms, stage=kind, error_type=type(error).__name__, error=str(error))
                continue
            handler = {"spec": self._on_spec_result, "llm": self._on_llm_result}[kind]
            handler(now_ms, job, result)

    def _poll_spec_orphans(self, now_ms):
        """Preempted chains: validate their late results and drop them, never reuse them."""
        for job in [job for job in self._spec_orphans if job["future"].done()]:
            self._spec_orphans.remove(job)
            try:
                result = job["future"].result()
            except Exception as error:
                self._emit("error", now_ms, stage="spec_orphan", error_type=type(error).__name__,
                           error=str(error))
                continue
            self.counters["spec_stale_results"] += 1
            self._emit("spec_discarded", now_ms, reason="canceled_late", text=job["text"],
                       sha256=(result or {}).get("sha256"))

    def _on_eot_result(self, now_ms, job, result):
        stale = job["version"] != self.version
        if stale:
            # The judgement was paid for but no longer describes the newest text: report it (metrics
            # need every prefix score) and drop it from the state machine.
            self.counters["eot_skipped"] += 1
            self.counters["eot_stale"] += 1
        self.eot_scores = result if not stale else self.eot_scores
        self._emit("eot_scores", now_ms, stale=stale, text=job["text"], **result)
        if stale:
            self._emit("eot_skipped", now_ms, reason="stale", text=job["text"])
            return
        if self.state == "CONFIRMED":
            return
        label = result["label"]
        margin = result["margin"]
        if label == "complete":
            if self.cfg.eot_policy == "margin" and margin < self.cfg.eot_margin:
                label = "cut" if result["cut"] >= result["wait"] else "wait"
            else:
                self._to_state("MAYBE_DONE", now_ms, "eot_complete")
                if self.cfg.speculative_llm and margin >= self.cfg.spec_margin:
                    self._maybe_start_llm(now_ms, speculative=True)
                return
        self._to_state("CUT_OFF_WAIT" if label == "cut" else "THINK_WAIT", now_ms, "eot_" + label)

    def _on_spec_result(self, now_ms, job, result):
        current = text_key(self.turn_text())
        if job["canceled"] or result.get("sha256") != current["sha256"]:
            # The result describes a text the user has already moved past: drop it, never execute.
            if not job["canceled"]:
                self.counters["spec_wasted"] += 1
            self.counters["spec_stale_results"] += 1
            self._emit("spec_discarded", now_ms,
                       reason="canceled" if job["canceled"] else "stale_text",
                       text=job["text"], sha256=job["sha256"])
            if not job["canceled"] and self.state != "CONFIRMED" and self.cfg.speculative_chain:
                self._maybe_speculate(now_ms, "stale_result")
            return
        self._spec_ready = result
        self._emit("spec_ready", now_ms, text=job["text"], fast=bool(result.get("fast")),
                   selection=(result.get("selection") or {}).get("selected", []),
                   waited_ms=now_ms - job["started_ms"], gated=bool(result.get("gated")))
        self._maybe_start_llm(now_ms, speculative=self.cfg.speculative_llm and self.state == "MAYBE_DONE")

    def _on_llm_result(self, now_ms, job, result):
        fast = bool(result.get("finite_fast"))
        record = result.get("finite_fast_record") or {}
        if job.get("spec_reused"):
            # The decision consumed a chain that was already computed for the final text; the gate
            # half only counts when the gate really ran before confirm (spec_gated / precomputed).
            if result.get("gate_precomputed"):
                self.counters["spec_reused_hit" if fast else "spec_reused_gate"] += 1
            else:
                self.counters["spec_reused_selection"] += 1
        if fast:
            self.counters["finite_fast_hits"] += 1
        else:
            self.counters["llm_calls"] += 1
            if record.get("finite_fast_reason"):
                self.counters["finite_fast_misses"] += 1
            if record.get("finite_fast_error"):
                self.counters["finite_fast_errors"] += 1
        if job["canceled"]:
            return
        key = text_key(self.turn_text())
        if job["version"] != self.version or job["sha256"] != key["sha256"]:
            self.counters["llm_wasted"] += 1
            self._emit("llm_cancel", now_ms, reason="stale_text", text=job["text"],
                       sha256=job["sha256"])
            return
        result = {**result, "text": job["text"], "sha256": job["sha256"],
                  "spec_reused": job.get("spec_reused")}
        if self.state != "CONFIRMED":
            # Speculative result arrived early; keep it until the turn is confirmed.
            self._llm_prepared = {"version": job["version"], "result": result}
            return
        self._apply_llm(now_ms, result)

    def _apply_llm(self, now_ms, result):
        fields = result.get("log_fields") or {}
        fast = bool(result.get("finite_fast"))
        record = result.get("finite_fast_record") or {}
        # Hard rule (spec A.3): an execution/writeback on any text other than the final text is a
        # bug, so the key is re-checked here - the last gate before mock execution and writeback.
        current = text_key(self.turn_text())
        if result.get("sha256") != current["sha256"]:
            self.counters["spec_stale_executions_blocked"] += 1
            self._emit("spec_stale_execution_blocked", now_ms, text=result.get("text"),
                       sha256=result.get("sha256"), final_sha256=current["sha256"])
            self._emit("spec_recompute", now_ms, reason="stale_before_execution", text=self.turn_text())
            self._jobs["llm"] = None
            self._spec_ready = None
            self._start_llm(now_ms, speculative=False)
            return
        if fields.get("routed_by") is None:
            fields = {**fields, "routed_by": "kev" if fast else "llm"}
        if fast:
            self._emit("finite_fast_hit", now_ms, tool=result["decision"].get("tool"),
                       tool_score=fields.get("tool_score"),
                       param_confidences=fields.get("param_confidences"), kev_ms=result.get("llm_ms"))
        elif record.get("finite_fast_reason"):
            self._emit("finite_fast_miss", now_ms, reason=record["finite_fast_reason"],
                       detail=record.get("finite_fast_detail"), tool=record.get("finite_fast_tool"),
                       tool_score=record.get("finite_fast_tool_score"),
                       error=record.get("finite_fast_error"))
        self.decision = result["decision"]
        self.t_decision = now_ms
        self.routed_by = "kev" if fast else "llm"
        decision_head = fields.get("decision_head") or ("kev_finite_fast" if fast else None)
        self._emit("llm_decision", now_ms, decision=result["decision"], llm_ms=result.get("llm_ms"),
                   decision_head=decision_head, routed_by=self.routed_by, finite_fast=fast,
                   text_sha256=result.get("sha256"), spec_reused=bool(result.get("spec_reused")))
        # The gap this closes: a HIT skipped DeepSeek, so nothing downstream knew the turn had run.
        # Both routes now end in the same mock execution + writeback before turn_done.
        self.miss_reason = None if fast else record.get("finite_fast_reason")
        self._writeback(now_ms, decision_head)
        self._finish_turn(now_ms)

    def _writeback(self, now_ms, decision_head):
        """Mock-execute this turn's decision and file it where the Agent can read it next turn."""
        writeback = writeback_result(utterance=self.turn_text(), decision=self.decision,
                                     routed_by=self.routed_by, decision_head=decision_head,
                                     miss_reason=self.miss_reason, ts_ms=time.time() * 1000,
                                     turn=self.turn_index, turn_ms=now_ms)
        append_writeback(self.context, writeback)
        self._emit(WRITEBACK_EVENT, now_ms, **writeback)
        return writeback

    # ---------- state ----------

    def _silence_ms(self, now_ms):
        anchors = [value for value in (self.last_change_ms, self.last_speech_end_ms,
                                       self.speech_start_ms, self.turn_start_ms) if value is not None]
        return now_ms - max(anchors) if anchors else 0

    def _maybe_start_llm(self, now_ms, speculative=False):
        if self.turn_done or self.state not in ("CONFIRMED", "MAYBE_DONE"):
            return
        if self.state == "CONFIRMED" and self._llm_prepared and self._llm_prepared["version"] == self.version:
            prepared, self._llm_prepared = self._llm_prepared, None
            self._apply_llm(now_ms, prepared["result"])
            return
        job = self._jobs["llm"]
        key = text_key(self.turn_text())
        if job is not None and not job["canceled"] and job["version"] == self.version \
                and job["sha256"] == key["sha256"]:
            return
        if self.state == "MAYBE_DONE" and not speculative:
            return
        self._start_llm(now_ms, speculative)

    def _cancel_inflight_llm(self, now_ms, reason):
        """Drop a decision that the user's continuation invalidated (in flight or already finished)."""
        prepared, self._llm_prepared = self._llm_prepared, None
        if prepared is not None:
            # A speculative result that already finished is discarded too, and counted as waste.
            self.counters["llm_wasted"] += 1
            self._emit("llm_cancel", now_ms, reason=reason, speculative=True, prepared=True)
        job = self._jobs["llm"]
        if job is None or job["canceled"]:
            return
        job["canceled"] = True
        self.counters["llm_wasted"] += 1
        self._emit("llm_cancel", now_ms, reason=reason, speculative=job["speculative"], prepared=False)

    def _to_state(self, state, now_ms, why):
        if state == self.state:
            return
        previous, self.state = self.state, state
        self._emit("state_change", now_ms, **{"from": previous, "to": state, "why": why})

    def _to_listening(self, now_ms, why):
        self._to_state("LISTENING", now_ms, why)

    def _confirm(self, now_ms, why):
        if self.state == "CONFIRMED":
            return
        self.t_confirmed = now_ms
        self._to_state("CONFIRMED", now_ms, why)
        self._maybe_start_llm(now_ms)

    def _finish_turn(self, now_ms):
        self.turn_done = True
        job = self._jobs["spec"]
        if job is not None and not job["canceled"] and not job["future"].done():
            job["canceled"] = True      # its late result is dropped, not executed
        summary = self.summary()
        self.turns.append(summary)
        path = self._timeline_path()
        self._emit("turn_done", now_ms, **summary, timeline_path=str(path) if path else None)
        self._dump_timeline()

    def _timeline_path(self):
        return Path(self.evidence_root) / self.session_id / "timeline.json" if self.evidence_root else None

    def _reset_turn(self, now_ms):
        # Deliberately NOT reset: self.context, and therefore context[COMPLETED_STEPS]. The writeback
        # of earlier turns is what makes the next decide() brief a complete picture; clearing it here
        # would restore the exact gap this PoC closes.
        self.turn_index += 1
        self.turn_done = False
        self.segment = ""
        self.segments = []
        self.segment_closed = False
        self.last_change_ms = None
        self.last_speech_end_ms = None
        self.speech_start_ms = now_ms
        self.t_last_word = None
        self.t_confirmed = None
        self.t_llm_start = None
        self.t_decision = None
        self.turn_start_ms = now_ms
        self.state = "LISTENING"
        job = self._jobs["spec"]
        if job is not None and not job["canceled"]:
            job["canceled"] = True
        self._jobs["llm"] = None
        self._jobs["spec"] = None
        self._eot_jobs = []
        self._eot_pending = None
        self._spec_pending = None
        self._spec_ready = None
        self._spec_orphans = []
        self._llm_prepared = None
        self.eot_scores = None
        self.decision = None
        self.routed_by = None
        self.miss_reason = None
        self.counters = {"eot_calls": 0, "eot_skipped": 0, "eot_stale": 0, "tool_calls": 0,
                         "llm_calls": 0, "llm_wasted": 0,
                         "finite_fast_hits": 0, "finite_fast_misses": 0, "finite_fast_errors": 0,
                         "spec_calls": 0, "spec_wasted": 0, "spec_reused_hit": 0,
                         "spec_reused_selection": 0, "spec_reused_gate": 0,
                         "spec_stale_results": 0,
                         "spec_stale_executions_blocked": 0, "spec_recomputes": 0,
                         "spec_waited": 0}

    # ---------- events / evidence ----------

    def _emit(self, name, t_ms, **payload):
        event = {"event": name, "t_ms": t_ms, "text_version": self.version, **payload}
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    @staticmethod
    def _delta(later, earlier):
        return None if later is None or earlier is None else later - earlier

    def _dump_timeline(self):
        if self.evidence_root is None:
            return
        folder = Path(self.evidence_root) / self.session_id
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"session_id": self.session_id, "config": asdict(self.cfg),
                   "events": self.events, "turns": self.turns, "summary": self.summary()}
        (folder / "timeline.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        self.timeline_path = folder / "timeline.json"
        return self.timeline_path

    def close(self):
        self._pool.shutdown(wait=False)
        self._spec_pool.shutdown(wait=False)