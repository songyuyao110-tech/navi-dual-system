"""Speculative decision chain (tmp/poc-speculative-toolselect-spec.md A.2/A.3).

The chain (KEV tool scoring -> finite-fast gate -> KEV parameter filling) runs on ASR partials,
in parallel with EOT judging. Every result is bound to normalized text + sha256 + text_version;
at confirm only the result whose text equals the final text may be reused, anything else is
discarded and recomputed. These tests pin the hard rule: an execution/writeback on a text other
than the final text must be impossible, including the "old chain lands after the user moved on"
race that a wall-clock worker makes inevitable.

All KEV and decisions are fakes: nothing here touches a tunnel, LiteLLM, or a vehicle.
"""
import json
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

from result_writeback import COMPLETED_STEPS
from turn_session import TurnConfig, TurnSession, normalize, text_key

CLIMATE_SCHEMA = {"name": "climate_set_power", "description": "开启或关闭座舱空调系统",
                  "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}},
                                 "required": ["enabled"], "additionalProperties": False}}
OPEN_SCHEMA = {"name": "climate_set_temperature", "description": "设置空调温度",
               "parameters": {"type": "object",
                              "properties": {"zone": {"type": "string", "enum": ["driver"]},
                                             "celsius": {"type": "number"}}}}


def eot_complete(text):
    return {"complete": 0.95, "cut": 0.03, "wait": 0.02, "label": "complete", "margin": 0.92,
            "wall_ms": 1.0, "kev_ms": 1.0}


class Recorder:
    """Thread-safe call log; `delay` simulates the KEV segment the chain must hide."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()

    def add(self, kind, payload):
        with self.lock:
            self.calls.append({"kind": kind, "at": time.monotonic() * 1000, "payload": payload})

    def of(self, kind):
        with self.lock:
            return [call for call in self.calls if call["kind"] == kind]


def make_tools(recorder, *, delay=None, tool="climate_set_power", score=0.95, schema=CLIMATE_SCHEMA):
    def select(text, tool_set, context):
        recorder.add("select", text)
        if delay:
            time.sleep(delay)
        return {"scores": {"tool:" + tool: score}, "selected": [tool], "schemas": [schema],
                "timing": {"stage1_ms": 1.0, "stage2_ms": 1.0, "total_ms": 2.0},
                "native_schemas": [schema], "domains": []}
    return select


def fast_hit(recorder, *, delay=0.0):
    def gate(utterance, selected, scores, schemas, cfg, record=None):
        recorder.add("gate", utterance)
        if delay:
            time.sleep(delay)
        if not selected:
            if record is not None:
                record["finite_fast_reason"] = "empty_shortlist"
            return None
        if selected[0] != "climate_set_power":
            if record is not None:
                record.update(finite_fast_reason="not_allowlisted", finite_fast_tool=selected[0],
                              finite_fast_tool_score=scores.get("tool:" + selected[0]))
            return None
        if record is not None:
            record.update(finite_fast_reason="hit", finite_fast_tool=selected[0],
                          finite_fast_tool_score=scores.get("tool:" + selected[0]),
                          finite_fast_param_confidences={"enabled": 0.95})
        decision = {"kind": "tool_call", "tool": "climate_set_power", "arguments": {"enabled": True}}
        return {"decision": decision, "llm_ms": 7.0,
                "log_fields": {"decision_head": "kev_finite_fast", "routed_by": "kev",
                               "finite_fast": True, "decision": decision,
                               "decision_latency_ms": 7.0}}
    return gate


def make_decide(recorder, *, delay=0.05):
    def decide(utterance, selected, schemas, tool_set="navigation", context=None, **kwargs):
        recorder.add("decide", {"utterance": utterance, "selected": list(selected)})
        if delay:
            time.sleep(delay)
        return {"decision": {"kind": "tool_call", "tool": selected[0] if selected else "ask_user",
                             "arguments": {"enabled": True}},
                "llm_ms": delay * 1000,
                "log_fields": {"decision_head": "project_existing_litellm"}}
    return decide


def new_session(recorder, *, cfg=None, tools=None, gate=None, decide=None, session_id="spec"):
    cfg = cfg or TurnConfig(speculative_llm=False, debounce_ms=0, confirm_silence_ms=220,
                            max_silence_ms=220, eot_margin=0.2, speculative_chain=True)
    session = TurnSession(cfg, lambda event: None, eot_fn=eot_complete,
                          select_tools_fn=tools or make_tools(recorder),
                          finite_fast_fn=gate or fast_hit(recorder),
                          decide_fn=decide or make_decide(recorder),
                          evidence_root=None, session_id=session_id)
    return session


def drive(session, timeline, *, budget_ms=8000):
    began = time.monotonic() * 1000
    now = lambda: time.monotonic() * 1000 - began
    for item in timeline:
        while now() < item[0] and not session.turn_done:
            session.tick(now())
            time.sleep(0.003)
        if session.turn_done:
            break
        if item[1] == "asr":
            session.on_asr(item[0], item[2], item[3])
        else:
            session.on_vad(item[0], item[2])
    limit = now() + budget_ms
    while not session.turn_done and now() < limit:
        session.tick(now())
        time.sleep(0.003)
    return session


class ReuseTests(unittest.TestCase):
    def test_the_chain_is_reused_when_its_text_is_the_final_text(self):
        recorder = Recorder()
        session = new_session(recorder, tools=make_tools(recorder, delay=0.12),
                              gate=fast_hit(recorder, delay=0.04))
        try:
            events = session.events
            drive(session, [(0, "asr", "打开空调", True)])
        finally:
            session.close()
        self.assertTrue(session.turn_done)
        # One chain only: the shortlist was not recomputed after confirm.
        self.assertEqual(len(recorder.of("select")), 1)
        self.assertEqual(len(recorder.of("gate")), 1)
        self.assertEqual(recorder.of("gate")[0]["payload"], "打开空调")
        self.assertEqual(len(recorder.of("decide")), 0)          # HIT: the model is never called
        self.assertEqual(session.counters["spec_calls"], 1)
        self.assertEqual(session.counters["spec_reused_hit"], 1)
        self.assertEqual(session.counters["spec_stale_executions_blocked"], 0)
        self.assertEqual(session.routed_by, "kev")
        [writeback] = session.context[COMPLETED_STEPS]
        self.assertEqual(writeback["utterance"], "打开空调")
        self.assertEqual(writeback["tool"], "climate_set_power")
        decision_event = [event for event in session.events if event["event"] == "llm_decision"][-1]
        self.assertTrue(decision_event["spec_reused"])
        # The decision lands right after confirm instead of after another KEV select segment.
        self.assertLessEqual(session.summary()["decision_after_confirm_ms"], 1500)


class StaleChainTests(unittest.TestCase):
    def test_a_chain_for_an_earlier_prefix_is_discarded_not_executed(self):
        recorder = Recorder()
        session = new_session(recorder, tools=make_tools(recorder, delay=0.15),
                              gate=fast_hit(recorder, delay=0.02))
        try:
            drive(session, [(0, "asr", "打开", False),
                            (40, "asr", "打开空", False),
                            (80, "asr", "打开空调", True)])
        finally:
            session.close()
        self.assertTrue(session.turn_done)
        # The stale "打开空" chain landed, and was dropped instead of executed.
        self.assertGreaterEqual(session.counters["spec_stale_results"], 1)
        self.assertGreaterEqual(session.counters["spec_wasted"], 1)
        self.assertEqual(session.counters["spec_stale_executions_blocked"], 0)
        [writeback] = session.context[COMPLETED_STEPS]
        self.assertEqual(writeback["utterance"], "打开空调")     # never the unfinished prefix
        executed = [call for call in recorder.of("gate") if call["payload"] == "打开空调"]
        self.assertTrue(executed, "the final text must be the one that is gated and executed")

    def test_a_late_result_cannot_overwrite_the_decision(self):
        # The chain for the prefix is slow; it finishes *after* the turn is already decided.
        recorder = Recorder()
        release = threading.Event()

        def slow_select(text, tool_set, context):
            recorder.add("select", text)
            if normalize(text) == "打开":
                release.wait(5.0)                # the old prefix chain stalls past turn_done
            return {"scores": {"tool:climate_set_power": 0.95}, "selected": ["climate_set_power"],
                    "schemas": [CLIMATE_SCHEMA], "timing": {}, "native_schemas": [], "domains": []}

        session = new_session(recorder, tools=slow_select, gate=fast_hit(recorder),
                              cfg=TurnConfig(speculative_llm=False, debounce_ms=0,
                                             confirm_silence_ms=150, max_silence_ms=150,
                                             speculative_chain=True))
        try:
            drive(session, [(0, "asr", "打开", False), (30, "asr", "打开空调", True)])
            self.assertTrue(session.turn_done)
            release.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                session.tick(time.monotonic() * 1000)
                time.sleep(0.005)
        finally:
            release.set()
            session.close()
        [writeback] = session.context[COMPLETED_STEPS]
        self.assertEqual(writeback["utterance"], "打开空调")
        self.assertEqual(session.counters["spec_stale_executions_blocked"], 0)
        self.assertEqual(len(session.context[COMPLETED_STEPS]), 1)

    def test_the_last_gate_before_execution_rejects_a_stale_key(self):
        # Direct unit check of the guard in _apply_llm: even a result that arrives through the
        # prepared path with a stale sha256 must not reach mock execution or the writeback.
        recorder = Recorder()
        session = new_session(recorder)
        try:
            session.on_asr(0, "打开空调", False)
            session.state = "CONFIRMED"
            stale = {"decision": {"kind": "tool_call", "tool": "climate_set_power",
                                  "arguments": {"enabled": True}},
                     "finite_fast": True, "finite_fast_record": {"finite_fast_reason": "hit"},
                     "log_fields": {"decision_head": "kev_finite_fast", "routed_by": "kev"},
                     "text": "打开空", "sha256": text_key("打开空")["sha256"]}
            session._apply_llm(100.0, stale)
            self.assertEqual(session.counters["spec_stale_executions_blocked"], 1)
            self.assertEqual(session.context[COMPLETED_STEPS], [])
            self.assertFalse(session.turn_done)
            self.assertTrue(any(event["event"] == "spec_stale_execution_blocked"
                                for event in session.events))
            self.assertTrue(any(event["event"] == "spec_recompute" for event in session.events))
        finally:
            session.close()

    def test_confirm_without_a_matching_chain_recomputes_synchronously(self):
        recorder = Recorder()

        def tools(text, tool_set, context):
            recorder.add("select", text)
            return {"scores": {"tool:climate_set_power": 0.95}, "selected": ["climate_set_power"],
                    "schemas": [CLIMATE_SCHEMA], "timing": {}, "native_schemas": [], "domains": []}

        session = new_session(recorder, tools=tools,
                              cfg=TurnConfig(speculative_llm=False, debounce_ms=0,
                                             confirm_silence_ms=60, max_silence_ms=60,
                                             speculative_chain=False))
        try:
            drive(session, [(0, "asr", "打开空调", True)])
        finally:
            session.close()
        self.assertTrue(session.turn_done)
        # speculative_chain=False is the pre-change behaviour: the shortlist may already be ready
        # (that was true before this change too), but the gate can only run after confirm.
        self.assertEqual(session.counters["spec_reused_hit"], 0)
        self.assertEqual(session.counters["spec_reused_selection"], 1)
        gates = recorder.of("gate")
        self.assertEqual(len(gates), 1)
        self.assertTrue(any(event["event"] == "llm_start" and not event.get("spec_gated")
                            for event in session.events))
        [writeback] = session.context[COMPLETED_STEPS]
        self.assertEqual(writeback["utterance"], "打开空调")


class MissReuseTests(unittest.TestCase):
    def test_a_miss_reuses_the_shortlist_and_still_calls_the_model(self):
        recorder = Recorder()
        session = new_session(recorder,
                              tools=make_tools(recorder, delay=0.1, tool="climate_set_temperature",
                                               score=0.93, schema=OPEN_SCHEMA),
                              gate=fast_hit(recorder, delay=0.02))
        try:
            drive(session, [(0, "asr", "空调调到二十二度", True)])
        finally:
            session.close()
        self.assertTrue(session.turn_done)
        self.assertEqual(len(recorder.of("select")), 1)          # shortlist reused, not recomputed
        # The gate itself ran before confirm (spec_reused_gate), not just the shortlist.
        self.assertEqual(session.counters["spec_reused_gate"], 1)
        self.assertEqual(session.counters["llm_calls"], 1)
        self.assertEqual(session.routed_by, "llm")
        [decision] = recorder.of("decide")
        self.assertEqual(decision["payload"]["selected"], ["climate_set_temperature"])
        [writeback] = session.context[COMPLETED_STEPS]
        self.assertEqual(writeback["uttterance"] if "uttterance" in writeback else writeback["utterance"],
                         "空调调到二十二度")
        self.assertEqual(writeback["miss_reason"], "not_allowlisted")


class KeyTests(unittest.TestCase):
    def test_the_key_is_the_normalized_text(self):
        a = text_key("打开空调。")
        b = text_key(" 打开 空调 ")
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["norm"], "打开空调")
        self.assertNotEqual(a["sha256"], text_key("打开空调了")["sha256"])
        self.assertEqual(json.loads(json.dumps(a))["sha256"], a["sha256"])


if __name__ == "__main__":
    unittest.main()