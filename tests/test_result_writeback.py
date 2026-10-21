"""Acceptance A-D for result writeback (tmp/poc-finite-fast-result-writeback-spec.md §6).

A  HIT writeback      fast path turn -> result_writeback event + one context record
B  MISS writeback     gate misses -> LLM decision -> same record shape, routed_by=llm
C  complete context   the next turn's decide() reads the previous tool/arguments/exec_result
D  non tool_call      ask_user/done still write one record with status=skipped

All execution is mock: nothing here touches a vehicle, KEV, or LiteLLM.
"""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import core_pipeline_run
from finite_fast_path import try_finite_fast_decide
from result_writeback import COMPLETED_STEPS, EVENT, mock_execute, writeback_result
from turn_session import TurnConfig, TurnSession

CLIMATE_SCHEMA = {"name": "climate_set_power", "description": "开启或关闭座舱空调系统",
                  "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}},
                                 "required": ["enabled"], "additionalProperties": False}}
# Open numeric slot (0-100 percent): the 58-tool allowlist covers every pure enum/boolean tool, so a
# guaranteed finite-fast MISS now has to be a schema with an open slot, not a finite tool left out.
BRIGHTNESS_SCHEMA = {"name": "light_set_ambient_brightness", "description": "设置氛围灯亮度百分比",
                     "parameters": {"type": "object",
                                    "properties": {"brightness": {"type": "integer", "minimum": 0,
                                                                  "maximum": 100}},
                                    "required": ["brightness"], "additionalProperties": False}}


def eot(text):
    return {"complete": 0.95, "cut": 0.03, "wait": 0.02, "label": "complete", "margin": 0.92,
            "wall_ms": 1.0, "kev_ms": 1.0}


def kev_true(state, questions):
    """Mock KEV parameter filling: every asked parameter says 'true' with 0.95 confidence."""
    return {"model": "kev-latest", "latency_ms": 1.0, "answers": {
        name: {"type": "choice", "choice": "true", "confidence": 0.95,
               "probabilities": {"true": 0.95, "false": 0.05}} for name in questions}}


def fast_gate(kev_call_fn=kev_true):
    def gate(utterance, selected, scores, schemas, cfg, record=None):
        return try_finite_fast_decide(utterance, selected, scores, schemas, cfg,
                                      kev_call_fn=kev_call_fn, record=record)
    return gate


def climate_tools(text, tool_set, context):
    return {"scores": {"tool:climate_set_power": 0.95}, "selected": ["climate_set_power"],
            "schemas": [CLIMATE_SCHEMA], "timing": {"stage1_ms": 1.0, "stage2_ms": 1.0, "total_ms": 2.0}}


def light_tools(text, tool_set, context):
    # Open slot (integer percent): never eligible for the gate, so decide() has to run.
    return {"scores": {"tool:light_set_ambient_brightness": 0.93},
            "selected": ["light_set_ambient_brightness"],
            "schemas": [BRIGHTNESS_SCHEMA], "timing": {"stage1_ms": 1.0, "stage2_ms": 1.0, "total_ms": 2.0}}


def light_tool_then_climate_tools(text, tool_set, context):
    return light_tools(text, tool_set, context) if "刚才" in text else climate_tools(text, tool_set, context)


def recording_tools(seen_contexts, tools_fn=light_tool_then_climate_tools):
    def select(text, tool_set, context):
        seen_contexts.append(dict(context))     # snapshot: the session keeps mutating it
        return tools_fn(text, tool_set, context)
    return select


def llm_tool_call(seen_contexts):
    def decide(utterance, selected, schemas, tool_set="navigation", context=None, **kwargs):
        # Snapshot: the session keeps mutating this dict after the call returns.
        seen_contexts.append(json.loads(json.dumps(context, ensure_ascii=False)))
        return {"decision": {"kind": "tool_call", "tool": "light_set_ambient_brightness",
                             "arguments": {"brightness": 60}},
                "llm_ms": 1.0, "log_fields": {"decision_head": "project_existing_litellm"}}
    return decide


def make_session(events, *, tools_fn=climate_tools, decide_fn=None):
    session = TurnSession(TurnConfig(speculative_llm=False, debounce_ms=0), events.append,
                          eot_fn=eot, select_tools_fn=tools_fn, finite_fast_fn=fast_gate(),
                          decide_fn=decide_fn or llm_tool_call([]),
                          evidence_root=None, session_id="writeback")
    return session


def wait_turn(session, began, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not session.turn_done and time.monotonic() < deadline:
        session.tick(time.monotonic() * 1000 - began)
        time.sleep(0.002)
    return session.turn_done


def writebacks(events):
    return [event for event in events if event["event"] == EVENT]


class UnitContractTests(unittest.TestCase):
    def test_tool_call_executes_and_missing_tool_fails(self):
        status, result = mock_execute({"kind": "tool_call", "tool": "climate_set_power",
                                       "arguments": {"enabled": True}})
        self.assertEqual(status, "completed")
        self.assertTrue(result["mock_only"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["executed_tools"], 0)       # mock: no real business execution
        self.assertIn("climate_set_power", result["message"])
        self.assertEqual(mock_execute({"kind": "tool_call"})[0], "failed")

    def test_routed_by_is_closed(self):
        with self.assertRaises(ValueError):
            writeback_result(utterance="x", decision={"kind": "done"},
                             routed_by="gateway", decision_head=None)

    def test_record_carries_the_contract_fields(self):
        record = writeback_result(utterance="打开空调",
                                  decision={"kind": "tool_call", "tool": "climate_set_power",
                                            "arguments": {"enabled": True}},
                                  routed_by="kev", decision_head="kev_finite_fast",
                                  turn=1, turn_ms=12.5)
        self.assertEqual(set(record), {"utterance", "tool", "arguments", "exec_result", "routed_by",
                                       "decision_head", "kind", "status", "ts_ms", "turn", "turn_ms"})
        self.assertIsInstance(record["ts_ms"], int)
        self.assertEqual(record["turn_ms"], 12.5)


class HitWritebackTests(unittest.TestCase):
    def test_hit_writes_back_tool_and_result(self):
        events = []
        session = make_session(events)
        began = time.monotonic() * 1000
        try:
            session.on_asr(0, "打开空调", True)
            self.assertTrue(wait_turn(session, began))
        finally:
            session.close()
        self.assertEqual(session.counters["finite_fast_hits"], 1)
        self.assertEqual(session.routed_by, "kev")
        [record] = writebacks(events)
        self.assertEqual(record["utterance"], "打开空调")
        self.assertEqual(record["tool"], "climate_set_power")
        self.assertEqual(record["arguments"], {"enabled": True})
        self.assertEqual(record["routed_by"], "kev")
        self.assertEqual(record["decision_head"], "kev_finite_fast")
        self.assertEqual(record["kind"], "tool_call")
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["exec_result"]["ok"])
        self.assertNotIn("miss_reason", record)
        [stored] = session.context[COMPLETED_STEPS]
        self.assertEqual(stored["tool"], "climate_set_power")
        self.assertEqual(session.summary()["writebacks"], 1)
        order = [event["event"] for event in events]
        self.assertLess(order.index("llm_decision"), order.index(EVENT))
        self.assertLess(order.index(EVENT), order.index("turn_done"))


class MissWritebackTests(unittest.TestCase):
    def test_miss_writes_back_the_llm_result(self):
        events, seen = [], []
        session = make_session(events, tools_fn=light_tools, decide_fn=llm_tool_call(seen))
        began = time.monotonic() * 1000
        try:
            session.on_asr(0, "氛围灯调成红色", True)
            self.assertTrue(wait_turn(session, began))
        finally:
            session.close()
        self.assertEqual(session.counters["finite_fast_hits"], 0)
        self.assertEqual(len(seen), 1)
        [record] = writebacks(events)
        self.assertEqual(record["routed_by"], "llm")
        self.assertEqual(record["decision_head"], "project_existing_litellm")
        self.assertEqual(record["miss_reason"], "not_allowlisted")
        self.assertEqual(record["tool"], "light_set_ambient_brightness")
        self.assertEqual(record["arguments"], {"brightness": 60})
        self.assertEqual(record["status"], "completed")
        [stored] = session.context[COMPLETED_STEPS]
        self.assertEqual(stored, {key: record[key] for key in stored})   # event envelope dropped

    def test_hit_and_miss_share_one_shape(self):
        hit_events, miss_events = [], []
        began = time.monotonic() * 1000
        for tools_fn, utterance, collected in ((climate_tools, "打开空调", hit_events),
                                                (light_tools, "氛围灯调成红色", miss_events)):
            session = make_session(collected, tools_fn=tools_fn)
            try:
                session.on_asr(0, utterance, True)
                self.assertTrue(wait_turn(session, began + len(collected)))
            finally:
                session.close()
        [hit], [miss] = writebacks(hit_events), writebacks(miss_events)
        self.assertEqual(set(miss) - set(hit), {"miss_reason"})
        self.assertLessEqual(set(hit), set(miss))


class CompleteContextTests(unittest.TestCase):
    def test_second_turn_sees_the_previous_tool_and_result(self):
        events, seen, scored = [], [], []
        session = make_session(events, tools_fn=recording_tools(scored),
                               decide_fn=llm_tool_call(seen))
        began = time.monotonic() * 1000
        try:
            session.on_asr(0, "打开空调", True)
            self.assertTrue(wait_turn(session, began))
            self.assertEqual(len(session.context[COMPLETED_STEPS]), 1)
            started = time.monotonic() * 1000 - began
            session.on_asr(started, "刚才调了什么", True)
            self.assertTrue(wait_turn(session, began))
        finally:
            session.close()
        self.assertEqual(session.turn_index, 2)
        self.assertEqual(len(seen), 1)                      # turn 2 really went to decide()
        [first, second] = session.context[COMPLETED_STEPS]
        self.assertEqual(first["tool"], "climate_set_power")
        self.assertEqual(first["arguments"], {"enabled": True})
        self.assertEqual(second["utterance"], "刚才调了什么")
        carried = seen[0][COMPLETED_STEPS]                  # what decide() was handed on turn 2
        self.assertEqual(len(carried), 1)
        self.assertEqual(carried[0]["tool"], "climate_set_power")
        self.assertEqual(carried[0]["arguments"], {"enabled": True})
        self.assertEqual(carried[0]["exec_result"], first["exec_result"])
        # KEV tool scoring stays keyed on the utterance: the writebacks go to decide(), not to KEV.
        self.assertTrue(all(COMPLETED_STEPS not in context for context in scored))

    def test_decide_brief_reads_completed_steps_from_context(self):
        record = writeback_result(utterance="打开空调",
                                  decision={"kind": "tool_call", "tool": "climate_set_power",
                                            "arguments": {"enabled": True}},
                                  routed_by="kev", decision_head="kev_finite_fast",
                                  turn=1, turn_ms=12.5)
        context = {"home_address_configured": True, "company_address_configured": True,
                   "synthetic": True, COMPLETED_STEPS: [record]}
        briefs = []
        os.environ.setdefault("SLOW_MODEL", "deepseek-v4.1-flash")

        def slow_turn(brief, messages, schemas, single_next_step=False):
            briefs.append(brief)
            return {"model": os.environ["SLOW_MODEL"], "content": [
                {"type": "tool_use", "name": "climate_set_power", "input": {"enabled": True}}]}

        with mock.patch.object(core_pipeline_run, "existing_slow_configuration", lambda: None), \
                mock.patch.object(core_pipeline_run, "slow_turn", slow_turn):
            outcome = core_pipeline_run.decide("刚才调了什么", ["climate_set_power"],
                                               [CLIMATE_SCHEMA], "navigation", context)
        self.assertEqual(outcome["decision"], {"kind": "tool_call", "tool": "climate_set_power",
                                               "arguments": {"enabled": True},
                                               "response_model": os.environ["SLOW_MODEL"]})
        [brief] = briefs
        self.assertEqual(len(brief["completed_steps"]), 1)
        step = brief["completed_steps"][0]
        self.assertEqual(step["tool"], "climate_set_power")
        self.assertEqual(step["arguments"], {"enabled": True})
        self.assertIn("mock 执行完成", step["exec_result"]["message"])
        self.assertEqual(step["exec_result"], record["exec_result"])
        self.assertNotIn(COMPLETED_STEPS, brief["context"])  # one copy of the facts, not two
        self.assertEqual(len(context[COMPLETED_STEPS]), 1)   # caller's context is not mutated

    def test_decide_without_writebacks_uses_an_empty_list(self):
        briefs = []
        os.environ.setdefault("SLOW_MODEL", "deepseek-v4.1-flash")

        def slow_turn(brief, messages, schemas, single_next_step=False):
            briefs.append(brief)
            return {"model": os.environ["SLOW_MODEL"], "content": [
                {"type": "tool_use", "name": "climate_set_power", "input": {"enabled": False}}]}

        with mock.patch.object(core_pipeline_run, "existing_slow_configuration", lambda: None), \
                mock.patch.object(core_pipeline_run, "slow_turn", slow_turn):
            core_pipeline_run.decide("打开空调", ["climate_set_power"], [CLIMATE_SCHEMA],
                                     "navigation", {"synthetic": True})
        self.assertEqual(briefs[0]["completed_steps"], [])


class NonToolCallWritebackTests(unittest.TestCase):
    def test_ask_user_and_done_are_skipped_not_dropped(self):
        for decision in ({"kind": "ask_user", "tool": "ask_user",
                          "arguments": {"question": "开还是关？"}},
                         {"kind": "done", "tool": "finish", "arguments": {"reason": "已完成"}}):
            events = []

            def decide(utterance, selected, schemas, tool_set="navigation", context=None,
                       _decision=decision, **kwargs):
                return {"decision": dict(_decision), "llm_ms": 1.0,
                        "log_fields": {"decision_head": "project_existing_litellm"}}

            session = make_session(events, tools_fn=light_tools, decide_fn=decide)
            began = time.monotonic() * 1000
            try:
                session.on_asr(0, "氛围灯", True)
                self.assertTrue(wait_turn(session, began))
            finally:
                session.close()
            [record] = writebacks(events)
            self.assertEqual(record["kind"], decision["kind"])
            self.assertIsNone(record["tool"])
            self.assertEqual(record["arguments"], {})
            self.assertEqual(record["status"], "skipped")
            self.assertTrue(record["exec_result"]["skipped"])
            self.assertEqual(record["exec_result"]["reason"], "kind=" + decision["kind"])
            self.assertEqual(len(session.context[COMPLETED_STEPS]), 1)


if __name__ == "__main__":
    unittest.main()