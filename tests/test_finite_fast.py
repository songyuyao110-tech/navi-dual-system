import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

from finite_fast_path import AllowlistError, is_finite_tool, load_allowlist, try_finite_fast_decide
from turn_session import TurnConfig, TurnSession

ALLOWLIST = load_allowlist()
CLIMATE_SCHEMA = {"name": "climate_set_power", "description": "开启或关闭座舱空调系统",
                  "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}},
                                 "required": ["enabled"], "additionalProperties": False}}
NAVI_SCHEMA = {"name": "navi_to_special_poi", "description": "导航到家庭或公司地址",
               "parameters": {"type": "object",
                              "properties": {"poi_type": {"type": "string", "enum": ["home", "company"]}},
                              "required": ["poi_type"], "additionalProperties": False}}


def kev(*choices):
    """mock kev_call(state, questions): one (choice, confidence) pair per asked parameter."""
    pairs = list(choices)

    def call(state, questions):
        answers = {}
        for (choice, confidence) in pairs:
            for name in questions:
                answers[name] = {"type": "choice", "choice": choice, "confidence": confidence,
                                 "probabilities": {value: (confidence if value == choice else (1 - confidence) / 2)
                                                   for value in questions[name]["criteria"]}}
        return {"model": "kev-latest", "answers": answers, "latency_ms": 1.0}
    return call


def execute(tool, score=0.95, selected=None):
    return {"scores": {"tool:" + tool: score}, "selected": selected or [tool],
            "schemas": [CLIMATE_SCHEMA if tool.startswith("climate") else NAVI_SCHEMA],
            "timing": {"stage1_ms": 1.0, "stage2_ms": 1.0, "total_ms": 2.0}}


class AllowlistTests(unittest.TestCase):
    def test_allowlist_is_the_fifty_eight_eligible_tools(self):
        # reopt 1a -> speculative-toolselect: 30 -> 58, i.e. every pure enum/boolean tool in
        # tools-multidomain.yaml. The order is the YAML order, so a missing or misspelt tool shows up.
        self.assertEqual(ALLOWLIST["order"], [
            "navi_to_special_poi", "navi_open_address_page", "climate_set_power", "climate_set_ac",
            "climate_set_auto", "music_skip", "music_set_mode", "phone_set_mic_mute",
            "light_set_ambient_power", "body_set_tailgate",
            "light_set_reading", "climate_set_front_defrost", "climate_set_sync", "seat_stop_massage",
            "navi_set_map_view", "navi_set_voice_guidance", "vehicle_get_trip", "energy_set_port",
            "vehicle_get_energy_usage", "video_resume", "video_set_subtitles",
            "message_set_notifications", "drive_set_mode", "drive_set_steering_weight",
            "system_set_do_not_disturb", "light_set_welcome", "light_set_high_beam_assist",
            "body_set_child_lock", "system_set_mute", "body_set_wipers",
            "climate_set_airflow", "climate_set_recirculation", "climate_set_rear_defrost",
            "climate_set_purification", "body_set_door_lock", "body_set_mirror_fold",
            "body_set_mirror_heat", "seat_get_status", "light_set_headlights", "light_set_fog",
            "light_set_ambient_color", "light_set_ambient_effect", "audio_set_playback",
            "video_pause", "video_set_quality", "video_get_status", "message_list_unread",
            "info_get_news", "drive_set_regeneration", "drive_set_lane_assist",
            "drive_set_auto_hold", "reminder_list", "system_set_screen_power", "system_set_theme",
            "system_set_language", "system_set_units", "system_set_bluetooth", "system_set_wifi"])
        self.assertEqual(ALLOWLIST["defaults"], {"tool_tau": 0.70, "param_tau": 0.70, "require_top1": True})

    def test_every_eligible_tool_is_listed_and_no_open_slot_sneaks_in(self):
        # The list is a full cover of the 1a rule over tools-multidomain.yaml, computed from the
        # fixture (not from a literal): 58 eligible, 29 with no parameters at all stay out.
        import yaml
        from core_pipeline_run import MULTIDOMAIN_TOOLS
        tools = yaml.safe_load(MULTIDOMAIN_TOOLS.read_text())["tools"]
        eligible = []
        for tool in tools:
            properties = (tool.get("parameters") or {}).get("properties") or {}
            if not properties:
                continue
            if all((item.get("type") == "boolean")
                   or (item.get("type") == "string" and item.get("enum"))
                   for item in properties.values()):
                eligible.append(tool["name"])
        self.assertEqual(len(eligible), 58)
        self.assertEqual(sorted(eligible), sorted(ALLOWLIST["order"]))

    def test_membership_is_declared_not_inferred(self):
        self.assertTrue(is_finite_tool("music_skip"))
        self.assertTrue(is_finite_tool("light_set_ambient_color"))   # now eligible, so listed
        # Open numeric slot: absent from the allowlist and ineligible by the 1a rule.
        self.assertFalse(is_finite_tool("climate_set_temperature"))
        self.assertFalse(is_finite_tool("navi_get_status"))     # no-parameter tools are not in scope


class FiniteFastHitTests(unittest.TestCase):
    def test_enum_parameter_hits_with_kev_head(self):
        record = {}
        result = try_finite_fast_decide("导航回家", ["navi_to_special_poi"], {"tool:navi_to_special_poi": 0.96},
                                        [NAVI_SCHEMA], TurnConfig(), kev_call_fn=kev(("home", 0.95)),
                                        record=record)
        self.assertIsNotNone(result)
        self.assertEqual(result["decision"], {"kind": "tool_call", "tool": "navi_to_special_poi",
                                              "arguments": {"poi_type": "home"}})
        self.assertEqual(result["log_fields"]["decision_head"], "kev_finite_fast")
        self.assertEqual(result["log_fields"]["routed_by"], "kev")
        self.assertIs(result["log_fields"]["finite_fast"], True)
        self.assertEqual(result["log_fields"]["param_confidences"], {"poi_type": 0.95})
        self.assertIsInstance(result["llm_ms"], float)
        self.assertEqual(record["finite_fast_reason"], "hit")

    def test_boolean_choice_maps_back_to_json_boolean(self):
        result = try_finite_fast_decide("打开空调", ["climate_set_power"], {"tool:climate_set_power": 0.94},
                                        [CLIMATE_SCHEMA], TurnConfig(), kev_call_fn=kev(("true", 0.93)))
        self.assertEqual(result["decision"]["arguments"], {"enabled": True})
        self.assertIs(result["decision"]["arguments"]["enabled"], True)
        json.dumps(result["decision"], allow_nan=False)   # must stay JSON-serialisable

    def test_second_enum_value_and_direct_attributes(self):
        cfg = TurnConfig(finite_fast_tool_tau=0.9, finite_fast_param_tau=0.5)
        result = try_finite_fast_decide("带我回公司", ["navi_to_special_poi"],
                                        {"tool:navi_to_special_poi": 0.92}, [NAVI_SCHEMA], cfg,
                                        kev_call_fn=kev(("company", 0.61)))
        self.assertEqual(result["decision"]["arguments"], {"poi_type": "company"})


class FiniteFastMissTests(unittest.TestCase):
    def call(self, utterance="打开空调", tool="climate_set_power", score=0.95, cfg=None,
             schema=CLIMATE_SCHEMA, kev_call_fn=None, record=None):
        record = {} if record is None else record
        result = try_finite_fast_decide(utterance, [tool], {"tool:" + tool: score}, [schema],
                                        cfg or TurnConfig(), kev_call_fn=kev_call_fn or kev(("true", 0.95)),
                                        record=record)
        return result, record

    def test_disabled_switch_forces_llm_without_calling_kev(self):
        calls = []
        result, record = self.call(cfg=TurnConfig(finite_fast_path=False),
                                   kev_call_fn=lambda state, questions: calls.append(1))
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "disabled")
        self.assertEqual(calls, [])

    def test_low_tool_score_misses(self):
        result, record = self.call(score=0.55)
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "tool_score_low")
        self.assertEqual(record["finite_fast_tool_score"], 0.55)

    def test_not_allowlisted_miss_logs_the_top1_tool_and_its_score(self):
        result, record = self.call(tool="climate_set_temperature", score=0.91, schema={
            "name": "climate_set_temperature",
            "parameters": {"type": "object", "properties": {
                "zone": {"type": "string", "enum": ["driver", "all"]},
                "celsius": {"type": "number"}}}})
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "not_allowlisted")
        # Part B: a "not in the allowlist" MISS must name the tool and its score, never null.
        self.assertEqual(record["finite_fast_tool"], "climate_set_temperature")
        self.assertIsInstance(record["finite_fast_tool_score"], float)
        self.assertEqual(record["finite_fast_tool_score"], 0.91)

    def test_low_parameter_confidence_misses_without_partial_arguments(self):
        result, record = self.call(kev_call_fn=kev(("true", 0.42)))
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "param_low_confidence")

    def test_open_slot_schema_misses(self):
        result, record = self.call(schema={"name": "climate_set_power", "parameters": {
            "type": "object", "properties": {"enabled": {"type": "boolean"}, "zone": {"type": "string"}},
            "required": ["enabled"]}})
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "schema_open_slot")

    def test_undeclared_finite_parameter_misses(self):
        # Both properties are booleans, but the allowlist only declares `enabled`: filling half the
        # arguments would be wrong, so the whole turn degrades.
        result, record = self.call(schema={"name": "climate_set_power", "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}, "sync": {"type": "boolean"}},
            "required": ["enabled"]}})
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "schema_undeclared_parameter")

    def test_kev_failure_degrades_and_records_error(self):
        def boom(state, questions):
            raise ValueError("KEV returned non-200")
        result, record = self.call(kev_call_fn=boom)
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "kev_error")
        self.assertIn("KEV returned non-200", record["finite_fast_error"])

    def test_illegal_choice_value_is_rejected(self):
        result, record = self.call(kev_call_fn=kev(("on", 0.99)))
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "kev_error")
        self.assertIn("schema-legal", record["finite_fast_error"])

    def test_missing_answers_are_rejected(self):
        result, record = self.call(kev_call_fn=lambda state, questions: {"answers": {}})
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "kev_error")

    def test_schema_enum_drift_misses(self):
        result, record = self.call(tool="navi_to_special_poi", schema={
            "name": "navi_to_special_poi", "parameters": {
                "type": "object", "properties": {"poi_type": {"type": "string", "enum": ["home"]}},
                "required": ["poi_type"]}})
        self.assertIsNone(result)
        self.assertEqual(record["finite_fast_reason"], "schema_enum_mismatch")


def drive(cfg, utterance="打开空调", *, selected=("climate_set_power",), score=0.95, schema=CLIMATE_SCHEMA):
    """Run one scripted turn through TurnSession with mocked EOT/tool selection."""
    events = []
    calls = {"decide": 0}

    def eot(text):
        return {"complete": 0.95, "cut": 0.03, "wait": 0.02, "label": "complete", "margin": 0.92,
                "wall_ms": 1.0, "kev_ms": 1.0}

    def tools(text, tool_set, context):
        return {"scores": {"tool:" + name: score for name in selected}, "selected": list(selected),
                "schemas": [schema], "timing": {"stage1_ms": 1.0, "stage2_ms": 1.0, "total_ms": 2.0}}

    def decide(utterance, selected_tools, schemas, tool_set="navigation", context=None, **kwargs):
        calls["decide"] += 1
        return {"decision": {"kind": "tool_call", "tool": "climate_set_power", "arguments": {"enabled": True}},
                "llm_ms": 1.0, "log_fields": {"decision_head": "project_existing_litellm",
                                              "decision": {"kind": "tool_call"}}}

    session = TurnSession(cfg, events.append, eot_fn=eot, select_tools_fn=tools, decide_fn=decide,
                          evidence_root=None, session_id="t")
    try:
        began = time.monotonic() * 1000
        session.on_asr(0, utterance, True)
        deadline = time.monotonic() * 1000 + 5000
        while not session.turn_done and time.monotonic() * 1000 < deadline:
            session.tick(time.monotonic() * 1000 - began)
            time.sleep(0.002)
    finally:
        session.close()
    return session, events, calls


class TurnSessionFastPathTests(unittest.TestCase):
    def test_hit_skips_the_decision_model(self):
        session, events, calls = drive(TurnConfig(speculative_llm=False, debounce_ms=0))
        self.assertTrue(session.turn_done)
        self.assertEqual(session.counters["finite_fast_hits"], 1)
        self.assertEqual(session.counters["llm_calls"], 0)
        self.assertEqual(calls["decide"], 0)                 # no decide() -> no slow_turn HTTP
        self.assertEqual(session.routed_by, "kev")
        decision = [event for event in events if event["event"] == "llm_decision"][-1]
        self.assertEqual(decision["decision_head"], "kev_finite_fast")
        self.assertEqual(decision["routed_by"], "kev")
        self.assertTrue(decision["finite_fast"])
        self.assertEqual(decision["decision"]["arguments"], {"enabled": True})
        self.assertTrue(any(event["event"] == "finite_fast_hit" for event in events))

    def test_miss_falls_back_to_the_decision_model(self):
        # A tool outside the allowlist always misses, so the original decide() must run.
        session, events, calls = drive(TurnConfig(speculative_llm=False, debounce_ms=0),
                                       selected=("climate_set_temperature",), score=0.93,
                                       schema={"name": "climate_set_temperature", "parameters": {
                                           "type": "object", "properties": {
                                               "zone": {"type": "string", "enum": ["driver", "all"]},
                                               "celsius": {"type": "number"}}}})
        self.assertTrue(session.turn_done)
        self.assertEqual(session.counters["finite_fast_hits"], 0)
        self.assertEqual(session.counters["llm_calls"], 1)
        self.assertEqual(calls["decide"], 1)
        self.assertEqual(session.routed_by, "llm")
        decision = [event for event in events if event["event"] == "llm_decision"][-1]
        self.assertEqual(decision["decision_head"], "project_existing_litellm")
        self.assertFalse(decision["finite_fast"])
        miss = [event for event in events if event["event"] == "finite_fast_miss"][-1]
        self.assertEqual(miss["reason"], "not_allowlisted")

    def test_disabled_switch_never_calls_the_gate(self):
        def explode(*args, **kwargs):
            raise AssertionError("finite fast gate must not run when the switch is off")
        cfg = TurnConfig(speculative_llm=False, debounce_ms=0, finite_fast_path=False)
        session = None
        events = []
        session = TurnSession(cfg, events.append,
                              eot_fn=lambda text: {"complete": 0.95, "cut": 0.03, "wait": 0.02,
                                                   "label": "complete", "margin": 0.92,
                                                   "wall_ms": 1.0, "kev_ms": 1.0},
                              select_tools_fn=lambda text, tool_set, context: execute("climate_set_power"),
                              decide_fn=lambda *args, **kwargs: {
                                  "decision": {"kind": "tool_call", "tool": "climate_set_power",
                                               "arguments": {"enabled": True}}, "llm_ms": 1.0,
                                  "log_fields": {"decision_head": "project_existing_litellm"}},
                              finite_fast_fn=explode, evidence_root=None, session_id="t2")
        try:
            began = time.monotonic() * 1000
            session.on_asr(0, "打开空调", True)
            deadline = time.monotonic() * 1000 + 5000
            while not session.turn_done and time.monotonic() * 1000 < deadline:
                session.tick(time.monotonic() * 1000 - began)
                time.sleep(0.002)
        finally:
            session.close()
        self.assertTrue(session.turn_done)
        self.assertEqual(session.counters["finite_fast_hits"], 0)
        self.assertEqual(session.counters["llm_calls"], 1)
        self.assertEqual(session.routed_by, "llm")
        self.assertFalse(any(event["event"].startswith("finite_fast") for event in events))


if __name__ == "__main__":
    unittest.main()