"""Offline protocol tests: actual registry/core, mocks only at adapter boundaries."""
import copy
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import model_clients as core
from registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


def envelope(kind="observe", **changes):
    data = {"kind": kind, "goal": {}, "actions": [], "continue_after_results": False,
            "question": None, "reason": None}
    if kind == "actions":
        data["actions"] = [{"tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}}]
    data.update(changes)
    return data


def react(*calls):
    return {"content": [{"type": "tool_use", "id": str(i), "name": name, "input": args}
                        for i, (name, args) in enumerate(calls)], "model": "offline-fixture"}


class ReoptProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
        cls.schemas = registry.schemas(frozenset(registry.tools))
        cls.contracts = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())["tools"]

    def parse(self, value):
        return core.parse_reopt_envelope(value, self.schemas, self.contracts)

    def test_four_legal_envelopes_and_defensive_copy(self):
        for value in (envelope(), envelope("actions"), envelope("clarify", question="去哪？"),
                      envelope("incomplete", reason="缺少目的地")):
            self.assertEqual(self.parse(value), value)
            self.assertIsNot(self.parse(value), value)
        self.assertTrue(self.parse(envelope("actions", continue_after_results=True))["continue_after_results"])

    def test_actual_probe_dry_run_zero_execution_network_and_credentials(self):
        from eval import slow_protocol_probe as probe
        stdout = io.StringIO()
        original_read = Path.read_text
        def deny_credentials(path, *args, **kwargs):
            self.assertNotEqual(path.name, ".credentials.yaml")
            return original_read(path, *args, **kwargs)
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(stdout), \
             patch.object(Path, "read_text", deny_credentials), \
             patch("socket.socket.connect", side_effect=AssertionError("No network")) as connect, \
             patch.object(probe.httpx2, "HTTPTransport", side_effect=AssertionError("Mock transport only")):
            self.assertEqual(probe.main(["--mode", "dry-run", "--no-execute"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual((result["envelope"]["kind"], result["executed_tools"], result["actual_network_requests"]),
                         ("actions", 0, 0))
        connect.assert_not_called()
        for config in ({"TYPESAFE_DEFAULT_MODEL": "wrong-model"}, {"TYPESAFE_BASE_URL": "https://wrong.invalid/v1"}):
            with patch.dict("os.environ", config, clear=True), self.assertRaises(core.ModelError):
                probe.configuration(credentials=True)

    def test_strict_six_keys_and_types(self):
        for key in envelope():
            value = envelope()
            del value[key]
            with self.subTest(key=key), self.assertRaises(core.ModelError):
                self.parse(value)
        for value in (envelope(extra=True), envelope(goal=[]), envelope(continue_after_results=1),
                      envelope(kind="finish"), envelope(goal={"x": float("nan")})):
            with self.subTest(value=value), self.assertRaises(core.ModelError):
                self.parse(value)
        from eval import slow_protocol_probe as probe
        _, schemas, contracts = probe.fixture()
        for text, finish in ((json.dumps(envelope("actions", question="bad")), "stop"),
                             ('{"kind":"observe","kind":"observe"}', "stop"),
                             (json.dumps(envelope()), "length")):
            raw = {"model": probe.MODEL, "choices": [{"finish_reason": finish,
                   "message": {"role": "assistant", "content": text}}]}
            with self.subTest(text=text, finish=finish), self.assertRaises(core.ModelError):
                probe.decode_response(raw, schemas, contracts, probe.MODEL)

    def test_kev_candidate_probe_matches_systemone_without_claiming_generation(self):
        from eval import slow_protocol_probe as probe
        brief, schemas, contracts = probe.fixture()
        body = probe.request_body(brief, schemas, {"model": "kev-latest", "endpoint": probe.BASE})
        self.assertEqual(body["model"], "kev-latest")
        self.assertEqual(body["max_output_tokens"], 512)
        self.assertIsInstance(body["state"], str)
        self.assertEqual(set(body["questions"]), {"protocol_kind"})
        self.assertEqual(body["questions"]["protocol_kind"]["type"], "choice")
        self.assertNotIn("messages", body)
        raw = {"answers": {"protocol_kind": {"choice": "actions", "confidence": 0.9,
               "probabilities": {"actions": 0.9, "clarify": 0.05, "incomplete": 0.03, "observe": 0.02}}}, "latency_ms": 17.3}
        result = probe.decode_response(raw, schemas, contracts, "kev-latest")
        self.assertEqual(result["envelope"]["kind"], "actions")
        self.assertFalse(result["generated_envelope"])
        self.assertIsNone(result["response_model"])
        self.assertEqual(result["executed_tools"], 0)

    def test_kev_choice_malformed_probabilities_ids_and_duplicate_keys_rejected(self):
        from eval import slow_protocol_probe as probe
        _, schemas, contracts = probe.fixture()
        raw = {"answers": {"protocol_kind": {"choice": "actions", "confidence": 0.9,
               "probabilities": {"actions": 0.9, "clarify": 0.05, "incomplete": 0.03, "observe": 0.02}}}}
        variants = []
        for field, value in (("choice", "missing"), ("confidence", True), ("confidence", float("nan")),
                             ("probabilities", {"actions": 0.9}),
                             ("probabilities", {"actions": 1.1, "clarify": 0, "incomplete": 0, "observe": 0})):
            bad = copy.deepcopy(raw);bad["answers"]["protocol_kind"][field] = value;variants.append(bad)
        variants.extend(({"answers": {}}, {"model": "wrong-model", **raw}, {"answers": {"extra": raw["answers"]["protocol_kind"]}}))
        for bad in variants:
            with self.subTest(shape=list(bad)), self.assertRaises(core.ModelError):
                probe.decode_response(bad, schemas, contracts, "kev-latest")
        with self.assertRaises(core.ModelError):
            json.loads('{"answers":{},"answers":{}}', object_pairs_hook=probe._unique)

    def test_illegal_semantic_combinations(self):
        for value in (envelope("actions", actions=[]), envelope("actions", question="去哪？"),
                      envelope("clarify", question="  "), envelope("clarify", question="去哪？", reason="缺项"),
                      envelope("incomplete"), envelope("incomplete", reason="缺项", question="去哪？"),
                      envelope(continue_after_results=True), envelope(reason="完成")):
            with self.subTest(value=value), self.assertRaises(core.ModelError):
                self.parse(value)

    def test_exact_tool_names_parameters_and_contracts(self):
        for action in ({"tool": "navigation.navi_to_special_poi", "arguments": {"poi_type": "home"}},
                       {"tool": "finish", "arguments": {}},
                       {"tool": "navi_to_special_poi", "arguments": {}},
                       {"tool": "navi_to_special_poi", "arguments": {"poi_type": "park"}},
                       {"tool": "navi_to_special_poi", "arguments": {"poi_type": "home", "extra": True}}):
            with self.subTest(action=action), self.assertRaises(core.ModelError):
                self.parse(envelope("actions", actions=[action]))
        bad = copy.deepcopy(self.contracts)
        bad[0]["parameters"]["required"] = []
        with self.assertRaises(core.ModelError):
            core.parse_reopt_envelope(envelope("actions"), self.schemas, bad)
        self.assertEqual(len(self.schemas), 14)

    def test_nested_future_references_rejected_mechanically(self):
        for arguments in ({"end_name": "${first.name}"},
                          {"end_name": "place", "via_points": [{"$ref": "first"}]},
                          {"end_name": "place", "via_points": [{"name": "${result}"}]}):
            with self.subTest(arguments=arguments), self.assertRaises(core.ModelError):
                self.parse(envelope("actions", actions=[{"tool": "navi_start_route", "arguments": arguments}]))
        self.parse(envelope("actions", actions=[{"tool": "navi_start_route", "arguments": {"end_name": "已提供地点"}}]))

    def test_react_control_mapping_and_business(self):
        for response, kind in ((react(("finish", {})), "observe"),
                               (react(("ask_user", {"question": "去哪？"})), "clarify"),
                               (react(("navi_to_special_poi", {"poi_type": "home"})), "actions")):
            self.assertEqual(core.normalize_reopt_react(response, self.schemas)["envelope"]["kind"], kind)

    def test_react_rejects_supplement_mix_duplicate_and_empty(self):
        duplicate = react(("navi_to_special_poi", {"poi_type": "home"}), ("navi_get_status", {}))
        duplicate["content"][1]["id"] = duplicate["content"][0]["id"]
        for response in (react(("request_tools", {"intent": "more"})),
                         react(("finish", {}), ("navi_to_special_poi", {"poi_type": "home"})),
                         react(("finish", {}), ("ask_user", {"question": "去哪？"})),
                         react(("finish", {"extra": True})), duplicate, react()):
            with self.subTest(response=response), self.assertRaises(core.ModelError):
                core.normalize_reopt_react(response, self.schemas)

    def test_opaque_metadata_is_standalone_and_survives_mock_next_round(self):
        response = react(("navi_to_special_poi", {"poi_type": "home"}))
        response["content"][0]["reasoning_content"] = "opaque-not-evidence"
        response["provider_metadata"] = {"opaque": [1, 2]}
        normalized = core.normalize_reopt_react(response, self.schemas)
        self.assertIs(normalized["_opaque"], response)
        self.assertNotIn("reasoning_content", normalized["envelope"])
        next_adapter = Mock(side_effect=core._slow_messages)
        messages = next_adapter([{"role": "assistant", "content": normalized["_opaque"]["content"]}])
        self.assertEqual(messages[0]["reasoning_content"], "opaque-not-evidence")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "navi_to_special_poi")

    def test_finish_observe_differential_all_four_outcomes(self):
        cases = [([], {}, "不满足"), ([{"status": "unknown"}], {}, "未知"),
                 ([{"status": "accepted"}], {"confirmation_remaining": 2}, "待确认"),
                 ([{"status": "effective", "verified": True}], {"goal_verified": True}, "已满足")]
        for facts, flags, expected in cases:
            with self.subTest(expected=expected):
                left = core.evaluate_reopt_exit("finish", facts, **flags)
                right = core.evaluate_reopt_exit(envelope(), facts, **flags)
                self.assertEqual(left, right)
                self.assertEqual(left["assessment"], expected)

    def test_confirmation_success_and_exhaustion_same_exit(self):
        for endpoint in ("finish", envelope()):
            confirm = Mock(return_value=[{"status": "effective", "verified": True}])
            result = core.evaluate_reopt_exit(endpoint, [{"status": "accepted"}], confirm=confirm,
                                              confirmation_remaining=3, goal_verified=True)
            self.assertEqual(result, {"assessment": "已满足", "status": "done", "query_count": 1})
            confirm.assert_called_once_with([{"status": "accepted"}])
            confirm = Mock(return_value=[{"status": "accepted"}])
            result = core.evaluate_reopt_exit(endpoint, [{"status": "accepted"}], confirm=confirm,
                                              confirmation_remaining=100, goal_verified=True)
            self.assertEqual(result, {"assessment": "未知", "status": "incomplete", "query_count": 3})
            self.assertEqual(confirm.call_count, 3)

    def test_unknown_and_unverified_echo_win_over_goal(self):
        for facts in ([{"status": "effective"}], [{"status": "effective", "verified": "true"}],
                      [{"status": "effective", "verified": True}, {"status": "unknown"}],
                      [{"status": "failed"}, {"status": "accepted"}]):
            self.assertEqual(core.check_reopt_outcome(facts, goal_verified=True)["assessment"], "未知")
        self.assertEqual(core.check_reopt_outcome([{"status": "effective", "verified": True}])["assessment"], "不满足")

    def test_termination_and_missing_user_confirmation(self):
        for flags in ({"cancelled": True}, {"deadline_expired": True}, {"confirmation_remaining": 0}):
            confirm = Mock()
            result = core.evaluate_reopt_exit("finish", [{"status": "accepted"}], confirm=confirm, **flags)
            self.assertEqual(result["assessment"], "未知")
            confirm.assert_not_called()
        result = core.check_reopt_outcome([], user_confirmation_missing=True, deadline_expired=True)
        self.assertEqual((result["assessment"], result["status"]), ("待确认", "incomplete"))
        for status in ("failed", "rejected"):
            self.assertEqual(core.check_reopt_outcome([{"status": status}])["assessment"], "不满足")

    def test_last_action_budget_uses_same_confirmation_without_model(self):
        for encoding, response in (("envelope", envelope("actions", continue_after_results=True)),
                                   ("react", react(("navi_to_special_poi", {"poi_type": "home"})))):
            dispatch = Mock(return_value={"status": "accepted"})
            confirm = Mock(return_value=[{"status": "effective", "verified": True}])
            result = core.run_reopt_round(response, encoding, self.schemas, [], dispatch=dispatch,
                                         confirm=confirm, last_round=True, goal_verified=True, confirmation_remaining=1)
            self.assertEqual((result["assessment"], result["query_count"], result["slow_continuation"]),
                             ("已满足", 1, False))
            dispatch.assert_called_once()
            confirm.assert_called_once()

    def test_finish_observe_missing_steps_never_resume_or_dispatch(self):
        for encoding, response in (("envelope", envelope()), ("react", react(("finish", {})))):
            dispatch, confirm = Mock(), Mock()
            result = core.run_reopt_round(response, encoding, self.schemas, [], dispatch=dispatch, confirm=confirm)
            self.assertEqual(result["assessment"], "不满足")
            self.assertFalse(result["slow_continuation"])
            dispatch.assert_not_called()
            confirm.assert_not_called()

    def test_fresh_state_noop_not_event_goal(self):
        dispatch = Mock(return_value={"status": "effective", "verified": True})
        result = core.run_reopt_round(envelope("actions"), "envelope", self.schemas, [], dispatch=dispatch,
                                     goal_verified=True, no_op_verified=True, state_goal=True)
        self.assertEqual(result["status"], "done")
        dispatch.assert_not_called()
        result = core.check_reopt_outcome([], goal_verified=True, no_op_verified=True, state_goal=True, event_goal=True)
        self.assertEqual(result["assessment"], "不满足")
        self.assertEqual(core.check_reopt_outcome([], goal_verified=True, state_goal=True)["assessment"], "不满足")

    def test_initial_done_never_dispatches_or_degrades_for_both_encodings(self):
        facts = [{"status": "effective", "verified": True}]
        initial = core.check_reopt_outcome(facts, goal_verified=True)
        self.assertEqual(initial["status"], "done")
        for encoding, response in (("envelope", envelope("actions", continue_after_results=True)),
                                   ("react", react(("navi_to_special_poi", {"poi_type": "home"})))):
            for returned in ({"status": "effective", "verified": True}, {"status": "accepted"}):
                with self.subTest(encoding=encoding, callback_result=returned["status"]):
                    dispatch, confirm = Mock(return_value=returned), Mock()
                    result = core.run_reopt_round(response, encoding, self.schemas, facts, dispatch=dispatch,
                                                 confirm=confirm, goal_verified=True, tool_contracts=self.contracts)
                    self.assertEqual({k: result[k] for k in initial}, initial)
                    self.assertFalse(result["slow_continuation"])
                    self.assertEqual(facts, [{"status": "effective", "verified": True}])
                    dispatch.assert_not_called()
                    confirm.assert_not_called()

    def test_kev_configuration_accepts_correct_public_environment_aliases(self):
        from eval import slow_protocol_probe as probe
        for values in ({}, {"SLOW_MODEL": "kev-latest", "LITELLM_BASE_URL": "http://127.0.0.1:8009/"},
                       {"TYPESAFE_DEFAULT_MODEL": "kev-latest", "TYPESAFE_BASE_URL": "http://127.0.0.1:8009/"},
                       {"SLOW_MODEL": "wrong", "LITELLM_BASE_URL": "https://wrong.invalid/v1",
                        "TYPESAFE_DEFAULT_MODEL": "kev-latest", "TYPESAFE_BASE_URL": "http://127.0.0.1:8009"}):
            with self.subTest(values=values), patch.dict("os.environ", values, clear=True):
                config = probe.configuration(credentials=True)
                self.assertEqual((config["model"], config["endpoint"]), ("kev-latest", "http://127.0.0.1:8009"))
                brief, schemas, _ = probe.fixture()
                body = probe.request_body(brief, schemas, config)
                self.assertIn("questions", body)
                self.assertNotIn("messages", body)
        for values in ({"SLOW_MODEL": "wrong"}, {"LITELLM_BASE_URL": "https://wrong.invalid/v1"}):
            with patch.dict("os.environ", values, clear=True), self.assertRaises(core.ModelError):
                probe.configuration()

    def test_token_and_protocol_rejection_before_any_callback(self):
        dispatch, confirm = Mock(), Mock()
        for response, token in ((envelope("actions"), False), (envelope("actions", question="bad"), True)):
            with self.assertRaises(core.ModelError):
                core.run_reopt_round(response, "envelope", self.schemas, [], token_valid=token,
                                     dispatch=dispatch, confirm=confirm)
        dispatch.assert_not_called()
        confirm.assert_not_called()

    def test_dispatch_stops_on_unresolved_fact_and_existing_unknown(self):
        response = envelope("actions", actions=[
            {"tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}},
            {"tool": "navi_get_status", "arguments": {}}])
        dispatch = Mock(return_value={"status": "unknown"})
        result = core.run_reopt_round(response, "envelope", self.schemas, [], dispatch=dispatch)
        self.assertEqual(result["assessment"], "未知")
        dispatch.assert_called_once()
        for facts, flags in (([{"status": "unknown"}], {}),
                             ([{"status": "accepted"}], {"confirmation_remaining": 1}),
                             ([], {"user_confirmation_missing": True})):
            dispatch.reset_mock()
            core.run_reopt_round(response, "envelope", self.schemas, facts, dispatch=dispatch, **flags)
            dispatch.assert_not_called()

    def test_failed_partial_actions_no_retry_and_one_actual_side_effect_bound(self):
        response = envelope("actions", continue_after_results=True)
        for status in ("failed", "rejected"):
            result = core.run_reopt_round(response, "envelope", self.schemas, [],
                                         dispatch=Mock(return_value={"status": status}))
            self.assertFalse(result["slow_continuation"])
            result = core.run_reopt_round(response, "envelope", self.schemas,
                                         [{"status": status}], dispatch=Mock(return_value={"status": "effective", "verified": True}))
            self.assertFalse(result["slow_continuation"])
        result = core.run_reopt_round(response, "envelope", self.schemas, [],
                                     dispatch=Mock(return_value={"status": "effective", "verified": True}))
        self.assertTrue(result["slow_continuation"])
        navigation = response["actions"][0]
        for second in ({"tool": "navi_open_address_page", "arguments": {"type": "home"}},
                       {"tool": "navi_search_keyword", "arguments": {"keyword": "place", "show_in_map": True}},
                       {"tool": "navi_search_keyword", "arguments": {"keyword": "place"}}):
            with self.subTest(second=second), self.assertRaises(core.ModelError):
                self.parse(envelope("actions", actions=[navigation, second]))
        self.parse(envelope("actions", actions=[navigation,
                   {"tool": "navi_search_keyword", "arguments": {"keyword": "place", "show_in_map": False}},
                   {"tool": "navi_get_status", "arguments": {}}]))

    def test_in_memory_token_and_echo_negative_controls_are_detected(self):
        # Execute actual source variants in memory; never alter checkout or mock the core to gold.
        token_source = inspect.getsource(core.run_reopt_round)
        token_source = token_source.replace('if token_valid is not True:', 'if False:', 1)
        namespace = vars(core).copy()
        exec(compile(token_source, "<token-negative-control>", "exec"), namespace)
        dispatch = Mock(return_value={"status": "effective", "verified": True})
        def token_assertion(runner):
            with self.assertRaises(core.ModelError):
                runner(envelope("actions"), "envelope", self.schemas, [], token_valid=False, dispatch=dispatch)
        token_assertion(core.run_reopt_round)
        with self.assertRaises(AssertionError):
            token_assertion(namespace["run_reopt_round"])
        dispatch.assert_called_once()
        source = inspect.getsource(core.check_reopt_outcome)
        source = source.replace('f.get("verified") is not True', 'False', 1)
        exec(compile(source, "<echo-negative-control>", "exec"), namespace)
        def echo_assertion(check):
            self.assertEqual(check([{"status": "effective"}], goal_verified=True)["assessment"], "未知")
        echo_assertion(core.check_reopt_outcome)
        with self.assertRaises(AssertionError):
            echo_assertion(namespace["check_reopt_outcome"])
        self.assertEqual(len(hashlib.sha256(inspect.getsource(core.run_reopt_round).encode()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
