import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx2
from typesafe_sdk import Noul
from typesafe_sdk.constants import API_KEY_ENV

from eval import jev_protocol_probe as probe
from eval.reopt_probe_support import Budget, GuardedTransport, Limits, Rejected, approval_digest, checkout_identity, payload_hash, sha256

MANIFEST = probe.ROOT / "eval/fixtures/reopt-manifest.json"


class JevProbeTests(unittest.TestCase):
    def test_full_fourteen_tools_and_two_synthetic_memories(self):
        state, questions = probe.build_request(MANIFEST)
        self.assertEqual(len(questions), 16)
        self.assertEqual(sum(name.startswith("tool:") for name in questions), 14)
        self.assertEqual({name for name in questions if name.startswith("memory:")}, {"memory:home", "memory:company"})
        self.assertTrue(all(isinstance(question, Noul) for question in questions.values()))
        self.assertIn("Synthetic", state)

    def call_response(self, response_factory, *, tunnel=False):
        with tempfile.TemporaryDirectory(prefix="reopt-jev-", dir="/Users/YUYSONG/Work/tmp") as directory:
            state = "synthetic protocol fixture"
            questions = {"probe": Noul(instructions="synthetic")}
            data = {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}}
            binding = {"code_sha": "b" * 40, "source_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                       "data_sha256": payload_hash(data),
                       "endpoint": "http://127.0.0.1:8009" if tunnel else "https://synthetic.example.invalid",
                       "model": "kev-latest" if tunnel else "jev-test",
                       "batch_id": "test-batch", "session": "synthetic", "execution_scope_hash": "e" * 64,
                       "acceptance_scope_hash": "f" * 64}
            limits = Limits(input_usd_per_million=1.0, output_usd_per_million=2.0, provider_caps_verified=True, output_cap_requested=True)
            human_sha = approval_digest(binding, limits, 4070908800.0)
            approval = {**binding, "limits": asdict(limits), "expires_at": 4070908800.0, "human_approval_sha256": human_sha}
            budget = Budget(Path(directory), "test-batch", binding, limits, approval, human_sha)
            requests = []
            def handler(request):
                requests.append(request)
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.url.path, "/v1/systemone")
                self.assertEqual(json.loads(request.content)["max_output_tokens"], 512)
                if tunnel:
                    self.assertNotIn("authorization", request.headers)
                    self.assertEqual(json.loads(request.content)["model"], "kev-latest")
                return response_factory()
            transport = GuardedTransport(httpx2.MockTransport(handler), budget, {**data, "model": binding["model"], "max_output_tokens": 512}, binding["endpoint"])
            original_get = os.environ.get
            def public_get(key, default=None):
                if key == API_KEY_ENV:
                    raise AssertionError("tunnel must not read an API key")
                return original_get(key, default)
            with patch.dict(os.environ, {API_KEY_ENV: "synthetic-" + "unit-only"}, clear=True), (
                    patch.object(probe, "AsyncTypeSafeClient", side_effect=AssertionError("no tunnel SDK"))
                    if tunnel else nullcontext()), (
                    patch.object(os.environ, "get", side_effect=public_get) if tunnel else nullcontext()):
                try:
                    result = asyncio.run(probe.sdk_probe(state, questions, binding, transport))
                except Rejected:
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(budget.requests, 1)  # Failed/unknown never reset the ledger.
                    raise
            self.assertEqual(len(requests), 1)
            self.assertEqual(budget.requests, 1)
            return result

    @staticmethod
    def response(answers=None, model="jev-test", usage=None):
        return httpx2.Response(200, json={"model": model, "usage": usage or {},
                                         "answers": {"probe": {"type": "noul", "noul": 0.9}} if answers is None else answers})

    def test_real_sdk_one_request_and_unknown_usage_is_null(self):
        result = self.call_response(self.response)
        self.assertEqual(result["probabilities"], {"probe": 0.9})
        self.assertIsNone(result["usage"])
        self.assertEqual(result["attempted_dispatch"], 1)
        self.assertEqual(result["actual_socket_network_calls"], 0)
        self.assertEqual(result["attempted_network"], 0)

    def test_missing_extra_wrong_type_and_invalid_probability_rejected(self):
        variants = ({}, {"extra": {"type": "noul", "noul": 0.9}},
                    {"probe": {"type": "choice", "choice": "x", "confidence": 0.9, "probabilities": {"x": 0.9}}},
                    {"probe": {"type": "noul", "noul": -0.1}}, {"probe": {"type": "noul", "noul": 1.1}})
        for answers in variants:
            with self.subTest(shape=len(answers)), self.assertRaises(Rejected):
                self.call_response(lambda: self.response(answers))

    def test_unknown_extra_raw_answer_is_rejected(self):
        answers = {"probe": {"type": "noul", "noul": 0.9},
                   "extra": {"type": "future-synthetic", "payload": "synthetic-private-body"}}
        with self.assertRaises(Rejected) as caught:
            self.call_response(lambda: self.response(answers))  # Real SDK discards extra, raw bytes must still reject it.
        self.assertTrue("synthetic-private-body" not in str(caught.exception), "redaction failed")

    def test_duplicate_top_level_and_answer_keys_rejected(self):
        variants = (
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"future-duplicate"},"probe":{"type":"noul","noul":0.9}}}',
            b'{"model":"jev-test","usage":{},"answers":{"extra":{"type":"future"}},"answers":{"probe":{"type":"noul","noul":0.9}}}',
            b'{"model":"synthetic-private-model","model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","noul":0.9}}}',
            b'{"model":"jev-test","usage":{"input_tokens":999999},"usage":{},"answers":{"probe":{"type":"noul","noul":0.9}}}',
        )
        for raw in variants:
            with self.subTest(size=len(raw)), self.assertRaises(Rejected) as caught:
                self.call_response(lambda: httpx2.Response(200, content=raw))
            self.assertTrue("synthetic-private-model" not in str(caught.exception), "redaction failed")

    def test_duplicate_probability_usage_and_type_keys_rejected(self):
        variants = (
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","noul":-0.5,"noul":0.9}}}',
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","noul":0.9,"noul":-0.5}}}',
            b'{"model":"jev-test","usage":{"input_tokens":999999,"input_tokens":1},"answers":{"probe":{"type":"noul","noul":0.9}}}',
            b'{"model":"jev-test","usage":{"output_tokens":999999,"output_tokens":1},"answers":{"probe":{"type":"noul","noul":0.9}}}',
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"future","type":"noul","noul":0.9}}}',
        )
        for raw in variants:
            with self.subTest(size=len(raw)), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, content=raw))

    def test_identical_escaped_and_nested_duplicate_keys_rejected(self):
        variants = (
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","noul":0.9,"noul":0.9}}}',
            b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","n\\u006ful":-0.5,"noul":0.9}}}',
            b'{"model":"jev-test","usage":{},"metadata":{"value":1,"value":2},"answers":{"probe":{"type":"noul","noul":0.9}}}',
        )
        for raw in variants:
            with self.subTest(size=len(raw)), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, content=raw))

    def test_nonfinite_probability_rejected(self):
        raw = b'{"model":"jev-test","usage":{},"answers":{"probe":{"type":"noul","noul":NaN}}}'
        with self.assertRaises(Rejected):
            self.call_response(lambda: httpx2.Response(200, content=raw))

    def test_reported_usage_cannot_exceed_reserved_caps(self):
        for usage in ({"input_tokens": 8193}, {"output_tokens": 513}, {"input_tokens": -1}):
            with self.subTest(field=next(iter(usage))), self.assertRaises(Rejected):
                self.call_response(lambda: self.response(usage=usage))

    def test_tunnel_documented_response_omits_model_usage_and_type(self):
        # Deployment revision 8: these fields are omitted, not fabricated.
        result = self.call_response(lambda: httpx2.Response(200, json={
            "answers": {"probe": {"noul": 0.92}}, "latency_ms": 17.3}), tunnel=True)
        self.assertEqual(result["model"], "kev-latest")
        self.assertIsNone(result["response_model"])
        self.assertEqual(result["model_source"], "request")
        self.assertIsNone(result["usage"])
        self.assertEqual(result["latency_ms"], 17.3)
        self.assertEqual(result["probabilities"], {"probe": 0.92})
        self.assertEqual(result["actual_socket_network_calls"], 0)
        self.assertEqual(result["attempted_dispatch"], 1)

    def test_tunnel_explicit_noul_type_model_usage_and_zero_latency(self):
        result = self.call_response(lambda: httpx2.Response(200, json={
            "model": "kev-latest", "usage": {"input_tokens": 1, "output_tokens": 0},
            "answers": {"probe": {"type": "noul", "noul": 0}}, "latency_ms": 0}), tunnel=True)
        self.assertEqual(result["response_model"], "kev-latest")
        self.assertEqual(result["model_source"], "response")
        self.assertEqual(result["usage"], {"input_tokens": 1, "output_tokens": 0})
        self.assertEqual(result["latency_ms"], 0)

    def test_tunnel_missing_extra_ids_wrong_type_and_extra_probability_rejected(self):
        variants = ({}, {"extra": {"noul": 0.9}},
                    {"probe": {"noul": 0.9}, "extra": {"noul": 0.8}},
                    {"probe": {"type": "choice", "noul": 0.9}},
                    {"probe": {"type": None, "noul": 0.9}},
                    {"probe": {"noul": 0.9, "probabilities": {"x": 0.9}}},
                    {"probe": {"noul": 0.9, "choice": "x"}}, {"probe": {}})
        for answers in variants:
            with self.subTest(answers=answers), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, json={"answers": answers}), tunnel=True)

    def test_tunnel_strict_probability_and_latency_rejected(self):
        for value in (True, False, "0.9", None, -0.1, 1.1):
            with self.subTest(probability=value), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, json={
                    "answers": {"probe": {"noul": value}}}), tunnel=True)
        for value in (True, False, "17.3", None, -1):
            with self.subTest(latency=value), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, json={
                    "answers": {"probe": {"noul": 0.9}}, "latency_ms": value}), tunnel=True)
        for field, value in (("noul", b"NaN"), ("noul", b"Infinity"), ("noul", b"-Infinity"),
                             ("latency_ms", b"NaN"), ("latency_ms", b"Infinity")):
            raw = (b'{"answers":{"probe":{"noul":' + value + b'}}}' if field == "noul" else
                   b'{"answers":{"probe":{"noul":0.9}},"latency_ms":' + value + b'}')
            with self.subTest(field=field, value=value), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, content=raw), tunnel=True)

    def test_tunnel_duplicate_keys_and_response_model_mismatch_rejected(self):
        variants = (
            b'{"answers":{"probe":{"noul":0.9,"noul":0.92}}}',
            b'{"answers":{"probe":{"noul":0.9,"n\\u006ful":0.92}}}',
            b'{"answers":{"probe":{"noul":0.9},"probe":{"noul":0.92}}}',
            b'{"answers":{},"answers":{"probe":{"noul":0.92}}}',
            b'{"answers":{"probe":{"noul":0.92}},"latency_ms":1,"latency_ms":2}',
            b'{"answers":{"probe":{"noul":0.92}},"model":"jev-test"}',
            b'{"answers":{"probe":{"noul":0.92}},"model":null}',
        )
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, content=raw), tunnel=True)

    def test_tunnel_wrong_response_format_rejected(self):
        for body in ([], "wrong format", {}, {"choices": []}, {"answers": []}):
            with self.subTest(body=body), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, json=body), tunnel=True)
        with self.assertRaises(Rejected):
            self.call_response(lambda: httpx2.Response(200, content=b"not json"), tunnel=True)

    def test_tunnel_invalid_usage_rejected(self):
        for usage in ([], "unknown", {"input_tokens": 8193}, {"output_tokens": 513},
                      {"input_tokens": -1}, {"output_tokens": True}, {"input_tokens": "1"}):
            with self.subTest(usage=usage), self.assertRaises(Rejected):
                self.call_response(lambda: httpx2.Response(200, json={
                    "answers": {"probe": {"noul": 0.92}}, "usage": usage}), tunnel=True)

    def test_tunnel_invalid_status_rejected_before_json_decode(self):
        with patch.object(probe.json, "loads", wraps=json.loads) as decoder, self.assertRaises(Rejected):
            self.call_response(lambda: httpx2.Response(500, content=b"not json"), tunnel=True)
        self.assertTrue(all(call.args[0] != b"not json" for call in decoder.call_args_list))

    def test_existing_output_rejected_before_mock_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="reopt-output-", dir=str(probe.EVIDENCE)) as directory:
            path = Path(directory) / "result.json"
            path.write_text("original\n")
            with patch.object(probe, "dry_run", side_effect=AssertionError("unexpected dispatch")) as dispatch, patch("sys.stderr", io.StringIO()):
                self.assertEqual(probe.main(["--mode", "dry-run", "--output", str(path)]), 1)
                dispatch.assert_not_called()
            self.assertEqual(path.read_text(), "original\n")

    def test_wrong_response_model_and_body_never_leak(self):
        marker = "synthetic-" + "private-provider-body"
        with self.assertRaises(Rejected) as caught:
            self.call_response(lambda: self.response(model=marker))
        self.assertTrue(marker not in str(caught.exception), "redaction failed")
        with self.assertRaises(Rejected) as caught:
            self.call_response(lambda: httpx2.Response(500, content=marker.encode()))
        self.assertTrue(marker not in str(caught.exception), "redaction failed")

    def test_dry_run_has_zero_network_and_no_live_certification(self):
        result = probe.dry_run(MANIFEST)
        self.assertEqual(result["mode"], "dry-run")
        self.assertTrue(result["simulation_only"])
        self.assertFalse(result["live_certified"])
        self.assertEqual(result["cloud_requests"], 0)
        self.assertEqual(result["actual_socket_network_calls"], 0)
        self.assertEqual(result["attempted_network"], 0)
        self.assertEqual(len(result["question_ids"]), 16)
        self.assertEqual(result["retry"], 0)
        self.assertIsNone(result["usage"])

    def test_test_role_and_run_identity_bind_original_journal(self):
        state, questions = probe.build_request(MANIFEST)
        data = {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}}
        expected = checkout_identity(MANIFEST, data)
        with tempfile.TemporaryDirectory(prefix="reopt-role-", dir="/Users/YUYSONG/Work/tmp") as directory, patch.object(probe, "EVIDENCE", Path(directory)):
            run = Path(directory) / "step-2" / expected["code_sha"] / "test" / "independent-run"
            with patch.object(probe, "ROOT", probe.ROOT.parent / "test-synthetic"):
                result = probe.dry_run(MANIFEST, "synthetic-test-batch", run_dir=run)
            self.assertEqual(result["role"], "test")
            self.assertEqual(result["session"], "test-dry-run")
            self.assertEqual(result["run_id"], "independent-run")
            self.assertEqual(result["batch_id"], "synthetic-test-batch")
            self.assertEqual(Path(result["run_dir"]), run)
            self.assertFalse((Path(directory) / "step-2" / expected["code_sha"] / "dev").exists())
            journal = run / result["journal_path"]
            self.assertTrue(journal.resolve().is_relative_to(run))
            self.assertEqual(result["journal_sha256"], sha256(journal.read_bytes()))
            expected.update(batch_id=result["batch_id"], session=result["session"],
                            execution_scope_hash=result["execution_scope_hash"], acceptance_scope_hash=result["acceptance_scope_hash"])
            self.assertEqual(json.loads(journal.read_text().splitlines()[0])["binding_sha256"], payload_hash(expected))

    def test_cli_test_role_writes_only_its_new_run(self):
        with tempfile.TemporaryDirectory(prefix="reopt-cli-role-", dir="/Users/YUYSONG/Work/tmp") as directory, patch.object(probe, "EVIDENCE", Path(directory)):
            state, questions = probe.build_request(MANIFEST)
            identity = checkout_identity(MANIFEST, {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}})
            run = Path(directory) / "step-2" / identity["code_sha"] / "test" / "cli-run"
            output = run / "result.json"
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                status = probe.main(["--mode", "dry-run", "--role", "test", "--run-dir", str(run), "--output", str(output)])
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["role"], "test")
            self.assertEqual(result["session"], "test-dry-run")
            self.assertTrue((run / result["journal_path"]).is_file())

    def test_foreign_role_or_approvals_output_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="reopt-scope-", dir="/Users/YUYSONG/Work/tmp") as directory, patch.object(probe, "EVIDENCE", Path(directory)):
            state, questions = probe.build_request(MANIFEST)
            identity = checkout_identity(MANIFEST, {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}})
            base = Path(directory) / "step-2" / identity["code_sha"]
            for output in (base / "dev" / "foreign" / "result.json", Path(directory) / "approvals" / "result.json"):
                with patch.object(probe, "sdk_probe", side_effect=AssertionError("unexpected dispatch")) as dispatch, patch("sys.stderr", io.StringIO()):
                    status = probe.main(["--mode", "dry-run", "--role", "test", "--run-dir", str(base / "test" / "run"), "--output", str(output)])
                    self.assertEqual(status, 1)
                    dispatch.assert_not_called()
            with patch.object(probe, "sdk_probe", side_effect=AssertionError("unexpected dispatch")) as dispatch, self.assertRaises(Rejected):
                probe.dry_run(MANIFEST, role="test", run_dir=base / "dev" / "wrong-role")
            dispatch.assert_not_called()
            other = Path(directory) / "other-sha-space"
            other.mkdir()
            base.parent.mkdir(parents=True)
            base.symlink_to(other, target_is_directory=True)
            with patch.object(probe, "sdk_probe", side_effect=AssertionError("unexpected dispatch")) as dispatch, self.assertRaises(Rejected):
                probe.dry_run(MANIFEST, role="test", run_dir=base / "test" / "redirected-run")
            dispatch.assert_not_called()

    def test_existing_run_cannot_be_reused(self):
        with tempfile.TemporaryDirectory(prefix="reopt-reuse-", dir="/Users/YUYSONG/Work/tmp") as directory, patch.object(probe, "EVIDENCE", Path(directory)):
            state, questions = probe.build_request(MANIFEST)
            identity = checkout_identity(MANIFEST, {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}})
            run = Path(directory) / "step-2" / identity["code_sha"] / "dev" / "run"
            run.mkdir(parents=True)
            with patch.object(probe, "sdk_probe", side_effect=AssertionError("unexpected dispatch")) as dispatch, self.assertRaises(FileExistsError):
                probe.dry_run(MANIFEST, role="dev", run_dir=run)
            dispatch.assert_not_called()

    def approved_fixture(self, root, *, env_file=None):
        state, questions = probe.build_request(MANIFEST)
        data = {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}}
        binding = checkout_identity(MANIFEST, data, env_file=env_file)
        # A synthetic clean snapshot isolates approval logic from this in-progress patch.
        binding["workspace_modified"] = False
        payload = {**data, "model": binding["model"], "max_output_tokens": 512}
        binding.update(batch_id="synthetic-approved-batch", session="dev",
                       execution_scope_hash=payload_hash({"payload": payload, "retry": 0, "timeout": 30.0,
                                                          "max_input_tokens": 8192, "max_requests": 1}),
                       acceptance_scope_hash=payload_hash({"ids": sorted(questions), "offline": False,
                                                           "output_cap_requested": True,
                                                           "provider_caps_verified": False}))
        limits = Limits(price_waived=True, output_cap_requested=True)
        human_sha = approval_digest(binding, limits, 4070908800.0)
        approval = {**binding, "limits": asdict(limits), "expires_at": 4070908800.0,
                    "human_approval_sha256": human_sha}
        path = root / "approvals" / "synthetic.json"
        path.parent.mkdir()
        path.write_text(json.dumps(approval))
        return binding, limits, approval, path, human_sha

    def test_live_preflight_requires_external_approval_before_credentials(self):
        with patch.object(probe, "resolve_project_config", side_effect=AssertionError("credentials forbidden")) as loader:
            for approval, batch, human in ((None, "synthetic", "a" * 64),
                                           (MANIFEST, None, "a" * 64), (MANIFEST, "synthetic", None)):
                with self.assertRaises(Rejected):
                    probe.live_preflight(MANIFEST, approval, batch, human)
            loader.assert_not_called()

    def test_live_preflight_checks_clean_binding_public_overrides_and_waiver(self):
        with tempfile.TemporaryDirectory(prefix="reopt-public-", dir="/Users/YUYSONG/Work/tmp") as directory:
            root = Path(directory)
            env_file = root / "public.env"
            env_file.write_text("TYPESAFE_BASE_URL=https://synthetic-file.example.invalid\n"
                                "TYPESAFE_DEFAULT_MODEL=jev-file\n")
            with patch.dict(os.environ, {"TYPESAFE_BASE_URL": "https://synthetic-env.example.invalid/",
                                         "TYPESAFE_DEFAULT_MODEL": "jev-env"}, clear=True), patch.object(probe, "EVIDENCE", root):
                binding, limits, doc, path, human = self.approved_fixture(root, env_file=env_file)
                identity = {k: v for k, v in binding.items() if k not in
                            {"batch_id", "session", "execution_scope_hash", "acceptance_scope_hash"}}
                with patch.object(probe, "checkout_identity", return_value=identity) as clean_check:
                    actual, actual_limits, actual_doc = probe.live_preflight(
                        MANIFEST, path, binding["batch_id"], human, env_file=env_file)
                clean_check.assert_called_once_with(MANIFEST, unittest.mock.ANY, clean=True, env_file=env_file)
                self.assertEqual(actual, binding)
                self.assertEqual(actual["endpoint"], "https://synthetic-env.example.invalid")
                self.assertEqual(actual["model"], "jev-env")
                self.assertEqual(actual_limits, limits)
                self.assertEqual(actual_doc, doc)
                self.assertIsNone(limits.cost)
                self.assertEqual(limits.price_status, "price unknown, user-waived")
                self.assertIsNone(limits.max_cost_usd)
                self.assertFalse(limits.provider_caps_verified)
                self.assertTrue(limits.output_cap_requested)
                for field in ("endpoint", "model", "execution_scope_hash", "acceptance_scope_hash"):
                    tampered = {**doc, field: "synthetic-tampered"}
                    path.write_text(json.dumps(tampered))
                    with patch.object(probe, "checkout_identity", return_value=dict(identity)), self.assertRaises(Rejected):
                        probe.live_preflight(MANIFEST, path, binding["batch_id"], human, env_file=env_file)

    def test_public_file_configuration_and_sdk_fallback(self):
        from eval.reopt_probe_support import resolve_project_config
        with tempfile.TemporaryDirectory(prefix="reopt-fallback-", dir="/Users/YUYSONG/Work/tmp") as directory:
            path = Path(directory) / "public.env"
            path.write_text("TYPESAFE_BASE_URL=https://synthetic-file.example.invalid/\n"
                            "TYPESAFE_DEFAULT_MODEL=jev-file\n")
            with patch.dict(os.environ, {}, clear=True):
                config = resolve_project_config(path)
                self.assertEqual(config["endpoint"], "https://synthetic-file.example.invalid")
                self.assertEqual(config["model"], "jev-file")
                self.assertNotIn(API_KEY_ENV, os.environ)
                binding = checkout_identity(MANIFEST, {"synthetic": True}, env_file=path)
                self.assertEqual(binding["endpoint"], config["endpoint"])
                self.assertEqual(binding["model"], config["model"])
            path.write_text("# Empty existing public configuration\n")
            with patch.dict(os.environ, {}, clear=True):
                config = resolve_project_config(path)
                self.assertEqual(config["endpoint"], "http://127.0.0.1:8009")
                self.assertEqual(config["model"], "kev-latest")
                binding = checkout_identity(MANIFEST, {"synthetic": True}, env_file=path)
                self.assertEqual(binding["endpoint"], config["endpoint"])
                self.assertEqual(binding["model"], config["model"])
                with self.assertRaises(Rejected):
                    resolve_project_config(Path(directory) / "missing.env")

    def test_approved_helper_selects_https_boundary_offline_and_records_unknown_price(self):
        # Mock DNS and the AsyncHTTPTransport method: no real DNS, sockets, keys,
        # models listing or LIVE CLI. The production HTTPS gate/guard remains real.
        for status in (200, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory(
                    prefix="reopt-https-boundary-", dir="/Users/YUYSONG/Work/tmp") as directory:
                root = Path(directory)
                with patch.dict(os.environ, {API_KEY_ENV: "synthetic-offline-only",
                                             "TYPESAFE_BASE_URL": "https://synthetic.example.invalid",
                                             "TYPESAFE_DEFAULT_MODEL": "jev-test"}, clear=True), patch.object(probe, "EVIDENCE", root):
                    binding, limits, doc, path, human = self.approved_fixture(root)
                    identity = {k: v for k, v in binding.items() if k not in
                                {"batch_id", "session", "execution_scope_hash", "acceptance_scope_hash"}}
                    run = root / "step-2" / binding["code_sha"] / "dev" / "offline-boundary-run"
                    requests = []
                    async def boundary(inner, request):
                        requests.append(request)
                        self.assertIsInstance(inner, httpx2.AsyncHTTPTransport)
                        self.assertEqual(str(request.url), binding["endpoint"] + "/v1/systemone")
                        self.assertEqual(json.loads(request.content)["max_output_tokens"], 512)
                        journal = root / "batches" / binding["batch_id"] / "journal.jsonl"
                        rows = [json.loads(line) for line in journal.read_text().splitlines()]
                        self.assertEqual(rows[-1]["reserved_request"], 1)  # Reserve precedes network boundary.
                        self.assertIsNone(rows[-1]["cost_usd"])
                        return httpx2.Response(status, json={"model": binding["model"], "usage": {},
                            "answers": {name: {"type": "noul", "noul": 0.9}
                                        for name in probe.build_request(MANIFEST)[1]}})
                    with patch.object(probe, "checkout_identity", return_value=dict(identity)), patch.object(
                            probe, "resolve_project_config", return_value={k: binding[k] for k in ("endpoint", "model")}) as loader, patch.object(
                            httpx2.AsyncHTTPTransport, "handle_async_request", boundary), patch.object(
                            socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]) as resolver:
                        result = probe.live_run(MANIFEST, path, binding["batch_id"], human, run_dir=run)
                    resolver.assert_called_once_with(probe.urlsplit(binding["endpoint"]).hostname, 443, 0, socket.SOCK_STREAM)
                    loader.assert_called_once_with(None, credentials=True)
                    if status == 200:
                        self.assertEqual(result["actual_socket_network_calls"], 0)
                        self.assertEqual(result["attempted_network"], 0)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(result["outcome"], "accepted" if status == 200 else "rejected")
                    self.assertEqual(result["pricing"], "price unknown, user-waived")
                    self.assertIsNone(result["actual_cost_usd"])
                    self.assertIsNone(result["reserved_cost_usd"])
                    self.assertIsNone(result["max_cost_usd"])
                    self.assertTrue(result["output_cap_requested"])
                    self.assertFalse(result["provider_caps_verified"])
                    self.assertFalse(result["live_certified"])
                    self.assertEqual(result["reserved_requests"], 1)
                    self.assertEqual(result["endpoint_host"], probe.urlsplit(binding["endpoint"]).hostname)
                    self.assertEqual(result["journal_path"], "batches/" + binding["batch_id"] + "/journal.jsonl")
                    journal = root / result["journal_path"]
                    self.assertEqual(result["journal_sha256"], sha256(journal.read_bytes()))
                    self.assertEqual(json.loads((run / "live-run.json").read_text()), result)
                    self.assertNotIn("synthetic-offline-only", json.dumps(result))
                    with patch.object(probe, "checkout_identity", return_value=dict(identity)), patch.object(
                            probe, "resolve_project_config", side_effect=AssertionError("unexpected credential access")), self.assertRaises(Rejected):
                        probe.live_run(MANIFEST, path, binding["batch_id"], human, run_dir=run)
                    with patch.object(probe, "checkout_identity", return_value=dict(identity)), patch.object(
                            probe, "resolve_project_config", return_value={k: binding[k] for k in ("endpoint", "model")}), patch.object(
                            probe, "HttpsTransport", side_effect=AssertionError("reused batch must not create transport")) as transport, self.assertRaises(Rejected):
                        probe.live_run(MANIFEST, path, binding["batch_id"], human, run_dir=run.with_name("different-run"))
                    transport.assert_not_called()

    def test_approved_tunnel_helper_skips_credentials_and_dns_offline(self):
        # Exercise the fixed HTTP transport only at its mocked async boundary.
        for status in (200, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory(
                    prefix="reopt-tunnel-boundary-", dir="/Users/YUYSONG/Work/tmp") as directory:
                root = Path(directory)
                env_file = root / "public.env"
                env_file.write_text("# Synthetic public defaults only\n")
                with patch.dict(os.environ, {API_KEY_ENV: "synthetic-offline-only"}, clear=True), patch.object(probe, "EVIDENCE", root):
                    binding, limits, doc, path, human = self.approved_fixture(root, env_file=env_file)
                    self.assertEqual(binding["endpoint"], "http://127.0.0.1:8009")
                    self.assertEqual(binding["model"], "kev-latest")
                    identity = {k: v for k, v in binding.items() if k not in
                                {"batch_id", "session", "execution_scope_hash", "acceptance_scope_hash"}}
                    run = root / "step-2" / binding["code_sha"] / "dev" / "offline-boundary-run"
                    requests = []
                    async def boundary(inner, request):
                        requests.append(request)
                        self.assertIsInstance(inner, httpx2.AsyncHTTPTransport)
                        self.assertEqual(str(request.url), "http://127.0.0.1:8009/v1/systemone")
                        self.assertNotIn("authorization", request.headers)
                        payload = json.loads(request.content)
                        self.assertEqual(payload["max_output_tokens"], 512)
                        self.assertEqual(payload["model"], "kev-latest")
                        journal = root / "batches" / binding["batch_id"] / "journal.jsonl"
                        rows = [json.loads(line) for line in journal.read_text().splitlines()]
                        self.assertEqual(rows[-1]["reserved_request"], 1)
                        return httpx2.Response(status, json={"latency_ms": 17.3,
                            "answers": {name: {"noul": 0.92} for name in probe.build_request(MANIFEST)[1]}})
                    original_get = os.environ.get
                    def public_get(key, default=None):
                        if key == API_KEY_ENV:
                            raise AssertionError("tunnel must not read an API key")
                        return original_get(key, default)
                    with patch.object(probe, "checkout_identity", return_value=dict(identity)), patch.object(
                            probe, "resolve_project_config", return_value={k: binding[k] for k in ("endpoint", "model")}) as loader, patch.object(
                            probe, "AsyncTypeSafeClient", side_effect=AssertionError("no tunnel SDK")), patch.object(
                            os.environ, "get", side_effect=public_get), patch.object(
                            httpx2.AsyncHTTPTransport, "handle_async_request", boundary), patch.object(
                            socket, "getaddrinfo", side_effect=AssertionError("tunnel must not use DNS")) as resolver:
                        result = probe.live_run(MANIFEST, path, binding["batch_id"], human,
                                                run_dir=run, env_file=env_file)
                    loader.assert_called_once_with(env_file, credentials=False)
                    resolver.assert_not_called()
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(result["outcome"], "accepted" if status == 200 else "rejected")
                    self.assertEqual(result["model"], "kev-latest")
                    self.assertEqual(result["auth_mode"], "ssh-tunnel")
                    self.assertIsNone(result["credential_variable"])
                    self.assertIsNone(result["usage"])
                    self.assertEqual(result["max_output_tokens"], 512)
                    self.assertEqual(result["max_requests"], 1)
                    self.assertEqual(result["retry"], 0)
                    self.assertEqual(result["timeout"], 30.0)
                    self.assertEqual(result["reserved_requests"], 1)
                    self.assertTrue(result["price_waived"])
                    self.assertFalse(result["provider_caps_verified"])
                    if status == 200:
                        self.assertEqual(result["actual_socket_network_calls"], 0)
                        self.assertIsNone(result["response_model"])
                        self.assertEqual(result["model_source"], "request")
                        self.assertEqual(result["latency_ms"], 17.3)
                    self.assertNotIn("synthetic-offline-only", json.dumps(result))

    def test_tunnel_dry_run_needs_no_key_or_sdk(self):
        with tempfile.TemporaryDirectory(prefix="reopt-kev-dry-", dir="/Users/YUYSONG/Work/tmp") as directory:
            root = Path(directory)
            env_file = root / "public.env"
            env_file.write_text("# Synthetic public defaults only\n")
            with patch.dict(os.environ, {}, clear=True), patch.object(probe, "EVIDENCE", root), patch.object(
                    probe, "AsyncTypeSafeClient", side_effect=AssertionError("no tunnel SDK")):
                result = probe.dry_run(MANIFEST, env_file=env_file)
                self.assertNotIn(API_KEY_ENV, os.environ)
            self.assertEqual(result["model"], "kev-latest")
            self.assertEqual(result["endpoint"], "http://127.0.0.1:8009")
            self.assertEqual(result["auth_mode"], "ssh-tunnel")
            self.assertIsNone(result["credential_variable"])
            self.assertIsNone(result["usage"])
            self.assertIsNone(result["response_model"])
            self.assertEqual(result["model_source"], "request")
            self.assertEqual(result["actual_socket_network_calls"], 0)
            self.assertEqual(result["cloud_requests"], 0)

    def test_cli_invalid_arguments_are_redacted(self):
        marker = "synthetic-" + "invalid-argument-value"
        errors = io.StringIO()
        with patch("sys.stderr", errors):
            code = probe.main(["--mode", marker])
        self.assertEqual(code, 1)
        self.assertTrue(marker not in errors.getvalue(), "redaction failed")

    def test_tampered_manifest_rejected_without_provider_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="reopt-manifest-", dir="/Users/YUYSONG/Work/tmp") as directory:
            manifest = json.loads(MANIFEST.read_text())
            manifest["contract_version"] = "0" * 64
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                probe.build_request(path)


if __name__ == "__main__":
    unittest.main()
