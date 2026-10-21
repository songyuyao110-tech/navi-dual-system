from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from eval import validate_reopt as contract
from execution import ExecutionError, Runtime, ToolCall
from registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/fixtures/reopt-manifest.json"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.frozen = contract.validate(MANIFEST)["contract"]

    def test_fourteen_actual_registry_schemas_and_runtime_validation(self):
        runtime = Runtime()
        snapshot = load_registry(ROOT / "tools.yaml", frozenset(runtime.handlers))
        self.assertEqual(len(self.frozen["tools"]), 14)
        self.assertEqual(len({t["name"] for t in self.frozen["tools"]}), 14)
        result = runtime.execute(snapshot, frozenset({"navi_get_status"}), ToolCall("navi_get_status", {}))
        self.assertTrue(result["mock"])
        for name, args in (("navi_start_route", {}), ("navi_add_waypoint", {"name": "synthetic"})):
            with self.subTest(name=name), self.assertRaises(ExecutionError):
                runtime.validate(snapshot, frozenset({name}), ToolCall(name, args))
        self.assertEqual(len(runtime.calls), 1)

    def test_idempotency_is_not_readonly_and_parameter_effects(self):
        tools = {t["name"]: t for t in self.frozen["tools"]}
        for name in ("navi_open_address_page", "navi_show_poi_list", "navi_search_in_address_page"):
            self.assertEqual(tools[name]["side_effect"], "idempotent")
            self.assertEqual(contract.effect(tools[name], {})[0], "ui_write")
        search = tools["navi_search_keyword"]
        self.assertEqual(contract.effect(search, {"show_in_map": False}), ("read_only", []))
        self.assertEqual(contract.effect(search, {"show_in_map": True}), ("ui_write", ["navigation_ui"]))
        self.assertEqual(contract.effect(search, {})[0], "unknown_effect")
        self.assertEqual(contract.effect(tools["navi_search_along_poi"], {})[0], "unknown_effect")

    def test_four_assessments_separate_endpoints_and_send_evidence(self):
        self.assertEqual(self.frozen["assessments"], ["已满足", "待确认", "不满足", "未知"])
        self.assertEqual(set(self.frozen["endpoints"]), {"task_delivery_correct", "operation_effect_correct"})
        self.assertNotEqual(*self.frozen["endpoints"].values())
        for case in ("queue_then_cancel", "partial_write", "write_then_ledger_error", "hidden_retry", "prepared_without_proof"):
            self.assertEqual(self.frozen["send_evidence_cases"][case], "unknown")
        self.assertEqual(self.frozen["send_evidence_cases"]["all_paths_revoked_before_handoff"], "not_sent")
        self.assertEqual(self.frozen["send_evidence_cases"]["device_rejected_after_receipt"], "rejected")

    def reject_variant(self, mutate, rehash=False):
        # Real validator and registry on disposable files; no formal fixture modifications.
        with tempfile.TemporaryDirectory(prefix="reopt-contract-", dir="/Users/YUYSONG/Work/tmp") as directory:
            root = Path(directory)
            (root / "eval/fixtures").mkdir(parents=True)
            for name in ("tools.yaml", "registry.py", "execution.py"):
                (root / name).write_bytes((ROOT / name).read_bytes())
            frozen, manifest = deepcopy(self.frozen), json.loads(MANIFEST.read_text())
            mutate(frozen, manifest)
            if rehash:
                frozen["contract_version"] = contract.digest({k: v for k, v in frozen.items() if k != "contract_version"})
                manifest["contract_version"] = frozen["contract_version"]
            path = root / manifest["contract_file"]
            path.write_text(json.dumps(frozen, ensure_ascii=False))
            if rehash:
                manifest["contract_file_sha256"] = contract.file_hash(path)
            local = root / "manifest.json"
            local.write_text(json.dumps(manifest))
            with patch.object(contract, "ROOT", root), self.assertRaises(ValueError):
                contract.validate(local)

    def test_swapped_contract_and_wrong_manifest_hash_rejected(self):
        self.reject_variant(lambda c, m: c["tools"][0]["postconditions"].append("wrong_contract"))
        self.reject_variant(lambda c, m: m.update(contract_version="0" * 64))

    def test_duplicate_ids_and_mixed_side_table_rejected_even_with_new_hash(self):
        self.reject_variant(lambda c, m: c["tools"][1].update(name=c["tools"][0]["name"]), rehash=True)
        self.reject_variant(lambda c, m: c["tools"][0].update(semantics_version="wrong-version"), rehash=True)

    def test_cli_and_acceptance_reject_wrong_manifest_without_receipt(self):
        with tempfile.TemporaryDirectory(prefix="reopt-cli-", dir="/Users/YUYSONG/Work/tmp") as directory:
            manifest = json.loads(MANIFEST.read_text())
            manifest["contract_version"] = "0" * 64
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            commands = [
                ["eval/validate_reopt.py", "--manifest", str(path), "--check", "contract", "--offline"],
                ["eval/run_step_acceptance.py", "--step", "1", "--sha", sha, "--manifest", str(path),
                 "--offline", "--receipt-dir", str(Path(directory) / "receipts")],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((Path(directory) / "receipts").exists())

    def test_rehashed_effect_resource_modes_rejected(self):
        # Every tool's reviewed classification remains mandatory after rehashing.
        for tool in self.frozen["tools"]:
            name = tool["name"]
            bad = {"kind": "ui_write", "resources": ["navigation_ui"]} if tool["effect"].get("kind") == "read_only" else {"kind": "read_only", "resources": []}
            with self.subTest(tool=name):
                self.reject_variant(lambda c, m: next(t for t in c["tools"] if t["name"] == name).update(effect=bad), rehash=True)
        self.reject_variant(lambda c, m: c["tools"][0]["effect"].update(resources=["navigation_ui"]), rehash=True)
        self.reject_variant(lambda c, m: next(t for t in c["tools"] if t["name"] == "navi_search_keyword")["effect"]["modes"].update(unspecified={"kind": "read_only", "resources": []}), rehash=True)

    def test_rehashed_policy_and_condition_types_rejected(self):
        for key in ("resource_policy", "noop_policy"):
            with self.subTest(policy=key):
                self.reject_variant(lambda c, m: c.pop(key), rehash=True)
                self.reject_variant(lambda c, m: c.update({key: "unsafe"}), rehash=True)
                self.reject_variant(lambda c, m: c.update({key: []}), rehash=True)
        for key in ("preconditions", "postconditions"):
            for bad in ("not_a_list", [], [""], [False]):
                with self.subTest(condition=key, bad_type=type(bad).__name__):
                    self.reject_variant(lambda c, m: c["tools"][0].update({key: bad}), rehash=True)

    def reject_acceptance_variant(self, mutate, after_commit=None):
        from eval.run_step_acceptance import REQUIRED_TESTS
        from eval.verify_delivery import APPROVED_STEP_PATHS, BASELINE
        # Disposable repo reads baseline objects only; no original refs/index writes.
        with tempfile.TemporaryDirectory(prefix="reopt-acceptance-", dir="/Users/YUYSONG/Work/tmp") as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            env = {"PATH": os.defpath, "HOME": directory, "PYTHONPATH": str(root),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                   "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                   "GIT_AUTHOR_NAME": "Synthetic Test", "GIT_COMMITTER_NAME": "Synthetic Test",
                   "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_EMAIL": "test@example.invalid"}

            def git(*args):
                result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(root), *args], env=env, capture_output=True)
                self.assertEqual(result.returncode, 0, "temporary Git command failed")
                return result.stdout.decode().strip()

            common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
            if not common.is_absolute():
                common = ROOT / common
            git("init", "--quiet")
            (root / ".git/objects/info/alternates").write_text(str(common.resolve() / "objects") + "\n")
            git("checkout", "--detach", BASELINE)
            for name in APPROVED_STEP_PATHS["1"]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            mutate(root)
            git("add", "--", *APPROVED_STEP_PATHS["1"])
            git("commit", "--quiet", "-m", "step 1: synthetic acceptance negative")
            sha = git("rev-parse", "HEAD")
            if after_commit is not None:
                after_commit(root, git)
            # Output stays outside the disposable checkout, inside its temp container.
            receipts = Path(directory) / "receipts"
            result = subprocess.run([sys.executable, "eval/run_step_acceptance.py", "--step", "1", "--sha", sha,
                                     "--manifest", "eval/fixtures/reopt-manifest.json", "--offline", "--receipt-dir", str(receipts)],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, "unsafe acceptance variant was accepted")
            self.assertNotIn("Traceback", result.stderr)
            path = receipts / "step-1.json"
            if path.exists():
                record = json.loads(path.read_text())
                self.assertFalse(record["passed"])
                self.assertEqual(set(record["missing_required_test_ids"]), REQUIRED_TESTS)
                return record

    def test_acceptance_rejects_missing_required_ids(self):
        def stub(root):
            for name in ("test_reopt_contract.py", "test_reopt_delivery.py", "test_reopt_scanner_bootstrap.py"):
                (root / "tests" / name).write_text('"""Synthetic missing Step 1 tests."""\n')
        record = self.reject_acceptance_variant(stub)
        self.assertIsNotNone(record)
        self.assertGreater(record["tests_run"], 0)  # Old regression still genuinely runs.
        self.assertEqual(record["skips"], 0)
        self.assertEqual(record["expected_failures"], 0)

    def test_acceptance_rejects_rehashed_unsafe_contract(self):
        def unsafe(root):
            path = root / "eval/fixtures/reopt-contract14.json"
            frozen = json.loads(path.read_text())
            frozen.pop("resource_policy")
            frozen.pop("noop_policy")
            next(t for t in frozen["tools"] if t["name"] == "navi_start_route")["effect"] = {"kind": "read_only", "resources": []}
            frozen["contract_version"] = contract.digest({k: v for k, v in frozen.items() if k != "contract_version"})
            path.write_text(json.dumps(frozen, ensure_ascii=False))
            manifest_path = root / "eval/fixtures/reopt-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(contract_version=frozen["contract_version"], contract_file_sha256=contract.file_hash(path))
            manifest_path.write_text(json.dumps(manifest))
        self.assertIsNone(self.reject_acceptance_variant(unsafe))

    def test_rehashed_legal_semantic_values_rejected(self):
        for tool in self.frozen["tools"]:
            name = tool["name"]
            for key, bad in (("preconditions", ["authorization_not_required"]),
                             ("postconditions", ["argument_echo_is_success"]),
                             ("confirmation", "argument_echo_is_effect_evidence"),
                             ("legacy_mock_limit", "legacy_binding_has_verified_real_effects")):
                with self.subTest(tool=name, field=key):
                    self.reject_variant(lambda c, m: next(t for t in c["tools"] if t["name"] == name).update({key: bad}), rehash=True)

    def test_rehashed_condition_removal_and_global_version_rejected(self):
        for tool in self.frozen["tools"]:
            name = tool["name"]
            for key in ("preconditions", "postconditions"):
                if len(tool[key]) > 1:
                    with self.subTest(tool=name, field=key):
                        self.reject_variant(lambda c, m: next(t for t in c["tools"] if t["name"] == name)[key].pop(0), rehash=True)
        def replace_all_versions(frozen, manifest):
            frozen["semantics_version"] = "self-consistent-wrong-version"
            for tool in frozen["tools"]:
                tool["semantics_version"] = frozen["semantics_version"]
        self.reject_variant(replace_all_versions, rehash=True)
        self.reject_variant(lambda c, m: c.update(scope="verified_real_binding"), rehash=True)

    def test_acceptance_rejects_rehashed_legal_semantics(self):
        def unsafe(root):
            path = root / "eval/fixtures/reopt-contract14.json"
            frozen = json.loads(path.read_text())
            for tool in frozen["tools"]:
                tool.update(preconditions=["authorization_not_required"], postconditions=["argument_echo_is_success"],
                            confirmation="argument_echo_is_effect_evidence")
            frozen["contract_version"] = contract.digest({k: v for k, v in frozen.items() if k != "contract_version"})
            path.write_text(json.dumps(frozen, ensure_ascii=False))
            manifest_path = root / "eval/fixtures/reopt-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(contract_version=frozen["contract_version"], contract_file_sha256=contract.file_hash(path))
            manifest_path.write_text(json.dumps(manifest))
        self.assertIsNone(self.reject_acceptance_variant(unsafe))

    def test_acceptance_rejects_index_hidden_source_drift(self):
        for mark in ("--assume-unchanged", "--skip-worktree"):
            with self.subTest(mark=mark):
                def hidden_drift(root, git):
                    git("update-index", mark, "app.py")
                    path = root / "app.py"
                    path.write_bytes(path.read_bytes() + b"\n# synthetic hidden drift\n")
                    self.assertEqual(git("diff", "HEAD", "--name-only"), "")
                self.assertIsNone(self.reject_acceptance_variant(lambda root: None, after_commit=hidden_drift))

    def test_no_false_heldout_or_fifth_assessment(self):
        self.reject_variant(lambda c, m: m["common32"].update(independent_held_out=True), rehash=True)
        self.reject_variant(lambda c, m: c["assessments"].append("done"), rehash=True)


if __name__ == "__main__":
    unittest.main()
