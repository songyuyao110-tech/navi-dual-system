"""Step 1 offline acceptance only; this is not a live/preflight guard."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from eval.validate_reopt import digest, file_hash, validate


# Reviewed Step 1 obligations, independent of candidate manifest and test count.
REQUIRED_TESTS = frozenset(
    f"{host}.{name}"
    for host, names in {
        "test_reopt_contract.ContractTests": (
            "test_cli_and_acceptance_reject_wrong_manifest_without_receipt",
            "test_duplicate_ids_and_mixed_side_table_rejected_even_with_new_hash",
            "test_four_assessments_separate_endpoints_and_send_evidence",
            "test_fourteen_actual_registry_schemas_and_runtime_validation",
            "test_idempotency_is_not_readonly_and_parameter_effects",
            "test_no_false_heldout_or_fifth_assessment",
            "test_swapped_contract_and_wrong_manifest_hash_rejected",
            "test_rehashed_effect_resource_modes_rejected",
            "test_rehashed_policy_and_condition_types_rejected",
            "test_acceptance_rejects_missing_required_ids",
            "test_acceptance_rejects_rehashed_unsafe_contract",
            "test_rehashed_legal_semantic_values_rejected",
            "test_rehashed_condition_removal_and_global_version_rejected",
            "test_acceptance_rejects_rehashed_legal_semantics",
            "test_acceptance_rejects_index_hidden_source_drift",
        ),
        "test_reopt_delivery.DeliveryTests": (
            "test_allowlist_is_staged_object_not_worktree",
            "test_checkout_rejects_worktree_index_untracked_and_ignored",
            "test_cli_success_and_invalid_arguments_redacted",
            "test_commit_and_clean_checkout",
            "test_commit_reads_object_not_worktree",
            "test_credential_and_key_formats_are_redacted",
            "test_full_sha_and_checkout_head_required",
            "test_invalid_allowlist_shapes",
            "test_out_of_allowlist_even_for_deletion",
            "test_staged_safe_delivery",
            "test_unsupported_step_and_symlink_blob",
            "test_full_baseline_short_tag_shadow_rejected",
            "test_baseline_requires_full_exact_identity",
            "test_allowlist_cannot_self_expand_staged_or_commit",
            "test_allowlist_cannot_shrink_or_add_steps",
            "test_allowlist_override_rejected",
            "test_checkout_rejects_assume_unchanged_clean_and_dirty",
            "test_checkout_rejects_skip_worktree_clean_and_dirty",
            "test_checkout_rejects_combined_index_marks",
            "test_checkout_rejects_byte_mismatch_when_git_diff_is_empty",
            "test_checkout_rejects_missing_symlink_and_outside_root",
            "test_checkout_binds_index_and_worktree_to_commit_blobs",
            "test_staged_safe_worktree_sensitive_is_accepted",
            "test_commit_safe_worktree_sensitive_is_accepted",
        ),
        "test_reopt_scanner_bootstrap.ScannerBootstrapTests": (
            "test_missing_git_object_and_sensitive_cli_values_never_echo",
            "test_scanner_source_is_scannable_and_receipt_identifies_exact_bytes",
            "test_staged_credential_worktree_safe_rejected",
            "test_staged_private_key_worktree_safe_rejected",
            "test_staged_safe_worktree_credential_is_not_scanned",
            "test_tracked_ignored_sensitive_filename_rejected",
        ),
    }.items()
    for name in names
)


class RequiredResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        self.passed.append(test.id())
        super().addSuccess(test)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1], required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.receipt_dir / "step-1.json"
    try:
        if receipt.exists():
            raise ValueError("receipt exists")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if len(args.sha) != 40 or args.sha != sha:
            raise ValueError("exact SHA required")
        for name in ("registry", "execution", "decisions", "model_clients", "app", "task_context", "gateway"):
            origin = Path(importlib.util.find_spec(name).origin).resolve()
            if not origin.is_relative_to(ROOT):
                raise ValueError("source import outside checkout")
        os.environ["TOOLS_FILE"] = str(ROOT / "tools.yaml")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        from eval.verify_delivery import verify_delivery
        delivery = verify_delivery("checkout", step=1, sha=sha, repo=ROOT)
        frozen = validate(args.manifest)
        attempted = []

        def deny(*unused, **kwargs):
            attempted.append("socket_call_blocked")
            raise OSError("offline acceptance forbids socket network/bind")

        with patch.object(socket.socket, "connect", deny), \
             patch.object(socket.socket, "connect_ex", deny), \
             patch.object(socket.socket, "bind", deny):
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
            count = suite.countTestCases()
            if count == 0:
                raise ValueError("empty tests")
            result = unittest.TextTestRunner(verbosity=2, resultclass=RequiredResult).run(suite)
        # Critical skip/expectedFailure and any socket attempt fail the entire Step.
        missing_required = sorted(REQUIRED_TESTS - set(result.passed))
        passed = (result.wasSuccessful() and not result.skipped and not result.expectedFailures
                  and not attempted and result.testsRun == count and not missing_required)
        verify_delivery("checkout", step=1, sha=sha, repo=ROOT)
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(ROOT.glob("*.py"))}
        sources.update({str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "tests").glob("test*.py"))})
        sources.update({str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "eval").glob("*.py"))})
        record = {"step": 1, "source_sha": sha, "suite_type": "offline_all_regression",
                  "role": args.receipt_dir.parent.parent.name,
                  "run_id": args.receipt_dir.parent.name,
                  "contract_version": frozen["contract_version"],
                  "manifest_sha256": file_hash(args.manifest),
                  "expectation_matrix_sha256": digest({k: frozen["contract"][k] for k in ("assessments", "endpoints", "send_evidence_cases")}),
                  "source_sha256": sources, "assertion_ids_passed": result.passed,
                  "required_test_ids": sorted(REQUIRED_TESTS), "missing_required_test_ids": missing_required,
                  "tests_run": result.testsRun, "skips": len(result.skipped),
                  "expected_failures": len(result.expectedFailures),
                  "attempted_socket_calls": len(attempted), "actual_socket_network_calls": 0,
                  "cloud_requests": 0, "cost": 0, "passed": passed,
                  "delivery": delivery}
        args.receipt_dir.mkdir(parents=True, exist_ok=True)
        with receipt.open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"passed": passed, "tests_run": result.testsRun,
                          "receipt_sha256": file_hash(receipt)}))
        return 0 if passed else 1
    except Exception:
        # No arbitrary exception/provider/body values in delivery output.
        print("step acceptance rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
