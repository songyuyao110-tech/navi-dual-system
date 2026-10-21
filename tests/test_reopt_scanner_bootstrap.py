"""Pre-staging negative controls using real temporary Git blobs and index."""

import hashlib
import unittest

from eval.verify_delivery import DeliveryError
from test_reopt_delivery import (
    ALLOWLIST, PAYLOAD, SCANNER, TemporaryRepo, credential_canary, private_key_canary,
)


class ScannerBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.repo = TemporaryRepo()
        self.addCleanup(self.repo.close)

    def assert_rejected_without_leak(self, value):
        with self.assertRaises(DeliveryError) as caught:
            self.repo.check()
        self.assertTrue(value not in str(caught.exception), "synthetic value leaked")
        result = self.repo.cli("--check", "staged", "--step", "1", "--allowlist", ALLOWLIST)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(value not in result.stdout + result.stderr, "synthetic value leaked")
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_staged_credential_worktree_safe_rejected(self):
        data, value = credential_canary()
        self.repo.stage(PAYLOAD, data, worktree=b"safe worktree replacement\n")
        self.assert_rejected_without_leak(value)

    def test_staged_private_key_worktree_safe_rejected(self):
        data, value = private_key_canary()
        self.repo.stage(PAYLOAD, data, worktree=b"safe worktree replacement\n")
        self.assert_rejected_without_leak(value)

    def test_staged_safe_worktree_credential_is_not_scanned(self):
        data, value = credential_canary()
        self.repo.stage(PAYLOAD, b"safe staged bytes\n", worktree=data)
        self.assertEqual(self.repo.check()["check"], "staged")
        result = self.repo.cli("--check", "staged", "--step", "1")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(value not in result.stdout + result.stderr, "synthetic value leaked")

    def test_tracked_ignored_sensitive_filename_rejected(self):
        path = "credentials" + ".json"
        self.repo.allow((path,))
        self.repo.stage(path, b"safe bytes\n")
        self.repo.stage(".gitignore", (path + "\n").encode())
        self.repo.commit()
        # It is tracked, ignored, and unchanged relative to HEAD: still rejected.
        self.assertEqual(self.repo.git("check-ignore", "--no-index", path).strip().decode(), path)
        self.assert_rejected_without_leak(path)

    def test_scanner_source_is_scannable_and_receipt_identifies_exact_bytes(self):
        source = SCANNER.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        path = "eval/verify_delivery.py"
        self.repo.approved["1"].append(path)  # Test-only fixed fixture mapping.
        self.repo.allow((path,))
        oid = self.repo.stage(path, source)
        staged = self.repo.git("cat-file", "blob", oid)
        self.assertEqual(hashlib.sha256(staged).hexdigest(), digest)
        self.assertEqual(self.repo.check()["checked_paths"], 3)
        # A value-free marker lets the external bootstrap receipt bind W.
        print("scanner bootstrap source sha256=" + digest)

    def test_missing_git_object_and_sensitive_cli_values_never_echo(self):
        _, value = credential_canary()
        for args in (("--check", "commit", "--step", "1", "--sha", value),
                     ("--check", "staged", "--step", "1", "--allowlist", value),
                     ("--check", "staged", "--step", value)):
            result = self.repo.cli(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(value not in result.stdout + result.stderr, "synthetic value leaked")
            self.assertNotIn("Traceback", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
