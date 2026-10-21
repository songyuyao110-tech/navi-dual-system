"""Delivery checks run only against standalone repositories under Work/tmp."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval.verify_delivery import DeliveryError, scan_blob, verify_delivery


SCANNER = Path(__file__).resolve().parents[1] / "eval" / "verify_delivery.py"
ALLOWLIST = "eval/acceptance/step-paths.json"
PAYLOAD = "eval/payload.txt"


def credential_canary():
    value = "synthetic-" + "canary-" + "never-real-983714"
    return ("LITELLM_" + "API_KEY" + '="' + value + '"\n').encode(), value


def private_key_canary():
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    value = "synthetic-" + "private-material-74631"
    return (marker + "\n" + value + "\n").encode(), value


class TemporaryRepo:
    """Real Git objects/index, never the development checkout's Git storage."""

    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(
            prefix="reopt-delivery-", dir="/Users/YUYSONG/Work/tmp"
        )
        self.root = Path(self.directory.name)
        self.env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
                    "PYTHONDONTWRITEBYTECODE": "1"}
        self.env.update({
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Synthetic Test", "GIT_COMMITTER_NAME": "Synthetic Test",
            "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
            "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
            "GIT_INDEX_FILE": str(self.root / ".git" / "temporary.index"),
        })
        self.git("init", "--quiet")
        self.stage(".gitignore", b".env*\n*.pem\nignored-output/\ntemporary.index\n")
        self.base = self.commit()
        self.approved = {"1": [ALLOWLIST, PAYLOAD]}
        self.allow()
        self.stage(PAYLOAD, b"safe payload\n")

    def close(self):
        self.directory.cleanup()

    def git(self, *args, data=None):
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
             "-C", str(self.root), *args], env=self.env, input=data, capture_output=True,
        )
        if result.returncode:
            raise AssertionError("Temporary Git operation failed (details redacted)")
        return result.stdout

    def stage(self, path, data, worktree=None, mode="100644"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if worktree is None else worktree)
        oid = self.git("hash-object", "-w", "--stdin", data=data).strip().decode()
        self.git("update-index", "--add", "--cacheinfo", mode, oid, path)
        return oid

    def allow(self, extra=()):
        self.stage(ALLOWLIST, json.dumps({"1": [ALLOWLIST, PAYLOAD, *extra]}).encode())

    def commit(self):
        self.git("commit", "--quiet", "-m", "step 1: synthetic delivery")
        return self.git("rev-parse", "HEAD").strip().decode()

    def check(self, check="staged", sha=None, allowlist=ALLOWLIST, step=1):
        with (patch.dict(os.environ, self.env, clear=True),
              patch("eval.verify_delivery.BASELINE", self.base),
              patch("eval.verify_delivery.APPROVED_STEP_PATHS", self.approved)):
            return verify_delivery(check, step, sha, allowlist, self.root)

    def cli(self, *args):
        # Test-only loader: production CLI has no baseline or approved-path override.
        loader = (
            "import importlib.util,json,sys; "
            "spec=importlib.util.spec_from_file_location('fixture_delivery',sys.argv[1]); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.BASELINE=sys.argv[2]; module.APPROVED_STEP_PATHS=json.loads(sys.argv[3]); "
            "raise SystemExit(module.main(sys.argv[4:]))"
        )
        return subprocess.run(
            [sys.executable, "-c", loader, str(SCANNER), self.base,
             json.dumps(self.approved), *args], cwd=self.root, env=self.env,
            capture_output=True, text=True,
        )


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.repo = TemporaryRepo()
        self.addCleanup(self.repo.close)

    def test_staged_safe_delivery(self):
        receipt = self.repo.check()
        self.assertEqual(receipt["checked_paths"], 2)
        self.assertEqual(receipt["step"], 1)

    def test_commit_and_clean_checkout(self):
        sha = self.repo.commit()
        self.assertEqual(self.repo.check("commit", sha)["sha"], sha)
        self.assertEqual(self.repo.check("checkout", sha)["sha"], sha)

    def test_commit_reads_object_not_worktree(self):
        data, value = credential_canary()
        self.repo.stage(PAYLOAD, data)
        sha = self.repo.commit()
        (self.repo.root / PAYLOAD).write_text("safe replacement\n")
        with self.assertRaises(DeliveryError) as caught:
            self.repo.check("commit", sha)
        self.assertTrue(value not in str(caught.exception), "synthetic value leaked")

    def test_out_of_allowlist_even_for_deletion(self):
        self.repo.stage("unapproved.txt", b"safe\n")
        with self.assertRaises(DeliveryError):
            self.repo.check()
        self.repo.git("update-index", "--force-remove", "unapproved.txt")
        self.repo.git("update-index", "--force-remove", ".gitignore")
        with self.assertRaises(DeliveryError):
            self.repo.check()

    def test_allowlist_is_staged_object_not_worktree(self):
        self.repo.stage("unapproved.txt", b"safe\n")
        (self.repo.root / ALLOWLIST).write_text(json.dumps({"1": [ALLOWLIST, PAYLOAD, "unapproved.txt"]}))
        with self.assertRaises(DeliveryError):
            self.repo.check()

    def test_invalid_allowlist_shapes(self):
        for mapping in ([], {"1": "all"}, {"1": []}, {"1": [PAYLOAD, PAYLOAD]}, {"1": ["../outside"]}):
            with self.subTest(shape=type(mapping).__name__):
                self.repo.stage(ALLOWLIST, json.dumps(mapping).encode())
                with self.assertRaises(DeliveryError):
                    self.repo.check()

    def test_full_sha_and_checkout_head_required(self):
        sha = self.repo.commit()
        for candidate in (None, sha[:7], "HEAD", "f" * 40):
            with self.subTest(candidate_type=type(candidate).__name__):
                with self.assertRaises(DeliveryError):
                    self.repo.check("commit", candidate)
        with self.assertRaises(DeliveryError):
            self.repo.check("checkout", self.repo.base)

    def test_checkout_rejects_worktree_index_untracked_and_ignored(self):
        sha = self.repo.commit()
        for path, ignored in ((PAYLOAD, False), ("new.txt", False), ("ignored-output/result.txt", True)):
            target = self.repo.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = target.read_bytes() if target.exists() else None
            target.write_bytes(b"changed\n")
            with self.subTest(ignored=ignored):
                with self.assertRaises(DeliveryError):
                    self.repo.check("checkout", sha)
            if previous is None:
                target.unlink()  # Only synthetic files in this disposable repository.
            else:
                target.write_bytes(previous)
        self.repo.stage(PAYLOAD, b"staged change\n", worktree=b"safe payload\n")
        with self.assertRaises(DeliveryError):
            self.repo.check("checkout", sha)

    def test_checkout_rejects_assume_unchanged_clean_and_dirty(self):
        sha = self.repo.commit()
        self.repo.git("update-index", "--assume-unchanged", PAYLOAD)
        self.assertIn(b"h " + PAYLOAD.encode() + b"\0", self.repo.git("ls-files", "-v", "-z"))
        for data in (b"safe payload\n", b"hidden drift\n"):
            with self.subTest(dirty=data != b"safe payload\n"):
                (self.repo.root / PAYLOAD).write_bytes(data)
                self.assertEqual(self.repo.git("diff", "--name-only", "HEAD", "--"), b"")
                with self.assertRaisesRegex(DeliveryError, "index flags"):
                    self.repo.check("checkout", sha)

    def test_checkout_rejects_skip_worktree_clean_and_dirty(self):
        sha = self.repo.commit()
        self.repo.git("update-index", "--skip-worktree", PAYLOAD)
        self.assertIn(b"S " + PAYLOAD.encode() + b"\0", self.repo.git("ls-files", "-v", "-z"))
        for data in (b"safe payload\n", b"hidden drift\n"):
            with self.subTest(dirty=data != b"safe payload\n"):
                (self.repo.root / PAYLOAD).write_bytes(data)
                self.assertEqual(self.repo.git("diff", "--name-only", "HEAD", "--"), b"")
                with self.assertRaisesRegex(DeliveryError, "index flags"):
                    self.repo.check("checkout", sha)

    def test_checkout_rejects_combined_index_marks(self):
        sha = self.repo.commit()
        self.repo.git("update-index", "--assume-unchanged", PAYLOAD)
        self.repo.git("update-index", "--skip-worktree", PAYLOAD)
        self.assertIn(b"s " + PAYLOAD.encode() + b"\0", self.repo.git("ls-files", "-v", "-z"))
        for data in (b"safe payload\n", b"hidden drift\n"):
            with self.subTest(dirty=data != b"safe payload\n"):
                (self.repo.root / PAYLOAD).write_bytes(data)
                self.assertEqual(self.repo.git("diff", "--name-only", "HEAD", "--"), b"")
                with self.assertRaisesRegex(DeliveryError, "index flags"):
                    self.repo.check("checkout", sha)

    def test_checkout_rejects_byte_mismatch_when_git_diff_is_empty(self):
        sha = self.repo.commit()
        target = self.repo.root / PAYLOAD
        # An old timestamp avoids Git's racy-index protection without flags.
        cached_time = target.stat().st_mtime_ns - 10_000_000_000
        os.utime(target, ns=(cached_time, cached_time))
        self.repo.git("config", "core.trustctime", "false")
        self.repo.git("config", "core.checkstat", "minimal")
        self.repo.git("update-index", "--refresh")
        before = target.stat()
        target.write_bytes(b"evil payload\n")  # Same length; restore index-cached times.
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.repo.git("config", "core.trustctime", "false")
        self.assertEqual(self.repo.git("diff", "--name-only", "HEAD", "--"), b"")
        self.assertIn(b"H " + PAYLOAD.encode() + b"\0", self.repo.git("ls-files", "-v", "-z"))
        with self.assertRaisesRegex(DeliveryError, "bytes differ"):
            self.repo.check("checkout", sha)

    def test_checkout_rejects_missing_symlink_and_outside_root(self):
        sha = self.repo.commit()
        target = self.repo.root / PAYLOAD
        target.unlink()
        with self.assertRaisesRegex(DeliveryError, "unsupported checkout file"):
            self.repo.check("checkout", sha)
        # Both targets are synthetic; the second is genuinely outside the repo.
        external = tempfile.TemporaryDirectory(prefix="reopt-outside-", dir="/Users/YUYSONG/Work/tmp")
        self.addCleanup(external.cleanup)
        inside = self.repo.root / "inside.txt"
        outside = Path(external.name) / "outside.txt"
        for source in (inside, outside):
            source.write_bytes(b"safe payload\n")
            target.symlink_to(source)
            with self.subTest(location=source.name):
                with self.assertRaises(DeliveryError):
                    self.repo.check("checkout", sha)
            target.unlink()
            source.unlink()
        # A symlinked directory must not redirect reads outside the checkout.
        target.write_bytes(b"safe payload\n")
        directory = self.repo.root / "eval"
        relocated = Path(external.name) / "relocated-eval"
        directory.rename(relocated)
        directory.symlink_to(relocated, target_is_directory=True)
        with self.assertRaises(DeliveryError):
            self.repo.check("checkout", sha)

    def test_checkout_binds_index_and_worktree_to_commit_blobs(self):
        sha = self.repo.commit()
        self.repo.stage(PAYLOAD, b"index-only drift\n", worktree=b"safe payload\n")
        # Commit validation stays object-bound despite a divergent index.
        self.assertEqual(self.repo.check("commit", sha)["sha"], sha)
        with self.assertRaisesRegex(DeliveryError, "not clean"):
            self.repo.check("checkout", sha)
        self.repo.stage(PAYLOAD, b"safe payload\n", worktree=b"worktree-only drift\n")
        self.assertEqual(self.repo.check("commit", sha)["sha"], sha)
        with self.assertRaisesRegex(DeliveryError, "bytes differ"):
            self.repo.check("checkout", sha)
        (self.repo.root / PAYLOAD).write_bytes(b"safe payload\n")
        # Baseline tracked files outside the delivery allowlist are also bound.
        (self.repo.root / ".gitignore").write_bytes(b"# baseline worktree drift\n")
        with self.assertRaisesRegex(DeliveryError, "bytes differ"):
            self.repo.check("checkout", sha)

    def test_staged_safe_worktree_sensitive_is_accepted(self):
        data, _ = credential_canary()
        self.repo.stage(PAYLOAD, b"safe payload\n", worktree=data)
        self.repo.git("update-index", "--skip-worktree", PAYLOAD)
        self.assertEqual(self.repo.check("staged")["check"], "staged")
        result = self.repo.cli("--check", "staged", "--step", "1")
        self.assertEqual(result.returncode, 0)

    def test_commit_safe_worktree_sensitive_is_accepted(self):
        sha = self.repo.commit()
        data, _ = credential_canary()
        (self.repo.root / PAYLOAD).write_bytes(data)
        self.repo.git("update-index", "--assume-unchanged", PAYLOAD)
        self.assertEqual(self.repo.check("commit", sha)["sha"], sha)
        result = self.repo.cli("--check", "commit", "--step", "1", "--sha", sha)
        self.assertEqual(result.returncode, 0)

    def test_unsupported_step_and_symlink_blob(self):
        with self.assertRaises(DeliveryError):
            self.repo.check(step=2)
        self.repo.stage(PAYLOAD, b"/outside", mode="120000")
        with self.assertRaises(DeliveryError):
            self.repo.check()

    def test_credential_and_key_formats_are_redacted(self):
        for data, value in (credential_canary(), private_key_canary()):
            with self.subTest(kind="synthetic"):
                with self.assertRaises(DeliveryError) as caught:
                    scan_blob(PAYLOAD, data)
                self.assertTrue(value not in str(caught.exception), "synthetic value leaked")
        for name in (".env", "folder/.env.local", "credentials.json", "private.pem", "id_rsa"):
            with self.assertRaises(DeliveryError):
                scan_blob(name, b"safe bytes")

    def test_full_baseline_short_tag_shadow_rejected(self):
        data, value = credential_canary()
        self.repo.stage(PAYLOAD, data, worktree=b"safe replacement\n")
        sha = self.repo.commit()
        self.repo.git("tag", "ac4eda6", sha)
        self.repo.git("tag", self.repo.base[:7], sha)
        for check in ("staged", "commit"):
            with self.subTest(check=check):
                with self.assertRaises(DeliveryError) as caught:
                    self.repo.check(check, sha)
                self.assertEqual(str(caught.exception), "Sensitive delivery content")
                result = self.repo.cli("--check", check, "--step", "1", "--sha", sha)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(value not in result.stdout + result.stderr, "synthetic value leaked")
                self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_baseline_requires_full_exact_identity(self):
        with (patch.dict(os.environ, self.repo.env, clear=True),
              patch("eval.verify_delivery.BASELINE", self.repo.base[:7]),
              patch("eval.verify_delivery.APPROVED_STEP_PATHS", self.repo.approved)):
            with self.assertRaises(DeliveryError) as caught:
                verify_delivery("staged", repo=self.repo.root)
            self.assertEqual(str(caught.exception), "Full exact baseline SHA required")
        sha = self.repo.commit()
        self.repo.git("tag", "-a", "fixture-baseline", "-m", "synthetic tag object", sha)
        # A full tag-object SHA peels to a different commit: not an exact baseline.
        self.repo.base = self.repo.git("rev-parse", "refs/tags/fixture-baseline").strip().decode()
        with self.assertRaises(DeliveryError) as caught:
            self.repo.check("commit", sha)
        self.assertEqual(str(caught.exception), "Baseline identity mismatch")

    def test_allowlist_cannot_self_expand_staged_or_commit(self):
        for path in ("unapproved.txt", "app.py"):
            with self.subTest(path=path):
                repo = TemporaryRepo()  # Isolate each expansion from other unapproved paths.
                self.addCleanup(repo.close)
                repo.allow((path,))
                repo.stage(path, b"# synthetic outside approved scope\n")
                with self.assertRaises(DeliveryError) as caught:
                    repo.check()
                self.assertEqual(str(caught.exception), "Invalid delivery allowlist")
                result = repo.cli("--check", "staged", "--step", "1")
                self.assertNotEqual(result.returncode, 0)
                sha = repo.commit()
                with self.assertRaises(DeliveryError) as caught:
                    repo.check("commit", sha)
                self.assertEqual(str(caught.exception), "Invalid delivery allowlist")
                result = repo.cli("--check", "commit", "--step", "1", "--sha", sha)
                self.assertNotEqual(result.returncode, 0)

    def test_allowlist_cannot_shrink_or_add_steps(self):
        for mapping in ({"1": [ALLOWLIST]},
                        {"1": [ALLOWLIST, PAYLOAD], "2": ["app.py"]},
                        {"1": [PAYLOAD, ALLOWLIST]}):
            with self.subTest(mapping=mapping):
                self.repo.stage(ALLOWLIST, json.dumps(mapping).encode())
                with self.assertRaises(DeliveryError):
                    self.repo.check()
                sha = self.repo.commit()
                with self.assertRaises(DeliveryError):
                    self.repo.check("commit", sha)

    def test_allowlist_override_rejected(self):
        alternate = "eval/alternate-paths.json"
        self.repo.stage(alternate, json.dumps(self.repo.approved).encode())
        with self.assertRaises(DeliveryError):
            self.repo.check(allowlist=alternate)
        for args in (("--allowlist", alternate), ("--baseline", self.repo.base)):
            result = self.repo.cli("--check", "staged", "--step", "1", *args)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_cli_success_and_invalid_arguments_redacted(self):
        result = self.repo.cli("--check", "staged", "--step", "1", "--allowlist", ALLOWLIST)
        self.assertEqual(result.returncode, 0, "safe CLI unexpectedly rejected")
        self.assertEqual(json.loads(result.stdout)["check"], "staged")
        _, value = credential_canary()
        result = self.repo.cli("--check", value, "--step", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(value not in result.stdout + result.stderr, "synthetic value leaked")


if __name__ == "__main__":
    unittest.main()
