"""Offline Step 1 delivery checks. Never print Git errors or matched content."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


BASELINE = "ac4eda67c8f2ab8f849d7451433b80d68a3c6ed8"
DEFAULT_ALLOWLIST = "eval/acceptance/step-paths.json"
APPROVED_STEP_PATHS = {
    "1": [
        "eval/fixtures/reopt-contract14.json",
        "eval/fixtures/reopt-manifest.json",
        "eval/validate_reopt.py",
        "eval/verify_delivery.py",
        "eval/run_step_acceptance.py",
        "eval/acceptance/step-paths.json",
        "tests/test_reopt_contract.py",
        "tests/test_reopt_delivery.py",
        "tests/test_reopt_scanner_bootstrap.py",
    ]
}


class DeliveryError(ValueError):
    """A safe, value-free delivery rejection."""


def _git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if result.returncode:
        raise DeliveryError("Git verification failed")
    return result.stdout


def _paths(data):
    return [p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p]


def _safe_path(path):
    parts = PurePosixPath(path).parts
    return bool(parts) and not path.startswith("/") and all(
        p not in (".", "..", ".git") for p in parts
    ) and "\\" not in path and not any(ord(c) < 32 for c in path)


def _sensitive_path(path):
    for part in PurePosixPath(path).parts:
        name = part.lower()
        if (name.startswith(".env") or name.startswith(("id_rsa", "id_ed25519", "id_ecdsa"))
                or re.search(r"(?:^|[._-])(?:credentials?|secrets?)(?:$|[._-])", name)
                or name.endswith((".pem", ".key", ".p12", ".pfx", ".keystore"))):
            return True
    return False


_PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_CREDENTIAL_PATTERN = re.compile(
    rb"(?im)[\"']?\b(?:[a-z0-9_]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    rb"client[_-]?secret|password|passwd|credential|secret[_-]?key))\b[\"']?"
    rb"\s*[:=]\s*(?:[\"'][^\"'\r\n]+[\"']|[a-z0-9_+/.=-]+)"
)
_AUTHORIZATION = re.compile(rb"(?i)authorization[\"']?\s*[:=]\s*[\"']?(?:bearer|basic)\s+\S+")


def scan_blob(path, data):
    """Reject risky names and credential-shaped bytes without echoing either."""
    if not _safe_path(path) or _sensitive_path(path):
        raise DeliveryError("Sensitive or invalid delivery path")
    if _PRIVATE_KEY.search(data) or _CREDENTIAL_PATTERN.search(data) or _AUTHORIZATION.search(data):
        raise DeliveryError("Sensitive delivery content")


def _index(repo):
    entries = {}
    for record in _git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise DeliveryError("Unmerged index")
        entries[path.decode("utf-8", "surrogateescape")] = (mode, oid)
    return entries


def _tree(repo, sha):
    entries = {}
    for record in _git(repo, "ls-tree", "-r", "-z", sha).split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        entries[path.decode("utf-8", "surrogateescape")] = (mode, oid)
    return entries


def _blob(repo, entries, path):
    if path not in entries or entries[path][0] not in (b"100644", b"100755"):
        raise DeliveryError("Missing or unsupported delivery blob")
    return _git(repo, "cat-file", "blob", entries[path][1].decode("ascii"))


def verify_delivery(check, step=1, sha=None, allowlist=DEFAULT_ALLOWLIST, repo=None):
    """Read staged blobs or exact commit objects; return only non-sensitive metadata."""
    if step != 1 or check not in ("staged", "commit", "checkout"):
        raise DeliveryError("Unsupported delivery check")
    repo = Path(repo or Path.cwd()).resolve()
    root = Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel")).strip()).resolve()
    if root != repo or allowlist != DEFAULT_ALLOWLIST:
        raise DeliveryError("Invalid repository or allowlist path")
    if not isinstance(BASELINE, str) or not re.fullmatch(r"[0-9a-f]{40}", BASELINE):
        raise DeliveryError("Full exact baseline SHA required")
    base = _git(repo, "rev-parse", "--verify", BASELINE + "^{commit}").strip().decode("ascii")
    if base != BASELINE:
        raise DeliveryError("Baseline identity mismatch")
    if check == "staged":
        target = _git(repo, "rev-parse", "HEAD^{commit}").strip().decode("ascii")
        entries = _index(repo)
        changed = _paths(_git(repo, "diff", "--cached", "--no-renames", "--name-only", "-z", base, "--"))
    else:
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise DeliveryError("Full exact commit SHA required")
        target = _git(repo, "rev-parse", "--verify", sha + "^{commit}").strip().decode("ascii")
        if target != sha:
            raise DeliveryError("Commit identity mismatch")
        entries = _tree(repo, sha)
        changed = _paths(_git(repo, "diff", "--no-renames", "--name-only", "-z", base, sha, "--"))
    _git(repo, "merge-base", "--is-ancestor", base, target)
    # Ignored tracked names remain visible here; ignore rules cannot bypass safety.
    for path in entries:
        if not _safe_path(path) or _sensitive_path(path):
            raise DeliveryError("Sensitive or invalid delivery path")
    try:
        mapping = json.loads(_blob(repo, entries, allowlist))
        allowed = mapping[str(step)]
        if (not isinstance(mapping, dict) or mapping != APPROVED_STEP_PATHS
                or not isinstance(allowed, list) or not allowed
                or any(not isinstance(p, str) or not _safe_path(p) or _sensitive_path(p) for p in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise DeliveryError("Invalid delivery allowlist") from None
    if set(changed) - set(allowed):
        raise DeliveryError("Delivery path outside step allowlist")
    for path in changed:
        if path in entries:  # Deletions still undergo path-allowlist checking.
            scan_blob(path, _blob(repo, entries, path))
    if check == "checkout":
        head = _git(repo, "rev-parse", "HEAD^{commit}").strip().decode("ascii")
        if head != sha:
            raise DeliveryError("Checkout identity mismatch")
        if any(not record.startswith(b"H ") for record in
               _git(repo, "ls-files", "-v", "-f", "-z").split(b"\0") if record):
            raise DeliveryError("Checkout index flags are not allowed")
        if _index(repo) != entries or _git(repo, "ls-files", "--others", "-z"):
            raise DeliveryError("Checkout is not clean")
        for path, (mode, oid) in entries.items():
            target_path = root / path
            try:
                if (mode not in (b"100644", b"100755")
                        or not stat.S_ISREG(target_path.lstat().st_mode)
                        or any(parent.is_symlink() for parent in target_path.parents if parent != root)
                        or not target_path.resolve(strict=True).is_relative_to(root)):
                    raise DeliveryError("Missing or unsupported checkout file")
                data = target_path.read_bytes()
            except OSError:
                raise DeliveryError("Missing or unsupported checkout file") from None
            actual = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest().encode("ascii")
            if actual != oid:
                raise DeliveryError("Checkout bytes differ from commit blob")
        if _git(repo, "diff", "--name-only", "-z", "HEAD", "--"):
            raise DeliveryError("Checkout is not clean")
    return {"check": check, "step": step, "sha": target, "checked_paths": len(changed)}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes invalid values, which may themselves be secrets.
        raise DeliveryError("Invalid delivery CLI arguments")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--check", required=True, choices=("staged", "commit", "checkout"))
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--sha")
    parser.add_argument("--allowlist", default=DEFAULT_ALLOWLIST)
    try:
        args = parser.parse_args(argv)
        receipt = verify_delivery(args.check, args.step, args.sha, args.allowlist)
    except (DeliveryError, OSError, UnicodeError):
        print("delivery check rejected (details redacted)", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
