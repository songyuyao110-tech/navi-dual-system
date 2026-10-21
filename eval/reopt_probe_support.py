"""One-request offline preflight primitives; no general-purpose network sandbox."""
from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import sys
from threading import Lock, get_ident
import time
from urllib.parse import urlsplit

import httpx2

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("/Users/YUYSONG/Work/tmp/jev-reopt-evidence")
DEFAULT_BASE_URL = "http://127.0.0.1:8009"
DEFAULT_MODEL = "kev-latest"
STEP_PATHS = (
    "eval/reopt_probe_support.py", "eval/jev_protocol_probe.py",
    "tests/test_reopt_guard.py", "tests/test_reopt_probe_budget.py", "tests/test_reopt_jev_probe.py",
)


class Rejected(ValueError):
    """Only static reason codes, never provider bodies, headers or credentials."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_hash(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode())


def is_tunnel(endpoint: str) -> bool:
    try:
        url = urlsplit(endpoint)
        return (url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == 8009
                and url.path in ("", "/") and not (url.username or url.password or url.query or url.fragment))
    except (ValueError, TypeError):
        return False


def endpoint_ok(endpoint: str) -> None:
    if is_tunnel(endpoint):
        return
    try:
        url = urlsplit(endpoint)
        host = url.hostname or ""
        if (url.scheme != "https" or url.port not in (None, 443) or url.username or url.password
                or url.query or url.fragment or url.path not in ("", "/")
                or not re.fullmatch(r"[a-zA-Z0-9.-]+", host)
                or host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local"))):
            raise Rejected("endpoint_rejected")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if "." not in host:
                raise Rejected("endpoint_rejected")
        else:
            raise Rejected("endpoint_rejected")  # No IP/loopback endpoints in this probe.
    except (ValueError, TypeError):
        raise Rejected("endpoint_rejected") from None


def resolve_project_config(env_file: Path | None = None, *, credentials: bool = False) -> dict:
    from jev_smoke import ENV_NAMES
    from typesafe_sdk.constants import API_KEY_ENV
    names = {"TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"}
    if credentials:
        names.add(API_KEY_ENV)
    if env_file is not None:
        path = Path(env_file)
        if not path.is_file():
            raise Rejected("config_missing")
        with path.open() as stream:
            for line in stream:
                line = line.strip().removeprefix("export ")
                name, sep, value = line.partition("=")
                name = name.strip()
                if not sep or name not in names or name not in ENV_NAMES:
                    continue
                try:
                    parsed = shlex.split(value, comments=True)
                except ValueError:
                    raise Rejected("config_rejected") from None
                if len(parsed) != 1:
                    raise Rejected("config_rejected")
                os.environ.setdefault(name, parsed[0])  # Same existing environment-wins rule.
    endpoint = os.environ.get("TYPESAFE_BASE_URL", "").strip() or DEFAULT_BASE_URL
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL
    endpoint_ok(endpoint)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", model):
        raise Rejected("model_rejected")
    return {"endpoint": endpoint.rstrip("/"), "model": model,
            "config_source": str(env_file) if env_file is not None else "environment or SDK defaults"}


def checkout_identity(manifest: Path, data: dict, *, clean: bool = False, env_file: Path | None = None) -> dict:
    from eval.validate_reopt import validate
    from eval.verify_delivery import _git, _index, _tree

    sha = _git(ROOT, "rev-parse", "HEAD^{commit}").strip().decode()
    entries = _tree(ROOT, sha)
    if _index(ROOT) != entries or any(not row.startswith(b"H ") for row in
                                    _git(ROOT, "ls-files", "-v", "-f", "-z").split(b"\0") if row):
        raise Rejected("checkout_rejected")
    workspace_modified = False
    for name, (mode, oid) in entries.items():
        path = ROOT / name
        if mode not in (b"100644", b"100755") or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise Rejected("checkout_rejected")
        content = path.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest().encode() != oid:
            if clean or name not in STEP_PATHS:
                raise Rejected("checkout_rejected")
            workspace_modified = True  # Offline prototype binds these bytes through source_sha256.
    others = {s.decode() for s in _git(ROOT, "ls-files", "--others", "-z").split(b"\0") if s}
    if others and (clean or others - set(STEP_PATHS)):
        raise Rejected("checkout_rejected")
    for name in ("decisions", "registry", "execution", "model_clients", "jev_smoke", "eval.reopt_probe_support"):
        if not Path(importlib.util.find_spec(name).origin).resolve().is_relative_to(ROOT):
            raise Rejected("import_rejected")
    validate(manifest)
    config = resolve_project_config(env_file)
    return {"code_sha": sha, "source_sha256": payload_hash({p: sha256((ROOT / p).read_bytes()) for p in STEP_PATHS}),
            "manifest_sha256": sha256(manifest.read_bytes()), "data_sha256": payload_hash(data),
            "endpoint": config["endpoint"], "model": config["model"], "config_source": config["config_source"],
            "workspace_modified": workspace_modified or bool(others)}


@dataclass(frozen=True)
class Limits:
    max_requests: int = 1
    max_input_tokens: int = 8192
    max_output_tokens: int = 512
    timeout: float = 30.0
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    provider_caps_verified: bool = False
    output_cap_requested: bool = False
    price_waived: bool = False
    max_cost_usd: float | None = None

    @property
    def cost(self) -> float | None:
        if (self.max_requests != 1 or type(self.max_requests) is not int
                or any(type(n) is not int or n <= 0 for n in (self.max_input_tokens, self.max_output_tokens))
                or isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or not 0 < self.timeout <= 30
                or self.max_output_tokens > 512
                or type(self.price_waived) is not bool or type(self.output_cap_requested) is not bool
                or (self.provider_caps_verified is not True and self.output_cap_requested is not True)
                or (self.price_waived and not self.output_cap_requested)):
            raise Rejected("limits_unverified")
        ceiling = self.max_cost_usd
        if ceiling is not None and (isinstance(ceiling, bool) or not isinstance(ceiling, (float, int))
                                    or not math.isfinite(ceiling) or ceiling < 0):
            raise Rejected("cost_ceiling_rejected")
        rates = (self.input_usd_per_million, self.output_usd_per_million)
        if any(r is not None and (isinstance(r, bool) or not isinstance(r, (int, float))
                                  or not math.isfinite(r) or r <= 0) for r in rates):
            raise Rejected("price_rejected")
        if any(r is None for r in rates):
            if not self.price_waived or ceiling is not None:
                raise Rejected("price_unknown")
            return None
        cost = (self.max_input_tokens * rates[0] + self.max_output_tokens * rates[1]) / 1_000_000
        if not math.isfinite(cost) or cost <= 0 or (ceiling is not None and cost > ceiling):
            raise Rejected("cost_ceiling_rejected")
        return cost

    @property
    def price_status(self) -> str:
        return "price unknown, user-waived" if self.cost is None else "price known"


_BINDING_FIELDS = (
    "code_sha", "source_sha256", "manifest_sha256", "data_sha256", "endpoint", "model",
    "batch_id", "session", "execution_scope_hash", "acceptance_scope_hash",
)


def approval_digest(binding: dict, limits: Limits, expires_at: float) -> str:
    """Hash trusted human-approved content, not a label or an untrusted LIVE candidate."""
    return payload_hash({**{key: binding[key] for key in _BINDING_FIELDS},
                         "limits": asdict(limits), "expires_at": expires_at})


def check_approval(doc: dict, binding: dict, limits: Limits, human_sha: str | None = None,
                   now: float | None = None) -> None:
    if not isinstance(human_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", human_sha):
        raise Rejected("human_approval_missing")
    if not isinstance(doc, dict) or doc.get("human_approval_sha256") != human_sha:
        raise Rejected("human_approval_mismatch")
    if not set(_BINDING_FIELDS) <= binding.keys():
        raise Rejected("approval_identity_missing")
    if any(k not in doc or doc[k] != binding[k] for k in _BINDING_FIELDS) or doc.get("limits") != asdict(limits):
        raise Rejected("approval_identity_mismatch")
    expires = doc.get("expires_at")
    if isinstance(expires, bool) or not isinstance(expires, (float, int)) or not math.isfinite(expires) or expires <= (time.time() if now is None else now):
        raise Rejected("approval_expired")
    _ = limits.cost  # Validate finite limits before canonicalizing their content.
    try:
        digest = approval_digest(doc, Limits(**doc["limits"]), expires)
    except (TypeError, ValueError):
        raise Rejected("approval_content_mismatch") from None
    if digest != human_sha:
        raise Rejected("approval_content_mismatch")


class Budget:
    """One-shot batch directory is the inter-process lock; never reopen/reuse it."""
    def __init__(self, root: Path, batch_id: str, binding: dict, limits: Limits,
                 approval: dict, human_sha: str):
        check_approval(approval, binding, limits, human_sha)
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", batch_id) or binding.get("batch_id") != batch_id:
            raise Rejected("batch_identity_mismatch")
        if root.is_symlink():
            raise Rejected("batch_root_rejected")
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / batch_id
        try:
            self.directory.mkdir(mode=0o700)
        except FileExistsError:
            raise Rejected("batch_reuse_or_concurrency") from None
        self.journal = self.directory / "journal.jsonl"
        self._append({"binding_sha256": payload_hash(binding), "limits": asdict(limits)}, exclusive=True)
        self._tail = sha256(self.journal.read_bytes())
        self._lock = Lock()
        self.limits = limits
        self.binding = json.loads(json.dumps(binding, allow_nan=False))
        self.approval = json.loads(json.dumps(approval, allow_nan=False))
        self._human_sha = human_sha
        self._requests = 0
        self._cost_usd = None if limits.cost is None else 0.0
        started = time.monotonic()
        self._time_window = (started, started + limits.timeout)  # Immutable batch snapshot.

    @property
    def started(self) -> float:
        return self._time_window[0]

    @property
    def deadline(self) -> float:
        return self._time_window[1]

    @property
    def requests(self) -> int:
        return self._requests

    @property
    def cost_usd(self) -> float | None:
        return self._cost_usd

    def _append(self, value: dict, *, exclusive: bool = False) -> None:
        fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_APPEND), 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            check_approval(self.approval, self.binding, self.limits, self._human_sha)
            journal = self.journal.read_bytes()
            if sha256(journal) != self._tail:
                raise Rejected("journal_tampered")
            try:
                rows = [json.loads(line) for line in journal.splitlines()]
            except (ValueError, UnicodeError):
                raise Rejected("journal_tampered") from None
            if not rows or rows[0] != {"binding_sha256": payload_hash(self.binding),
                                       "limits": asdict(self.limits)}:
                raise Rejected("journal_state_mismatch")
            unit_cost = self.limits.cost
            cost = None if unit_cost is None else 0.0
            for sequence, row in enumerate(rows[1:], 1):
                cost = None if unit_cost is None else cost + unit_cost
                if row != {"reserved_request": sequence, "input_tokens": self.limits.max_input_tokens,
                           "output_tokens": self.limits.max_output_tokens, "cost_usd": cost, "price_status": self.limits.price_status}:
                    raise Rejected("journal_state_mismatch")
            if (type(self._requests) is not int or self._requests != len(rows) - 1
                    or self._cost_usd != cost or self._requests > self.limits.max_requests):
                raise Rejected("journal_state_mismatch")
            if (time.monotonic() >= self.deadline
                    or time.time() >= self.approval["expires_at"]):
                raise Rejected("batch_expired")
            if any(type(n) is not int or n <= 0 for n in (input_tokens, output_tokens)):
                raise Rejected("invalid_reservation")
            if (self.requests >= self.limits.max_requests or input_tokens > self.limits.max_input_tokens
                    or output_tokens > self.limits.max_output_tokens):
                raise Rejected("budget_exhausted")
            total = None if unit_cost is None else self.cost_usd + unit_cost  # No failure/unknown refund.
            self._append({"reserved_request": self.requests + 1, "input_tokens": self.limits.max_input_tokens,
                          "output_tokens": self.limits.max_output_tokens, "cost_usd": total, "price_status": self.limits.price_status})
            self._tail = sha256(self.journal.read_bytes())
            self._requests += 1
            self._cost_usd = total


_active_fence = None
_fence_lock = Lock()
_live_permit = ContextVar("single_approved_https_dispatch", default=None)


def _audit(event, args):
    if _active_fence is not None:
        if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"}:
            fence = _active_fence
            fence.attempted_network += 1
            within_time = time.monotonic() < fence._network_deadline and time.time() < fence._approval_expires
            if event == "socket.getaddrinfo" and fence._host and within_time:
                host = args[0].decode("ascii") if isinstance(args[0], bytes) else args[0]
                if host == fence._host and args[1] == fence._port:
                    return  # Async resolver may run on its library worker thread.
            if (event == "socket.connect" and _live_permit.get() is fence and within_time
                    and get_ident() == fence._dispatch_thread):
                address = args[1]
                if isinstance(address, tuple) and len(address) >= 2 and address[0] in fence._addresses and address[1] == fence._port:
                    fence.actual_socket_network_calls += 1  # Authorized pre-syscall connects, not OS success telemetry.
                    return
            raise Rejected("network_forbidden")
        if event in {"subprocess.Popen", "os.system", "os.fork", "os.posix_spawn", "os.exec", "os.spawn"}:
            raise Rejected("subprocess_forbidden")


sys.addaudithook(_audit)


class NetworkFence(AbstractContextManager):
    """Process audit fence includes threads; child creation is forbidden, not sandboxed."""
    def __init__(self):
        self.attempted_network = 0
        self.actual_socket_network_calls = 0
        self._host, self._addresses, self._network_deadline = None, set(), 0.0
        self._approval_expires, self._dispatch_thread, self._port = 0.0, None, None

    def __enter__(self):
        global _active_fence
        if not _fence_lock.acquire(blocking=False):
            raise Rejected("concurrent_probe_forbidden")
        _active_fence = self
        return self

    def __exit__(self, *unused):
        global _active_fence
        _active_fence = None
        _fence_lock.release()


class HttpsTransport(httpx2.AsyncHTTPTransport):
    """Existing HTTPX transport for exact SSH tunnel or HTTPS override; no proxies/retries."""
    def __init__(self):
        super().__init__(verify=True, trust_env=False, retries=0)


class GuardedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, inner, budget: Budget, payload: dict, endpoint: str):
        endpoint_ok(endpoint)
        if (endpoint.rstrip("/") != budget.binding["endpoint"].rstrip("/")
                or payload.get("model") != budget.binding["model"]
                or payload_hash({k: payload[k] for k in ("state", "questions")}) != budget.binding["data_sha256"]):
            raise Rejected("transport_identity_mismatch")
        self.inner, self.budget = inner, budget
        self.payload = json.loads(json.dumps(payload, allow_nan=False))
        self.endpoint = str(httpx2.URL(endpoint.rstrip("/") + "/v1/systemone"))
        self.attempted_dispatch = 0

    async def handle_async_request(self, request):
        self.attempted_dispatch += 1
        if _active_fence is None:
            raise Rejected("fence_required")
        if request.method != "POST" or str(request.url) != self.endpoint:
            raise Rejected("dispatch_rejected")
        permit = None
        try:
            body = json.loads(request.content)
            if payload_hash(body) != payload_hash(self.payload):
                raise Rejected("request_identity_mismatch")
            real = isinstance(self.inner, HttpsTransport)
            if not real and not isinstance(self.inner, httpx2.MockTransport):
                raise Rejected("transport_type_rejected")
            if self.budget.limits.output_cap_requested or real:
                if type(body.get("max_output_tokens")) is not int or body["max_output_tokens"] != self.budget.limits.max_output_tokens:
                    raise Rejected("output_cap_rejected")
            self.budget.reserve(len(request.content), self.budget.limits.max_output_tokens)
            if real:
                fence = _active_fence
                fence._host = request.url.host
                fence._port = request.url.port or 443
                fence._network_deadline = self.budget.deadline
                fence._approval_expires = self.budget.approval["expires_at"]
                fence._dispatch_thread = get_ident()
                permit = _live_permit.set(fence)
                if is_tunnel(self.budget.binding["endpoint"]):
                    ips = {"127.0.0.1"}  # Exact literal SSH forward; no external DNS or global-IP test.
                else:
                    addresses = await asyncio.to_thread(socket.getaddrinfo, fence._host, fence._port, 0, socket.SOCK_STREAM)
                    ips = {item[4][0] for item in addresses}
                    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
                        raise Rejected("endpoint_address_rejected")
                fence._addresses = ips
                if time.monotonic() >= fence._network_deadline or time.time() >= fence._approval_expires:
                    raise Rejected("network_permit_expired")
            response = await self.inner.handle_async_request(request)
            if response.status_code != 200:
                raise Rejected("provider_status_rejected")
            if len(await response.aread()) > 8192:
                raise Rejected("response_size_rejected")
            return response
        except Exception:
            raise Rejected("transport_rejected") from None
        finally:
            if permit is not None:
                _live_permit.reset(permit)
                fence._host, fence._addresses, fence._network_deadline = None, set(), 0.0
                fence._approval_expires, fence._dispatch_thread, fence._port = 0.0, None, None

    async def aclose(self):
        if self.inner is not None:
            await self.inner.aclose()
