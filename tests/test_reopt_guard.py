"""Offline guard contract: native audit events and httpx2 MockTransport only."""

import asyncio
from contextvars import copy_context
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy
from typesafe_sdk.constants import API_KEY_ENV

from eval.reopt_probe_support import (
    Budget, GuardedTransport, HttpsTransport, Limits, NetworkFence, Rejected, approval_digest, endpoint_ok, is_tunnel, payload_hash,
)


ENDPOINT = "https://synthetic.example.invalid"
CANARY = "synthetic-" + "private-response-" + "983714"


class GuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="reopt-guard-", dir="/Users/YUYSONG/Work/tmp"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = {
            "model": "jev-test", "state": "synthetic task",
            "questions": {"probe": {"type": "noul", "instructions": "synthetic"}},
            "max_output_tokens": 32,
        }
        self.delegated = 0
        self.batches = 0

    def budget(self, *, expires_at=4070908800.0, endpoint=ENDPOINT, **overrides):
        limits = Limits(input_usd_per_million=1.0, output_usd_per_million=2.0,
                        provider_caps_verified=True, **overrides)
        self.batches += 1
        batch_id = f"synthetic-batch-{self.batches}"
        binding = {
            "code_sha": "b" * 40, "source_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "data_sha256": payload_hash({
                "state": self.payload["state"], "questions": self.payload["questions"],
            }),
            "execution_scope_hash": "e" * 64, "acceptance_scope_hash": "f" * 64,
            "endpoint": endpoint, "model": self.payload["model"],
            "batch_id": batch_id, "session": "synthetic-session",
        }
        human_sha = approval_digest(binding, limits, expires_at)
        approval = {
            **binding, "limits": asdict(limits),
            "human_approval_sha256": human_sha, "expires_at": expires_at,
        }
        return Budget(self.root, batch_id, binding, limits, approval, human_sha)

    def transport(self, response=None, **limits):
        def delegate(request):
            self.delegated += 1
            return response if response is not None else httpx2.Response(200, json={"ok": True})
        return GuardedTransport(
            httpx2.MockTransport(delegate), self.budget(**limits), self.payload, ENDPOINT
        )

    def request(self, *, method="POST", url=None, payload=None, content=None):
        return httpx2.Request(
            method, url or ENDPOINT + "/v1/systemone",
            content=content if content is not None else json.dumps(
                self.payload if payload is None else payload, separators=(",", ":")
            ).encode(),
        )

    def assert_redacted(self, exception):
        self.assertTrue(CANARY not in str(exception), "redaction failed")
        self.assertTrue(CANARY not in repr(exception), "redaction failed")

    async def test_thread_socket_audit_blocked_before_network(self):
        results = []
        def worker():
            try:
                sys.audit("socket.connect", None, ("127.0.0.1", 8260))
            except Exception as error:
                results.append(error)
        with NetworkFence() as fence:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive(), "fenced worker did not terminate")
            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], Rejected)
            self.assertGreaterEqual(fence.attempted_network, 1)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_bind_dns_sendto_and_child_audits_never_execute(self):
        events = [
            ("socket.bind", (None, ("127.0.0.1", 0))),
            ("socket.getaddrinfo", ("synthetic.example.invalid", 443, 0, 0, 0)),
            ("socket.sendto", (None, ("127.0.0.1", 8260))),
            ("subprocess.Popen", ("synthetic-child", ["synthetic-child"], None, None)),
            ("os.system", (b"synthetic-child",)),
            ("os.fork", ()),
            ("os.posix_spawn", ("synthetic-child", ["synthetic-child"], {})),
        ]
        with NetworkFence() as fence:
            for event, arguments in events:
                with self.subTest(event=event), self.assertRaises(Rejected):
                    # Emit exactly the native pre-syscall event; never perform a syscall.
                    sys.audit(event, *arguments)
            self.assertGreaterEqual(fence.attempted_network, 3)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_models_get_and_unapproved_urls_do_not_delegate(self):
        transport = self.transport()
        cases = [
            self.request(method="GET", url=ENDPOINT + "/v1/models"),
            self.request(url="http://127.0.0.1:8260/v1/systemone"),
            self.request(url="https://other.example.invalid/v1/systemone"),
            self.request(url=ENDPOINT + "/v1/systemone?unexpected=1"),
        ]
        with NetworkFence() as fence:
            for request in cases:
                with self.subTest(url=str(request.url)), self.assertRaises(Rejected):
                    await transport.handle_async_request(request)
            self.assertEqual(transport.attempted_dispatch, len(cases))
            self.assertEqual(self.delegated, 0)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_wrong_model_body_and_data_identity_do_not_delegate(self):
        transport = self.transport()
        cases = [
            self.request(payload={**self.payload, "model": "wrong-model"}),
            self.request(payload={**self.payload, "state": CANARY}),
            self.request(payload={**self.payload, "questions": {}}),
            self.request(content=b"not-json-" + CANARY.encode()),
        ]
        with NetworkFence() as fence:
            for request in cases:
                with self.assertRaises(Rejected) as caught:
                    await transport.handle_async_request(request)
                self.assert_redacted(caught.exception)
            self.assertEqual(self.delegated, 0)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_reservation_precedes_inner_dispatch(self):
        transport = self.transport()
        with patch.object(transport.budget, "reserve", side_effect=Rejected("synthetic rejection")) as reserve:
            with NetworkFence() as fence, self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            reserve.assert_called_once_with(
                len(self.request().content), transport.budget.limits.max_output_tokens
            )
            self.assertEqual(self.delegated, 0)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_actual_sdk_retry_zero_mock_transport_succeeds(self):
        def delegate(request):
            self.delegated += 1
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), ENDPOINT + "/v1/systemone")
            return httpx2.Response(200, json={
                "model": "jev-test", "usage": {},
                "answers": {"probe": {"type": "noul", "noul": 0.9}},
            })
        with patch.dict(os.environ, {API_KEY_ENV: "synthetic-" + "test-only"}, clear=True), NetworkFence() as fence:
            async with AsyncTypeSafeClient(
                base_url=ENDPOINT, retry=RetryPolicy(max_retries=0),
                transport=httpx2.MockTransport(delegate),
            ) as client:
                response = await client.system_one(
                    model="jev-test", state="synthetic task",
                    questions={"probe": Noul(instructions="synthetic")},
                )
            self.assertEqual(response.answers["probe"].noul, 0.9)
            self.assertEqual(self.delegated, 1)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_redirect_5xx_and_inner_error_redact_canary(self):
        for status in (301, 302, 307, 308, 500, 503):
            before = self.delegated
            transport = self.transport(httpx2.Response(
                status, text=CANARY, headers={"location": "http://127.0.0.1:8260/"}
            ))
            with self.subTest(status=status), NetworkFence() as fence:
                with self.assertRaises(Rejected) as caught:
                    await transport.handle_async_request(self.request())
                self.assert_redacted(caught.exception)
                self.assertEqual(transport.attempted_dispatch, 1)
                self.assertEqual(self.delegated, before + 1)
                self.assertEqual(fence.actual_socket_network_calls, 0)
        def failing(request):
            raise RuntimeError(CANARY)
        transport = GuardedTransport(httpx2.MockTransport(failing), self.budget(), self.payload, ENDPOINT)
        with NetworkFence(), self.assertRaises(Rejected) as caught:
            await transport.handle_async_request(self.request())
        self.assert_redacted(caught.exception)

    async def test_excessive_response_bytes_rejected(self):
        transport = self.transport(httpx2.Response(200, content=b"x" * (2 * 1024 * 1024)))
        with NetworkFence() as fence, self.assertRaises(Rejected):
            await transport.handle_async_request(self.request())
        self.assertEqual(self.delegated, 1)
        self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_n_plus_one_counts_attempts_but_delegates_once(self):
        transport = self.transport()
        with NetworkFence() as fence:
            self.assertEqual((await transport.handle_async_request(self.request())).status_code, 200)
            with self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            self.assertEqual(transport.attempted_dispatch, 2)
            self.assertEqual(self.delegated, 1)
            self.assertEqual(fence.actual_socket_network_calls, 0)

    async def test_counter_reset_cannot_dispatch_second_request(self):
        transport = self.transport()
        with NetworkFence():
            await transport.handle_async_request(self.request())
            journal = transport.budget.journal.read_bytes()
            with self.assertRaises(AttributeError):
                transport.budget.requests = 0
            transport.budget._requests = 0  # Accidental private-state drift must disagree with the durable record.
            with self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
        self.assertEqual(self.delegated, 1)
        self.assertEqual(transport.budget.journal.read_bytes(), journal)

    async def test_unapproved_expiry_extension_never_dispatches(self):
        transport = self.transport()
        journal = transport.budget.journal.read_bytes()
        transport.budget.approval["expires_at"] += 3600
        with NetworkFence(), self.assertRaises(Rejected):
            await transport.handle_async_request(self.request())
        self.assertEqual(self.delegated, 0)
        self.assertEqual(transport.budget.requests, 0)
        self.assertEqual(transport.budget.journal.read_bytes(), journal)

    async def test_expired_time_window_cannot_be_restarted_for_dispatch(self):
        transport = self.transport()
        budget = transport.budget
        journal = budget.journal.read_bytes()
        with patch("time.monotonic", return_value=budget.deadline + 1), NetworkFence():
            with self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            with self.assertRaises(AttributeError):
                budget.started = budget.deadline + 1
            with self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
        self.assertEqual(self.delegated, 0)
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.journal.read_bytes(), journal)

    async def test_http_exception_is_only_exact_127_8009(self):
        for endpoint in ("http://127.0.0.1:8009", "http://127.0.0.1:8009/"):
            endpoint_ok(endpoint)
            self.assertTrue(is_tunnel(endpoint))
        for endpoint in ("http://127.0.0.1:8260", "http://127.0.0.1:443", "http://127.0.0.2:8009",
                         "http://localhost:8009", "http://106.53.0.235:8009", "http://example.invalid:8009",
                         "http://127.0.0.1:8009/models", "http://127.0.0.1:8009?x=1", "http://user@127.0.0.1:8009"):
            with self.subTest(endpoint=endpoint), self.assertRaises(Rejected):
                endpoint_ok(endpoint)

    async def test_tunnel_transport_uses_8009_without_public_dns(self):
        endpoint = "http://127.0.0.1:8009"
        self.payload.update(model="kev-latest", max_output_tokens=512)
        budget = self.budget(endpoint=endpoint, output_cap_requested=True)
        transport = GuardedTransport(HttpsTransport(), budget, self.payload, endpoint)
        async def boundary(request):
            self.assertEqual(str(request.url), endpoint + "/v1/systemone")
            self.assertEqual(json.loads(request.content)["model"], "kev-latest")
            self.assertEqual(budget.requests, 1)
            sys.audit("socket.connect", None, ("127.0.0.1", 8009))  # Synthetic, no syscall.
            for address in (("127.0.0.1", 443), ("127.0.0.1", 8260), ("127.0.0.2", 8009)):
                with self.assertRaises(Rejected):
                    sys.audit("socket.connect", None, address)
            return httpx2.Response(200, json={"answers": {"probe": {"noul": 0.9}}, "latency_ms": 17.3})
        with patch.object(socket, "getaddrinfo", side_effect=AssertionError("unexpected DNS")) as dns, patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", side_effect=boundary) as sender, NetworkFence() as fence:
            response = await transport.handle_async_request(self.request(url=endpoint + "/v1/systemone"))
            self.assertEqual(response.status_code, 200)
            dns.assert_not_called()
            sender.assert_awaited_once()
            self.assertEqual(fence.actual_socket_network_calls, 1)
        await transport.aclose()

    async def test_https_constructor_fixes_tls_proxy_and_retries(self):
        with patch.object(httpx2.AsyncHTTPTransport, "__init__", return_value=None) as initialize:
            HttpsTransport()
        initialize.assert_called_once_with(verify=True, trust_env=False, retries=0)

    async def test_https_boundary_scopes_network_after_reservation(self):
        self.payload["max_output_tokens"] = 512
        budget = self.budget(output_cap_requested=True)
        transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        async def boundary(request):
            self.assertEqual(budget.requests, 1)
            self.assertEqual(json.loads(request.content)["max_output_tokens"], 512)
            sys.audit("socket.getaddrinfo", "synthetic.example.invalid", 443, 0, 0, 0)
            sys.audit("socket.connect", None, ("93.184.216.34", 443))  # Audit only, never a real syscall.
            for address in (("127.0.0.1", 443), ("93.184.216.34", 8260), ("8.8.8.8", 443)):
                with self.assertRaises(Rejected):
                    sys.audit("socket.connect", None, address)
            errors = []
            def worker():
                try:
                    sys.audit("socket.connect", None, ("93.184.216.34", 443))
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=worker, daemon=True)
            thread.start(); thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], Rejected)
            return httpx2.Response(200, json={"ok": True})
        with patch.object(socket, "getaddrinfo", return_value=answers) as resolver, patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", side_effect=boundary) as sender, NetworkFence():
            self.assertEqual((await transport.handle_async_request(self.request())).status_code, 200)
            sender.assert_awaited_once()
            resolver.assert_called_once_with("synthetic.example.invalid", 443, 0, socket.SOCK_STREAM)
            with self.assertRaises(Rejected):
                sys.audit("socket.connect", None, ("93.184.216.34", 443))  # Permission is revoked.
            with self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            sender.assert_awaited_once()
        await transport.aclose()

    async def test_approval_expiry_during_dns_prevents_http_and_preserves_reservation(self):
        self.payload["max_output_tokens"] = 512
        for aligned in (False, True):
            now = {"wall": 5000.0, "mono": 1000.0}
            with self.subTest(aligned_clocks=aligned), patch("time.time", side_effect=lambda: now["wall"]), patch("time.monotonic", side_effect=lambda: now["mono"]):
                budget = self.budget(expires_at=5001.0, output_cap_requested=True)
                transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
                def resolve(*unused):
                    now["wall"] = 5001.0
                    if aligned:
                        now["mono"] = 1001.0  # Batch deadline is still 1030.
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
                with patch.object(socket, "getaddrinfo", side_effect=resolve), patch.object(httpx2.AsyncHTTPTransport, "handle_async_request") as sender, NetworkFence() as fence, self.assertRaises(Rejected):
                    await transport.handle_async_request(self.request())
                sender.assert_not_called()
                self.assertEqual(budget.requests, 1)
                self.assertEqual(len(budget.journal.read_text().splitlines()), 2)
                self.assertEqual(fence.actual_socket_network_calls, 0)
                self.assertIsNone(fence._dispatch_thread)
                self.assertEqual(fence._approval_expires, 0)
                await transport.aclose()

    async def test_each_connect_and_dns_rechecks_human_expiry(self):
        self.payload["max_output_tokens"] = 512
        now = {"wall": 5000.0}
        with patch("time.time", side_effect=lambda: now["wall"]):
            budget = self.budget(expires_at=5001.0, output_cap_requested=True)
            transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
            dns_rejected = []
            async def boundary(request):
                sys.audit("socket.connect", None, ("93.184.216.34", 443))  # Valid owner before expiry.
                now["wall"] = 5001.0
                try:
                    sys.audit("socket.getaddrinfo", "synthetic.example.invalid", 443, 0, 0, 0)
                except Rejected:
                    dns_rejected.append(True)
                sys.audit("socket.connect", None, ("93.184.216.34", 443))  # Must reject on the connection boundary itself.
                raise AssertionError("expired connect was authorized")
            addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
            with patch.object(socket, "getaddrinfo", return_value=addresses), patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", side_effect=boundary) as sender, NetworkFence() as fence, self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            sender.assert_awaited_once()
            self.assertEqual(dns_rejected, [True])
            self.assertEqual(fence.actual_socket_network_calls, 1)  # One synthetic audit before expiry, none after.
            self.assertEqual(budget.requests, 1)
            await transport.aclose()

    async def test_copied_context_to_thread_cannot_gain_tcp_permission(self):
        self.payload["max_output_tokens"] = 512
        budget = self.budget(output_cap_requested=True)
        transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
        def connect_audit():
            sys.audit("socket.connect", None, ("93.184.216.34", 443))
        async def boundary(request):
            with self.assertRaises(Rejected):
                await asyncio.to_thread(connect_audit)
            copied = copy_context()
            with self.assertRaises(Rejected):
                await asyncio.get_running_loop().run_in_executor(None, copied.run, connect_audit)
            connect_audit()  # The original dispatch thread is still permitted.
            return httpx2.Response(200, json={"ok": True})
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch.object(socket, "getaddrinfo", return_value=addresses), patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", side_effect=boundary), NetworkFence() as fence:
            self.assertEqual((await transport.handle_async_request(self.request())).status_code, 200)
            self.assertEqual(fence.actual_socket_network_calls, 1)
        await transport.aclose()

    async def test_dns_worker_can_only_resolve_approved_host_not_connect(self):
        self.payload["max_output_tokens"] = 512
        transport = GuardedTransport(HttpsTransport(), self.budget(output_cap_requested=True), self.payload, ENDPOINT)
        def resolve(host, port, *unused):
            sys.audit("socket.getaddrinfo", host, port, 0, 0, 0)
            with self.assertRaises(Rejected):
                sys.audit("socket.connect", None, ("93.184.216.34", 443))
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch.object(socket, "getaddrinfo", side_effect=resolve), patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", return_value=httpx2.Response(200, json={"ok": True})), NetworkFence() as fence:
            self.assertEqual((await transport.handle_async_request(self.request())).status_code, 200)
            self.assertEqual(fence.actual_socket_network_calls, 0)
        await transport.aclose()

    async def test_https_private_dns_address_never_reaches_http(self):
        self.payload["max_output_tokens"] = 512
        budget = self.budget(output_cap_requested=True)
        transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch.object(socket, "getaddrinfo", return_value=answers), patch.object(httpx2.AsyncHTTPTransport, "handle_async_request") as sender, NetworkFence(), self.assertRaises(Rejected):
            await transport.handle_async_request(self.request())
        sender.assert_not_called()
        self.assertEqual(budget.requests, 1)  # Failed reserved batch is not refunded/reused.
        await transport.aclose()

    async def test_requested_output_cap_missing_or_changed_rejected_before_dns(self):
        for cap in (None, 32, True):
            self.payload["max_output_tokens"] = cap
            budget = self.budget(output_cap_requested=True)
            transport = GuardedTransport(HttpsTransport(), budget, self.payload, ENDPOINT)
            with patch.object(socket, "getaddrinfo") as resolver, NetworkFence(), self.assertRaises(Rejected):
                await transport.handle_async_request(self.request())
            resolver.assert_not_called()
            self.assertEqual(budget.requests, 0)
            await transport.aclose()

    async def test_arbitrary_tls_disabled_transport_is_not_accepted(self):
        inner = httpx2.AsyncHTTPTransport(verify=False, retries=0)
        transport = GuardedTransport(inner, self.budget(), self.payload, ENDPOINT)
        with NetworkFence(), self.assertRaises(Rejected):
            await transport.handle_async_request(self.request())
        self.assertEqual(transport.budget.requests, 0)
        await transport.aclose()

    async def test_real_inner_none_refused_before_network(self):
        transport = GuardedTransport(None, self.budget(), self.payload, ENDPOINT)
        with NetworkFence() as fence, self.assertRaises(Rejected) as caught:
            await transport.handle_async_request(self.request())
        self.assert_redacted(caught.exception)
        self.assertEqual(transport.attempted_dispatch, 1)
        self.assertEqual(fence.attempted_network, 0)
        self.assertEqual(fence.actual_socket_network_calls, 0)


if __name__ == "__main__":
    unittest.main()
