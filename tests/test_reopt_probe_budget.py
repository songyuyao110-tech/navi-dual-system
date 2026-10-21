"""Synthetic approval/budget checks; no transport, credentials, or live provider."""

from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from eval.reopt_probe_support import Budget, Limits, Rejected, approval_digest, check_approval


class ProbeBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="reopt-probe-budget-", dir="/Users/YUYSONG/Work/tmp"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.expires_at = time.time() + 3600
        self.binding = {
            "code_sha": "b" * 40,
            "manifest_sha256": "c" * 64,
            "source_sha256": "f" * 64,
            "execution_scope_hash": "1" * 64,
            "acceptance_scope_hash": "2" * 64,
            "endpoint": "https://synthetic.example.invalid/v1/chat/completions",
            "model": "synthetic-model",
            "batch_id": "synthetic-batch",
            "session": "synthetic-session",
            "data_sha256": "d" * 64,
        }
        self.limits = Limits(input_usd_per_million=1.0,
                             output_usd_per_million=2.0,
                             provider_caps_verified=True)
        self.human_sha = approval_digest(self.binding, self.limits, self.expires_at)
        self.approval = self.document()

    def document(self, limits=None):
        return {**self.binding, "human_approval_sha256": self.human_sha,
                "expires_at": self.expires_at,
                "limits": asdict(limits or self.limits)}

    def check(self, doc=None, limits=None, human_sha=None):
        return check_approval(self.approval if doc is None else doc,
                              self.binding, limits or self.limits,
                              human_sha=self.human_sha if human_sha is None else human_sha)

    def budget(self, limits=None):
        limits = limits or self.limits
        return Budget(self.root, self.binding["batch_id"], self.binding,
                      limits, self.document(limits), self.human_sha)

    def authorize(self, limits):
        self.limits = limits
        self.human_sha = approval_digest(self.binding, limits, self.expires_at)
        self.approval = self.document(limits)

    def waived_unknown_limits(self, **changes):
        return replace(self.limits, input_usd_per_million=None,
                       output_usd_per_million=None, price_waived=True,
                       output_cap_requested=True, provider_caps_verified=False,
                       **changes)

    def test_valid_approval_and_limits_are_frozen(self):
        self.check()
        self.assertAlmostEqual(self.limits.cost, (8192 + 512 * 2) / 1_000_000)
        with self.assertRaises(FrozenInstanceError):
            self.limits.max_requests = 2

    def test_every_binding_mismatch_or_missing_field_rejected(self):
        for field in self.binding:
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    doc = deepcopy(self.approval)
                    if missing:
                        del doc[field]
                    else:
                        doc[field] = "different-synthetic-value"
                    with self.assertRaises(Rejected):
                        self.check(doc)

    def test_human_reference_required_and_exact_hash(self):
        for value in (None, "", "a" * 40, "z" * 64, "e" * 64):
            with self.subTest(value=value):
                doc = deepcopy(self.approval)
                doc["human_approval_sha256"] = value
                with self.assertRaises(Rejected):
                    self.check(doc)
        for known in (None, "", "a" * 40, "z" * 64, "e" * 64):
            with self.subTest(known=known), self.assertRaises(Rejected):
                check_approval(self.approval, self.binding, self.limits, human_sha=known)
        doc = deepcopy(self.approval)
        del doc["human_approval_sha256"]
        doc["approved"] = True
        with self.assertRaises(Rejected):
            self.check(doc)

    def test_rejection_does_not_echo_untrusted_approval_value(self):
        marker = "synthetic-private-approval-value-983714"
        doc = deepcopy(self.approval)
        doc["model"] = marker
        with self.assertRaises(Rejected) as caught:
            self.check(doc)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertNotIn(marker, str(caught.exception))

    def test_expiry_missing_expired_or_nonfinite_rejected(self):
        for expiry in (None, 0, -1, time.time() - 1,
                       float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(expiry=expiry):
                doc = deepcopy(self.approval)
                doc["expires_at"] = expiry
                with self.assertRaises(Rejected):
                    self.check(doc)
        doc = deepcopy(self.approval)
        del doc["expires_at"]
        with self.assertRaises(Rejected):
            self.check(doc)
        with self.assertRaises(Rejected):
            check_approval(self.approval, self.binding, self.limits,
                           human_sha=self.human_sha, now=self.approval["expires_at"])

    def test_approval_limits_must_match_exactly(self):
        for field, value in asdict(self.limits).items():
            with self.subTest(field=field):
                doc = deepcopy(self.approval)
                doc["limits"][field] = (not value if isinstance(value, bool)
                                        else 1 if value is None else value + 1)
                with self.assertRaises(Rejected):
                    self.check(doc)
        for value in (None, {}, {"max_requests": 1}):
            doc = deepcopy(self.approval)
            doc["limits"] = value
            with self.assertRaises(Rejected):
                self.check(doc)

    def test_expiry_extension_under_same_external_content_anchor_rejected(self):
        trusted_expiry = 946684800
        anchor = approval_digest(self.binding, self.limits, trusted_expiry)
        trusted = {**self.document(), "expires_at": trusted_expiry,
                   "human_approval_sha256": anchor}
        with self.assertRaises(Rejected):
            check_approval(trusted, self.binding, self.limits, anchor)
        extended = {**trusted, "expires_at": self.expires_at}
        with self.assertRaises(Rejected):
            check_approval(extended, self.binding, self.limits, anchor)
        extended["human_approval_sha256"] = approval_digest(
            self.binding, self.limits, extended["expires_at"])
        with self.assertRaises(Rejected):
            check_approval(extended, self.binding, self.limits, anchor)

    def test_future_expiry_extension_and_embedded_rehash_rejected(self):
        for rehash in (False, True):
            with self.subTest(rehash=rehash):
                candidate = deepcopy(self.approval)
                candidate["expires_at"] += 3600
                if rehash:
                    candidate["human_approval_sha256"] = approval_digest(
                        self.binding, self.limits, candidate["expires_at"])
                with self.assertRaises(Rejected):
                    self.check(candidate)

    def test_swapped_trusted_binding_content_cannot_reuse_external_anchor(self):
        for field in self.binding:
            with self.subTest(field=field):
                swapped = {**self.binding, field: "different-synthetic-value"}
                candidate = {**self.approval, **swapped}
                with self.assertRaises(Rejected):
                    check_approval(candidate, swapped, self.limits, self.human_sha)

    def test_swapped_trusted_limits_cannot_reuse_external_anchor(self):
        swapped = replace(self.limits, input_usd_per_million=3.0)
        with self.assertRaises(Rejected):
            self.check(self.document(swapped), swapped)

    def test_extra_document_fields_are_not_authorized_content(self):
        candidate = {**self.approval, "approved": True, "synthetic_note": "not-authorized"}
        binding = {**self.binding, "synthetic_note": "different-note"}
        self.assertEqual(approval_digest(candidate, self.limits, self.expires_at), self.human_sha)
        self.assertEqual(approval_digest(binding, self.limits, self.expires_at), self.human_sha)
        check_approval(candidate, binding, self.limits, self.human_sha)

    def test_missing_zero_negative_or_nonfinite_prices_rejected(self):
        for field in ("input_usd_per_million", "output_usd_per_million"):
            for value in (None, 0, -1, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(Rejected):
                    limits = replace(self.limits, **{field: value})
                    self.check(self.document(limits), limits)
        with self.assertRaises(Rejected):
            _ = Limits().cost

    def test_invalid_budgets_and_unverified_provider_caps_rejected(self):
        for field in ("max_requests", "max_input_tokens", "max_output_tokens", "timeout"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(Rejected):
                    limits = replace(self.limits, **{field: value})
                    self.check(self.document(limits), limits)
        for changes in ({"provider_caps_verified": False}, {"max_requests": 2}, {"timeout": 31}):
            with self.subTest(changes=changes), self.assertRaises(Rejected):
                limits = replace(self.limits, **changes)
                self.check(self.document(limits), limits)

    def test_invalid_approval_creates_no_batch(self):
        doc = deepcopy(self.approval)
        doc["model"] = "different-synthetic-model"
        with self.assertRaises(Rejected):
            Budget(self.root, self.binding["batch_id"], self.binding,
                   self.limits, doc, self.human_sha)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_reserve_is_durable_and_charges_worst_case_before_transport(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        with patch("os.fsync", wraps=os.fsync) as sync:
            budget.reserve(1, 1)
        self.assertGreater(sync.call_count, 0)
        self.assertEqual(budget.requests, 1)
        self.assertAlmostEqual(budget.cost_usd, self.limits.cost)
        self.assertNotEqual(budget.journal.read_bytes(), before)
        self.assertEqual(budget.journal.parent, self.root / self.binding["batch_id"])
        reservation = json.loads(budget.journal.read_text().splitlines()[-1])
        self.assertEqual(reservation["reserved_request"], 1)
        self.assertEqual(reservation["input_tokens"], self.limits.max_input_tokens)
        self.assertEqual(reservation["output_tokens"], self.limits.max_output_tokens)
        self.assertAlmostEqual(reservation["cost_usd"], self.limits.cost)

    def test_reservation_timeout_or_approval_expiry_fail_closed(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        with patch("time.monotonic", return_value=budget.started + self.limits.timeout):
            with self.assertRaises(Rejected):
                budget.reserve(1, 1)
        with patch("time.time", return_value=budget.approval["expires_at"]):
            with self.assertRaises(Rejected):
                budget.reserve(1, 1)
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.cost_usd, 0)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_expired_batch_cannot_reset_started_or_deadline(self):
        with patch("time.monotonic", return_value=100.0):
            budget = self.budget()
        self.assertEqual(budget.started, 100.0)
        self.assertEqual(budget.deadline, 130.0)
        before = budget.journal.read_bytes()
        with patch("time.monotonic", return_value=131.0):
            with self.assertRaisesRegex(Rejected, "batch_expired"):
                budget.reserve(1, 1)
            for field, value in (("started", 131.0), ("deadline", 161.0)):
                with self.subTest(field=field), self.assertRaises(AttributeError):
                    setattr(budget, field, value)
            with self.assertRaisesRegex(Rejected, "batch_expired"):
                budget.reserve(1, 1)
        self.assertEqual(budget.started, 100.0)
        self.assertEqual(budget.deadline, 130.0)
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.cost_usd, 0)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_frozen_time_window_boundary_allows_only_before_deadline(self):
        with patch("time.monotonic", return_value=100.0):
            budget = self.budget()
        before = budget.journal.read_bytes()
        with patch("time.monotonic", return_value=130.0), self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.journal.read_bytes(), before)
        with patch("time.monotonic", return_value=129.999):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.deadline, 130.0)

    def test_request_n_plus_one_rejected_without_refund(self):
        budget = self.budget()
        budget.reserve(self.limits.max_input_tokens, self.limits.max_output_tokens)
        journal = budget.journal.read_bytes()
        # Transport failure/unknown is deliberately not a refund or retry permission.
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertAlmostEqual(budget.cost_usd, self.limits.cost)
        self.assertEqual(budget.journal.read_bytes(), journal)

    def test_invalid_tokens_do_not_consume_request_or_change_journal(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        invalid = (-1, True, False, float("nan"), float("inf"), 1.5, "1", None)
        for input_tokens, output_tokens in (
            *((value, 1) for value in invalid),
            *((1, value) for value in invalid),
            (self.limits.max_input_tokens + 1, 1),
            (1, self.limits.max_output_tokens + 1),
        ):
            with self.subTest(input=input_tokens, output=output_tokens):
                with self.assertRaises(Rejected):
                    budget.reserve(input_tokens, output_tokens)
                self.assertEqual(budget.requests, 0)
                self.assertEqual(budget.cost_usd, 0)
                self.assertEqual(budget.journal.read_bytes(), before)
        budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)

    def test_concurrent_reservations_exactly_one_success_finite_join(self):
        budget = self.budget()
        start = threading.Event()
        outcomes, lock = [], threading.Lock()

        def reserve():
            if not start.wait(2):
                outcome = TimeoutError("synthetic start timeout")
            else:
                try:
                    budget.reserve(1, 1)
                    outcome = None
                except Exception as error:
                    outcome = error
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=reserve, daemon=True) for _ in range(12)]
        for thread in threads:
            thread.start()
        start.set()
        deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(outcomes), len(threads))
        self.assertEqual(sum(outcome is None for outcome in outcomes), 1)
        self.assertTrue(all(outcome is None or isinstance(outcome, Rejected)
                            for outcome in outcomes))
        self.assertEqual(budget.requests, 1)
        self.assertAlmostEqual(budget.cost_usd, self.limits.cost)

    def test_existing_batch_rejected_even_after_crash_without_reservation(self):
        budget = self.budget()
        journal, before = budget.journal, budget.journal.read_bytes()
        del budget
        with self.assertRaises(Rejected):
            self.budget()
        self.assertEqual(journal.read_bytes(), before)

    def test_existing_batch_rejected_after_reservation_preserves_journal(self):
        budget = self.budget()
        budget.reserve(1, 1)
        journal, before = budget.journal, budget.journal.read_bytes()
        del budget
        with self.assertRaises(Rejected):
            self.budget()
        self.assertEqual(journal.read_bytes(), before)

    def test_public_consumption_counters_are_read_only(self):
        budget = self.budget()
        budget.reserve(1, 1)
        before = budget.journal.read_bytes()
        for field in ("requests", "cost_usd"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(budget, field, 0)
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.cost_usd, self.limits.cost)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_private_counter_drift_cannot_reset_durable_consumption(self):
        budget = self.budget()
        budget.reserve(1, 1)
        before = budget.journal.read_bytes()
        for requests, cost in ((0, self.limits.cost), (1, 0.0), (0, 0.0)):
            with self.subTest(requests=requests, cost=cost):
                budget._requests, budget._cost_usd = requests, cost
                with self.assertRaisesRegex(Rejected, "journal_state_mismatch"):
                    budget.reserve(1, 1)
                self.assertEqual(budget.journal.read_bytes(), before)
        budget._requests, budget._cost_usd = 1, self.limits.cost
        with self.assertRaisesRegex(Rejected, "budget_exhausted"):
            budget.reserve(1, 1)

    def test_approval_snapshot_tamper_rejected_before_consumption(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        original = deepcopy(budget.approval)
        for field, value in (("expires_at", self.expires_at + 3600),
                             ("model", "different-synthetic-model"),
                             ("human_approval_sha256", "e" * 64)):
            with self.subTest(field=field):
                budget.approval = {**original, field: value}
                with self.assertRaises(Rejected):
                    budget.reserve(1, 1)
                self.assertEqual(budget.requests, 0)
                self.assertEqual(budget.journal.read_bytes(), before)
        budget.approval = deepcopy(original)
        budget.approval["expires_at"] += 3600
        budget.approval["human_approval_sha256"] = approval_digest(
            budget.binding, budget.limits, budget.approval["expires_at"])
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_limits_snapshot_tamper_rejected_even_with_matching_document(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        budget.limits = replace(self.limits, max_input_tokens=16384)
        budget.approval["limits"] = asdict(budget.limits)
        with self.assertRaisesRegex(Rejected, "approval_content_mismatch"):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_binding_snapshot_tamper_rejected_even_with_matching_document(self):
        budget = self.budget()
        before = budget.journal.read_bytes()
        budget.binding["session"] = "different-synthetic-session"
        budget.approval["session"] = budget.binding["session"]
        with self.assertRaisesRegex(Rejected, "approval_content_mismatch"):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_constructor_copies_caller_owned_authorization_documents(self):
        budget = Budget(self.root, self.binding["batch_id"], self.binding,
                        self.limits, self.approval, self.human_sha)
        self.binding["model"] = "different-synthetic-model"
        self.approval["expires_at"] += 3600
        budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.binding["model"], "synthetic-model")
        self.assertEqual(budget.approval["expires_at"], self.expires_at)

    def test_journal_tamper_before_next_reservation_rejected(self):
        budget = self.budget()
        budget.reserve(1, 1)
        before = budget.journal.read_bytes()
        budget.journal.write_bytes(before + b'{"synthetic_tamper":true}\n')
        tampered = budget.journal.read_bytes()
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.journal.read_bytes(), tampered)

    def test_journal_truncation_before_next_reservation_rejected(self):
        budget = self.budget()
        budget.reserve(1, 1)
        budget.journal.write_bytes(b"")
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.journal.read_bytes(), b"")

    def test_missing_prices_without_explicit_waiver_rejected(self):
        for changes in ({}, {"input_usd_per_million": 1.0},
                        {"output_usd_per_million": 2.0}):
            with self.subTest(changes=changes):
                limits = Limits(output_cap_requested=True, **changes)
                with self.assertRaises(Rejected):
                    _ = limits.cost
                self.authorize(limits)
                with self.assertRaises(Rejected):
                    self.budget()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_waived_unknown_prices_return_none_and_journal_records_unknown(self):
        self.authorize(self.waived_unknown_limits())
        self.check()
        self.assertIsNone(self.limits.cost)
        self.assertIsNone(self.limits.max_cost_usd)
        self.assertFalse(self.limits.provider_caps_verified)
        self.assertTrue(self.limits.output_cap_requested)
        self.assertEqual(self.limits.max_output_tokens, 512)
        budget = self.budget()
        self.assertIsNone(budget.cost_usd)
        budget.reserve(1, 1)
        self.assertEqual(budget.requests, 1)
        self.assertIsNone(budget.cost_usd)
        rows = [json.loads(line) for line in budget.journal.read_text().splitlines()]
        self.assertIsNone(rows[-1]["cost_usd"])
        self.assertEqual(rows[-1]["input_tokens"], self.limits.max_input_tokens)
        self.assertEqual(rows[-1]["output_tokens"], 512)
        self.assertEqual(self.limits.price_status, "price unknown, user-waived")
        self.assertEqual(rows[-1]["price_status"], "price unknown, user-waived")
        self.assertFalse(rows[0]["limits"]["provider_caps_verified"])
        self.assertTrue(rows[0]["limits"]["output_cap_requested"])

    def test_partially_unknown_prices_with_waiver_return_none(self):
        original = self.limits
        for field in ("input_usd_per_million", "output_usd_per_million"):
            with self.subTest(field=field):
                limits = replace(original, **{field: None},
                                 price_waived=True, output_cap_requested=True)
                self.assertIsNone(limits.cost)
                self.authorize(limits)
                self.check()

    def test_waived_unknown_prices_require_requested_output_cap(self):
        limits = replace(self.waived_unknown_limits(), output_cap_requested=False)
        self.authorize(limits)
        with self.assertRaises(Rejected):
            _ = limits.cost
        with self.assertRaises(Rejected):
            self.budget()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_requested_output_cap_does_not_claim_provider_verification(self):
        limits = replace(self.limits, provider_caps_verified=False,
                         output_cap_requested=True)
        self.authorize(limits)
        self.check()
        self.assertAlmostEqual(limits.cost, (8192 + 512 * 2) / 1_000_000)
        self.assertFalse(limits.provider_caps_verified)
        self.assertEqual(asdict(limits)["max_output_tokens"], 512)
        self.assertFalse(asdict(limits)["provider_caps_verified"])

    def test_known_price_ceiling_is_optional_and_enforced_when_specified(self):
        expected = self.limits.cost
        for ceiling in (None, expected, expected + 1):
            with self.subTest(ceiling=ceiling):
                limits = replace(self.limits, max_cost_usd=ceiling)
                self.authorize(limits)
                self.check()
                self.assertEqual(limits.cost, expected)
        for ceiling in (0, expected / 2):
            with self.subTest(ceiling=ceiling):
                limits = replace(self.limits, max_cost_usd=ceiling)
                self.authorize(limits)
                with self.assertRaises(Rejected):
                    _ = limits.cost
                with self.assertRaises(Rejected):
                    self.check()

    def test_invalid_optional_cost_ceilings_rejected(self):
        for ceiling in (-1, float("nan"), float("inf"), -float("inf"),
                        True, False, "1"):
            with self.subTest(ceiling=ceiling):
                limits = replace(self.limits, max_cost_usd=ceiling)
                with self.assertRaises(Rejected):
                    _ = limits.cost

    def test_unknown_price_with_specified_ceiling_rejected_even_if_waived(self):
        for ceiling in (0, 1, 1000000):
            with self.subTest(ceiling=ceiling):
                limits = self.waived_unknown_limits(max_cost_usd=ceiling)
                self.authorize(limits)
                with self.assertRaises(Rejected):
                    _ = limits.cost
                with self.assertRaises(Rejected):
                    self.budget()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_price_rates_are_not_ignored_by_waiver(self):
        for field in ("input_usd_per_million", "output_usd_per_million"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf"),
                          True, False, "1"):
                for other_unknown in (False, True):
                    with self.subTest(field=field, value=value,
                                      other_unknown=other_unknown):
                        limits = self.waived_unknown_limits() if other_unknown else replace(
                            self.limits, price_waived=True, output_cap_requested=True)
                        limits = replace(limits, **{field: value})
                        with self.assertRaises(Rejected):
                            _ = limits.cost
        limits = replace(self.limits, input_usd_per_million=1e308,
                         price_waived=True, output_cap_requested=True)
        with self.assertRaises(Rejected):
            _ = limits.cost

    def test_price_waiver_cannot_be_toggled_after_approved_content_hash(self):
        original = self.limits
        for initially_waived in (False, True):
            for embedded_rehash in (False, True):
                with self.subTest(initially_waived=initially_waived,
                                  embedded_rehash=embedded_rehash):
                    self.authorize(replace(original, price_waived=initially_waived,
                                           output_cap_requested=True))
                    changed = replace(self.limits, price_waived=not initially_waived)
                    candidate = self.document(changed)
                    if embedded_rehash:
                        candidate["human_approval_sha256"] = approval_digest(
                            self.binding, changed, self.expires_at)
                    with self.assertRaises(Rejected):
                        self.check(candidate, changed)
        self.authorize(self.waived_unknown_limits())
        budget = self.budget()
        before = budget.journal.read_bytes()
        budget.limits = replace(budget.limits, price_waived=False)
        budget.approval["limits"] = asdict(budget.limits)
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.requests, 0)
        self.assertIsNone(budget.cost_usd)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_waived_unknown_same_batch_n_plus_one_and_reuse_fail_closed(self):
        self.authorize(self.waived_unknown_limits())
        budget = self.budget()
        budget.reserve(1, 1)
        before = budget.journal.read_bytes()
        with self.assertRaisesRegex(Rejected, "budget_exhausted"):
            budget.reserve(1, 1)
        with self.assertRaises(Rejected):
            self.budget()
        self.assertEqual(budget.requests, 1)
        self.assertIsNone(budget.cost_usd)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_waived_unknown_frozen_clock_and_approval_expiry_fail_closed(self):
        self.authorize(self.waived_unknown_limits())
        with patch("time.monotonic", return_value=100.0):
            budget = self.budget()
        before = budget.journal.read_bytes()
        for field, value in (("started", 131.0), ("deadline", 161.0)):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(budget, field, value)
        with patch("time.monotonic", return_value=130.0), self.assertRaises(Rejected):
            budget.reserve(1, 1)
        with patch("time.monotonic", return_value=129.999), patch(
                "time.time", return_value=self.expires_at), self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual((budget.started, budget.deadline), (100.0, 130.0))
        self.assertEqual(budget.requests, 0)
        self.assertIsNone(budget.cost_usd)
        self.assertEqual(budget.journal.read_bytes(), before)

    def test_waived_unknown_concurrency_count_and_read_only_counters_hold(self):
        self.authorize(self.waived_unknown_limits())
        budget = self.budget()
        start = threading.Event()
        outcomes, lock = [], threading.Lock()

        def reserve():
            if not start.wait(2):
                outcome = TimeoutError("synthetic start timeout")
            else:
                try:
                    budget.reserve(1, 1)
                    outcome = None
                except Exception as error:
                    outcome = error
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=reserve, daemon=True) for _ in range(12)]
        for thread in threads:
            thread.start()
        start.set()
        deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(outcomes), 12)
        self.assertEqual(sum(outcome is None for outcome in outcomes), 1)
        self.assertTrue(all(outcome is None or isinstance(outcome, Rejected)
                            for outcome in outcomes))
        before = budget.journal.read_bytes()
        for field, value in (("requests", 0), ("cost_usd", 0.0)):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(budget, field, value)
        self.assertEqual(budget.requests, 1)
        self.assertIsNone(budget.cost_usd)
        with self.assertRaises(Rejected):
            budget.reserve(1, 1)
        self.assertEqual(budget.journal.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
