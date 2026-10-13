from __future__ import annotations

import json
import math

from typesafe_sdk import AsyncTypeSafeClient, NoulAnswer

from decisions import run_b_decisions, serialize_results
from registry import RegistrySnapshot


def disclosed_names(results, threshold: float) -> frozenset[str]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("B threshold must be a finite probability")
    names = set()
    for result in results:
        for name, answer in result.response.answers.items():
            if not isinstance(answer, NoulAnswer):
                raise ValueError("Unexpected B answer type")
            if answer.noul >= threshold:
                names.add(name)
    return frozenset(names)


async def additional_disclosure(
    client: AsyncTypeSafeClient, snapshot: RegistrySnapshot, brief: dict, intent: str,
    threshold: float = 0.5,
) -> frozenset[str]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("B threshold must be a finite probability")
    state = json.dumps(
        {"task": brief, "next_intent": intent}, ensure_ascii=False, allow_nan=False
    )
    results = await run_b_decisions(client, state, snapshot.domains())
    return disclosed_names(results, threshold)


async def judge(
    client: AsyncTypeSafeClient,
    utterance: str,
    snapshot: RegistrySnapshot,
    context: dict | None = None,
    b_threshold: float = 0.5,
) -> dict:
    """Disclose the all-domain B union; the route is fixed, not model-decided."""
    if not math.isfinite(b_threshold) or not 0 <= b_threshold <= 1:
        raise ValueError("B threshold must be a finite probability")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("Utterance must be nonempty")
    state = json.dumps(
        {"utterance": utterance, "context": context or {}},
        ensure_ascii=False,
        allow_nan=False,
    )
    results = await run_b_decisions(client, state, snapshot.domains())
    disclosed = disclosed_names(results, b_threshold)
    report = serialize_results(results)
    report["response_count"] = len(results)
    return {
        "execution_policy": "b_only_slow",
        "route": "slow",
        "registry_version": snapshot.version,
        "disclosed_tools": sorted(disclosed),
        "schemas": snapshot.schemas(disclosed),
        "thresholds": {"B": b_threshold},
        "decisions": report,
    }
