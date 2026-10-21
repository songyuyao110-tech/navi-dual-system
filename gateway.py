from __future__ import annotations

import hashlib
import json
import math

from typesafe_sdk import AsyncTypeSafeClient, NoulAnswer

from decisions import Tool, mixed_questions, run_b_decisions, select_mixed, serialize_results
from registry import RegistrySnapshot


def _catalog_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _memory_pool(memories: list | tuple) -> None:
    if not isinstance(memories, (list, tuple)):
        raise ValueError("Memory pool must be a sequence")
    for entry in memories:
        if (not isinstance(entry, dict) or set(entry) != {"id", "summary", "content", "source", "valid"}
                or entry["source"] not in {"synthetic_fixture", "user_approved"} or entry["valid"] is not True):
            raise ValueError("Memory pool contains an unauthorized/invalid entry")
        if not isinstance(entry["summary"], str) or not entry["summary"].strip():
            raise ValueError("Memory summary must be nonempty")
        content = entry["content"]
        if type(content) not in (str, dict, list) or not content or (isinstance(content, str) and not content.strip()):
            raise ValueError("Memory content must be nonempty JSON data")
    _catalog_hash(list(memories))  # Reject non-JSON/NaN without filling or repairing content.


def _selection_catalog(catalog: str) -> dict:
    from eval.validate_reopt import FROZEN_CONTRACT_VERSION
    bundle = json.loads(catalog)
    if not isinstance(bundle, dict) or set(bundle) != {"contract", "memories", "contract_version", "registry_version", "memory_version"}:
        raise ValueError("Invalid selection catalog")
    contract = bundle["contract"]
    version = _catalog_hash({k: v for k, v in contract.items() if k != "contract_version"})
    if (version != FROZEN_CONTRACT_VERSION or contract["contract_version"] != version
            or bundle["contract_version"] != version
            or bundle["registry_version"] != contract["source_sha256"]["tools.yaml"][:16]):
        raise ValueError("Mixed contract/registry version")
    if any(t["semantics_version"] != contract["semantics_version"] for t in contract["tools"]):
        raise ValueError("Mixed semantic side-table version")
    _memory_pool(bundle["memories"])
    if bundle["memory_version"] != _catalog_hash(bundle["memories"]):
        raise ValueError("Mixed memory version")
    mixed_questions(tuple(Tool(t["name"], t["description"]) for t in contract["tools"]), tuple(bundle["memories"]))
    return bundle


def freeze_selection_catalog(snapshot: RegistrySnapshot, contract: dict, memories: tuple[dict, ...]) -> str:
    """Capture existing registry + frozen semantic table + full legal memory as JSON bytes."""
    _memory_pool(memories)
    if set(snapshot.tools) != {t["name"] for t in contract["tools"]}:
        raise ValueError("Registry/catalog tool identifiers differ")
    for entry in contract["tools"]:
        tool = snapshot.tools[entry["name"]]
        if (any(entry[k] != getattr(tool, k) for k in ("name", "description", "domain", "binding", "side_effect"))
                or entry["parameters"] != tool.parameters):
            raise ValueError("Registry schema/metadata differ from contract")
    bundle = {"contract": contract, "memories": list(memories), "contract_version": contract["contract_version"],
              "registry_version": snapshot.version, "memory_version": _catalog_hash(list(memories))}
    catalog = json.dumps(bundle, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    _selection_catalog(catalog)
    return catalog  # No mutable reference to registry, schema or caller-owned memory survives.


def prepare_reopt(catalog: str, utterance: str, context: dict) -> dict:
    bundle = _selection_catalog(catalog)
    if not isinstance(utterance, str) or not utterance.strip() or not isinstance(context, dict):
        raise ValueError("Selection requires utterance and fixed public context")
    public_context = json.loads(json.dumps(context, ensure_ascii=False, allow_nan=False))
    tools = bundle["contract"]["tools"]
    return {**{k: bundle[k] for k in ("contract_version", "registry_version", "memory_version")},
            "state": {"utterance": utterance, "context": public_context, "tools": tools, "memories": bundle["memories"]},
            "questions": mixed_questions(tuple(Tool(t["name"], t["description"]) for t in tools), tuple(bundle["memories"]))}


def select_reopt(catalog: str, probabilities: dict, *, tool_threshold: float = 0.5,
                 memory_threshold: float = 0.5, max_selected: int | None = None) -> dict:
    bundle = _selection_catalog(catalog)
    questions = mixed_questions(tuple(Tool(t["name"], t["description"]) for t in bundle["contract"]["tools"]), tuple(bundle["memories"]))
    result = select_mixed(questions, probabilities, tool_threshold=tool_threshold,
                          memory_threshold=memory_threshold, max_selected=max_selected)
    return {**result, **{k: bundle[k] for k in ("contract_version", "registry_version", "memory_version")}}


def assemble_reopt(catalog: str, selection: dict) -> dict:
    """Only selected schemas/semantics/memory; no private oracle, closure or fallback input."""
    bundle = _selection_catalog(catalog)
    try:
        expected = select_reopt(catalog, selection["probabilities"], tool_threshold=selection["thresholds"]["tool"],
                                memory_threshold=selection["thresholds"]["memory"], max_selected=selection["max_selected"])
    except (KeyError, TypeError):
        raise ValueError("Malformed selection") from None
    if selection != expected:
        raise ValueError("Selection identifiers or versions were changed")
    if selection["status"] == "capacity_failure":
        raise ValueError("Cannot assemble a capacity failure")
    tools = [t for t in bundle["contract"]["tools"] if t["name"] in selection["selected_tool_ids"]]
    memories = [m for m in bundle["memories"] if m["id"] in selection["selected_memory_ids"]]
    return {**{k: bundle[k] for k in ("contract_version", "registry_version", "memory_version")},
            "status": selection["status"], "selected_tool_ids": list(selection["selected_tool_ids"]),
            "selected_memory_ids": list(selection["selected_memory_ids"]),
            "schemas": [{"name": t["name"], "description": t["description"], "parameters": json.loads(json.dumps(t["parameters"], allow_nan=False))} for t in tools],
            "tool_contracts": tools, "memories": memories}


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
