"""Step4 kev-9b/systemone candidate-envelope format probe, never tool execution.

Systemone chooses among supplied legal candidates; it does not generate arbitrary
JSON plans. The shared executor protocol is validated independently offline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import httpx2
from model_clients import ModelError, REOPT_ENVELOPE_SCHEMA, parse_reopt_envelope
from registry import load_registry
from eval.reopt_probe_support import (Budget, EVIDENCE, GuardedTransport, HttpsTransport, Limits,
                                      NetworkFence, is_tunnel, resolve_project_config, payload_hash, sha256)

BASE = "http://127.0.0.1:8009"
MODEL = "kev-latest"


def configuration(credentials=False):
    try:
        config = resolve_project_config(credentials=False)  # SSH tunnel never reads API keys.
    except ValueError:
        raise ModelError("Invalid kev tunnel configuration") from None
    # Accept the existing public slow-model names as aliases; TYPESAFE names win.
    config["endpoint"] = (os.environ.get("TYPESAFE_BASE_URL", "").strip()
                          or os.environ.get("LITELLM_BASE_URL", "").strip() or config["endpoint"]).rstrip("/")
    config["model"] = (os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip()
                       or os.environ.get("SLOW_MODEL", "").strip() or config["model"])
    if config["endpoint"] != BASE or config["model"] != MODEL or not is_tunnel(config["endpoint"]):
        raise ModelError("Step4 requires the approved kev-latest tunnel")
    return config


def fixture():
    snapshot = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
    ids = frozenset({"navi_start_route", "navi_get_status"})
    schemas = snapshot.schemas(ids)
    frozen = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
    contracts = [tool for tool in frozen["tools"] if tool["name"] in ids]
    brief = {"original_utterance": "Synthetic: plan navigation to Test Place B avoiding highways. Do not execute.",
             "independent_constraints": {"destination": "Test Place B", "route_option": "avoid_highway"},
             "selected_memories": [], "occurred_facts": [], "navigation_state": {"route_active": False},
             "contract_version": frozen["contract_version"]}
    return brief, schemas, contracts


def candidates():
    common = {"goal": {"destination": "Test Place B"}, "actions": [], "continue_after_results": False,
              "question": None, "reason": None}
    return {
        "actions": {**common, "kind": "actions", "actions": [{"tool": "navi_start_route", "arguments": {"end_name": "Test Place B", "route_option": "avoid_highway"}}]},
        "clarify": {**common, "kind": "clarify", "question": "Please confirm the synthetic destination."},
        "incomplete": {**common, "kind": "incomplete", "reason": "No suitable capability can complete this request."},
        "observe": {**common, "kind": "observe"},
    }


def request_body(brief, schemas, config):
    return {"model": config["model"], "state": json.dumps({"brief": brief, "schemas": schemas}, ensure_ascii=False, allow_nan=False),
            "questions": {"protocol_kind": {"type": "choice", "instructions": "Choose the appropriate predeclared envelope for the whole synthetic task. No execution or parameter generation.",
                                             "criteria": {kind: json.dumps(value, ensure_ascii=False) for kind, value in candidates().items()}}},
            "max_output_tokens": 512}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelError("Duplicate response key")
        result[key] = value
    return result


def decode_response(data, schemas, contracts, expected_model):
    if not isinstance(data, dict) or ("model" in data and data["model"] != expected_model):
        raise ModelError("Unexpected kev response model")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"protocol_kind"}:
        raise ModelError("Incomplete or extra kev questions")
    answer = answers["protocol_kind"]
    if (not isinstance(answer, dict) or ("type" in answer and answer["type"] != "choice")
            or set(answer) - {"type", "choice", "confidence", "probabilities"}
            or answer.get("choice") not in candidates() or not isinstance(answer.get("probabilities"), dict)
            or set(answer["probabilities"]) != set(candidates())):
        raise ModelError("Invalid kev Choice answer")
    values = [answer.get("confidence"), *answer["probabilities"].values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ModelError("Invalid kev probabilities")
    latency = data.get("latency_ms")
    if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
        raise ModelError("Invalid kev latency")
    usage = data.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise ModelError("Invalid kev usage")
        for name, cap in (("input_tokens", 8192), ("output_tokens", 512)):
            if name in usage and (type(usage[name]) is not int or not 0 <= usage[name] <= cap):
                raise ModelError("Invalid kev usage tokens")
    envelope = parse_reopt_envelope(candidates()[answer["choice"]], schemas, contracts)
    return {"model": expected_model, "response_model": data.get("model"), "envelope": envelope,
            "choice": answer["choice"], "probabilities": answer["probabilities"], "confidence": answer["confidence"],
            "latency_ms": latency, "usage": data.get("usage"), "executed_tools": 0, "format_valid": True,
            "probe_kind": "predeclared_envelope_choice", "generated_envelope": False, "auth_mode": "ssh-tunnel"}


def binding(manifest, body, batch):
    from eval.verify_delivery import _git, _index, _tree
    sha = _git(ROOT, "rev-parse", "HEAD").strip().decode()
    if _index(ROOT) != _tree(ROOT, sha) or _git(ROOT, "diff", "--name-only", "HEAD") or _git(ROOT, "ls-files", "--others", "-z"):
        raise ModelError("Live source must be clean")
    config = configuration()
    files = {p: sha256((ROOT / p).read_bytes()) for p in ("model_clients.py", "execution.py", "eval/slow_protocol_probe.py", "tests/test_reopt_protocol.py")}
    return {"code_sha": sha, "source_sha256": payload_hash(files), "manifest_sha256": sha256(manifest.read_bytes()),
            "data_sha256": payload_hash({"state": body["state"], "questions": body["questions"]}),
            "endpoint": config["endpoint"], "model": config["model"], "batch_id": batch, "session": "dev",
            "execution_scope_hash": payload_hash({"body": body, "request_count": 1, "no_execute": True}),
            "acceptance_scope_hash": payload_hash({"schema": REOPT_ENVELOPE_SCHEMA, "executed_tools": 0, "generated_envelope": False})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "live"), required=True)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--credentials", choices=("project-existing",), default="project-existing")
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/fixtures/reopt-manifest.json")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--human-approval-sha256")
    parser.add_argument("--max-requests", type=int, choices=(1,), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    status = None
    began = time.monotonic()
    try:
        brief, schemas, contracts = fixture()
        config = configuration();body = request_body(brief, schemas, config)
        for candidate in candidates().values():
            parse_reopt_envelope(candidate, schemas, contracts)
        if args.mode == "dry-run":
            raw = {"answers": {"protocol_kind": {"choice": "actions", "confidence": 0.9,
                    "probabilities": {"actions": 0.9, "clarify": 0.05, "incomplete": 0.03, "observe": 0.02}}}, "latency_ms": 0}
            with NetworkFence(), httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=raw)), trust_env=False) as client:
                response = client.post(config["endpoint"] + "/v1/systemone", json=body)
                status = response.status_code
                result = decode_response(json.loads(response.content, object_pairs_hook=_unique), schemas, contracts, config["model"])
            result.update(mode="dry-run", actual_network_requests=0, simulation_only=True)
        else:
            if not args.no_execute or args.approval is None or args.batch_id is None or args.human_approval_sha256 is None:
                raise ModelError("Explicit no-execute and approval required")
            b = binding(args.manifest, body, args.batch_id)
            limits = Limits(price_waived=True, output_cap_requested=True)
            budget = Budget(EVIDENCE / "batches", args.batch_id, b, limits,
                            json.loads(args.approval.read_text(), object_pairs_hook=_unique), args.human_approval_sha256)
            transport = GuardedTransport(HttpsTransport(), budget, body, config["endpoint"])
            async def call():
                import asyncio
                with NetworkFence():
                    async with httpx2.AsyncClient(transport=transport, trust_env=False, follow_redirects=False, timeout=30.0) as client:
                        return await asyncio.wait_for(client.post(config["endpoint"] + "/v1/systemone", json=body), 30.0)
            import asyncio
            response = asyncio.run(call());status = response.status_code
            if status != 200:
                raise ModelError("Unexpected HTTP status")
            result = decode_response(json.loads(response.content, object_pairs_hook=_unique), schemas, contracts, config["model"])
            result.update(mode="live", actual_network_requests=1, simulation_only=False, source_sha=b["code_sha"],
                          batch_id=args.batch_id, journal_path=str(budget.journal), journal_sha256=sha256(budget.journal.read_bytes()),
                          pricing="price unknown, user-waived", actual_cost_usd=None)
        result.update(endpoint=config["endpoint"], http_status=status, elapsed_ms=(time.monotonic()-began)*1000)
        if args.output:
            if not args.output.resolve().is_relative_to(EVIDENCE):
                raise ModelError("Output must be in evidence")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False));return 0
    except Exception:
        error = {"mode": args.mode, "outcome": "rejected", "http_status": status, "elapsed_ms": (time.monotonic()-began)*1000,
                 "executed_tools": 0, "retry": 0, "reason": "kev protocol probe rejected; details redacted"}
        if args.output and args.output.resolve().is_relative_to(EVIDENCE) and not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:json.dump(error, stream, indent=2);stream.write("\n")
        print(json.dumps(error));return 1


if __name__ == "__main__":
    raise SystemExit(main())
