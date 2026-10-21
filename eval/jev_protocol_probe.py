"""Checkpoint 2.1: one mixed Noul request to kev-9b / systemone.

The SSH tunnel uses direct HTTP without API credentials; HTTPS overrides retain
SDK compatibility. The documented response may omit model, usage and answer type.

LIVE requires an external human approval bound to the clean checkout and public
configuration. Prices may be explicitly user-waived; backend caps are not certified.
Prospective invocation (not an approval):
  python -m eval.jev_protocol_probe --mode live --approval APPROVAL --batch-id BATCH --human-approval-sha256 HUMAN_SHA256 --role dev --run-dir NEW_RUN --output NEW_RUN/result.json
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy
from typesafe_sdk.constants import API_KEY_ENV
from decisions import Tool, b_questions, validate_response
from jev_smoke import ENV_NAMES  # Reuse the existing project credential allowlist.
from eval.validate_reopt import validate
from eval.reopt_probe_support import (Budget, EVIDENCE, GuardedTransport, Limits, NetworkFence,
                                      HttpsTransport, Rejected, approval_digest, check_approval,
                                      checkout_identity, is_tunnel, payload_hash, resolve_project_config, sha256)


def build_request(manifest: Path) -> tuple[str, dict]:
    frozen = validate(manifest)["contract"]
    tools = tuple(Tool(t["name"], t["description"]) for t in frozen["tools"])
    questions = {"tool:" + name: question for name, question in b_questions(tools).items()}
    questions.update({"memory:home": Noul(instructions="Is synthetic Test Home A relevant to going home?"),
                      "memory:company": Noul(instructions="Is synthetic Test Company B relevant to going home?")})
    state = "Synthetic protocol fixture: navigate home; memory pool is Test Home A and Test Company B. No device execution."
    return state, questions


def _unique_response_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejected("duplicate_response_key")
        result[key] = value
    return result


async def sdk_probe(state: str, questions: dict, binding: dict, transport: GuardedTransport) -> dict:
    loggers = [logging.getLogger(name) for name in ("typesafe_sdk", "httpx2", "httpcore")]
    previous = [logger.disabled for logger in loggers]
    for logger in loggers:
        logger.disabled = True  # SDK bodies are not redacted by its own logging implementation.
    try:
        with NetworkFence() as fence:
            async with httpx2.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                         verify=True, timeout=30.0) as http_client:
                tunnel = is_tunnel(binding["endpoint"])
                if tunnel:
                    payload = {"model": binding["model"], "state": state,
                               "questions": {name: question.model_dump() for name, question in questions.items()},
                               "max_output_tokens": 512}
                    raw_response = await asyncio.wait_for(http_client.post(
                        binding["endpoint"].rstrip("/") + "/v1/systemone", json=payload), 30.0)
                else:
                    async with AsyncTypeSafeClient(http_client=http_client, base_url=binding["endpoint"],
                                                   model=binding["model"], retry=RetryPolicy(max_retries=0), timeout=30.0) as client:
                        response = await asyncio.wait_for(client.system_one(
                            state=state, questions=questions, extra_body={"max_output_tokens": 512}), 30.0)
                    raw_response = response.raw_http_response
            raw = json.loads(raw_response.content, object_pairs_hook=_unique_response_object)
            answers = raw.get("answers") if isinstance(raw, dict) else None
            if (not isinstance(answers, dict) or set(answers) != set(questions)
                    or any(not isinstance(answer, dict) or set(answer) - {"type", "noul"}
                           or ("type" in answer and answer["type"] != "noul")
                           or type(answer.get("noul")) not in (int, float)
                           or not math.isfinite(answer["noul"]) or not 0 <= answer["noul"] <= 1
                           for answer in answers.values())):
                raise Rejected("raw_answers_rejected")
            if not tunnel:
                validate_response(questions, response)
            if "model" in raw and raw["model"] != binding["model"]:
                raise Rejected("response_model_mismatch")
            usage = raw.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise Rejected("usage_rejected")
                for name, maximum in (("input_tokens", transport.budget.limits.max_input_tokens),
                                      ("output_tokens", transport.budget.limits.max_output_tokens)):
                    value = usage.get(name)
                    if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                        raise Rejected("usage_rejected")
                if not usage or all(value is None for value in usage.values()):
                    usage = None
            latency = raw.get("latency_ms")
            if "latency_ms" in raw and (type(latency) not in (int, float)
                                       or not math.isfinite(latency) or latency < 0):
                raise Rejected("latency_rejected")
            return {"model": binding["model"], "response_model": raw.get("model"),
                    "model_source": "response" if "model" in raw else "request",
                    "question_ids": sorted(questions),
                    "probabilities": {name: answer["noul"] for name, answer in answers.items()},
                    "usage": usage, "latency_ms": latency,
                    "attempted_dispatch": transport.attempted_dispatch,
                    "actual_socket_network_calls": fence.actual_socket_network_calls,
                    "attempted_network": fence.attempted_network}
    except Exception:
        raise Rejected("probe_rejected") from None
    finally:
        for logger, disabled in zip(loggers, previous):
            logger.disabled = disabled


def dry_run(manifest: Path, batch_id: str | None = None, *, role: str | None = None,
            run_dir: Path | None = None, env_file: Path | None = None) -> dict:
    state, questions = build_request(manifest)
    data = {"state": state, "questions": {name: question.model_dump() for name, question in questions.items()}}
    binding = checkout_identity(manifest, data, env_file=env_file)
    role = role or ("test" if ROOT.name.startswith("test-") else "dev")
    if role not in {"dev", "test"}:
        raise Rejected("role_rejected")
    role_path = EVIDENCE / "step-2" / binding["code_sha"] / role
    if role_path.is_symlink() or any(p.is_symlink() for p in role_path.parents if p.is_relative_to(EVIDENCE)):
        raise Rejected("run_scope_rejected")
    role_root = role_path.resolve()
    run = Path(run_dir).resolve() if run_dir is not None else role_root / ("dry-" + uuid.uuid4().hex)
    if (run.parent != role_root or not run.name
            or (run_dir is not None and Path(run_dir).is_symlink())):
        raise Rejected("run_scope_rejected")
    batch = batch_id or "step2-kev-dry-" + uuid.uuid4().hex
    binding.update(batch_id=batch, session=role + "-dry-run", execution_scope_hash=payload_hash(data),
                   acceptance_scope_hash=payload_hash({"ids": sorted(questions), "offline": True}))
    payload = {**data, "model": binding["model"], "max_output_tokens": 512}
    # Explicit synthetic in-memory fixture, not a persisted human/live approval.
    limits = replace(Limits(), input_usd_per_million=1.0, output_usd_per_million=2.0, provider_caps_verified=True, output_cap_requested=True)
    expires_at = time.time() + 60
    human_fixture = approval_digest(binding, limits, expires_at)  # Synthetic content, never live permission.
    approval = {**binding, "limits": asdict(limits), "human_approval_sha256": human_fixture, "expires_at": expires_at}
    role_root.mkdir(parents=True, exist_ok=True)
    run.mkdir(mode=0o700)  # Existing runs are never reused.
    budget = Budget(run / "mock-batches", batch, binding, limits, approval, human_fixture)

    tunnel = is_tunnel(binding["endpoint"])

    def mock_provider(request):
        body = {"answers": {name: {"noul": 0.9} for name in questions}, "latency_ms": 17.3}
        if not tunnel:
            body = {"model": binding["model"], "usage": {},
                    "answers": {name: {"type": "noul", "noul": 0.9} for name in questions}}
        return httpx2.Response(200, json=body)

    transport = GuardedTransport(httpx2.MockTransport(mock_provider), budget, payload, binding["endpoint"])
    if tunnel:
        result = asyncio.run(sdk_probe(state, questions, binding, transport))
    else:
        if API_KEY_ENV not in ENV_NAMES:
            raise Rejected("credential_source_rejected")
        with patch.dict(os.environ, {API_KEY_ENV: "synthetic-" + "offline-only"}):
            result = asyncio.run(sdk_probe(state, questions, binding, transport))
    result.update(mode="dry-run", simulation_only=True, cloud_requests=0, actual_cost_usd=0,
                  code_sha=binding["code_sha"], source_sha256=binding["source_sha256"],
                  workspace_modified=binding.get("workspace_modified", False),
                  manifest_sha256=binding["manifest_sha256"], data_sha256=binding["data_sha256"],
                  endpoint=binding["endpoint"], endpoint_host=urlsplit(binding["endpoint"]).hostname,
                  config_source=binding.get("config_source"),
                   auth_mode="ssh-tunnel" if tunnel else "api-key",
                   credential_variable=None if tunnel else API_KEY_ENV,
                  retry=0, timeout=30.0, max_input_tokens=Limits().max_input_tokens,
                  max_output_tokens=Limits().max_output_tokens, live_certified=False,
                  role=role, session=binding["session"], batch_id=batch, run_id=run.name, run_dir=str(run),
                  execution_scope_hash=binding["execution_scope_hash"], acceptance_scope_hash=binding["acceptance_scope_hash"],
                  journal_path=budget.journal.relative_to(run).as_posix(), journal_sha256=sha256(budget.journal.read_bytes()))
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    with (run / "dry-run.json").open("x") as stream:
        stream.write(encoded + "\n")
    return result


def live_preflight(manifest: Path, approval: Path | None, batch_id: str | None,
                   human_sha: str | None, *, role: str = "dev",
                   env_file: Path | None = None) -> tuple[dict, Limits, dict]:
    # Resolve public configuration only; never load credentials before approval.
    if approval is None or not batch_id or human_sha is None:
        raise Rejected("human_approval_missing")
    if role not in {"dev", "test"}:
        raise Rejected("role_rejected")
    state, questions = build_request(manifest)
    data = {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}}
    binding = checkout_identity(manifest, data, clean=True, env_file=env_file)
    path = approval.resolve()
    if (approval.is_symlink() or not path.is_relative_to(EVIDENCE.resolve())
            or any(p.is_symlink() for p in approval.parents if p.is_relative_to(EVIDENCE))):
        raise Rejected("approval_path_rejected")
    payload = {**data, "model": binding["model"], "max_output_tokens": 512}
    binding.update(batch_id=batch_id, session=role,
                   execution_scope_hash=payload_hash({"payload": payload, "retry": 0, "timeout": 30.0,
                                                      "max_input_tokens": 8192, "max_requests": 1}),
                   acceptance_scope_hash=payload_hash({"ids": sorted(questions), "offline": False,
                                                       "output_cap_requested": True,
                                                       "provider_caps_verified": False}))
    limits = Limits(price_waived=True, output_cap_requested=True)
    doc = json.loads(path.read_text(), object_pairs_hook=_unique_response_object)
    check_approval(doc, binding, limits, human_sha)
    return binding, limits, doc


def live_run(manifest: Path, approval: Path | None, batch_id: str | None,
             human_sha: str | None, *, role: str = "dev", run_dir: Path | None = None,
             env_file: Path | None = None) -> dict:
    binding, limits, doc = live_preflight(manifest, approval, batch_id, human_sha,
                                        role=role, env_file=env_file)
    role_path = EVIDENCE / "step-2" / binding["code_sha"] / role
    if (EVIDENCE.is_symlink() or role_path.is_symlink()
            or any(p.is_symlink() for p in role_path.parents if p.is_relative_to(EVIDENCE))):
        raise Rejected("run_scope_rejected")
    role_root = role_path.resolve()
    run = Path(run_dir).resolve() if run_dir is not None else role_root / ("live-" + uuid.uuid4().hex)
    if (run.parent != role_root or not run.name or run.exists()
            or (run_dir is not None and (Path(run_dir).is_symlink()
                or any(p.is_symlink() for p in Path(run_dir).parents if p.is_relative_to(EVIDENCE))))):
        raise Rejected("run_scope_rejected")
    tunnel = is_tunnel(binding["endpoint"])
    # SSH authenticates the tunnel; this branch never loads or inspects API keys.
    config = resolve_project_config(env_file, credentials=not tunnel)
    if any(config[k] != binding[k] for k in ("endpoint", "model")):
        raise Rejected("configuration_changed")
    if not tunnel and (API_KEY_ENV not in ENV_NAMES or not os.environ.get(API_KEY_ENV)):
        raise Rejected("credential_source_rejected")
    state, questions = build_request(manifest)
    data = {"state": state, "questions": {k: v.model_dump() for k, v in questions.items()}}
    payload = {**data, "model": binding["model"], "max_output_tokens": 512}
    role_root.mkdir(parents=True, exist_ok=True)
    run.mkdir(mode=0o700)
    # The fixed shared batch root, not the role/run directory, prevents reuse.
    budget = Budget(EVIDENCE / "batches", batch_id, binding, limits, doc, human_sha)
    transport = GuardedTransport(HttpsTransport(), budget, payload, binding["endpoint"])
    try:
        result = asyncio.run(sdk_probe(state, questions, binding, transport))
        outcome = "accepted"
    except Rejected:
        result = {"usage": None, "response_model": None, "model_source": "request", "latency_ms": None,
                  "attempted_dispatch": transport.attempted_dispatch,
                  "actual_socket_network_calls": None, "attempted_network": None}
        outcome = "rejected"
    result.update(mode="live", outcome=outcome, simulation_only=False,
                  cloud_requests=budget.requests if outcome == "accepted" else None,
                  reserved_requests=budget.requests, actual_cost_usd=None,
                  reserved_cost_usd=budget.cost_usd, pricing=limits.price_status,
                  price_waived=limits.price_waived, max_cost_usd=limits.max_cost_usd,
                  output_cap_requested=True, provider_caps_verified=False, live_certified=False,
                  code_sha=binding["code_sha"], source_sha256=binding["source_sha256"],
                  manifest_sha256=binding["manifest_sha256"], data_sha256=binding["data_sha256"],
                  workspace_modified=binding.get("workspace_modified", False),
                  endpoint=binding["endpoint"], endpoint_host=urlsplit(binding["endpoint"]).hostname,
                  model=binding["model"], config_source=binding.get("config_source"),
                  auth_mode="ssh-tunnel" if tunnel else "api-key",
                   credential_variable=None if tunnel else API_KEY_ENV, retry=0, timeout=30.0,
                  max_input_tokens=8192, max_output_tokens=512, max_requests=1,
                  role=role, session=binding["session"], batch_id=batch_id,
                  run_id=run.name, run_dir=str(run), human_approval_sha256=human_sha,
                  execution_scope_hash=binding["execution_scope_hash"],
                  acceptance_scope_hash=binding["acceptance_scope_hash"],
                  journal_path=budget.journal.relative_to(EVIDENCE).as_posix(),
                  journal_sha256=sha256(budget.journal.read_bytes()))
    with (run / "live-run.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Rejected("arguments_rejected")


def main(argv=None) -> int:
    parser = Parser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "live"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/fixtures/reopt-manifest.json")
    parser.add_argument("--credentials", choices=("project-existing",), default="project-existing")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--human-approval-sha256")
    parser.add_argument("--max-requests", type=int, choices=(1,), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", choices=("dev", "test"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    try:
        args = parser.parse_args(argv)
        if args.manifest.is_symlink() or args.manifest.resolve() != (ROOT / "eval/fixtures/reopt-manifest.json").resolve():
            raise Rejected("manifest_path_rejected")
        output = None
        if args.output is not None:
            output = args.output.resolve()
            if (args.run_dir is None or args.output.is_symlink() or output.exists()
                    or output.parent != args.run_dir.resolve() or output.name in {"dry-run.json", "live-run.json", "mock-batches"}):
                raise Rejected("output_path_rejected")
        if args.mode == "live":
            result = live_run(args.manifest, args.approval, args.batch_id, args.human_approval_sha256,
                              role=args.role or "dev", run_dir=args.run_dir, env_file=args.env_file)
        else:
            result = dry_run(args.manifest, args.batch_id, role=args.role, run_dir=args.run_dir,
                             env_file=args.env_file)
        if output is not None:
            with output.open("x") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if result.get("outcome") == "rejected" else 0
    except Exception:
        print("kev-9b / systemone probe rejected (details redacted)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
