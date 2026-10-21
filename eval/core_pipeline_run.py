"""C1/C2: text stand-in -> KEV scores -> top-k schemas -> one next-step decision."""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import statistics
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import httpx2
import yaml
from jsonschema import Draft202012Validator
from gateway import freeze_selection_catalog, prepare_reopt, select_reopt, assemble_reopt
from model_clients import (CONTROL_TOOLS, MULTIDOMAIN_CONTROLS, ModelError, slow_turn,
                           multidomain_disclosure, validate_multidomain_arguments)
from registry import load_registry
from decisions import Tool, mixed_questions
from result_writeback import COMPLETED_STEPS

KEV_BASE = "http://127.0.0.1:8009"
# EOT judging goes through its own SSH tunnel so it never queues behind tool scoring on KEV_BASE.
# Both local ports forward to the same remote KEV service (kev-latest @ 106.53.0.235:8009).
KEV_EOT_BASE = os.environ.get("KEV_EOT_BASE", "http://127.0.0.1:8010")
KEV_MODEL = "kev-latest"
# Shared across runs and fallback; retain the tunnel HTTP connection between queries.
KEV_CLIENT = httpx2.Client(timeout=30.0, trust_env=False, follow_redirects=False,
                           limits=httpx2.Limits(keepalive_expiry=60.0))
atexit.register(KEV_CLIENT.close)
MULTIDOMAIN_TOOLS = ROOT / "eval/fixtures/tools-multidomain.yaml"
DOMAIN_DESCRIPTIONS = ROOT / "eval/fixtures/kev-domain-descriptions.json"
# Stage 1 and tool batches reuse two connections; the default 14-tool pool is unchanged.
KEV_MD_CLIENT = httpx2.Client(timeout=30.0, trust_env=False, follow_redirects=False,
                              limits=httpx2.Limits(max_connections=2, max_keepalive_connections=2,
                                                  keepalive_expiry=60.0))
atexit.register(KEV_MD_CLIENT.close)
# Turn-taking (EOT) judging gets its own tunnel + connection pool so it never competes with tool
# scoring, and concurrent per-prefix EOT calls can be in flight without queueing behind each other.
KEV_EOT_CLIENT = httpx2.Client(timeout=30.0, trust_env=False, follow_redirects=False,
                               limits=httpx2.Limits(max_connections=4, max_keepalive_connections=4,
                                                   keepalive_expiry=60.0))
atexit.register(KEV_EOT_CLIENT.close)
TOOL_THRESHOLD = 0.5


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON key")
        result[name] = value
    return result


def kev_call(client, state, questions, base=KEV_BASE):
    response = client.post(base + "/v1/systemone", json={"model": KEV_MODEL, "state": state,
                          "questions": questions, "max_output_tokens": 512})
    if response.status_code != 200:
        raise ValueError("KEV returned non-200")
    raw = json.loads(response.content, object_pairs_hook=unique_object)
    if not isinstance(raw, dict) or ("model" in raw and raw["model"] != KEV_MODEL):
        raise ValueError("Wrong KEV response model")
    if not isinstance(raw.get("answers"), dict) or set(raw["answers"]) != set(questions):
        raise ValueError("Missing/extra KEV questions")
    return raw


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid KEV probability")
    return value


def score_multidomain(snapshot, utterance, context):
    groups = snapshot.domains()
    domain_items = json.loads(DOMAIN_DESCRIPTIONS.read_text())
    if len(snapshot.tools) != 200 or len(groups) != 20 or {t["name"] for t in domain_items} != set(groups):
        raise ValueError("Two-stage fixture requires 200 tools and the fixed 20 domain descriptions")
    origin = time.monotonic()

    def score_items(items):
        started = time.monotonic()
        # Identical to probe.payload(): no root domain tag, stage1 score or gold label.
        questions = {name: q.model_dump() for name, q in
                     mixed_questions(tuple(Tool(t["name"], t["description"]) for t in items), ()).items()}
        state = json.dumps({"utterance": utterance, "context": context, "tools": items},
                           ensure_ascii=False, allow_nan=False)
        raw = kev_call(KEV_MD_CLIENT, state, questions)
        scores = {}
        for name, answer in raw["answers"].items():
            if not isinstance(answer, dict) or set(answer) - {"type", "noul"} or answer.get("type", "noul") != "noul":
                raise ValueError("Malformed two-stage KEV Noul")
            scores[name] = probability(answer.get("noul"))
        finished = time.monotonic()
        return scores, {"tool_names": [t["name"] for t in items], "tool_count": len(items),
                        "latency_ms": (finished-started)*1000, "started_ms": (started-origin)*1000,
                        "finished_ms": (finished-origin)*1000, "model_latency_ms": raw.get("latency_ms")}

    stage1_scores, stage1_timing = score_items(domain_items)
    ranked_domains = [{"name": name[5:], "score": value} for name, value in
                      sorted(stage1_scores.items(), key=lambda item: (-item[1], item[0]))]
    recalled = [item for item in ranked_domains if item["score"] >= 0.4]
    fallback = not recalled
    if fallback:
        recalled = ranked_domains[:1]
    stage1_ms = (time.monotonic()-origin)*1000
    stage2_started = time.monotonic()
    definitions = [{"name": t.name, "domain": t.domain, "description": t.description,
                    "binding": t.binding, "side_effect": t.side_effect, "parameters": t.parameters}
                   for domain in recalled for t in snapshot.tools.values() if t.domain == domain["name"]]
    batches = [definitions[i:i+40] for i in range(0, len(definitions), 40)]
    if len(batches) == 1:
        results = [score_items(batches[0])]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(score_items, batches))
    scores = {name: value for values, _ in results for name, value in values.items()}
    if set(scores) != {"tool:" + t["name"] for t in definitions}:
        raise ValueError("Incomplete recalled-tool KEV scores")
    scores = dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
    stage2_ms = (time.monotonic()-stage2_started)*1000
    wall = (time.monotonic()-origin)*1000
    return scores, {"kev_scoring_strategy": "two_stage", "kev_wall_ms": wall, "kev_latency_ms": wall,
                    "kev_stage1_ms": stage1_ms, "kev_stage2_ms": stage2_ms,
                    "kev_domain_scores": {item["name"]: item["score"] for item in ranked_domains},
                    "recalled_domains": recalled, "domain_threshold": 0.4,
                    "domain_min1_fallback": fallback, "scored_tool_count": len(definitions),
                    "kev_stage1": stage1_timing, "kev_stage2_batches": [timing for _, timing in results],
                    "kev_stage2_concurrency": 1 if len(batches) == 1 else 2, "kev_http_pool_size": 2}


def decision_from_response(response, schemas, *, multidomain=False, native_schemas=None, utterance=None):
    if response.get("model") != os.environ["SLOW_MODEL"]:
        raise ValueError("Wrong decision response model")
    calls = [item for item in response["content"] if item.get("type") == "tool_use"]
    if len(calls) != 1:
        raise ValueError("C1 requires one parseable next step")
    call = calls[0]
    definitions = {tool["name"]: tool["parameters"] for tool in schemas}
    controls = MULTIDOMAIN_CONTROLS if multidomain else CONTROL_TOOLS
    definitions.update({tool["name"]: tool["input_schema"] for tool in controls if tool["name"] in {"ask_user", "finish"}})
    if call["name"] not in definitions:
        raise ValueError("Decision selected an undisclosed tool")
    Draft202012Validator(definitions[call["name"]]).validate(call["input"])
    decision = {"kind": "ask_user" if call["name"] == "ask_user" else "done" if call["name"] == "finish" else "tool_call",
                "tool": call["name"], "arguments": call["input"], "response_model": response["model"]}
    if multidomain and decision["kind"] == "tool_call":
        native = {t["name"]: t["parameters"] for t in native_schemas or []}
        if call["name"] not in native or not isinstance(utterance, str):
            raise ValueError("Multidomain selection requires native schemas and original utterance")
        references = validate_multidomain_arguments(call["name"], call["input"], native[call["name"]], utterance)
        decision.update(argument_mode="unresolved_intent" if references else "concrete_parameters",
                        unresolved_references=references, native_schema_valid=not references,
                        source_guards_passed=True, source_validation_scope="references_and_supplied_navigation_data",
                        execution_ready=False)
    return decision


def existing_slow_configuration():
    # httpx2 turns bracketed [::1] NO_PROXY into an invalid wildcard URL.
    # Keep the equivalent bare ::1 entry and every other proxy setting intact.
    for name in ("NO_PROXY", "no_proxy"):
        if name in os.environ:
            os.environ[name] = ",".join(host for host in os.environ[name].split(",") if host.strip() != "[::1]")
    os.environ.setdefault("LITELLM_BASE_URL", "https://litellm.swfcn.cn.bg.corpintra.net/v1")
    os.environ.setdefault("SLOW_MODEL", "deepseek-v4.1-flash")
    if os.environ["LITELLM_BASE_URL"].rstrip("/") != "https://litellm.swfcn.cn.bg.corpintra.net/v1" or os.environ["SLOW_MODEL"] != "deepseek-v4.1-flash":
        raise ValueError("Wrong existing decision endpoint/model")
    variable = "LITELLM_" + "API_KEY"
    if not os.environ.get(variable):
        doc = yaml.safe_load((Path.home() / ".dsh/.credentials.yaml").read_text())
        value = doc.get("refs", {}).get("SWF_LITELLM_" + "API_KEY")
        if not isinstance(value, str) or not value.strip():
            raise FileNotFoundError("Existing decision credentials unavailable")
        os.environ[variable] = value  # Original endpoint-bound credential only; never print/copy it.


def select_tools(utterance, tool_set="navigation", context=None, *, top_k=3, mock=False, on_start=None):
    """KEV tool scoring + shortlist assembly, no LLM call.

    Returns {scores, selected, schemas, native_schemas, domains, timing, log_fields}.
    `log_fields` carries exactly the evidence fields this stage used to contribute to run().
    `on_start(log_fields)` is called with the pre-scoring fields so callers can emit their
    "KEV starts here" event with the same payload the original run() used.
    """
    if tool_set not in {"navigation", "multidomain"}:
        raise ValueError("Unknown tool set")
    if mock and tool_set == "multidomain":
        raise ValueError("Multidomain uses real KEV scoring; execution remains mock-only")
    context = dict(context) if context is not None else {
        "home_address_configured": True, "company_address_configured": True, "synthetic": True}
    origin = time.monotonic()
    fields, native_schemas, domains, questions, state = {}, None, None, None, None
    if tool_set == "multidomain":
        snapshot = load_registry(MULTIDOMAIN_TOOLS, frozenset({"mock"}))
        domains = list(snapshot.domains())
        fields.update(tool_set=tool_set, tool_count=len(snapshot.tools), domain_count=len(domains),
                      registry_version=snapshot.version)
    else:
        snapshot = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
        contract = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
        catalog = freeze_selection_catalog(snapshot, contract, ())
        prepared = prepare_reopt(catalog, utterance, context)
        questions = {name: question.model_dump() for name, question in prepared["questions"].items()}
        state = json.dumps(prepared["state"], ensure_ascii=False, allow_nan=False)
    if on_start:
        on_start(dict(fields))
    stage = time.monotonic()
    if mock:
        scores = {name: 0.9 if name == "tool:navi_to_special_poi" else 0.1 for name in questions}
    elif tool_set == "multidomain":
        scores, timing = score_multidomain(snapshot, utterance, context)
        fields.update(timing)
    else:
        raw = kev_call(KEV_CLIENT, state, questions)
        fields["kev_model_latency_ms"] = raw.get("latency_ms")
        scores = {}
        for name, answer in raw["answers"].items():
            if not isinstance(answer, dict) or set(answer) - {"type", "noul"} or answer.get("type", "noul") != "noul":
                raise ValueError("Malformed KEV Noul")
            scores[name] = probability(answer.get("noul"))
    fields["scores"] = scores
    if tool_set == "navigation":
        fields["kev_latency_ms"] = (time.monotonic() - stage) * 1000
    stage = time.monotonic()
    if tool_set == "multidomain":
        selected = sorted((name[5:] for name, value in scores.items() if value >= TOOL_THRESHOLD),
                          key=lambda name: (-scores["tool:" + name], name))[:top_k]
        native_schemas = snapshot.schemas(frozenset(selected))
        schemas = multidomain_disclosure(native_schemas, {name: snapshot.tools[name].domain for name in selected})
        fields.update(native_selected_schemas=native_schemas,
                      decision_schema_profile="selection_only_resource_references",
                      selected_tool_domains={name: snapshot.tools[name].domain for name in selected},
                      selected=[{"name": name, "domain": snapshot.tools[name].domain, "score": scores["tool:" + name]}
                                for name in selected])
    else:
        selection = select_reopt(catalog, scores, tool_threshold=TOOL_THRESHOLD)
        assembled = assemble_reopt(catalog, selection)
        selected = sorted(assembled["selected_tool_ids"], key=lambda name: (-scores["tool:" + name], name))[:top_k]
        schemas = [tool for tool in assembled["schemas"] if tool["name"] in selected]
    assembled_ms = (time.monotonic() - stage) * 1000
    fields.update(selected_tools=selected, assembled_schemas=schemas, assembly_latency_ms=assembled_ms)
    if tool_set == "multidomain":
        timing = {"stage1_ms": fields.get("kev_stage1_ms"), "stage2_ms": fields.get("kev_stage2_ms")}
    else:
        timing = {"stage1_ms": fields.get("kev_latency_ms"), "stage2_ms": assembled_ms}
    timing["total_ms"] = (time.monotonic() - origin) * 1000
    return {"scores": scores, "selected": selected, "schemas": schemas, "native_schemas": native_schemas,
            "domains": domains, "timing": timing, "log_fields": fields}


def decide(utterance, selected, schemas, tool_set="navigation", context=None, *, native_schemas=None,
           available_domains=None, mock=False):
    """One synchronous, non-streaming decision call (existing slow_turn path).

    Returns {decision, llm_ms, log_fields}; decision.kind is tool_call / ask_user / done.
    """
    if tool_set not in {"navigation", "multidomain"}:
        raise ValueError("Unknown tool set")
    context = dict(context) if context is not None else {
        "home_address_configured": True, "company_address_configured": True, "synthetic": True}
    fields = {}
    stage = time.monotonic()
    # What earlier turns actually did, written back by the fast path and the LLM path alike
    # (result_writeback). Popping it from the copy keeps one copy of the same facts in the prompt:
    # the public context stays public context, completed_steps stays a sibling field - the shape
    # app.py's brief_context already uses.
    completed_steps = context.pop(COMPLETED_STEPS, None) or []
    brief = {"original_utterance": utterance, "context": context, "completed_steps": completed_steps,
             "selected_tools": selected,
             "decision_contract": "只输出一个下一步 tool_call、ask_user 或 finish；缺参数或无合适候选时 ask_user，不用 request_tools；不把未执行动作说成已完成。"}
    if tool_set == "multidomain":
        brief.update(tool_set=tool_set, available_domains=list(available_domains or []),
                     decision_contract="严格一个已披露业务下一步；受支持但缺信息用ask_user；不支持/越权/LIVE用finish(reason=明确拒绝)。不执行、不编造ID、不把未执行动作说成完成。")
    if mock:
        decision = {"kind": "tool_call", "tool": "navi_to_special_poi", "arguments": {"poi_type": "home"}}
        if decision["tool"] not in selected: raise ValueError("Mock decision not in top-k")
        fields["decision_head"] = "mock"
    else:
        try:
            existing_slow_configuration()
            fields["decision_head"] = "project_existing_litellm"
            response = slow_turn(brief, [{"role": "user", "content": utterance}], schemas, single_next_step=True)
            fields["decision_response"] = response
        except (FileNotFoundError, ModelError, httpx2.HTTPError, httpx2.InvalidURL) as error:
            # Only unavailable transport/config may fall back; invalid model/schema is a result error.
            if isinstance(error, ModelError) and not str(error).startswith(("Slow model transport failed", "LiteLLM configuration required")):
                raise
            fields["decision_head"] = "kev_fallback"
            fields["fallback_note"] = type(error).__name__ + ": existing decision path unavailable"
            criteria = {tool["name"]: tool["description"] for tool in schemas} or {"ask_user": "无合适业务工具，追问用户需求"}
            raw = kev_call(KEV_MD_CLIENT if tool_set == "multidomain" else KEV_CLIENT,
                           json.dumps({"utterance": utterance, "schemas": schemas}, ensure_ascii=False),
                           {"next_step": {"type": "choice", "instructions": "选择下一步使用的已披露工具，只选一个。", "criteria": criteria}})
            answer = raw["answers"]["next_step"]
            if (not isinstance(answer, dict) or answer.get("type", "choice") != "choice" or answer.get("choice") not in criteria
                    or not isinstance(answer.get("probabilities"), dict) or set(answer["probabilities"]) != set(criteria)):
                raise ValueError("Malformed KEV decision Choice")
            for value in [answer.get("confidence"), *answer["probabilities"].values()]: probability(value)
            # Choice cannot generate parameters; log an explicit question, not an incomplete call.
            decision = {"kind": "ask_user", "tool": "ask_user", "selected_tool": answer["choice"],
                        "arguments": {"question": f"临时决策头选择了 {answer['choice']}，但不能生成参数；请确认调用参数。"},
                        "confidence": answer["confidence"], "probabilities": answer["probabilities"], "parameters_generated": False}
        else:
            decision = decision_from_response(response, schemas, multidomain=tool_set == "multidomain",
                                              native_schemas=native_schemas, utterance=utterance)
    fields["decision"] = decision
    fields["decision_latency_ms"] = (time.monotonic() - stage) * 1000
    return {"decision": decision, "llm_ms": fields["decision_latency_ms"], "log_fields": fields}


def run(utterance="导航回家", *, top_k=3, mock=False, folder=None, on_stage=None, tool_set="navigation"):
    if tool_set not in {"navigation", "multidomain"}:
        raise ValueError("Unknown tool set")
    if not utterance.strip() or type(top_k) is not int or not 1 <= top_k <= (200 if tool_set == "multidomain" else 14):
        raise ValueError("Invalid text/top-k")
    if mock and tool_set == "multidomain":
        raise ValueError("Multidomain uses real KEV scoring; execution remains mock-only")
    checkpoint = "MD" if tool_set == "multidomain" else "C2" if folder is not None else "C1"
    run_id = folder.name if folder is not None else ("md-" if tool_set == "multidomain" else "c1-") + uuid.uuid4().hex
    folder = folder or ROOT / "eval/evidence/core-pipeline" / run_id
    folder.mkdir(parents=True)
    log = {"checkpoint": checkpoint, "run_id": run_id, "utterance": utterance, "asr": "text_standin",
           "kev_model": KEV_MODEL, "kev_endpoint": KEV_BASE + "/v1/systemone", "top_k": top_k,
           "tool_threshold": TOOL_THRESHOLD,
           "mock": mock, "executed_tools": 0}
    began = time.monotonic()
    stage = began
    active_stage = None
    try:
        context = {"home_address_configured": True, "company_address_configured": True, "synthetic": True}

        def started(fields):
            log.update(fields)
            if on_stage:
                on_stage("kev_start", dict(log))

        stage = time.monotonic()
        active_stage = "kev_latency_ms"
        selection = select_tools(utterance, tool_set, context, top_k=top_k, mock=mock,
                                 on_start=started if on_stage else None)
        log.update(selection["log_fields"])
        if on_stage:
            on_stage("kev_scores", dict(log))
            on_stage("llm_start", dict(log))
        stage = time.monotonic()
        active_stage = "decision_latency_ms"
        outcome = decide(utterance, selection["selected"], selection["schemas"], tool_set, context,
                         native_schemas=selection["native_schemas"], available_domains=selection["domains"], mock=mock)
        log.update(outcome["log_fields"])
        log["outcome"] = "PASS"
        if on_stage:
            on_stage("llm_decision", dict(log))
    except Exception as error:
        if active_stage is not None:
            log[active_stage] = (time.monotonic() - stage) * 1000
        reason = str(error) if type(error) in (ValueError, ModelError) else "Result rejected: " + type(error).__name__
        log.update(outcome="FAIL", error_type=type(error).__name__, error=reason)
    log["latency_ms"] = (time.monotonic()-began)*1000
    with (folder / "result.json").open("x") as stream:
        json.dump(log, stream, ensure_ascii=False, allow_nan=False, indent=2);stream.write("\n")
    print(json.dumps({"outcome": log["outcome"], "result_path": str(folder / "result.json"),
                      "selected_tools": log.get("selected_tools"), "decision": log.get("decision"),
                      "decision_head": log.get("decision_head"), "latency_ms": log["latency_ms"]}, ensure_ascii=False))
    return log


def run_corpus(path, *, top_k=3):
    path = Path(path).resolve()
    corpus_bytes = path.read_bytes()
    cases = [json.loads(line, object_pairs_hook=unique_object) for line in corpus_bytes.decode().splitlines() if line.strip()]
    registry = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
    ids = set()
    for case in cases:
        if (set(case) != {"id", "utterance", "expected_tools"} or not isinstance(case["id"], str)
                or not case["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in case["id"])
                or case["id"] in ids or not isinstance(case["utterance"], str) or not case["utterance"].strip()
                or not isinstance(case["expected_tools"], list) or any(t not in registry.tools for t in case["expected_tools"])):
            raise ValueError("Invalid corpus row")
        ids.add(case["id"])
    if not cases:
        raise ValueError("Empty corpus")
    folder = ROOT / "eval/evidence/core-pipeline" / ("c2-" + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    rows = []
    for case in cases:
        # expected_tools is evaluation-only; never enters KEV state or the decision brief.
        log = run(case["utterance"], top_k=top_k, folder=folder / case["id"])
        expected = set(case["expected_tools"])
        selected = log.get("selected_tools", [])
        decision = log.get("decision", {})
        rows.append({"id": case["id"], "utterance": case["utterance"], "expected_tools": case["expected_tools"],
                     "result_path": case["id"] + "/result.json", "outcome": log["outcome"],
                     "top1_hit": bool(expected.intersection(selected[:1])),
                     "top3_hit": bool(expected.intersection(selected[:3])),
                     "decision_expected_tool_hit": decision.get("kind") == "tool_call" and decision.get("tool") in expected,
                     "empty_selection_correct": not expected and "scores" in log and not selected,
                     "no_expected_control_correct": not expected and decision.get("kind") in {"ask_user", "done"},
                     "decision_head": log.get("decision_head"), "decision_kind": decision.get("kind"),
                     "error_type": log.get("error_type"), "error": log.get("error"),
                     "latency_ms": {"kev": log.get("kev_latency_ms"), "decision": log.get("decision_latency_ms"),
                                    "total": log["latency_ms"]}})
    labelled = [r for r in rows if r["expected_tools"]]
    empty = [r for r in rows if not r["expected_tools"]]
    summary = {"checkpoint": "C2", "run_id": folder.name, "asr": "text_standin", "mock": False,
               "kev_endpoint": KEV_BASE + "/v1/systemone", "kev_model": KEV_MODEL, "tool_threshold": 0.5, "top_k": top_k,
               "corpus_path": str(path), "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
               "case_count": len(rows), "expected_tool_case_count": len(labelled), "no_expected_tool_case_count": len(empty),
               "metric_definitions": {
                   "top1_top3": "Any expected tool intersects threshold-selected ranked top-1/top-3; denominator=all nonempty-expected rows, failures included.",
                   "decision_expected_tool": "Parsed tool_call names an expected business tool; denominator=all nonempty-expected rows. ask_user/done and fallback selection alone are not hits.",
                   "no_expected": "Empty expected rows excluded from tool hit rates; empty selection and ask_user/done correctness reported separately.",
                   "latency": "Milliseconds; all available per-row stage timings including failures. Missing stage timings excluded with sample_count. p50=median, p95=linear interpolation (inclusive)."},
               "metrics": {}, "latency_ms": {}, "decision_heads": dict(Counter(r["decision_head"] for r in rows)),
               "decision_kinds": dict(Counter(r["decision_kind"] for r in rows)),
               "failed_count": sum(r["outcome"] != "PASS" for r in rows),
               "failures": [{k: r[k] for k in ("id", "error_type", "error", "result_path")} for r in rows if r["outcome"] != "PASS"],
               "rows": rows}
    for name, key, group in (("top1", "top1_hit", labelled), ("top3", "top3_hit", labelled),
                             ("decision_expected_tool", "decision_expected_tool_hit", labelled),
                             ("empty_selection", "empty_selection_correct", empty),
                             ("no_expected_control", "no_expected_control_correct", empty)):
        hits = sum(r[key] for r in group)
        summary["metrics"][name] = {"hits": hits, "denominator": len(group), "rate": hits / len(group) if group else None}
    for stage in ("kev", "decision", "total"):
        values = [r["latency_ms"][stage] for r in rows if r["latency_ms"][stage] is not None]
        summary["latency_ms"][stage] = {"sample_count": len(values), "mean": statistics.mean(values) if values else None,
            "p50": statistics.median(values) if values else None,
            "p95": statistics.quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0] if values else None,
            "max": max(values) if values else None}
    summary["outcome"] = "PASS" if summary["failed_count"] == 0 else "FAIL"
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("outcome", "run_id", "metrics", "latency_ms", "failed_count", "decision_heads")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterance", default="导航回家")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tool-set", choices=("navigation", "multidomain"), default="navigation")
    args = parser.parse_args()
    if args.corpus and args.tool_set == "multidomain":
        parser.error("Use eval/multidomain_eval.py for the 200-utterance corpus")
    if args.corpus and (args.mock or args.top_k != 3):
        parser.error("C2 corpus uses real models and top-k=3")
    result = run_corpus(args.corpus, top_k=args.top_k) if args.corpus else run(args.utterance, top_k=args.top_k, mock=args.mock, tool_set=args.tool_set)
    raise SystemExit(0 if result["outcome"] == "PASS" else 1)
