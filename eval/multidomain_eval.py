"""Evaluate multi-domain selection using only the YAML sample_utterances comments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS_YAML = ROOT / "eval/fixtures/tools-multidomain.yaml"
SMOKE_TOOLS = ("navi_to_special_poi", "climate_set_temperature", "music_pause")


def load_cases(path):
    """Associate each tool's single comment utterance with its name and domain."""
    text = Path(path).read_text(encoding="utf-8")
    tools = yaml.safe_load(text)["tools"]
    samples = {}
    for line in text.splitlines():
        tool_text, marker, metadata = line.partition("# sample_utterances:")
        if not marker:
            continue
        tool = yaml.safe_load(tool_text)[0]
        utterances = yaml.safe_load("sample_utterances:" + metadata)["sample_utterances"]
        if (not isinstance(utterances, list) or len(utterances) != 1
                or not isinstance(utterances[0], str) or not utterances[0].strip()):
            raise ValueError("Each tool must have one nonempty sample_utterances comment")
        name = tool["name"]
        if name in samples:
            raise ValueError("Duplicate tool sample_utterances comment: " + name)
        samples[name] = utterances[0]
    if not tools or len(samples) != len(tools) or set(samples) != {tool["name"] for tool in tools}:
        raise ValueError("Every tool must have exactly one sample_utterances comment")
    return [{"id": f"MD{index:03d}", "utterance": samples[tool["name"]],
             "expected_tool": tool["name"], "expected_domain": tool["domain"]}
            for index, tool in enumerate(tools, 1)]


def evaluate(path, smoke=False):
    """Run top-k=3 and retain misses and failures in the full-sample denominators."""
    path = Path(path).resolve()
    corpus_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = load_cases(path)
    corpus_case_count = len(cases)
    tool_domains = {case["expected_tool"]: case["expected_domain"] for case in cases}
    if smoke:
        by_tool = {case["expected_tool"]: case for case in cases}
        cases = [by_tool[name] for name in SMOKE_TOOLS]

    # Import only when evaluating: load_cases and --help never initialize model clients.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import core_pipeline_run as pipeline
    pipeline.MULTIDOMAIN_TOOLS = path  # Score the same YAML whose samples and hash are evaluated.
    run = pipeline.run

    folder = ROOT / "eval/evidence/multidomain" / ("md-" + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    rows = []
    for case in cases:
        case_folder = folder / case["id"]
        began = time.monotonic()
        try:
            # Labels and metadata are evaluation-only; only utterance enters the pipeline.
            log = run(case["utterance"], tool_set="multidomain", top_k=3, folder=case_folder)
        except Exception as error:
            # A raised pipeline failure still gets a row, evidence and the same denominator.
            # Do not persist arbitrary exception text, which might contain credentials.
            log = {"outcome": "FAIL", "error_type": type(error).__name__,
                   "error": "Pipeline raised " + type(error).__name__ + " before returning a result",
                   "latency_ms": (time.monotonic() - began) * 1000, "executed_tools": 0}
            case_folder.mkdir(parents=True, exist_ok=True)
            result_path = case_folder / "result.json"
            if not result_path.exists():
                result_path.write_text(json.dumps(log, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                       encoding="utf-8")

        selected_tools = log.get("selected_tools", [])
        selected_domains = log.get("selected_tool_domains", {})
        selected = log.get("selected", [{"name": name,
                                         "domain": selected_domains.get(name, tool_domains.get(name)),
                                         "score": log.get("scores", {}).get("tool:" + name)}
                                        for name in selected_tools])
        decision = log.get("decision", {})
        top1 = selected_tools[0] if selected_tools else None
        expected = case["expected_tool"]
        top1_hit = top1 == expected
        top3_hit = expected in selected_tools[:3]
        llm_hit = (decision.get("kind") == "tool_call" and decision.get("tool") == expected
                   and log.get("decision_head") == "project_existing_litellm")
        reasons = []
        if not top1_hit:
            reasons.append("kev_top1_miss")
        if not top3_hit:
            reasons.append("kev_top3_miss")
        if not llm_hit:
            if log.get("decision_head") != "project_existing_litellm":
                reasons.append("decision_not_project_existing_litellm")
            elif decision.get("kind") != "tool_call":
                reasons.append("decision_not_tool_call")
            else:
                reasons.append("decision_tool_miss")
        if log.get("outcome") != "PASS":
            reasons.append("pipeline_failure")
        rows.append({**case, "actual_top1_tool": top1,
                     "actual_top1_domain": selected_domains.get(top1, tool_domains.get(top1)),
                     "selected_tools": selected_tools, "selected_tool_domains": selected_domains,
                     "selected": selected, "scores": log.get("scores", {}),
                     "decision": decision, "decision_head": log.get("decision_head"),
                     "top1_hit": top1_hit, "top3_hit": top3_hit, "llm_in_expected_hit": llm_hit,
                     "outcome": log.get("outcome", "FAIL"),
                     "error": log.get("error"), "error_type": log.get("error_type"), "reasons": reasons,
                     "result_path": str(case_folder / "result.json"), "case_folder": str(case_folder),
                     "kev_wall_ms": log.get("kev_wall_ms"),
                     "kev_stage1_ms": log.get("kev_stage1_ms"), "kev_stage2_ms": log.get("kev_stage2_ms"),
                     "latency_ms": log.get("latency_ms"),
                     "recalled_domains": log.get("recalled_domains", []),
                     "kev_stage2_batches": log.get("kev_stage2_batches", []),
                     "executed_tools": log.get("executed_tools", 0)})

    summary = {"batch_id": folder.name, "tool_set": "multidomain", "top_k": 3, "smoke": smoke,
               "corpus_path": str(path), "corpus_sha256": corpus_sha256,
               "corpus_source": "YAML sample_utterances comments (not MD U001-U200)",
               "corpus_case_count": corpus_case_count, "case_count": len(rows),
               "metric_definitions": {
                   "kev_top1_top3": "Expected tool in ranked selected_tools[:1]/[:3]; denominator=all evaluated samples including FAIL.",
                   "llm_in_expected": "kind=tool_call AND tool=expected_tool AND decision_head=project_existing_litellm; fallback alone is not a hit; denominator=all evaluated samples including FAIL.",
                   "latency": "Milliseconds, all available timings including FAIL; missing timings excluded with sample_count; p50/p95 use inclusive linear interpolation."},
               "metrics": {}, "latency_ms": {},
               "failed_count": sum(row["outcome"] != "PASS" for row in rows),
               "failures": [row for row in rows if row["outcome"] != "PASS"],
               "confusions": [row for row in rows if not all((row["top1_hit"], row["top3_hit"],
                                                             row["llm_in_expected_hit"]))],
               "rows": rows, "executed_tools": sum(row["executed_tools"] for row in rows)}
    for name, key in (("kev_top1", "top1_hit"), ("kev_top3", "top3_hit"),
                      ("llm_in_expected", "llm_in_expected_hit")):
        hits = sum(row[key] for row in rows)
        summary["metrics"][name] = {"hits": hits, "denominator": len(rows), "rate": hits / len(rows)}
    for name, key in (("kev_wall", "kev_wall_ms"), ("kev_stage1", "kev_stage1_ms"), ("kev_stage2", "kev_stage2_ms"),
                      ("e2e", "latency_ms")):
        values = [row[key] for row in rows if row[key] is not None]
        summary["latency_ms"][name] = {
            "sample_count": len(values), "mean": statistics.mean(values) if values else None,
            "p50": statistics.median(values) if values else None,
            "p95": statistics.quantiles(values, n=100, method="inclusive")[94] if len(values) > 1
                   else values[0] if values else None,
            "max": max(values) if values else None}
    summary["outcome"] = "FAIL" if summary["failed_count"] else "PASS"
    summary_path = folder / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "outcome": summary["outcome"],
                      "metrics": summary["metrics"], "latency_ms": summary["latency_ms"]},
                     ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Evaluate three samples from navigation, climate and music")
    parser.add_argument("--tools-yaml", type=Path, default=DEFAULT_TOOLS_YAML,
                        help="YAML registry containing sample_utterances comments (default: eval fixture)")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.tools_yaml, smoke=args.smoke)["outcome"] == "PASS" else 1)
