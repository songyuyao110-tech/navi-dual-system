"""Measure how much of the existing corpus the current allowlist (N tools) could cover.

Product question (tmp/poc-finite-fast-followup-spec.md Part C): of the utterances we already have,
how many put an allowlisted tool at KEV shortlist rank 1? That share is the ceiling on the fast
path's end-to-end benefit - everything else can never skip the LLM no matter how τ is tuned.

    # real KEV (:8009 tool shortlist + parameter filling), writes the markdown + raw JSON
    .venv/bin/python -B eval/finite_fast_coverage.py \
        --out /Users/YUYSONG/Work/tmp/poc-finite-fast-coverage.md

    # no tunnel: only use the expected_tool labels that some corpora already carry (degraded)
    .venv/bin/python -B eval/finite_fast_coverage.py --kev label --out /tmp/coverage-label.md

Corpus sources are the ones already on disk (see SOURCES); a missing/empty file is skipped and
reported, never silently dropped. Each row is one unique utterance per source bucket.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from core_pipeline_run import select_tools
from finite_fast_path import is_finite_tool, load_allowlist, try_finite_fast_decide
from turn_session import TurnConfig

WORK = Path("/Users/YUYSONG/Work")
TOOL_SET = "multidomain"
CONTEXT = {"home_address_configured": True, "company_address_configured": True, "synthetic": True}


# ---------- corpus ----------

def _jsonl(path):
    """Parse objects only: a few corpora end with a free-text line such as `DONE ... 2026-10-19`."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def _text_of(kind, payload):
    if kind == "mic_session":
        finals = [item.get("text") for item in payload.get("asr_updates") or [] if item.get("final")]
        return finals[-1] if finals else None
    return payload.get(kind)


# Each bucket states the field carrying the utterance and (when the corpus has one) the label used
# by --kev label. `path` may be a directory: every *.jsonl inside is read.
SOURCES = [
    {"bucket": "finite-fast-cases", "path": HERE / "fixtures/finite-fast-cases.jsonl",
     "field": "utterance", "label": lambda c: c.get("expect_tool")},
    {"bucket": "finite-fast-extra", "path": WORK / "tmp/finite-fast-extra-cases.jsonl",
     "field": "utterance", "label": lambda c: c.get("expect_tool")},
    {"bucket": "mic-skin-cases", "path": WORK / "tmp/poc-mic-skin-test-cases.jsonl",
     "field": "say", "label": lambda c: c.get("expect_tool")},
    {"bucket": "mic-sessions", "path": WORK / "tmp/poc-streaming-mic-cases",
     "field": "mic_session", "label": lambda c: None},
    {"bucket": "eval-a", "path": WORK / "tmp/poc-eval-a/results.jsonl",
     "field": "text", "label": lambda c: (c.get("gold_tools") or [None])[0]},
    {"bucket": "decision-heldout-60", "path": WORK / "tmp/jev-research/heldout/heldout-60.jsonl",
     "field": "utterance", "label": lambda c: (c.get("expected_tools") or [None])[0]},
    {"bucket": "decision-split-opt-holdout",
     "path": WORK / "tmp/jev-research/split-opt-holdout/holdout.jsonl",
     "field": "utterance",
     "label": lambda c: next((name for name in c.get("acceptable_actions") or []
                              if isinstance(name, str) and "_" in name), None)},
]


def load_source(source):
    """Return (rows, raw, skip_reason): unique utterances per bucket, keeping first occurrence."""
    path = Path(source["path"])
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files or not any(file.exists() for file in files):
        return [], 0, "missing"
    rows, seen, raw = [], set(), 0
    for file in files:
        for payload in _jsonl(file):
            text = (_text_of(source["field"], payload) or "").strip()
            raw += 1
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": source["label"](payload),
                         "id": payload.get("id"), "file": file.name})
    return rows, raw, None if rows else "empty"


# ---------- measurement ----------

def measure_kev(row, cfg, run_gate):
    began = time.monotonic()
    selection = select_tools(row["text"], TOOL_SET, dict(CONTEXT))
    row["select_ms"] = (time.monotonic() - began) * 1000
    scores = selection.get("scores") or {}
    selected = selection.get("selected") or []
    row["top1"] = selected[0] if selected else None
    row["top1_score"] = scores.get("tool:" + row["top1"]) if row["top1"] else None
    row["allowlisted"] = bool(row["top1"]) and is_finite_tool(row["top1"])
    if run_gate:
        record = {}
        fast = try_finite_fast_decide(row["text"], selected, scores, selection.get("schemas") or [],
                                      cfg, record=record)
        row["hit"] = fast is not None
        row["hit_reason"] = record.get("finite_fast_reason")
        row["hit_ms"] = fast["llm_ms"] if fast else None
    return row


def measure_label(row):
    label = row.get("label")
    row["top1"] = label
    row["top1_score"] = None
    row["allowlisted"] = bool(label) and is_finite_tool(label)
    row["hit"] = False
    row["hit_reason"] = None
    return row


def summarise(bucket_rows, bucket_raw, skipped, mode, gate, elapsed_ms):
    buckets = []
    for bucket, rows in bucket_rows.items():
        counted = [row for row in rows if row.get("top1") is not None or mode == "real"]
        hits = sum(1 for row in counted if row["allowlisted"])
        buckets.append({"bucket": bucket, "n": len(counted), "raw": bucket_raw.get(bucket, 0),
                        "allowlisted": hits, "rate": hits / len(counted) if counted else None,
                        "errors": sum(1 for row in rows if row.get("error")),
                        "outside": Counter(row["top1"] for row in counted
                                           if not row["allowlisted"] and row["top1"])})
    total_n = sum(item["n"] for item in buckets)
    total_hits = sum(item["allowlisted"] for item in buckets)
    gate_rows = [row for rows in bucket_rows.values() for row in rows if row.get("hit")]
    outside = Counter()
    for item in buckets:
        outside.update(item["outside"])
    return {"bucket_rows": buckets, "skipped": skipped, "mode": mode, "gate": gate,
            "n": total_n, "allowlisted": total_hits,
            "rate": total_hits / total_n if total_n else None,
            "gate_hits": len(gate_rows),
            "gate_rate": len(gate_rows) / total_n if total_n else None,
            "gate_reasons": Counter(row["hit_reason"] for rows in bucket_rows.values()
                                    for row in rows if row.get("hit_reason")),
            "outside": outside, "elapsed_ms": elapsed_ms}


# ---------- report ----------

def _pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def compose(path, summary, allowlist, json_path):
    count = len(allowlist["order"])
    lines = [f"# {count} 工具 allowlist 覆盖率（finite fast path）", "",
             f"日期：{time.strftime('%Y-%m-%d')} · 名单：`{allowlist['path']}`（{len(allowlist['order'])} 个工具）· "
             f"测量方式：`--kev {summary['mode']}`",
             "",
             f"问题：现有语料里，短名单 top-1 有多少落在当前 {count} 个工具上。这个比例就是快路径收益的上界",
             "——top-1 不在名单里时，无论 τ 怎么调都不可能跳过 DeepSeek。", "",
             "## 语料与结果", "",
             f"| 来源 | N（唯一语句） | 原始条数 | top-1 ∈ {count} 工具 | 覆盖率 | KEV 报错 |",
             "|---|---|---|---|---|---|"]
    for item in summary["bucket_rows"]:
        lines.append(f"| {item['bucket']} | {item['n']} | {item['raw']} | {item['allowlisted']} | "
                     f"{_pct(item['rate'])} | {item['errors']} |")
    lines.append(f"| **合计** | **{summary['n']}** | — | **{summary['allowlisted']}** | "
                 f"**{_pct(summary['rate'])}** | "
                 f"{sum(item['errors'] for item in summary['bucket_rows'])} |")
    if summary["skipped"]:
        lines += ["", "跳过的来源：" + "、".join(f"`{name}`（{reason}）"
                                                 for name, reason in summary["skipped"].items())]
    lines += ["", "## 未进名单的 top-1 工具（全部语料，top 10）", "",
              "| 工具（短名单 rank 1） | 次数 |", "|---|---|"]
    for name, times in summary["outside"].most_common(10):
        lines.append(f"| `{name}` | {times} |")
    if summary["gate"]:
        lines += ["", "## 过完整闸门后的真 HIT（默认 τ=0.70，无任何词表/规则）", "",
                  f"- 真 HIT：**{summary['gate_hits']} / {summary['n']} = "
                  f"{_pct(summary['gate_rate'])}**（同一批语料、同一 KEV 服务）",
                  "- MISS 原因分布：" + "、".join(f"`{reason}` {count}"
                                                 for reason, count in summary["gate_reasons"].most_common())]
    total_ms = summary["elapsed_ms"] / 1000
    lines += ["", "## 结论", ""]
    if summary["mode"] == "real":
        lines.append(f"在本批语料下，约 {_pct(summary['rate'])} 的说法 top-1 落在当前 {count} 工具；"
                     f"端到端快路径收益上界约 ≤{_pct(summary['rate'])}（未计 τ 过滤）。")
        if summary["gate"] and summary["gate_hits"] != summary["allowlisted"]:
            lines.append("")
            lines.append(f"按默认闸门实测（工具分/参数置信 τ=0.70），真 HIT 上界约 "
                         f"{_pct(summary['gate_rate'])}——比名单覆盖率低 "
                         f"{(summary['rate'] - summary['gate_rate']) * 100:.1f} 个百分点，"
                         f"差额来自工具分/参数置信 τ 与 schema 一致性检查。")
        elif summary["gate"]:
            lines += ["", "按默认闸门实测（工具分/参数置信 τ=0.70），真 HIT 数与名单覆盖率相同："
                          "本批 top-1 命中名单的句子都过了 τ。"]
    else:
        lines.append("KEV 隧道不可用，本表只按各语料自带的 expected_tool 标签退化统计（无 top-1 短名单），"
                     "不能替代真实 top-1 覆盖率。")
    lines += ["", "## 数据局限", "",
              "- 语料不是真实车载全量流量：评测 A 的 66 条是合成的多意图/长句样本，决策 holdout 是"
              "人工构造的分层集，mic 会话是同一句「导航回家」的 8 次录音（唯一语句 1 条）。",
              "- 只测 `tool_set=multidomain`；同一句在别的 tool_set 下 top-1 可能不同。",
              "- 覆盖率是「top-1 命中名单」的比例，不等于「快路径一定执行」：还要过工具分 τ、参数置信 τ、"
              "参数置信 τ 与 schema 一致性。",
              "- 本地另有 `jev-research/split-generation/part-a|b.jsonl`（各 150 条）与 "
              "`jev-research/split-opt/opt.jsonl`（150 条），与本次纳入的两个 holdout 集高度重叠，"
              "为控制 KEV 调用量未跑；要扩样可直接把它们加进 SOURCES。", "",
              "## 复现", "", "```bash",
              "cd /Users/YUYSONG/Work/tmp/jev-reopt-checkouts/dev-streaming-eot",
              ".venv/bin/python -B eval/finite_fast_coverage.py \\",
              f"    --out {path}" + ("" if summary["mode"] == "real" else " --kev label"),
              "```", "", f"逐条结果（含 top-1/分数/是否名单内）：`{json_path}`",
              f"· 总耗时 {total_ms:.1f}s（选工具 p50 见 JSON）", ""]
    p50 = statistics.median([row["select_ms"] for rows in summary["rows"].values() for row in rows
                             if isinstance(row.get("select_ms"), (int, float))] or [0])
    lines.append(f"KEV 选工具 p50：{p50:.0f} ms/句（本机隧道）。")
    lines.append("")
    Path(path).write_text("\n".join(lines))
    return path


# ---------- cli ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev", choices=("real", "label"), default="real")
    parser.add_argument("--out", type=Path, default=Path("/Users/YUYSONG/Work/tmp/poc-finite-fast-coverage.md"))
    parser.add_argument("--json", type=Path, help="raw per-utterance JSON (default: next to --out)")
    parser.add_argument("--no-gate", action="store_true", help="skip the full-gate HIT estimate")
    parser.add_argument("--limit", type=int, default=0, help="cap utterances per bucket (smoke runs)")
    args = parser.parse_args(argv)
    json_path = args.json or args.out.with_suffix(".json")
    allowlist = load_allowlist()
    cfg = TurnConfig()
    began = time.monotonic()
    bucket_rows, bucket_raw, skipped = {}, {}, {}
    for source in SOURCES:
        rows, raw, reason = load_source(source)
        if reason:
            skipped[source["bucket"]] = reason
            print(json.dumps({"bucket": source["bucket"], "skipped": reason}, ensure_ascii=False),
                  flush=True)
            continue
        if args.limit:
            rows = rows[:args.limit]
        for row in rows:
            if args.kev == "label":
                measure_label(row)
                continue
            for attempt in (1, 2):
                try:
                    measure_kev(row, cfg, not args.no_gate)
                    row.pop("error", None)
                    break
                except Exception as error:  # one retry, then record and keep going
                    row["error"] = f"{type(error).__name__}: {error}"
                    if attempt == 2:
                        print(json.dumps({"error": row["error"], "text": row["text"]},
                                         ensure_ascii=False), flush=True)
            print(json.dumps({"src": source["bucket"], "top1": row.get("top1"),
                              "allowlisted": row.get("allowlisted"), "hit": row.get("hit"),
                              "reason": row.get("hit_reason"), "text": row["text"]},
                             ensure_ascii=False), flush=True)
        bucket_rows[source["bucket"]] = rows
        bucket_raw[source["bucket"]] = raw
    summary = summarise(bucket_rows, bucket_raw, skipped, args.kev, not args.no_gate,
                        (time.monotonic() - began) * 1000)
    summary["rows"] = {bucket: rows for bucket, rows in bucket_rows.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"summary": {key: value for key, value in summary.items()
                                                 if key not in {"rows", "bucket_rows"}},
                                     "buckets": summary["bucket_rows"], "rows": summary["rows"]},
                                    ensure_ascii=False, indent=1, allow_nan=False, default=str) + "\n")
    compose(args.out, summary, allowlist, json_path)
    print(json.dumps({"out": str(args.out), "n": summary["n"], "allowlisted": summary["allowlisted"],
                      "rate": summary["rate"], "gate_hits": summary["gate_hits"],
                      "skipped": skipped}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())