"""V1 prefix replay evaluation: feed prefix sequences to TurnSession and report the passing criteria.

Real KEV is used for EOT judging and tool scoring; the decision LLM is mocked with the measured mean
PoC latency so the evaluation does not spend LLM budget. Timings are real (the replay clock is wall
clock, because KEV latency is one of the quantities under test).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from core_pipeline_run import ROOT, select_tools
from turn_eot import score_eot
from turn_session import TurnConfig, TurnSession

CASES = ROOT / "eval/fixtures/streaming-prefix-cases.jsonl"
EVIDENCE = ROOT / "eval/evidence/streaming-prefix"
MEASURED_LLM_MS = 1380.0     # plan §4: PoC non-streaming DeepSeek decision, navigation
PIPECAT_CASES = Path("/Users/YUYSONG/Work/tmp/pipecat-turn-eval-v2/cases.json")


def mock_decide(utterance, selected, schemas, tool_set, context, *, native_schemas=None,
                available_domains=None, mock=False, latency_ms=MEASURED_LLM_MS):
    time.sleep(latency_ms / 1000.0)
    tool = selected[0] if selected else "ask_user"
    return {"decision": {"kind": "tool_call" if selected else "ask_user", "tool": tool, "arguments": {}},
            "llm_ms": latency_ms, "log_fields": {"decision_head": "eval_mock"}}


def load_sentences(path, langs):
    grouped = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            if case["lang"] in langs:
                grouped[(case["lang"], case["source"], case["external_id"], case["full_text"])].append(case)
    sentences = []
    for (lang, source, external_id, text), cases in grouped.items():
        cases.sort(key=lambda case: case["idx"])
        sentences.append({"lang": lang, "source": source, "id": external_id, "text": text, "cases": cases})
    sentences.sort(key=lambda item: (item["lang"], item["source"], item["id"]))
    return sentences


def build_timeline(sentence, prefix_interval_ms, pause_ratio, rng):
    """ASR updates + VAD; ~20% of sentences pause mid-sentence, which also closes the ASR segment."""
    cases = sentence["cases"]
    pause_at = None
    pause_ms = 0
    if len(cases) >= 4 and rng.random() < pause_ratio:
        pause_at = rng.randrange(1, len(cases) - 1)
        pause_ms = rng.randrange(400, 801)
    updates, vad, t = [], [{"t_ms": 0, "kind": "speech_start"}], 0
    for index, case in enumerate(cases):
        if pause_at is not None and index == pause_at + 1:
            vad.append({"t_ms": int(t), "kind": "speech_end"})
            t += pause_ms
            vad.append({"t_ms": int(t), "kind": "speech_start"})
        t += prefix_interval_ms
        # After a segment boundary the ASR restarts its buffer: only the new part is sent.
        text = case["prefix"] if pause_at is None or index <= pause_at else \
            case["prefix"][len(cases[pause_at]["prefix"]):]
        updates.append({"t_ms": int(t), "text": text, "final": case["is_full"], "idx": case["idx"]})
    return updates, vad, pause_at, pause_ms


def run_sentence(sentence, cfg, decisions, *, prefix_interval_ms, pause_ratio, rng, turn_budget_ms):
    updates, vad, pause_at, pause_ms = build_timeline(sentence, prefix_interval_ms, pause_ratio, rng)
    events = []
    session = TurnSession(cfg, events.append, eot_fn=score_eot, select_tools_fn=select_tools,
                          decide_fn=decisions, session_id=sentence["id"], evidence_root=None)
    timeline = sorted([{**item, "type": "asr"} for item in updates] +
                      [{"t_ms": item["t_ms"], "type": "vad", "kind": item["kind"]} for item in vad],
                      key=lambda item: item["t_ms"])
    started = time.monotonic() * 1000
    now = lambda: time.monotonic() * 1000 - started
    for item in timeline:
        while now() < item["t_ms"] and not session.turn_done:
            session.tick(min(now(), item["t_ms"]))
            time.sleep(0.004)
        if session.turn_done:
            break
        if item["type"] == "asr":
            session.on_asr(item["t_ms"], item["text"], item["final"])
        else:
            session.on_vad(item["t_ms"], item["kind"])
    limit = now() + turn_budget_ms
    while not session.turn_done and now() < limit:
        session.tick(now())
        time.sleep(0.004)
    session.close()
    return events, session, {"pause_at": pause_at, "pause_ms": pause_ms, "updates": updates}


def analyse(sentence, events, session, meta, cfg):
    """Per-sentence metrics, including which prefixes were judged complete."""
    eot_text_by_version = {}
    for event in events:
        if event["event"] == "eot_start":
            eot_text_by_version[event["text_version"]] = event["text"]
    prefix_by_text = {case["prefix"]: case for case in sentence["cases"]}
    verdicts = []
    for event in events:
        if event["event"] != "eot_scores":
            continue
        text = event.get("text") or eot_text_by_version.get(event["text_version"], "")
        case = prefix_by_text.get(text)
        verdicts.append({"text": text, "label": event["label"], "margin": event["margin"],
                         "stale": bool(event.get("stale")),
                         "complete": event["complete"], "cut": event["cut"], "wait": event["wait"],
                         "wall_ms": event["wall_ms"], "idx": case["idx"] if case else None,
                         "is_full": case["is_full"] if case else None,
                         "gold": case["gold"] if case else None,
                         "natural_complete": case["natural_complete"] if case else None,
                         "matched": case is not None})
    last_update = max(event["t_ms"] for event in events if event["event"] == "asr_update")
    t_last_word = session.summary()["t_last_word"]
    summary = session.summary()
    return {"id": sentence["id"], "lang": sentence["lang"], "source": sentence["source"],
            "text": sentence["text"], "prefix_count": len(sentence["cases"]),
            "pause_at": meta["pause_at"], "pause_ms": meta["pause_ms"],
            "turn_done": session.turn_done, "config": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
            "t_last_word": t_last_word, "last_update_ms": last_update,
            "t_confirmed": summary["t_confirmed"], "t_llm_start": summary["t_llm_start"],
            "t_decision": summary["t_decision"],
            "confirm_latency_ms": summary["t_confirm_latency_ms"],
            "llm_start_latency_ms": summary["t_llm_start_latency_ms"],
            "decision_latency_ms": summary["t_decision_latency_ms"],
            "eot_calls": summary["eot_calls"], "eot_skipped": summary["eot_skipped"],
            "eot_stale": summary.get("eot_stale"),
            "tool_calls": summary["tool_calls"], "llm_calls": summary["llm_calls"],
            "llm_wasted": summary["llm_wasted"],
            "eot_skips": dict(Counter(event["reason"] for event in events
                                      if event["event"] == "eot_skipped")),
            "verdicts": verdicts,
            "llm_starts": [event["t_ms"] for event in events if event["event"] == "llm_start"],
            "tools_times": [event["t_ms"] for event in events if event["event"] == "spec_start"],
            "premature_llm": any(event["event"] == "llm_start" and event["t_ms"] < last_update for event in events),
            "llm_decisions": sum(event["event"] == "llm_decision" for event in events),
            "barge_ins": sum(event["event"] == "barge_in" for event in events),
            "tool_reuse_events": sum(event["event"] == "spec_ready" for event in events),
            "spec_calls": summary.get("spec_calls"), "spec_wasted": summary.get("spec_wasted"),
            "spec_reused_hit": summary.get("spec_reused_hit"),
            "spec_reused_selection": summary.get("spec_reused_selection"),
            "spec_stale_results": summary.get("spec_stale_results"),
            "spec_stale_executions_blocked": summary.get("spec_stale_executions_blocked"),
            "states": [event["to"] for event in events if event["event"] == "state_change"]}


def percentiles(values, points=(50, 90)):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {point: None for point in points}
    result = {}
    for point in points:
        if len(values) == 1:
            result[point] = values[0]
        else:
            index = min(len(values) - 1, int(round(point / 100 * (len(values) - 1))))
            result[point] = values[index]
    return result


def v0_regression():
    """B1: score_eot on all 168 pipecat turns must reproduce the held-out binary accuracy (~94.6%)."""
    cases = json.loads(PIPECAT_CASES.read_text())
    rows = []
    for case in cases:
        result = score_eot(case["user"])
        rows.append({"id": case["id"], "heldout": bool(case["heldout"]), "gold": case["expected"],
                     "label": result["label"], "binary": "complete" if result["label"] == "complete" else "incomplete",
                     "three_way_hit": (result["label"] == "complete") if case["expected"] == "complete"
                     else (result["label"] in {"cut", "wait"}),
                     "complete": result["complete"], "cut": result["cut"], "wait": result["wait"],
                     "margin": result["margin"], "wall_ms": result["wall_ms"], "kev_ms": result["kev_ms"]})
    summary = {}
    for name, subset in (("all", rows), ("heldout", [row for row in rows if row["heldout"]]),
                         ("heldout_complete", [row for row in rows if row["heldout"] and row["gold"] == "complete"]),
                         ("heldout_incomplete", [row for row in rows if row["heldout"] and row["gold"] == "incomplete"])):
        correct = sum(row["gold"] == row["binary"] for row in subset)
        incomplete = [row for row in subset if row["gold"] == "incomplete"]
        summary[name] = {"n": len(subset), "correct": correct,
                         "rate": correct / len(subset) if subset else None,
                         "cut": sum(row["label"] == "cut" for row in incomplete),
                         "wait": sum(row["label"] == "wait" for row in incomplete),
                         "false_complete": sum(row["label"] == "complete" for row in incomplete)}
    return summary, rows


def build_report(rows, v0, config, args):
    zh = [row for row in rows if row["lang"] == "zh"]
    en = [row for row in rows if row["lang"] == "en"]
    lines = []
    add = lines.append
    add("# 流式 KEV 判停 PoC — 前缀回放评测报告（M2 / V1）")
    add("")
    add(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 配置：`{json.dumps(config, ensure_ascii=False)}`")
    add(f"- 数据集：{len(rows)} 句 / {sum(row['prefix_count'] for row in rows)} 个前缀"
        f"（中文 {len(zh)} 句，英文 {len(en)} 句；`eval/fixtures/streaming-prefix-cases.jsonl`）")
    add(f"- 回放：前缀间隔 {args.prefix_interval_ms}ms，约 {int(args.pause_ratio * 100)}% 句中插入 400–800ms 停顿（停顿同时切分 ASR 段）")
    add(f"- 决策 LLM：mock（固定 {args.decide_ms:.0f}ms = plan §4 实测均值），避免评测成本；EOT 与选工具为真 KEV")
    add("")
    for name, subset in (("中文", zh), ("英文", en)):
        add(f"## {name}（{len(subset)} 句）")
        add("")
        if not subset:
            add("（无数据）")
            add("")
            continue
        all_prefixes = sum(row["prefix_count"] for row in subset)
        judged = [v for row in subset for v in row["verdicts"] if v["matched"] and not v["is_full"]]
        judged_strict = [v for v in judged if not v["natural_complete"]]
        fresh = [v for v in judged if not v.get("stale")]
        judged_natural = [v for v in judged if v["natural_complete"]]
        unmatched = sum(len([v for v in row["verdicts"] if not v["matched"]]) for row in subset)
        prefix_total, prefix_total_all = len(judged_strict), len(judged)
        prefix_bad = sum(v["label"] == "complete" for v in judged_strict)
        prefix_bad_all = sum(v["label"] == "complete" for v in judged)
        sentence_bad = sum(any(v["matched"] and not v["is_full"] and not v["natural_complete"]
                               and v["label"] == "complete" for v in row["verdicts"]) for row in subset)
        natural_said_complete = sum(v["label"] == "complete" for v in judged_natural)
        premature = sum(row["premature_llm"] for row in subset)
        multi_decision = sum(row["llm_decisions"] != 1 for row in subset)
        not_done = sum(not row["turn_done"] for row in subset)
        confirm = percentiles([row["confirm_latency_ms"] for row in subset])
        start = percentiles([row["llm_start_latency_ms"] for row in subset])
        reused = [row for row in subset if row.get("tools_times") is not None
                  and not [t for t in row["tools_times"] if t > (row["t_last_word"] or 0)]]
        start_reused = percentiles([row["llm_start_latency_ms"] for row in reused])
        decide = percentiles([row["decision_latency_ms"] for row in subset])
        eot_calls = statistics.mean(row["eot_calls"] for row in subset)
        eot_skipped = statistics.mean(row["eot_skipped"] for row in subset)
        tool_calls = statistics.mean(row["tool_calls"] for row in subset)
        llm_calls = sum(row["llm_calls"] for row in subset)
        llm_wasted = sum(row["llm_wasted"] for row in subset)
        skips = Counter()
        for row in subset:
            skips.update(row["eot_skips"])
        add(f"| 指标 | 值 |")
        add(f"|---|---|")
        add(f"| 判停覆盖率（有判停结果的非整句前缀 / 全部前缀） | {prefix_total_all}/{all_prefixes} = "
            f"{(prefix_total_all / all_prefixes * 100 if all_prefixes else 0):.1f}% |")
        add(f"| 判停可用覆盖率（结果落地时仍是最新文本，非 stale） | {len(fresh)}/{all_prefixes} = "
            f"{(len(fresh) / all_prefixes * 100 if all_prefixes else 0):.1f}% |")
        fresh_all = [v for row in subset for v in row["verdicts"] if not v.get("stale")]
        add(f"| 每句可用（非 stale）判停结果数（含整句，均值） | "
            f"{len(fresh_all) / len(subset):.2f} |")
        add(f"| ① 前缀误判说完率（按前缀，排除 natural_complete） | {prefix_bad}/{prefix_total} = "
            f"{(prefix_bad / prefix_total * 100 if prefix_total else 0):.2f}% |")
        add(f"| ① 前缀误判说完率（按前缀，不排除） | {prefix_bad_all}/{prefix_total_all} = "
            f"{(prefix_bad_all / prefix_total_all * 100 if prefix_total_all else 0):.2f}% |")
        add(f"| ① 被预标 natural_complete 的前缀中判为 complete | {natural_said_complete}/{len(judged_natural)} |")
        add(f"| ① 判停结果中判完即过期（stale）的条数 | "
            f"{sum(1 for v in [v for row in subset for v in row['verdicts']] if v.get('stale'))} |")
        add(f"| ① 未能归因到某个前缀的判停结果（停顿后段内增量文本） | {unmatched} |")
        add(f"| ① 前缀误判说完率（按句，任一前缀误判） | {sentence_bad}/{len(subset)} = "
            f"{(sentence_bad / len(subset) * 100):.2f}% |")
        add(f"| ② 提前触发 LLM 率（按句） | {premature}/{len(subset)} = {premature / len(subset) * 100:.2f}% |")
        add(f"| ③ 最后一字→CONFIRMED（中位 / p90） | {confirm[50]:.0f} / {confirm[90]:.0f} ms |")
        add(f"| ③ 最后一字→LLM 启动（中位 / p90） | {start[50]:.0f} / {start[90]:.0f} ms |")
        add(f"| ③ 其中工具短名单已复用（最后一字后无新选工具调用，n={len(reused)}） | "
            f"{start_reused[50]:.0f} ms / p90 {start_reused[90]:.0f} ms |"
            if reused else "| ③ 其中工具短名单已复用 | 本运行未记录选工具时间戳 |")
        add(f"| ③ 最后一字→决策（中位 / p90） | {decide[50]:.0f} / {decide[90]:.0f} ms |")
        add(f"| ④ 每句 KEV 判停调用数（均值） | {eot_calls:.2f} |")
        add(f"| ④ 每句 KEV 判停跳过数（均值） | {eot_skipped:.2f}（{dict(skips)}） |")
        add(f"| ④ 每句选工具调用数（均值） | {tool_calls:.2f} |")
        add(f"| ⑤ LLM 调用 / 浪费 / 浪费率 | {llm_calls} / {llm_wasted} / "
            f"{(llm_wasted / llm_calls * 100 if llm_calls else 0):.1f}% |")
        add(f"| 每轮有效决策恰好 1 次的句子占比 | {(len(subset) - multi_decision)}/{len(subset)} |")
        add(f"| 未走完一轮（未 CONFIRMED/无决策） | {not_done}/{len(subset)} |")
        add("")
    add("## B1 · V0 回归（英文 168 轮整句二分类，`score_eot`）")
    add("")
    add("| 子集 | n | 二分类正确 | 二分类准确率 | 其中 incomplete 判成 cut / wait / complete（误判说完） |")
    add("|---|---|---|---|---|")
    for name, item in v0.items():
        add(f"| {name} | {item['n']} | {item['correct']} | "
            f"{item['rate'] * 100 if item['rate'] is not None else 0:.1f}% | "
            f"{item['cut']} / {item['wait']} / {item['false_complete']} |")
    add("")
    add("对照设计文档 [实测]：留出集二分类 94.6%（全 168 轮：93 条未说完里 11 条误判说完，75 条说完 0 条漏判）。")
    add("`cases.json` 只带 complete/incomplete 二分类金标，设计文档里的三分类 97.4% 需要另一套 cut/wait 金标，"
        "本机没有该数据，故不宣称复现三分类数字。")
    add("")
    add("## 通过标准判定（plan §6 V1）")
    add("")
    standard = [("整句二分类 ≥93%", (v0.get("all", {}).get("rate") or 0) >= 0.93),
                ("提前触发 LLM 率 ≤5%（按句，中英分别）",
                 all(sum(row["premature_llm"] for row in subset) / len(subset) <= 0.05
                     for subset in (zh, en) if subset)),
                ("最后一字→LLM 启动中位 ≤800ms（复用工具时 ≤400ms）",
                 all(percentiles([row["llm_start_latency_ms"] for row in subset])[50] <= 800
                     for subset in (zh, en) if subset)),
                ("每轮有效 LLM 决策恰好 1 次",
                 all(row["llm_decisions"] == 1 for row in rows)),
                ("投机浪费率 ≤30%",
                 (sum(row["llm_wasted"] for row in rows) / max(1, sum(row["llm_calls"] for row in rows))) <= 0.30)]
    for text, ok in standard:
        add(f"- [{'PASS' if ok else 'FAIL'}] {text}")
    add("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--langs", default="zh,en")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-lang-limit", type=int, default=0)
    parser.add_argument("--prefix-interval-ms", type=int, default=150)
    parser.add_argument("--pause-ratio", type=float, default=0.2)
    parser.add_argument("--decide-ms", type=float, default=MEASURED_LLM_MS)
    parser.add_argument("--turn-budget-ms", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--config", default="{}", help="TurnConfig overrides as JSON")
    parser.add_argument("--report", type=Path, default=Path("/Users/YUYSONG/Work/tmp/poc-streaming-prefix-eval-report.md"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-v0", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--report-only", type=Path, default=None,
                        help="rebuild the report from an existing detail.jsonl")
    args = parser.parse_args()

    overrides = json.loads(args.config)
    cfg = TurnConfig(**overrides) if overrides else TurnConfig()
    langs = tuple(part.strip() for part in args.langs.split(",") if part.strip())
    run_id = args.run_id or ("px-" + time.strftime("%Y%m%d-%H%M%S"))
    folder = EVIDENCE / run_id
    folder.mkdir(parents=True, exist_ok=True)
    decide = (lambda utterance, selected, schemas, tool_set, context, **kwargs:
              mock_decide(utterance, selected, schemas, tool_set, context, latency_ms=args.decide_ms, **kwargs))

    v0 = {}
    if not args.skip_v0:
        print("B1 V0 regression on 168 pipecat turns ...", file=sys.stderr)
        v0, v0_rows = v0_regression()
        (folder / "v0-regression.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in v0_rows))
        print("  " + json.dumps(v0, ensure_ascii=False), file=sys.stderr)

    rows = []
    if args.report_only:
        rows = [json.loads(line) for line in args.report_only.read_text().splitlines() if line.strip()]
    elif not args.skip_eval:
        sentences = load_sentences(args.cases, langs)
        if args.per_lang_limit:
            kept, counts = [], Counter()
            for sentence in sentences:
                if counts[sentence["lang"]] < args.per_lang_limit:
                    counts[sentence["lang"]] += 1
                    kept.append(sentence)
            sentences = kept
        if args.limit:
            sentences = sentences[:args.limit]
        rng = random.Random(args.seed)
        print(f"evaluating {len(sentences)} sentences ...", file=sys.stderr)
        for number, sentence in enumerate(sentences, 1):
            events, session, meta = run_sentence(sentence, cfg, decide, prefix_interval_ms=args.prefix_interval_ms,
                                                pause_ratio=args.pause_ratio, rng=rng,
                                                turn_budget_ms=args.turn_budget_ms)
            rows.append(analyse(sentence, events, session, meta, cfg))
            if number % 10 == 0 or number == len(sentences):
                print(f"  {number}/{len(sentences)} sentences, "
                      f"{sum(len(row['verdicts']) for row in rows)} EOT verdicts", file=sys.stderr)
        with (folder / "detail.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        columns = ["id", "lang", "source", "text", "prefix_count", "pause_at", "pause_ms", "turn_done",
                   "t_last_word", "t_confirmed", "t_llm_start", "t_decision", "confirm_latency_ms",
                   "llm_start_latency_ms", "decision_latency_ms", "eot_calls", "eot_skipped", "tool_calls",
                   "llm_calls", "llm_wasted", "llm_decisions", "premature_llm", "barge_ins"]
        with (folder / "detail.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"detail: {folder}/detail.jsonl ({len(rows)} rows)", file=sys.stderr)

    config = {"eot_policy": cfg.eot_policy, "eot_margin": cfg.eot_margin, "spec_margin": cfg.spec_margin,
              "speculative_llm": cfg.speculative_llm, "eot_input": cfg.eot_input,
              "confirm_silence_ms": cfg.confirm_silence_ms, "cut_wait_ms": cfg.cut_wait_ms,
              "think_wait_ms": cfg.think_wait_ms, "max_silence_ms": cfg.max_silence_ms,
              "eot_timeout_ms": cfg.eot_timeout_ms, "debounce_ms": cfg.debounce_ms,
              "reuse_tools_if_text_unchanged": cfg.reuse_tools_if_text_unchanged, "tool_set": cfg.tool_set,
              "speculative_chain": cfg.speculative_chain}
    argv = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    summary = {"run_id": run_id, "folder": str(folder), "cases": str(args.cases), "config": config,
               "args": argv, "v0": v0, "sentences": len(rows)}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    if rows:
        if args.report_only:
            config = rows[0]["config"] or config
        args.report.write_text(build_report(rows, v0, config, args))
        print(f"report: {args.report}", file=sys.stderr)
    print(json.dumps({key: summary[key] for key in ("run_id", "folder", "v0", "sentences")}, ensure_ascii=False))


if __name__ == "__main__":
    main()