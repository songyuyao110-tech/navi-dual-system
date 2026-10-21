"""Latency A/B for the speculative decision chain (tmp/poc-speculative-toolselect-spec.md, acceptance 1-2).

One turn per sentence, driven in real wall-clock time against the real KEV (EOT on :8010, tool
scoring + parameter filling on :8009). The decision model is mocked with the measured PoC latency
unless --llm real is passed, so the A/B isolates the stage this change moves (the tool-select /
gate segment), not DeepSeek variance.

Two corpora:

  commands  the finite-fast case corpus: real car commands, so HIT (allowlisted finite tool) and
            MISS (open slot / ambiguous) both get samples. Partial ASR updates are synthesised by
            cutting the command at fixed character ratios (150 ms apart), the way ASR grows a buffer.
  prefix    the recorded prefix corpus (`eval/fixtures/streaming-prefix-cases.jsonl`), which is what
            the streaming PoC already measures; used for speculation churn on realistic partial counts.

  # after (speculative chain on) vs before (the a5272c1 behaviour) on the command corpus
  .venv/bin/python -B eval/speculative_latency_eval.py --corpus commands --mode both \
      --out /Users/YUYSONG/Work/tmp/poc-speculative-latency-commands.json

  # speculation churn on recorded prefixes (Chinese, 40 sentences)
  .venv/bin/python -B eval/speculative_latency_eval.py --corpus prefix --langs zh --limit 40 \
      --mode both --out /Users/YUYSONG/Work/tmp/poc-speculative-latency-prefix.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from core_pipeline_run import select_tools
from streaming_prefix_eval import MEASURED_LLM_MS, load_sentences, mock_decide
from turn_eot import score_eot
from turn_session import TurnConfig, TurnSession, normalize

CASE_CORPUS = HERE / "fixtures/finite-fast-cases.jsonl"
PREFIX_CORPUS = HERE / "fixtures/streaming-prefix-cases.jsonl"
WORK = Path("/Users/YUYSONG/Work/tmp")


# ---------- timelines ----------

def command_updates(text, prefix_interval_ms):
    """ASR-like growth of one command: cut at 40%/70%/100% of the characters."""
    length = len(text)
    cuts = sorted({min(length, max(2, math.ceil(length * ratio))) for ratio in (0.4, 0.7, 1.0)})
    updates, t = [], 0
    for index, cut in enumerate(cuts):
        t += prefix_interval_ms
        updates.append({"t_ms": t, "text": text[:cut], "final": index == len(cuts) - 1})
    return updates


def command_sentences(path, limit=0):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    sentences = [{"lang": "zh", "source": "finite-fast-cases", "id": case["id"],
                  "text": case["utterance"], "cases": None} for case in cases]
    return sentences[:limit] if limit else sentences


def prefix_updates(sentence, prefix_interval_ms):
    updates, t = [], 0
    for case in sentence["cases"]:
        t += prefix_interval_ms
        updates.append({"t_ms": t, "text": case["prefix"], "final": bool(case["is_full"])})
    return updates


# ---------- one turn ----------

def run_turn(sentence, updates, cfg, decide_fn, *, turn_budget_ms=20000):
    events = []
    session = TurnSession(cfg, events.append, eot_fn=score_eot, select_tools_fn=select_tools,
                          decide_fn=decide_fn, evidence_root=None, session_id=sentence["id"])
    started = time.monotonic() * 1000
    now = lambda: time.monotonic() * 1000 - started
    try:
        for item in updates:
            while now() < item["t_ms"] and not session.turn_done:
                session.tick(min(now(), item["t_ms"]))
                time.sleep(0.003)
            if session.turn_done:
                break
            session.on_asr(item["t_ms"], item["text"], item["final"])
        limit = now() + turn_budget_ms
        while not session.turn_done and now() < limit:
            session.tick(now())
            time.sleep(0.003)
    finally:
        session.close()
    summary = session.summary()
    final_text = session.turn_text()
    writebacks = [event for event in events if event["event"] == "result_writeback"]
    executed = [writeback.get("utterance") for writeback in writebacks]
    row = {
        "id": sentence["id"], "lang": sentence["lang"], "source": sentence["source"],
        "text": sentence["text"], "final_text": final_text,
        "prefix_count": len(updates), "turn_done": session.turn_done,
        "route": session.routed_by, "miss_reason": session.miss_reason,
        "decision_tool": (session.decision or {}).get("tool"),
        "decision_arguments": (session.decision or {}).get("arguments"),
        "t_last_word": summary["t_last_word"], "t_confirmed": summary["t_confirmed"],
        "t_decision": summary["t_decision"],
        "last_word_to_decision_ms": summary["t_decision_latency_ms"],
        "confirm_to_decision_ms": summary["decision_after_confirm_ms"],
        "spec_calls": summary["spec_calls"], "spec_wasted": summary["spec_wasted"],
        "spec_reused_hit": summary["spec_reused_hit"],
        "spec_reused_selection": summary["spec_reused_selection"],
        "spec_reused_gate": summary["spec_reused_gate"], "spec_waited": summary["spec_waited"],
        "spec_stale_results": summary["spec_stale_results"],
        "spec_stale_executions_blocked": summary["spec_stale_executions_blocked"],
        "spec_recomputes": summary["spec_recomputes"],
        "llm_calls": summary["llm_calls"], "llm_wasted": summary["llm_wasted"],
        "finite_fast_hits": summary["finite_fast_hits"],
        "eot_calls": summary["eot_calls"],
        "confirm_window_ms": summary["t_confirm_latency_ms"],
        "premature_llm": any(event["event"] == "llm_start" and summary["t_last_word"] is not None
                             and event["t_ms"] < summary["t_last_word"] for event in events),
        # Independent of the session's own guard: every writeback must carry the final text.
        "executed_utterances": executed,
        "stale_executions": sum(1 for text in executed if normalize(text) != normalize(final_text)),
        "decision_events": [{"event": event["event"], "t_ms": event["t_ms"],
                             "spec_gated": event.get("spec_gated"),
                             "spec_reused": event.get("spec_reused")}
                            for event in events
                            if event["event"] in {"spec_start", "spec_ready", "spec_cancel",
                                                  "spec_discarded", "spec_recompute", "llm_start",
                                                  "llm_decision", "turn_done"}],
    }
    return row


# ---------- statistics ----------

def points(values, wanted=(50, 90)):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return {point: None for point in wanted}
    result = {}
    for point in wanted:
        if len(values) == 1:
            result[point] = values[0]
        else:
            index = min(len(values) - 1, int(round(point / 100 * (len(values) - 1))))
            result[point] = values[index]
    return result


def chain_started_before_confirm(row):
    """Was the chain that produced this decision started before the turn was confirmed?"""
    confirmed = row["t_confirmed"]
    if confirmed is None:
        return False
    return any(event["event"] == "llm_start" and event.get("spec_gated")
               and event["t_ms"] <= confirmed for event in row["decision_events"])


def summarise(rows, mode, corpus):
    groups = {"all": rows}
    for route in ("kev", "llm"):
        groups["hit" if route == "kev" else "miss"] = [row for row in rows if row["route"] == route]
    by_lang = defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row)
    summary = {"mode": mode, "corpus": corpus, "n": len(rows),
               "turn_done": sum(row["turn_done"] for row in rows),
               "stale_executions": sum(row["stale_executions"] for row in rows),
               "stale_executions_blocked": sum(row["spec_stale_executions_blocked"] for row in rows),
               "routes": dict(Counter(row["route"] for row in rows)),
               "miss_reasons": dict(Counter(row["miss_reason"] for row in rows if row["miss_reason"])),
               "chain_before_confirm": sum(chain_started_before_confirm(row) for row in rows),
               "spec": {}, "latency": {}}
    for name, subset in list(groups.items()) + [("lang:" + lang, rows) for lang, rows in by_lang.items()]:
        if not subset:
            continue
        summary["spec"][name] = {
            "n": len(subset),
            "spec_calls_mean": statistics.mean(row["spec_calls"] for row in subset),
            "spec_wasted_mean": statistics.mean(row["spec_wasted"] for row in subset),
            "reused_hit": sum(row["spec_reused_hit"] for row in subset),
            "reused_selection": sum(row["spec_reused_selection"] for row in subset),
            "stale_results": sum(row["spec_stale_results"] for row in subset),
            "reused_gate": sum(row.get("spec_reused_gate") or 0 for row in subset),
            "premature_llm": sum(bool(row.get("premature_llm")) for row in subset),
            "recomputes": sum(row["spec_recomputes"] for row in subset),
            "llm_calls": sum(row["llm_calls"] for row in subset),
            "chain_before_confirm": sum(chain_started_before_confirm(row) for row in subset),
        }
        summary["latency"][name] = {
            "confirm_to_decision": points([row["confirm_to_decision_ms"] for row in subset]),
            "last_word_to_decision": points([row["last_word_to_decision_ms"] for row in subset]),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("commands", "prefix"), default="commands")
    parser.add_argument("--cases", type=Path, default=CASE_CORPUS)
    parser.add_argument("--prefix-cases", type=Path, default=PREFIX_CORPUS)
    parser.add_argument("--mode", choices=("before", "after", "both"), default="both")
    parser.add_argument("--langs", default="zh,en")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prefix-interval-ms", type=int, default=150)
    parser.add_argument("--decide-ms", type=float, default=MEASURED_LLM_MS)
    parser.add_argument("--llm", choices=("mock", "real"), default="mock")
    parser.add_argument("--turn-budget-ms", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.corpus == "commands":
        sentences = command_sentences(args.cases, args.limit)
        updates_for = lambda sentence: command_updates(sentence["text"], args.prefix_interval_ms)
    else:
        langs = tuple(part.strip() for part in args.langs.split(",") if part.strip())
        sentences = load_sentences(args.prefix_cases, langs)
        if args.limit:
            kept, counts = [], Counter()
            for sentence in sentences:
                if counts[sentence["lang"]] < args.limit:
                    counts[sentence["lang"]] += 1
                    kept.append(sentence)
            sentences = kept
        updates_for = lambda sentence: prefix_updates(sentence, args.prefix_interval_ms)

    if args.llm == "real":
        from core_pipeline_run import decide as decide_fn
        decisions = decide_fn
    else:
        decisions = (lambda utterance, selected, schemas, tool_set, context, **kwargs:
                     mock_decide(utterance, selected, schemas, tool_set, context,
                                 latency_ms=args.decide_ms, **kwargs))

    modes = ["before", "after"] if args.mode == "both" else [args.mode]
    rows, began = [], time.monotonic()
    for mode in modes:
        # The finite-fast allowlist covers tools-multidomain.yaml, so the replay must select from
        # that registry (the streaming PoC default is the 14-tool navigation set).
        cfg = TurnConfig(tool_set="multidomain", speculative_chain=(mode == "after"), debounce_ms=80)
        for number, sentence in enumerate(sentences, 1):
            row = run_turn(sentence, updates_for(sentence), cfg, decisions,
                           turn_budget_ms=args.turn_budget_ms)
            row["mode"] = mode
            rows.append(row)
            print(json.dumps({"mode": mode, "id": row["id"], "route": row["route"],
                              "confirm_to_decision_ms": round(row["confirm_to_decision_ms"] or -1),
                              "last_word_to_decision_ms": round(row["last_word_to_decision_ms"] or -1),
                              "spec_calls": row["spec_calls"], "spec_wasted": row["spec_wasted"],
                              "reused": row["spec_reused_hit"] + row["spec_reused_selection"],
                              "stale": row["stale_executions"]}, ensure_ascii=False), flush=True)
            if number % 10 == 0 or number == len(sentences):
                print(f"  {mode}: {number}/{len(sentences)}", file=sys.stderr, flush=True)

    summaries = {mode: summarise([row for row in rows if row["mode"] == mode], mode, args.corpus)
                 for mode in modes}
    payload = {"corpus": args.corpus, "llm": args.llm, "decide_ms": args.decide_ms,
               "prefix_interval_ms": args.prefix_interval_ms, "n": len(sentences),
               "elapsed_ms": (time.monotonic() - began) * 1000, "summaries": summaries, "rows": rows}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(sentences),
                      "summaries": {mode: {"latency": item["latency"].get("all"),
                                           "spec": item["spec"].get("all")}
                                    for mode, item in summaries.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())