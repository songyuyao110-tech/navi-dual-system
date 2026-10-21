"""Build streaming prefix cases for the V1 replay evaluation.

Sources: tools-multidomain.yaml sample_utterances (200 zh) + core-pipeline-corpus.jsonl (30 zh),
pipecat-turn-eval-v2/cases.json (168 en). Prefixes grow 1-2 characters (zh) / one word (en) at a time.
True prefixes default to gold=incomplete; prefixes that can stand alone are pre-labelled by DeepSeek
and flagged `natural_complete` (reported separately, not counted as misjudgements).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from core_pipeline_run import ROOT, existing_slow_configuration
import httpx2

PIPECAT_CASES = Path("/Users/YUYSONG/Work/tmp/pipecat-turn-eval-v2/cases.json")
OUTPUT = ROOT / "eval/fixtures/streaming-prefix-cases.jsonl"
SAMPLE_COMMENT = re.compile(r"# sample_utterances: (\[.*\])")

NATURAL_PROMPT = """下面是一个完整句子，以及它从开头开始的若干前缀（按顺序递增，最后一个是整句）。
请挑出"本身已经是一个自然、可以独立结束的话轮"的前缀：用户说到那里停住，助手此时回应是合理的。
例如"导航回家吧"的"导航回家"可以独立成句；"帮我导航到人民广场"的"帮我导航到"不可以。
只输出 JSON 数组，元素是这些前缀的序号 idx（从 0 开始）。没有任何一个可独立结束就输出 []。
不要输出解释、markdown 或代码块。

句子：{sentence}
候选前缀：
{prefixes}
"""


def chinese_sentences():
    text = (ROOT / "eval/fixtures/tools-multidomain.yaml").read_text()
    rows = []
    for index, item in enumerate(SAMPLE_COMMENT.findall(text)):
        for utterance in json.loads(item):
            rows.append({"lang": "zh", "source": "tools-multidomain-sample", "text": utterance,
                         "external_id": f"md-{index:03d}"})
    for line in (ROOT / "eval/fixtures/core-pipeline-corpus.jsonl").read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            rows.append({"lang": "zh", "source": "core-pipeline-corpus", "text": case["utterance"],
                         "external_id": case["id"]})
    return rows


def english_sentences():
    cases = json.loads(PIPECAT_CASES.read_text())
    return [{"lang": "en", "source": "pipecat-turn-eval-v2", "text": case["user"],
             "external_id": case["id"], "gold_full": "complete" if case["expected"] == "complete" else "incomplete",
             "heldout": bool(case["heldout"])} for case in cases]


def slice_prefixes(case):
    """Return [(idx, prefix, is_full)] for one sentence."""
    text = case["text"].strip()
    if case["lang"] == "zh":
        characters = list(text)
        step = 1 if len(characters) <= 10 else 2
        lengths = list(range(step, len(characters), step)) + [len(characters)]
        return [(index, "".join(characters[:length]), length == len(characters))
                for index, length in enumerate(lengths)]
    words = text.split()
    return [(index, " ".join(words[:index + 1]), index + 1 == len(words)) for index in range(len(words))]


def build_cases(rows):
    cases = []
    for case in rows:
        prefixes = slice_prefixes(case)
        if not prefixes:
            continue
        for idx, prefix, is_full in prefixes:
            gold = case.get("gold_full", "complete") if is_full else "incomplete"
            cases.append({"id": f"{case['lang']}-{case['external_id']}-{idx:02d}", "lang": case["lang"],
                          "source": case["source"], "full_text": case["text"], "prefix": prefix, "idx": idx,
                          "is_full": is_full, "gold": gold, "natural_complete": False,
                          "heldout": case.get("heldout"), "external_id": case["external_id"]})
    return cases


def llm_ids(prompt):
    body = {"model": "deepseek-v4.1-flash", "stream": False, "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": {"thinking": {"type": "disabled"}}}
    base = "https://litellm.swfcn.cn.bg.corpintra.net/v1"
    with httpx2.Client(timeout=90.0, trust_env=False) as client:
        response = client.post(base + "/chat/completions",
                               headers={"Authorization": "Bearer" + " " + __import__("os").environ["LITELLM_API_KEY"]},
                               json=body)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
    match = re.search(r"\[[^\]]*\]", content)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except ValueError:
        return []
    return [int(value) for value in values if isinstance(value, (int, float))]


def label_natural(cases, workers=4):
    by_sentence = {}
    for case in cases:
        by_sentence.setdefault((case["lang"], case["full_text"]), []).append(case)
    existing = {}
    if OUTPUT.is_file():
        for line in OUTPUT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("natural_complete"):
                    existing.setdefault((row["lang"], row["full_text"]), set()).add(row["idx"])
    existing = {key: value for key, value in existing.items()}

    def label(item):
        key, group = item
        if key in existing:
            return key, existing[key]
        candidates = sorted(group, key=lambda row: row["idx"])
        # A one-character prefix or single short word is never a natural turn end.
        eligible = [row for row in candidates
                    if not row["is_full"] and (len(row["prefix"]) >= 2 if key[0] == "zh" else len(row["prefix"].split()) >= 1)]
        listing = "\n".join(f"idx={row['idx']}: {row['prefix']}" for row in eligible)
        if not listing:
            return key, set()
        prompt = NATURAL_PROMPT.replace("{sentence}", key[1]).replace("{prefixes}", listing)
        try:
            ids = set(llm_ids(prompt)) & {row["idx"] for row in eligible}
        except Exception as error:
            print(f"  ! label failed for {key[1]!r}: {type(error).__name__}", file=sys.stderr)
            ids = set()
        return key, ids

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (key, ids) in enumerate(pool.map(label, sorted(by_sentence.items())), 1):
            existing[key] = ids
            if number % 40 == 0:
                print(f"  labelled {number}/{len(by_sentence)} sentences", file=sys.stderr)
    for case in cases:
        case["natural_complete"] = case["idx"] in existing.get((case["lang"], case["full_text"]), set())
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-label", action="store_true")
    args = parser.parse_args()
    rows = chinese_sentences() + english_sentences()
    cases = build_cases(rows)
    if not args.no_label:
        existing_slow_configuration()
        cases = label_natural(cases, workers=args.workers)
    args.out.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    sentences = {case["external_id"] for case in cases}
    natural = sum(case["natural_complete"] for case in cases)
    print(json.dumps({"out": str(args.out), "cases": len(cases), "sentences": len(sentences),
                      "zh": sum(case["lang"] == "zh" for case in cases),
                      "en": sum(case["lang"] == "en" for case in cases),
                      "natural_complete": natural}, ensure_ascii=False))


if __name__ == "__main__":
    main()