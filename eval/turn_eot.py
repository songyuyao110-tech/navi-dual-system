"""Turn-taking (EOT) judging: three Noul questions on the current ASR segment, R2 config.

EOT_LABELS is copied verbatim from pipecat-turn-eval-v2/kev_tune2.py VARS["r2_en_utter_only"]["L"].
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from core_pipeline_run import KEV_EOT_BASE, KEV_EOT_CLIENT, kev_call, probability

EOT_LABELS = {
    "complete": "The user's last utterance is a complete turn that can be answered now.",
    "cut": "The user's last utterance stopped mid-sentence; they are still going to continue (e.g. ends on a conjunction, preposition, or unfinished list).",
    "wait": "The user needs time to think or asks the assistant to wait a moment.",
}
EOT_ORDER = ("complete", "cut", "wait")


def score_eot(text: str) -> dict:
    """One KEV call on `text`; label = argmax, margin = complete - max(cut, wait)."""
    questions = {name: {"type": "noul", "instructions": EOT_LABELS[name]} for name in EOT_ORDER}
    state = json.dumps({"utterance": text}, ensure_ascii=False, allow_nan=False)
    began = time.monotonic()
    raw = kev_call(KEV_EOT_CLIENT, state, questions, base=KEV_EOT_BASE)
    wall_ms = (time.monotonic() - began) * 1000
    answers = raw["answers"]
    scores = {}
    for name in EOT_ORDER:
        answer = answers[name]
        if not isinstance(answer, dict) or set(answer) - {"type", "noul"} or answer.get("type", "noul") != "noul":
            raise ValueError("Malformed KEV EOT Noul")
        scores[name] = probability(answer.get("noul"))
    label = max(EOT_ORDER, key=lambda name: scores[name])
    return {**scores, "label": label,
            "margin": scores["complete"] - max(scores["cut"], scores["wait"]),
            "wall_ms": wall_ms, "kev_ms": raw.get("latency_ms")}


if __name__ == "__main__":
    print(json.dumps(score_eot(sys.argv[1] if len(sys.argv) > 1 else "导航回家"), ensure_ascii=False))