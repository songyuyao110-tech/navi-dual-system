"""One result-writeback contract shared by both routes: KEV finite fast path (HIT) and LLM (MISS).

The fast path is allowed to skip the DeepSeek decision; it is not allowed to skip the outcome.
Without a writeback a HIT turn is invisible to the session, so the next turn's decide() brief plans
blind ("调用完了但 Agent 不知道"). This module is the single place both routes go through, so a HIT
and a MISS never produce two different record shapes.

Record fields (tmp/poc-finite-fast-result-writeback-spec.md §4):

    utterance      本轮确认文本
    tool           业务工具名；非 tool_call（ask_user/done）为 null
    arguments      参数 dict；非 tool_call 为 {}
    exec_result    mock 执行摘要：{mock_only, skipped, executed_tools, ok?, reason?, error?, message?}
    routed_by      "kev" | "llm"
    decision_head  如 kev_finite_fast / project_existing_litellm / kev_fallback / mock
    miss_reason    仅 MISS 闸门降级时有（not_allowlisted / tool_score_low / ...），省略即 HIT
    kind           decision.kind：tool_call / ask_user / done
    status         completed / skipped / failed（mock 语义）
    ts_ms          epoch 毫秒（墙钟，int）
    turn_ms        会话相对毫秒（TurnSession 内为 turn_done 时刻；独立单次调用为 null）
    turn           会话内轮次（1 起；独立单次调用为 null）

Timestamps: the spec allows "会话相对 ms 或 ISO, 实现选一种". Chosen: `ts_ms` is epoch ms (works for
both the session and the single-shot /api/run path), and `turn_ms` carries the session-relative value
so the timeline can be read next to TurnSession's own `t_ms` events. The record does not use `ts`/`t`
key names because TurnSession._emit already puts its relative `t_ms` on the event envelope.

`exec_result` is the one key for mock execution (the spec allows exec_result / mock_result, one or
both); `mock_only: true` inside it states that no business tool ran. `executed_tools` counts real
business executions, hence it stays 0 on the PoC.
"""
from __future__ import annotations

import time

COMPLETED_STEPS = "completed_steps"   # session context key that survives turn boundaries
EVENT = "result_writeback"            # timeline event name, HIT and MISS alike
STATUSES = ("completed", "skipped", "failed")
ROUTED_BY = ("kev", "llm")


def mock_execute(decision):
    """Mock the business tool (never a real vehicle) and return (status, exec_result)."""
    decision = decision if isinstance(decision, dict) else {}
    kind = decision.get("kind")
    if kind != "tool_call":
        # ask_user / done still write one record so "a complete turn" has no hole in it.
        return "skipped", {"mock_only": True, "skipped": True, "executed_tools": 0,
                           "reason": "kind=" + str(kind)}
    tool = decision.get("tool")
    arguments = decision.get("arguments")
    if not isinstance(tool, str) or not tool:
        return "failed", {"mock_only": True, "skipped": False, "executed_tools": 0,
                          "error": "tool_call without a tool name"}
    if not isinstance(arguments, dict):
        return "failed", {"mock_only": True, "skipped": False, "executed_tools": 0,
                          "error": "tool_call without an argument object"}
    return "completed", {"mock_only": True, "skipped": False, "executed_tools": 0, "ok": True,
                         "tool": tool, "arguments": dict(arguments),
                         "message": "mock 执行完成：" + tool + "（未调用真实业务工具）"}


def writeback_result(*, utterance, decision, routed_by, decision_head=None, miss_reason=None,
                     ts_ms=None, turn=None, turn_ms=None):
    """Build the one record both routes write into context[COMPLETED_STEPS] and the timeline."""
    if routed_by not in ROUTED_BY:
        raise ValueError("routed_by must be one of " + ", ".join(ROUTED_BY))
    decision = decision if isinstance(decision, dict) else {}
    kind = decision.get("kind")
    tool_call = kind == "tool_call"
    arguments = decision.get("arguments")
    status, exec_result = mock_execute(decision)
    record = {"utterance": utterance,
              "tool": decision.get("tool") if tool_call else None,
              "arguments": dict(arguments) if tool_call and isinstance(arguments, dict) else {},
              "exec_result": exec_result,
              "routed_by": routed_by,
              "decision_head": decision_head,
              "kind": kind,
              "status": status,
              "ts_ms": int(ts_ms if ts_ms is not None else time.time() * 1000),
              "turn": turn,
              "turn_ms": None if turn_ms is None else round(float(turn_ms), 3)}
    if miss_reason:
        record["miss_reason"] = miss_reason
    return record


def append_writeback(context, record):
    """Append to the session's completed-steps list and return it (created when absent)."""
    steps = context.setdefault(COMPLETED_STEPS, [])
    steps.append(record)
    return steps