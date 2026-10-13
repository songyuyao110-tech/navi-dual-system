from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import httpx2
from jsonschema import Draft202012Validator


class ModelError(ValueError):
    pass


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
        "question": {"type": ["string", "null"]},
    },
    "required": ["steps", "question"],
    "additionalProperties": False,
}


def fast_plan(brief: dict, schemas: list[dict]) -> dict:
    base = os.environ.get("FAST_MODEL_URL", "http://127.0.0.1:11434")
    if urlparse(base).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelError("The fast model must be a loopback local endpoint")
    body = {
        "model": os.environ.get("FAST_MODEL", "qwen3:4b"),
        "think": False,
        "stream": False,
        "format": PLAN_SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一次性工具计划器，不执行工具，不做 ReAct，不根据未知结果规划。"
                    "只使用披露的工具和参数要求，按顺序输出完整 steps。"
                    "填参优先级：最新用户表达 > 对话上下文 > 已提供偏好 > 合法默认。"
                    "家庭或公司地址已配置且用户要求导航到家或公司时，"
                    "navi_to_special_poi 的 poi_type=home 或 company 已足够，无须地址坐标。"
                    "不猜测未知地点、坐标或缺失必要信息。缺项时 steps=[]，question=具体追问。"
                    "有效计划 question=null。不得同时出计划与追问，不加无关的状态查询。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"brief": brief, "disclosed_tools": schemas},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            },
        ],
    }
    # The approved fast chain is local; proxy settings are only relevant to cloud clients.
    try:
        with httpx2.Client(timeout=90.0, trust_env=False) as client:
            response = client.post(base.rstrip("/") + "/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
    except httpx2.HTTPError as error:
        raise ModelError(f"Local fast model transport failed: {type(error).__name__}") from None
    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise ModelError("Local fast model returned no message")
    try:
        plan = json.loads(data["message"]["content"])
    except (KeyError, TypeError, ValueError):
        raise ModelError("Local fast model returned invalid JSON") from None
    errors = list(Draft202012Validator(PLAN_SCHEMA).iter_errors(plan))
    if errors:
        raise ModelError(f"Invalid fast plan shape: {errors[0].message}")
    question = plan["question"]
    if (question is not None and not question.strip()) or bool(plan["steps"]) == bool(question):
        raise ModelError("Fast plan must contain either steps or a concrete question")
    return {
        **plan,
        "model": data.get("model"),
        "usage": {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        },
        "total_ms": data.get("total_duration", 0) / 1_000_000
        if isinstance(data.get("total_duration"), (int, float)) else None,
    }


CONTROL_TOOLS = [
    {
        "name": "ask_user",
        "description": "执行层需要具体缺项或用户选择时，提出一个问题并挂起原任务",
        "input_schema": {
            "type": "object", "properties": {"question": {"type": "string", "minLength": 1}},
            "required": ["question"], "additionalProperties": False,
        },
    },
    {
        "name": "request_tools",
        "description": "已披露工具不够时，描述下一步意图，重新并发全部域 B 请求追加披露",
        "input_schema": {
            "type": "object", "properties": {"intent": {"type": "string", "minLength": 1}},
            "required": ["intent"], "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "任务目标已由执行事实完成时结束；不允许把未执行或失败动作说成成功",
        "input_schema": {
            "type": "object", "properties": {}, "additionalProperties": False,
        },
    },
]


def _slow_messages(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant"}:
            raise ModelError("Unsupported slow conversation role")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ModelError("Invalid slow conversation content")
        if role == "assistant":
            text, calls = [], []
            reasoning = None
            for item in content:
                if not isinstance(item, dict):
                    raise ModelError("Invalid slow assistant block")
                if "reasoning_content" in item:
                    value = item["reasoning_content"]
                    if not isinstance(value, str) or (reasoning is not None and reasoning != value):
                        raise ModelError("Invalid slow reasoning metadata")
                    reasoning = value
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text.append(item["text"])
                elif item.get("type") == "tool_use" and (
                    isinstance(item.get("id"), str) and item["id"].strip()
                    and isinstance(item.get("name"), str) and item["name"].strip()
                    and isinstance(item.get("input"), dict)
                ):
                    calls.append({
                        "id": item["id"], "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item["input"], ensure_ascii=False, allow_nan=False),
                        },
                    })
                else:
                    raise ModelError("Unsupported slow assistant block")
            converted = {"role": "assistant", "content": "\n".join(text) if text else None}
            if calls:
                converted["tool_calls"] = calls
            if reasoning is not None:
                converted["reasoning_content"] = reasoning
            result.append(converted)
        else:
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "tool_result" or (
                    not isinstance(item.get("tool_use_id"), str) or not item["tool_use_id"].strip()
                    or not isinstance(item.get("content"), str)
                ):
                    raise ModelError("Invalid slow tool result")
                value = item["content"]
                if item.get("is_error"):
                    value = json.dumps(
                        {"is_error": True, "content": value}, ensure_ascii=False, allow_nan=False,
                    )
                result.append({
                    "role": "tool", "tool_call_id": item["tool_use_id"], "content": value,
                })
    return result


def _slow_response(data: dict) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or (
        len(data["choices"]) != 1 or not isinstance(data["choices"][0], dict)
    ):
        raise ModelError("Slow model returned invalid choices")
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ModelError("Slow model response was truncated")
    if choice.get("finish_reason") == "content_filter":
        raise ModelError("Slow model response was filtered")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ModelError("Slow model returned no assistant message")
    if message.get("refusal"):
        raise ModelError("Slow model refused the request")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise ModelError("Slow model returned unsupported text content")
    content = [{"type": "text", "text": text}] if text else []
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ModelError("Slow model returned invalid tool calls")
    identifiers = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function" or (
            not isinstance(call.get("id"), str) or not call["id"].strip()
            or call["id"] in identifiers
        ):
            raise ModelError("Slow model returned an invalid tool call identifier")
        function = call.get("function")
        if not isinstance(function, dict) or (
            not isinstance(function.get("name"), str) or not function["name"].strip()
            or not isinstance(function.get("arguments"), str)
        ):
            raise ModelError("Slow model returned an invalid function call")
        try:
            arguments = json.loads(function["arguments"])
            json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError):
            raise ModelError("Slow model returned invalid JSON tool arguments") from None
        if not isinstance(arguments, dict):
            raise ModelError("Slow model tool arguments must be an object")
        identifiers.add(call["id"])
        content.append({
            "type": "tool_use", "id": call["id"], "name": function["name"], "input": arguments,
        })
    reasoning = message.get("reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ModelError("Slow model returned invalid reasoning metadata")
        if content:
            # DeepSeek may require this opaque metadata on the next tool-call round.
            content[0]["reasoning_content"] = reasoning
    usage = data.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ModelError("Slow model returned invalid usage")
    return {
        "model": data.get("model"),
        "content": content,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") if usage else None,
            "output_tokens": usage.get("completion_tokens") if usage else None,
        },
    }


def slow_turn(brief: dict, messages: list[dict], schemas: list[dict]) -> dict:
    settings = {
        name: os.environ.get(name, "").strip()
        for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY", "SLOW_MODEL")
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ModelError("LiteLLM configuration required: " + ", ".join(missing))
    base = settings["LITELLM_BASE_URL"]
    token = settings["LITELLM_API_KEY"]
    model = settings["SLOW_MODEL"]
    endpoint = urlparse(base)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or (
        endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
    ):
        raise ModelError("LiteLLM base URL must be an HTTP(S) API root without credentials or query")
    if endpoint.scheme == "http" and endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelError("Non-loopback LiteLLM endpoints require HTTPS")
    base = base.rstrip("/")
    if not endpoint.path.rstrip("/").endswith("/v1"):
        base += "/v1"
    body = {
        "model": model,
        "stream": False,
        "max_tokens": 2048,
        "messages": [{"role": "system", "content": (
            "你是单 Agent 的慢执行链，按执行结果逐步规划，最多十轮。"
            "只调用已披露业务工具，缺工具用 request_tools；缺信息用 ask_user。"
            "ask_user、request_tools、finish 是流程控制，不得与业务动作混在同一轮。"
            "简报 completed_steps 是已完成事实，不重做其中已成功的非幂等动作。"
            "不能猜测地址、坐标等缺项。只能以工具回传事实判断是否成功。"
            "完成原任务用 finish。未完成时不得 finish，也不以普通文本代替流程控制。"
            "所有当前工具都是 PoC mock，无真实车辆动作。\n任务简报："
            + json.dumps(brief, ensure_ascii=False, allow_nan=False)
        )}] + _slow_messages(messages),
        "tool_choice": "auto",
        "tools": [
            {"type": "function", "function": {
                "name": tool["name"], "description": tool["description"],
                "parameters": tool["parameters"],
            }} for tool in schemas
        ] + [
            {"type": "function", "function": {
                "name": tool["name"], "description": tool["description"],
                "parameters": tool["input_schema"],
            }} for tool in CONTROL_TOOLS
        ],
    }
    try:
        with httpx2.Client(timeout=90.0) as client:
            response = client.post(
                base + "/chat/completions",
                headers={"Authorization": "Bearer " + token}, json=body,
            )
            response.raise_for_status()
            data = response.json()
    except httpx2.HTTPError as error:
        raise ModelError(f"Slow model transport failed: {type(error).__name__}") from None
    except ValueError:
        raise ModelError("Slow model returned invalid JSON") from None
    return _slow_response(data)


def review_task(utterance: str, pending: dict[str, str], destination: dict, context: dict) -> str:
    response = slow_turn(
        {"original_utterance": utterance, "pending_tasks": pending, "context": context,
         "uncertain_destination": {
             name: destination[name] for name in ("task_kind", "target_task", "route")
         }},
        [{"role": "user", "content": (
            "核对的是新建任务还是修订/取消列表内未完成任务，不是重新索要已知业务参数。"
            "车辆仍导航中不等于台账任务未完成。保留上下文中已经配置的家庭/公司地址。"
            "只用 ask_user 提问核对目标任务或任务意图，不执行其他动作。"
        )}],
        [],
    )
    calls = [item for item in response["content"] if item["type"] == "tool_use"]
    if len(calls) != 1 or calls[0]["name"] != "ask_user":
        raise ModelError("Task review did not return a clarification")
    errors = list(
        Draft202012Validator(CONTROL_TOOLS[0]["input_schema"]).iter_errors(calls[0]["input"])
    )
    if errors:
        raise ModelError("Task review returned an invalid clarification")
    return calls[0]["input"]["question"]
