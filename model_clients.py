from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urlparse
from jsonschema.exceptions import SchemaError, ValidationError

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


# Selection-only multidomain profile; legacy navigation/execution controls stay unchanged.
MULTIDOMAIN_CONTROLS = [{
    "name": "ask_user",
    "description": "意图对象/渠道未确定时先中立澄清；不能从候选工具推定用户已选空调、屏幕、歌曲或电话。对象确定后才问真正缺项或越界目标。",
    "input_schema": {
        "type": "object", "properties": {"question": {
            "type": "string", "minLength": 1,
            "description": "先问未确定的对象或渠道；冷热可指空气/座椅，明暗可指灯/屏幕，播放可指不同媒体，联系需明确电话/消息。不能只问已假定域的温区或歌名。",
        }}, "required": ["question"], "additionalProperties": False,
    },
}, {
    "name": "finish",
    "description": "明确拒绝当前不支持、越权或禁止的请求并结束；reason说明能力/授权边界，不声称业务动作已执行。",
    "input_schema": {
        "type": "object", "properties": {"reason": {"type": "string", "minLength": 1}},
        "required": ["reason"], "additionalProperties": False,
    },
}]
MULTIDOMAIN_NEXT_STEP_PROMPT = (
    "你是多域工具决策器，本轮只决定一个合理下一步，不执行业务工具。"
    "决策顺序：先排除不支持/禁止目标；再消歧未确定对象/渠道；然后按原话明确优先选一个动作-对象配对；最后生成参数。不要先看披露工具顺序选工具。"
    "选动作时先识别先/再/之后：如果要求先暂停某个明确音源，即使同时有灯光或屏幕设置，也先暂停该音源，不先做设置。"
    "不同分句的音源与动词必须配对：暂停A并播放B，不能变成暂停B。如果A的暂停工具未披露，不能用B的暂停代替，可先选择原请求另一个独立可做目标。"
    "对象/联系渠道未确定时必须中立ask_user，不能先按候选schema填槽或查联系人；selected_tools和schema排列仅是候选集合，不是用户优先顺序。"
    "澄清对象不限于当前已披露工具；冷热要区分车内空气与座椅，明暗要区分灯与屏幕，泛指播放先问媒体类型/对象，联系先问电话还是消息。"
    "严格只产生一次工具调用：一个已披露业务工具，或ask_user，或finish；不能同时输出两个动作，也不能混合控制与业务。"
    "即使用户提出多个独立目标，也只选一个本轮可做的下一步，余下目标留待后续；存在结果依赖时先取所需结果。"
    "先判断用户要求的动作是否受支持和授权，再判断对象与参数。股票买卖成交、ECU刷写、读取他人私聊、绕过mock连接LIVE等不受支持/禁止的动作，"
    "必须finish(reason=明确的拒绝和能力边界)，不能追问如何交易、越权或连接实车，也不能用普通查询冒充这些执行能力。"
    "需要消歧或有效必填参数的受支持请求才ask_user；不要为了多确认而追问已明确的对象、数值或开关。"
    "对于没有对象上下文的冷热、明暗、播放、联系或指代性关闭，先中立追问对象/渠道，再问目标；"
    "不能把KEV候选当用户已经确认的域，不能擅自将热理解为空调、暗理解为屏幕、播放理解为歌曲或联系理解为电话。"
    "按用户的动作、对象和作用范围选工具，KEV相关性不等于下一步；不要加用户没要求且不构成真实依赖的状态查询或搜索。"
    "用户已说当前、正在、已选、已确认、已授权的状态可作为语义前提；空参的当前对象操作无需先查状态。"
    "但这些词不是缺失ID/坐标的具体值来源：不得把current、selected、姓名、工具名或自行编号冒充ID。"
    "只有用户/上下文/真实工具结果提供的值或schema明确声明的默认/对象引用可以填参；schema未支持引用且缺真实句柄时，做必要准备查询或ask_user。"
    "这是决策意图而不是实际执行：schema资源字段若允许unresolved_resource对象，用户明确已选/当前/已确认/刚才的资源应直接用该引用表达目标动作，"
    "按声明填ref_kind、namespace、selector、source_text（逐字复制原句对象片段）；不要重复搜同一对象或向用户索取后台ID。引用仍需后续可信解析，不证明唯一绑定/授权。"
    "引用selector由源片段实际词决定，不凭当前语境默认current：当前/正在/现在对应current；已确认/已配对/已接好/已连接/已授权对应confirmed。source_text必须连续原文片段，不改写。"
    "若schema允许preserve_existing_value且用户明确某字段不变，用该保留引用表达，不猜测旧值、不把不变当作标题。"
    "仅补齐schema实际required参数，不把可选字段额外当必填；普通名称和已给歌曲标题可直接用，不先惯例性检索。"
    "没有明确电话还是消息的联系请求，应先ask_user确认渠道，不以查通讯录替代这一澄清。"
    "从用户表达直接转换合法枚举、百分比、摄氏度、档位和时间；不编造未给出的资源、句柄、坐标或结果。"
    "已确认坐标、刚才搜到、已选地点等文字不提供实际坐标/结果列表/目的地名称；不可用0/0、空列表、合成地点或那家/已选餐厅的字串充当具体数据。"
    "原schema没有这些字段的引用分支且缺必填实际数据时ask_user；可选坐标或数据未给时省略，不额外追问、不自行补齐。"
    "独立目标可选一个先做，但用户明确先/再/之后的优先或依赖必须遵循；不把逗号排列机械当依赖。"
    "选择前先找用户显式优先动作；例如多个设置目标中另要求先暂停明确播放源，下一步先暂停该源。不得因工具披露排列或相关性排名先选其他设置。"
    "消息和草稿正文保留用户给定的事实、时态和语义，不加等待、到达或其他未给信息；明确给定原文时直接复制。"
    "参数越界须ask_user请求有效目标，不裁剪数值、不执行错误值。"
    "复合句分别识别每个动词及其对象，暂停音乐不等于暂停播客，播放视频不等于关闭屏幕；不要把一个分句的动词套在另一个对象上。"
    "用户说先别播放、别发送、别导航时，选择搜索/草稿/查询而不是被禁止的动作。"
    "completed_steps才是已完成事实，不重做已成功的非幂等动作，不把未执行任务说成完成。"
    "所有工具都是mock选择意图，绝不连接真车或LIVE。\n任务简报："
)


def multidomain_disclosure(schemas: list[dict], domains: dict[str, str]) -> list[dict]:
    """Selection-only view; registered execution schemas and navigation14 are not mutated."""
    views = json.loads(json.dumps(schemas, ensure_ascii=False, allow_nan=False))
    for tool in views:
        if domains[tool["name"]] == "navigation":
            continue
        changed = False
        for name, schema in list(tool["parameters"].get("properties", {}).items()):
            if name.endswith("_id") and schema.get("type") == "string":
                reference = {"type": "object", "properties": {
                    "ref_kind": {"const": "unresolved_resource"},
                    "namespace": {"const": name[:-3]},
                    "selector": {"enum": ["selected", "current", "confirmed", "previous"],
                        "description": "严格按source_text实际词选择：selected有已选/选中/选定，current有当前/正在/现在，confirmed有确认/已配对/已接好/已连接/已授权，previous有刚才/之前/上次。没有当前词不默认current。"},
                    "source_text": {"type": "string", "minLength": 1},
                }, "required": ["ref_kind", "namespace", "selector", "source_text"], "additionalProperties": False}
                tool["parameters"]["properties"][name] = {
                    "description": "真实ID必须来自输入；原句明确已选/当前/已确认/刚才对象时，用未解析资源引用，不用current或姓名冒充ID。",
                    "anyOf": [schema, reference],
                }
                changed = True
        if tool["name"] == "calendar_update_event":
            title = tool["parameters"]["properties"]["title"]
            tool["parameters"]["properties"]["title"] = {"anyOf": [title, {
                "type": "object", "properties": {"ref_kind": {"const": "preserve_existing_value"},
                    "source_text": {"type": "string", "minLength": 1}},
                "required": ["ref_kind", "source_text"], "additionalProperties": False,
            }], "description": "明确标题不变时表达preserve_existing_value，不猜旧标题；执行前仍需解析真实原标题。"}
            changed = True
        if changed:
            tool["description"] += "；本轮可用schema声明的未解析资源/保留值引用表达此目标意图，引用不是实际ID或已执行结果。"
    return views


def _validate_multidomain_navigation_sources(name: str, arguments: dict, utterance: str) -> None:
    """Current MD context has no bound navigation data; accept only supplied literal values."""
    labels = {
        "lat": r"lat|latitude|end_lat|纬度|北纬", "end_lat": r"lat|latitude|end_lat|纬度|北纬",
        "latitude": r"lat|latitude|end_lat|纬度|北纬",
        "lng": r"lng|longitude|end_lng|经度|东经", "end_lng": r"lng|longitude|end_lng|经度|东经",
        "longitude": r"lng|longitude|end_lng|经度|东经",
    }
    def coordinates(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in labels:
                    pattern = r"(?:" + labels[key] + r")[\"'”]?\s*(?:[:：=]|为|是)?\s*([-+]?\d+(?:\.\d+)?)"
                    supplied = [float(v) for v in re.findall(pattern, utterance, re.IGNORECASE)]
                    if item not in supplied:
                        raise ModelError("Multidomain coordinate has no supplied target value; placeholders are not data")
                coordinates(item)
        elif isinstance(value, list):
            for item in value:
                coordinates(item)
    coordinates(arguments)
    if "poi_list" in arguments:
        supplied = []
        for match in re.finditer(r"(?:poi_list|地点(?:结果)?列表|搜索结果列表|POI列表)[\"'”]?\s*[:：=]\s*", utterance):
            try:
                value, _ = json.JSONDecoder().raw_decode(utterance[match.end():])
            except ValueError:
                continue
            if isinstance(value, list):
                supplied.append(value)
        if arguments["poi_list"] not in supplied:
            raise ModelError("Multidomain POI results were not supplied; an empty array cannot stand for unknown results")
    if name == "navi_start_route" and "end_name" in arguments:
        target = arguments["end_name"]
        relative = r"已选|选中|选定|那家|这家|那个|那里|刚才"
        if re.search(relative, utterance):
            quoted = re.findall(r"[\"“「]([^\"”」]+)[\"”」]", utterance)
            explicit = re.findall(r"(?:end_name|目的地名称)[\"'”]?\s*[:：=]\s*([^，。;；\n]+)", utterance)
            located = all(k in arguments for k in ("end_lat", "end_lng"))
            category_only = bool(re.fullmatch(r"(?:川菜|中餐|西餐|附近)?(?:餐厅|咖啡店|停车场|酒店|商家|目的地|地方)", target))
            named_in_input = target in utterance and not re.search(relative, target) and not category_only
            if not located and target not in quoted + explicit and not named_in_input:
                raise ModelError("Multidomain selected destination is unresolved; supply an actual name/address or clarify")


def validate_multidomain_arguments(name: str, arguments: dict, native: dict, utterance: str) -> list[dict]:
    """Validate provenance of selection references; never resolve or authorize them."""
    _validate_multidomain_navigation_sources(name, arguments, utterance)
    unresolved = []
    markers = {"selected": r"已(?:经)?选|选中|选好|选定(?!后|再|以后)", "current": r"当前|正在|现在",
               "confirmed": r"确认|已配对|已接好|已连接|已授权", "previous": r"刚才|之前|上次"}
    for key, value in arguments.items():
        schema = native.get("properties", {}).get(key, {})
        if key.endswith("_id") and schema.get("type") == "string":
            if isinstance(value, str):
                pattern = r"(?:" + re.escape(key) + r"|ID|编号)\s*(?:[:：=]|为|是)?\s*[\"'“]?" + re.escape(value)
                if not value or not re.search(pattern, utterance, re.IGNORECASE):
                    raise ModelError("Multidomain resource ID has no input provenance; use an explicit selection reference or clarify")
            else:
                source = value["source_text"]
                match = re.search(markers[value["selector"]], source)
                if (value["ref_kind"] != "unresolved_resource" or value["namespace"] != key[:-3]
                        or source not in utterance or not match
                        or re.search(r"未|没有|还没|尚未|不", source[:match.start()])):
                    raise ModelError("Multidomain selection reference has no affirmative input source")
                unresolved.append({"argument": key, **value})
        elif name == "calendar_update_event" and key == "title":
            preserve = bool(re.search(r"标题.*(?:不变|保留|保持)|不改.*标题", utterance))
            if isinstance(value, dict):
                if (value.get("ref_kind") != "preserve_existing_value" or value.get("source_text") not in utterance
                        or not re.search(r"不变|保留|保持", value["source_text"]) or not preserve):
                    raise ModelError("Multidomain preserved title has no input source")
                unresolved.append({"argument": key, **value})
            elif preserve:
                raise ModelError("Preserving an unprovided title requires an explicit preserve reference or clarification")
    if not unresolved:
        Draft202012Validator(native).validate(arguments)
    return unresolved


_SLOW_CLIENT = None  # lazily created: proxy env (NO_PROXY fix) must be applied before construction


def _slow_client() -> "httpx2.Client":
    """Process-lifetime client so every decision reuses the keep-alive connection (no TCP/TLS per call)."""
    global _SLOW_CLIENT
    if _SLOW_CLIENT is None:
        _SLOW_CLIENT = httpx2.Client(
            timeout=90.0,
            limits=httpx2.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=120.0),
        )
    return _SLOW_CLIENT


def _slow_request(base: str, token: str, body: dict) -> dict:
    """POST with one reconnect retry: a keep-alive connection can be closed by the gateway while idle."""
    payload = {"Authorization": "Bearer" + " " + token}
    for attempt in (1, 2):
        try:
            response = _slow_client().post(base + "/chat/completions", headers=payload, json=body)
            response.raise_for_status()
            return response.json()
        except httpx2.RemoteProtocolError:
            if attempt == 2:
                raise
            globals()["_SLOW_CLIENT"] = None      # drop the stale pool and retry once
    raise ModelError("Slow model transport failed")


def warm_slow_model() -> dict:
    """One tiny completion so the gateway/model is loaded and the connection is established.

    Called while the user is still speaking, so the cold-start cost never lands on the critical path.
    """
    settings = {name: os.environ.get(name, "").strip()
                for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY", "SLOW_MODEL")}
    if any(not value for value in settings.values()):
        return {"warm_ms": None, "skipped": "no configuration"}
    base = settings["LITELLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    started = time.monotonic()
    try:
        _slow_request(base, settings["LITELLM_API_KEY"],
                      {"model": settings["SLOW_MODEL"], "stream": False, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "ping"}]})
    except Exception as error:                      # PoC: warming is best effort, never fatal
        return {"warm_ms": (time.monotonic() - started) * 1000, "error": type(error).__name__}
    return {"warm_ms": (time.monotonic() - started) * 1000}


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


def slow_turn(brief: dict, messages: list[dict], schemas: list[dict], *, single_next_step: bool = False) -> dict:
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
    if model == "deepseek-v4.1-flash":
        body["extra_body"] = {"thinking": {"type": "disabled"}}
    if single_next_step:
        body["parallel_tool_calls"] = False
        body["tools"] = [tool for tool in body["tools"] if tool["function"]["name"] != "request_tools"]
        body["messages"][0]["content"] += (
            "\n本轮仅决定一个下一步动作。只调用一个业务工具或 ask_user/finish；"
            "缺信息或没有合适候选工具时用 ask_user；不追加披露，不用 request_tools。"
        )
    if single_next_step and brief.get("tool_set") == "multidomain":
        body["tool_choice"] = "required"
        body["messages"][0]["content"] = MULTIDOMAIN_NEXT_STEP_PROMPT + json.dumps(
            brief, ensure_ascii=False, allow_nan=False)
        body["tools"] = [tool for tool in body["tools"]
                         if tool["function"]["name"] not in {t["name"] for t in CONTROL_TOOLS}] + [
            {"type": "function", "function": {
                "name": tool["name"], "description": tool["description"],
                "parameters": tool["input_schema"],
            }} for tool in MULTIDOMAIN_CONTROLS]
    try:
        data = _slow_request(base, token, body)
    except httpx2.HTTPError as error:
        raise ModelError(f"Slow model transport failed: {type(error).__name__}") from None
    except ValueError:
        raise ModelError("Slow model returned invalid JSON") from None
    return _slow_response(data)


# Experimental kernel; legacy slow/default execution above is unchanged.
REOPT_ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"enum": ["actions", "clarify", "incomplete", "observe"]},
        "goal": {"type": "object"},
        "actions": json.loads(json.dumps(PLAN_SCHEMA["properties"]["steps"])),
        "continue_after_results": {"type": "boolean"},
        "question": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["kind", "goal", "actions", "continue_after_results", "question", "reason"],
    "additionalProperties": False,
}


def _reopt_validate(data: object, schema: dict) -> None:
    try:
        json.dumps(data, allow_nan=False)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(data)
    except (TypeError, ValueError, RecursionError, ValidationError, SchemaError):
        raise ModelError("Invalid experimental protocol schema or data") from None


def _reopt_future_ref(value: object) -> bool:
    # Mechanical syntax exclusion only; not natural-language guess detection.
    if isinstance(value, dict):
        return "$ref" in value or any(_reopt_future_ref(v) for v in value.values())
    if isinstance(value, list):
        return any(_reopt_future_ref(v) for v in value)
    return isinstance(value, str) and "${" in value


def parse_reopt_envelope(data: dict, schemas: list[dict], tool_contracts: list[dict] | None = None) -> dict:
    """Validate syntax, never authorize/execute; candidate goal is not evidence."""
    _reopt_validate(data, REOPT_ENVELOPE_SCHEMA)
    kind, actions, question, reason = (data[k] for k in ("kind", "actions", "question", "reason"))
    concrete = lambda value: isinstance(value, str) and bool(value.strip())
    valid = {
        "actions": bool(actions) and question is None and reason is None,
        "clarify": not actions and concrete(question) and reason is None,
        "incomplete": not actions and question is None and concrete(reason),
        "observe": not actions and question is None and reason is None,
    }
    if not valid[kind] or (data["continue_after_results"] and kind != "actions"):
        raise ModelError("Illegal experimental envelope combination")
    disclosed = {}
    for schema in schemas:
        if not isinstance(schema, dict) or not isinstance(schema.get("name"), str) or (
            schema["name"] in disclosed or not isinstance(schema.get("parameters"), dict)
        ):
            raise ModelError("Invalid experimental tool disclosure")
        disclosed[schema["name"]] = schema["parameters"]
    contracts = {}
    for contract in tool_contracts or []:
        if not isinstance(contract, dict) or not isinstance(contract.get("name"), str) or (
            contract["name"] in contracts or not isinstance(contract.get("parameters"), dict)
        ):
            raise ModelError("Invalid experimental tool contracts")
        contracts[contract["name"]] = contract
    side_effects = 0
    for action in actions:
        name, arguments = action["tool"], action["arguments"]
        if name not in disclosed or name in {t["name"] for t in CONTROL_TOOLS}:
            raise ModelError("Undisclosed or control tool in experimental actions")
        if tool_contracts is not None and (name not in contracts or contracts[name]["parameters"] != disclosed[name]):
            raise ModelError("Experimental tool contract mismatch")
        if _reopt_future_ref(arguments):
            raise ModelError("Future-result references are forbidden")
        _reopt_validate(arguments, disclosed[name])
        if tool_contracts is not None:
            effect = contracts[name].get("effect", {})
            if "parameter" in effect:
                value = arguments.get(effect["parameter"])
                mode = "true" if value is True else "false" if value is False else "unspecified"
                effect = effect.get("modes", {}).get(mode, {})
            side_effects += effect.get("kind") != "read_only"
            if side_effects > 1:
                raise ModelError("Experimental round allows at most one side effect")
    return json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))


def normalize_reopt_react(normalizedresponse: dict, schemas: list[dict], tool_contracts: list[dict] | None = None) -> dict:
    """Return common envelope plus unmodified opaque response for future rounds."""
    if not isinstance(normalizedresponse, dict) or not isinstance(normalizedresponse.get("content"), list):
        raise ModelError("Invalid experimental ReAct response")
    _slow_messages([{"role": "assistant", "content": normalizedresponse["content"]}])
    calls = [b for b in normalizedresponse["content"] if b.get("type") == "tool_use"]
    if not calls or len({c["id"] for c in calls}) != len(calls):
        raise ModelError("Missing calls or duplicate experimental identifiers")
    controls = [c for c in calls if c["name"] in {t["name"] for t in CONTROL_TOOLS}]
    data = {"kind": "actions", "goal": {}, "actions": [], "continue_after_results": True,
            "question": None, "reason": None}
    if controls:
        if len(calls) != 1 or controls[0]["name"] == "request_tools":
            raise ModelError("Experimental supplementation or mixed controls forbidden")
        call = controls[0]
        schema = next(t["input_schema"] for t in CONTROL_TOOLS if t["name"] == call["name"])
        _reopt_validate(call["input"], schema)
        data.update(kind="observe" if call["name"] == "finish" else "clarify",
                    continue_after_results=False, question=call["input"].get("question"))
    else:
        data["actions"] = [{"tool": c["name"], "arguments": c["input"]} for c in calls]
    return {"envelope": parse_reopt_envelope(data, schemas, tool_contracts), "_opaque": normalizedresponse}


def check_reopt_outcome(facts: list[dict], *, goal_verified: bool = False,
                        user_confirmation_missing: bool = False, deadline_expired: bool = False,
                        cancelled: bool = False, confirmation_remaining: int = 0,
                        no_op_verified: bool = False, state_goal: bool = False, event_goal: bool = False) -> dict:
    """Use runtime facts/independent verification, never model argument echoes."""
    if not isinstance(facts, list) or any(not isinstance(f, dict) or f.get("status") not in
            {"effective", "accepted", "failed", "rejected", "unknown"} for f in facts):
        raise ModelError("Invalid experimental outcome facts")
    if type(confirmation_remaining) is not int or confirmation_remaining < 0:
        raise ModelError("Invalid confirmation budget")
    if any(type(v) is not bool for v in (goal_verified, user_confirmation_missing, deadline_expired,
                                        cancelled, no_op_verified, state_goal, event_goal)):
        raise ModelError("Verification flags must be runtime booleans")
    unresolved = any(f["status"] == "unknown" or (f["status"] == "effective" and f.get("verified") is not True)
                     for f in facts)
    accepted = any(f["status"] == "accepted" for f in facts)
    stopped = deadline_expired or cancelled
    if unresolved or (accepted and (stopped or confirmation_remaining == 0)):
        assessment, status = "未知", "incomplete"
    elif accepted:
        assessment, status = "待确认", "waiting_confirmation"
    elif any(f["status"] in {"failed", "rejected"} for f in facts):
        assessment, status = "不满足", "incomplete"
    elif user_confirmation_missing:
        assessment, status = "待确认", "incomplete" if stopped else "waiting_confirmation"
    elif not stopped and goal_verified and (bool(facts) or (no_op_verified and state_goal and not event_goal)):
        assessment, status = "已满足", "done"
    else:
        assessment, status = "不满足", "incomplete"
    return {"assessment": assessment, "status": status, "query_count": 0}


def evaluate_reopt_exit(envelope_or_finish: dict | str, facts: list[dict], *, confirm=None, **flags) -> dict:
    """Common exit; confirm(facts)->replacement facts is a read-only adapter.

    The caller owns independent verification and deadline/cancellation snapshots.
    This synchronous helper is not a scheduler, dispatcher, CAS or resource manager.
    """
    kind = envelope_or_finish if isinstance(envelope_or_finish, str) else envelope_or_finish.get("kind")
    if kind not in {"finish", "observe", "actions", "clarify", "incomplete"}:
        raise ModelError("Invalid experimental exit")
    flags = dict(flags)
    remaining = flags.get("confirmation_remaining", 0)
    if type(remaining) is not int or remaining < 0:
        raise ModelError("Invalid confirmation budget")
    flags["confirmation_remaining"] = min(3, remaining)
    if kind == "clarify":
        flags["user_confirmation_missing"] = True
    outcome = check_reopt_outcome(facts, **flags)
    queries = 0
    while confirm is not None and outcome["status"] == "waiting_confirmation" and (
        any(f["status"] == "accepted" for f in facts) and flags["confirmation_remaining"] > 0
    ):
        facts = confirm(json.loads(json.dumps(facts, allow_nan=False)))
        flags["confirmation_remaining"] -= 1
        queries += 1
        outcome = check_reopt_outcome(facts, **flags)
    return {**outcome, "query_count": queries}


def run_reopt_round(response: dict, encoding: str, schemas: list[dict], facts: list[dict], *,
                    tool_contracts: list[dict] | None = None, token_valid: bool = True,
                    dispatch=None, confirm=None, last_round: bool = False, **flags) -> dict:
    """Opt-in callback harness only; no connection to legacy or real dispatch."""
    if token_valid is not True:
        raise ModelError("Experimental authorization token invalid")
    if encoding not in {"envelope", "react"}:
        raise ModelError("Unsupported experimental encoding")
    normalized = normalize_reopt_react(response, schemas, tool_contracts) if encoding == "react" else {
        "envelope": parse_reopt_envelope(response, schemas, tool_contracts)}
    envelope = normalized["envelope"]
    initial = check_reopt_outcome(facts, **flags)
    if initial["status"] == "done":
        return {**normalized, **initial, "slow_continuation": False}
    if envelope["kind"] == "actions" and dispatch is not None and not (
        flags.get("cancelled") or flags.get("deadline_expired") or initial["assessment"] in {"未知", "待确认"}
    ):
        facts = list(facts)
        for action in envelope["actions"]:
            fact = dispatch(action)
            facts.append(fact)
            if fact.get("status") != "effective" or fact.get("verified") is not True:
                break
    outcome = evaluate_reopt_exit(envelope, facts, confirm=confirm, **flags)
    continuation = envelope["kind"] == "actions" and envelope["continue_after_results"] and not last_round and (
        outcome["assessment"] == "不满足" and bool(facts) and
        all(f["status"] == "effective" and f.get("verified") is True for f in facts) and
        not flags.get("cancelled") and not flags.get("deadline_expired")
    )
    return {**normalized, **outcome, "slow_continuation": bool(continuation)}


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
