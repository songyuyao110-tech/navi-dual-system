"""Finite fast path evaluation: run the case corpus and emit metrics + the short report.

    # real KEV (tool shortlist + parameter filling) and real DeepSeek on every MISS
    .venv/bin/python -B eval/finite_fast_eval.py --kev real --llm real --out /tmp/ff-real.json

    # fully offline: mock shortlist + mock KEV answers, proving the gate logic only
    .venv/bin/python -B eval/finite_fast_eval.py --kev mock --llm skip --out /tmp/ff-mock.json

    # compose the report from the JSON runs above (adds --notes, ends with the DONE line)
    .venv/bin/python -B eval/finite_fast_eval.py --report /Users/YUYSONG/Work/tmp/poc-finite-fast-path-report.md \
        --notes /Users/YUYSONG/Work/tmp/poc-finite-fast-path-notes.md

A HIT means the turn was routed to kev_finite_fast and the LLM decision was never called; a MISS
means try_finite_fast_decide() returned None and the existing decide() ran. The report keeps those
two apart so a MISS group false HIT (the failure this PoC must not have) is visible on its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from core_pipeline_run import MULTIDOMAIN_TOOLS, decide, select_tools
from finite_fast_path import load_allowlist, try_finite_fast_decide
from model_clients import multidomain_disclosure
from registry import load_registry
from turn_session import TurnConfig

CASES_PATH = HERE / "fixtures/finite-fast-cases.jsonl"
DONE_LINE = "DONE finite-fast-path 2026-10-19"
TOOL_SET = "multidomain"
CONTEXT = {"home_address_configured": True, "company_address_configured": True, "synthetic": True}
REQUIRED = {"id", "utterance", "expect_route", "expect_tool", "expect_arguments", "note"}


# ---------- corpus ----------

def load_cases(path=CASES_PATH):
    cases, ids = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = REQUIRED - set(case)
        unknown = set(case) - REQUIRED - {"group", "config"}
        if missing or unknown:
            raise ValueError(f"{path}:{number}: bad fields missing={sorted(missing)} unknown={sorted(unknown)}")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError(f"{path}:{number}: duplicate or empty id")
        if case["expect_route"] not in {"kev", "llm"}:
            raise ValueError(f"{path}:{number}: expect_route must be kev|llm")
        if not isinstance(case["note"], str) or not case["note"].strip():
            raise ValueError(f"{path}:{number}: note must be nonempty")
        if (case["expect_tool"] is None) != (case["expect_arguments"] is None):
            raise ValueError(f"{path}:{number}: expect_tool/expect_arguments must both be set or both null")
        if case["expect_route"] == "kev" and case["expect_tool"] is None:
            raise ValueError(f"{path}:{number}: a kev case needs an expected tool")
        case.setdefault("group", "ungrouped")
        case.setdefault("config", {})
        if not isinstance(case["config"], dict):
            raise ValueError(f"{path}:{number}: config must be an object")
        ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("empty corpus")
    return cases


# ---------- mock KEV (offline proof of the gate logic only) ----------

# Shortlist the mock selector returns per utterance; mirrors what real KEV produced on 2026-10-19 so
# the gate sees the same shapes without a tunnel. Scores are the real measured values, rounded.
MOCK_SHORTLIST = {
    "导航回家": [("navi_to_special_poi", 0.973), ("navi_start_route", 0.773)],
    "带我回公司": [("navi_to_special_poi", 0.963), ("navi_start_route", 0.766)],
    "打开公司地址设置页面": [("navi_open_address_page", 0.968), ("navi_search_in_address_page", 0.303)],
    "打开空调": [("climate_set_power", 0.950), ("climate_set_auto", 0.298)],
    "关闭空调": [("climate_set_power", 0.940), ("climate_get_status", 0.242)],
    "把AC关掉": [("climate_set_ac", 0.952), ("climate_get_status", 0.348)],
    "打开空调自动模式": [("climate_set_auto", 0.965), ("climate_get_status", 0.259)],
    "音乐下一首": [("music_skip", 0.963), ("music_get_now_playing", 0.381)],
    "单曲循环": [("music_set_mode", 0.945), ("music_get_now_playing", 0.315)],
    "通话麦克风静音": [("phone_set_mic_mute", 0.936), ("phone_get_status", 0.519)],
    "打开氛围灯": [("light_set_ambient_power", 0.944), ("light_set_ambient_brightness", 0.375)],
    "打开后备箱尾门": [("body_set_tailgate", 0.964), ("body_get_status", 0.288)],
    "导航去望京": [("navi_search_keyword", 0.901), ("navi_start_route", 0.893)],
    "导航去中关村软件园": [("navi_search_keyword", 0.897), ("navi_start_route", 0.885)],
    "音乐模式随便换换": [("music_set_mode", 0.910), ("light_set_ambient_effect", 0.179)],
    "氛围灯那个": [("light_set_ambient_power", 0.846), ("light_set_ambient_effect", 0.705)],
    "空调搞一下": [("climate_set_power", 0.841), ("climate_set_ac", 0.450)],
    "把氛围灯调成蓝色": [("light_set_ambient_color", 0.957), ("light_get_status", 0.342)],
    # Extra phrasing used by tmp/finite-fast-extra-cases.jsonl (same measured scores, 2026-10-19).
    "回家": [("navi_to_special_poi", 0.968), ("navi_start_route", 0.700)],
    "帮我开下空调": [("climate_set_power", 0.940), ("climate_set_ac", 0.300)],
    "把尾门关上": [("body_set_tailgate", 0.966), ("body_get_status", 0.300)],
    "打开后备箱": [("body_set_tailgate", 0.963), ("body_get_status", 0.300)],
    "循环播放": [("music_set_mode", 0.961), ("music_get_now_playing", 0.300)],
    "打开导航": [("navi_to_special_poi", 0.777), ("navi_start_route", 0.700)],
    "音乐切一下": [("music_skip", 0.935), ("music_get_now_playing", 0.300)],
    "取消静音": [("system_set_mute", 0.792), ("system_get_settings", 0.500)],
    "有点热": [("climate_set_temperature", 0.788), ("climate_set_ac", 0.500)],
    # Negation guard corpus (real shortlist top-1 measured 2026-10-19; every one of these used to
    # HIT climate_set_power {enabled: false} before the guard existed).
    "别开空调": [("climate_set_power", 0.9218), ("climate_get_status", 0.3991)],
    "别打开空调": [("climate_set_power", 0.9299), ("climate_set_ac", 0.3988)],
    "别开空调了": [("climate_set_power", 0.9162), ("climate_get_status", 0.3991)],
    "不要开空调": [("climate_set_power", 0.9251), ("climate_set_ac", 0.3988)],
    "不用开空调": [("climate_set_power", 0.8957), ("climate_set_ac", 0.6859)],
    "别关空调": [("climate_set_power", 0.8723), ("climate_get_status", 0.3991)],
    # 2026-10-20: the 28 tools added to the 58-tool allowlist, plus the new must-MISS / negation
    # cases. Scores are the measured real-KEV shortlist top-1 values from eval/speculative probes.
    "空调风吹脚": [("climate_set_airflow", 0.9623)],
    "空调切到内循环": [("climate_set_recirculation", 0.9599)],
    "打开后风挡除雾": [("climate_set_rear_defrost", 0.9622), ("climate_set_front_defrost", 0.5979)],
    "打开空气净化": [("climate_set_purification", 0.9651)],
    "把全车车门锁上": [("body_set_door_lock", 0.9528)],
    "折叠后视镜": [("body_set_mirror_fold", 0.9550)],
    "打开后视镜加热": [("body_set_mirror_heat", 0.9657)],
    "副驾座椅什么状态": [("seat_get_status", 0.9707)],
    "前大灯切到自动": [("light_set_headlights", 0.9493)],
    "打开后雾灯": [("light_set_fog", 0.9618)],
    "别开后雾灯": [("light_set_fog", 0.9074), ("light_get_status", 0.5025)],
    "把氛围灯调成蓝色": [("light_set_ambient_color", 0.9568)],
    "把氛围灯颜色换一下": [("light_set_ambient_color", 0.9536)],
    "氛围灯呼吸效果": [("light_set_ambient_effect", 0.9553)],
    "把播客暂停": [("audio_set_playback", 0.9718)],
    "暂停中控视频": [("video_pause", 0.9553)],
    "中控视频切到高清": [("video_set_quality", 0.9305)],
    "中控屏在放什么视频": [("video_get_status", 0.9599)],
    "看下未读短信": [("message_list_unread", 0.9645)],
    "看下未读消息": [("message_list_unread", 0.9635)],
    "看看科技新闻": [("info_get_news", 0.9624)],
    "能量回收调到高": [("drive_set_regeneration", 0.9465)],
    "关闭车道保持": [("drive_set_lane_assist", 0.9518)],
    "打开自动驻车": [("drive_set_auto_hold", 0.9658)],
    "看我未完成的提醒": [("reminder_list", 0.9513)],
    "关闭中控屏": [("system_set_screen_power", 0.9522)],
    "切到深色主题": [("system_set_theme", 0.9552)],
    "主题换一下": [("system_set_theme", 0.8840), ("system_get_settings", 0.5162)],
    "把车机语言改成英文": [("system_set_language", 0.9530), ("system_get_settings", 0.5400)],
    "距离单位改成英里，温度单位改成华氏度": [("system_set_units", 0.9568)],
    "打开蓝牙": [("system_set_bluetooth", 0.9468)],
    "蓝牙弄一下": [("system_set_bluetooth", 0.8690), ("system_pair_device", 0.7003)],
    "不要关蓝牙": [("system_set_bluetooth", 0.8471), ("system_get_settings", 0.6642)],
    "关掉车机WiFi": [("system_set_wifi", 0.9553), ("system_get_settings", 0.5065)],
    "氛围灯调亮一点": [("light_set_ambient_brightness", 0.9238), ("light_get_status", 0.5024)],
    "地图放大一点": [("navi_set_map_zoom", 0.9358)],
    "视频快进三十秒": [("video_seek", 0.8296), ("video_get_status", 0.5047)],
    "把屏幕亮度调到百分之七十": [("system_set_brightness", 0.9627)],
    "空调风向往哪吹来着": [("climate_set_airflow", 0.9423), ("climate_get_status", 0.6653)],
    # 2026-10-20: entries for the remaining corpus rows (measured real-KEV top-1/top-2 scores),
    # so --kev mock covers all 98 cases - the offline proof of the gate, no tunnel needed.
    "打开主驾阅读灯": [("light_set_reading", 0.9682)],
    "地图切到三维视图": [("navi_set_map_view", 0.9601)],
    "导航播报静音": [("navi_set_voice_guidance", 0.9587)],
    "打开前风挡除雾": [("climate_set_front_defrost", 0.9586)],
    "打开空调温区同步": [("climate_set_sync", 0.9686)],
    "关掉主驾按摩": [("seat_stop_massage", 0.963)],
    "打开充电口盖": [("energy_set_port", 0.9583)],
    "切换到运动模式": [("drive_set_mode", 0.9623)],
    "打开免打扰": [("system_set_do_not_disturb", 0.958)],
    "打开左侧儿童锁": [("body_set_child_lock", 0.9603)],
    "把短信通知关掉": [("message_set_notifications", 0.9577)],
    "雨刷调到自动": [("body_set_wipers", 0.9712)],
    "导航去三里屯太古里": [("navi_start_route", 0.8877), ("navi_search_keyword", 0.8859)],
    "放一首稻香": [("music_play_track", 0.9178), ("music_search", 0.7714)],
    "找找附近的充电站": [("energy_search_chargers", 0.9248), ("energy_get_charger_details", 0.6792)],
    "空调调到二十四度": [("climate_set_temperature", 0.9522)],
    "主驾座椅加热开到二档": [("seat_set_heating", 0.9678)],
    "把车窗关上": [("body_set_window", 0.8973)],
    "打开天窗": [("body_set_sunroof", 0.9402)],
    "视频那个继续播": [("video_resume", 0.9307), ("video_get_status", 0.548)],
    "字幕换成英文": [("video_set_subtitles", 0.953)],
    "看一眼能耗": [("vehicle_get_energy_usage", 0.9327)],
    "看下我的行程": [("calendar_list_events", 0.9502)],
    "把迎宾灯弄一下": [("light_set_welcome", 0.9679), ("light_get_status", 0.5395)],
    "远光辅助开着没": [("light_get_status", 0.9032), ("light_set_high_beam_assist", 0.8794)],
    "方向盘手感调一下": [("drive_set_steering_weight", 0.9578)],
    "座椅按摩搞一下": [("seat_start_massage", 0.9615)],
    "不要打开座椅通风": [("seat_set_ventilation", 0.6247)],
    "把车速限制到八十": [("drive_set_cruise_speed", 0.8617)],
    "帮我订个明天上午的会议室": [("calendar_create_event", 0.9127), ("calendar_list_events", 0.6671)],
    "把阅读灯的": [("light_set_reading", 0.8795)],
    "空调温度设为": [("climate_set_temperature", 0.9325)],
    "打开那个": [("navi_search_keyword", 0.7069), ("navi_show_poi_list", 0.6128)],
    "切换到第": [],
}
AMBIGUOUS = ("随便", "搞一下", "弄一下", "那个")
MOCK_CONFIDENT, MOCK_UNSURE = 0.95, 0.45
_SNAPSHOT = None


def snapshot():
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = load_registry(MULTIDOMAIN_TOOLS, frozenset({"mock"}))
    return _SNAPSHOT


def mock_select_tools(utterance):
    if utterance not in MOCK_SHORTLIST:
        raise ValueError("mock shortlist has no entry for: " + utterance)
    ranked = MOCK_SHORTLIST[utterance]      # an empty list is a real measured result (no candidate)
    names = [name for name, _ in ranked]
    registry = snapshot()
    native = registry.schemas(frozenset(names))
    schemas = multidomain_disclosure(native, {name: registry.tools[name].domain for name in names})
    return {"scores": {"tool:" + name: score for name, score in ranked}, "selected": names,
            "schemas": schemas, "native_schemas": native,
            "domains": sorted({registry.tools[name].domain for name in names}),
            "timing": {"stage1_ms": 0.0, "stage2_ms": 0.0, "total_ms": 0.0}, "mock": True}


# Mock only: KEV reads the verb and flips polarity when the sentence is negated (别开空调 -> false
# at 0.97 confidence). Mirrored here so the offline run reproduces the measured answers - the gate
# itself has no cue list, it always asks KEV.
MOCK_NEGATIONS = ("别", "不要", "不用", "甭", "勿", "禁止")


def _pick(text, mapping):
    """First keyword that appears in the utterance; None when the text does not say it."""
    for needle, value in mapping:
        if needle in text:
            return value
    return None


# Per-(tool, parameter) keyword fills for the tools added on 2026-10-20. Booleans stay on the
# generic verb-polarity rule below; enums that share a parameter name across tools need this table.
MOCK_SPECIFIC = {
    ("climate_set_airflow", "direction"): lambda t: _pick(t, (("脚", "feet"), ("面", "face"), ("脸", "face"))),
    ("climate_set_recirculation", "mode"): lambda t: _pick(t, (("内循环", "inside"), ("外循环", "outside"),
                                                               ("自动", "auto"))),
    ("body_set_door_lock", "door"): lambda t: _pick(t, (("全车", "all"), ("主驾", "driver"), ("副驾", "passenger"),
                                                        ("左后", "rear_left"), ("右后", "rear_right"))),
    ("body_set_door_lock", "locked"): lambda t: ("false" if "解锁" in t else "true"
                                                 if any(w in t for w in ("锁上", "上锁", "锁车")) else None),
    ("body_set_mirror_fold", "folded"): lambda t: ("true" if "折叠" in t else "false" if "展开" in t else None),
    ("light_set_reading", "position"): lambda t: _pick(t, (("主驾", "driver"), ("副驾", "passenger"),
                                                           ("左后", "rear_left"), ("右后", "rear_right"),
                                                           ("全车", "all"), ("所有", "all"))),
    ("navi_set_map_view", "view"): lambda t: _pick(t, (("三维", "three_d"), ("二维", "two_d"),
                                                       ("卫星", "satellite"))),
    ("navi_set_voice_guidance", "mode"): lambda t: _pick(t, (("静音", "muted"), ("简洁", "brief"),
                                                             ("完整", "full"))),
    ("seat_stop_massage", "seat"): lambda t: _pick(t, (("主驾", "driver"), ("副驾", "passenger"))),
    ("drive_set_mode", "mode"): lambda t: _pick(t, (("运动", "sport"), ("经济", "eco"), ("舒适", "comfort"))),
    ("body_set_child_lock", "side"): lambda t: _pick(t, (("左侧", "left"), ("右侧", "right"), ("两侧", "both"))),
    ("message_set_notifications", "channel"): lambda t: _pick(t, (("短信", "sms"), ("即时", "im"))),
    ("body_set_wipers", "mode"): lambda t: _pick(t, (("自动", "auto"), ("关闭", "off"), ("低速", "low"),
                                                     ("高速", "high"))),
    ("seat_get_status", "seat"): lambda t: _pick(t, (("副驾", "passenger"), ("主驾", "driver"),
                                                     ("左后", "rear_left"), ("右后", "rear_right"))),
    ("light_set_headlights", "mode"): lambda t: _pick(t, (("自动", "auto"), ("近光", "low_beam"),
                                                          ("关闭", "off"))),
    ("light_set_ambient_color", "color"): lambda t: _pick(t, (("蓝", "blue"), ("红", "red"), ("绿", "green"),
                                                              ("紫", "purple"), ("琥珀", "amber"), ("黄", "amber"),
                                                              ("白", "white"))),
    ("light_set_ambient_effect", "effect"): lambda t: _pick(t, (("呼吸", "breathing"), ("律动", "music_sync"),
                                                                ("静态", "static"))),
    ("audio_set_playback", "source"): lambda t: _pick(t, (("播客", "podcast"), ("广播", "radio"),
                                                          ("有声书", "audiobook"))),
    ("audio_set_playback", "action"): lambda t: _pick(t, (("暂停", "pause"), ("继续", "resume"))),
    ("video_pause", "screen"): lambda t: _pick(t, (("中控", "center"), ("左后", "rear_left"), ("右后", "rear_right"))),
    ("video_set_quality", "screen"): lambda t: _pick(t, (("中控", "center"), ("左后", "rear_left"),
                                                         ("右后", "rear_right"))),
    ("video_set_quality", "quality"): lambda t: _pick(t, (("高清", "hd"), ("标清", "sd"), ("自动", "auto"))),
    ("video_get_status", "screen"): lambda t: _pick(t, (("中控", "center"), ("左后", "rear_left"),
                                                        ("右后", "rear_right"))),
    ("message_list_unread", "channel"): lambda t: _pick(t, (("短信", "sms"), ("即时", "im"))),
    ("info_get_news", "category"): lambda t: _pick(t, (("科技", "technology"), ("财经", "finance"),
                                                       ("体育", "sports"), ("综合", "general"))),
    ("drive_set_regeneration", "level"): lambda t: _pick(t, (("高", "high"), ("中", "medium"), ("低", "low"),
                                                             ("自动", "auto"))),
    ("reminder_list", "status"): lambda t: _pick(t, (("未完成", "pending"), ("全部", "all"))),
    ("system_set_screen_power", "screen"): lambda t: _pick(t, (("中控", "center"), ("左后", "rear_left"),
                                                               ("右后", "rear_right"))),
    ("system_set_theme", "theme"): lambda t: _pick(t, (("深色", "dark"), ("明亮", "light"), ("自动", "auto"))),
    ("system_set_language", "language"): lambda t: _pick(t, (("英文", "en_US"), ("中文", "zh_CN"),
                                                             ("德文", "de_DE"))),
    ("system_set_units", "distance"): lambda t: _pick(t, (("英里", "mi"), ("公里", "km"))),
    ("system_set_units", "temperature"): lambda t: _pick(t, (("华氏", "fahrenheit"), ("摄氏", "celsius"))),
}


def _mock_value(utterance, tool, param):
    text = utterance
    specific = MOCK_SPECIFIC.get((tool, param))
    if specific is not None:
        return specific(text)
    if param in {"poi_type", "type"}:
        return "company" if "公司" in text else "home" if "家" in text else None
    if tool == "music_skip" and param == "direction":
        return "previous" if "上一" in text else "next" if "下一" in text else None
    if tool == "music_set_mode" and param == "mode":
        if "单曲" in text:
            return "repeat_one"
        if "随机" in text:
            return "shuffle"
        if "顺序" in text:
            return "sequence"
        if "列表" in text or "全部" in text:
            return "repeat_all"
        # Bare "循环" is repeat_one or repeat_all: real KEV scores ~0.60-0.66 here, so stay vague.
        return None
    if param in {"enabled", "open"}:
        if any(word in text for word in ("关闭", "关掉", "关一下", "关上", "关空调")):
            value = "false"
        elif any(word in text for word in ("打开", "开启", "开一下", "开下", "开空调")):
            value = "true"
        elif "开" in text:
            value = "true"
        elif "关" in text:
            value = "false"
        else:
            return None
        if any(cue in text for cue in MOCK_NEGATIONS):
            value = "true" if value == "false" else "false"
        return value
    if param == "muted":
        return "false" if "取消" in text else "true" if "静音" in text else None
    return None


def mock_kev_call(state, questions):
    """Deterministic stand-in for KEV: keyword fill, low confidence when the text is vague."""
    payload = json.loads(state)
    utterance, tool = payload.get("utterance", ""), (payload.get("tool") or {}).get("name", "")
    vague = any(word in utterance for word in AMBIGUOUS)
    # KEV's own confidence collapses when negation meets the closing verb (0.4165 on 别关空调), which
    # is why that phrasing MISSes on param_tau without anyone matching words.
    vague = vague or (any(cue in utterance for cue in MOCK_NEGATIONS) and "关" in utterance)
    answers = {}
    for name, question in questions.items():
        criteria = question["criteria"]
        value = _mock_value(utterance, tool, name)
        if value is None or vague:
            # Illegal-free but low confidence: the caller must reject it via param_tau.
            value = value or next(iter(criteria))
            confidence = MOCK_UNSURE
        else:
            confidence = MOCK_CONFIDENT
        probabilities = {key: (confidence if key == value else (1 - confidence) / max(1, len(criteria) - 1))
                         for key in criteria}
        answers[name] = {"type": "choice", "choice": value, "confidence": confidence,
                         "probabilities": probabilities}
    return {"model": "kev-latest", "answers": answers, "latency_ms": 0.0, "mock": True}


# ---------- one case ----------

def run_case(case, kev_mode, llm_mode, llm_baseline):
    overrides = {"tool_set": TOOL_SET, **case["config"]}
    cfg = TurnConfig(**overrides)
    began = time.monotonic()
    selection = select_tools(case["utterance"], TOOL_SET, dict(CONTEXT)) if kev_mode == "real" \
        else mock_select_tools(case["utterance"])
    select_ms = (time.monotonic() - began) * 1000
    record, fast = {}, None
    if cfg.finite_fast_path:
        fast = try_finite_fast_decide(case["utterance"], selection["selected"], selection["scores"],
                                      selection["schemas"], cfg, record=record,
                                      kev_call_fn=mock_kev_call if kev_mode == "mock" else None)
    decision, decision_ms, head, route, baseline_ms = None, None, None, "llm", None
    if fast is not None:
        route, decision, decision_ms = "kev", fast["decision"], fast["llm_ms"]
        head = fast["log_fields"]["decision_head"]
    elif llm_mode == "real":
        outcome = decide(case["utterance"], selection["selected"], selection["schemas"], TOOL_SET,
                         dict(CONTEXT), native_schemas=selection.get("native_schemas"),
                         available_domains=selection.get("domains"))
        decision, decision_ms = outcome["decision"], outcome["llm_ms"]
        head = (outcome.get("log_fields") or {}).get("decision_head")
    if llm_mode == "real" and llm_baseline and route == "kev":
        # Same utterance, same environment: what the LLM decision would have cost.
        began = time.monotonic()
        outcome = decide(case["utterance"], selection["selected"], selection["schemas"], TOOL_SET,
                         dict(CONTEXT), native_schemas=selection.get("native_schemas"),
                         available_domains=selection.get("domains"))
        baseline_ms = outcome["llm_ms"]
        baseline_head = (outcome.get("log_fields") or {}).get("decision_head")
        baseline_decision = outcome["decision"]
    else:
        baseline_head, baseline_decision = None, None
    tool = (decision or {}).get("tool")
    arguments = (decision or {}).get("arguments")
    return {
        "id": case["id"], "group": case["group"], "utterance": case["utterance"],
        "note": case["note"], "config": case["config"],
        "expect_route": case["expect_route"], "expect_tool": case["expect_tool"],
        "expect_arguments": case["expect_arguments"],
        "route": route, "decision_head": head, "decision": decision,
        "tool": tool, "arguments": arguments,
        "route_ok": route == case["expect_route"],
        "tool_ok": None if case["expect_tool"] is None else tool == case["expect_tool"],
        "args_ok": None if case["expect_arguments"] is None else arguments == case["expect_arguments"],
        "false_hit": case["expect_route"] == "llm" and route != "llm",
        "missed_hit": case["expect_route"] == "kev" and route != "kev",
        "finite_fast_reason": record.get("finite_fast_reason"),
        "finite_fast_detail": record.get("finite_fast_detail"),
        "finite_fast_error": record.get("finite_fast_error"),
        "finite_fast_tool": record.get("finite_fast_tool"),
        "tool_score": record.get("finite_fast_tool_score"),
        "param_confidences": record.get("finite_fast_param_confidences"),
        "select_ms": select_ms, "decision_ms": decision_ms,
        "llm_baseline_ms": baseline_ms, "llm_baseline_head": baseline_head,
        "llm_baseline_ok": None if baseline_decision is None else
        (baseline_decision.get("tool") == case["expect_tool"]
         and baseline_decision.get("arguments") == case["expect_arguments"]),
    }


# ---------- metrics ----------

def p50(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.median(values) if values else None


def summarise(rows, kev_mode, llm_mode, elapsed_ms):
    expected_kev = [row for row in rows if row["expect_route"] == "kev"]
    expected_llm = [row for row in rows if row["expect_route"] == "llm"]
    hits = [row for row in rows if row["route"] == "kev"]
    correct = [row for row in hits if row["tool_ok"] and row["args_ok"]]
    llm_rows = [row for row in rows if row["route"] == "llm"]
    negated = [row for row in rows if row["group"] == "negation"]
    baseline = [row["llm_baseline_ms"] for row in rows if row["llm_baseline_ms"] is not None]
    groups = {}
    for row in rows:
        bucket = groups.setdefault(row["group"], {"n": 0, "route_ok": 0, "false_hit": 0, "correct": 0})
        bucket["n"] += 1
        bucket["route_ok"] += bool(row["route_ok"])
        bucket["false_hit"] += bool(row["false_hit"])
        bucket["correct"] += bool(row["route"] == "kev" and row["tool_ok"] and row["args_ok"])
    metrics = {
        "case_count": len(rows),
        "expected_kev": len(expected_kev), "expected_llm": len(expected_llm),
        "hit_rate": len(hits) / len(expected_kev) if expected_kev else None,
        "hit_accuracy": len(correct) / len(hits) if hits else None,
        "miss_false_hits": sum(row["false_hit"] for row in expected_llm),
        "missed_hits": sum(row["missed_hit"] for row in expected_kev),
        "negation_cases": len(negated),
        "negation_hits": sum(1 for row in negated if row["route"] == "kev"),
        "negation_correct": sum(1 for row in negated
                                if row["route"] == "kev" and row["tool_ok"] and row["args_ok"]),
        # Part B: a "top-1 is not in the allowlist" MISS must carry the tool name + a real score.
        "not_allowlisted_without_score": sum(
            1 for row in rows if row["finite_fast_reason"] == "not_allowlisted"
            and not isinstance(row["tool_score"], (int, float))),
        "route_accuracy": sum(row["route_ok"] for row in rows) / len(rows),
        "kev_decision_p50_ms": p50([row["decision_ms"] for row in hits]),
        "llm_decision_p50_ms": p50([row["decision_ms"] for row in llm_rows]),
        "llm_baseline_p50_ms": p50(baseline),
        "llm_baseline_tool_args_ok": (sum(1 for row in rows if row["llm_baseline_ok"]) /
                                      max(1, sum(1 for row in rows if row["llm_baseline_ok"] is not None))
                                      if baseline else None),
        "select_p50_ms": p50([row["select_ms"] for row in rows]),
        "finite_fast_errors": sum(1 for row in rows if row["finite_fast_error"]),
    }
    passed = (metrics["miss_false_hits"] == 0 and metrics["missed_hits"] == 0
              and metrics["negation_correct"] == metrics["negation_hits"]
              and metrics["not_allowlisted_without_score"] == 0
              and (metrics["hit_accuracy"] or 0) >= 0.80 and metrics["finite_fast_errors"] == 0)
    return {"mode": {"kev": kev_mode, "llm": llm_mode}, "tool_set": TOOL_SET,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_ms": elapsed_ms,
            "allowlist": load_allowlist()["order"], "metrics": metrics, "groups": groups,
            "outcome": "PASS" if passed else "FAIL", "cases": rows}


# ---------- report ----------

def _ms(value):
    return "—" if value is None else f"{value:.0f} ms"


def _pct(value):
    return "—" if value is None else f"{value * 100:.0f}%"


def compose_report(path, runs, notes_path):
    lines = ["# PoC 报告：有限参 KEV 快路径（finite fast path）", "",
             f"日期：2026-10-19 · spec：`tmp/poc-finite-fast-spec.md` · 用例："
             f"`eval/fixtures/finite-fast-cases.jsonl` · 名单：`eval/fixtures/finite-fast-allowlist.yaml`", "",
             "在现有「KEV 选工具短名单 → 确认后一次 DeepSeek」框架里加了一层 pre-dispatch gate：",
             "短名单第一名在 allowlist 内、参数全为 enum/boolean、KEV 工具分与每个参数 Choice 都过 τ，",
             "就直接产出 `tool_call`（`decision_head=kev_finite_fast`、`routed_by=kev`），**不调用 DeepSeek**；",
             "任何一步不满足就返回 None，走原 `decide()`。执行仍是 mock。", ""]
    for run in runs:
        label = f"KEV={run['mode']['kev']} / LLM={run['mode']['llm']}"
        metrics = run["metrics"]
        lines += [f"## 验收结果 · {label}", "",
                  f"- 结论：**{run['outcome']}** · 用例 {metrics['case_count']} 条 · 总耗时 "
                  f"{run['elapsed_ms'] / 1000:.1f}s",
                  f"- HIT 率：{_pct(metrics['hit_rate'])}（期望 kev 的 {metrics['expected_kev']} 条里 "
                  f"{round((metrics['hit_rate'] or 0) * metrics['expected_kev'])} 条走了快路径）",
                  f"- HIT 正确率（tool+args 全对）：{_pct(metrics['hit_accuracy'])}",
                  f"- MISS 组误 HIT：**{metrics['miss_false_hits']}**（必须 0）· HIT 组漏 HIT："
                  f"{metrics['missed_hits']}",
                  f"- 否定组（{metrics['negation_cases']} 条）：HIT {metrics['negation_hits']} 条、"
                  f"tool+args 全对 **{metrics['negation_correct']}** 条（必须等于 HIT 数）· "
                  f"not_allowlisted MISS 缺 tool_score："
                  f"{metrics['not_allowlisted_without_score']}（必须 0）",
                  f"- 路由准确率：{_pct(metrics['route_accuracy'])} · 快路径异常（finite_fast_error）："
                  f"{metrics['finite_fast_errors']}",
                  f"- 决策耗时 p50：快路径 {_ms(metrics['kev_decision_p50_ms'])} vs 同环境 LLM "
                  f"{_ms(metrics['llm_decision_p50_ms'])}"
                  + (f" · 同句 LLM 基线 {_ms(metrics['llm_baseline_p50_ms'])}" if
                     metrics['llm_baseline_p50_ms'] is not None else ""),
                  f"- KEV 选工具 p50：{_ms(metrics['select_p50_ms'])}", ""]
        lines += ["| 组 | 用例 | 路由正确 | 误 HIT | tool+args 全对 |", "|---|---|---|---|---|"]
        for name, bucket in run["groups"].items():
            lines.append(f"| {name} | {bucket['n']} | {bucket['route_ok']} | {bucket['false_hit']} | "
                         f"{bucket['correct']} |")
        lines.append("")
    chosen = next((run for run in runs if run["mode"]["kev"] == "real"), runs[0])
    lines += [f"## 逐条结果（KEV={chosen['mode']['kev']}）", "",
              "| id | 语料 | 期望 | 实际路由 | tool | args | 耗时 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    for row in chosen["cases"]:
        expected = "kev" if row["expect_route"] == "kev" else "llm"
        decision = "—" if row["decision"] is None else \
            f"`{row['tool']}` " + json.dumps(row["arguments"], ensure_ascii=False)
        reason = row["finite_fast_reason"] or "—"
        if row["finite_fast_error"]:
            reason += f" / {row['finite_fast_error']}"
        if isinstance(row["tool_score"], (int, float)):
            # Part B visibility: every MISS that had a shortlist top-1 also shows its score.
            reason += f" · top1 {row['finite_fast_tool'] or row['tool'] or '—'} {row['tool_score']:.4f}"
        expected_args = "—" if row["expect_arguments"] is None else \
            json.dumps(row["expect_arguments"], ensure_ascii=False)
        lines.append(f"| {row['id']} | {row['utterance']} | {expected} {row['expect_tool'] or ''} "
                     f"{expected_args} | **{row['route']}** | `{row['tool'] or '—'}` | "
                     f"{json.dumps(row['arguments'], ensure_ascii=False) if row['arguments'] is not None else '—'} | "
                     f"{_ms(row['decision_ms'])} | {reason} |")
    lines.append("")
    if notes_path and Path(notes_path).exists():
        lines += [Path(notes_path).read_text().rstrip(), ""]
    lines += ["---", "", DONE_LINE, ""]
    Path(path).write_text("\n".join(lines))
    return path


# ---------- cli ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev", choices=("real", "mock"), default="mock")
    parser.add_argument("--llm", choices=("real", "skip"), default="skip")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, help="write the run summary JSON here")
    parser.add_argument("--no-llm-baseline", action="store_true",
                        help="skip the extra decide() call that measures LLM latency on HIT cases")
    parser.add_argument("--report", type=Path, help="compose the markdown report from --run files")
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="summary JSON to include in the report (repeatable)")
    parser.add_argument("--notes", type=Path, help="markdown inserted before the DONE line")
    args = parser.parse_args(argv)
    if args.report:
        if not args.run:
            parser.error("--report needs at least one --run summary JSON")
        runs = [json.loads(path.read_text()) for path in args.run]
        print(json.dumps({"report": str(compose_report(args.report, runs, args.notes))}, ensure_ascii=False))
        return 0
    cases = load_cases(args.cases)
    began = time.monotonic()
    rows = []
    for case in cases:
        row = run_case(case, args.kev, args.llm, not args.no_llm_baseline)
        rows.append(row)
        print(json.dumps({"id": row["id"], "route": row["route"], "tool": row["tool"],
                          "arguments": row["arguments"], "ok": row["route_ok"],
                          "reason": row["finite_fast_reason"], "ms": row["decision_ms"]},
                         ensure_ascii=False), flush=True)
    summary = summarise(rows, args.kev, args.llm, (time.monotonic() - began) * 1000)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outcome": summary["outcome"], "metrics": summary["metrics"]}, ensure_ascii=False))
    return 0 if summary["outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())