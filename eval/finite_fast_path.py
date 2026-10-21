"""Finite fast path: KEV fills an allowlisted tool's enum/boolean arguments, skipping the LLM.

Gate (see tmp/poc-finite-fast-spec.md §6), evaluated after the KEV tool shortlist:

    selected[0] in allowlist                                   (YAML is the only list source)
      -> scores["tool:<tool>"] >= finite_fast_tool_tau
      -> every parameters.properties entry is enum or boolean   (defensive; no open slots)
      -> one KEV Choice call for all required parameters
      -> each Choice confidence >= finite_fast_param_tau
      -> decision {kind: tool_call, tool, arguments}            (no DeepSeek / slow_turn call)

Anything else - disabled switch, no shortlist, tool not allowlisted, low score, non-finite
schema, KEV unavailable/timeout/malformed, low parameter confidence - returns None so the caller
falls back to the existing decide() path. A failed turn never produces a partial argument set.

Negation is deliberately NOT special-cased: there is no cue list and no string rule in this gate.
KEV fills the polarity and the case corpus measures whether that is right (negated phrasings are
HIT cases with an expected argument), so a wrong fill shows up as a failing case instead of being
silently bypassed by matching words.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import yaml
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALLOWLIST_PATH = FIXTURES / "finite-fast-allowlist.yaml"
# TurnConfig names -> allowlist default keys; the config always wins when it has a value.
DEFAULT_TOOL_TAU = 0.70
DEFAULT_PARAM_TAU = 0.70
DEFAULT_REQUIRE_TOP1 = True
HEAD = "kev_finite_fast"
_ROUTE = "kev"
_CACHE = {}


class AllowlistError(ValueError):
    """The YAML allowlist itself is malformed; never silently widen it."""


# ---------- allowlist ----------

def _spec_error(where, text):
    raise AllowlistError(f"{where}: {text}")


def _validate_param(where, param):
    if not isinstance(param, dict):
        _spec_error(where, "parameter spec must be a mapping")
    unknown = set(param) - {"name", "kind", "values", "labels", "question"}
    if unknown:
        _spec_error(where, "unknown parameter keys " + ", ".join(sorted(unknown)))
    name = param.get("name")
    if not isinstance(name, str) or not name or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in name):
        _spec_error(where, "invalid parameter name")
    kind = param.get("kind")
    if kind not in {"enum", "boolean"}:
        _spec_error(where + "." + name, "kind must be enum or boolean")
    question = param.get("question")
    if not isinstance(question, str) or not question.strip():
        _spec_error(where + "." + name, "question must be nonempty")
    labels = param.get("labels")
    if not isinstance(labels, dict) or not labels:
        _spec_error(where + "." + name, "labels must be a nonempty mapping")
    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        _spec_error(where + "." + name, "labels must map to nonempty strings")
    if kind == "boolean":
        if set(labels) != {"true", "false"}:
            # Criteria keys are JSON object keys, so booleans are addressed as "true"/"false"
            # and mapped back to a Python bool after the answer is validated.
            _spec_error(where + "." + name, 'boolean labels must be exactly "true" and "false"')
        if "values" in param:
            _spec_error(where + "." + name, "boolean parameters do not declare values")
        return {"name": name, "kind": "boolean", "question": question.strip(),
                "values": [True, False], "labels": {key: labels[key].strip() for key in ("true", "false")},
                "criteria": {key: labels[key].strip() for key in ("true", "false")}}
    values = param.get("values")
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        _spec_error(where + "." + name, "enum values must be a nonempty list of unique strings")
    if set(labels) != set(values):
        _spec_error(where + "." + name, "labels must cover exactly the enum values")
    # criteria keys must be schema-legal values, so they are the enum values themselves.
    return {"name": name, "kind": "enum", "question": question.strip(), "values": list(values),
            "labels": {value: labels[value].strip() for value in values},
            "criteria": {value: labels[value].strip() for value in values}}


def _validate_tool(where, tool):
    if not isinstance(tool, dict):
        _spec_error(where, "tool spec must be a mapping")
    unknown = set(tool) - {"name", "description", "params"}
    if unknown:
        _spec_error(where, "unknown tool keys " + ", ".join(sorted(unknown)))
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        _spec_error(where, "tool name must be nonempty")
    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        _spec_error(name, "description must be nonempty")
    params = tool.get("params")
    if not isinstance(params, list) or not params:
        _spec_error(name, "params must be a nonempty list")
    checked = [_validate_param(name, param) for param in params]
    if len({param["name"] for param in checked}) != len(checked):
        _spec_error(name, "duplicate parameter")
    return {"name": name, "description": description.strip(), "params": checked}


def load_allowlist(path=None):
    """Parse + validate the YAML allowlist; cached by (path, mtime)."""
    path = Path(path or ALLOWLIST_PATH).resolve()
    try:
        stamp = path.stat().st_mtime_ns
    except OSError as error:
        raise AllowlistError(f"allowlist missing: {path}") from error
    key = (str(path), stamp)
    if key in _CACHE:
        return _CACHE[key]
    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise AllowlistError(f"allowlist is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        _spec_error(str(path), "document must be a mapping")
    unknown = set(document) - {"version", "defaults", "tools", "notes"}
    if unknown:
        _spec_error(str(path), "unknown top-level keys " + ", ".join(sorted(unknown)))
    version = document.get("version")
    if version != 1:
        _spec_error(str(path), "version must be 1")
    defaults = document.get("defaults") or {}
    unknown = set(defaults) - {"tool_tau", "param_tau", "require_top1"}
    if unknown:
        _spec_error(str(path), "unknown default keys " + ", ".join(sorted(unknown)))
    for key in ("tool_tau", "param_tau"):
        if key in defaults:
            defaults[key] = _check_probability(defaults[key], str(path) + "." + key)
    if "require_top1" in defaults and not isinstance(defaults["require_top1"], bool):
        _spec_error(str(path), "require_top1 must be boolean")
    tools = document.get("tools")
    if not isinstance(tools, list) or not tools:
        _spec_error(str(path), "tools must be a nonempty list")
    checked = [_validate_tool(str(path), tool) for tool in tools]
    if len({tool["name"] for tool in checked}) != len(checked):
        _spec_error(str(path), "duplicate tool name")
    allowlist = {"path": str(path), "version": version,
                 "defaults": {"tool_tau": defaults.get("tool_tau", DEFAULT_TOOL_TAU),
                              "param_tau": defaults.get("param_tau", DEFAULT_PARAM_TAU),
                              "require_top1": defaults.get("require_top1", DEFAULT_REQUIRE_TOP1)},
                 "tools": {tool["name"]: tool for tool in checked},
                 "order": [tool["name"] for tool in checked]}
    _CACHE.clear()
    _CACHE[key] = allowlist
    return allowlist


def is_finite_tool(name, allowlist=None):
    """True only for an allowlisted tool whose declared parameters are all enum/boolean."""
    try:
        allowlist = allowlist or load_allowlist()
    except (AllowlistError, ValueError):
        return False
    tool = allowlist["tools"].get(name)
    return tool is not None and bool(tool["params"])


# ---------- config ----------

def _check_probability(value, where):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{where}: must be a finite probability in [0,1]")
    return float(value)


def finite_fast_config(cfg, allowlist=None):
    """Read the four TurnConfig fields, falling back to the allowlist defaults."""
    defaults = (allowlist or {}).get("defaults") or {}

    def read(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    enabled = read("finite_fast_path", True)
    require_top1 = read("finite_fast_require_top1", defaults.get("require_top1", DEFAULT_REQUIRE_TOP1))
    return {"enabled": bool(enabled), "require_top1": bool(require_top1),
            "tool_tau": _check_probability(read("finite_fast_tool_tau", defaults.get("tool_tau", DEFAULT_TOOL_TAU)),
                                           "finite_fast_tool_tau"),
            "param_tau": _check_probability(read("finite_fast_param_tau", defaults.get("param_tau", DEFAULT_PARAM_TAU)),
                                            "finite_fast_param_tau")}


# ---------- schema agreement ----------

def _schema_parameters(schema):
    if not isinstance(schema, dict):
        return None
    parameters = schema.get("parameters") or schema.get("input_schema")
    return parameters if isinstance(parameters, dict) else None


def _check_finite_schema(tool, parameters):
    """Every property must be a string enum or a boolean; allowlist and schema must agree."""
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "schema_empty_parameters"
    for name, item in properties.items():
        if not isinstance(item, dict):
            return "schema_open_slot"
        if item.get("type") == "boolean":
            continue
        values = item.get("enum")
        if item.get("type") == "string" and isinstance(values, list) and values:
            continue
        return "schema_open_slot"
    unknown = set(properties) - {param["name"] for param in tool["params"]}
    if unknown:
        # An argument we cannot fill would make the call incomplete.
        return "schema_undeclared_parameter"
    required = parameters.get("required") or []
    if not isinstance(required, list) or not set(required) <= set(properties):
        return "schema_bad_required"
    missing = set(required) - {param["name"] for param in tool["params"]}
    if missing:
        return "schema_required_not_declared"
    for param in tool["params"]:
        item = properties.get(param["name"])
        if item is None:
            return "schema_parameter_missing"
        if param["kind"] == "boolean":
            if item.get("type") != "boolean":
                return "schema_type_mismatch"
            continue
        if item.get("type") != "string" or set(item.get("enum") or []) != set(param["values"]):
            return "schema_enum_mismatch"
    return None


# ---------- decisions ----------

def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid KEV probability")
    return float(value)


def default_kev_call(state, questions):
    """The project's KEV client (127.0.0.1:8009 tunnel); imported lazily so tests need no env."""
    from core_pipeline_run import KEV_BASE, KEV_CLIENT, kev_call
    return kev_call(KEV_CLIENT, state, questions, base=KEV_BASE)


def _miss(record, reason, detail, tool=None, score=None):
    if record is not None:
        record["finite_fast_reason"] = reason
        record["finite_fast_detail"] = detail
        if tool is not None:
            record["finite_fast_tool"] = tool
        if score is not None:
            record["finite_fast_tool_score"] = score
    return None


def _names(selected):
    names = []
    for item in selected or []:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str):
            names.append(name)
    return names


def _tool_score(scores, name):
    value = (scores or {}).get("tool:" + name)
    if value is None:
        return None
    try:
        return _probability(value)
    except ValueError:
        return None


def _ask_parameters(utterance, tool, questions, kev_call_fn):
    """One KEV request for every parameter of the tool; returns (answers, wall_ms)."""
    state = json.dumps({"utterance": utterance,
                        "tool": {"name": tool["name"], "description": tool["description"]}},
                       ensure_ascii=False, allow_nan=False)
    began = time.monotonic()
    raw = kev_call_fn(state, questions)
    wall_ms = (time.monotonic() - began) * 1000
    if not isinstance(raw, dict):
        raise ValueError("KEV response is not a mapping")
    if "model" in raw and raw["model"] != "kev-latest":
        raise ValueError("Wrong KEV response model")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Missing/extra KEV answers")
    return answers, wall_ms


def _read_choice(answer, param):
    if not isinstance(answer, dict):
        raise ValueError("KEV answer is not a mapping")
    unknown = set(answer) - {"type", "choice", "confidence", "probabilities"}
    if unknown:
        raise ValueError("Unexpected KEV answer keys")
    if answer.get("type", "choice") != "choice":
        raise ValueError("Expected a Choice answer")
    criteria = param["criteria"]
    choice = answer.get("choice")
    if choice not in criteria:
        raise ValueError("KEV choice is not a schema-legal value")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
        raise ValueError("Incomplete KEV Choice probabilities")
    confidence = _probability(answer.get("confidence"))
    for value in probabilities.values():
        _probability(value)
    if param["kind"] == "boolean":
        return choice == "true", confidence
    return choice, confidence


def try_finite_fast_decide(utterance, selected, scores, schemas, cfg, *, allowlist=None,
                           kev_call_fn=None, record=None):
    """Return None (fall back to decide()) or {decision, llm_ms, log_fields} for the fast path.

    `record` is an optional dict the function fills with MISS diagnostics
    (`finite_fast_reason`, `finite_fast_detail`, `finite_fast_error`, tool/score). It exists
    because the contract for every MISS is a bare None.
    """
    try:
        allowlist = allowlist or load_allowlist()
        settings = finite_fast_config(cfg, allowlist)
    except (AllowlistError, ValueError) as error:
        return _miss(record, "invalid_config", f"{type(error).__name__}: {error}")
    if not settings["enabled"]:
        return _miss(record, "disabled", "finite_fast_path is false")
    names = _names(selected)
    if not names:
        return _miss(record, "empty_shortlist", "KEV selected no candidate tool")
    tool_name = names[0]
    # Read the top-1 score before the guards so every MISS below can log tool + tool_score; a null
    # score is what made "not in the allowlist" MISSes undiagnosable in the timeline.
    score = _tool_score(scores, tool_name)
    tool = allowlist["tools"].get(tool_name)
    if tool is None:
        return _miss(record, "not_allowlisted", "shortlist top-1 is not in the allowlist",
                     tool_name, score)
    if score is None:
        return _miss(record, "tool_score_missing", "no KEV tool score for the shortlist top-1", tool_name)
    if score < settings["tool_tau"]:
        return _miss(record, "tool_score_low", f"{score:.4f} < {settings['tool_tau']}", tool_name, score)
    if not settings["require_top1"]:
        # Only when the hard rank-1 requirement is waived may a lower candidate stand in - and it
        # still has to be allowlisted and above tau.
        for candidate in names[1:]:
            alternative = allowlist["tools"].get(candidate)
            alternative_score = _tool_score(scores, candidate) if alternative else None
            if alternative and alternative_score is not None and alternative_score >= settings["tool_tau"]:
                tool, tool_name, score = alternative, candidate, alternative_score
                break
    parameters = None
    for schema in schemas or []:
        if isinstance(schema, dict) and schema.get("name") == tool_name:
            parameters = _schema_parameters(schema)
            break
    if parameters is None:
        return _miss(record, "schema_missing", "no disclosed schema for the shortlisted tool", tool_name, score)
    problem = _check_finite_schema(tool, parameters)
    if problem:
        return _miss(record, problem, "schema and allowlist do not describe a finite tool", tool_name, score)
    questions = {}
    for param in tool["params"]:
        questions[param["name"]] = {
            "type": "choice",
            "instructions": (f"业务工具 {tool_name}（{tool['description']}）。判断参数 {param['name']}："
                             f"{param['question']}。只依据用户原话判断，不要推测用户未说出的偏好；"
                             "必须从给定的枚举值里选一个。"),
            "criteria": dict(param["criteria"])}
    try:
        answers, wall_ms = _ask_parameters(utterance, tool, questions,
                                           kev_call_fn or default_kev_call)
        arguments, confidences = {}, {}
        for param in tool["params"]:
            value, confidence = _read_choice(answers[param["name"]], param)
            if confidence < settings["param_tau"]:
                return _miss(record, "param_low_confidence",
                             f"{param['name']}={value} confidence {confidence:.4f} < {settings['param_tau']}",
                             tool_name, score)
            arguments[param["name"]] = value
            confidences[param["name"]] = round(confidence, 4)
    except Exception as error:  # KEV unavailable/timeout/malformed -> degrade, never kill the turn
        if record is not None:
            record["finite_fast_error"] = f"{type(error).__name__}: {error}"
        return _miss(record, "kev_error", "KEV parameter filling failed", tool_name, score)
    if list(arguments) != [param["name"] for param in tool["params"]]:
        return _miss(record, "arguments_mismatch", "assembled arguments are not the declared parameters",
                     tool_name, score)
    try:
        Draft202012Validator(parameters).validate(arguments)
    except Exception as error:
        if record is not None:
            record["finite_fast_error"] = f"{type(error).__name__}: {error}"
        return _miss(record, "arguments_invalid", "assembled arguments failed schema validation",
                     tool_name, score)
    if record is not None:
        record.update({"finite_fast_reason": "hit", "finite_fast_tool": tool_name,
                       "finite_fast_tool_score": score, "finite_fast_kev_ms": wall_ms,
                       "finite_fast_param_confidences": dict(confidences)})
    decision = {"kind": "tool_call", "tool": tool_name, "arguments": arguments}
    return {"decision": decision, "llm_ms": wall_ms,
            "log_fields": {"decision_head": HEAD, "routed_by": _ROUTE, "finite_fast": True,
                           "tool_score": score, "param_confidences": confidences,
                           "finite_fast_kev_ms": wall_ms, "allowlist_version": allowlist["version"],
                           "decision": decision, "decision_latency_ms": wall_ms}}


if __name__ == "__main__":
    allowlist = load_allowlist()
    print(json.dumps({"path": allowlist["path"], "defaults": allowlist["defaults"],
                      "tools": allowlist["order"]}, ensure_ascii=False, indent=2))