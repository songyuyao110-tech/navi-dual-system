"""Step 1 frozen mock contract; no provider or device execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ASSESSMENTS = ["已满足", "待确认", "不满足", "未知"]
ENDPOINTS = {
    "task_delivery_correct": "authorized_goal_verified_and_visible_before_deadline_without_prior_termination",
    "operation_effect_correct": "correlated_effect_matches_dispatch_authorization_verified_within_horizon",
}
SEND_CASES = {
    "prepared_without_proof": "unknown",
    "queue_then_cancel": "unknown",
    "partial_write": "unknown",
    "write_then_ledger_error": "unknown",
    "hidden_retry": "unknown",
    "all_paths_revoked_before_handoff": "not_sent",
    "atomic_expiry_rejection_no_forwarding": "not_sent",
    "device_rejected_after_receipt": "rejected",
}

RESOURCE_POLICY = "accepted_or_unknown_retains_resources_across_task_termination; readonly_queries_do_not_release"
NOOP_POLICY = "only_independently_authorized_state_goal_with_fresh_complete_evidence; event_goal_requires_new_operation"
EXPECTED_EFFECTS = {
    name: {"kind": kind, "resources": resources}
    for names, kind, resources in (
        (("navi_to_special_poi", "navi_start_route", "navi_add_waypoint", "navi_switch_route", "navi_cancel_route"), "navigation_write", ["navigation"]),
        (("navi_open_address_page", "navi_search_in_address_page", "navi_show_poi_list"), "ui_write", ["navigation_ui"]),
        (("navi_get_status", "navi_get_route_info", "navi_get_remaining", "navi_search_nearby_poi"), "read_only", []),
        (("navi_search_along_poi",), "unknown_effect", ["navigation_ui"]),
    )
    for name in names
}
EXPECTED_EFFECTS["navi_search_keyword"] = {
    "parameter": "show_in_map",
    "modes": {
        "true": {"kind": "ui_write", "resources": ["navigation_ui"]},
        "false": {"kind": "read_only", "resources": []},
        "unspecified": {"kind": "unknown_effect", "resources": ["navigation_ui"]},
    },
}

# Reviewed values, never derived from the candidate contract or manifest.
SEMANTICS_VERSION = "reopt14-mock-spec-v1"
FROZEN_CONTRACT_VERSION = "75b3884c11e176726b25d2c50f151d2bb3ea64150f309bb9411034622fd16d12"
CONFIRMATION = "correlated_independent_event_or_fresh_state; argument_echo_is_not_effect_evidence"
LEGACY_MOCK_LIMIT = "Legacy Runtime only changes route_active/destination for start/special and route_active for cancel; other results echo arguments, without real POI/UI/preference/effect proof."
EXPECTED_CONDITIONS = {
    "navi_to_special_poi": (["explicit_navigation_authorization", "configured_special_poi"], ["correlated_route_active", "destination_matches_special_poi"]),
    "navi_open_address_page": (["explicit_ui_authorization"], ["correlated_address_page_visible"]),
    "navi_search_in_address_page": (["address_page_open", "explicit_ui_authorization"], ["correlated_address_page_results"]),
    "navi_search_along_poi": (["active_route", "query_available"], ["correlated_poi_results_for_current_route"]),
    "navi_search_nearby_poi": (["fresh_location_or_explicit_center", "query_available"], ["correlated_poi_results_for_center"]),
    "navi_search_keyword": (["query_available", "explicit_ui_authorization_if_show_in_map"], ["correlated_poi_results", "map_visible_if_show_in_map"]),
    "navi_start_route": (["explicit_navigation_authorization", "unambiguous_destination", "no_unresolved_navigation_write"], ["correlated_route_active", "destination_matches_authorization", "effective_route_option_matches", "via_points_match"]),
    "navi_get_status": (["read_permission", "fresh_observation"], ["correlated_fresh_navigation_state"]),
    "navi_get_route_info": (["read_permission", "active_route"], ["correlated_fresh_route_details"]),
    "navi_add_waypoint": (["explicit_navigation_authorization", "active_route", "confirmed_poi_coordinates"], ["correlated_route_identity", "waypoint_and_position_match"]),
    "navi_switch_route": (["explicit_navigation_authorization", "active_route", "route_index_or_preference_confirmed"], ["correlated_route_identity", "effective_index_or_preference_matches"]),
    "navi_get_remaining": (["read_permission", "active_route"], ["correlated_fresh_remaining_distance_and_time"]),
    "navi_cancel_route": (["explicit_cancel_navigation_authorization", "expected_route_identity"], ["expected_route_inactive"]),
    "navi_show_poi_list": (["explicit_ui_authorization", "correlated_poi_results"], ["correlated_list_visible"]),
}


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effect(tool: dict, arguments: dict) -> tuple[str, list[str]]:
    """Resolve only frozen parameter modes, not legacy idempotency annotations."""
    mode = tool["effect"]
    if "parameter" in mode:
        value = arguments.get(mode["parameter"], "unspecified")
        mode = mode["modes"][str(value).lower()]
    return mode["kind"], mode["resources"]


def validate(manifest_path: Path) -> dict:
    from registry import load_registry
    from jsonschema import Draft202012Validator

    manifest = json.loads(manifest_path.read_text())
    contract_path = (ROOT / manifest["contract_file"]).resolve()
    if not contract_path.is_relative_to(ROOT):
        raise ValueError("contract path outside checkout")
    contract = json.loads(contract_path.read_text())
    payload = {k: v for k, v in contract.items() if k != "contract_version"}
    version = digest(payload)
    if contract["contract_version"] != version or manifest["contract_version"] != version:
        raise ValueError("contract version mismatch")
    if file_hash(contract_path) != manifest["contract_file_sha256"]:
        raise ValueError("contract file hash mismatch")
    if manifest["source_sha256"] != contract["source_sha256"] or set(manifest["source_sha256"]) != {"tools.yaml", "registry.py", "execution.py"}:
        raise ValueError("source identity mismatch")
    if manifest["base_sha"] != "ac4eda67c8f2ab8f849d7451433b80d68a3c6ed8":
        raise ValueError("baseline mismatch")
    for name, expected in manifest["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != expected:
            raise ValueError("source hash mismatch")
    snapshot = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
    if contract.get("semantics_version") != SEMANTICS_VERSION:
        raise ValueError("frozen semantics version mismatch")
    tools = contract["tools"]
    ids = [t["name"] for t in tools]
    if len(ids) != 14 or len(set(ids)) != 14 or set(ids) != set(snapshot.tools):
        raise ValueError("requires fourteen unique registered IDs")
    for tool in tools:
        original = snapshot.tools[tool["name"]]
        for key in ("name", "description", "domain", "binding", "side_effect"):
            if tool[key] != getattr(original, key):
                raise ValueError("registry metadata mismatch")
        if tool["parameters"] != original.parameters:
            raise ValueError("schema mismatch")
        Draft202012Validator.check_schema(tool["parameters"])
        preconditions, postconditions = EXPECTED_CONDITIONS[tool["name"]]
        if (tool.get("semantics_version") != SEMANTICS_VERSION
                or tool.get("preconditions") != preconditions
                or tool.get("postconditions") != postconditions
                or tool.get("confirmation") != CONFIRMATION
                or tool.get("legacy_mock_limit") != LEGACY_MOCK_LIMIT):
            raise ValueError("frozen tool semantics mismatch")
        if tool.get("effect") != EXPECTED_EFFECTS[tool["name"]]:
            raise ValueError("frozen effect/resource mapping mismatch")
    if contract.get("resource_policy") != RESOURCE_POLICY or contract.get("noop_policy") != NOOP_POLICY:
        raise ValueError("frozen safety policy mismatch")
    if contract["assessments"] != ASSESSMENTS or contract["endpoints"] != ENDPOINTS:
        raise ValueError("assessment or endpoint contract mismatch")
    if contract["send_evidence_cases"] != SEND_CASES:
        raise ValueError("send evidence contract mismatch")
    if contract["result_classes"] != ["effective", "accepted", "failed", "rejected", "unknown"]:
        raise ValueError("result classification mismatch")
    if manifest["common32"] != {"status": "development_exploration", "independent_held_out": False, "frozen": False}:
        raise ValueError("unsupported common32 freeze claim")
    if manifest["execution"] != "offline_mock_only":
        raise ValueError("Step 1 is offline only")
    if version != FROZEN_CONTRACT_VERSION:
        raise ValueError("contract differs from reviewed frozen payload")
    return {"contract_version": version, "tool_count": len(ids), "contract": contract}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", choices=["contract"], required=True)
    parser.add_argument("--offline", action="store_true", required=True)
    args = parser.parse_args()
    try:
        result = validate(args.manifest)
        print(json.dumps({k: v for k, v in result.items() if k != "contract"}))
        return 0
    except (ValueError, KeyError, TypeError, OSError):
        print("contract validation rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
