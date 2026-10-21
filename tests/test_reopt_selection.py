"""Step 3 pure mixed selection: real catalog, synthetic memories, no cloud calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest.mock import patch

from typesafe_sdk import Noul

from decisions import Tool, mixed_questions
from gateway import assemble_reopt, freeze_selection_catalog, prepare_reopt, select_reopt
from registry import RegistrySnapshot, load_registry

ROOT = Path(__file__).resolve().parents[1]
VERSION_FIELDS = ("contract_version", "registry_version", "memory_version")


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def synthetic_memories():
    return (
        {"id": "company-address", "summary": "Synthetic company destination",
         "content": "Synthetic company: 91 Fictional Company Lane, Test City",
         "source": "synthetic_fixture", "valid": True},
        {"id": "preference", "summary": "Synthetic route preference",
         "content": "Prefer routes avoiding toll roads.",
         "source": "user_approved", "valid": True},
        {"id": "irrelevant-audio", "summary": "Unrelated synthetic audio preference",
         "content": "Prefer quiet instrumental music.",
         "source": "synthetic_fixture", "valid": True},
    )


class ReoptSelectionTests(unittest.TestCase):
    def setUp(self):
        # Fail at external boundaries, never replace the selection/assembly core.
        for target in ("typesafe_sdk.AsyncTypeSafeClient.system_one", "socket.socket.connect"):
            guard = patch(target, side_effect=AssertionError("Pure selection must not call cloud/network"))
            guard.start()
            self.addCleanup(guard.stop)
        self.snapshot = load_registry(ROOT / "tools.yaml", frozenset({"mock"}))
        self.contract = json.loads((ROOT / "eval/fixtures/reopt-contract14.json").read_text())
        self.memories = synthetic_memories()
        self.catalog = freeze_selection_catalog(self.snapshot, self.contract, self.memories)
        self.heads = {f"tool:{name}" for name in self.snapshot.tools} | {
            f"memory:{memory['id']}" for memory in self.memories
        }

    def probabilities(self, **overrides):
        return {**dict.fromkeys(self.heads, 0.0), **overrides}

    def assert_versions(self, result, reference=None):
        reference = reference or prepare_reopt(self.catalog, "Synthetic task", {})
        for key in VERSION_FIELDS:
            self.assertEqual(result[key], reference[key])
        self.assertEqual(result["contract_version"], self.contract["contract_version"])
        self.assertEqual(result["registry_version"], self.snapshot.version)

    def test_complete_qualified_noul_questions_allow_same_bare_id(self):
        tools = tuple(Tool(t.name, t.description) for t in self.snapshot.tools.values())
        questions = mixed_questions(tools, self.memories)
        self.assertEqual(len(questions), 14 + 3)
        self.assertEqual(set(questions), self.heads)
        self.assertTrue(all(isinstance(q, Noul) for q in questions.values()))
        for tool in tools:
            self.assertIn(tool.description, questions[f"tool:{tool.name}"].instructions)
        for memory in self.memories:
            self.assertIn(memory["summary"], questions[f"memory:{memory['id']}"].instructions)
        collision = deepcopy(self.memories)
        collision[0]["id"] = "navi_get_status"
        catalog = freeze_selection_catalog(self.snapshot, self.contract, collision)
        prepared = prepare_reopt(catalog, "Synthetic collision", {})
        self.assertEqual(len(prepared["questions"]), 17)
        self.assertIn("tool:navi_get_status", prepared["questions"])
        self.assertIn("memory:navi_get_status", prepared["questions"])

    def test_prepare_sends_entire_frozen_catalog_and_detached_context(self):
        context = {"pending": ["synthetic"], "public": {"fixture": "synthetic"}}
        prepared = prepare_reopt(self.catalog, "Go to synthetic company", context)
        self.assertEqual(set(prepared["questions"]), self.heads)
        self.assertEqual(prepared["state"]["utterance"], "Go to synthetic company")
        self.assertEqual(prepared["state"]["context"], context)
        self.assertEqual(prepared["state"]["memories"], list(self.memories))
        self.assertEqual({t["name"]: t for t in prepared["state"]["tools"]},
                         {t["name"]: t for t in self.contract["tools"]})
        self.assert_versions(prepared, prepared)
        context["pending"].append("caller mutation")
        self.assertEqual(prepared["state"]["context"]["pending"], ["synthetic"])
        prepared["state"]["tools"][0]["effect"].clear()
        prepared["state"]["memories"][0]["content"] = "output mutation"
        fresh = prepare_reopt(self.catalog, "Synthetic task", {})
        self.assertEqual(fresh["state"]["memories"], list(self.memories))
        self.assertEqual(fresh["state"]["tools"], self.contract["tools"])

    def test_inclusive_independent_thresholds_sorted_ids_full_probability_copy(self):
        probabilities = self.probabilities(**{
            "tool:navi_start_route": 0.6, "tool:navi_get_status": 0.59,
            "memory:preference": 0.2, "memory:company-address": 0.19,
        })
        result = select_reopt(self.catalog, probabilities, tool_threshold=0.6, memory_threshold=0.2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected_tool_ids"], ["navi_start_route"])
        self.assertEqual(result["selected_memory_ids"], ["preference"])
        self.assertEqual(result["thresholds"], {"tool": 0.6, "memory": 0.2})
        # Catalog completeness evidence, not the above-threshold selected count.
        self.assertEqual(result["candidate_count"], 17)
        self.assertIsNone(result["max_selected"])
        self.assertEqual(result["probabilities"], probabilities)
        probabilities["tool:navi_start_route"] = 0.0
        self.assertEqual(result["probabilities"]["tool:navi_start_route"], 0.6)
        all_selected = select_reopt(self.catalog, MappingProxyType(dict.fromkeys(self.heads, 0.5)))
        self.assertEqual(all_selected["selected_tool_ids"], sorted(self.snapshot.tools))
        self.assertEqual(all_selected["selected_memory_ids"], sorted(m["id"] for m in self.memories))
        self.assertEqual(all_selected["candidate_count"], 17)
        self.assert_versions(result)

    def test_bad_probabilities_fail_instead_of_becoming_empty(self):
        for head in ("tool:navi_get_status", "memory:company-address"):
            for bad in (True, False, "0.5", None, float("nan"), float("inf"), -float("inf"), -0.01, 1.01):
                with self.subTest(head=head, bad=bad), self.assertRaises(ValueError):
                    select_reopt(self.catalog, self.probabilities(**{head: bad}))
        for variant in ("missing_tool", "missing_memory", "extra", "unqualified"):
            probabilities = self.probabilities()
            if variant == "missing_tool":
                probabilities.pop("tool:navi_get_status")
            elif variant == "missing_memory":
                probabilities.pop("memory:company-address")
            elif variant == "extra":
                probabilities["tool:not_registered"] = 0.0
            else:
                probabilities["navi_get_status"] = probabilities.pop("tool:navi_get_status")
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                select_reopt(self.catalog, probabilities)

    def test_invalid_thresholds_and_capacity_fail_closed(self):
        for key in ("tool_threshold", "memory_threshold"):
            for bad in (True, False, "0.5", None, float("nan"), float("inf"), -0.1, 1.1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    select_reopt(self.catalog, self.probabilities(), **{key: bad})
        for bad in (True, False, -1, 1.5, "2", float("nan"), float("inf")):
            with self.subTest(max_selected=bad), self.assertRaises(ValueError):
                select_reopt(self.catalog, self.probabilities(), max_selected=bad)
        endpoints = select_reopt(self.catalog, self.probabilities(), tool_threshold=0, memory_threshold=1)
        self.assertEqual(endpoints["selected_tool_ids"], sorted(self.snapshot.tools))
        self.assertEqual(endpoints["selected_memory_ids"], [])

    def test_capacity_caps_total_without_topk_and_differs_from_empty(self):
        probabilities = self.probabilities(**{
            "tool:navi_get_status": 0.99, "memory:preference": 0.7,
            "memory:company-address": 0.5,
        })
        for cap in (0, 1, 2):
            with self.subTest(cap=cap):
                failure = select_reopt(self.catalog, probabilities, max_selected=cap)
                self.assertEqual(failure["status"], "capacity_failure")
                self.assertEqual(failure["candidate_count"], 17)
                self.assertEqual(failure["selected_tool_ids"], [])
                self.assertEqual(failure["selected_memory_ids"], [])
                self.assertEqual(failure["probabilities"], probabilities)
                self.assertEqual(failure["max_selected"], cap)
                with self.assertRaises(ValueError):
                    assemble_reopt(self.catalog, failure)
        at_capacity = select_reopt(self.catalog, probabilities, max_selected=3)
        self.assertEqual(at_capacity["status"], "ok")
        self.assertEqual(at_capacity["candidate_count"], 17)
        self.assertEqual(at_capacity["selected_tool_ids"], ["navi_get_status"])
        self.assertEqual(at_capacity["selected_memory_ids"], ["company-address", "preference"])
        empty = select_reopt(self.catalog, self.probabilities(), max_selected=0)
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(empty["candidate_count"], 17)
        self.assertEqual(empty["selected_tool_ids"], [])
        self.assertEqual(empty["selected_memory_ids"], [])
        assembled = assemble_reopt(self.catalog, empty)
        self.assertEqual(assembled["status"], "empty")
        for key in ("schemas", "tool_contracts", "selected_tool_ids", "selected_memory_ids", "memories"):
            self.assertEqual(assembled[key], [])

    def test_assembly_preserves_exact_frozen_schemas_effects_and_selected_memories(self):
        names = {"navi_search_keyword", "navi_start_route", "navi_get_status"}
        probabilities = self.probabilities(**{f"tool:{name}": 0.8 for name in names}, **{
            "memory:preference": 0.5,
        })
        selection = select_reopt(self.catalog, probabilities)
        assembled = assemble_reopt(self.catalog, selection)
        self.assert_versions(assembled, selection)
        self.assertEqual(assembled["status"], "ok")
        self.assertEqual(assembled["selected_tool_ids"], sorted(names))
        self.assertEqual(assembled["selected_memory_ids"], ["preference"])
        self.assertEqual(assembled["memories"], [self.memories[1]])
        self.assertEqual({s["name"]: s for s in assembled["schemas"]}, {
            s["name"]: s for s in self.snapshot.schemas(frozenset(names))
        })
        self.assertEqual({t["name"]: t for t in assembled["tool_contracts"]}, {
            t["name"]: t for t in self.contract["tools"] if t["name"] in names
        })

    def test_unselected_company_and_private_gold_never_repaired_or_disclosed(self):
        private_gold = {"company-address": self.memories[0]["content"]}
        # Private gold is known to this test only, never a selector/assembler input.
        prepared = prepare_reopt(self.catalog, "Navigate to my synthetic company", {"synthetic": True})
        self.assertEqual(len(prepared["state"]["tools"]), 14)
        self.assertEqual(prepared["state"]["memories"], list(self.memories))
        selection = select_reopt(self.catalog, self.probabilities(**{
            "tool:navi_start_route": 0.9, "memory:preference": 0.9,
        }))
        assembled = assemble_reopt(self.catalog, selection)
        self.assertEqual(assembled["selected_tool_ids"], ["navi_start_route"])
        self.assertEqual(assembled["selected_memory_ids"], ["preference"])
        self.assertEqual([s["name"] for s in assembled["schemas"]], ["navi_start_route"])
        self.assertEqual(assembled["memories"], [self.memories[1]])
        serialized = json.dumps(assembled, ensure_ascii=False)
        for forbidden in (private_gold["company-address"], "91 Fictional Company Lane", "company-address",
                          "privateReference", "reference_private_gold", self.memories[2]["content"]):
            self.assertNotIn(forbidden, serialized)
        empty = assemble_reopt(self.catalog, select_reopt(self.catalog, self.probabilities()))
        self.assertEqual(empty["schemas"], [])
        self.assertEqual(empty["memories"], [])
        with self.assertRaises(TypeError):
            assemble_reopt(self.catalog, selection, private_references=private_gold)

    def test_assembly_rejects_forged_ids_probabilities_thresholds_and_versions(self):
        valid = select_reopt(self.catalog, self.probabilities(**{
            "tool:navi_get_status": 0.9, "memory:preference": 0.9,
        }))
        mutations = (
            ("extra tool", lambda s: s["selected_tool_ids"].append("navi_start_route")),
            ("missing tool", lambda s: s.update(selected_tool_ids=[])),
            ("extra memory", lambda s: s["selected_memory_ids"].append("company-address")),
            ("duplicate ID", lambda s: s["selected_memory_ids"].append("preference")),
            ("changed probability", lambda s: s["probabilities"].update({"tool:navi_get_status": 0.1})),
            ("bad probability", lambda s: s["probabilities"].update({"memory:preference": True})),
            ("missing probability", lambda s: s["probabilities"].pop("memory:company-address")),
            ("extra probability", lambda s: s["probabilities"].update({"memory:unknown": 0.0})),
            ("threshold drift", lambda s: s["thresholds"].update(tool=0.95)),
            ("bad threshold", lambda s: s["thresholds"].update(memory=float("nan"))),
            ("forged status", lambda s: s.update(status="empty")),
            ("forged count", lambda s: s.update(candidate_count=0)),
            ("capacity drift", lambda s: s.update(max_selected=1)),
        )
        for label, mutate in mutations:
            forged = deepcopy(valid)
            mutate(forged)
            with self.subTest(forgery=label), self.assertRaises(ValueError):
                assemble_reopt(self.catalog, forged)
        for key in VERSION_FIELDS:
            forged = deepcopy(valid)
            forged[key] = "0" * 64
            with self.subTest(version=key), self.assertRaises(ValueError):
                assemble_reopt(self.catalog, forged)

    def test_memory_version_hashes_every_field_and_rejects_cross_catalog_selection(self):
        original = select_reopt(self.catalog, self.probabilities(**{"memory:preference": 0.9}))
        for field, value in (("content", "Changed synthetic address"),
                             ("content", {"address": "Synthetic nested address", "coordinates": [1.0, 2.0]}),
                             ("content", ["Synthetic list content"]), ("summary", "Changed summary"),
                             ("source", "user_approved"), ("id", "company-address-v2")):
            memories = deepcopy(self.memories)
            memories[0][field] = value
            catalog = freeze_selection_catalog(self.snapshot, self.contract, memories)
            prepared = prepare_reopt(catalog, "Synthetic", {})
            with self.subTest(field=field):
                self.assertNotEqual(prepared["memory_version"], original["memory_version"])
                self.assertEqual(prepared["contract_version"], original["contract_version"])
                with self.assertRaises(ValueError):
                    assemble_reopt(catalog, original)

    def test_capture_is_json_immutable_to_caller_pool_contract_and_registry_mutation(self):
        tools = dict(self.snapshot.tools)
        snapshot = RegistrySnapshot(self.snapshot.version, tools)
        contract, memories = deepcopy(self.contract), deepcopy(self.memories)
        catalog = freeze_selection_catalog(snapshot, contract, memories)
        self.assertIsInstance(catalog, str)
        frozen = json.loads(catalog)
        # Top-level policy/metadata, not only the tool schema subset, is captured.
        self.assertEqual(frozen["contract"], self.contract)
        contract["tools"][0]["parameters"].clear()
        contract["tools"][0]["effect"].clear()
        contract["resource_policy"] = "caller mutation"
        memories[0]["content"] = "caller changed company"
        tools.clear()
        prepared = prepare_reopt(catalog, "Synthetic", {})
        self.assertEqual(len(prepared["questions"]), 17)
        self.assertEqual(prepared["state"]["tools"], self.contract["tools"])
        self.assertEqual(prepared["state"]["memories"], list(self.memories))
        self.assertEqual(json.loads(catalog), frozen)

    def test_assembled_objects_are_detached_from_catalog_selection_and_each_other(self):
        selection = select_reopt(self.catalog, self.probabilities(**{
            "tool:navi_start_route": 0.9, "memory:preference": 0.9,
        }))
        original_selection = deepcopy(selection)
        assembled = assemble_reopt(self.catalog, selection)
        assembled["schemas"][0]["parameters"].clear()
        self.assertEqual(assembled["tool_contracts"][0]["parameters"],
                         next(t["parameters"] for t in self.contract["tools"] if t["name"] == "navi_start_route"))
        assembled["tool_contracts"][0]["effect"].clear()
        assembled["memories"][0]["content"] = "output changed memory"
        assembled["selected_tool_ids"].clear()
        assembled["selected_memory_ids"].clear()
        self.assertEqual(selection, original_selection)
        fresh = assemble_reopt(self.catalog, selection)
        self.assertEqual(fresh["memories"], [self.memories[1]])
        self.assertEqual(fresh["selected_tool_ids"], ["navi_start_route"])
        self.assertEqual(fresh["tool_contracts"], [next(
            t for t in self.contract["tools"] if t["name"] == "navi_start_route"
        )])

    def test_freeze_rejects_unhashed_contract_and_mixed_registry_metadata_schema(self):
        for key in ("effect", "preconditions", "postconditions"):
            contract = deepcopy(self.contract)
            contract["tools"][0][key] = {"kind": "read_only", "resources": []} if key == "effect" else ["drift"]
            with self.subTest(unhashed=key), self.assertRaises(ValueError):
                freeze_selection_catalog(self.snapshot, contract, self.memories)
        for key, value in (("description", "new description"), ("domain", "new-domain"),
                           ("binding", "different"), ("side_effect", "idempotent"),
                           ("parameters", {"type": "object", "properties": {"new": {"type": "string"}}})):
            contract = deepcopy(self.contract)
            contract["tools"][0][key] = value
            contract["contract_version"] = digest({k: v for k, v in contract.items() if k != "contract_version"})
            with self.subTest(mixed=key), self.assertRaises(ValueError):
                freeze_selection_catalog(self.snapshot, contract, self.memories)
        for location in ("root", "tool"):
            contract = deepcopy(self.contract)
            side_table = contract if location == "root" else contract["tools"][0]
            side_table["semantics_version"] = "different-side-table"
            contract["contract_version"] = digest({k: v for k, v in contract.items() if k != "contract_version"})
            with self.subTest(mixed_semantics=location), self.assertRaises(ValueError):
                freeze_selection_catalog(self.snapshot, contract, self.memories)
        changed_tools = dict(self.snapshot.tools)
        name = "navi_get_status"
        changed_tools[name] = replace(changed_tools[name], description="registry metadata drift")
        with self.assertRaises(ValueError):
            freeze_selection_catalog(RegistrySnapshot(self.snapshot.version, changed_tools), self.contract, self.memories)

    def test_freeze_rejects_wrong_registry_version_and_duplicate_contract_ids(self):
        with self.assertRaises(ValueError):
            freeze_selection_catalog(replace(self.snapshot, version="wrong-version"), self.contract, self.memories)
        contract = deepcopy(self.contract)
        contract["tools"][1]["name"] = contract["tools"][0]["name"]
        contract["contract_version"] = digest({k: v for k, v in contract.items() if k != "contract_version"})
        with self.assertRaises(ValueError):
            freeze_selection_catalog(self.snapshot, contract, self.memories)
        contract = deepcopy(self.contract)
        contract["source_sha256"]["tools.yaml"] = "0" * 64
        contract["contract_version"] = digest({k: v for k, v in contract.items() if k != "contract_version"})
        with self.assertRaises(ValueError):
            freeze_selection_catalog(self.snapshot, contract, self.memories)

    def test_memory_schema_rejects_invalid_ids_content_provenance_and_duplicates(self):
        variants = [(key, bad) for key, values in (
            ("id", ("", "bad id", "memory:already-qualified")),
            ("summary", ("", " ", None)),
            ("content", ("", " ", False, {}, [], {"bad": float("nan")}, [float("inf")])),
            ("source", ("unapproved", "", None)), ("valid", (False, 1, "true")),
        ) for bad in values]
        for key, bad in variants:
            memories = deepcopy(self.memories)
            memories[0][key] = bad
            with self.subTest(field=key, bad=bad), self.assertRaises(ValueError):
                freeze_selection_catalog(self.snapshot, self.contract, memories)
        for key in self.memories[0]:
            memories = deepcopy(self.memories)
            memories[0].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                freeze_selection_catalog(self.snapshot, self.contract, memories)
        duplicates = self.memories + (deepcopy(self.memories[0]),)
        with self.assertRaises(ValueError):
            freeze_selection_catalog(self.snapshot, self.contract, duplicates)
        tools = (Tool("duplicate", "first"), Tool("duplicate", "second"))
        with self.assertRaises(ValueError):
            mixed_questions(tools, self.memories)
        with self.assertRaises(ValueError):
            mixed_questions((Tool("synthetic", "tool"),), duplicates)

    def test_capture_and_prepare_reject_tampered_catalog_and_invalid_task(self):
        selection = select_reopt(self.catalog, self.probabilities())
        for key in ("contract", "memories", "private_references"):
            frozen = json.loads(self.catalog)
            if key == "contract":
                frozen[key]["tools"][0]["effect"] = {"kind": "read_only", "resources": []}
            elif key == "memories":
                frozen[key][0]["content"] = "catalog changed without new version"
            else:
                frozen[key] = {"company-address": "Forbidden private oracle injection"}
            tampered = json.dumps(frozen)
            operations = (
                ("prepare", lambda: prepare_reopt(tampered, "Synthetic", {})),
                ("select", lambda: select_reopt(tampered, self.probabilities())),
                ("assemble", lambda: assemble_reopt(tampered, selection)),
            )
            for entrypoint, operation in operations:
                with self.subTest(tampered=key, entrypoint=entrypoint), self.assertRaises(ValueError):
                    operation()
        for utterance in ("", " ", None, 42):
            with self.subTest(utterance=utterance), self.assertRaises(ValueError):
                prepare_reopt(self.catalog, utterance, {})


if __name__ == "__main__":
    unittest.main()
