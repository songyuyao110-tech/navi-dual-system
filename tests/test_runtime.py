from pathlib import Path
import tempfile
import unittest

import yaml

from execution import ExecutionError, Runtime, ToolCall
from registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.snapshot = load_registry(ROOT / "tools.yaml", frozenset(self.runtime.handlers))

    def test_fourteen_navigation_tools_and_one_domain(self):
        self.assertEqual(len(self.snapshot.tools), 14)
        self.assertEqual(set(self.snapshot.domains()), {"navigation"})
        self.assertEqual(len(self.snapshot.domains()["navigation"]), 14)

    def test_fast_and_slow_test_drivers_share_runtime(self):
        disclosed = frozenset({"navi_to_special_poi", "navi_get_status"})
        first = self.runtime.execute(
            self.snapshot, disclosed, ToolCall("navi_to_special_poi", {"poi_type": "home"})
        )
        second = self.runtime.execute(self.snapshot, disclosed, ToolCall("navi_get_status", {}))
        self.assertTrue(first["mock"])
        self.assertTrue(second["facts"]["navigation"]["route_active"])
        self.assertEqual(len(self.runtime.calls), 2)

    def test_undisclosed_unknown_and_invalid_parameters(self):
        for disclosed, call in [
            (frozenset(), ToolCall("navi_get_status", {})),
            (frozenset({"unknown"}), ToolCall("unknown", {})),
            (frozenset({"navi_to_special_poi"}), ToolCall("navi_to_special_poi", {})),
            (frozenset({"navi_to_special_poi"}), ToolCall("navi_to_special_poi", {"poi_type": "wrong"})),
        ]:
            with self.subTest(call=call), self.assertRaises(ExecutionError):
                self.runtime.execute(self.snapshot, disclosed, call)
        self.assertEqual(self.runtime.calls, [])

    def test_schema_copy_cannot_change_snapshot(self):
        schema = self.snapshot.tools["navi_to_special_poi"].parameters
        schema["required"] = []
        self.assertEqual(
            self.snapshot.tools["navi_to_special_poi"].parameters["required"], ["poi_type"]
        )

    def test_bad_registry_rejected(self):
        original = yaml.safe_load((ROOT / "tools.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.yaml"
            original["tools"].append(original["tools"][0])
            path.write_text(yaml.safe_dump(original))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_registry(path, frozenset(self.runtime.handlers))
            original["tools"].pop()
            original["tools"][0]["binding"] = "missing"
            path.write_text(yaml.safe_dump(original))
            with self.assertRaisesRegex(ValueError, "Unknown execution binding"):
                load_registry(path, frozenset(self.runtime.handlers))


if __name__ == "__main__":
    unittest.main()
