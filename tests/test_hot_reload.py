from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from app import create_app
from execution import ExecutionError, Runtime, ToolCall
from registry import Registry
from task_context import TaskLedger

ROOT = Path(__file__).resolve().parents[1]


class HotReloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tools.yaml"
        self.data = yaml.safe_load((ROOT / "tools.yaml").read_text())
        self.write()
        self.runtime = Runtime()
        self.registry = Registry(self.path, frozenset(self.runtime.handlers))

    def write(self):
        self.path.write_text(yaml.safe_dump(self.data, allow_unicode=True))

    def test_old_task_pins_schema_description_and_binding(self):
        old = self.registry.current
        ledger = TaskLedger()
        task = ledger.create("synthetic", {}, old, {
            "route": "fast", "disclosed_tools": ["navi_get_status"],
        })
        self.data["tools"][7]["binding"] = "mock_error"
        self.data["tools"][7]["description"] = "new synthetic description"
        self.write()
        new = self.registry.reload()
        self.assertNotEqual(old.version, new.version)
        ledger.revise(task.id, "synthetic refill", {})
        self.assertIs(task.snapshot, old)
        self.assertEqual(task.snapshot.domains()["navigation"][7].description,
                         "获取当前是否导航中及当前导航状态")
        result = self.runtime.execute(task.snapshot, task.disclosed, ToolCall("navi_get_status", {}))
        self.assertTrue(result["mock"])
        with self.assertRaises(ExecutionError):
            self.runtime.execute(new, task.disclosed, ToolCall("navi_get_status", {}))

    def test_added_domain_is_used_only_by_new_snapshot(self):
        old = self.registry.current
        added = deepcopy(self.data["tools"][7])
        added.update(name="synthetic_status", domain="synthetic")
        self.data["tools"].append(added)
        self.write()
        new = self.registry.reload()
        self.assertEqual(set(old.domains()), {"navigation"})
        self.assertEqual(set(new.domains()), {"navigation", "synthetic"})
        self.assertNotIn("synthetic_status", old.tools)
        self.assertEqual(new.domains()["synthetic"][0].name, "synthetic_status")

    def test_failed_reload_never_publishes_partial_version(self):
        old = self.registry.current
        for mutate in [
            lambda: self.data["tools"].append(deepcopy(self.data["tools"][0])),
            lambda: self.data["tools"][0].update(binding="unregistered"),
            lambda: self.data["tools"][0].update(parameters={"type": "invalid"}),
            lambda: self.data["tools"][0].update(name="ask_user"),
        ]:
            self.data = yaml.safe_load((ROOT / "tools.yaml").read_text())
            mutate()
            self.write()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.registry.reload()
            self.assertIs(self.registry.current, old)

    def test_http_reload_returns_active_version_on_rejection(self):
        with patch.dict(os.environ, {"TOOLS_FILE": str(self.path)}):
            client = create_app().test_client()
        old = client.get("/health").get_json()["registry_version"]
        self.path.write_text("tools: [")
        response = client.post("/api/registry/reload")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["active_version"], old)
        self.assertEqual(client.get("/health").get_json()["registry_version"], old)
        self.data["tools"][0]["description"] = "new description"
        self.write()
        response = client.post("/api/registry/reload")
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.get_json()["version"], old)


if __name__ == "__main__":
    unittest.main()
