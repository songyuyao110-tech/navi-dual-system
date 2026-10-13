from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml

from decisions import Tool
from model_clients import CONTROL_TOOLS


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    domain: str
    description: str
    schema_json: str
    binding: str
    side_effect: str

    @property
    def parameters(self) -> dict:
        return json.loads(self.schema_json)


@dataclass(frozen=True)
class RegistrySnapshot:
    version: str
    tools: Mapping[str, RegisteredTool]

    def domains(self) -> dict[str, tuple[Tool, ...]]:
        result: dict[str, list[Tool]] = {}
        for tool in self.tools.values():
            result.setdefault(tool.domain, []).append(Tool(tool.name, tool.description))
        return {domain: tuple(tools) for domain, tools in result.items()}

    def schemas(self, names: frozenset[str]) -> list[dict]:
        unknown = names - self.tools.keys()
        if unknown:
            raise ValueError(f"Unregistered disclosed tools: {sorted(unknown)}")
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in self.tools.values()
            if tool.name in names
        ]


def load_registry(path: Path, bindings: frozenset[str]) -> RegistrySnapshot:
    text = path.read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ValueError("Registry YAML is malformed") from None
    if not isinstance(data, dict) or set(data) != {"tools"}:
        raise ValueError("Registry requires a tools list")
    if not isinstance(data["tools"], list) or not data["tools"]:
        raise ValueError("Registry tools must be a nonempty list")
    tools = {}
    required = {"name", "domain", "description", "parameters", "binding", "side_effect"}
    for entry in data["tools"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Tool registration must contain exactly {sorted(required)}")
        for field in required - {"parameters"}:
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"Tool registration requires nonempty {field}")
        name = entry["name"]
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", entry[field]) is None
            for field in ("name", "domain")
        ):
            raise ValueError("Tool and domain identifiers must be ASCII identifiers")
        if entry["domain"] == "A" or name in {tool["name"] for tool in CONTROL_TOOLS}:
            raise ValueError(f"Reserved tool or domain identifier: {name}")
        if name in tools:
            raise ValueError(f"Duplicate tool identifier: {name}")
        if entry["binding"] not in bindings:
            raise ValueError(f"Unknown execution binding for {name}")
        if entry["side_effect"] not in {"idempotent", "non_idempotent", "conditional"}:
            raise ValueError(f"Invalid side-effect annotation for {name}")
        schema = entry["parameters"]
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"Parameters for {name} must be an object schema")
        try:
            Draft202012Validator.check_schema(schema)
            schema_json = json.dumps(schema, ensure_ascii=False, allow_nan=False)
        except (SchemaError, TypeError, ValueError):
            raise ValueError(f"Invalid JSON parameter schema for {name}") from None
        tools[name] = RegisteredTool(
            name=name,
            domain=entry["domain"],
            description=entry["description"],
            schema_json=schema_json,
            binding=entry["binding"],
            side_effect=entry["side_effect"],
        )
    return RegistrySnapshot(
        hashlib.sha256(text.encode()).hexdigest()[:16], MappingProxyType(tools)
    )


class Registry:
    def __init__(self, path: Path, bindings: frozenset[str]):
        self.path = path
        self.bindings = bindings
        self.lock = RLock()
        self._snapshot = load_registry(path, bindings)

    @property
    def current(self) -> RegistrySnapshot:
        with self.lock:
            return self._snapshot

    def reload(self) -> RegistrySnapshot:
        with self.lock:
            candidate = load_registry(self.path, self.bindings)
            self._snapshot = candidate
            return candidate
