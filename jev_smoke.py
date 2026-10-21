from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import shlex

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from decisions import Tool, run_b_decisions, serialize_results

logger = logging.getLogger(__name__)
ENV_NAMES = {"TYPESAFE_API_KEY", "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"}


def load_environment(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in ENV_NAMES:
            continue
        try:
            values = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f"Malformed environment assignment for {name}") from None
        if len(values) != 1:
            raise ValueError(f"Expected one environment value for {name}")
        os.environ.setdefault(name, values[0])


async def probe(model: str | None) -> dict:
    domains = {
        "navigation": (
            Tool("navi_to_special_poi", "导航到用户已设置的家庭或公司地址"),
            Tool("navi_get_status", "读取当前导航状态"),
            Tool("navi_cancel_route", "取消正在进行的导航"),
        )
    }
    async with AsyncTypeSafeClient(
        model=model, retry=RetryPolicy(max_retries=0), timeout=30.0
    ) as client:
        models = await client.models.list()
        results = await run_b_decisions(
            client,
            '用户当前话："导航回家"。测试上下文：已设置家庭地址。本轮无挂起任务。',
            domains,
        )
    report = serialize_results(results)
    report["available_models"] = [item.name for item in models.models]
    report["sample"] = "synthetic_navigation_home"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real JEV B-only tool disclosure")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        if args.env_file:
            load_environment(args.env_file)
        report = asyncio.run(probe(args.model))
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
        return 0
    except TypeSafeError as error:
        logger.error(
            "JEV request failed: %s, HTTP status=%s",
            type(error).__name__,
            getattr(error, "status_code", "not reported"),
        )
        return 1
    except (OSError, ValueError) as error:
        logger.error("JEV probe failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
