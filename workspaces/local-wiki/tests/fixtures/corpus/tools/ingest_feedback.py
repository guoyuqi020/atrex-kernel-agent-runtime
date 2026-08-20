#!/usr/bin/env python3
"""Minimal upstream-compatible feedback fixture."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "kernel_wiki" / "feedback" / "events.jsonl"


def report(record_id: str, outcome: str, note: str, now: float) -> list[dict[str, object]]:
    if record_id != "nvidia.hopper.triton.kernel-opt.reduction":
        raise SystemExit(f"unknown record id: {record_id}")
    return [
        {
            "ts": now,
            "record_id": record_id,
            "kind": "increment",
            "source": "agent-report",
            "key": f"report:{record_id}:{outcome}:{now:f}",
            "counts": {"served_count": 1},
            "note": note,
        }
    ]


def append(events: list[dict[str, object]]) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
