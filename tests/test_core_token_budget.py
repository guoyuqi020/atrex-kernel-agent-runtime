"""Black-box checks for Core's live token-only session quota."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "atrex-kernel-agent-core"
sys.path.insert(0, str(CORE_ROOT / "src"))

from backends.adapter import ClaudeAdapter, QoderAdapter  # noqa: E402
from backends.process import run_bounded  # noqa: E402
from backends.runtime import TokenBudgetObserver  # noqa: E402


def test_live_token_budget_terminates_complete_process_group(tmp_path: Path) -> None:
    observer = TokenBudgetObserver(ClaudeAdapter(), 10)
    event = json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": "message-1",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    )
    script = f"import time; print({event!r}, flush=True); time.sleep(30)"
    started = time.monotonic()

    result = run_bounded(
        [sys.executable, "-c", script],
        tmp_path,
        timeout=20,
        observer=observer,
    )

    assert event in result.stdout
    assert observer.exhausted is True
    assert result.returncode != 0
    assert result.timed_out is False
    assert time.monotonic() - started < 5


def test_live_token_budget_deduplicates_repeated_provider_message() -> None:
    observer = TokenBudgetObserver(ClaudeAdapter(), 15)
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": "same-message",
                "usage": {
                    "input_tokens": 6,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    )

    assert observer.on_stdout_line(line) is False
    assert observer.on_stdout_line(line) is False
    assert observer.exhausted is False


def test_qoder_ignores_intermediate_usage_until_credits_arrive() -> None:
    observer = TokenBudgetObserver(QoderAdapter(), 100)
    intermediate = json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": "qoder-message",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    )
    billed = json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": "qoder-message",
                "usage": {"credits": 12.5},
            },
        }
    )

    assert observer.on_stdout_line(intermediate) is False
    assert observer.monitoring_failed is False
    assert observer.on_stdout_line(billed) is False
    assert observer.exhausted is False


def test_qoder_credit_budget_is_recorded_and_enforced() -> None:
    observer = TokenBudgetObserver(QoderAdapter(), 10.0)
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": "qoder-message",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "credits": 10.5,
                },
            },
        }
    )

    assert observer.on_stdout_line(line) is True
    assert observer.exhausted is True
    assert observer.monitoring_failed is False
