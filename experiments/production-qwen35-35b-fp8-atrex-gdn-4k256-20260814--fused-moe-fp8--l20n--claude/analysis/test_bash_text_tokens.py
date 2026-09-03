"""Small offline regression tests; run with python -m unittest from this directory."""

import json
import unittest

from bash_text_tokens import content_text, digest, extract_bash


def use(tool_id: str, command: str = "pwd") -> dict:
    return {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": command}}


def result(tool_id: str, text: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": text}


def event(*blocks: dict) -> dict:
    return {"message": {"content": list(blocks)}}


def encode(*events: dict) -> bytes:
    return "\n".join(json.dumps(value) for value in events).encode()


class ExtractionTests(unittest.TestCase):
    def test_duplicate_tool_and_result_blocks_are_not_recounted(self) -> None:
        call, response = event(use("t1")), event(result("t1", "/workspace"))
        tools = extract_bash(encode(call, call, response, response), False)
        tool = tools[(1, digest(b"pwd"))][0]
        self.assertEqual(tool["results"], ["/workspace"])
        self.assertEqual(sum(map(len, tools.values())), 1)

    def test_same_command_same_line_with_distinct_ids_remains_two_calls(self) -> None:
        tools = extract_bash(
            encode(
                event(use("t1"), use("t2")),
                event(result("t1", "first"), result("t2", "second")),
            ),
            False,
        )
        calls = tools[(1, digest(b"pwd"))]
        self.assertEqual([tool["results"] for tool in calls], [["first"], ["second"]])

    def test_distinct_results_remain_visible(self) -> None:
        tools = extract_bash(
            encode(
                event(use("t1")),
                event(result("t1", "progress")),
                event(result("t1", "finished")),
            ),
            False,
        )
        self.assertEqual(tools[(1, digest(b"pwd"))][0]["results"], ["progress", "finished"])

    def test_runtime_wrapper_and_missing_result(self) -> None:
        tools = extract_bash(
            encode(
                {"type": "message", "content": "initial prompt"},
                {"type": "provider_event", "event": event(use("t1", "cat kernel.py"))},
            ),
            True,
        )
        self.assertEqual(tools[(2, digest(b"cat kernel.py"))][0]["results"], [])

    def test_text_block_boundary_matches_existing_character_audit(self) -> None:
        self.assertEqual(
            content_text(
                [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]
            ),
            "one\ntwo",
        )


if __name__ == "__main__":
    unittest.main()
