import unittest

from bash_adjacent_thinking import counts, thinking_fragments
from bash_context_tokens import replay
from test_bash_context_tokens import CharacterEncoder, assistant, command, compact, result


def thought(mid, uid, text="abcd", chain=None):
    return assistant(mid, uid, [{"type": "thinking", "thinking": text}], chain)


def run(events, tools=("a", "b")):
    numbered = list(enumerate(events, 1))
    fragments, diagnostics = thinking_fragments(numbered, dict.fromkeys(tools), CharacterEncoder())
    replay(numbered, fragments)
    return fragments, diagnostics


class AdjacentThinkingTests(unittest.TestCase):
    def test_shared_after_before_thinking_counts_once(self):
        fragments, _ = run(
            [
                thought("m1", "think1"),
                assistant("m1", "call1", [command("a", "ls")]),
                result("a", "files", "out1"),
                thought("m2", "think2"),
                assistant("m2", "call2", [command("b", "cat file")]),
                result("b", "data", "out2"),
                thought("m3", "think3"),
            ]
        )
        self.assertEqual([f["relation"] for f in fragments], ["before", "both", "after"])
        self.assertEqual(fragments[1]["neighbor_tool_ids"], ["a", "b"])
        self.assertEqual(sum(counts(f)["generated"] for f in fragments), 12)
        self.assertEqual(sum(counts(f)["preserved_messages_total"] for f in fragments), 24)

    def test_prose_metadata_transparent_and_stream_duplicates_removed(self):
        repeated = thought("m", "thinking")
        fragments, diagnostics = run(
            [
                repeated,
                {"type": "attachment"},
                repeated,
                assistant("m", "explain", [{"type": "text", "text": "I will read it"}]),
                assistant("m", "cmd", [command("a", "cat x")]),
            ]
        )
        self.assertEqual(len(fragments), 1)
        self.assertEqual(diagnostics["duplicate_thinking_blocks"], 1)

    def test_parallel_batch_results_can_interleave_with_calls(self):
        fragments, _ = run(
            [
                thought("m1", "t1"),
                assistant("m1", "a", [command("a", "ls")]),
                result("a", "files", "r1"),
                assistant("m1", "b", [command("b", "pwd")]),
                result("b", "path", "r2"),
                thought("m2", "t2"),
            ]
        )
        self.assertEqual(fragments[0]["before_tool_ids"], ["a", "b"])
        self.assertEqual(fragments[1]["after_tool_ids"], ["a", "b"])

    def test_non_bash_action_not_skipped_to_find_older_bash(self):
        read = {"type": "tool_use", "name": "Read", "id": "r", "input": {}}
        fragments, _ = run(
            [
                assistant("m1", "a", [command("a", "ls")]),
                result("a", "files", "r1"),
                assistant("m2", "read", [read]),
                result("r", "file", "r2"),
                thought("m3", "t3"),
            ]
        )
        self.assertEqual(fragments, [])

    def test_mixed_non_bash_neighbor_is_flagged(self):
        read = {"type": "tool_use", "name": "Read", "id": "r", "input": {}}
        fragments, _ = run(
            [
                assistant("m1", "read", [read]),
                result("r", "file", "out"),
                thought("m2", "t2"),
                assistant("m2", "a", [command("a", "ls")]),
            ]
        )
        self.assertEqual(fragments[0]["non_bash_neighbors"], ["r"])

    def test_compaction_is_adjacency_barrier_and_uuid_controls_replay(self):
        fragments, _ = run(
            [
                thought("m1", "t1"),
                assistant("m1", "a", [command("a", "ls")]),
                result("a", "files", "r1"),
                compact(["a"]),
                thought("m2", "t2"),
            ]
        )
        self.assertEqual(len(fragments), 1)
        self.assertEqual(counts(fragments[0])["preserved_messages_input"], 0)
        self.assertEqual(counts(fragments[0])["ignore_compaction_input"], 4)

    def test_user_message_is_barrier(self):
        fragments, _ = run(
            [
                assistant("m1", "a", [command("a", "ls")]),
                result("a", "files", "r1"),
                {"type": "user", "message": {"content": [{"type": "text", "text": "new topic"}]}},
                thought("m2", "t2"),
            ]
        )
        self.assertEqual(fragments, [])

    def test_chain_local_adjacency_and_rereads(self):
        fragments, _ = run(
            [
                thought("m1", "t1"),
                assistant("m1", "a", [command("a", "ls")]),
                result("a", "files", "r1"),
                thought("c1", "ct1", chain="child"),
                assistant("c1", "b", [command("b", "pwd")], chain="child"),
                result("b", "path", "cr1", chain="child"),
                thought("m2", "t2"),
            ]
        )
        child = next(f for f in fragments if f["chain"] == "child")
        self.assertEqual(child["relation"], "before")
        self.assertEqual(counts(child)["preserved_messages_input"], 0)
        main = next(f for f in fragments if f["uuid"] == "t1")
        self.assertEqual(counts(main)["preserved_messages_input"], 4)


if __name__ == "__main__":
    unittest.main()
