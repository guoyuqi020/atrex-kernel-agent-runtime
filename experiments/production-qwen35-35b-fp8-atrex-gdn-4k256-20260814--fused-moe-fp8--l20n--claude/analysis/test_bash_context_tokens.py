import unittest

from bash_context_tokens import fragment_counts, prepare_fragments, replay


class CharacterEncoder:
    def encode_ordinary(self, text):
        return list(text)


def assistant(mid, uuid, blocks=(), chain=None, usage=None):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parent_tool_use_id": chain,
        "message": {"id": mid, "content": list(blocks), "usage": usage or {}},
    }


def command(tid, text):
    return {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": text}}


def result(tid, text, uuid, chain=None):
    return {
        "type": "user",
        "uuid": uuid,
        "parent_tool_use_id": chain,
        "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]},
    }


def compact(keep):
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "compact_metadata": {"preserved_messages": {"all_uuids": keep}},
    }


def run(events):
    numbered = list(enumerate(events, 1))
    annotations = {"t": {"flags": ["gpu_entry"]}, "child_t": {"flags": []}}
    fragments = prepare_fragments(numbered, annotations, {"test": CharacterEncoder()})
    diagnostics, requests = replay(numbered, fragments)
    return fragments, diagnostics, requests


class ContextReplayTests(unittest.TestCase):
    def test_generation_once_result_only_input_and_stream_dedup(self):
        fragments, diagnostics, _ = run(
            [
                assistant("m1", "thinking"),
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
                assistant("m2", "r2"),
                assistant("m2", "r2b"),
                assistant("m3", "r3"),
            ]
        )
        self.assertEqual(diagnostics["observed_requests"], 3)
        self.assertEqual([f["reads"]["preserved_messages"] for f in fragments], [2, 2])
        self.assertEqual(
            sum(fragment_counts(f)["test_preserved_messages_total"] for f in fragments), 19
        )

    def test_compaction_preserves_only_exact_fragment_uuids(self):
        fragments, diagnostics, _ = run(
            [
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
                assistant("m2", "r2"),
                compact(["out"]),
                assistant("m3", "r3"),
            ]
        )
        self.assertEqual([f["reads"]["preserved_messages"] for f in fragments], [1, 2])
        self.assertEqual([f["reads"]["reset_at_compaction"] for f in fragments], [1, 1])
        self.assertEqual([f["reads"]["ignore_compaction"] for f in fragments], [2, 2])
        self.assertEqual([f["compaction_passes"] for f in fragments], [1, 1])
        self.assertEqual(diagnostics["bash_fragments_preserved_at_boundaries"], 1)

    def test_native_camelcase_compaction_and_unobserved_uuid(self):
        boundary = {
            "type": "system",
            "subtype": "compact_boundary",
            "compactMetadata": {"preservedMessages": {"allUuids": ["cmd", "unknown"]}},
        }
        fragments, diagnostics, _ = run(
            [
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
                boundary,
                assistant("m2", "r2"),
            ]
        )
        self.assertEqual([f["reads"]["preserved_messages"] for f in fragments], [1, 0])
        self.assertEqual(diagnostics["unobserved_preserved_uuids"], 1)

    def test_child_requests_do_not_read_main_context(self):
        fragments, diagnostics, _ = run(
            [
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
                assistant("c1", "child_cmd", [command("child_t", "x")], chain="agent1"),
                result("child_t", "yy", "child_out", chain="agent1"),
                assistant("c2", "child2", chain="agent1"),
                assistant("m2", "r2"),
            ]
        )
        self.assertEqual(diagnostics["child_requests"], 2)
        self.assertEqual([f["reads"]["preserved_messages"] for f in fragments], [1, 1, 1, 1])

    def test_duplicate_result_is_not_replayed_as_new_text(self):
        fragments, _, _ = run(
            [
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
                result("t", "12345", "out2"),
                assistant("m2", "r2"),
            ]
        )
        self.assertEqual(len(fragments), 2)
        self.assertEqual([f["reads"]["preserved_messages"] for f in fragments], [1, 1])

    def test_no_later_request_does_not_charge_tool_output(self):
        fragments, _, _ = run(
            [
                assistant("m1", "cmd", [command("t", "abc")]),
                result("t", "12345", "out"),
            ]
        )
        self.assertEqual(
            sum(fragment_counts(f)["test_preserved_messages_total"] for f in fragments), 3
        )

    def test_usage_updated_not_summed(self):
        _, _, requests = run(
            [
                assistant("m1", "a", usage={"input_tokens": 10}),
                assistant("m1", "b", usage={"input_tokens": 20}),
            ]
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["usage_last_observed"], {"input_tokens": 20})

    def test_missing_boundary_metadata_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing preserved-message"):
            run([{"type": "system", "subtype": "compact_boundary"}])


if __name__ == "__main__":
    unittest.main()
