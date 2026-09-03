import unittest

from shell_read_audit import classify, summarize


class ShellReadAuditTests(unittest.TestCase):
    def test_heredoc_body_is_not_an_executed_read(self):
        bucket, detail = classify("cat > scratch/r.json <<'EOF'\ncat kernel.py | head\nEOF\n", [])
        self.assertEqual(bucket, "write_only_name_match")
        self.assertEqual(detail["reader_matches"], 0)

    def test_writer_then_api_is_not_a_file_read(self):
        bucket, _ = classify(
            "cat > scratch/q.json <<'EOF'\n{}\nEOF\npython3 runtime_tools.py wiki-query",
            ["runtime_wiki-query"],
        )
        self.assertEqual(bucket, "write_only_name_match")

    def test_gpu_pipe_is_execution_not_reading_source(self):
        bucket, _ = classify(
            "python tools/sandbox.py --kind run -- python test_kernel.py 2>&1 | tail -40",
            ["gpu_entry"],
        )
        self.assertEqual(bucket, "gpu_execution_and_filter")

    def test_help_is_not_a_real_gpu_submission(self):
        bucket, _ = classify("python tools/sandbox.py --help | head -40", ["gpu_entry"])
        self.assertEqual(bucket, "cli_help")

    def test_memory_and_reference_sources(self):
        self.assertEqual(
            classify("cat memory/v7.json | head -200", [])[0], "history_plan_report_files"
        )
        self.assertEqual(
            classify("sed -n '100,200p' reference-projects/cutlass/example.py", [])[0],
            "reference_library_code",
        )

    def test_sed_inplace_is_editing(self):
        self.assertEqual(classify("sed -i.bak 's/a/b/' kernel.py", [])[0], "write_only_name_match")

    def test_harness_source_is_not_measured_output(self):
        self.assertEqual(
            classify("grep -n ncu tools/profile_nvidia.sh", [])[0], "harness_framework_code"
        )

    def test_exclusive_buckets_sum_to_total(self):
        rows = []
        for group in ("AKA", "retained"):
            for bucket in ("cli_help", "write_only_name_match"):
                rows.append(
                    {
                        "group": group,
                        "session": "test",
                        "command_sha256": bucket,
                        "result_sha256": bucket,
                        "bucket": bucket,
                        "visible_tokens": 10,
                        "command_total_tokens": 20,
                        "result_total_tokens": 30,
                        "total_tokens": 50,
                    }
                )
        summary = summarize(rows)
        for group in summary["groups"].values():
            self.assertEqual(group["total"]["total_tokens"], 100)
            self.assertEqual(sum(b["calls"] for b in group["buckets"].values()), 2)


if __name__ == "__main__":
    unittest.main()
