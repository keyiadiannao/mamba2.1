from __future__ import annotations

import unittest

from src.tracing import TraceRecord, make_run_id


class TraceSchemaTest(unittest.TestCase):
    def test_make_run_id_uses_utc_suffix_z(self) -> None:
        run_id = make_run_id("prefix")
        self.assertTrue(run_id.startswith("prefix_"))
        self.assertTrue(run_id.endswith("Z"))
        self.assertRegex(run_id, r"^prefix_\d{8}_\d{6}Z$")

    def test_trace_to_dict_includes_stable_fingerprint(self) -> None:
        trace = TraceRecord(routing_mode="rule", context_source="t1_visited_leaves_ordered")
        trace.nav_success = True
        trace.rollback_count = 1
        trace.finalize()
        d1 = trace.to_dict()
        self.assertIn("trace_fingerprint_sha256", d1)
        self.assertEqual(len(d1["trace_fingerprint_sha256"]), 64)
        d2 = trace.to_dict()
        self.assertEqual(d1["trace_fingerprint_sha256"], d2["trace_fingerprint_sha256"])

    def test_trace_fingerprint_changes_when_scalar_metric_changes(self) -> None:
        t1 = TraceRecord(routing_mode="rule", context_source="x")
        t1.nav_success = True
        t1.finalize()
        fp1 = t1.to_dict()["trace_fingerprint_sha256"]

        t2 = TraceRecord(routing_mode="rule", context_source="x")
        t2.nav_success = False
        t2.finalize()
        fp2 = t2.to_dict()["trace_fingerprint_sha256"]

        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
