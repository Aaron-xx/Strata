"""Offline and live tests for tools/dsh_serve_regression.py.

    python -m unittest tools.test_dsh_serve_regression -v
    python tools/test_dsh_serve_regression.py
    python -m pytest tools/test_dsh_serve_regression.py

The live case skips itself when no Strata server with a loaded model is up, so
the file is safe to include in a GPU-less test sweep. When a server IS up, it
runs the three-column gate once (--runs 1) and asserts every gate passes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dsh_serve_regression as R  # noqa: E402


def _tool_out(arguments):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "call_x", "type": "function",
         "function": {"name": "glob", "arguments": arguments}}]}}]}


class Offline(unittest.TestCase):
    def test_garble_detects_historical_doc_mode(self):
        samples = [
            "It seems like your message is garbled or contains typos.",
            "Based on the fragmented text provided, it appears you are asking about glob.",
            "This appears to be a corrupted or obfuscated version of a Python signature.",
            "The text is heavily corrupted or phonetically distorted.",
        ]
        for s in samples:
            self.assertTrue(R.is_garbled(s), s)

    def test_garble_allows_clean_answers(self):
        samples = ["4", "grey", "BLUE-HERON-4471 9312",
                   '{"pattern": "src/prefill/**/*.cu"}',
                   "A transformer attention layer weighs each token by its key."]
        for s in samples:
            self.assertFalse(R.is_garbled(s), s)

    def test_tool_call_ok_requires_pattern(self):
        self.assertTrue(R.tool_call_ok(_tool_out('{"pattern":"src/prefill/*.cu"}')))
        self.assertFalse(R.tool_call_ok(_tool_out('{}')))
        self.assertFalse(R.tool_call_ok(_tool_out('not json')))
        self.assertFalse(R.tool_call_ok({"choices": [{"message": {"content": "no call"}}]}))

    def test_doc_needle_is_at_the_end(self):
        line = R._doc(3, "CODENAME-TEST", 9312).splitlines()[-1]
        self.assertIn("CODENAME-TEST", line)
        self.assertIn("9312", line)

    def test_needle_pair_changes_every_call(self):
        a, b = R._needle_pair(), R._needle_pair()
        self.assertNotEqual(a, b)
        self.assertTrue(a[0].startswith("CODENAME-"))
        self.assertGreaterEqual(a[1], 4000)

    def test_server_ready_is_false_when_nothing_listens(self):
        self.assertFalse(R.server_ready("http://127.0.0.1:1", timeout=1))


@unittest.skipUnless(R.server_ready(),
                     "no live Strata server with a loaded model")
class Live(unittest.TestCase):
    def test_serve_regression_three_columns(self):
        res = R.run(base=R.DEFAULT_BASE, runs=1, long=False,
                    min_prefill=180.0, min_decode=20.0, timeout=300)
        self.assertEqual([s["name"] for s in res["sections"]],
                         ["PREFILL", "DECODE", "QUALITY"])
        self.assertTrue(res["ok"] is True, res["summary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
