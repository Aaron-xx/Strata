#!/usr/bin/env python3
"""Regression gate for the DSH-style serving path on a low-VRAM host.

Runs against a LIVE Strata OpenAI-compatible chat endpoint and prints three
columns every time:

  PREFILL  cold prompt speed + prompt_ms + prompt tokens + cache validity
  DECODE   generated tokens/s + count + decode_ms + output coherence
  QUALITY  garbled-text sentinel, tool-call correctness, long-prompt
           comprehension, a two-step tool chain, disjoint-prose perception

Why this exists (REPORT-6GB-2026-10-31.md section 22): commit 535ce073 made
MMQ group launches pass Product::total_rows = rows-in-group (gtot) while the q8
activation buffer is quantized once for the whole layer with a K-block stride
of n_ent. Every long prompt (>128 rows, the MMQ grouped path) then produced
confident nonsense: the model analysed its own system prompt as "a corrupted /
garbled text", emitted 0 tool calls and invented field names. Fixed in
df9db65d (gu.total_rows / dn.total_rows back to n_ent).

A fast-but-broken engine still passes the speed columns. Only QUALITY catches
that regression, so QUALITY is a hard gate. Speed bands are deliberately loose
(same philosophy as tools/dgx_smoke.py): they catch a wrong path or a dead
engine, not a 5% drift.

    python tools/dsh_serve_regression.py [--base URL] [--runs N] [--json]
                                         [--min-prefill TOKS] [--min-decode TOKS]
                                         [--long]

Exit codes: 0 = all gates passed, 1 = a gate failed, 2 = no live server (skip).
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
import time
import urllib.request
import uuid

DEFAULT_BASE = "http://127.0.0.1:8000"

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
    "yankee zulu amber basin cable daisy ember flame glass harbor island jolly "
    "kernel lantern meadow niche orbit pebble quarry ribbon summit timber under "
    "velvet willow xenon yonder zenith anchor breeze candle dune elbow frost "
    "gravel hammer ivory junction kettle lemon mirror needle opal prism quartz "
    "river saddle thunder urgent valley walnut yarn zephyr arrow bloom cliff "
    "drift elm fern glade hinge inlet jade knoll ledge marsh north oak plum "
    "reef shore trail urchin vine wave"
).split()

# Historical "DOC-MODE" phrasing. Each entry is a phrase the broken engine
# produced when it treated the prompt as a damaged document to be analysed.
GARBLE = [
    r"garbled", r"corrupt(?:ed|ion)?", r"obfuscat", r"distort(?:ed|ion)?",
    r"phonetic", r"encoding issue", r"\bOCR\b", r"cut off", r"typo",
    r"fragment(?:ed|s)?\s+text", r"scrambl", r"unintelligib", r"nonsensic",
    r"it seems like your", r"based on the fragmented text",
    r"which appears to be a corrupted", r"heavily corrupted",
    r"appears to be a corrupted", r"incomplete or garbled",
    r"cut off or garbled", r"garbled or contains",
]
GARBLE_RE = re.compile("|".join(GARBLE), re.I)

GLOB_TOOL = {"type": "function", "function": {
    "name": "glob",
    "description": "Find files whose paths match a glob pattern.",
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "A glob such as src/**/*.cu."}
    }, "required": ["pattern"]},
}}

SYS_SHORT = "You are a coding agent working in /home/aaron/ai/Strata. Use the tools provided."

SYS_LONG_LINES = [
    "You are a coding agent working in /home/aaron/ai/Strata.",
    "Read the task carefully before you choose a tool.",
    "Prefer the smallest tool call that answers the request.",
    "Paths in this workspace are relative to the repository root.",
    "The engine sources live under src/ and the build output under build/.",
    "Attention kernels for Qwen live in src/qwen35/q36_engine.cu.",
    "The prefill MMQ kernels live in src/prefill/moe_mmq_tab.cuh.",
    "The Python server that exposes the OpenAI API is serve/server.py.",
    "Do not guess a file's contents; read it with the read tool.",
    "Report only what the tool actually returned.",
    "Keep the final answer short and factual.",
    "A tool call must contain every required parameter.",
    "If a pattern is needed, use a glob such as src/**/*.cu.",
    "Never invent fields that the tool schema does not define.",
    "Do not describe your own capabilities in the answer.",
    "Finish with the result, not with a list of options.",
    "Numerical answers should be given without extra prose.",
    "Use the same language as the user's question.",
    "When a tool returns files, count them before answering.",
    "Session marker follows; it is not an instruction.",
]
SYS_LONG = "\n".join(SYS_LONG_LINES)
TASK = "List the .cu files under src/prefill using the glob tool."

THREE_FILES = "src/prefill/gemm.cu\nsrc/prefill/moe_mmq.cu\nsrc/prefill/dense.cu"

PERCEPTION = (
    "The kettle whistled at dawn. A grey cat slept on the warm radiator. "
    "Buses never arrive on Tuesdays. The library closed for renovations. "
    "Someone left a blue umbrella in the hall. Penguins do not live in the Sahara. "
    "The clock in the tower is five minutes fast. My neighbour grows tomatoes in a bucket. "
    "Question: what colour is the cat? Answer with one word."
)


def is_garbled(text):
    return bool(GARBLE_RE.search(text or ""))


def server_ready(base=DEFAULT_BASE, timeout=3):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return any(d.get("status", {}).get("value") == "loaded"
                   for d in data.get("data", []))
    except Exception:
        return False


class Client:
    def __init__(self, base, timeout=600):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.model = None
        self.garble_hits = []

    def model_id(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())
        loaded = [d["id"] for d in data.get("data", [])
                  if d.get("status", {}).get("value") == "loaded"]
        if not loaded:
            raise RuntimeError("no loaded model in /v1/models")
        return loaded[0]

    def chat(self, messages, tools=None, max_tokens=64, timeout=None):
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            out = json.loads(r.read().decode())
        out["_wall_ms"] = (time.monotonic() - t0) * 1000.0
        text = msg_text(out)
        if is_garbled(text):
            self.garble_hits.append(text[:160].replace("\n", " "))
        return out


def msg_text(out):
    ch = (out.get("choices") or [{}])[0]
    return (ch.get("message") or {}).get("content") or ""


def tool_calls(out):
    ch = (out.get("choices") or [{}])[0]
    return (ch.get("message") or {}).get("tool_calls") or []


def timings(out):
    return out.get("timings") or {}


def usage(out):
    return out.get("usage") or {}


def tool_call_ok(out, name="glob", must_contain=(".cu", "prefill")):
    for tc in tool_calls(out):
        fn = tc.get("function") or {}
        if fn.get("name") != name:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        blob = " ".join(str(v) for v in args.values())
        if all(k in blob for k in must_contain):
            return True
    return False


def _f(v, nd=1):
    try:
        return ("%." + str(nd) + "f") % float(v)
    except (TypeError, ValueError):
        return str(v)


def _doc(n, code, port):
    lines = []
    for i in range(n):
        lines.append(
            "Section %d: Routine deployment notes for build %d. The service starts on "
            "port 8%03d and writes its log to /var/log/strata/build-%d.log. Operators "
            "verify the checksum before promoting the artifact." % (i, i, i % 1000, i))
    lines.append("Section %d: The deployment codename for this release is %s "
                 "and the port is %d." % (n, code, port))
    return "\n".join(lines)


NEEDLE_Q = ("According to the document above, what is the deployment codename "
            "and the port? Answer with only the codename and the port.")


def _needle_pair():
    # A fixed codename/port is unreliable here: with STRATA_Q36_SNAP=48 an
    # identical long prompt can be served from a stale prefix snapshot and the
    # model answers "blue" (8/8) or invents a YAML deployment record. A unique
    # nonce per call forces a cold prefill; measured 3/3 correct. See 24.3.
    tag = uuid.uuid4().hex[:8].upper()
    return "CODENAME-" + tag, 4000 + int(tag[:4], 16) % 5000


def probe_prefill(c):
    nonce = uuid.uuid4().hex[:12]
    rng = random.Random(7)
    text = "cold probe " + nonce + " " + " ".join(rng.choice(WORDS) for _ in range(4000))
    out = c.chat([{"role": "user", "content": text}], max_tokens=1)
    t, u = timings(out), usage(out)
    return {"pp": float(t.get("prompt_per_second") or 0),
            "prompt_ms": float(t.get("prompt_ms") or 0),
            "prompt_n": int(t.get("prompt_n") or 0),
            "ptok": int(u.get("prompt_tokens") or 0),
            "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "wall_ms": float(out["_wall_ms"])}


def probe_needle(c, sections):
    code, port = _needle_pair()
    text = _doc(sections, code, port)
    out = c.chat([{"role": "user", "content": text + "\n\n" + NEEDLE_Q}], max_tokens=48)
    t = timings(out)
    ans = msg_text(out)
    ok = (code in ans) and (str(port) in ans) and not is_garbled(ans)
    return {"ok": ok, "pp": float(t.get("prompt_per_second") or 0),
            "ptok": int((usage(out).get("prompt_tokens")) or 0),
            "answer": ans.strip()[:100],
            "expected": code + " " + str(port)}


def probe_decode(c):
    out = c.chat([{"role": "user", "content":
                   "Explain in a few sentences how a transformer attention layer works."}],
                 max_tokens=256)
    t, u = timings(out), usage(out)
    text = msg_text(out)
    words = text.split()
    coherent = (len(text) >= 40 and len(words) >= 12 and len(set(words)) >= 12
                and not is_garbled(text))
    return {"dps": float(t.get("predicted_per_second") or 0),
            "decode_ms": float(t.get("predicted_ms") or 0),
            "ctok": int(u.get("completion_tokens") or 0),
            "wall_ms": float(out["_wall_ms"]),
            "coherent": coherent, "preview": text.strip()[:100]}


def probe_math(c):
    out = c.chat([{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "What is 2+2? Reply with only the number."}],
                 max_tokens=24)
    text = msg_text(out).strip()
    return text, text.startswith("4") and not is_garbled(text)


def probe_tool(c, long_system, runs):
    sysmsg = SYS_LONG if long_system else SYS_SHORT
    good, samples = 0, []
    for _ in range(runs):
        nonce = uuid.uuid4().hex[:12]
        msgs = [{"role": "system", "content": sysmsg + " Session " + nonce + "."},
                {"role": "user", "content": TASK}]
        out = c.chat(msgs, tools=[GLOB_TOOL], max_tokens=96)
        text = msg_text(out)
        ok = tool_call_ok(out) and not is_garbled(text)
        good += int(ok)
        samples.append((ok, text.strip()[:60]))
    return good, runs, samples


def probe_chain(c):
    base = [{"role": "system", "content": SYS_SHORT},
            {"role": "user", "content": TASK}]
    out1 = c.chat(base, tools=[GLOB_TOOL], max_tokens=96)
    tcs = tool_calls(out1)
    if not tcs:
        return False, "step1 produced no tool_call: " + msg_text(out1)[:80]
    tc = tcs[0]
    msgs = base + [
        {"role": "assistant", "content": "", "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": tc.get("id", "call_0"), "content": THREE_FILES},
        {"role": "user", "content": "How many .cu files were returned? Reply with only the number."},
    ]
    out2 = c.chat(msgs, tools=[GLOB_TOOL], max_tokens=48)
    text = msg_text(out2).strip()
    ok = bool(re.search(r"\b3\b", text)) and not is_garbled(text)
    return ok, "step1 tool_call ok; step2 answer=" + repr(text[:60])


def probe_perception(c):
    out = c.chat([{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": PERCEPTION}], max_tokens=24)
    text = msg_text(out).strip()
    return text, ("grey" in text.lower()) and not is_garbled(text)


def row(label, value, gate=None, ok=None, note=None):
    return {"label": label, "value": value, "gate": gate, "ok": ok, "note": note}


def run(base=DEFAULT_BASE, runs=3, long=False, min_prefill=180.0,
        min_decode=20.0, timeout=600):
    started = time.monotonic()
    c = Client(base, timeout)
    c.model = c.model_id()
    p = probe_prefill(c)
    nl = probe_needle(c, 60)
    nl_long = probe_needle(c, 500) if long else None
    d = probe_decode(c)
    m_text, m_ok = probe_math(c)
    g_short, g_short_n, g_short_s = probe_tool(c, False, runs)
    g_long, g_long_n, g_long_s = probe_tool(c, True, runs)
    chain_ok, chain_note = probe_chain(c)
    perc_text, perc_ok = probe_perception(c)

    prefill_rows = [
        row("cold prompt speed", _f(p["pp"]) + " tok/s", ">= " + _f(min_prefill),
            p["pp"] >= min_prefill),
        row("prompt_ms / prompt tokens", _f(p["prompt_ms"], 0) + " ms / " + str(p["ptok"]) + " tok",
            None, None),
        row("cached_tokens (cold required)", str(p["cached"]), "== 0", p["cached"] == 0),
        row("long-doc prefill speed", _f(nl["pp"]) + " tok/s / " + str(nl["ptok"]) + " tok",
            None, None),
        row("long-doc needle", nl["answer"], "codename+port", nl["ok"],
            "expected " + nl["expected"]),
    ]
    if nl_long is not None:
        prefill_rows.append(row("22K-doc needle (--long)", nl_long["answer"],
                                "codename+port", nl_long["ok"],
                                "expected " + nl_long["expected"]))

    decode_rows = [
        row("generated speed", _f(d["dps"]) + " tok/s", ">= " + _f(min_decode),
            d["dps"] >= min_decode),
        row("tokens / decode_ms", str(d["ctok"]) + " / " + _f(d["decode_ms"], 0) + " ms",
            None, None),
        row("output coherence", d["preview"], "no garble", d["coherent"]),
    ]

    quality_rows = [
        row("short 2+2", repr(m_text[:40]), "== '4'", m_ok),
        row("glob tool (short system)", "%d/%d tool_calls" % (g_short, g_short_n),
            "%d/%d" % (runs, runs), g_short == runs),
        row("glob tool (long system)", "%d/%d tool_calls" % (g_long, g_long_n),
            "%d/%d" % (runs, runs), g_long == runs),
        row("two-step tool chain", "3 files -> 3", "chain ok", chain_ok, chain_note),
        row("disjoint-prose perception", repr(perc_text[:40]), "== 'grey'", perc_ok),
        row("garbled-text sentinel", "%d hits" % len(c.garble_hits), "== 0",
            len(c.garble_hits) == 0),
    ]
    if g_long_s and not g_long_s[0][0]:
        quality_rows[2]["note"] = "sample: " + repr(g_long_s[0][1])

    sections = [{"name": "PREFILL", "rows": prefill_rows},
                {"name": "DECODE", "rows": decode_rows},
                {"name": "QUALITY", "rows": quality_rows}]
    gates = [r["ok"] for s in sections for r in s["rows"] if r["ok"] is not None]
    ok = all(gates)
    failed = [s["name"] + "/" + r["label"] for s in sections for r in s["rows"]
              if r["ok"] is False]
    return {
        "ok": ok, "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": c.model, "base": base, "runs": runs, "long": long,
        "min_prefill": min_prefill, "min_decode": min_decode,
        "sections": sections,
        "failed": failed,
        "summary": "; ".join(failed) if failed else "all gates passed",
        "elapsed_s": time.monotonic() - started,
    }


def report(res):
    W = 96
    print("=" * W)
    print(" DSH serve regression  %s  model=%s  base=%s" % (res["time"], res["model"], res["base"]))
    print("=" * W)
    for sec in res["sections"]:
        print(sec["name"])
        for r in sec["rows"]:
            status = "PASS" if r["ok"] is True else ("FAIL" if r["ok"] is False else "info")
            print("  {:<31} {:<41} {:<14} {}".format(
                str(r["label"])[:31], str(r["value"])[:41],
                str(r["gate"] or "")[:14], status))
            if r.get("note"):
                print("      " + str(r["note"])[:92])
        print()
    print("-" * W)
    print("RESULT: %s   (elapsed %.1f s)" % ("PASS" if res["ok"] else "FAIL", res["elapsed_s"]))
    if res["failed"]:
        print("FAILED GATES: " + "; ".join(res["failed"]))
    print("=" * W)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--runs", type=int, default=3, help="repeats per tool probe (default 3)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the table")
    ap.add_argument("--long", action="store_true", help="also run the ~22K-token needle probe")
    ap.add_argument("--min-prefill", type=float, default=180.0)
    ap.add_argument("--min-decode", type=float, default=20.0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    if not server_ready(args.base):
        print("SKIP: no live Strata server with a loaded model at " + args.base)
        return 2
    try:
        res = run(args.base, runs=args.runs, long=args.long,
                  min_prefill=args.min_prefill, min_decode=args.min_decode,
                  timeout=args.timeout)
    except Exception as e:
        print("ERROR: %s: %s" % (type(e).__name__, e))
        return 1
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        report(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
