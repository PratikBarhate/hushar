#!/usr/bin/env python3
"""Re-derives every number in the wire test's `summary.md` from `raw/` and fails on any
disagreement.

    verify.py

A check that cannot find its input FAILS rather than passing quietly: "no data" and
"agrees" are different answers and only one of them is good news.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


RUN = os.environ.get("RUN", "config-sweep-gpu-g5-wire-10M-20M")
BENCH = _benchmark_data(HERE)
RESULTS = BENCH / "results" / RUN
RAW = RESULTS / "raw"
SIBLING = "config-sweep-gpu-g5-10M-20M"
UNLABEL = {"10M": "10m", "20M": "20m"}
UNWIRE = {"210 features": "features", "690-float tensor": "tensor"}
MARK = {"sustained": "ok", "partial": "partial", "failed": "no"}

CHECKS = [0]
FAILS = []


def ck(name, got, want, tol=None):
    CHECKS[0] += 1
    if want is None and got is None:
        return
    ok = (got == want) if tol is None else (
        got is not None and want is not None and abs(got - want) <= tol)
    if not ok:
        FAILS.append(f"{name}: report says {got!r}, raw says {want!r}")


def num(text):
    text = (text or "").strip().strip("`").rstrip("%")
    if text in ("--", ""):
        return None
    m = re.match(r"^([+-]?[\d.]+)", text)
    return float(m.group(1)) if m else None


def rows_of(text, fragment):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and fragment in line:
            out = []
            for body in lines[i + 2:]:
                if not body.startswith("|"):
                    break
                out.append([c.strip() for c in body.strip().strip("|").split("|")])
            return out
    return None


def main():
    report = RESULTS / "summary.md"
    if not report.exists():
        sys.exit(f"no {report}; run report.py first")
    text = report.read_text()
    cells = json.loads((RAW / "results.json").read_text())
    if not cells:
        sys.exit("raw/results.json is empty; nothing to verify against")
    status = json.loads((RAW / "run-status.json").read_text()) \
        if (RAW / "run-status.json").exists() else {}
    idx = {(c["size"], c["wire"], c["tps"]): c for c in cells}

    def res(c, k):
        return (c.get("result") or {}).get(k)

    def cw(c, m):
        return ((c.get("cloudwatch") or {}).get(m) or {}).get("p50_ms")

    def state(c):
        return (c.get("verdict") or {}).get("state")

    # ---------------------------------------------------------- the answer
    table = rows_of(text, "switching to tensor")
    if table is None:
        FAILS.append("answer table: not found")
    else:
        for r in table:
            size = UNLABEL.get(r[0])
            tps = int(r[1].split(" x ")[1])
            f, t = idx.get((size, "features", tps)), idx.get((size, "tensor", tps))
            tag = f"answer {r[0]} {tps}"
            if f is None or t is None:
                FAILS.append(f"{tag}: missing a wire ({f is None=}, {t is None=})")
                continue
            for col, c, who in ((2, f, "features"), (3, t, "tensor")):
                if state(c) == "sustained":
                    ck(f"{tag} {who} p50", num(r[col]), res(c, "p50"), 0.005)
                else:
                    # An unsustained cell is reported by achieved rate, not latency.
                    CHECKS[0] += 1
                    if not r[col].startswith("no "):
                        FAILS.append(f"{tag} {who}: did not sustain but the report shows "
                                     f"{r[col]!r}")
                    ck(f"{tag} {who} achieved", num(r[col].split()[-1].split('/')[0]),
                       res(c, "achieved"), 0.05)
            # The cost is the claim the whole run exists to make, so it is recomputed.
            # tensor MINUS features, the direction the report's prose uses. The two
            # disagreed once, which made the same number read as +0.21 and -0.21 in
            # adjacent lines; the sign is now part of the assertion.
            want = None
            if state(f) == "sustained" and state(t) == "sustained":
                want = res(t, "p50") - res(f, "p50")
            ck(f"{tag} cost of switching to tensor", num(r[4]), want, 0.01)
            # The Triton reference must match the sibling campaign, not be retyped.
            sib = json.loads(
                (BENCH / "results" / SIBLING / "raw" / "results.json").read_text())
            ref = next((( c.get("result") or {}).get("p50") for c in sib
                        if c.get("stack") == "triton" and c.get("size") == size
                        and c.get("tps") == tps and c.get("sustained")), None)
            ck(f"{tag} triton reference", num(r[5]), ref, 0.005)

    # ---------------------------------------------------------- the detail
    table = rows_of(text, "everything else")
    if table is None:
        FAILS.append("detail table: not found")
    else:
        ck("detail table covers every cell", len(table), len(cells))
        for r in table:
            size = UNLABEL.get(r[0])
            tps = int(r[1].split(" x ")[1])
            wire = UNWIRE.get(r[2])
            c = idx.get((size, wire, tps))
            tag = f"detail {r[0]} {tps} {r[2]}"
            if c is None:
                FAILS.append(f"{tag}: no raw cell")
                continue
            ck(f"{tag} id", r[12].strip("`"), c["id"])
            ck(f"{tag} verdict", r[3].split()[0], MARK.get(state(c), "?"))
            ck(f"{tag} achieved", num(r[3].split()[-1].split("/")[0]),
               res(c, "achieved"), 0.05)
            ck(f"{tag} p50", num(r[4]), res(c, "p50"), 0.005)
            ck(f"{tag} p95", num(r[5]), res(c, "p95"), 0.005)
            ck(f"{tag} p99", num(r[6]), res(c, "p99"), 0.005)
            ck(f"{tag} graph", num(r[7]), cw(c, "InferenceTime"), 0.0005)
            ck(f"{tag} vectorize", num(r[8]), cw(c, "VectorizationTime"), 0.0005)
            other = (None if res(c, "p50") is None or cw(c, "InferenceTime") is None
                     else res(c, "p50") - cw(c, "InferenceTime")
                     - (cw(c, "VectorizationTime") or 0.0))
            ck(f"{tag} everything else", num(r[9]), other, 0.002)
            if None not in (num(r[4]), num(r[7]), num(r[8]), num(r[9])):
                ck(f"{tag} parts sum to the median",
                   round(num(r[7]) + num(r[8]) + num(r[9]), 2),
                   round(num(r[4]), 2), 0.011)
            gpu = (c.get("gpu_utilization") or {}).get("gpu_util_pct")
            ck(f"{tag} gpu", num(r[10]), None if gpu is None else round(gpu, 1), 0.001)
            cpu = c.get("server_cpu_pct")
            ck(f"{tag} cpu", num(r[11]), None if cpu is None else round(cpu, 1), 0.001)

    # ---------------------------------------------------------- integrity
    for c in cells:
        tag = f"integrity {c['id']}"
        ck(f"{tag} rows", c["rows"], 100)
        ck(f"{tag} stack", c["stack"], "hushar")
        ck(f"{tag} wire is one of the two", c["wire"] in ("features", "tensor"), True)
        ck(f"{tag} mini batch is rows", c.get("mini_batch"), c["rows"])
        ck(f"{tag} clean slate", c.get("clean_slate_verified"), True)
        ck(f"{tag} provider is TensorRT",
           any("TensorRT(device=0" in b for b in (c.get("banner") or [])), True)
        ck(f"{tag} has GPU samples", bool(c.get("gpu_utilization")), True)
        ck(f"{tag} has server percentiles",
           bool((c.get("cloudwatch") or {}).get("InferenceTime")), True)
        # The banner must show the wire the cell claims, which is the one assertion that
        # stops a mislabelled arm from being the finding.
        want = ("<- [features] transformed and concatenated" if c["wire"] == "tensor"
                else "<- 210 features transformed and concatenated")
        ck(f"{tag} banner shows the {c['wire']} wire",
           any(want in b for b in (c.get("banner") or [])), True)
        slices = c.get("slices") or []
        ck(f"{tag} slice count", len(slices), c.get("segments"))
        held = [s for s in slices
                if s["shed"] == 0 and s["failed"] == 0
                and s["achieved"] is not None and s["achieved"] >= 0.98 * c["tps"]]
        want_state = "sustained" if (slices and len(held) == len(slices)
                                     and res(c, "shed") == 0 and res(c, "failed") == 0
                                     and res(c, "achieved") >= 0.98 * res(c, "offered")) \
            else ("partial" if held else "failed")
        ck(f"{tag} verdict rule", state(c), want_state)

    # Both wires must share the parallelism, or the subtraction is not one variable.
    for size in {c["size"] for c in cells}:
        pars = {c["parallelism"] for c in cells if c["size"] == size}
        ck(f"{size}: one parallelism across both wires", len(pars), 1)
        mbs = {c["mini_batch"] for c in cells if c["size"] == size}
        ck(f"{size}: one mini batch across both wires", len(mbs), 1)
        md5s = {c["model_md5"] for c in cells if c["size"] == size}
        ck(f"{size}: one graph across both wires", len(md5s), 1)

    if status:
        ck("status cell count", status.get("main", {}).get("cells"), len(cells))

    print(f"{CHECKS[0]} checks, {len(FAILS)} failure(s)")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
