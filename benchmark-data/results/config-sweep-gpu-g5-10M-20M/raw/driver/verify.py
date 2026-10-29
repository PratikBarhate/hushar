#!/usr/bin/env python3
"""Re-derives every number in `summary.md` from `raw/` and fails if one disagrees.

    verify.py

This exists because a report is generated once and read many times, and the failure mode
that matters is not a crash -- it is a figure that was true of an earlier run and is
quietly false of this one. So every cell of every table in the generated file is parsed
back out and recomputed from the raw record it claims to describe.

A check that cannot find its input FAILS rather than passing quietly. "No data" and
"agrees" are different answers and only one of them is good news.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _benchmark_data(start):
    """The `benchmark-data` directory, found by walking up rather than by counting `..`.

    This file lives in two places: the working copy under `benchmark-data/private/<run>/
    driver`, and the archived copy under `benchmark-data/results/<run>/raw/driver`. Those
    are different depths, so a fixed number of `.parent`s is correct in one and silently
    wrong in the other -- which is exactly what happened, and the archived copy could not
    find the tree it was archived beside.
    """
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")
RUN = os.environ.get("RUN", "config-sweep-gpu-g5-10M-20M")
BENCH = _benchmark_data(HERE)
RESULTS = BENCH / "results" / RUN
RAW = RESULTS / "raw"
BASELINE = (BENCH / "old-results" / "config-sweep-neuron-trn1-10M-50M"
            / "raw" / "results.json")
LABEL = {"10m": "10M", "20m": "20M"}
UNLABEL = {v: k for k, v in LABEL.items()}
STACK = {"hushar": "hushar", "Triton": "triton"}

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
    text = (text or "").strip().strip("`")
    if text in ("--", "", "did not hold"):
        return None
    m = re.match(r"^(-?[\d.]+)", text)
    return float(m.group(1)) if m else None


def rows_of(text, header_fragment):
    """The data rows of the first table whose header contains `header_fragment`."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and header_fragment in line:
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
    probes = json.loads((RAW / "probe-results.json").read_text()) \
        if (RAW / "probe-results.json").exists() else []
    status = json.loads((RAW / "run-status.json").read_text()) \
        if (RAW / "run-status.json").exists() else {}
    if not cells:
        sys.exit("raw/results.json is empty; nothing to verify against")

    idx = {(c["size"], c["stack"], c["rows"], c["tps"]): c for c in cells}

    def res(c, key):
        return (c.get("result") or {}).get(key)

    def cw(c, metric, key="p50_ms"):
        return ((c.get("cloudwatch") or {}).get(metric) or {}).get(key)

    def mark(c):
        return {"sustained": "ok", "partial": "partial",
                "failed": "no"}.get((c.get("verdict") or {}).get("state"), "?")

    # ---------------------------------------------------------- answer table
    table = rows_of(text, "lower median")
    if table is None:
        FAILS.append("answer table: not found in the report")
    else:
        ck("answer table row count", len(table) > 0, True)
        for r in table:
            size = UNLABEL.get(r[0])
            rows, tps = (int(x) for x in r[1].split(" x "))
            h = idx.get((size, "hushar", rows, tps))
            t = idx.get((size, "triton", rows, tps))
            tag = f"answer {r[0]} {r[1]}"
            if h is None and t is None:
                FAILS.append(f"{tag}: no raw cell for either stack")
                continue
            ck(f"{tag} hushar p50", num(r[2]), res(h, "p50") if h else None, 0.005)
            ck(f"{tag} triton p50", num(r[3]), res(t, "p50") if t else None, 0.005)
            ck(f"{tag} hushar p99", num(r[5]), res(h, "p99") if h else None, 0.005)
            ck(f"{tag} triton p99", num(r[6]), res(t, "p99") if t else None, 0.005)
            # The table carries no verdict columns, so the verdict is asserted from the
            # raw record. "not measured" must mean the cell is genuinely ABSENT: the
            # phrase must not be able to hide a measured failure.
            for who, c, col in (("hushar", h, 2), ("triton", t, 3)):
                CHECKS[0] += 1
                if r[col] == "not measured":
                    if c is not None:
                        FAILS.append(f"{tag}: {who} is shown as 'not measured' but a raw "
                                     f"cell exists for it ({(c.get('verdict') or {}).get('state')})")
                elif c is None or (c.get("verdict") or {}).get("state") != "sustained":
                    FAILS.append(f"{tag}: {who} carries a latency in the comparison "
                                 f"table but did not sustain its rate")
            # The claim the headline rests on: WHICH stack is faster and by how much.
            # Both halves are recomputed, because the report's first draft printed a
            # ratio whose direction its own prose then reversed.
            want_who, want_by = None, None
            if h and t and res(h, "p50") and res(t, "p50") \
                    and h["verdict"]["state"] == "sustained" \
                    and t["verdict"]["state"] == "sustained":
                if res(t, "p50") <= res(h, "p50"):
                    want_who, want_by = "Triton", res(h, "p50") / res(t, "p50")
                else:
                    want_who, want_by = "hushar", res(t, "p50") / res(h, "p50")
            cell_text = r[4].strip()
            CHECKS[0] += 1
            if want_who is None:
                if cell_text not in ("--", ""):
                    FAILS.append(f"{tag} lower median: report says {cell_text!r} but "
                                 f"not both cells sustained")
            else:
                if not cell_text.startswith(want_who):
                    FAILS.append(f"{tag} lower median: report credits {cell_text!r}, "
                                 f"raw says {want_who}")
                ck(f"{tag} lower median ratio", num(cell_text.split()[-1]),
                   want_by, 0.02)

    # ---------------------------------------------------------- every cell
    table = rows_of(text, "srv p50")
    if table is None:
        FAILS.append("per-cell table: not found in the report")
    else:
        ck("per-cell table covers every cell", len(table), len(cells))
        for r in table:
            size = UNLABEL.get(r[0])
            rows, tps = (int(x) for x in r[1].split(" x "))
            stack = STACK.get(r[2])
            c = idx.get((size, stack, rows, tps))
            tag = f"cell {r[0]} {r[1]} {r[2]}"
            if c is None:
                FAILS.append(f"{tag}: no raw cell")
                continue
            marked = r[13].rstrip().endswith("*")
            ck(f"{tag} id", r[13].rstrip().rstrip("*").strip().strip("`"), c["id"])
            ck(f"{tag} marked as measured elsewhere", marked, c.get("run") != RUN)
            ck(f"{tag} layout", r[3], f"{c['parallelism_kind']} {c['parallelism']}")
            ck(f"{tag} p50", num(r[5]), res(c, "p50"), 0.005)
            ck(f"{tag} p95", num(r[6]), res(c, "p95"), 0.005)
            ck(f"{tag} p99", num(r[7]), res(c, "p99"), 0.005)
            ck(f"{tag} srv p50", num(r[8]), cw(c, "InferenceTime"), 0.0005)
            # The percentage columns are compared against the ROUNDED value, because that
            # is what the report prints. Comparing against the unrounded one with a
            # half-ulp tolerance fails on exact halves -- 31.95 formats to "31.9" and then
            # sits exactly 0.05 away from its own source.
            gpu = (c.get("gpu_utilization") or {}).get("gpu_util_pct")
            ck(f"{tag} gpu util", num(r[9]),
               None if gpu is None else round(gpu, 1), 0.001)
            power = (c.get("gpu_utilization") or {}).get("power_w")
            ck(f"{tag} power", num(r[10]),
               None if power is None else round(power), 0.6)
            cpu = c.get("server_cpu_pct")
            ck(f"{tag} cpu", num(r[11]),
               None if cpu is None else round(cpu, 1), 0.001)
            ck(f"{tag} shed", num(r[12]), float(res(c, "shed") or 0), 0.5)
            ck(f"{tag} verdict word", r[4].split()[0], mark(c))
            # And the rate beside the word. Checking only the word left the verifier
            # blind to an achieved figure that had been edited -- including one ABOVE the
            # offered rate, which is impossible and still passed.
            got = r[4].split()
            CHECKS[0] += 1
            if len(got) < 2 or "/" not in got[1]:
                FAILS.append(f"{tag} result: {r[4]!r} carries no achieved/offered")
            else:
                ach_text, off_text = got[1].split("/", 1)
                ck(f"{tag} achieved", num(ach_text), res(c, "achieved"), 0.05)
                ck(f"{tag} offered", num(off_text), float(res(c, "offered")), 0.5)
                ck(f"{tag} offered is the cell's tps", num(off_text), float(c["tps"]), 0.5)
                CHECKS[0] += 1
                if num(ach_text) is not None and num(ach_text) > float(c["tps"]) * 1.001:
                    FAILS.append(f"{tag}: achieved {ach_text} exceeds the offered "
                                 f"{c['tps']}, which an open-loop client cannot do")

    foreign_runs = {c.get("run") for c in cells if c.get("run") != RUN}
    for run in sorted(foreign_runs):
        CHECKS[0] += 1
        if run not in text:
            FAILS.append(f"provenance: cells measured by {run!r} are present but that "
                         f"run is never named in the report")

    # ---------------------------------------------------------- probes
    table = rows_of(text, "chosen")
    if table is None:
        FAILS.append("probe table: not found in the report")
    else:
        ck("probe table covers every probe", len(table), len(probes))
        pidx = {(p["size"], p["stack"], p["parallelism"]): p for p in probes}
        chosen = {}
        for r in table:
            size, stack = UNLABEL.get(r[0]), STACK.get(r[1])
            par = int(r[3].split()[-1])
            p = pidx.get((size, stack, par))
            tag = f"probe {r[0]} {r[1]} p{par}"
            if p is None:
                FAILS.append(f"{tag}: no raw probe")
                continue
            ck(f"{tag} p50", num(r[5]), res(p, "p50"), 0.005)
            ck(f"{tag} p95", num(r[6]), res(p, "p95"), 0.005)
            if "<-" in r[7]:
                chosen.setdefault((size, stack), []).append(par)
        # The selection rule, re-applied. A report that marked a parallelism as chosen
        # and then measured a different one is the failure this catches.
        for (size, stack), picked in chosen.items():
            ck(f"selection {size} {stack} marked once", len(picked), 1)
            used = sorted({c["parallelism"] for c in cells
                           if c["size"] == size and c["stack"] == stack})
            ck(f"selection {size} {stack} is what ran", used, [picked[0]])
            mine = [p for p in probes if p["size"] == size and p["stack"] == stack
                    and res(p, "p50") is not None]
            held = [p for p in mine if p.get("sustained")]
            if held:
                want = min(held, key=lambda p: res(p, "p50"))["parallelism"]
                ck(f"selection {size} {stack} is lowest sustained p50", picked[0], want)

    # ---------------------------------------------------------- decomposition
    table = rows_of(text, "everything else")
    if table is None:
        FAILS.append("decomposition table: not found in the report")
    else:
        for r in table:
            size = UNLABEL.get(r[0])
            rows, tps = (int(x) for x in r[1].split(" x "))
            c = idx.get((size, "hushar", rows, tps))
            tag = f"decompose {r[0]} {r[1]}"
            if c is None:
                FAILS.append(f"{tag}: no raw hushar cell")
                continue
            p50, graph, vec = res(c, "p50"), cw(c, "InferenceTime"), \
                cw(c, "VectorizationTime")
            ck(f"{tag} p50", num(r[2]), p50, 0.005)
            ck(f"{tag} graph", num(r[3]), graph, 0.0005)
            ck(f"{tag} vectorize", num(r[4]), vec, 0.0005)
            # The remainder is the report's load-bearing claim, so it is recomputed and
            # also checked to add back up to the median.
            other = (p50 - graph - (vec or 0.0)) if (p50 and graph) else None
            ck(f"{tag} everything else", num(r[5]), other, 0.002)
            if None not in (num(r[2]), num(r[3]), num(r[4]), num(r[5])):
                ck(f"{tag} parts sum to the median",
                   round(num(r[3]) + num(r[4]) + num(r[5]), 2), round(num(r[2]), 2), 0.01)
            share = (graph / p50) if (graph and p50) else None
            ck(f"{tag} graph share", num(r[6]),
               round(share * 100) if share else None, 0.6)
            per_row = (vec * 1000 / rows) if vec is not None else None
            ck(f"{tag} per row us", num(r[7]), per_row, 0.01)

    # ---------------------------------------------------------- scaling
    table = rows_of(text, "graph ratio")
    if table is None:
        FAILS.append("scaling table: not found in the report")
    else:
        for r in table:
            tps = int(r[0].split(" x ")[1])
            tag = f"scaling 100 x {tps}"
            pairs = [("hushar", 1, 2, 3), ("triton", 4, 5, 6)]
            for stack, c10, c20, cr in pairs:
                a = idx.get(("10m", stack, 100, tps))
                b = idx.get(("20m", stack, 100, tps))
                ck(f"{tag} {stack} 10M", num(r[c10]),
                   res(a, "p50") if a else None, 0.005)
                ck(f"{tag} {stack} 20M", num(r[c20]),
                   res(b, "p50") if b else None, 0.005)
                want = None
                if a and b and res(a, "p50") and res(b, "p50") \
                        and a["verdict"]["state"] == "sustained" \
                        and b["verdict"]["state"] == "sustained":
                    want = res(b, "p50") / res(a, "p50")
                ck(f"{tag} {stack} ratio", num(r[cr]), want, 0.02)
            # The graph columns are hushar's CloudWatch InferenceTime, which is the
            # column the section tells the reader to trust, so it is checked hardest.
            a = idx.get(("10m", "hushar", 100, tps))
            b = idx.get(("20m", "hushar", 100, tps))
            g10 = cw(a, "InferenceTime") if a else None
            g20 = cw(b, "InferenceTime") if b else None
            ck(f"{tag} graph 10M", num(r[7]), g10, 0.0005)
            ck(f"{tag} graph 20M", num(r[8]), g20, 0.0005)
            want = None
            if g10 and g20 and a["verdict"]["state"] == "sustained" \
                    and b["verdict"]["state"] == "sustained":
                want = g20 / g10
            ck(f"{tag} graph ratio", num(r[9]), want, 0.02)

    # ---------------------------------------------------------- Trainium
    table = rows_of(text, "trn1 p50")
    if table is None:
        # Absent on purpose: the Trainium figures were withdrawn as stale and the section
        # is behind INCLUDE_TRAINIUM. What IS checked is that the report then makes no
        # Trainium comparison anywhere -- a withdrawn table must not leave a live claim
        # behind it.
        CHECKS[0] += 1
        for claim in ("against trn1", "trn1 p50", "Against Trainium"):
            if claim in text:
                FAILS.append(f"Trainium table is absent but the report still says "
                             f"{claim!r}")
    elif not BASELINE.exists():
        FAILS.append(f"Trainium table present but {BASELINE} is missing")
    else:
        base = json.loads(BASELINE.read_text())
        best = {}
        for c in base:
            p50 = (c.get("result") or {}).get("p50")
            if p50 is None:
                continue
            key = (c.get("size"), c.get("rows"), c.get("tps"))
            rec = (bool(c.get("sustained")), -p50, p50)
            if key not in best or rec[:2] > best[key][:2]:
                best[key] = rec
        for r in table:
            size = UNLABEL.get(r[0])
            rows, tps = (int(x) for x in r[1].split(" x "))
            tag = f"trainium {r[0]} {r[1]}"
            b = best.get((size, rows, tps))
            if b is None:
                FAILS.append(f"{tag}: no baseline cell")
                continue
            ck(f"{tag} trn1 p50", num(r[2]), b[2] if b[0] else None, 0.005)
            h = idx.get((size, "hushar", rows, tps))
            ck(f"{tag} hushar p50", num(r[3]), res(h, "p50") if h else None, 0.005)
            # Direction and magnitude, both recomputed.
            want = None
            if b[0] and h and res(h, "p50") and h["verdict"]["state"] == "sustained":
                want = res(h, "p50") / b[2]
            cell_text = r[5].strip()
            CHECKS[0] += 1
            if want is None:
                if cell_text not in ("--", ""):
                    FAILS.append(f"{tag}: report says {cell_text!r} with no comparable "
                                 f"pair")
            elif want < 0.98:
                if "faster" not in cell_text:
                    FAILS.append(f"{tag}: hushar is faster ({want:.3f}) but the report "
                                 f"says {cell_text!r}")
                ck(f"{tag} magnitude", num(cell_text), 1 / want, 0.02)
            elif want > 1.02:
                if "slower" not in cell_text:
                    FAILS.append(f"{tag}: hushar is slower ({want:.3f}) but the report "
                                 f"says {cell_text!r}")
                ck(f"{tag} magnitude", num(cell_text), want, 0.02)
            else:
                if cell_text != "level":
                    FAILS.append(f"{tag}: within 2% ({want:.3f}) but the report says "
                                 f"{cell_text!r}")

    # ---------------------------------------------------------- provenance
    seen = {}
    if (RAW / "server").is_dir():
        for p in sorted((RAW / "server").glob("*-stackinfo.txt")):
            for line in p.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    seen.setdefault(k.strip(), set()).add(v.strip())
    if not seen:
        FAILS.append("provenance: no *-stackinfo.txt under raw/server")
    else:
        for key in ("gpu", "nvidia_driver", "image_tensorrt", "onnxruntime",
                    "triton_image", "kernel"):
            vals = sorted(seen.get(key) or [])
            CHECKS[0] += 1
            if not vals:
                FAILS.append(f"provenance: {key} was never recorded")
            elif not all(v in text for v in vals):
                FAILS.append(f"provenance: {key}={vals} not in the report")
        # The claim that both stacks used one TensorRT is structural, not stylistic.
        ck("one TensorRT across hosts", len(seen.get("image_tensorrt") or []), 1)
        ck("one ONNX Runtime across hosts", len(seen.get("onnxruntime") or []), 1)
        ck("one Triton image across hosts", len(seen.get("triton_image") or []), 1)

    # ---------------------------------------------------------- integrity
    for c in cells:
        tag = f"integrity {c['id']}"
        ck(f"{tag} md5 recorded", bool(c.get("model_md5")), True)
        ck(f"{tag} clean slate", c.get("clean_slate_verified"), True)
        ck(f"{tag} rows", c["rows"], 100)
        if c["stack"] == "hushar":
            ck(f"{tag} provider is TensorRT",
               any("TensorRT(device=0" in b for b in (c.get("banner") or [])), True)
            ck(f"{tag} feature wire", c["wire"], "features")
            ck(f"{tag} mini batch is rows", c.get("mini_batch"), c["rows"])
            ck(f"{tag} has GPU samples", bool(c.get("gpu_utilization")), True)
            ck(f"{tag} has server percentiles",
               bool((c.get("cloudwatch") or {}).get("InferenceTime")), True)
        else:
            ck(f"{tag} tensorrt backend", c.get("backend"), "tensorrt")
            ck(f"{tag} tensorrt_plan", c.get("platform"), "tensorrt_plan")
            ck(f"{tag} no dynamic batching", c.get("dynamic_batching"), None)
            ck(f"{tag} max_batch 0", c.get("max_batch"), 0)
        slices = c.get("slices") or []
        ck(f"{tag} slice count", len(slices), c.get("segments"))
        # The verdict is the report's load column, so the rule is re-applied here.
        held = [s for s in slices
                if s["shed"] == 0 and s["failed"] == 0
                and s["achieved"] is not None and s["achieved"] >= 0.98 * c["tps"]]
        state = (c.get("verdict") or {}).get("state")
        want = "sustained" if (slices and len(held) == len(slices)
                               and res(c, "shed") == 0 and res(c, "failed") == 0
                               and res(c, "achieved") >= 0.98 * res(c, "offered")) \
            else ("partial" if held else "failed")
        ck(f"{tag} verdict rule", state, want)

    if status:
        ck("status cell count", status.get("main", {}).get("cells"), len(cells))
        ck("status sustained count", status.get("main", {}).get("sustained"),
           sum(bool(c.get("sustained")) for c in cells))

    print(f"{CHECKS[0]} checks, {len(FAILS)} failure(s)")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
