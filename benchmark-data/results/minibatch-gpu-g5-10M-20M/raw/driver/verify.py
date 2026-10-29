#!/usr/bin/env python3
"""Re-derives every number in `summary.md` from `raw/` and fails if one disagrees.

    verify.py

This exists because a report is generated once and read many times, and the failure mode
that matters is not a crash -- it is a figure that was true of an earlier run and is
quietly false of this one. So every cell of every table in the generated file is parsed
back out and recomputed from the raw record it claims to describe.

A check that cannot find its input FAILS rather than passing quietly. "No data" and
"agrees" are different answers and only one of them is good news.

WHAT THIS RUN NEEDS CHECKED THAT A SPEED COMPARISON DOES NOT.

  * **The split is real.** A cell that silently ran unsplit would look exactly like a
    split that bought nothing -- which is this run's expected answer, so it is the one way
    the report can lie and be believed. Every hushar cell's `mini_batch` is therefore
    checked against its own id, its engine cache path, its engine shape, its piece count
    AND the server banner it recorded at startup.
  * **Direction and magnitude, separately.** `against unsplit` carries words
    (`1.22x slower`). A verifier that only checks the number cannot catch a sign flip,
    and a sign flip is how an earlier report in this series said "0.6x faster" about the
    slower stack. Each is recomputed on its own.
  * **The baseline is the UNSPLIT cell of the same load.** Not the largest split, not a
    cell from another run, not another row count. A gain divided by the wrong baseline is
    arithmetically fine and completely wrong.
  * **Nothing is imported.** `config-sweep-gpu-g5-10M-20M` measured these loads over a
    300s window; this run uses 900s. Every cell must name THIS run, so a pasted-in figure
    from the prior campaign fails rather than blending two windows.
  * **A cell that did not hold its rate must not appear in any comparison**, and the
    `sustained` verdict is re-applied from the slices rather than trusted.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    """The `benchmark-data` directory, found by walking up rather than by counting `..`."""
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


RUN = os.environ.get("RUN", "minibatch-gpu-g5-10M-20M")
BENCH = _benchmark_data(HERE)
RESULTS = BENCH / "results" / RUN
RAW = RESULTS / "raw"
PRIOR_GPU = "config-sweep-gpu-g5-10M-20M"
LABEL = {"10m": "10M", "20m": "20M", "50m": "50M"}
UNLABEL = {v: k for k, v in LABEL.items()}
TIE_PCT = 8.4

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


def ck_true(name, cond, why=""):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(f"{name}: {why}")


def num(text):
    text = (text or "").strip().strip("`").strip("*")
    if text in ("--", "", "n/a", "n/a (split)", "baseline", "level"):
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


# ---------------------------------------------------------------- the raw truth

def load_cells():
    cells = json.loads((RAW / "results.json").read_text())
    if not cells:
        sys.exit("results.json is empty")
    return cells


def key_of(c):
    return (c["size"], c["stack"], c["rows"], c["tps"], c.get("mini_batch"))


def sustained_from_slices(c):
    """Re-apply the rule rather than trusting the recorded verdict.

    Sustained means EVERY slice of the measured window served at least 98% of the offered
    rate with nothing shed and nothing failed. A cell that held some slices is `partial`,
    which is its own state and neither a pass nor a fail.
    """
    slices = c.get("slices") or []
    if not slices:
        return None
    offered = (c.get("result") or {}).get("offered")
    if not offered:
        return None
    held = [s for s in slices
            if s.get("achieved", 0) >= 0.98 * offered
            and not s.get("shed") and not s.get("failed")]
    if len(held) == len(slices):
        return "sustained"
    return "partial" if held else "failed"


def expected_label(base_p50, p50):
    """The same words report.py must have produced, recomputed independently."""
    if base_p50 in (None, 0) or p50 in (None, 0):
        return "--"
    change = (p50 - base_p50) / base_p50 * 100.0
    if abs(change) < TIE_PCT:
        return "level"
    if p50 < base_p50:
        return f"{base_p50 / p50:.2f}x faster"
    return f"{p50 / base_p50:.2f}x slower"


def split_label(text):
    """`1.22x slower` -> (1.22, 'slower'). Direction and magnitude, separately."""
    text = text.strip().strip("*")
    if text in ("level", "baseline", "--"):
        return None, text
    m = re.match(r"^([\d.]+)x (faster|slower)$", text)
    if not m:
        return None, text
    return float(m.group(1)), m.group(2)


# ---------------------------------------------------------------- checks

def check_structure(cells):
    """What must be true of every cell for any table built from it to mean anything."""
    for c in cells:
        cid = c["id"]
        ck(f"{cid} run", c.get("run"), RUN)
        state = sustained_from_slices(c)
        if state is not None:
            ck(f"{cid} verdict re-derived from slices",
               (c.get("verdict") or {}).get("state"), state)
        if c["stack"] == "hushar":
            mb = c.get("mini_batch")
            ck_true(f"{cid} declares a mini batch", mb is not None,
                    "a hushar cell with no mini_batch cannot be placed in any group")
            if mb is None:
                continue
            # The id, the cache, the shape and the piece count must all agree with it.
            ck_true(f"{cid} id names its mini batch", f"-mb{mb}-" in cid,
                    f"id does not contain -mb{mb}-")
            ck(f"{cid} engine cache", c.get("engine_cache"),
               f"/root/trt-cache/{c['size']}/mb{mb}")
            ck(f"{cid} engine shape", tuple(c.get("engine_shape") or []), (mb, 690))
            ck(f"{cid} pieces", c.get("pieces"), c["rows"] // mb)
            ck(f"{cid} split flag", bool(c.get("split")), c["rows"] // mb > 1)
            ck_true(f"{cid} mini batch divides rows", c["rows"] % mb == 0,
                    f"mini batch {mb} does not divide {c['rows']} rows, so the cell "
                    f"presented the engine two shapes")
            banner = "\n".join(c.get("banner") or [])
            ck_true(f"{cid} banner named the TensorRT provider",
                    "TensorRT(device=0" in banner,
                    "a graph that fell back to the CPU provider still produces a "
                    "latency, and that latency is a lie about a GPU")
            ck_true(f"{cid} banner echoed its own engine cache",
                    f"trt_engine_cache_path={c['engine_cache']}" in banner,
                    "the cell may have borrowed another shape's engine")
            # THE one that matters: the server's own words about the split.
            ck_true(f"{cid} banner reported mini batches of {mb}",
                    f"mini batches of {mb}, scored together" in banner,
                    "the server did not confirm the split this cell claims to measure")
            ck(f"{cid} onnxruntime", c.get("onnxruntime"), c.get("ort_version"))
        else:
            ck(f"{cid} backend", c.get("backend"), "tensorrt")
            ck(f"{cid} max_batch", c.get("max_batch"), 0)
            ck_true(f"{cid} has no mini batch", c.get("mini_batch") is None,
                    "Triton cannot split; a mini batch on a Triton cell is a bug")

    # One ONNX Runtime across every hushar cell, and one pinned parallelism per size.
    orts = {c.get("ort_version") for c in cells if c["stack"] == "hushar"}
    ck_true("one ONNX Runtime across the run", len(orts) == 1,
            f"hushar cells report {sorted(orts)}")
    for size in sorted({c["size"] for c in cells}):
        for stack in ("hushar", "triton"):
            pars = {c["parallelism"] for c in cells
                    if c["size"] == size and c["stack"] == stack}
            if pars:
                ck_true(f"{size} {stack} parallelism pinned", len(pars) == 1,
                        f"parallelism varies across cells ({sorted(pars)}), so the "
                        f"mini-batch column is not a one-variable subtraction")


def check_gain_table(text, cells):
    rows = rows_of(text, "against unsplit")
    ck_true("gain table present", rows is not None, "no table with an 'against unsplit' "
                                                    "column was found in the report")
    if not rows:
        return
    by_key = {key_of(c): c for c in cells}
    seen = 0
    for r in rows:
        (model, load, mb_s, pieces_s, p50_s, p95_s, p99_s,
         label, gpu_s, cpu_s, cid_s) = r[:11]
        size = UNLABEL[model]
        rws, tps = (int(x) for x in load.split(" x "))
        mb = int(mb_s)
        c = by_key.get((size, "hushar", rws, tps, mb))
        ck_true(f"{cid_s} exists in raw", c is not None,
                "the report names a cell that is not in results.json -- a latency was "
                "invented for a load that was never measured")
        if c is None:
            continue
        seen += 1
        res = c["result"]
        ck(f"{c['id']} in gain table is sustained",
           (c.get("verdict") or {}).get("state"), "sustained")
        ck(f"{c['id']} p50", num(p50_s), res["p50"], tol=0.005)
        ck(f"{c['id']} p95", num(p95_s), res["p95"], tol=0.005)
        ck(f"{c['id']} p99", num(p99_s), res["p99"], tol=0.005)
        ck(f"{c['id']} pieces", int(pieces_s), c["pieces"])
        ck(f"{c['id']} cell id", cid_s.strip("`"), c["id"])
        ck(f"{c['id']} gpu", num(gpu_s),
           (c.get("gpu_utilization") or {}).get("gpu_util_pct"), tol=0.05)
        ck(f"{c['id']} cpu", num(cpu_s), c.get("server_cpu_pct"), tol=0.05)

        base = by_key.get((size, "hushar", rws, tps, rws))
        if mb == rws:
            ck(f"{c['id']} baseline row is labelled", label, "baseline")
            continue
        ck_true(f"{c['id']} has a baseline", base is not None,
                f"no unsplit cell for {size} {rws}x{tps} to divide by")
        if base is None:
            continue
        # DIRECTION and MAGNITUDE, independently.
        want = expected_label(base["result"]["p50"], res["p50"])
        got_mag, got_dir = split_label(label)
        want_mag, want_dir = split_label(want)
        ck(f"{c['id']} gain direction", got_dir, want_dir)
        ck(f"{c['id']} gain magnitude", got_mag, want_mag, tol=0.011)
    ck_true("gain table is not empty", seen > 0, "no row resolved to a raw cell")

    # And the converse: nothing unsustained leaked in.
    ids = {r[10].strip("`") for r in rows}
    for c in cells:
        if c["id"] in ids:
            ck_true(f"{c['id']} sustained to be in the gain table",
                    (c.get("verdict") or {}).get("state") == "sustained",
                    "an unsustained cell is in a comparison table; its median is queue "
                    "depth, not a latency")


def check_answer_table(text, cells):
    """The summary row per load: the best split must be the fastest SUSTAINED split."""
    rows = rows_of(text, "best split")
    ck_true("answer table present", rows is not None, "no 'best split' table found")
    if not rows:
        return
    groups = {}
    for c in cells:
        if c["stack"] == "hushar" and c.get("mini_batch") is not None:
            groups.setdefault((c["size"], c["rows"], c["tps"]), {})[c["mini_batch"]] = c
    for r in rows:
        model, load, rps_s, base_s, best_s, best_p50_s, verdict = r[:7]
        size = UNLABEL[model]
        rws, tps = (int(x) for x in load.split(" x "))
        ck(f"{size} {rws}x{tps} rows/s", int(rps_s.replace(",", "")), rws * tps)
        group = groups.get((size, rws, tps), {})
        base = group.get(rws)
        if base is None or (base.get("verdict") or {}).get("state") != "sustained":
            continue
        ck(f"{size} {rws}x{tps} unsplit p50", num(base_s), base["result"]["p50"],
           tol=0.005)
        splits = [c for mb, c in group.items()
                  if mb != rws and (c.get("verdict") or {}).get("state") == "sustained"]
        if not splits:
            ck_true(f"{size} {rws}x{tps} says no split sustained",
                    "sustained" in verdict or "none" in verdict,
                    f"no split sustained the rate but the verdict reads {verdict!r}")
            continue
        want = min(splits, key=lambda c: c["result"]["p50"])
        ck(f"{size} {rws}x{tps} best split p50", num(best_p50_s),
           want["result"]["p50"], tol=0.005)
        ck_true(f"{size} {rws}x{tps} best split names mini batch "
                f"{want['mini_batch']}",
                str(want["mini_batch"]) == best_s.split(" ")[0],
                f"report names {best_s!r}, raw's fastest sustained split is mini batch "
                f"{want['mini_batch']}")
        got_mag, got_dir = split_label(verdict)
        want_mag, want_dir = split_label(
            expected_label(base["result"]["p50"], want["result"]["p50"]))
        ck(f"{size} {rws}x{tps} verdict direction", got_dir, want_dir)
        ck(f"{size} {rws}x{tps} verdict magnitude", got_mag, want_mag, tol=0.011)


def check_decomposition(text, cells):
    rows = rows_of(text, "everything else")
    ck_true("decomposition table present", rows is not None,
            "no 'everything else' table found")
    if not rows:
        return
    by_key = {key_of(c): c for c in cells}
    for r in rows:
        model, load, mb_s, pieces_s, p50_s, graph_s, vec_s, rest_s, share_s = r[:9]
        size = UNLABEL[model]
        rws, tps = (int(x) for x in load.split(" x "))
        c = by_key.get((size, "hushar", rws, tps, int(mb_s)))
        ck_true(f"decomposition row {model} {load} mb{mb_s} exists", c is not None,
                "names a cell not in results.json")
        if c is None:
            continue
        cw = c.get("cloudwatch") or {}
        graph = (cw.get("InferenceTime") or {}).get("p50_ms")
        vec = (cw.get("VectorizationTime") or {}).get("p50_ms")
        ck(f"{c['id']} graph", num(graph_s), graph, tol=0.0005)
        ck(f"{c['id']} vectorize", num(vec_s), vec, tol=0.0005)
        if c.get("split"):
            # A subtraction against a maximum-over-pieces figure is not a remainder, and
            # the report must say so rather than print a number.
            ck_true(f"{c['id']} split row leaves 'everything else' blank",
                    "split" in rest_s,
                    f"a split cell printed a remainder ({rest_s!r}) computed from a "
                    f"stage figure that is the MAXIMUM over its pieces")
            ck_true(f"{c['id']} split share is marked as the max piece",
                    "max piece" in share_s,
                    "a split cell's graph share must be marked, or it reads as the "
                    "request's total device time")
        else:
            want = c["result"]["p50"] - graph - vec
            ck(f"{c['id']} everything else", num(rest_s), want, tol=0.0015)


def check_packaging(text, cells):
    rows = rows_of(text, "per candidate, against the other")
    if rows is None:
        return                        # only present when both packagings sustained
    unsplit = {}
    for c in cells:
        if (c["stack"] == "hushar" and not c.get("split")
                and (c.get("verdict") or {}).get("state") == "sustained"):
            unsplit[(c["size"], c["rows"])] = c
    for r in rows:
        model, rps_s, pack_s, tps_s, p50_s, per_s, graph_s, gpu_s, cpu_s, label = r[:10]
        size = UNLABEL[model]
        rws = int(pack_s.split(" ")[0])
        c = unsplit.get((size, rws))
        ck_true(f"packaging row {model} {pack_s} exists", c is not None,
                "names an unsplit sustained cell that is not in results.json")
        if c is None:
            continue
        ck(f"{c['id']} packaging p50", num(p50_s), c["result"]["p50"], tol=0.005)
        per_mine = c["result"]["p50"] / c["rows"] * 1000
        ck(f"{c['id']} per candidate us", num(per_s), per_mine, tol=0.05)
        ck(f"{c['id']} packaging rows/s", int(rps_s.replace(",", "")),
           c["rows"] * c["tps"])
        other = next((o for (s, rr), o in unsplit.items()
                      if s == size and rr != rws
                      and o["rows"] * o["tps"] == c["rows"] * c["tps"]), None)
        if other is None:
            continue
        # THE check this column needs: it must compare cost PER CANDIDATE, not the two
        # request medians. An earlier draft compared medians and printed "2.83x faster"
        # directly beneath a finding that said the other packaging was cheaper. Both were
        # true of different quantities, which is exactly how a report contradicts itself.
        per_other = other["result"]["p50"] / other["rows"] * 1000
        got_mag, got_dir = split_label(label.split(" than ")[0])
        want_mag, want_dir = split_label(expected_label(per_other, per_mine))
        ck(f"{c['id']} packaging direction (per candidate)", got_dir, want_dir)
        ck(f"{c['id']} packaging magnitude (per candidate)", got_mag, want_mag,
           tol=0.011)
        # And a guard against the regression itself: whenever the per-REQUEST ratio
        # points the other way, the column must not be reporting that instead.
        _, median_dir = split_label(
            expected_label(other["result"]["p50"], c["result"]["p50"]))
        if median_dir != want_dir and median_dir in ("faster", "slower"):
            ck_true(f"{c['id']} packaging column is not the per-request ratio",
                    got_dir != median_dir,
                    "the column reports the direction of the per-REQUEST medians, which "
                    "contradicts the per-candidate finding above it")


def check_triton(text, cells):
    rows = rows_of(text, "lower median")
    if rows is None:
        return
    by_key = {key_of(c): c for c in cells}
    for r in rows:
        model, load, h_s, t_s, lower, h99_s, t99_s, cid_s = r[:8]
        size = UNLABEL[model]
        rws, tps = (int(x) for x in load.split(" x "))
        h = by_key.get((size, "hushar", rws, tps, rws))
        t = by_key.get((size, "triton", rws, tps, None))
        ck_true(f"triton row {model} {load} resolves", h is not None and t is not None,
                "the reference row names a cell that is not in results.json")
        if not h or not t:
            continue
        # The control the prior campaign needed: a cell is identified by its STACK, not
        # by its rate or its wire. A hushar tensor-wire arm reads like a Triton cell at a
        # glance and copying it across would attribute a latency to a server that never
        # served that load.
        ck(f"{t['id']} is really the Triton stack", t["stack"], "triton")
        ck(f"{t['id']} hushar p50", num(h_s), h["result"]["p50"], tol=0.005)
        ck(f"{t['id']} triton p50", num(t_s), t["result"]["p50"], tol=0.005)
        ck(f"{t['id']} hushar p99", num(h99_s), h["result"]["p99"], tol=0.005)
        ck(f"{t['id']} triton p99", num(t99_s), t["result"]["p99"], tol=0.005)
        ck(f"{t['id']} cell id", cid_s.strip("`"), t["id"])
        if t["result"]["p50"] < h["result"]["p50"]:
            want = f"Triton {h['result']['p50'] / t['result']['p50']:.2f}x"
        elif t["result"]["p50"] > h["result"]["p50"]:
            want = f"hushar {t['result']['p50'] / h['result']['p50']:.2f}x"
        else:
            want = "level"
        ck(f"{t['id']} lower median", lower, want)


def check_overload(text, cells):
    """Every cell that did not hold its rate appears, and only those."""
    bad = {c["id"] for c in cells
           if (c.get("verdict") or {}).get("state") not in (None, "sustained")}
    rows = rows_of(text, "slices held")
    listed = {r[-1].strip("`") for r in rows} if rows else set()
    ck("unsustained cells reported", listed, bad)
    if rows:
        by_id = {c["id"]: c for c in cells}
        for r in rows:
            cid = r[-1].strip("`")
            c = by_id.get(cid)
            if not c:
                continue
            ck(f"{cid} achieved", num(r[5]), c["result"]["achieved"], tol=0.05)
            ck(f"{cid} shed", int(r[6].replace(",", "")), c["result"]["shed"])


def check_every_cell(text, cells):
    rows = rows_of(text, "| shape |")
    ck_true("every-cell table present", rows is not None, "no '| shape |' table found")
    if not rows:
        return
    listed = {r[-1].strip("`") for r in rows}
    ck("every cell is listed", listed, {c["id"] for c in cells})


def check_counts(text, cells):
    m = re.search(r"(\d+) measured cells: (\d+) sustained, (\d+) partial, "
                  r"(\d+) failed", text)
    ck_true("the closing count is present", m is not None,
            "the report does not state how many cells it measured")
    if not m:
        return
    total, sus, part, fail = (int(g) for g in m.groups())
    states = [(c.get("verdict") or {}).get("state") for c in cells]
    ck("cell count", total, len(cells))
    ck("sustained count", sus, states.count("sustained"))
    ck("partial count", part, states.count("partial"))
    ck("failed count", fail, states.count("failed"))


def check_no_imports(text, cells):
    """Nothing from the prior campaign may be quoted as if measured here."""
    ck_true("no cell claims the prior run",
            all(c.get("run") == RUN for c in cells),
            "a cell names another run; this run's window differs from "
            f"{PRIOR_GPU}'s, so its cells are not comparable")
    # The prior run may be NAMED (it is, in Method and limits) but must be named as a
    # thing not imported from.
    if PRIOR_GPU in text:
        ck_true("the prior run is named as not imported from",
                "re-measured" in text or "not copied" in text,
                f"{PRIOR_GPU} is mentioned without saying its cells were not reused")


def control_checks(cells):
    """The verifier must FAIL on a fabricated row. Checked by actually trying it.

    A verifier nobody has seen fail is a verifier nobody has tested.
    """
    before = len(FAILS)
    fake = "| 10M | 999 x 999 | 999 | 1 | 1.00 | 1.00 | 1.00 | baseline | 1.0% | 1.0% " \
           "| `g5-hushar-10m-r999t999-mb999-p2` |"
    header = ("| model | rows x TPS | mini batch | pieces | p50 ms | p95 ms | p99 ms "
              "| against unsplit | GPU | CPU | cell |")
    check_gain_table("\n".join([header, "|---|", fake]), cells)
    produced = len(FAILS) - before
    del FAILS[before:]
    CHECKS[0] += 1
    if produced == 0:
        FAILS.append("CONTROL: a fabricated cell passed the gain check, so the gain "
                     "table is not actually being verified")


def main():
    report = RESULTS / "summary.md"
    if not report.exists():
        sys.exit(f"no {report}; run report.py first")
    text = report.read_text()
    cells = load_cells()

    check_structure(cells)
    check_answer_table(text, cells)
    check_gain_table(text, cells)
    check_decomposition(text, cells)
    check_packaging(text, cells)
    check_triton(text, cells)
    check_overload(text, cells)
    check_every_cell(text, cells)
    check_counts(text, cells)
    check_no_imports(text, cells)
    control_checks(cells)

    print(f"{CHECKS[0]} checks, {len(FAILS)} failures")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
