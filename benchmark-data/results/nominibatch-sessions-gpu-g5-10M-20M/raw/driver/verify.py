#!/usr/bin/env python3
"""Re-derives every number in `summary.md` from `raw/` and fails if one disagrees.

    verify.py

A report is generated once and read many times, and the failure mode that matters is not a
crash -- it is a figure that was true of an earlier run and quietly false of this one. So
every cell of every table is parsed back out of the generated file and recomputed from the
raw record it claims to describe.

A check that cannot find its input FAILS rather than passing quietly. "No data" and
"agrees" are different answers and only one of them is good news.

WHAT THIS RUN NEEDS CHECKED THAT A SPEED COMPARISON DOES NOT

  * **No cell ran a mini batch.** The run's premise is hushar WITHOUT its split against
    Triton as it ships. A cell that silently configured one would be comparing a hushar
    feature against a server that has no counterpart for it, which is this run's one way
    to lie and be believed. So every hushar cell is checked for `pieces == 1`, an absent
    `mini_batch_size`, the `whole request` plan, `sessions_per_request == 1` -- and the
    server banner it recorded at startup is checked for the plan name AND for the absence
    of any mini-batch line.
  * **The pair really is matched.** `request_lanes == sessions_per_model` on hushar is
    what makes its session count the same quantity as Triton's instance count. Recomputed
    per cell and read back out of the banner's threads line.
  * **Both arms ran one TensorRT.** The comparison is only about the servers if the
    library underneath them is identical. Every cell carries the version read off its host
    at preflight, and a lane whose two arms disagree fails here.
  * **Direction and magnitude, separately.** A verifier that only checks a ratio cannot
    catch a sign flip.
  * **A cell that did not hold its rate is visibly marked**, and every derived column on
    such a row is withheld. A reader who takes a shedding cell's median for a latency has
    been misled by the table rather than by the measurement.
  * **Nothing is imported.** Every cell must name THIS run, so a figure pasted from an
    earlier campaign fails rather than blending two windows. The mini-batch campaign is
    named explicitly, because it measured this grid's geometry and is the one a figure
    could plausibly drift in from.
  * **A control.** A fabricated row must make a check fail. A verifier nobody has seen
    fail is a verifier nobody has tested.
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


RUN = os.environ.get("RUN", "nominibatch-sessions-gpu-g5-10M-20M")
BENCH = _benchmark_data(HERE)
RESULTS = BENCH / "results" / RUN
RAW = RESULTS / "raw"
SUMMARY = RESULTS / "summary.md"
LABEL = {"10m": "10M", "20m": "20M"}
MARK = {"sustained": "ok", "partial": "part", "failed": "FAIL"}

FAILURES = []
CHECKS = [0]


def ck(name, got, want, tol=None):
    CHECKS[0] += 1
    if tol is not None and got is not None and want is not None:
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    if not ok:
        FAILURES.append(f"{name}: report says {got!r}, raw says {want!r}")
    return ok


def ck_true(name, cond, why=""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(f"{name}: {why}")
    return bool(cond)


def num(text):
    text = (text or "").strip()
    if text in ("--", "", "baseline"):
        return None
    try:
        return float(text.rstrip("x%").lstrip("+"))
    except ValueError:
        return None


# ---------------------------------------------------------------- the raw truth

def load_cells():
    path = RAW / "results.json"
    if not path.is_file():
        sys.exit(f"no {path}; run collect.py first")
    cells = [c for c in json.loads(path.read_text()) if c.get("phase") != "probe"]
    if not cells:
        sys.exit("results.json holds no main cells")
    return cells


def banner_of(c):
    return "\n".join(c.get("banner") or [])


def config_text(c):
    """Must match report.py's own `config_text`, which is what the tables key on."""
    return f"{c['parallelism']} {'sessions' if c['stack'] == 'hushar' else 'instances'}"


def p50_of(c):
    return (c.get("result") or {}).get("p50")


def state_of(c):
    return (c.get("verdict") or {}).get("state")


def held(c):
    return state_of(c) == "sustained"


def graph_of(c):
    return ((c.get("cloudwatch") or {}).get("InferenceTime") or {}).get("p50_ms")


def by_load(cells, stack):
    """{(size, rows, tps): [cells of that stack, ascending parallelism]}."""
    out = {}
    for c in cells:
        if c["stack"] != stack:
            continue
        out.setdefault((c["size"], c["rows"], c["tps"]), []).append(c)
    for group in out.values():
        group.sort(key=lambda c: c["parallelism"])
    return out


def sustained_from_slices(c):
    """Re-applies the verdict from the slices rather than trusting `sustained`."""
    slices = c.get("slices") or []
    r = c.get("result") or {}
    if not slices or not r:
        return None
    ok = [s["shed"] == 0 and s["failed"] == 0 and s["achieved"] is not None
          and s["achieved"] >= 0.98 * c["tps"] for s in slices]
    whole = (r.get("shed") == 0 and r.get("failed") == 0
             and r.get("achieved", 0) >= 0.98 * r.get("offered", 1))
    return bool(slices and all(ok) and whole)


# ---------------------------------------------------------------- table parsing

def rows_of(text, header_fragment):
    """The data rows of the first markdown table whose header contains the fragment."""
    out, seen = [], False
    for line in text.splitlines():
        if not seen:
            if line.startswith("|") and header_fragment in line:
                seen = True
            continue
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        out.append(cells)
    return out


# ---------------------------------------------------------------- checks

def check_structure(cells):
    """Each cell's own record, and the server banner it recorded, must agree."""
    for c in cells:
        cid = c["id"]
        ck_true(f"{cid} names this run", c.get("run") == RUN,
                f"cell names run {c.get('run')!r}, expected {RUN!r}")
        ck_true(f"{cid} sample_rate is 0", str(c.get("sample_rate")) in ("0.0", "0"),
                f"sample_rate is {c.get('sample_rate')!r}; feature logging must be off")
        ck_true(f"{cid} id carries its parallelism", f"-p{c['parallelism']}" in cid,
                "id does not carry parallelism")
        ck(f"{cid} pieces", c.get("pieces"), 1)
        ck(f"{cid} split flag", bool(c.get("split")), False)
        # The TensorRT version this run claims both arms share. Stamped from preflight on
        # every cell of both arms; its absence is a failure, not a blank column.
        ck_true(f"{cid} records a TensorRT version", bool(c.get("host_tensorrt")),
                "no host_tensorrt on the cell; the shared-library claim is unsupported")

        if c["stack"] != "hushar":
            ck(f"{cid} max_batch", c.get("max_batch"), 0)
            ck_true(f"{cid} Triton cell declares no mini batch",
                    c.get("mini_batch") is None
                    and c.get("mini_batch_configured") is None,
                    "a Triton cell records a mini batch")
            continue

        # THE INVARIANT: no mini batch, anywhere, by any name.
        ck(f"{cid} no mini_batch_size configured", c.get("mini_batch_configured"), None)
        ck(f"{cid} plan", c.get("plan"), "whole request")
        ck(f"{cid} sessions_per_request", c.get("sessions_per_request"), 1)
        ck(f"{cid} lanes = sessions", c.get("request_lanes"), c["parallelism"])
        ck(f"{cid} engine shape is the whole request",
           c.get("engine_shape"), [c["rows"], 690])
        ck(f"{cid} shape row count", c.get("mini_batch"), c["rows"])
        ck(f"{cid} engine cache names the shape",
           (c.get("engine_cache") or "").split("/")[-1], f"r{c['rows']}")

        b = banner_of(c)
        ck_true(f"{cid} banner records a plan", "plan     :" in b,
                "no plan line in the recorded banner")
        ck_true(f"{cid} banner plan matches", "plan     : whole request" in b,
                "banner does not report the 'whole request' plan")
        ck_true(f"{cid} banner names no mini batch", "mini batches of" not in b,
                "the banner names a mini batch; this run configures none")
        ck_true(f"{cid} banner session arithmetic",
                f"{c['request_lanes']} concurrent request(s) x 1 session(s) each" in b,
                "banner does not report one session per request")
        ck_true(f"{cid} banner session count",
                f"x {c['parallelism']} ONNX Runtime session(s)" in b,
                f"banner does not report {c['parallelism']} sessions")
        ck_true(f"{cid} banner names TensorRT", "TensorRT(device=0" in b,
                "banner does not name the TensorRT provider")
        ck_true(f"{cid} banner echoes the cache path",
                f"trt_engine_cache_path={c['engine_cache']}" in b,
                "banner does not echo this cell's engine cache")
        ck_true(f"{cid} banner shows the tensor wire",
                "<- [features] transformed and concatenated" in b,
                "banner does not show the single-binding tensor wire")
        ck_true(f"{cid} banner shows logging off", "logging no requests" in b,
                "banner does not report feature logging off")
        ck_true(f"{cid} no candidate model", "A/B" not in b,
                "banner names an A/B split; this run has no candidate model")

        v = sustained_from_slices(c)
        if v is not None:
            ck(f"{cid} sustained re-derived from slices", bool(c.get("sustained")), v)


def check_one_tensorrt(cells):
    """Both arms of a lane must have recorded the same TensorRT version."""
    by_host = {}
    for c in cells:
        by_host.setdefault(c.get("host"), {}).setdefault(
            c.get("host_tensorrt"), set()).add(c["arm"])
    for host, versions in sorted(by_host.items()):
        ck_true(f"{host} ran one TensorRT for both arms", len(versions) == 1,
                f"cells on {host} record TensorRT "
                + ", ".join(f"{v} ({', '.join(sorted(a))})"
                            for v, a in sorted(versions.items(),
                                               key=lambda kv: str(kv[0])))
                + "; this run claims one library underneath both servers")
        arms = {a for s in versions.values() for a in s}
        ck_true(f"{host} carries both arms", len(arms) >= 2,
                f"only {sorted(arms)} measured on {host}, so the shared-library claim "
                f"has nothing to compare")


def check_answer(text, cells):
    """Each arm at its own best sustained parallelism, and the ratio between them."""
    rows = rows_of(text, "at sessions")
    ck_true("answer table present", bool(rows), "no answer table in the summary")
    hus, tri = by_load(cells, "hushar"), by_load(cells, "triton")
    size_of = {v: k for k, v in LABEL.items()}
    ck("answer row count", len(rows), len(set(hus) | set(tri)))
    for r in rows:
        key = (size_of[r[0]], int(r[1]), int(r[2]))
        h, t = hus.get(key, []), tri.get(key, [])
        if not ck_true(f"answer row {key}", bool(h or t), "no cells for this load"):
            continue
        hb = min((c for c in h if held(c) and p50_of(c) is not None),
                 key=p50_of, default=None)
        tb = min((c for c in t if held(c) and p50_of(c) is not None),
                 key=p50_of, default=None)
        if hb:
            ck(f"answer {key} hushar p50", num(r[3]), p50_of(hb), tol=0.005)
            ck(f"answer {key} hushar sessions", int(r[4]), hb["parallelism"])
            ck_true(f"answer {key} hushar best is the lowest sustained median",
                    all(p50_of(hb) <= p50_of(c)
                        for c in h if held(c) and p50_of(c) is not None),
                    "a sustained hushar cell has a lower median than the one reported")
        else:
            ck_true(f"answer {key} hushar withheld", r[3] == "--" and r[4] == "--",
                    "a figure is printed where no hushar cell held its rate")
        if tb:
            ck(f"answer {key} Triton p50", num(r[5]), p50_of(tb), tol=0.005)
            ck(f"answer {key} Triton instances", int(r[6]), tb["parallelism"])
        else:
            ck_true(f"answer {key} Triton withheld", r[5] == "--" and r[6] == "--",
                    "a figure is printed where no Triton cell held its rate")
        if hb and tb:
            ck(f"answer {key} ratio", num(r[7]), p50_of(hb) / p50_of(tb), tol=0.011)
            ck_true(f"answer {key} ratio direction",
                    (num(r[7]) > 1.0) == (p50_of(hb) > p50_of(tb)),
                    f"ratio {r[7]!r} claims the wrong winner")
        else:
            ck_true(f"answer {key} ratio withheld", r[7] == "--",
                    "a ratio is printed with only one arm sustained")
        if hb and graph_of(hb) is not None:
            ck(f"answer {key} hushar graph", num(r[8]), graph_of(hb), tol=0.0011)
        ck(f"answer {key} hushar held", r[9],
           f"{sum(1 for c in h if held(c))}/{len(h)}")
        ck(f"answer {key} Triton held", r[10],
           f"{sum(1 for c in t if held(c))}/{len(t)}")


def check_grid(text, cells):
    """Every pair: hushar at N sessions beside Triton at N instances."""
    rows = rows_of(text, "hushar result")
    ck_true("grid table present", bool(rows), "no grid table in the summary")
    hus = {(c["size"], c["rows"], c["tps"], c["parallelism"]): c
           for c in cells if c["stack"] == "hushar"}
    tri = {(c["size"], c["rows"], c["tps"], c["parallelism"]): c
           for c in cells if c["stack"] != "hushar"}
    size_of = {v: k for k, v in LABEL.items()}
    ck("grid row count", len(rows), len(set(hus) | set(tri)))
    for r in rows:
        rows_, tps = (int(x) for x in r[1].split(" x "))
        key = (size_of[r[0]], rows_, tps, int(r[2]))
        h, t = hus.get(key), tri.get(key)
        if not ck_true(f"grid row {key}", h is not None or t is not None,
                       "no cell matches this row"):
            continue
        for cell_, cols, who in ((h, (3, 4, 5, 6), "hushar"),
                                 (t, (7, 8, 9, 10), "Triton")):
            if cell_ is None:
                ck_true(f"grid {key} {who} absent", all(r[i] == "--" for i in cols),
                        f"figures are printed for a {who} cell that does not exist")
                continue
            res = cell_["result"] or {}
            ck_true(f"grid {key} {who} verdict marked",
                    r[cols[0]].split()[0] == MARK.get(state_of(cell_)),
                    f"result cell {r[cols[0]]!r} does not mark {state_of(cell_)!r}")
            ck(f"grid {key} {who} p50", num(r[cols[1]]), res.get("p50"), tol=0.005)
            ck(f"grid {key} {who} p95", num(r[cols[2]]), res.get("p95"), tol=0.005)
            ck(f"grid {key} {who} p99", num(r[cols[3]]), res.get("p99"), tol=0.005)
        if h and t and held(h) and held(t):
            ck(f"grid {key} hushar/Triton", num(r[11]),
               p50_of(h) / p50_of(t), tol=0.011)
            ck_true(f"grid {key} ratio direction",
                    (num(r[11]) > 1.0) == (p50_of(h) > p50_of(t)),
                    f"ratio {r[11]!r} claims the wrong winner")
        else:
            ck_true(f"grid {key} ratio withheld", r[11] == "--",
                    f"a ratio {r[11]!r} is printed against a pair where one cell did "
                    f"not hold its rate")


def check_parallelism(text, cells, stack, noun):
    """One arm at each parallelism, against its own fewest."""
    groups = by_load(cells, stack)
    if not groups:
        # No cell of this arm exists yet, so an absent table is correct rather than
        # missing input. This is what makes the verifier usable mid-run as well as at the
        # end.
        return
    rows = rows_of(text, f"p50 vs fewest {noun}")
    ck_true(f"{stack} parallelism table present", bool(rows),
            f"no {stack} parallelism table in the summary")
    ck(f"{stack} parallelism row count", len(rows),
       sum(len(g) for g in groups.values()))
    size_of = {v: k for k, v in LABEL.items()}
    for r in rows:
        rows_, tps = (int(x) for x in r[1].split(" x "))
        key = (size_of[r[0]], rows_, tps)
        group = groups.get(key, [])
        c = next((g for g in group if g["parallelism"] == int(r[2])), None)
        if not ck_true(f"{stack} row {key} p{r[2]}", c is not None,
                       "no cell matches this row"):
            continue
        base = min(group, key=lambda g: g["parallelism"])
        ck(f"{stack} {key} p{r[2]} p50", num(r[4]), p50_of(c), tol=0.005)
        ck(f"{stack} {key} p{r[2]} p95", num(r[5]), (c["result"] or {}).get("p95"),
           tol=0.005)
        ck(f"{stack} {key} p{r[2]} p99", num(r[6]), (c["result"] or {}).get("p99"),
           tol=0.005)
        ck_true(f"{stack} {key} p{r[2]} verdict marked",
                r[3].split()[0] == MARK.get(state_of(c)),
                f"result cell {r[3]!r} does not mark {state_of(c)!r}")
        if c is base:
            ck_true(f"{stack} {key} p{r[2]} marked baseline", r[7] == "baseline",
                    f"the fewest-{noun} cell shows {r[7]!r} instead of 'baseline'")
            continue
        if not (held(c) and held(base)):
            ck_true(f"{stack} {key} p{r[2]} delta withheld", r[7] == "--",
                    f"a delta {r[7]!r} is printed against a cell that did not hold "
                    f"its rate")
            continue
        ck(f"{stack} {key} p{r[2]} delta magnitude", abs(num(r[7])),
           abs(p50_of(c) - p50_of(base)), tol=0.011)
        ck_true(f"{stack} {key} p{r[2]} delta direction",
                (num(r[7]) > 0) == (p50_of(c) > p50_of(base)),
                f"delta {r[7]!r} has the wrong sign for {p50_of(c)} against "
                f"{p50_of(base)}")


def check_stage(text, cells):
    rows = rows_of(text, "remainder")
    ck_true("stage table present", bool(rows), "no stage table in the summary")
    hus = {(LABEL[c["size"]], f"{c['rows']} x {c['tps']}", config_text(c)): c
           for c in cells if c["stack"] == "hushar"}
    ck("stage row count", len(rows), len(hus))
    for r in rows:
        key = (r[0], r[1], r[2])
        c = hus.get(key)
        if not ck_true(f"stage row {key}", c is not None, "no cell matches this row"):
            continue
        cw = c.get("cloudwatch") or {}

        def got(m):
            return (cw.get(m) or {}).get("p50_ms")

        ck(f"stage {key} p50", num(r[3]), p50_of(c), tol=0.005)
        ck(f"stage {key} InferenceTime", num(r[4]), got("InferenceTime"), tol=0.0011)
        ck(f"stage {key} VectorizationTime", num(r[5]), got("VectorizationTime"),
           tol=0.0011)
        ck(f"stage {key} ToTensorTime", num(r[6]), got("ToTensorTime"), tol=0.0011)
        if None not in (p50_of(c), got("InferenceTime")):
            want = p50_of(c) - got("InferenceTime") - (got("VectorizationTime") or 0) \
                - (got("ToTensorTime") or 0)
            ck(f"stage {key} remainder", num(r[7]), want, tol=0.0025)


def check_device(text, cells):
    rows = rows_of(text, "GPU max")
    ck_true("device table present", bool(rows), "no device table in the summary")
    by = {(LABEL[c["size"]], f"{c['rows']} x {c['tps']}", c["arm"], config_text(c)): c
          for c in cells}
    ck("device row count", len(rows), len(by))
    for r in rows:
        key = (r[0], r[1], r[2], r[3])
        c = by.get(key)
        if not ck_true(f"device row {key}", c is not None, "no cell matches this row"):
            continue
        gpu = c.get("gpu_utilization") or {}
        srv = c.get("server_utilization") or {}
        ck(f"device {key} GPU", num(r[4]), gpu.get("gpu_util_pct"), tol=0.051)
        ck(f"device {key} GPU max", num(r[5]), gpu.get("gpu_util_max_pct"), tol=0.051)
        ck(f"device {key} GPU MiB", num(r[6]), gpu.get("memory_used_mib"), tol=0.51)
        ck(f"device {key} power", num(r[7]), gpu.get("power_w"), tol=0.051)
        ck(f"device {key} host CPU", num(r[8]), srv.get("cpu_pct"), tol=0.051)
        ck(f"device {key} CPU max", num(r[9]), srv.get("cpu_max_pct"), tol=0.051)


def check_hold(text, cells):
    rows = rows_of(text, "slices held")
    ck_true("hold table present", bool(rows), "no hold table in the summary")
    ck("hold row count", len(rows), len(cells))
    by = {(LABEL[c["size"]], c["arm"], config_text(c),
           f"{c['rows']} x {c['tps']}"): c for c in cells}
    for r in rows:
        key = (r[0], r[1], r[2], r[3])
        c = by.get(key)
        if not ck_true(f"hold row {key}", c is not None, "no cell matches this row"):
            continue
        res = c["result"] or {}
        ck(f"hold {key} shed", int(r[6]), res.get("shed"))
        ck(f"hold {key} failed", int(r[7]), res.get("failed"))
        v = c.get("verdict") or {}
        ck(f"hold {key} slices", r[5], f"{v.get('slices_held')}/{v.get('slices_total')}")


def check_library(text, cells):
    """The versions table: one row per host and arm, from the cells' own stamps."""
    rows = rows_of(text, "ORT in use")
    ck_true("library table present", bool(rows), "no library table in the summary")
    first = {}
    for c in cells:
        first.setdefault((c.get("host"), c["arm"]), c)
    ck("library row count", len(rows), len(first))
    for r in rows:
        c = first.get((r[0], r[1]))
        if not ck_true(f"library row {r[0]} {r[1]}", c is not None,
                       "no cell for this host and arm"):
            continue
        ck(f"library {r[0]} {r[1]} TensorRT", r[4], c.get("host_tensorrt"))
        ck(f"library {r[0]} {r[1]} CUDA", r[5], c.get("host_cuda_major"))
        ck(f"library {r[0]} {r[1]} driver", r[3], c.get("host_nvidia_driver"))
        ck(f"library {r[0]} {r[1]} ONNX Runtime", r[6], c.get("onnxruntime"))


def check_every_cell(text, cells):
    rows = rows_of(text, "per request")
    ck_true("every-cell table present", bool(rows), "no every-cell table")
    ck("every-cell row count", len(rows), len(cells))
    by = {c["id"]: c for c in cells}
    for r in rows:
        cid = r[0].strip("`")
        c = by.get(cid)
        if not ck_true(f"every-cell row {cid}", c is not None, "no cell with this id"):
            continue
        ck(f"every-cell {cid} parallelism", int(r[3]), c["parallelism"])
        ck(f"every-cell {cid} p50", num(r[6]), p50_of(c), tol=0.005)
        if c["stack"] == "hushar":
            ck(f"every-cell {cid} plan", r[1], "whole request")
            ck(f"every-cell {cid} per request", r[4], "1")


def check_prime(text):
    """The engine table, against the prime records the lane wrote."""
    primes = [json.loads(p.read_text())
              for p in sorted((RAW / "prime").glob("*.json"))] \
        if (RAW / "prime").is_dir() else []
    if not primes:
        return
    rows = rows_of(text, "cold build s")
    ck_true("prime table present", bool(rows), "no engine table in the summary")
    want = {(LABEL.get(d.get("size"), d.get("size")), s.get("rows")): s
            for d in primes for s in d.get("shapes", [])}
    ck("prime row count", len(rows), len(want))
    for r in rows:
        s = want.get((r[0], int(r[1])))
        if not ck_true(f"prime row {r[0]} r{r[1]}", s is not None,
                       "no primed shape matches this row"):
            continue
        ck(f"prime {r[0]} r{r[1]} cold", num(r[3]), s.get("build_seconds"), tol=0.051)
        ck(f"prime {r[0]} r{r[1]} cached", num(r[4]), s.get("cached_seconds"), tol=0.051)
        ck(f"prime {r[0]} r{r[1]} shape", r[2], str([int(r[1]), 690]))


def check_counts(text, cells):
    m = re.search(r"\|\s*cells\s*\|\s*(\d+) measured, (\d+) sustained", text)
    if not ck_true("cell counts present", m is not None,
                   "the configuration block does not state the cell counts"):
        return
    ck("measured count", int(m.group(1)), len(cells))
    ck("sustained count", int(m.group(2)),
       sum(1 for c in cells if c.get("sustained")))
    t = re.search(r"\|\s*TensorRT\s*\|\s*([0-9][0-9.]*),", text)
    if ck_true("TensorRT version stated", t is not None,
               "the configuration block does not state the TensorRT version"):
        ck("TensorRT version", t.group(1),
           next((c.get("host_tensorrt") for c in cells if c.get("host_tensorrt")), None))


def check_no_imports(text, cells):
    for c in cells:
        ck_true(f"{c['id']} is this run's", c.get("run") == RUN,
                f"cell belongs to {c.get('run')!r}")
    # THIS RUN'S OWN NAME IS REMOVED FIRST, and that is not a nicety. `nominibatch-...`
    # CONTAINS `minibatch-...` as a substring, so a plain `in text` reports the previous
    # campaign as imported on every well-formed report -- which the selftest caught. The
    # names are compared against a haystack this run's name has been taken out of.
    haystack = text.replace(RUN, "")
    for other in ("minibatch-sessions-gpu-g5-10M-20M", "minibatch-gpu-g5-10M-20M",
                  "config-sweep-gpu-g5-10M-20M", "config-sweep-gpu-g5-wire-10M-20M"):
        if other in RUN:
            continue
        ck_true(f"no figure imported from {other}",
                other not in haystack or f"]({other}" in text or f"[{other}]" in text,
                f"the summary mentions {other} outside a link")


def control_checks(cells):
    """A fabricated row must make a check fail, or none of the above means anything."""
    saved = list(FAILURES)
    FAILURES.clear()
    fake = json.loads(json.dumps(cells[0]))
    fake["result"]["p50"] = (fake["result"]["p50"] or 1.0) * 3 + 7
    check_grid(SUMMARY.read_text(), [fake] + list(cells[1:]))
    caught_p50 = bool(FAILURES)
    FAILURES.clear()
    # And the invariant specifically: a cell that configured a mini batch must fail, or
    # the one thing this run has to rule out is not actually being checked.
    split = json.loads(json.dumps(cells[0]))
    if split["stack"] != "hushar":
        split = json.loads(json.dumps(next(
            (c for c in cells if c["stack"] == "hushar"), cells[0])))
    split["mini_batch_configured"] = split["rows"] // 2
    split["pieces"] = 2
    check_structure([split])
    caught_split = bool(FAILURES)
    FAILURES.clear()
    FAILURES.extend(saved)
    ck_true("CONTROL: a fabricated p50 is caught", caught_p50,
            "tripling a cell's p50 did not fail any check, so the grid check is not "
            "actually comparing the report against the raw record")
    return ck_true("CONTROL: a configured mini batch is caught", caught_split,
                   "a cell carrying a mini batch did not fail any check, so this run's "
                   "central invariant is not actually being verified")


def main():
    if not SUMMARY.is_file():
        sys.exit(f"no {SUMMARY}; run report.py first")
    text = SUMMARY.read_text()
    cells = load_cells()

    check_structure(cells)
    check_one_tensorrt(cells)
    check_answer(text, cells)
    check_grid(text, cells)
    check_parallelism(text, cells, "hushar", "sessions")
    check_parallelism(text, cells, "triton", "instances")
    check_stage(text, cells)
    check_device(text, cells)
    check_hold(text, cells)
    check_library(text, cells)
    check_every_cell(text, cells)
    check_prime(text)
    check_counts(text, cells)
    check_no_imports(text, cells)
    control_checks(cells)

    print(f"{CHECKS[0]} check(s) over {len(cells)} cell(s)")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
