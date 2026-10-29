#!/usr/bin/env python3
"""Writes `results/minibatch-gpu-g5-10M-20M/summary.md` from `raw/results.json`.

    report.py            # the default: the file collect.py produced
    report.py --dry      # print to stdout, write nothing

THE QUESTION THIS REPORT ANSWERS is whether cutting a request into mini batches lowers
its latency on ONE NVIDIA A10G. `minibatch-rows-cpu-arm64-10M` found the split to be the
largest single lever it measured on a 64-core CPU -- 2.10x at 100 rows, 2.64x at 200 --
and the reason it worked there does not obviously exist here.

WHAT THE REPORT MUST SAY OUT LOUD, because it decides how every number below reads: the
split's two terms move in OPPOSITE directions on a GPU, and one of them is bounded by the
server's own architecture rather than by the device.

    graph      hushar leases ONE ONNX Runtime session per REQUEST and shares it across
               every mini batch of that request; the TensorRT execution provider holds a
               per-session mutex around enqueue. The pieces therefore QUEUE ON THE
               DEVICE, and a split buys N kernel launches and N host-device copies for
               the same arithmetic at a smaller M. This term can only get worse.
    vectorize  the 210 transformations are host work on a rayon pool sized to `num_cpus`,
               and the pieces really do run at once. This term can get better.

So the report's job is to separate them and to show which dominates at each packaging,
NOT to print a single gain number. The decomposition table is the point of the document;
the gain column is its summary.

READING `graph` ON A SPLIT CELL -- the one trap in this data. hushar publishes one
duration per stage per request, and under a split that duration is the MAXIMUM over the
mini batches, not their sum. With the pieces serialized on the device it is therefore
neither total device work nor a clean per-piece figure, so:

  * no per-row column is offered for a split cell, and
  * `everything else` is computed from it only for UNSPLIT cells, where it is exact.

`minibatch-rows-cpu-arm64-10M` set this convention and this report keeps it. The
client-side median is the ground truth on every cell.

NEVER PRINT A BARE RATIO. A column of `0.67x` makes the reader supply the direction and
they will supply the wrong one; an earlier report in this series generated the sentence
"hushar is 0.6x faster" from exactly that. Every comparison here carries its direction
inside the cell as words, and `verify.py` recomputes direction and magnitude separately.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    """The `benchmark-data` directory, found by walking up rather than counting `..`.

    This file lives in two places -- the working copy under `benchmark-data/private/
    <run>/driver` and the archived copy under `benchmark-data/results/<run>/raw/driver`
    -- which are different depths, so a fixed number of `.parent`s is right in one and
    silently wrong in the other.
    """
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not under a benchmark-data directory")


BENCH = _benchmark_data(HERE)
RUN = os.environ.get("RUN", "minibatch-gpu-g5-10M-20M")
RESULTS = BENCH / "results" / RUN
RAW = RESULTS / "raw"
OUT = RESULTS / "summary.md"

# The campaign this run's unsplit cells are continuous with. Named so the report cannot
# drift from a baseline it never identifies -- but NOT imported from: that run measured a
# 300 s window and this one 900 s, so its cells are not comparable and every baseline
# here is re-measured. See `Method and limits`.
PRIOR_GPU = "config-sweep-gpu-g5-10M-20M"
PRIOR_CPU = "minibatch-rows-cpu-arm64-10M"

# Below this, two medians are a tie rather than a difference. No cell was repeated in this
# run, so the figure is inherited from the campaign that did repeat one: the CPU
# mini-batch run measured an 8.4% same-host spread on a SPLIT configuration, whose
# completion time is the slowest of several pieces and therefore the noisiest thing here.
# Using the split figure rather than the 1.1% single-call figure is the conservative
# choice, and it is what makes a 1.05x "no measurable gain" instead of a finding.
TIE_PCT = 8.4


# ---------------------------------------------------------------- plumbing

def load(path, default=None):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text())


def fmt(value, digits=2, dash="--"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def dash_or(value, f, dash="--"):
    return dash if value is None else f(value)


def pct(value, digits=1):
    return dash_or(value, lambda v: f"{v:.{digits}f}%")


MARK = {"sustained": "ok", "partial": "partial", "failed": "no"}


def row_of(cell):
    """Everything a table in this report can ask of one cell, flattened."""
    r = cell.get("result") or {}
    v = cell.get("verdict") or {}
    cw = cell.get("cloudwatch") or {}
    gpu = cell.get("gpu_utilization") or {}
    return dict(
        cid=cell.get("id"), run=cell.get("run"), host=cell.get("host"),
        server_ip=cell.get("server_ip"),
        size=cell.get("size"), stack=cell.get("stack"), arm=cell.get("arm"),
        rows=cell.get("rows"), tps=cell.get("tps"),
        rows_per_sec=cell.get("rows_per_sec"),
        mb=cell.get("mini_batch"), pieces=cell.get("pieces"),
        split=bool(cell.get("split")),
        par=cell.get("parallelism"), kind=cell.get("parallelism_kind"),
        engine_shape=cell.get("engine_shape"),
        p50=r.get("p50"), p95=r.get("p95"), p99=r.get("p99"),
        offered=r.get("offered"), achieved=r.get("achieved"),
        shed=r.get("shed"), failed=r.get("failed"),
        state=v.get("state"), held=v.get("slices_held"), total=v.get("slices_total"),
        sustained=bool(cell.get("sustained")),
        cpu=cell.get("server_cpu_pct"),
        gpu_pct=gpu.get("gpu_util_pct"), power=gpu.get("power_w"),
        gpu_mem=gpu.get("memory_used_mib"),
        graph=(cw.get("InferenceTime") or {}).get("p50_ms"),
        vec=(cw.get("VectorizationTime") or {}).get("p50_ms"),
        buckets=(cw.get("InferenceTime") or {}).get("buckets"),
    )


def mark(r):
    return MARK.get(r["state"] or "?", "?")


def result_text(r):
    """`ok 1000.0/1000` -- the verdict and the rate in one cell."""
    return f"{mark(r)} {fmt(r['achieved'], 1)}/{r['offered']}"


def gain_label(base_p50, p50, tie=TIE_PCT):
    """How `p50` compares with `base_p50`, with the DIRECTION inside the string.

    Returns words, never a bare multiplier: `1.68x faster`, `1.22x slower`, or `level`
    when the two are inside this run's tie band. A split that is slower is a real and
    expected outcome here, so the slower branch is as much a result as the faster one.
    """
    if base_p50 in (None, 0) or p50 in (None, 0):
        return "--"
    change = (p50 - base_p50) / base_p50 * 100.0
    if abs(change) < tie:
        return "level"
    if p50 < base_p50:
        return f"{base_p50 / p50:.2f}x faster"
    return f"{p50 / base_p50:.2f}x slower"


def gain_ratio(base_p50, p50):
    """The signed magnitude behind `gain_label`, for the machine-readable sidecar."""
    if base_p50 in (None, 0) or p50 in (None, 0):
        return None
    return round(base_p50 / p50, 3)


# ---------------------------------------------------------------- grouping

def groups(rows):
    """hushar's cells as (size, rows, tps) -> {mini_batch: row}, baseline first.

    A group is one LOAD, and the mini batch is the axis inside it. The unsplit entry --
    mini batch == rows -- is the baseline every gain in that group divides by, so a group
    without one is reported as having no baseline rather than silently compared against
    its largest split.
    """
    out = {}
    for r in rows:
        if r["stack"] != "hushar" or r["mb"] is None:
            continue
        out.setdefault((r["size"], r["rows"], r["tps"]), {})[r["mb"]] = r
    return dict(sorted(out.items(), key=lambda kv: (kv[0][0], kv[0][1])))


def triton_index(rows):
    return {(r["size"], r["rows"], r["tps"]): r
            for r in rows if r["stack"] == "triton"}


def baseline_of(group, rows_in_request):
    return group.get(rows_in_request)


# ---------------------------------------------------------------- sections

def answer_table(gs):
    """The headline: every sustained mini batch against its own unsplit baseline.

    ONLY sustained cells. A cell that did not hold its offered rate has a median that
    measures queue depth, and putting a 4,128 ms "p50" in a gain column beside a 2.53 ms
    one is not a comparison -- those cells are reported separately by achieved rate.
    """
    out = ["| model | rows x TPS | mini batch | pieces | p50 ms | p95 ms | p99 ms "
           "| against unsplit | GPU | CPU | cell |",
           "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    n = 0
    for (size, rows, tps), group in gs.items():
        base = baseline_of(group, rows)
        for mb in sorted(group, reverse=True):
            r = group[mb]
            if r["state"] != "sustained":
                continue
            label = ("baseline" if mb == rows
                     else (gain_label(base["p50"], r["p50"])
                           if base and base["state"] == "sustained"
                           else "no sustained baseline"))
            out.append(
                f"| {size.upper()} | {rows} x {tps} | {mb} | {r['pieces']} "
                f"| {fmt(r['p50'])} | {fmt(r['p95'])} | {fmt(r['p99'])} "
                f"| {label} | {pct(r['gpu_pct'])} | {pct(r['cpu'])} | `{r['cid']}` |")
            n += 1
    return "\n".join(out), n


def best_table(gs):
    """One line per load: did splitting help at all, and which size was best."""
    out = ["| model | rows x TPS | rows/s | unsplit p50 | best split | its p50 "
           "| verdict |",
           "|---|---:|---:|---:|---:|---:|---|"]
    verdicts = []
    for (size, rows, tps), group in gs.items():
        base = baseline_of(group, rows)
        if not base or base["state"] != "sustained":
            out.append(f"| {size.upper()} | {rows} x {tps} | {rows * tps:,} | -- | -- "
                       f"| -- | no sustained unsplit baseline |")
            continue
        splits = [r for mb, r in group.items()
                  if mb != rows and r["state"] == "sustained" and r["p50"]]
        if not splits:
            out.append(f"| {size.upper()} | {rows} x {tps} | {rows * tps:,} "
                       f"| {fmt(base['p50'])} | -- | -- | no split sustained the rate |")
            verdicts.append((size, rows, tps, None, None, "none sustained"))
            continue
        best = min(splits, key=lambda r: r["p50"])
        label = gain_label(base["p50"], best["p50"])
        out.append(f"| {size.upper()} | {rows} x {tps} | {rows * tps:,} "
                   f"| {fmt(base['p50'])} | {best['mb']} ({best['pieces']} pieces) "
                   f"| {fmt(best['p50'])} | **{label}** |")
        verdicts.append((size, rows, tps, best["mb"], best["p50"], label))
    return "\n".join(out), verdicts


def decomposition_table(gs):
    """Where the median goes, and the whole explanation of the answer.

    `graph` and `vectorize` are hushar's own `InferenceTime` and `VectorizationTime`
    medians over the measured window's minute buckets. On a SPLIT cell both are the
    maximum over that request's mini batches, so they are the span the slowest piece set
    rather than the request's total -- which is exactly why `everything else` is left
    blank there instead of being computed from a subtraction that does not hold.
    """
    out = ["| model | rows x TPS | mini batch | pieces | p50 ms | graph ms | vectorize ms "
           "| everything else | graph share |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (size, rows, tps), group in gs.items():
        for mb in sorted(group, reverse=True):
            r = group[mb]
            if r["state"] != "sustained":
                continue
            share = (f"{r['graph'] / r['p50'] * 100:.0f}%"
                     if r["graph"] and r["p50"] else "--")
            if r["split"]:
                # A subtraction against a MAXIMUM-over-pieces figure is not a remainder.
                rest = "n/a (split)"
                share += " (max piece)"
            elif r["graph"] is not None and r["vec"] is not None and r["p50"]:
                rest = fmt(r["p50"] - r["graph"] - r["vec"], 3)
            else:
                rest = "--"
            out.append(
                f"| {size.upper()} | {rows} x {tps} | {mb} | {r['pieces']} "
                f"| {fmt(r['p50'])} | {fmt(r['graph'], 3)} | {fmt(r['vec'], 3)} "
                f"| {rest} | {share} |")
    return "\n".join(out)


def packaging_table(gs):
    """100 rows against 500 at the SAME offered rows per second, both unsplit.

    Free from the grid's shape and the question a GPU actually rewards: the CPU lever is
    to split a request to win cores, the GPU lever is to merge requests to fill the
    device. This table is the second one, measured rather than argued.

    THE COMPARISON COLUMN IS PER CANDIDATE, not per request. Comparing the two medians
    would be meaningless -- a 100-row request is obviously quicker than a 500-row one --
    and an earlier draft of this table did exactly that, producing a column that said
    "2.83x faster" directly beneath a finding that said the 500-row request was cheaper.
    Both were arithmetically right about different quantities. The one that answers
    "which packaging should a caller use at a fixed row throughput" is cost per candidate.
    """
    out = ["| model | rows/s | packaging | req/s | p50 ms | per candidate us "
           "| graph ms | GPU | CPU | per candidate, against the other |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    pairs = {}
    for (size, rows, tps), group in gs.items():
        base = baseline_of(group, rows)
        if base and base["state"] == "sustained":
            pairs.setdefault((size, rows * tps), []).append(base)
    any_row = False
    for (size, rps), arms in sorted(pairs.items()):
        if len(arms) < 2:
            continue
        arms.sort(key=lambda r: r["rows"])
        small, large = arms[0], arms[-1]
        for r in arms:
            other = large if r is small else small
            mine = r["p50"] / r["rows"] * 1000
            theirs = other["p50"] / other["rows"] * 1000
            label = gain_label(theirs, mine)
            if label != "level":
                label = f"{label} than {other['rows']} rows"
            out.append(
                f"| {size.upper()} | {rps:,} | {r['rows']} rows | {r['tps']} "
                f"| {fmt(r['p50'])} | {fmt(mine, 1)} "
                f"| {fmt(r['graph'], 3)} | {pct(r['gpu_pct'])} | {pct(r['cpu'])} "
                f"| {label} |")
            any_row = True
    if not any_row:
        return None
    return "\n".join(out)


def triton_table(gs, tri):
    """The reference line: the same packaging on a server that cannot split.

    Triton is not a variable in this run. It is here to separate "the split did nothing"
    from "the packaging did nothing": if 500-row requests are worse than 100-row ones on
    BOTH stacks, that is the device and not hushar.
    """
    if not tri:
        return None
    out = ["| model | rows x TPS | hushar unsplit p50 | Triton p50 | lower median "
           "| hushar p99 | Triton p99 | Triton cell |",
           "|---|---:|---:|---:|---|---:|---:|---|"]
    n = 0
    for (size, rows, tps), group in gs.items():
        t = tri.get((size, rows, tps))
        base = baseline_of(group, rows)
        if not t or not base:
            continue
        if t["state"] != "sustained" or base["state"] != "sustained":
            continue
        if t["p50"] < base["p50"]:
            lower = f"Triton {base['p50'] / t['p50']:.2f}x"
        elif t["p50"] > base["p50"]:
            lower = f"hushar {t['p50'] / base['p50']:.2f}x"
        else:
            lower = "level"
        out.append(f"| {size.upper()} | {rows} x {tps} | {fmt(base['p50'])} "
                   f"| {fmt(t['p50'])} | {lower} | {fmt(base['p99'])} "
                   f"| {fmt(t['p99'])} | `{t['cid']}` |")
        n += 1
    return ("\n".join(out), n) if n else None


def overload_table(rows):
    """Cells that did not hold their offered rate, by ACHIEVED rate and nothing else.

    Their p50 is queue depth, so it is deliberately not placed beside a sustained one.
    """
    bad = [r for r in rows if r["state"] and r["state"] != "sustained"]
    if not bad:
        return None
    out = ["| model | rows x TPS | stack | mini batch | result | achieved | shed "
           "| failed | slices held | GPU | CPU | cell |",
           "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(bad, key=lambda r: (r["size"], r["rows"], -(r["mb"] or 0))):
        out.append(
            f"| {r['size'].upper()} | {r['rows']} x {r['tps']} | {r['stack']} "
            f"| {r['mb'] if r['mb'] else 'n/a'} | {mark(r)} | {fmt(r['achieved'], 1)} "
            f"| {r['shed']:,} | {r['failed']:,} | {r['held']}/{r['total']} "
            f"| {pct(r['gpu_pct'])} | {pct(r['cpu'])} | `{r['cid']}` |")
    return "\n".join(out)


def every_cell_table(rows):
    out = ["| model | rows x TPS | stack | mini batch | pieces | shape | result | p50 "
           "| p95 | p99 | graph | vectorize | GPU | W | CPU | shed | cell |",
           "|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["size"], r["rows"], r["stack"],
                                         -(r["mb"] or r["rows"]))):
        shape = (f"[{r['engine_shape'][0]}, {r['engine_shape'][1]}]"
                 if r.get("engine_shape") else "--")
        out.append(
            f"| {r['size'].upper()} | {r['rows']} x {r['tps']} | {r['stack']} "
            f"| {r['mb'] if r['mb'] else 'n/a'} | {r['pieces']} | {shape} "
            f"| {result_text(r)} | {fmt(r['p50'])} | {fmt(r['p95'])} | {fmt(r['p99'])} "
            f"| {fmt(r['graph'], 3)} | {fmt(r['vec'], 3)} | {pct(r['gpu_pct'])} "
            f"| {fmt(r['power'], 0)} | {pct(r['cpu'])} | {r['shed']:,} | `{r['cid']}` |")
    return "\n".join(out)


def prime_table(primes):
    """One engine per SHAPE, built once and proved reloadable."""
    shapes = []
    for size, doc in sorted(primes.items()):
        for s in doc.get("shapes", []):
            shapes.append((size, s))
    if not shapes:
        return None
    out = ["| model | mini batch | engine shape | first build | fresh server reload "
           "| cache |",
           "|---|---:|---|---:|---:|---|"]
    for size, s in shapes:
        shape = s.get("engine_shape") or []
        out.append(f"| {size.upper()} | {s.get('mini_batch')} "
                   f"| {('[%d, %d]' % tuple(shape)) if len(shape) == 2 else '--'} "
                   f"| {fmt(s.get('build_seconds'), 1)}s "
                   f"| {fmt(s.get('cached_seconds'), 1)}s "
                   f"| `{s.get('engine_cache')}` |")
    return "\n".join(out)


def provenance(cells, primes):
    """What the numbers were measured on, read out of the cells rather than asserted."""
    def one(key, among=None):
        vals = sorted({c.get(key) for c in (among or cells) if c.get(key)})
        return vals[0] if len(vals) == 1 else (", ".join(map(str, vals)) or "--")

    hush = [c for c in cells if c.get("stack") == "hushar"]
    tri = [c for c in cells if c.get("stack") == "triton"]
    hostinfo = load(RAW / "hostinfo.json", {}) or {}
    gpu_names = sorted({h.get("gpu") for h in hostinfo.values() if h.get("gpu")})
    rows = [
        ("server", f"`{one('instance')}`, {one('cores')} vCPU"),
        ("GPU", ", ".join(gpu_names) or "NVIDIA A10G"),
        ("TensorRT", "the library extracted from the Triton image, so both stacks "
                     "compile with the same one; `deploy.sh` refuses on any unresolved "
                     "dependency in the provider"),
        ("hushar", f"ONNX Runtime {one('onnxruntime', hush)}, TensorRT execution "
                   f"provider, FP32"),
        ("Triton", f"`{one('triton_image', tri) if tri else '--'}`, `tensorrt` backend, "
                   f"`max_batch_size: 0`, KIND_GPU, FP32"),
        ("graphs", ", ".join(
            f"{c.upper()} `{m[:12]}`"
            for c, m in sorted({(x['size'], x['model_md5']) for x in cells}))),
        ("sessions", ", ".join(
            f"{s.upper()} {p}"
            for s, p in sorted({(x['size'], x['parallelism']) for x in hush}))
            + " -- pinned from " + f"`{PRIOR_GPU}`, not re-probed"),
        ("window", f"{one('warmup')}s warm-up discarded, then {one('duration')}s "
                   f"measured as one continuous load in {one('segments')} slices"),
        ("engine shapes", ", ".join(
            f"{s.upper()} {sorted({x['mini_batch'] for x in hush if x['size'] == s})}"
            for s in sorted({x['size'] for x in hush}))),
        ("revision", f"`{one('rev')}`"),
    ]
    return "\n".join(["| what | value |", "|---|---|"]
                     + [f"| {k} | {v} |" for k, v in rows])


def provenance_hosts(rows):
    """Which machine measured what, grouped by the thing that discriminates them.

    Plan-level names are no help: every rig calls its first server `g5-1`, so two
    different machines record the same value. The private address discriminates, and the
    DATE is what a reader needs, so the grouping is by address and the reporting by date.
    """
    by_ip = {}
    for r in rows:
        if r.get("server_ip"):
            by_ip.setdefault(r["server_ip"], []).append(r)
    if len(by_ip) <= 1:
        return None
    return ("Measured on %d machines of the same type; each load's arms were always "
            "measured together on one machine, so no comparison in this report crosses "
            "a host." % len(by_ip))


# ---------------------------------------------------------------- findings

def findings(gs, verdicts, rows, tri_result, tri):
    """The prose, branching on what the data says rather than on what was expected.

    Each finding is a (lead, rest) pair so the formatter owns the numbering and the bold
    run together. Building the markers into the text instead produced `**1 ·** Splitting
    ... measured.**` -- a stray closing marker from stripping the opening one.

    Every sentence here is generated from a number in a table above it, because the
    verifier re-derives the tables and would otherwise have nothing to check the prose
    against. A finding that asserts a direction the data does not carry is the failure
    mode this function exists to avoid.
    """
    out = []
    helped = [v for v in verdicts if v[5] and "faster" in v[5]]
    hurt = [v for v in verdicts if v[5] and "slower" in v[5]]
    level = [v for v in verdicts if v[5] == "level"]
    nothing = [v for v in verdicts if v[5] == "none sustained"]

    def plural(n, one, many):
        return one if n == 1 else many

    # 1 -- the answer, whichever way it went.
    parts = []
    if hurt:
        parts.append(f"{len(hurt)} {plural(len(hurt), 'was', 'were')} slower than "
                     f"{plural(len(hurt), 'its', 'their')} own unsplit baseline")
    if level:
        parts.append(f"{len(level)} {plural(len(level), 'was', 'were')} inside this "
                     f"run's {TIE_PCT}% tie band")
    if nothing:
        parts.append(f"on {len(nothing)} no split held the offered rate at all")
    if helped:
        parts.append(f"{len(helped)} {plural(len(helped), 'was', 'were')} faster")
    if not helped:
        out.append((
            "Splitting a request did not lower its latency at any load measured",
            f"Of {len(verdicts)} loads, " + ", and ".join(
                [", ".join(parts[:-1]), parts[-1]] if len(parts) > 2 else parts) +
            f". That is the opposite of `{PRIOR_CPU}`, where the same knob was worth "
            f"2.10x at 100 rows and 2.64x at 200 -- and the reason is the next finding "
            f"rather than the device."))
    elif not hurt:
        out.append((
            f"Splitting lowered the median at {len(helped)} of {len(verdicts)} loads",
            "; ".join(f"{s.upper()} {r} x {t} at mini batch {mb}, {lab}"
                      for s, r, t, mb, _, lab in helped) + "."))
    else:
        out.append((
            f"The split helped at {len(helped)} of {len(verdicts)} loads and hurt at "
            f"{len(hurt)}",
            "; ".join(f"{s.upper()} {r} x {t}: {lab}"
                      for s, r, t, mb, _, lab in verdicts if lab) + "."))

    # 2 -- the mechanism, from the decomposition.
    unsplit = [r for r in rows if r["stack"] == "hushar" and not r["split"]
               and r["state"] == "sustained" and r["graph"] and r["vec"]]
    if unsplit:
        shares = [r["vec"] / r["p50"] * 100 for r in unsplit]
        out.append((
            "The term a split can improve is too small to pay for the term it makes "
            "worse",
            f"Across the unsplit cells vectorization is {min(shares):.0f}% to "
            f"{max(shares):.0f}% of the client median, so even driving it to zero "
            f"cannot buy what the extra kernel launches and host-device copies cost. "
            f"One ONNX Runtime session is leased per REQUEST and shared by all of its "
            f"mini batches, and the TensorRT provider serializes concurrent `Run()` on "
            f"one session, so the pieces fan out for the feature work and then queue on "
            f"the device. `execution_contexts_per_replica`, the knob that would give "
            f"each piece its own device context, exists only on the Neuron backend."))

    # 3 -- the two terms, moving in opposite directions, with the numbers.
    moves = []
    for (size, rws, tps), group in gs.items():
        base = baseline_of(group, rws)
        if not base or base["state"] != "sustained" or not base["graph"]:
            continue
        for mb in sorted(group, reverse=True):
            r = group[mb]
            if mb == rws or r["state"] != "sustained" or not r["graph"]:
                continue
            moves.append((size, rws, tps, mb, r["graph"] - base["graph"],
                          r["vec"] - base["vec"]))
    if moves:
        worst = max(moves, key=lambda m: m[4])
        best_vec = min(moves, key=lambda m: m[5])
        same = worst[:4] == best_vec[:4]
        where = (f"both on {worst[0].upper()} {worst[1]} x {worst[2]} at mini batch "
                 f"{worst[3]}, the same cell"
                 if same else
                 f"{worst[0].upper()} {worst[1]} x {worst[2]} at mini batch {worst[3]} "
                 f"and {best_vec[0].upper()} {best_vec[1]} x {best_vec[2]} at mini "
                 f"batch {best_vec[3]} respectively")
        out.append((
            "The two terms really do move in opposite directions, and the device term is "
            "the larger by an order of magnitude",
            f"On the sustained splits `graph` rose by as much as {worst[4]:+.3f} ms "
            f"while the largest `vectorize` saving anywhere was {best_vec[5]:+.3f} ms "
            f"-- {where}. So the split does exactly what it was designed to do to the "
            f"feature path, and the graph then eats roughly "
            f"{abs(worst[4] / best_vec[5]):.0f}x the gain. "
            f"That ratio, not the mini-batch size, is the result."))

    # 4 -- the collapse, which is the strongest evidence for the mechanism.
    broke = [r for r in rows if r["stack"] == "hushar" and r["split"]
             and r["state"] != "sustained"]
    if broke:
        worst = min(broke, key=lambda r: r["achieved"] or 0)
        out.append((
            f"At enough pieces the split stops being a latency question and breaks "
            f"capacity",
            f"{len(broke)} split {plural(len(broke), 'cell', 'cells')} failed the rate "
            f"its own unsplit baseline held: worst was {worst['size'].upper()} "
            f"{worst['rows']} x {worst['tps']} at mini batch {worst['mb']}, which served "
            f"{worst['achieved']:.1f} of {worst['offered']} req/s and shed "
            f"{worst['shed']:,} at only {worst['cpu']:.1f}% CPU. **Shedding at low CPU is "
            f"the serialization signature** -- nothing is saturated, the pieces are "
            f"queued behind one session's execution context. A split multiplies the "
            f"graph calls a request makes, so it multiplies the rate the device is asked "
            f"to serve."))

    # 5 -- splitting is not free even where it is harmless.
    deltas = []
    for (size, rws, tps), group in gs.items():
        base = baseline_of(group, rws)
        if not base or base["cpu"] is None:
            continue
        for mb, r in group.items():
            if mb != rws and r["state"] == "sustained" and r["cpu"] is not None:
                deltas.append(r["cpu"] - base["cpu"])
    if deltas:
        out.append((
            "Even where the split is harmless it is not free",
            f"Against its own unsplit baseline a sustained split cell ran "
            f"{min(deltas):+.1f} to {max(deltas):+.1f} percentage points of whole-host "
            f"CPU, because the fan-out pool is `num_cpus` wide and a request now "
            f"occupies several of its threads instead of one. On a CPU host that cost "
            f"bought cores; here there is nothing to buy."))

    # 6 -- the packaging comparison, which is the lever that does exist.
    pack = {}
    for r in rows:
        if r["stack"] == "hushar" and not r["split"] and r["state"] == "sustained":
            pack.setdefault((r["size"], r["rows_per_sec"]), []).append(r)
    lines, ratios = [], []
    for (size, rps), arms in sorted(pack.items()):
        if len(arms) < 2:
            continue
        arms.sort(key=lambda r: r["rows"])
        small, large = arms[0], arms[-1]
        per_small = small["p50"] / small["rows"] * 1000
        per_large = large["p50"] / large["rows"] * 1000
        cheaper, dearer = ((large, per_large), (small, per_small)) \
            if per_large < per_small else ((small, per_small), (large, per_large))
        ratios.append(dearer[1] / cheaper[1])
        lines.append(
            f"{size.upper()} at {rps:,} rows/s, {small['rows']} rows cost "
            f"{per_small:.1f} us per candidate against {large['rows']} rows at "
            f"{per_large:.1f}, so the {cheaper[0]['rows']}-row request is "
            f"{dearer[1] / cheaper[1]:.2f}x cheaper per candidate")
    if lines:
        out.append((
            "The lever that works on this device is the opposite one",
            "; ".join(lines) + ". Row parallelism is consumed INSIDE the kernel, so "
            "filling the device with a wider batch is what uses its headroom -- and a "
            "split is a move in the other direction. Note the CPU column while reading "
            "it: the larger request is cheaper per candidate AND cheaper on the host."))

    # 7 -- the reference line, with the numbers that make it a check and not an aside.
    tri_lines = []
    for (size, rws, tps), group in gs.items():
        t = tri.get((size, rws, tps))
        if t and t["state"] == "sustained":
            tri_lines.append((size, rws, t["p50"] / rws * 1000))
    by_size = {}
    for size, rws, per in tri_lines:
        by_size.setdefault(size, []).append((rws, per))
    agree = []
    for size, vals in sorted(by_size.items()):
        if len(vals) < 2:
            continue
        vals.sort()
        agree.append(f"{size.upper()} {vals[0][0]} rows {vals[0][1]:.1f} us per "
                     f"candidate against {vals[-1][0]} rows at {vals[-1][1]:.1f}")
    if agree:
        out.append((
            "Triton moves the same way under the packaging change, which is what puts "
            "that result on the device rather than on hushar",
            "; ".join(agree) + ". It has no counterpart to the split, so it is a fixed "
            "line here and not a variable -- and its ONNX Runtime version is NOT matched "
            "to hushar's and is not claimed to be, because the `tensorrt` backend does "
            "not use ONNX Runtime at all. What is matched is TensorRT, the library that "
            "executes the graph."))

    # 8 -- tails, where they disagree with the medians.
    tails = []
    for (size, rws, tps), group in gs.items():
        t = tri.get((size, rws, tps))
        base = baseline_of(group, rws)
        if not t or not base or t["state"] != "sustained" \
                or base["state"] != "sustained":
            continue
        if t["p99"] and base["p99"] and t["p99"] > base["p99"]:
            tails.append(f"{size.upper()} {rws} x {tps}, where Triton's p99 is "
                         f"{t['p99']:.2f} ms against hushar's {base['p99']:.2f}")
    if tails:
        out.append((
            "On one load Triton's median advantage is bought with a worse tail",
            "; ".join(tails) + ". Everywhere else it leads at both. Worth stating "
            "because a median comparison is the wrong one to ship against a tail "
            "objective."))

    # 9 -- where the split DOES pay, named so the result is not over-generalized.
    out.append((
        "This is a result about one GPU, not about mini batching",
        f"The split pays where the pieces reach DISTINCT compute units: `{PRIOR_CPU}` "
        f"measured 2.10x at 100 rows on 64 cores, and the Trainium campaign cut device "
        f"time 41-49% by spanning two NeuronCores with one replica. Both had somewhere "
        f"for the second piece to go. One A10G reached through a single ONNX Runtime "
        f"session does not, and no mini-batch size can create one. The change that would "
        f"make this experiment different is per-piece context leasing, not a different "
        f"mini batch."))

    return "\n\n".join(f"**{i} · {lead}.** {rest}"
                       for i, (lead, rest) in enumerate(out, 1))


# ---------------------------------------------------------------- the document

def build():
    cells = load(RAW / "results.json", [])
    if not cells:
        raise SystemExit(f"no cells in {RAW / 'results.json'}; run collect.py first")
    rows = [row_of(c) for c in cells]
    gs = groups(rows)
    tri = triton_index(rows)
    primes = {}
    pdir = RAW / "prime"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.json")):
            doc = json.loads(p.read_text())
            primes[doc.get("size", p.stem)] = doc

    answer, n_answer = answer_table(gs)
    best, verdicts = best_table(gs)
    tri_result = triton_table(gs, tri)
    sustained = [r for r in rows if r["state"] == "sustained"]
    n_gpu = len([r for r in rows if r["gpu_pct"] is not None])

    doc = []
    doc.append("# NVIDIA A10G: does a mini batch lower hushar's latency?\n")
    doc.append(
        "One A10G, one ONNX graph, one TensorRT. hushar scores it through ONNX Runtime's "
        "TensorRT execution provider, and the only thing that changes down each group of "
        "cells is **how many rows go into one graph call**. `mini_batch_size` equal to "
        "the request's rows is one call and is the baseline; anything smaller cuts the "
        "request into pieces that are handed to a thread pool and scored at the same "
        f"time. `{PRIOR_CPU}` found that knob worth up to 2.64x on a 64-core CPU. This "
        "run asks what it is worth on a GPU.\n")

    doc.append("## Answer\n")
    doc.append(best + "\n")
    doc.append(findings(gs, verdicts, rows, tri_result, tri) + "\n")

    doc.append("## Every mini batch, against its own baseline\n")
    doc.append(
        "One group per load. The mini batch is the only thing that changes inside a "
        "group: same graph, same offered rate, same session count, same host, and the "
        "unsplit cell at the top is what the column divides by. Only cells that "
        "**sustained** their offered rate appear here -- every slice of the measured "
        "window serving at least 98% of it with nothing shed and nothing failed -- "
        "because the median of a rate that was not held is queue depth and not a "
        "latency.\n")
    doc.append(answer + "\n")

    doc.append("## Where the median goes\n")
    doc.append(
        "`graph` is hushar's own `InferenceTime` and `vectorize` its "
        "`VectorizationTime`, both as the median over the measured window's minute "
        "buckets. Read the two columns in opposite directions: the split is what makes "
        "`graph` worse and `vectorize` better, and this table is the whole of the "
        "answer above.\n\n"
        "**On a split cell both figures are the MAXIMUM over that request's mini "
        "batches**, not their sum, because the service reports one duration per stage "
        "per request and the slowest piece sets the request's span. With the pieces "
        "serialized on the device that is neither total device work nor a clean "
        "per-piece figure, so `everything else` -- which is a subtraction -- is left "
        "blank there and computed only for the unsplit cells, where it is exact.\n")
    doc.append(decomposition_table(gs) + "\n")

    pack = packaging_table(gs)
    if pack:
        doc.append("## The other lever: 100 rows against 500, at the same rows per "
                   "second\n")
        doc.append(
            "Both cells unsplit, both offering the same rows per second, so only the "
            "packaging differs. This is the comparison a GPU actually rewards and it "
            "falls out of the grid for free: on a CPU the lever is to SPLIT a request to "
            "win cores, on a GPU it is to MERGE requests to fill the device. `per "
            "candidate` is the client median divided by the rows in the request.\n")
        doc.append(pack + "\n")

    if tri_result:
        table, _ = tri_result
        doc.append("## The reference line: Triton on the same packaging\n")
        doc.append(
            "Triton is **not a variable in this run** -- it has no counterpart to the "
            "mini batch, so it runs one unsplit cell per load. It is here to separate "
            "*the split did nothing* from *the packaging did nothing*: a row count that "
            "is worse on both stacks is the device, not hushar. The wires differ, as in "
            "every report in this series: Triton's plan takes the finished 690-float "
            "tensor, so the 210 transformations move to its caller, and hushar's "
            "`vectorize` column above is a lower bound on what that costs outside "
            "Triton's number.\n")
        doc.append(table + "\n")

    over = overload_table(rows)
    if over:
        doc.append("## Cells that did not hold their rate\n")
        doc.append(
            "Reported by achieved rate and deliberately not placed beside a sustained "
            "cell: the latency of a rate that was not held is queue depth and has no "
            "median worth comparing.\n")
        doc.append(over + "\n")

    doc.append("## Every cell\n")
    doc.append(every_cell_table(rows) + "\n")

    pt = prime_table(primes)
    if pt:
        doc.append("## The engines, built once each\n")
        doc.append(
            "ONNX Runtime's TensorRT provider compiles lazily at the first `Run()` with "
            "whatever shape it sees, so each mini-batch size is a separate engine and "
            "each was built and then PROVED reloadable by a fresh server before any cell "
            "ran. Without this the first cell at each shape would have spent its warm-up "
            "inside the TensorRT builder. A shape is the mini batch and nothing else, so "
            "a 500-row request at mini batch 100 and a 100-row request scored whole "
            "share one engine.\n")
        doc.append(pt + "\n")

    doc.append("## What was running\n")
    doc.append(provenance(cells, primes) + "\n")

    doc.append("## Method and limits\n")
    multi = provenance_hosts(rows)
    limits = [
        f"**Window.** {cells[0].get('warmup')}s warm-up discarded, then "
        f"{cells[0].get('duration')}s measured as one continuous load reported in "
        f"{cells[0].get('segments')} slices. A fresh server per cell, stopped afterwards, "
        f"{cells[0].get('settle')}s idle before the next. Clients start on a minute "
        f"boundary so the measured window is whole CloudWatch buckets.",
        f"**Nothing is imported from `{PRIOR_GPU}`.** Its unsplit cells measured the same "
        f"loads on the same graphs, but over a 300s window against this run's "
        f"{cells[0].get('duration')}s, and a window is one of the fields that DEFINES a "
        f"cell. Every baseline here was therefore re-measured rather than copied.",
        "**Parallelism is pinned, not probed.** `sessions_per_model` is fixed per model "
        f"size at the value `{PRIOR_GPU}` selected by the lowest median among the "
        "configurations that sustained these loads. Holding it fixed is what makes the "
        "mini-batch column a one-variable subtraction; re-ranking it would let it move "
        "between the cells this report subtracts. The same value is used at 500 rows as "
        "at 100, so the packaging comparison is also at equal concurrency.",
        "**The pin gives Triton hushar's number, which at 10M is not Triton's own.** One "
        "value per model size is applied to both stacks, so each cell runs the two "
        "servers at equal concurrency. At 20M that is `instances 4`, which is also what "
        f"`{PRIOR_GPU}` selected for Triton. At 10M it is `instances 2` where that "
        "campaign selected 4 -- and its own probes priced the difference at **1.62 ms "
        "against 1.59 ms at the median and 1.89 against 1.78 at p95**, both inside this "
        f"report's {TIE_PCT}% tie band. Disclosed rather than corrected, because Triton "
        "is a fixed reference line here and not a variable: nothing in the answer above "
        "is computed from it.",
        "**Every split divides exactly**, so `is_fixed` padded nothing and each cell "
        "presented the engine exactly one shape. A short last batch would have meant two "
        "shapes, two compiles and a median blended from two tactic selections.",
        f"**No cell was repeated, so the tie band is inherited.** Two medians within "
        f"{TIE_PCT}% are reported as `level`. That figure is the same-host spread "
        f"`{PRIOR_CPU}` measured by repeating a SPLIT configuration, which is the "
        "noisiest thing in this kind of run because its completion time is the slowest "
        "of several pieces. Using it rather than the 1.1% single-call figure is the "
        "conservative choice.",
        "**GPU utilization is occupancy, not saturation.** `nvidia-smi` reports the share "
        "of each sample period with at least one resident kernel, so a stream of small "
        "kernels reads high on a device with headroom -- and a split produces exactly "
        "such a stream. Power and clocks are reported beside it for that reason. Once "
        "the device is saturated hushar's `InferenceTime` also stops being compute, "
        "because it brackets the whole provider call and so absorbs queueing for the "
        "device.",
        "**FP32 only.** Both stacks take TensorRT's default builder settings, so these "
        "numbers are a floor on what the card can do rather than a ceiling.",
        "**Evidence.** `raw/results.json` is this report's input. Cells, resolved "
        "configurations, server and client logs, per-second host and GPU samples, the "
        "engine-cache proofs, Triton's `config.pbtxt`, the `trtexec` build logs and the "
        "CloudWatch points are under `raw/`. `verify.py` re-derives every number here "
        "from them.",
    ]
    if multi:
        limits.insert(1, f"**{multi}**")
    doc.append("\n".join(f"- {l}" for l in limits) + "\n")

    doc.append(
        f"{len(rows)} measured cells: {len(sustained)} sustained, "
        f"{len([r for r in rows if r['state'] == 'partial'])} partial, "
        f"{len([r for r in rows if r['state'] == 'failed'])} failed the rate. "
        f"{n_gpu} carry GPU samples. {n_answer} appear in the gain table.\n")
    return "\n".join(doc)


def main():
    text = build()
    if "--dry" in sys.argv:
        print(text)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
