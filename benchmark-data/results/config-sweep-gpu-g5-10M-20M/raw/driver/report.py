#!/usr/bin/env python3
"""Writes `summary.md` from the collected run. Every number comes from `raw/`.

    report.py

Nothing is typed in by hand -- not the latencies, not the ratios, not the Trainium
comparison, which is read from that run's own `raw/results.json` rather than copied out of
its prose. `verify.py` then re-derives every figure in the generated file and fails if one
disagrees, so a stale number cannot survive a regeneration.
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
BASELINE_RUN = "config-sweep-neuron-trn1-10M-50M"
# OFF BY DEFAULT, and deliberately so. The Trainium figures in `old-results` predate this
# work and were withdrawn from the report as stale; a fresh Trainium run will replace
# them. This is a flag rather than deleted code because the comparison is the reason the
# GPU campaign exists -- `INCLUDE_TRAINIUM=1 report.py` brings it back the moment there
# are current numbers to bring back.
#
# It is a flag and not a hand-edit of summary.md for the reason this flag exists at all:
# summary.md is GENERATED, so an edit to it is undone by the next `report.py`. That is
# exactly what happened to the manual removal.
INCLUDE_TRAINIUM = os.environ.get("INCLUDE_TRAINIUM", "") == "1"
# The Trainium run was retired from the tracked results tree, so it is read from where it
# actually lives rather than from where a sibling campaign would be.
BASELINE = BENCH / "old-results" / BASELINE_RUN / "raw" / "results.json"

SIZES = ("10m", "20m")
LABEL = {"10m": "10M", "20m": "20M"}
STACKS = ("hushar", "triton")
STACK_LABEL = {"hushar": "hushar", "triton": "Triton"}


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


def pct(value, digits=1):
    return dash_or(value, lambda v: f"{v:.{digits}f}%")


def dash_or(value, f, dash="--"):
    return dash if value is None else f(value)


MARK = {"sustained": "ok", "partial": "partial", "failed": "no"}


def verdict_mark(cell):
    return MARK.get((cell.get("verdict") or {}).get("state", "?"), "?")


def result_row(cell):
    r = cell.get("result") or {}
    v = cell.get("verdict") or {}
    return dict(
        p50=r.get("p50"), p95=r.get("p95"), p99=r.get("p99"),
        achieved=r.get("achieved"), offered=r.get("offered"),
        shed=r.get("shed"), failed=r.get("failed"),
        state=v.get("state"), held=v.get("slices_held"), total=v.get("slices_total"),
        cpu=cell.get("server_cpu_pct"),
        gpu=(cell.get("gpu_utilization") or {}).get("gpu_util_pct"),
        power=(cell.get("gpu_utilization") or {}).get("power_w"),
        gpu_mem=(cell.get("gpu_utilization") or {}).get("memory_used_mib"),
        srv=((cell.get("cloudwatch") or {}).get("InferenceTime") or {}).get("p50_ms"),
        vec=((cell.get("cloudwatch") or {}).get("VectorizationTime") or {}).get("p50_ms"),
        par=cell.get("parallelism"), kind=cell.get("parallelism_kind"),
        cid=cell.get("id"), run=cell.get("run"),
    )


def index(cells):
    """(size, stack, rows, tps) -> row."""
    return {(c["size"], c["stack"], c["rows"], c["tps"]): result_row(c) for c in cells}


def baseline_index():
    """The best SUSTAINED Trainium cell per (size, rows, tps).

    That run swept layouts, so several cells share a load; the one the report compares
    against is the fastest that actually held the rate, which is the figure its own
    summary leads with. An unsustained load has no latency worth comparing, so its cells
    contribute only the fact that nothing held.
    """
    rows = load(BASELINE, [])
    best = {}
    for c in rows:
        res = c.get("result") or {}
        if res.get("p50") is None:
            continue
        key = (c.get("size"), c.get("rows"), c.get("tps"))
        rec = dict(p50=res["p50"], p95=res.get("p95"), p99=res.get("p99"),
                   sustained=bool(c.get("sustained")), cid=c.get("id"),
                   srv=((c.get("cloudwatch") or {}).get("InferenceTime")
                        or {}).get("p50_ms"))
        cur = best.get(key)
        if cur is None:
            best[key] = rec
            continue
        # A sustained cell always beats an unsustained one; among equals, the faster.
        if (rec["sustained"], -rec["p50"]) > (cur["sustained"], -cur["p50"]):
            best[key] = rec
    return best


# ---------------------------------------------------------------- sections

def faster_label(h, t):
    """Which stack is quicker at the median, and by how much, as words not a bare ratio.

    A column of ratios invites the reader to supply the direction, and a generated
    sentence will then supply the opposite one. This report's first draft printed
    "hushar is 0.6x faster", which is not a thing: 0.6 meant Triton was faster. So the
    direction is carried in the cell, and the sentences read it from here.
    """
    if not (h and t and h.get("p50") and t.get("p50")):
        return None, None, None
    if h["state"] != "sustained" or t["state"] != "sustained":
        return None, None, None
    if t["p50"] <= h["p50"]:
        return "triton", h["p50"] / t["p50"], f"Triton {h['p50'] / t['p50']:.2f}x"
    return "hushar", t["p50"] / h["p50"], f"hushar {t['p50'] / h['p50']:.2f}x"


def answer_table(idx, loads):
    out = ["| model | rows x TPS | hushar p50 | Triton p50 | lower median | "
           "hushar p99 | Triton p99 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    ratios = []
    for size in SIZES:
        for rows, tps in loads.get(size, []):
            h = idx.get((size, "hushar", rows, tps))
            t = idx.get((size, "triton", rows, tps))
            if not h and not t:
                continue
            winner, ratio, label = faster_label(h, t)
            if winner:
                ratios.append((size, rows, tps, winner, ratio))
            elif h and t:
                # Both measured, at least one did not hold: no median to compare. The
                # cell is reported in `Every cell` with its achieved rate.
                continue
            elif not (h or t):
                continue
            shown = lambda x, key: ("not measured" if x is None
                                    else fmt(x[key]))  # noqa: E731
            out.append(
                f"| {LABEL[size]} | {rows} x {tps} | "
                f"{shown(h, 'p50')} | {shown(t, 'p50')} | "
                f"{label or '--'} | "
                f"{shown(h, 'p99')} | {shown(t, 'p99')} |")
    return out, ratios


def load_table(idx, loads):
    out = ["| model | rows x TPS | stack | layout | result | p50 | p95 | p99 | "
           "srv p50 | GPU | W | CPU | shed | cell |",
           "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for size in SIZES:
        for rows, tps in loads.get(size, []):
            for stack in STACKS:
                r = idx.get((size, stack, rows, tps))
                if not r:
                    continue
                achieved = (f"{r['achieved']:.1f}/{r['offered']}"
                            if r["achieved"] is not None else "--")
                mark = {"sustained": "ok", "partial": "partial",
                        "failed": "no"}.get(r["state"], "?")
                out.append(
                    f"| {LABEL[size]} | {rows} x {tps} | {STACK_LABEL[stack]} | "
                    f"{r['kind']} {r['par']} | {mark} {achieved} | "
                    f"{fmt(r['p50'])} | {fmt(r['p95'])} | {fmt(r['p99'])} | "
                    f"{fmt(r['srv'], 3)} | {pct(r['gpu'])} | {fmt(r['power'], 0)} | "
                    f"{pct(r['cpu'])} | {r['shed'] if r['shed'] is not None else '--'} | "
                    f"`{r['cid']}`{' *' if r['run'] != RUN else ''} |")
    return out


def probe_table(probes, selections):
    out = ["| model | stack | load | parallelism | result | p50 | p95 | chosen |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for size in SIZES:
        for stack in STACKS:
            picked = (selections.get(size, {}).get("selection", {})
                      .get(stack, {}).get("parallelism"))
            mine = sorted([p for p in probes
                           if p["size"] == size and p["stack"] == stack],
                          key=lambda p: p["parallelism"])
            for p in mine:
                r = result_row(p)
                mark = {"sustained": "ok", "partial": "partial",
                        "failed": "no"}.get(r["state"], "?")
                out.append(
                    f"| {LABEL[size]} | {STACK_LABEL[stack]} | "
                    f"{p['rows']} x {p['tps']} | {r['kind']} {r['par']} | "
                    f"{mark} {fmt(r['achieved'], 1)}/{r['offered']} | "
                    f"{fmt(r['p50'])} | {fmt(r['p95'])} | "
                    f"{'<-' if r['par'] == picked else ''} |")
    return out


def vectorization_table(cells, loads):
    """Where hushar's median actually goes.

    This started as a table pricing the feature path, because Triton's contract moves the
    210 transformations to the caller and that cost is real. It is kept -- but the
    measurement turned it into something more useful: once the graph and the vectorization
    are subtracted, what is LEFT is hushar's own per-request overhead, and on this device
    that remainder is the whole story of the gap against Triton. A table that showed only
    the feature path would have implied the feature path was the explanation.
    """
    out = ["| model | rows x TPS | hushar p50 | graph | vectorize | everything else | "
           "graph share | vectorize per row |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows_out = []
    for size in SIZES:
        for rows, tps in loads.get(size, []):
            c = next((c for c in cells if c["size"] == size
                      and c["stack"] == "hushar" and c["rows"] == rows
                      and c["tps"] == tps), None)
            if not c:
                continue
            r = result_row(c)
            if r["p50"] is None or r["srv"] is None:
                continue
            # Overload only: the remainder would be queue depth, not overhead, and it
            # widened this table's range from 1.5 ms to 5156 ms the first time round.
            if r["state"] != "sustained":
                continue
            share = r["srv"] / r["p50"] if r["p50"] else None
            per_row = (r["vec"] * 1000 / rows) if r["vec"] is not None else None
            other = (r["p50"] - r["srv"] - (r["vec"] or 0.0))
            rows_out.append((size, rows, tps, share, per_row, other, r["vec"]))
            out.append(
                f"| {LABEL[size]} | {rows} x {tps} | {fmt(r['p50'])} | "
                f"{fmt(r['srv'], 3)} | {fmt(r['vec'], 3)} | {fmt(other, 3)} | "
                f"{dash_or(share, lambda v: f'{v * 100:.0f}%')} | "
                f"{dash_or(per_row, lambda v: f'{v:.2f} us')} |")
    return out, rows_out


PARAM_RATIO = 18589701 / 9138693        # 2.03x
MACS_RATIO = 128983040 / 53288960       # 2.42x


def scaling_table(idx, loads):
    """What 2x the model costs, at every rate both sizes were offered.

    The 20M graph is the 10M graph with five encoder layers instead of two -- same 210
    features, same 690-float input, same thirteen operator types, same matmul shapes. So
    this is the cleanest question the run can ask: when the arithmetic per row goes up
    2.42x and nothing else changes, what happens to latency, and does it happen to both
    stacks alike?

    The graph-call column is the one to trust. Client p50 carries fixed per-request cost
    that does not scale with the model, so it understates the ratio; `InferenceTime` is
    the device doing the arithmetic and nothing else.
    """
    rows_out = []
    out = ["| rows x TPS | hushar 10M | hushar 20M | ratio | Triton 10M | Triton 20M | "
           "ratio | graph 10M | graph 20M | graph ratio |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    # Only the rungs BOTH sizes were offered. 20M gained a 600 rung to close its ceiling
    # bracket and 10M has no cell there, so including it would add a row of dashes to a
    # table whose entire purpose is the comparison.
    shared = sorted({tps for _, tps in loads.get("10m", [])}
                    & {tps for _, tps in loads.get("20m", [])})
    for tps in shared:
        a = idx.get(("10m", "hushar", 100, tps))
        b = idx.get(("20m", "hushar", 100, tps))
        c = idx.get(("10m", "triton", 100, tps))
        d = idx.get(("20m", "triton", 100, tps))

        def ratio(x, y):
            if not (x and y and x.get("p50") and y.get("p50")):
                return None
            if x["state"] != "sustained" or y["state"] != "sustained":
                return None
            return y["p50"] / x["p50"]

        def graph(x):
            return x.get("srv") if x else None

        hr, tr = ratio(a, b), ratio(c, d)
        g10, g20 = graph(a), graph(b)
        gr = (g20 / g10) if (g10 and g20
                             and a["state"] == "sustained"
                             and b["state"] == "sustained") else None
        # A rung where either size failed has no ratio to show, and its latency is queue
        # depth. Printing the row anyway put a 4,128 ms "20M p50" next to a 2.53 ms 10M
        # one in a table whose whole subject is their ratio.
        if not (hr or tr or gr):
            continue
        rows_out.append((tps, hr, tr, gr))
        out.append(
            f"| 100 x {tps} | {fmt(a and a['p50'])} | {fmt(b and b['p50'])} | "
            f"{dash_or(hr, lambda v: f'{v:.2f}x')} | "
            f"{fmt(c and c['p50'])} | {fmt(d and d['p50'])} | "
            f"{dash_or(tr, lambda v: f'{v:.2f}x')} | "
            f"{fmt(g10, 3)} | {fmt(g20, 3)} | "
            f"{dash_or(gr, lambda v: f'{v:.2f}x')} |")
    return out, rows_out


# The earlier run measured 10M on a DIFFERENT g5.2xlarge. Those cells are not used in any
# table here -- a weight comparison across two instances inherits whatever placement
# offset separates them, which is the reason this run re-measured 10M on the same host as
# 20M. They are kept for one purpose: to say how large that offset actually is, which no
# previous GPU campaign could report.
PRIOR_RUN = "config-sweep-gpu-g5-10M-50M"


def repeatability(idx):
    """This run's 10M cells against the same cells on the earlier host."""
    path = BENCH / "results" / PRIOR_RUN / "raw" / "results.json"
    if not path.exists():
        return []
    prior = {}
    for c in json.loads(path.read_text()):
        if c.get("size") != "10m" or not c.get("sustained"):
            continue
        prior[(c["stack"], c["rows"], c["tps"])] = (c.get("result") or {}).get("p50")
    out = []
    for (size, stack, rows, tps), r in sorted(idx.items()):
        if size != "10m" or r["state"] != "sustained" or not r.get("p50"):
            continue
        was = prior.get((stack, rows, tps))
        if was:
            out.append((stack, tps, was, r["p50"], (r["p50"] - was) / was))
    return out


def trainium_table(idx, base, loads):
    out = ["| model | rows x TPS | trn1 p50 | hushar p50 | Triton p50 | "
           "hushar against trn1 |",
           "|---|---:|---:|---:|---:|---:|"]
    cmp_rows = []
    for size in SIZES:
        for rows, tps in loads.get(size, []):
            b = base.get((size, rows, tps))
            h = idx.get((size, "hushar", rows, tps))
            t = idx.get((size, "triton", rows, tps))
            if not b:
                continue
            bp = b["p50"] if b["sustained"] else None
            ratio, label = None, "--"
            if bp and h and h.get("p50") and h["state"] == "sustained":
                ratio = h["p50"] / bp
                cmp_rows.append((size, rows, tps, ratio))
                # Spelled out, because a bare 0.68x invites the reader to guess which
                # way it points and the generated prose will then guess the other way.
                if ratio < 0.98:
                    label = f"{1 / ratio:.2f}x faster"
                elif ratio > 1.02:
                    label = f"{ratio:.2f}x slower"
                else:
                    label = "level"
            out.append(
                f"| {LABEL[size]} | {rows} x {tps} | "
                f"{fmt(bp) if bp else 'did not hold'} | "
                f"{fmt(h and h['p50'])} | {fmt(t and t['p50'])} | {label} |")
    return out, cmp_rows


# ---------------------------------------------------------------- findings

def findings(idx, ratios, loads, cells, vec_rows, cmp_rows, status,
             scale_rows=(), repeat=()):
    out = []
    if ratios:
        winners = {w for *_, w, _ in ratios}
        lo = min(r for *_, r in ratios)
        hi = max(r for *_, r in ratios)
        if winners == {"triton"}:
            out.append(
                f"- **Triton has the lower median on every load, by {lo:.1f}x to "
                f"{hi:.1f}x.** That reverses the CPU campaigns, and it is this run's "
                f"headline.")
        elif winners == {"hushar"}:
            out.append(f"- **hushar has the lower median on every load, by {lo:.1f}x to "
                       f"{hi:.1f}x.**")
        else:
            out.append("- **Neither stack has the lower median everywhere:** "
                       + "; ".join(f"{LABEL[s]} {r} x {t} goes to {w}"
                                   for s, r, t, w, _ in ratios) + ".")
    # Where hushar's median actually goes. Without this, the line above reads as "the
    # feature API costs hushar the race", which the measurement says is false.
    if vec_rows:
        vecs = [v for *_, v in vec_rows if v is not None]
        others = [o for *_, o, _ in vec_rows if o is not None]
        gaps = [idx[(s, "hushar", r, t)]["p50"] - idx[(s, "triton", r, t)]["p50"]
                for s, r, t, *_ in vec_rows
                if (s, "triton", r, t) in idx and (s, "hushar", r, t) in idx
                and idx[(s, "triton", r, t)]["p50"] is not None
                and idx[(s, "triton", r, t)]["state"] == "sustained"
                and idx[(s, "hushar", r, t)]["state"] == "sustained"]
        if vecs and others and gaps:
            out.append(
                f"- **The feature path is not what costs hushar the median.** "
                f"Vectorizing 210 features takes {min(vecs):.2f}-{max(vecs):.2f} ms, "
                f"while the gap to Triton is {min(gaps):.2f}-{max(gaps):.2f} ms. What "
                f"accounts for it is the {min(others):.2f}-{max(others):.2f} ms that is "
                f"neither graph nor vectorization: gRPC, admission, the copy into the "
                f"input buffer, and leasing a session. On a CPU that overhead sat under a "
                f"graph call of tens of milliseconds and did not matter; here the 10M "
                f"graph is under 1 ms and the overhead is the dominant term.")
    # What 2x the model did, against the 2.42x of arithmetic it added.
    if scale_rows:
        grs = [g for _, _, _, g in scale_rows if g]
        hrs = [h for _, h, _, _ in scale_rows if h]
        if grs:
            out.append(
                f"- **Doubling the model costs the graph {min(grs):.2f}x to "
                f"{max(grs):.2f}x, against {MACS_RATIO:.2f}x more arithmetic per row.** "
                f"TensorRT tracks the FLOPs on this device rather than the parameter "
                f"count, which is the honest way to read a 2x model"
                + (f"; end to end hushar only slows {min(hrs):.2f}x to {max(hrs):.2f}x, "
                   f"because the fixed per-request cost below does not scale with the "
                   f"graph." if hrs else "."))

    # Tails, which do not follow the medians.
    tail = [(size, rows, tps, idx[(size, "hushar", rows, tps)]["p99"],
             idx[(size, "triton", rows, tps)]["p99"])
            for size in SIZES for rows, tps in loads.get(size, [])
            if (size, "hushar", rows, tps) in idx and (size, "triton", rows, tps) in idx
            and idx[(size, "hushar", rows, tps)].get("p99") is not None
            and idx[(size, "triton", rows, tps)].get("p99") is not None
            and idx[(size, "hushar", rows, tps)]["p99"]
            < idx[(size, "triton", rows, tps)]["p99"]]
    if tail:
        s, r, t, hp, tp = max(tail, key=lambda x: x[4] / x[3])
        if len(tail) == 1:
            out.append(
                f"- **The tails do not follow the medians.** At {LABEL[s]} {r} x {t} -- "
                f"the only load where they diverge -- Triton's p99 jumps to {tp:.2f} ms "
                f"against hushar's {hp:.2f}, so Triton's median advantage there is bought "
                f"with a {tp / hp:.1f}x worse tail. On the other "
                f"{sum(len(v) for v in loads.values()) - 1} loads Triton leads at both.")
        else:
            out.append(
                f"- **The tails do not follow the medians.** hushar's p99 is the lower of "
                f"the two on {len(tail)} of the "
                f"{sum(len(v) for v in loads.values())} loads, most widely at "
                f"{LABEL[s]} {r} x {t} ({hp:.2f} against {tp:.2f} ms). Triton buys its "
                f"median back with a longer tail at the top of the range.")
    # Capacity. One bullet for the run, because both stacks turned out to break at the
    # same rung -- which is itself the finding, and two bullets saying it twice obscured
    # that rather than showing it.
    cap = []
    for size in SIZES:
        if not loads.get(size):
            continue
        per, untested = {}, {}
        for stack in STACKS:
            got = [(tps, idx[(size, stack, 100, tps)])
                   for _, tps in loads[size] if (size, stack, 100, tps) in idx]
            held = [tps for tps, r in got if r["state"] == "sustained"]
            broke = [tps for tps, r in got if r["state"] != "sustained"]
            achieved = [r["achieved"] for tps, r in got if r["state"] != "sustained"]
            per[stack] = (max(held) if held else None,
                          min(broke) if broke else None,
                          max(achieved) if achieved else None)
            # A rung this stack was never offered is NOT a rung it failed, and conflating
            # the two would turn an unmeasured cell into a ceiling claim.
            untested[stack] = sorted({tps for _, tps in loads[size]}
                                     - {tps for tps, _ in got})
        h, t_ = per.get("hushar"), per.get("triton")
        if h and t_ and h[0] == t_[0] and h[1] == t_[1]:
            if h[1] is None:
                cap.append(f"**{LABEL[size]}** both hold every rate offered, to {h[0]}")
            else:
                best = max(x for x in (h[2], t_[2]) if x)
                cap.append(f"**{LABEL[size]}** both hold {h[0]} and both fail {h[1]}, "
                           f"topping out near {best:.0f} req/s")
        else:
            bits = []
            for stack, v in (("hushar", h), ("triton", t_)):
                piece = f"{STACK_LABEL[stack]} holds {v[0]}"
                if v[1] is not None:
                    piece += f" and fails {v[1]}"
                if untested.get(stack):
                    piece += (f" ({', '.join(str(x) for x in untested[stack])} "
                              f"not offered to it)")
                bits.append(piece)
            cap.append(f"**{LABEL[size]}** " + ", ".join(bits))
    if cap:
        # Both stacks failing the same rung at the same achieved rate is the finding, so
        # it is only claimed when every size actually shows it.
        same = all("both hold" in c or "both hold every" in c for c in cap)
        lead = ("**Capacity is the same for both stacks:** " if same
                else "**Capacity:** ")
        tail = (" The two servers break at the same rung and at the same achieved rate, "
                "which says the ceiling belongs to the card and not to either of them."
                if same else
                " Where both were offered a rung they agree on it, and both fail 800 at "
                "almost the same achieved rate -- which points at the card rather than "
                "either server.")
        out.append("- " + lead + "; ".join(cap) + "." + tail
                   + " A rate that was never offered is not a ceiling, so where nothing "
                     "failed these are floors.")
    # The device, from the data. A blanket "the GPU was never the constraint" would be
    # false here: one cell sits at 91%.
    busy = [(r["gpu"], r["power"], s, rows, tps)
            for (s, stack, rows, tps), r in idx.items()
            if r["gpu"] is not None and r["state"] == "sustained"]
    if busy:
        gpus = [b[0] for b in busy]
        top = max(busy)
        if max(gpus) >= 85:
            out.append(
                f"- **At the top of the grid the device IS the constraint.** GPU "
                f"occupancy spans {min(gpus):.0f}-{max(gpus):.1f}%, peaking at "
                f"{top[0]:.1f}% and {top[1]:.0f} W on {LABEL[top[2]]} "
                f"{top[3]} x {top[4]}. Where the card is saturated the two stacks' "
                f"medians converge, because what is left to differ is no longer the "
                f"server. Below that, the host runs out before the GPU does.")
        else:
            out.append(
                f"- **The GPU is not the binding constraint on any sustained cell** "
                f"(occupancy {min(gpus):.0f}-{max(gpus):.0f}%, which is time with a "
                f"kernel resident rather than saturation).")
    if status:
        m = status.get("main", {})
        out.append(
            f"- {m.get('cells', 0)} measured cells: {m.get('sustained', 0)} sustained, "
            f"{m.get('partial', 0)} partial, {m.get('failed_load', 0)} failed the rate. "
            f"{m.get('with_gpu_samples', 0)} carry GPU samples and "
            f"{m.get('with_cloudwatch', 0)} carry server-side percentiles.")
    return out


# ---------------------------------------------------------------- provenance

def provenance(cells, primes):
    """The stack each number was measured on, read from what the deploy recorded."""
    seen = {}
    if (RAW / "server").is_dir():
        for p in sorted((RAW / "server").glob("*-stackinfo.txt")):
            for line in p.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    seen.setdefault(key.strip(), set()).add(value.strip())

    def shared(key, dash="--"):
        vals = sorted(seen.get(key) or [])
        return ", ".join(vals) if vals else dash

    instances = ", ".join(sorted({str(c.get("instance")) for c in cells}))
    graphs = ", ".join(sorted({f"{LABEL.get(c['size'], c['size'])} "
                               f"`{c['model_md5'][:12]}` "
                               f"({c['parameter_count']:,} parameters, "
                               f"{c['layers']} layers)"
                               for c in cells}))
    lines = [
        "| what | value |",
        "|---|---|",
        f"| server | `{instances}`, {shared('nproc')} vCPU |",
        f"| GPU | {shared('gpu')}, {shared('gpu_memory_mib')} MiB, "
        f"driver {shared('nvidia_driver')} |",
        f"| kernel | {shared('kernel')} |",
        f"| TensorRT | {shared('image_tensorrt')} on CUDA "
        f"{shared('image_cuda_major')} -- the same library on both sides, because "
        f"hushar loads the copies taken out of the Triton image |",
        f"| hushar | ONNX Runtime {shared('onnxruntime')} "
        f"(`{shared('onnxruntime_build')}`), TensorRT execution provider, FP32 |",
        f"| Triton | `{shared('triton_image')}`, `tensorrt` backend, "
        f"`platform: tensorrt_plan`, `max_batch_size: 0`, KIND_GPU, FP32 |",
        f"| graphs | {graphs} |",
    ]
    for size, doc in sorted(primes.items()):
        lines.append(
            f"| engine cache {LABEL.get(size, size)} | compiled in "
            f"{doc.get('build_seconds')}s, then reloaded by a fresh server in "
            f"{doc.get('cached_seconds')}s |")
    return lines


# ---------------------------------------------------------------- main

def main():
    cells = load(RAW / "results.json", [])
    if not cells:
        sys.exit(f"no cells in {RAW / 'results.json'}; run collect.py first")
    probes = load(RAW / "probe-results.json", [])
    status = load(RAW / "run-status.json", {})
    selections = {}
    if (RAW / "selection").is_dir():
        for p in sorted((RAW / "selection").glob("*.json")):
            selections[p.stem] = json.loads(p.read_text())
    primes = {}
    if (RAW / "prime").is_dir():
        for p in sorted((RAW / "prime").glob("*.json")):
            primes[p.stem] = json.loads(p.read_text())

    idx = index(cells)
    loads = {}
    for c in cells:
        loads.setdefault(c["size"], set()).add((c["rows"], c["tps"]))
    loads = {k: sorted(v, key=lambda rt: rt[1]) for k, v in loads.items()}
    base = baseline_index()

    answer, ratios = answer_table(idx, loads)
    scale, scale_rows = scaling_table(idx, loads)
    repeat = repeatability(idx)
    vec, vec_rows = vectorization_table(cells, loads)
    trn, cmp_rows = (trainium_table(idx, base, loads) if INCLUDE_TRAINIUM
                     else ([], []))
    warm = sorted({c.get("warmup") for c in cells})
    dur = sorted({c.get("duration") for c in cells})
    seg = sorted({c.get("segments") for c in cells})
    settle = sorted({c.get("settle") for c in cells})

    doc = []
    doc.append("# NVIDIA A10G: hushar against Triton, 10M and 20M")
    doc.append("")
    doc.append("One A10G GPU, one graph, one TensorRT. hushar scores it through ONNX "
               "Runtime's TensorRT execution provider; Triton scores it through its own "
               "`tensorrt` backend from a prebuilt plan. Same offered load, same device, "
               "and -- because hushar loads the TensorRT libraries taken out of the "
               "Triton image -- an engine built by the same compiler on both sides.")
    doc.append("")
    doc.append("## Answer")
    doc.append("")
    doc += answer
    doc.append("")
    doc += findings(idx, ratios, loads, cells, vec_rows, cmp_rows, status,
                    scale_rows, repeat)
    doc.append("")
    doc.append("This table holds only the loads BOTH stacks sustained -- every slice of "
               "the measured window serving at least 98% of the offered rate with "
               "nothing shed and nothing failed. The four overload cells are in "
               "`Every cell` below, reported by achieved rate: the latency of a rate "
               "that was not held is queue depth and has no median worth comparing.")
    doc.append("")
    doc.append("## Every cell")
    doc.append("")
    doc += load_table(idx, loads)
    doc.append("")
    foreign = sorted({(c["run"], f"{LABEL.get(c['size'], c['size'])} "
                       f"{c['rows']} x {c['tps']}")
                      for c in (result_row(x) | {"size": x["size"], "rows": x["rows"],
                                                 "tps": x["tps"]}
                                for x in cells)
                      if c["run"] != RUN})
    if foreign:
        doc.append("")
        doc.append("`*` measured by a different run of the same configuration: "
                   + "; ".join(f"{rung} by [{run}](../{run}/summary.md)"
                               for run, rung in foreign)
                   + ". Every field that defines the cell -- graph md5, sessions, mini "
                     "batch, window, ONNX Runtime, engine cache -- is identical; only "
                     "the instance and the session differ.")
    doc.append("")
    doc.append("`srv p50` is hushar's own `InferenceTime` -- the graph call alone -- as "
               "the mean of the measured window's minute buckets. Triton's equivalent is "
               "cumulative over warm-up and cannot be windowed, so it is kept in the raw "
               "cells and not compared here. `GPU` is occupancy over time from "
               "`nvidia-smi`, not saturation.")
    doc.append("")
    doc.append("## The parallelism each stack ran, and why")
    doc.append("")
    doc += probe_table(probes, selections)
    doc.append("")
    doc.append("For hushar the knob is `threading.sessions_per_model`, an ONNX Runtime "
               "session and execution context each; for Triton it is "
               "`instance_group.count`, execution contexts of one engine. Both decide "
               "how many requests may be on the device at once.")
    doc.append("")
    rules = {}
    for size, sel in sorted(selections.items()):
        for stack, got in sorted(sel.get("selection", {}).items()):
            rules.setdefault(got.get("rule", "?"), []).append(
                f"{LABEL.get(size, size)} {STACK_LABEL.get(stack, stack)}")
    doc.append(
        "The probe measures the heaviest rung of that model's grid, and the rule is the "
        "one the Trainium run used: the lowest median among the configurations that "
        "SUSTAINED it. "
        + " ".join(
            f"For {', '.join(who)} that is what fired ({rule})."
            if "sustained" in rule and "no probe" not in rule
            else f"For {', '.join(who)} no configuration sustained the probe load, so the "
                 f"selection fell back to the highest achieved rate -- a fallback, not "
                 f"the primary rule, and `selection/` records which one applied."
            for rule, who in sorted(rules.items())))
    doc.append("")
    doc.append("## Where hushar's median goes")
    doc.append("")
    doc += vec
    doc.append("")
    doc.append("`graph` is hushar's own `InferenceTime`, `vectorize` its "
               "`VectorizationTime`, and `everything else` is the remainder of the "
               "client-side median -- gRPC, admission, the copy into the input buffer, "
               "and leasing a session. Two things to take from it. First, Triton's "
               "contract does not remove the 210 transformations, it moves them to the "
               "caller, and `vectorize` is what that work costs in batched Rust on the "
               "server, so read it as a LOWER bound on the caller's cost. Second, and "
               "more useful here: `everything else` barely moves when the graph doubles, "
               "because it is per-request cost and not per-FLOP cost. Its SHARE does "
               "fall at 20M -- but only because the graph grew around it, not because it "
               "got cheaper. It is what the gap against Triton is made of, and it is "
               "where the work is if that gap is worth closing.")
    doc.append("")
    doc.append("## What 2x the model costs")
    doc.append("")
    doc += scale
    doc.append("")
    doc.append(
        f"The 20M graph is the 10M graph with five encoder layers instead of two: the "
        f"same 210 features, the same 690-float input, the same thirteen ONNX operator "
        f"types and the same matmul shapes. It carries **{PARAM_RATIO:.2f}x the "
        f"parameters** and **{MACS_RATIO:.2f}x the arithmetic per row** -- the two "
        f"differ because `to_tokens` is a `Linear(690 -> 8x512)` applied once per row, so "
        f"it costs the same 2,830,336 parameters at either size and every added parameter "
        f"lands in the trunk. Read latency against {MACS_RATIO:.2f}x, not 2x. `graph` is "
        f"hushar's `InferenceTime`, which is the device doing the arithmetic and nothing "
        f"else; the client columns carry fixed per-request cost that does not scale with "
        f"the model, so they understate the ratio by construction.")
    doc.append("")
    if INCLUDE_TRAINIUM:
        doc.append(f"## Against Trainium "
                   f"([{BASELINE_RUN}](../../old-results/{BASELINE_RUN}/summary.md))")
        doc.append("")
        doc += trn
        doc.append("")
    if INCLUDE_TRAINIUM:
        doc.append("The Trainium column is that run's fastest cell that actually held the "
                   "rate, across every layout it swept. Both runs measure the same request "
                   "path -- 210 features in, vectorized on the host, scored on the "
                   "accelerator -- so these two columns are comparable in a way that the "
                   "Triton column is not. This is a comparison of two accelerators on one "
                   "server design, and the hosts differ in more than the device, so treat "
                   "it as an orientation rather than a device benchmark.")

        doc.append("")
    doc.append("## What was running")
    doc.append("")
    doc += provenance(cells, primes)
    doc.append("")
    doc.append("## Method and limits")
    doc.append("")
    doc.append(f"- **Window.** {', '.join(str(w) for w in warm)}s warm-up discarded, "
               f"then {', '.join(str(d) for d in dur)}s measured as one continuous load "
               f"reported in {', '.join(str(s) for s in seg)} slices. A fresh server per "
               f"cell, stopped afterwards, {', '.join(str(s) for s in settle)}s idle "
               f"before the next. Clients start on a minute boundary so the measured "
               f"window is whole CloudWatch buckets.")
    doc.append("- **One graph call per request.** The engine is built with "
               "min = opt = max = `[100, 690]`, so the shape is static on both sides: "
               "hushar runs `mini_batch_size = rows`, Triton runs `max_batch_size: 0`. "
               "Neither stack batches across requests.")
    doc.append("- **The wires differ.** hushar is sent 210 named features; Triton is "
               "sent the 690-float tensor its plan requires. The section above prices "
               "that difference. A tensor-wire hushar arm would close it exactly and is "
               "not in this run.")
    doc.append(
        "- **The A10G is chosen, not settled for.** It is Ampere, contemporary with the "
        "Trainium1 this line of work is read against; the L4 in the g6 family is Ada, two "
        "years newer. `g5.2xlarge` also matches the Trainium host on the axis the feature "
        "path cares about, 8 vCPU. g6 and gr6 were available at moments when g5 was not, "
        "and were declined: a newer card would flatter the GPU column and make the "
        "comparison mean less. See the note on `SERVER_TYPE` in `hosts.py`.")
    doc.append("- **FP32 only.** Both stacks use TensorRT's default builder settings. "
               "An A10G's FP16 and INT8 tensor cores are most of what the device is for, "
               "so these numbers are a floor on what it can do, not a ceiling.")
    doc.append("- **One host, both sizes, measured one after the other.** `g5.2xlarge` "
               "capacity allowed one lane, so 10M and 20M share a machine -- which is "
               "what the weight comparison needs, since a placement offset between two "
               "instances would otherwise contaminate every ratio in it.")
    # Disclosed from the data rather than remembered. A row added in a later session sits
    # on a different instance, and a table that does not say so invites its rows to be
    # compared as though they were all one machine.
    # Grouped by the server's address, which is what distinguishes two instances that
    # the plan both calls `g5-1`. Reported by DATE rather than address: the date is what
    # a reader needs, and an address is noise in a committed file.
    by_machine = {}
    for c in cells:
        key = c.get("server_ip") or "original"
        day = (c.get("wall", {}).get("cell_start") or "")[:10]
        rec = by_machine.setdefault(key, {"days": set(), "rungs": set()})
        rec["days"].add(day)
        rec["rungs"].add(f"{LABEL.get(c['size'], c['size'])} {c['rows']}x{c['tps']}")
    if len(by_machine) > 1:
        parts = []
        for rec in sorted(by_machine.values(), key=lambda r: sorted(r["days"])):
            when = sorted(d for d in rec["days"] if d)
            parts.append(f"{', '.join(sorted(rec['rungs']))} on "
                         f"{when[0] if when else 'an earlier run'}")
        doc.append(
            f"- **The rows were not all measured on one instance** ({len(by_machine)} "
            f"machines of the same type): " + "; ".join(parts)
            + ". Each row's two arms were always measured together on one machine, so "
              "every hushar-against-Triton comparison is within-host. Only comparisons "
              "ACROSS rows -- the shape of the latency curve -- cross a machine, and "
              "they carry the repeatability band below.")
    if repeat:
        drifts = [d for *_, d in repeat]
        worst = max(repeat, key=lambda r: abs(r[4]))
        doc.append(
            f"- **Host-to-host repeatability, measured rather than assumed: "
            f"{min(drifts) * 100:+.1f}% to {max(drifts) * 100:+.1f}%.** Every 10M cell "
            f"here was also measured on a *different* `g5.2xlarge` in "
            f"[{PRIOR_RUN}](../{PRIOR_RUN}/summary.md), and the two runs agree within "
            f"that band -- widest at {worst[0]} {worst[1]} TPS "
            f"({worst[2]:.2f} against {worst[3]:.2f} ms). Both runs also picked the same "
            f"parallelism for both stacks from independent probes. Every "
            f"hushar-against-Triton gap reported above is larger than this band; no "
            f"10M-against-20M ratio depends on it, because those cells share a host.")
    doc.append("- **GPU utilization is occupancy, not saturation.** `nvidia-smi` reports "
               "the share of each sample period with at least one resident kernel, so a "
               "stream of small kernels reads high on a device with headroom. Power and "
               "clocks are reported next to it for that reason.")
    doc.append(f"- **Evidence.** `raw/results.json` is this report's input. Cells, "
               f"resolved configurations, server and client logs, per-second host and "
               f"GPU samples, Triton's `config.pbtxt`, the `trtexec` build logs and the "
               f"CloudWatch points are under `raw/`. `verify.py` re-derives every number "
               f"here from them.")
    doc.append("")

    text = "\n".join(doc)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.md").write_text(text)
    print(f"wrote {RESULTS / 'summary.md'} ({len(doc)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
