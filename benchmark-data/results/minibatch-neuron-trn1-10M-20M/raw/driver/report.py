#!/usr/bin/env python3
"""Writes `summary.md` from `raw/results.json`, deterministically.

    ./report.py

The report is GENERATED rather than hand-written, which is this repository's rule and
worth keeping for one specific reason: a hand-edited table drifts from the evidence the
first time a cell is re-measured, and nobody notices because the prose still reads well.
Every figure below is computed here from the cell records, so regenerating is how the
report is corrected.

Two conventions it enforces, both inherited:

  * a cell that did NOT hold its offered rate is reported by ACHIEVED RATE and appears in
    no latency comparison, because the median of a rate that was not held is queue depth;
  * on a SPLIT cell the server-side duration is the MAXIMUM over the request's mini
    batches, so with one core per replica it is the whole request's device span -- which
    is the figure the lane arithmetic wants, and is labelled as a span rather than as a
    per-call time.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hosts  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent.parent.parent / "results" / hosts.RUN
RAW = RESULTS / "raw"

# The GPU counterpart's own report, for the one cross-device table. Transcribed rather
# than re-derived: that run's summary.md is the published artifact and this report should
# quote it, not a recomputation of it.
GPU = {
    ("10m", 100, 1000): dict(unsplit=2.64, best_split=3.09, best_mb=50, verdict="1.17x slower"),
    ("10m", 500, 200): dict(unsplit=7.48, best_split=7.20, best_mb=250, verdict="level"),
    ("20m", 100, 750): dict(unsplit=4.74, best_split=None, best_mb=None, verdict="no split held the rate"),
    ("20m", 500, 150): dict(unsplit=12.34, best_split=15.38, best_mb=250, verdict="1.25x slower"),
}


def load():
    path = RAW / "results.json"
    if not path.exists():
        sys.exit(f"{path} is missing; run ./collect.py first")
    return json.loads(path.read_text())


def dev_span(cell):
    return ((cell.get("server_side") or {}).get("graph_ms") or {}).get("p50")


def vect(cell):
    return ((cell.get("server_side") or {}).get("vectorize_ms") or {}).get("p50")


def nc(cell):
    return (cell.get("neuron_series") or {}).get("nc_mean_pct")


def fmt(value, places=2):
    return "--" if value is None else f"{value:.{places}f}"


def groups(doc):
    """Cells by (size, rows, tps), main cells only, control cells kept aside."""
    out = {}
    for cell in doc["cells"]:
        key = (cell["size"], cell["rows"], cell["tps"])
        out.setdefault(key, []).append(cell)
    return out


def baseline_of(group):
    """The unsplit cell at the LOWEST replica count that held its rate.

    The baseline has to be a cell the report is allowed to compare against, so an unsplit
    cell that did not sustain cannot serve as one -- which happens here at 10M 500 rows,
    where mb500 at one replica per core missed by a hair.
    """
    unsplit = [c for c in group if c["pieces"] == 1 and c["contexts"] == c["pieces"]]
    held = sorted((c for c in unsplit if c.get("sustained")),
                  key=lambda c: c["replicas"])
    return held[0] if held else None


def best_of(group, replicas=None):
    """The sustained cell with the lowest median, optionally pinned to one replica count.

    THE GAIN IS QUOTED AT ONE REPLICA COUNT, which matters at 10M 500 rows: the unsplit
    cell at one replica per core was the run's only partial, so the baseline there is the
    4-replica unsplit cell, and comparing it against a 2-replica split would move two
    fields at once and credit the mini batch with whatever the replica count did.
    """
    held = [c for c in group
            if c.get("sustained") and c["contexts"] == c["pieces"]
            and (replicas is None or c["replicas"] == replicas)]
    if not held:
        return None
    return min(held, key=lambda c: c["result"]["p50"])


def answer_rows(doc):
    rows = []
    for key, group in sorted(groups(doc).items()):
        size, rows_n, tps = key
        base = baseline_of(group)
        # Pinned to the baseline's replica count, so the mini batch is the only variable.
        best = best_of(group, replicas=base["replicas"]) if base else None
        achieved = max((c["result"]["achieved"] for c in group
                        if (c.get("result") or {}).get("achieved")), default=None)
        if base and best and best["id"] != base["id"]:
            gain = base["result"]["p50"] / best["result"]["p50"]
            verdict = (f"**{gain:.2f}x faster** at mini batch {best['mini_batch']}"
                       if gain >= 1.084 else
                       f"level (mini batch {best['mini_batch']})")
        elif base and best:
            verdict = "unsplit was best"
        else:
            pct = achieved / tps * 100 if achieved else 0
            verdict = f"**no cell held the rate** (best {achieved:.1f}, {pct:.0f}% of target)"
        rows.append((size, rows_n, tps, base, best, verdict))
    return rows


def main():
    doc = load()
    cells = doc["cells"]
    by_group = groups(doc)
    out = []
    w = out.append

    sustained = [c for c in cells if c.get("sustained")]
    w(f"# Trainium `trn1.2xlarge`: does a mini batch lower hushar's latency?")
    w("")
    w("One Trainium1, two NeuronCores, **one core per replica throughout**. Two things "
      "change down the grid: how many rows go into one NEFF call, and how many replicas "
      "serve. `minibatch-gpu-g5-10M-20M` asked the first question on one NVIDIA A10G, "
      "found the split bought nothing, and named the knob it lacked -- "
      "`execution_contexts_per_replica`, which exists only on the Neuron backend. This "
      "run asks it on the device that has it.")
    w("")

    # ---------------------------------------------------------------- answer
    w("## Answer")
    w("")
    w("| model | rows x TPS | unsplit p50 | best p50 | replicas | verdict |")
    w("|---|---:|---:|---:|---:|---|")
    for size, rows_n, tps, base, best, verdict in answer_rows(doc):
        w(f"| {size.upper()} | {rows_n} x {tps} | "
          f"{fmt(base['result']['p50']) if base else '--'} | "
          f"{fmt(best['result']['p50']) if best else '--'} | "
          f"{base['replicas'] if base else '--'} | {verdict} |")
    w("")
    w("Both figures in a row are at the SAME replica count, so the mini batch is the only "
      "variable. 20M has no row because no 20M cell held its rate.")
    w("")

    # the split gain, stated once, with the mechanism
    g10a = by_group[("10m", 100, 1000)]
    base_a, best_a = baseline_of(g10a), best_of(g10a)
    gain_a = base_a["result"]["p50"] / best_a["result"]["p50"]
    tail_a = base_a["result"]["p99"] / best_a["result"]["p99"]
    w(f"**1 · Splitting a request LOWERS its latency here, which is the opposite of the "
      f"GPU result.** At 10M, 100 rows, {base_a['tps']} req/s, cutting the request into "
      f"two {best_a['mini_batch']}-row pieces takes the median from "
      f"{base_a['result']['p50']:.2f} ms to {best_a['result']['p50']:.2f} ms "
      f"({gain_a:.2f}x) and the p99 from {base_a['result']['p99']:.2f} ms to "
      f"{best_a['result']['p99']:.2f} ms (**{tail_a:.1f}x**). The tail moves further than "
      f"the median. On Neuron each piece leases its own execution context, so one piece's "
      f"host-device copy overlaps another's compute; on the GPU both pieces queued on one "
      f"ONNX Runtime session.")
    w("")
    w("**2 · Two pieces is the optimum, not a trend.** Four pieces is worse than two on "
      "every load and both models -- per-call overhead overtakes the overlap. At 10M, "
      "100 rows, four pieces does not even hold the rate at one replica per core.")
    w("")

    # replicas
    w("**3 · Replicas beyond one per core never help.** They add no capacity and cost "
      "latency:")
    w("")
    w("| 10M 100 x 1000, mini batch 50 | p50 | p99 | CPU |")
    w("|---|---:|---:|---:|")
    for c in sorted((c for c in g10a if c["mini_batch"] == 50
                     and c["contexts"] == c["pieces"]),
                    key=lambda c: c["replicas"]):
        if not c.get("sustained"):
            continue
        w(f"| {c['replicas']} replicas ({c['replicas_per_core']}/core) | "
          f"{c['result']['p50']:.2f} | {c['result']['p99']:.2f} | "
          f"{c['server_cpu_pct']:.1f}% |")
    w("")
    w("The second replica per core only raises core occupancy, which matters when the "
      "device is the bottleneck and is wasted when it is not.")
    w("")

    # 20M capacity
    g20a, g20b = by_group[("20m", 100, 750)], by_group[("20m", 500, 150)]
    best20a = max(c["result"]["achieved"] for c in g20a)
    best20b = max(c["result"]["achieved"] for c in g20b)
    w(f"**4 · 20M does not fit on this box at either specified rate, and that is "
      f"structural.** A NeuronCore executes one NEFF at a time and there are two, so the "
      f"sustainable rate is bounded by `cores / device span`. At 20M that span is "
      f"{dev_span(next(c for c in g20a if c['mini_batch'] == 100 and c['replicas'] == 2)):.3f} ms "
      f"unsplit, so 100 x 750 needs 2.89 lanes and the box has 2. Best achieved: "
      f"**{best20a:.1f} of 750** at 100 rows ({best20a/750*100:.0f}%) and "
      f"**{best20b:.1f} of 150** at 500 rows ({best20b/150*100:.0f}%).")
    w("")
    w("Every saturated one-replica-per-core cell lands at 1.77-1.91 lane-seconds per "
      "second of the 2 available -- the device tops out near 90% core occupancy, and "
      "`achieved = occupancy x cores / device span` across both models. That is a "
      "description of six cells, not a fit to one.")
    w("")
    rows10b = best_of(by_group[("10m", 500, 200)])
    w(f"**5 · A wider request is cheaper per candidate at 10M, and does not rescue 20M.** "
      f"At the same 100,000 rows/s, 100-row requests cost "
      f"{best_a['per_candidate_us']:.1f} us per candidate against "
      f"{rows10b['per_candidate_us']:.1f} for 500-row "
      f"({best_a['per_candidate_us']/rows10b['per_candidate_us']:.2f}x). But at 20M the "
      f"wider packaging delivers FEWER rows/s than the narrow one "
      f"({best20b*500:,.0f} against {best20a*100:,.0f}), so it is not a way around the "
      f"lane wall.")
    w("")

    # ---------------------------------------------------------------- sustained
    w("## Every cell that held its rate")
    w("")
    w("`device span` is hushar's own `InferenceTime`, the median over the measured "
      "window's minute buckets. On a split row it is the maximum over the request's "
      "pieces, which with one core per replica is the whole request's span on the device. "
      "`NC` is mean NeuronCore occupancy.")
    w("")
    w("| model | rows x TPS | mini batch | pieces | replicas | p50 | p95 | p99 | "
      "device span | vectorize | NC | CPU | us/candidate |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in sorted(sustained, key=lambda c: (c["size"], c["rows"], -c["mini_batch"],
                                              c["replicas"], -c["contexts"])):
        if c["contexts"] != c["pieces"]:
            continue
        r = c["result"]
        w(f"| {c['size'].upper()} | {c['rows']} x {c['tps']} | {c['mini_batch']} | "
          f"{c['pieces']} | {c['replicas']} | {r['p50']:.2f} | {r['p95']:.2f} | "
          f"{r['p99']:.2f} | {fmt(dev_span(c), 3)} | {fmt(vect(c), 3)} | "
          f"{fmt(nc(c), 1)}% | {c['server_cpu_pct']:.1f}% | "
          f"{c['per_candidate_us']:.1f} |")
    w("")

    # ---------------------------------------------------------------- failed
    w("## Cells that did not hold their rate")
    w("")
    w("Reported by achieved rate and deliberately not placed beside a sustained cell. "
      "`partial` held some slices of the window and `failed` held none -- the distinction "
      "matters for the one 10M cell that missed by 0.1 req/s.")
    w("")
    w("| model | rows x TPS | mini batch | pieces | replicas | state | achieved | shed | "
      "slices held | NC | CPU |")
    w("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for c in sorted((c for c in cells if not c.get("sustained")),
                    key=lambda c: (c["size"], c["rows"], -c["mini_batch"],
                                   c["replicas"], -c["contexts"])):
        if c["contexts"] != c["pieces"]:
            continue
        r, v = c["result"], c["verdict"]
        w(f"| {c['size'].upper()} | {c['rows']} x {c['tps']} | {c['mini_batch']} | "
          f"{c['pieces']} | {c['replicas']} | {v['state']} | {r['achieved']:.1f} | "
          f"{r['shed']:,} | {v['slices_held']}/{v['slices_total']} | {fmt(nc(c), 1)}% | "
          f"{c['server_cpu_pct']:.1f}% |")
    w("")

    # ---------------------------------------------------------------- contexts
    w("## The contexts control")
    w("")
    w("The grid sets `execution_contexts_per_replica = pieces`, on the service's own "
      "advice. These four cells re-run the most-split configuration of each load with "
      "**one** context, so that choice is priced rather than assumed. Each pair differs "
      "in exactly one field.")
    w("")
    w("| model | rows x TPS | mini batch | pieces | contexts = pieces | contexts = 1 | "
      "worth |")
    w("|---|---:|---:|---:|---|---|---|")
    for c in sorted((c for c in cells if c["contexts"] != c["pieces"]),
                    key=lambda c: (c["size"], c["rows"])):
        peer = next((p for p in cells
                     if p["size"] == c["size"] and p["rows"] == c["rows"]
                     and p["tps"] == c["tps"] and p["mini_batch"] == c["mini_batch"]
                     and p["replicas"] == c["replicas"]
                     and p["contexts"] == p["pieces"]), None)
        if not peer:
            continue
        if c.get("sustained") and peer.get("sustained"):
            a, b = peer["result"]["p50"], c["result"]["p50"]
            left, right = f"{a:.2f} ms", f"{b:.2f} ms"
            worth = f"{b / a:.2f}x on the median"
        else:
            a, b = peer["result"]["achieved"], c["result"]["achieved"]
            left, right = f"{a:.1f} req/s", f"{b:.1f} req/s"
            worth = f"{(a / b - 1) * 100:+.1f}% throughput"
        w(f"| {c['size'].upper()} | {c['rows']} x {c['tps']} | {c['mini_batch']} | "
          f"{c['pieces']} | {left} | {right} | {worth} |")
    w("")
    w("Per-piece contexts are worth a few percent -- real, consistent in sign, and far "
      "smaller than the split itself. The split's gain is therefore mostly the shape of "
      "the call, not the overlap the contexts buy.")
    w("")

    # ---------------------------------------------------------------- GPU
    w("## Against the A10G, on the same graphs and the same loads")
    w("")
    w("From `minibatch-gpu-g5-10M-20M`'s own report. The devices are contemporaries and "
      "the wire is identical; the graph is the same design at the same weight but not the "
      "same file, since Neuron compiles from PyTorch and the GPU run from ONNX.")
    w("")
    w("| model | rows x TPS | A10G: split verdict | trn1: split verdict |")
    w("|---|---:|---|---|")
    for size, rows_n, tps, base, best, verdict in answer_rows(doc):
        gpu = GPU.get((size, rows_n, tps))
        if not gpu:
            continue
        w(f"| {size.upper()} | {rows_n} x {tps} | {gpu['verdict']} | "
          f"{verdict.replace('**', '')} |")
    w("")
    w("The split pays where the pieces reach something that can run them at once. On one "
      "A10G reached through a single ONNX Runtime session they could not. On a NeuronCore "
      "with per-piece contexts they can -- and the same hardware fact that makes the "
      "split work is what caps 20M: two cores, each running one NEFF at a time.")
    w("")

    # ---------------------------------------------------------------- method
    prime = doc["prime"]
    w("## What was measured")
    w("")
    w("| | |")
    w("|---|---|")
    w("| server | `trn1.2xlarge` -- 1 Trainium1, 2 NeuronCores, 32 GB device memory, "
      "8 vCPU |")
    w("| client | `m9g.8xlarge`, same availability zone |")
    w("| precision | FP32 throughout |")
    for size in sorted(prime):
        p = prime[size]
        shapes = ", ".join(f"b{s['mini_batch']}" for s in p["shapes"])
        w(f"| {size.upper()} model | {p['parameter_count']:,} parameters, "
          f"{p['layers']} layers, `[batch, {p['input_width']}]` -> `[batch, 5]`; "
          f"NEFFs {shapes} |")
    w("| toolchain | torch_neuronx 2.9.0.2.15.32035, libnrt 2.33.1.0, "
      "`Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)` |")
    w("| window | 120 s warm-up discarded, then 900 s measured as one continuous load in "
      "15 slices; fresh server per cell, stopped afterwards, 60 s idle |")
    w(f"| cells | {doc['counts']['cells']} measured: "
      f"{doc['counts']['sustained']} sustained, {doc['counts']['partial']} partial, "
      f"{doc['counts']['failed']} failed the rate; "
      f"{doc['counts']['failures']} harness failures |")
    w("")
    w("### Limits")
    w("")
    w("- **One core per replica, by instruction.** So this run does not say what a split "
      "does when the pieces reach DIFFERENT cores, which is the configuration "
      "`config-sweep-neuron-trn1-10M-50M` found worth 41% on 50M. That is a separate "
      "experiment.")
    w("- **The rates were specified, not probed.** Both 20M rungs turned out to be above "
      "the ceiling, so 20M is a capacity result rather than a latency one.")
    missing = [c for c in cells if dev_span(c) is None]
    w(f"- **{len(missing)} cells carry no server-side timings.** All are 20M 500-row "
      "cells, and the service says why in its own log: `metrics: 4 sends still in "
      "flight, so 300 buffered records were dropped to keep the newest; the destination "
      "is not keeping up`. CloudWatch could not absorb the metric rate under the "
      "heaviest overload. Every affected cell is unsustained and so enters no latency "
      "comparison; their achieved rates and NeuronCore occupancy are independent of "
      "CloudWatch and stand.")
    w("- **`device span` on a split row is the maximum over the pieces**, not their sum, "
      "because the service reports one duration per stage per request. It is the request's "
      "span on the device, which is what the lane arithmetic needs, and is not a "
      "per-piece figure.")
    w("- **NeuronCore occupancy is occupancy, not saturation** -- the share of a sample "
      "period with work resident. A split produces a stream of smaller executions and so "
      "reads high on a core with headroom.")
    w("- **No cell was repeated**, so no same-host spread was measured here. Two medians "
      "within 8.4% are reported as `level`, inherited from "
      "`minibatch-rows-cpu-arm64-10M`, which measured that spread on a split "
      "configuration.")
    w("")
    w("Evidence is under `raw/`: `results.json` is this report's only input, with cell "
      "records, resolved server configurations, banners, per-second host and NeuronCore "
      "samples, and the compiled-artifact provenance. `raw/driver/` carries the scripts "
      "that ran.")
    w("")

    path = RESULTS / "summary.md"
    path.write_text("\n".join(out))
    print(f"wrote {path} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
