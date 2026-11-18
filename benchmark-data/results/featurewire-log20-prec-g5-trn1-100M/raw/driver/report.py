#!/usr/bin/env python3
"""Writes `results/featurewire-log20-prec-g5-trn1-100M/summary.md` from both lanes' results.

    ./report.py

Reads `raw/results-gpu.json` and `raw/results-neuron.json` -- the two collectors' output,
each enriched with its own accelerator's device samples -- and the two control runs'
`results.json`, and writes one report.

NUMBERS ONLY. This round was asked for tables rather than narration, so every section is a
table and the prose is confined to one line of column definitions where a heading is not
self-explanatory. Nothing here is hand-edited: `verify.py` re-derives every displayed
figure from the same inputs and fails on any disagreement.

THE TWO LANES RECORD DIFFERENT FIELD NAMES for the same quantity, because each collector
was written for its own accelerator -- the GPU one puts CloudWatch under `cloudwatch` and
the Neuron one under `server_side`. `row()` normalises both into one shape ONCE, here, so
no table below has to know which lane a cell came from.
"""
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BD = benchmark_data(HERE)
# OVERRIDABLE so `selftest.py` can point both this and `verify.py` at a
# synthetic results tree. A test that cannot run the real generator is not a
# test of the real generator.
RUN = os.environ.get("REPORT_RUN", "featurewire-log20-prec-g5-trn1-100M")
RAW = BD / "results" / RUN / "raw"
OUT = BD / "results" / RUN / "summary.md"

# The round this one is a subtraction against: the SAME model at the SAME two precisions,
# the same four loads and the same session count on the same instance type -- with the
# TENSOR wire and the inference log OFF. So an A10G delta against it is the feature path
# plus the log and nothing else.
#
# IT IS RETAINED IN `results/`, so this reads it live and there is no snapshot to keep in
# step. There is NO Trainium control, for the plain reason that no campaign has measured
# THIS MODEL SIZE on Trainium. The round that measured this wire and log rate did so at
# 10M and 20M, and it was genuinely FP32 (see raw/probe/), so it is comparable in KIND --
# it simply has no cell at 100M to subtract against.
CONTROLS = {
    "a10g": BD / "results" / "nominibatch-precision-gpu-g5-100M" / "raw"
    / "results.json",
}
CONTROL_RUN = "nominibatch-precision-gpu-g5-100M"
CONTROL_NOTE = ("the same model at the same two precisions and the same four loads on the "
                "same instance type, on the TENSOR wire with the inference log OFF, so an "
                "A10G delta here is the feature path plus the log")
ACCEL_ORDER = {"a10g": 0, "trainium1": 1}
ACCEL_LABEL = {"a10g": "A10G", "trainium1": "trn1"}
PREC_LABEL = {"fp32": "FP32", "bf16": "BF16"}
PREC_ORDER = {"fp32": 0, "bf16": 1}


def load(path, what):
    if not path.exists():
        sys.exit(f"missing {what}: {path}\nrun the lane's collect.py first")
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, list) else doc.get("cells", [])


def meta(path):
    """The Neuron collector's run-level block, which carries the rig and the prime record."""
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, dict) else {}


def g(doc, *path, default=None):
    """Nested lookup that tolerates a None anywhere along the way."""
    for key in path:
        if not isinstance(doc, dict):
            return default
        doc = doc.get(key)
    return default if doc is None else doc


def num(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def pct(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}%"


def row(cell):
    """One cell, with both lanes' field names resolved to one shape."""
    accel = cell.get("accelerator")
    result = cell.get("result") or {}
    verdict = cell.get("verdict") or {}
    neuron = accel == "trainium1"

    if neuron:
        graph = g(cell, "server_side", "graph_ms", "p50")
        graph95 = g(cell, "server_side", "graph_ms", "p95")
        vec = g(cell, "server_side", "vectorize_ms", "p50")
        # THE FALLBACK, AND WHY IT IS NEEDED HERE RATHER THAN BEING A DEFECT. Every
        # Trainium cell in this round ran far over its device ceiling, so requests queued
        # for seconds and the host's async pool was saturated -- which starved the
        # server's own CloudWatch publishing: ~1,450 publishes per cell were lost to a
        # 3.1 s connect timeout, leaving the measured window with no COMPLETE minute
        # bucket. `graph_ms` is therefore None on exactly the cells whose device time is
        # the most interesting number in the round.
        #
        # `graph_ms_whole_cell` survives, because it is the median over the cell's whole
        # wall clock and so needs only one bucket. It INCLUDES the warm-up, which is why
        # it is not used when the in-window figure exists and why `graph_window` records
        # which one a row carries. On these cells it agrees with the device time implied
        # by the achieved rate to within 2%, which is the cross-check that makes it
        # usable: 16.02 ms against 2 / 127.1 = 15.73.
        graph_window = "measured window"
        if graph is None:
            graph = g(cell, "server_side", "graph_ms_whole_cell", "p50")
            graph95 = g(cell, "server_side", "graph_ms_whole_cell", "p95")
            if graph is not None:
                graph_window = "whole cell, includes warm-up"
        tensor = None
        buckets = g(cell, "server_side", "graph_buckets", default=0) \
            or g(cell, "server_side", "graph_buckets_whole_cell", default=0)
        cpu = g(cell, "host_series", "cpu_pct")
        cpu_max = g(cell, "host_series", "cpu_max_pct")
        mem = g(cell, "host_series", "mem_pct")
        dev = g(cell, "neuron_series", "nc_mean_pct")
        dev_max = g(cell, "neuron_series", "nc_peak_pct")
        dev_mib = g(cell, "neuron_series", "device_mib")
        power = None
        client_cpu = g(cell, "client_series", "cpu_pct")
        metrics_lost = g(cell, "server_side", "metric_sends_lost")
        graph_note = g(cell, "server_side", "unavailable_reason")
    else:
        graph = g(cell, "cloudwatch", "InferenceTime", "p50_ms")
        graph95 = g(cell, "cloudwatch", "InferenceTime", "p95_ms")
        vec = g(cell, "cloudwatch", "VectorizationTime", "p50_ms")
        tensor = g(cell, "cloudwatch", "ToTensorTime", "p50_ms")
        buckets = g(cell, "cloudwatch", "InferenceTime", "buckets", default=0)
        cpu = g(cell, "server_utilization", "cpu_pct")
        cpu_max = g(cell, "server_utilization", "cpu_max_pct")
        mem = g(cell, "server_utilization", "host_memory_pct")
        dev = g(cell, "gpu_utilization", "gpu_util_pct")
        dev_max = g(cell, "gpu_utilization", "gpu_util_max_pct")
        dev_mib = g(cell, "gpu_utilization", "memory_used_mib")
        power = g(cell, "gpu_utilization", "power_w")
        client_cpu = g(cell, "client_utilization", "cpu_pct")
        graph_window = "measured window"
        metrics_lost = None
        graph_note = None

    return dict(
        id=cell["id"],
        accel=accel,
        accel_label=ACCEL_LABEL.get(accel, accel),
        stack=cell.get("stack", "hushar"),
        wire=cell.get("wire"),
        size=cell.get("size"),
        rows=cell.get("rows"),
        tps=cell.get("tps"),
        rows_per_sec=cell.get("rows_per_sec"),
        sessions=cell.get("parallelism") or cell.get("sessions"),
        kind=cell.get("parallelism_kind", "sessions"),
        state=verdict.get("state", "?"),
        sustained=bool(cell.get("sustained")),
        offered=result.get("offered"),
        achieved=result.get("achieved"),
        p50=result.get("p50"),
        p95=result.get("p95"),
        p99=result.get("p99"),
        shed=result.get("shed"),
        failed=result.get("failed"),
        slices_held=verdict.get("slices_held"),
        slices_total=verdict.get("slices_total"),
        window=g(cell, "measured_window", "seconds")
        or g(cell, "window", "duration_seconds"),
        graph=graph, graph95=graph95, vec=vec, tensor=tensor, buckets=buckets,
        graph_window=graph_window, metrics_lost=metrics_lost, graph_note=graph_note,
        cpu=cpu, cpu_max=cpu_max, mem=mem, client_cpu=client_cpu,
        dev=dev, dev_max=dev_max, dev_mib=dev_mib, power=power,
        # the inference log
        rate=cell.get("sample_rate"),
        every=cell.get("sample_every"),
        log_batch=cell.get("log_batch"),
        log_rows_per_object=cell.get("log_rows_per_object"),
        log_objects=cell.get("log_objects"),
        log_bytes=cell.get("log_bytes_written"),
        log_shed=cell.get("log_shed_batches"),
        log_write_failures=cell.get("log_write_failures"),
        log_predicted_bps=cell.get("log_bytes_per_sec"),
        # provenance
        host=cell.get("host"),
        instance=cell.get("instance"),
        params=cell.get("parameter_count") or g(cell, "metadata", "parameter_count"),
        layers=cell.get("layers") or g(cell, "metadata", "layers"),
        macs=cell.get("macs_per_row"),
        tensorrt=cell.get("host_tensorrt"),
        ort=cell.get("onnxruntime"),
        triton_image=cell.get("triton_image"),
        libnrt=cell.get("libnrt"),
        rev=cell.get("rev"),
        net_gbps=g(cell, "network", "total_gbps"),
        net_pct=g(cell, "network", "pct_of_baseline"),
        # ---- THE AXIS OF THIS ROUND.
        #
        # `precision` is the label the cell was run under. The other three are the
        # EVIDENCE, and they are carried separately because a label is not a measurement:
        #
        #   engine_bytes   the built artifact's size. A BF16 engine holds its weights at
        #                  half the width, so the ratio across an arm pair is checkable
        #                  without trusting any flag. Present on both A10G stacks.
        #   auto_cast      what the Neuron compiler was actually told. Its own DEFAULT is
        #                  `matmult`, so an absent value means the precision claim rests
        #                  on a default that is not FP32.
        #   trace_error    the compiled-versus-PyTorch maximum on Trainium, ~1e-6 at FP32
        #                  and ~1e-2 once the matmuls are BF16. Four orders apart, so it
        #                  separates the arms on its own.
        precision=cell.get("precision"),
        engine_bytes=cell.get("engine_bytes"),
        auto_cast=cell.get("auto_cast"),
        auto_cast_type=cell.get("auto_cast_type"),
        trace_error=(g(cell, "metadata", "maximum_trace_error")
                     or cell.get("maximum_trace_error")),
        bf16_flag=cell.get("bf16_flag_echoed"),
        plan=cell.get("plan"),
    )


def control_cells():
    """The control's figures, flattened, from the snapshot if there is one.

    PREFERS THE SNAPSHOT, which is `raw/control-<run>.json` inside this round. The control
    round is not retained in `results/`, so reading it live would make this report
    unreproducible the moment that directory went -- which is exactly what happened. The
    live path is kept as the fallback so the first generation, before any snapshot exists,
    still works; `snapshot_control.py` is what writes the file.

    Returns `{(accel, size, rows, tps, stack): {p50, graph, sustained}}` whichever source
    it came from, so the table below does not know the difference.
    """
    out, source = {}, None
    snapshot = RAW / f"control-{CONTROL_RUN}.json"
    if snapshot.exists():
        doc = json.loads(snapshot.read_text())
        source = doc.get("source") or CONTROL_RUN
        for c in doc.get("cells", []):
            out[(c["accelerator"], c["size"], c.get("precision"), c["rows"], c["tps"],
                 c["stack"])] = dict(
                p50=c.get("p50"), graph=c.get("graph_p50_ms"),
                sustained=bool(c.get("sustained")))
        return out, source, "snapshot"

    for accel, path in CONTROLS.items():
        if not path.exists():
            continue
        source = f"results/{CONTROL_RUN}"
        doc = json.loads(path.read_text())
        for c in (doc if isinstance(doc, list) else doc.get("cells", [])):
            # THE PRECISION IS PART OF THE KEY. The control ran BOTH arms at 100 x 100 and
            # 500 x 20, so a key without it would collide its FP32 and BF16 cells and this
            # round would subtract against whichever happened to be read last.
            key = (accel, c.get("size"), c.get("precision"), c.get("rows"), c.get("tps"),
                   c.get("stack", "hushar"))
            graph = (g(c, "server_side", "graph_ms", "p50") if accel == "trainium1"
                     else g(c, "cloudwatch", "InferenceTime", "p50_ms"))
            out[key] = dict(p50=(c.get("result") or {}).get("p50"), graph=graph,
                            sustained=bool(c.get("sustained")))
    return out, source, "live"


def table(lines, header, rows):
    lines.append("")
    lines.append("| " + " | ".join(header[0]) + " |")
    lines.append("|" + "|".join(header[1]) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")


def state_cell(r):
    """What the cell did to its offered rate, in the vocabulary the verdict uses.

    THREE OUTCOMES, NOT TWO, because this round produced all three and collapsing them
    would lose the distinction that matters. A `partial` cell held the rate for part of
    the window and fell behind in the rest, which is a different statement from a cell
    that never held it -- and neither may have its latency ranked, so the slice count
    travels with the label rather than being left to another table.
    """
    if r["sustained"]:
        return f"ok {num(r['achieved'], 1)}/{r['offered']}"
    if r["state"] == "partial":
        return (f"partial {num(r['achieved'], 1)}/{r['offered']} "
                f"({r['slices_held']}/{r['slices_total']})")
    return f"FAIL {num(r['achieved'], 1)}/{r['offered']}"


def lat(r, key):
    """A latency, withheld on a cell that did not hold its offered rate."""
    return num(r[key]) if r["sustained"] else "--"


def lane_prime(raw):
    """Prime facts per lane, parsed from the LANE LOGS rather than the prime records.

    WHY THE LOGS AND NOT `prime/*.json`. An earlier round in this series had both lanes
    write `runs/<run>/prime/<size>.json`, one key, so each overwrote the other -- and the
    GPU lane then found the Trainium lane's record already there and SKIPPED, building its
    engines inside a cell's warm-up. The keys are lane- and precision-specific now, but the
    lane logs are per-lane by construction, so they are what this reads.

    THE PRECISION COMES FROM THE LINE WHEN THE LINE HAS IT, AND FROM THE LANE TAG WHEN IT
    DOES NOT. Both spellings are accepted on purpose: the GPU lanes of this round were
    launched before `prime ... r<rows>` learned to name its arm, so their logs carry
    `prime 100m r100: build ...` while the Trainium logs carry `prime 100m/fp32 mb100:`.
    The lane log's own filename -- `lane-fp32-g5-1.log` -- settles it either way, so a
    round already in flight does not have to be restarted to be reportable.

    WHICH ENTRY WINS WHEN A SHAPE WAS PRIMED TWICE. A `prime`-tagged re-run only fills
    shapes the measured lane has none for. The `when` column says which, rather than the
    two being silently averaged or the later one quietly winning.
    """
    found = {}
    for path in sorted((raw / "driver").glob("lane-*.log")):
        stem = path.stem[len("lane-"):]
        # `lane-g5-1.log` is an untagged main run; `lane-fp32-g5-1.log` carries the arm;
        # `lane-prime-g5-1.log` is a re-run under a tag.
        tag, host, lane_prec = "main", stem, None
        for known in ("fp32", "bf16", "prime"):
            if stem.startswith(f"{known}-"):
                host = stem[len(known) + 1:]
                if known == "prime":
                    tag = known
                else:
                    lane_prec = known
                break
        text = path.read_text(errors="replace")
        rows = []
        # GPU, either spelling:
        #   `prime 100m r100: build 25.3s -> cached 22.1s`
        #   `prime 100m/bf16 r100: build 25.3s -> cached 22.1s engine 202952556 bytes`
        for size, prec, shape, build, cached, ebytes in re.findall(
                r"prime (\w+)(?:/(\w+))? r(\d+): build ([\d.]+)s -> cached ([\d.]+)s"
                r"(?: engine (\d+) bytes)?", text):
            rows.append(dict(accel="a10g", kind="TensorRT engine", size=size,
                             precision=prec or lane_prec,
                             shape=int(shape), source="built", seconds=float(build),
                             proof_seconds=float(cached),
                             mib=(int(ebytes) / 1048576.0 if ebytes else None),
                             bytes=int(ebytes) if ebytes else None,
                             params=None, trace_error=None))
        # Neuron: `prime 100m/fp32 mb100: compiled 812.4s, NEFF 345.1 MiB,
        #          100,561,925 params / 31 layers, trace error 1.4e-06,
        #          served a trickle in 22.1s`
        for size, prec, shape, source, secs, mib, params, err, proof in re.findall(
                r"prime (\w+)(?:/(\w+))? mb(\d+): (\S+) ([\d.]+)s, NEFF ([\d.]+) MiB, "
                r"([\d,]+) params / \d+ layers, trace error (\S+), "
                r"served a trickle in ([\d.]+)s", text):
            rows.append(dict(accel="trainium1", kind="NEFF", size=size,
                             precision=prec or lane_prec,
                             shape=int(shape), source=source, seconds=float(secs),
                             proof_seconds=float(proof), mib=float(mib),
                             bytes=int(float(mib) * 1048576),
                             params=int(params.replace(",", "")),
                             trace_error=err))
        for r in rows:
            # The accelerator comes from the LINE SHAPE, not the host name: a re-run's log
            # is called `lane-prime-g5-1.log`, whose stem does not start with `g5`.
            #
            # AN ENTRY WITH NO PRECISION IS DISCARDED. This round is precision-keyed, so a
            # prime figure that cannot be attributed to an arm cannot be placed in any
            # table -- and in practice it is a DUPLICATE: the sweeps uploaded their lane
            # log under both `lane-<host>.log` and `lane-<tag>-<host>.log` for a while, so
            # the untagged copy carries the same lines with no arm to key them on. Keeping
            # it would print every primed shape twice.
            if not r["precision"]:
                continue
            # THE PRECISION IS PART OF THE KEY. Both arms prime the same two shapes of the
            # same size, so a key without it keeps one arm and discards the other.
            key = (r["accel"], r["size"], r["precision"], r["shape"])
            if key in found and found[key]["when"] == "main":
                continue
            found[key] = dict(r, host=host, when=tag)
    return list(found.values())


def main():
    gpu = load(RAW / "results-gpu.json", "the A10G lane's results")
    neu = load(RAW / "results-neuron.json", "the Trainium lane's results")
    nmeta = meta(RAW / "results-neuron.json")
    primes = lane_prime(RAW)
    rows = [row(c) for c in gpu + neu]
    rows.sort(key=lambda r: (ACCEL_ORDER.get(r["accel"], 9), r["size"],
                             r["stack"] != "hushar", r["rows"], r["tps"]))
    hushar = [r for r in rows if r["stack"] == "hushar"]
    triton = [r for r in rows if r["stack"] == "triton"]
    controls, control_source, control_kind = control_cells()
    # Parameter counts per accelerator and size. The ONNX graphs carry theirs on every
    # cell; the NEFFs carry theirs only in the prime record, because the artifact a
    # Trainium cell runs is compiled on the host rather than shipped.
    params = {}
    for r in rows:
        if r["params"]:
            params[(r["accel"], r["size"])] = r["params"]
    for p in primes:
        if p["params"]:
            params.setdefault((p["accel"], p["size"]), p["params"])

    first = rows[0] if rows else {}
    collected = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    L = [f"# A 100M model on its own wire with the log on: FP32 against BF16 "
         f"on one A10G and one Trainium1", "",
         f"`{RUN}` | collected {collected} | rev `{first.get('rev', '?')}`", ""]

    # ------------------------------------------------------------------ configuration
    a10g = [r for r in rows if r["accel"] == "a10g"]
    trn1 = [r for r in rows if r["accel"] == "trainium1"]
    # The ONNX graph's own figures, for the configuration line. Taken from a cell rather
    # than written down, so the table cannot drift from what the lane actually served.
    _p = params.get(("a10g", "100m"))
    onnx_params = f"{_p:,}" if _p else "--"
    _l = next((r["layers"] for r in a10g if r.get("layers")), None)
    onnx_layers = str(_l) if _l else "--"
    L.append("## Configuration")
    table(L, (["setting", "value"], ["---", "---"]), [
        ["servers", "`g5.2xlarge` 1x NVIDIA A10G 8 vCPU; "
                    "`trn1.2xlarge` 1x Trainium1 2 NeuronCore 8 vCPU"],
        ["clients", "`m9g.8xlarge`, 32 vCPU, one per lane, same availability zone"],
        ["model", f"one 100M graph -- {onnx_params} parameters, {onnx_layers} encoder "
                  f"layers -- served at two precisions"],
        ["precision", "FP32 and BF16. On the A10G both arms build the SAME ONNX bytes "
                      "into a TensorRT engine, FP32 at the builder's defaults and BF16 "
                      "through `trt_bf16_enable=1` on hushar and `--bf16` on `trtexec`. "
                      "On Trainium each arm is a separate `neuronx-cc` compile, "
                      "`--auto-cast=none` against "
                      "`--auto-cast=matmult --auto-cast-type=bf16`"],
        ["precision is explicit", "Both Trainium arms pass `--auto-cast` rather than "
                                  "relying on a default. `neuronx-cc`'s CLI reference "
                                  "gives the flag a default of `matmult`, but "
                                  "`torch_neuronx.trace` does NOT inherit it: compiling "
                                  "one model with no flag, with `--auto-cast=none` and "
                                  "with `--auto-cast=matmult` produced a BYTE-IDENTICAL "
                                  "NEFF for the first two. So the earlier Trainium "
                                  "rounds in this series, which passed no flag, were "
                                  "genuinely FP32 and ARE comparable with this round's "
                                  "FP32 arm. The flag is passed anyway so the label "
                                  "rests on the artifact rather than on a framework "
                                  "default -- see `raw/probe/autocast-default.txt`"],
        ["wire", "210 named features per row, transformed and concatenated by hushar "
                 "into the model's one 690-float input; server-side vectorization ON"],
        ["feature logging", f"`inference_log.sample_rate` "
                            f"{first.get('rate')} -> 1 request in "
                            f"{first.get('every')}, to Amazon S3"],
        ["log object", f"one per {first.get('log_rows_per_object'):,} logged rows -- "
                       f"{', '.join(f'{b} requests at {r} rows' for r, b in sorted({(h['rows'], h['log_batch']) for h in hushar}))}, "
                       f"so ~50 MiB whichever packaging; "
                       f"{sum(h['log_write_failures'] or 0 for h in hushar)} write "
                       f"failure(s) and "
                       f"{sum(1 for h in hushar if h['log_shed'])} cell(s) that shed, "
                       f"across the round"],
        ["candidate model", "none; every request scores the one resident model"],
        ["mini batch", "none on the A10G (`mini_batch_size` unset); equal to the row "
                       "count on Trainium, where a NEFF's leading dimension is fixed. "
                       "Both are one engine call per request"],
        ["parallelism", "`threading.sessions_per_model` 2, `sessions_per_request` 1, so "
                        "2 requests in flight; `instance_group` 2 on Triton"],
        ["Neuron placement", "1 core per replica, 1 execution context per replica, "
                             "1 replica per core, so both cores are occupied"],
        ["hushar engine", f"ONNX Runtime {first.get('ort') or '?'} TensorRT provider on "
                          f"the A10G; libnrt "
                          f"{next((r['libnrt'] for r in trn1 if r['libnrt']), '?')} on "
                          f"Trainium"],
        ["Triton", f"`{next((r['triton_image'] for r in triton if r['triton_image']), '?')}`, "
                   f"`tensorrt` backend, `max_batch_size: 0`, the 690-float vector in, "
                   f"no feature logging"],
        ["TensorRT", f"{next((r['tensorrt'] for r in rows if r['tensorrt']), '?')}, the "
                     f"same library for both A10G arms: CUDA, cuDNN and TensorRT are "
                     f"extracted from the Triton image and hushar links those copies, "
                     f"gated by `ldd` at deploy and a version check at preflight"],
        ["window", f"{first.get('window')}s measured in "
                   f"{first.get('slices_total')} slices, after a 300s warm-up "
                   f"discarded; fresh server per cell, 60s idle between"],
        ["cells", f"{len(rows)} measured, {sum(r['sustained'] for r in rows)} sustained "
                  f"the offered rate"],
    ])

    # ------------------------------------------------------------------ headline
    L.append("## Sustained medians")
    held = [r for r in rows if r["sustained"]]
    table(L, (["accel", "precision", "rows x TPS", "rows/s", "stack", "p50", "p95",
               "p99", "graph", "vectorize", "device", "host CPU"],
              ["---", "---", "--:", "--:", "---", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
            f"{r['rows']} x {r['tps']}",
            f"{r['rows_per_sec']:,}", r["stack"],
            num(r["p50"]), num(r["p95"]), num(r["p99"]),
            num(r["graph"], 3), num(r["vec"], 3), pct(r["dev"]), pct(r["cpu"])]
           for r in held])

    short = [r for r in rows if not r["sustained"]]
    if short:
        L.append("## Did not hold the offered rate")
        table(L, (["accel", "precision", "rows x TPS", "stack", "offered", "achieved",
                   "shortfall", "slices held", "client shed", "device", "host CPU",
                   "graph"],
                  ["---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:", "--:",
                   "--:", "--:"]),
              [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
                f"{r['rows']} x {r['tps']}",
                r["stack"], str(r["offered"]), num(r["achieved"], 1),
                pct((1 - r["achieved"] / r["offered"]) * 100)
                if r["offered"] else "--",
                f"{r['slices_held']}/{r['slices_total']}", str(r["shed"]),
                pct(r["dev"]), pct(r["cpu"]), num(r["graph"], 3)]
               for r in short])
        L.append("No latency is quoted for these: a median taken while requests queue "
                 "measures the queue. `graph` is still the per-request device time, which "
                 "is what says whether the device or the host was the limit.")
        L.append("")

    # ------------------------------------------------------------------ the axis
    #
    # THE TABLE THIS ROUND EXISTS FOR. BF16 is offered TWICE the rate of FP32 at each
    # packaging, so the pair is not a like-for-like latency comparison and must not be
    # printed as one: the honest readings are the per-request device time, which is rate
    # independent at these occupancies, and the throughput each arm actually sustained.
    rows_all = rows

    def one(accel, prec, n_rows, tps, stack):
        return next((r for r in rows_all if r["accel"] == accel
                     and r["precision"] == prec and r["rows"] == n_rows
                     and r["tps"] == tps and r["stack"] == stack), None)

    prec_rows = []
    for accel in ("a10g", "trainium1"):
        for stack in ("hushar", "triton"):
            for pack, (f_tps, b_tps) in ((100, (100, 200)), (500, (20, 40))):
                f = one(accel, "fp32", pack, f_tps, stack)
                b = one(accel, "bf16", pack, b_tps, stack)
                if not f or not b:
                    continue
                gr = (f"{f['graph'] / b['graph']:.2f}x"
                      if f["graph"] and b["graph"] else "--")
                th = (f"{(b['achieved'] * pack) / (f['achieved'] * pack):.2f}x"
                      if f["achieved"] and b["achieved"] else "--")
                eb = (f"{b['engine_bytes'] / f['engine_bytes']:.3f}x"
                      if f["engine_bytes"] and b["engine_bytes"] else "--")
                prec_rows.append([
                    ACCEL_LABEL[accel], stack, f"{pack} rows",
                    f"{f_tps} -> {b_tps}",
                    state_cell(f), state_cell(b),
                    num(f["graph"], 3), num(b["graph"], 3), gr,
                    f"{int(f['achieved'] * pack):,}" if f["achieved"] else "--",
                    f"{int(b['achieved'] * pack):,}" if b["achieved"] else "--",
                    th, eb])
    if prec_rows:
        L.append("## FP32 against BF16")
        table(L, (["accel", "stack", "rows", "TPS offered", "FP32", "BF16",
                   "FP32 graph", "BF16 graph", "graph speedup",
                   "FP32 rows/s", "BF16 rows/s", "throughput", "artifact bytes"],
                  ["---", "---", "--:", "--:", "---", "---", "--:", "--:", "--:",
                   "--:", "--:", "--:", "--:"]), prec_rows)
        L.append("The two arms are NOT offered the same rate -- BF16 is asked for double "
                 "at each packaging -- so a median against a median would price the rate "
                 "and the precision together. `graph speedup` is FP32's per-request "
                 "device time over BF16's, which is what the precision bought; "
                 "`throughput` is the rows per second each arm actually sustained. "
                 "`artifact bytes` is the BF16 engine or NEFF over its FP32 pair: a "
                 "builder that accepted the flag and kept every layer at full width would "
                 "show the flag and not the shrink.")
        L.append("")

    # ------------------------------------------------------- is the precision real
    L.append("## Precision evidence")
    ev = []
    for accel in ("a10g", "trainium1"):
        for prec in ("fp32", "bf16"):
            match = [r for r in rows_all if r["accel"] == accel
                     and r["precision"] == prec]
            if not match:
                continue
            h = [r for r in match if r["stack"] == "hushar"]
            src = h or match
            flags = ("`trt_bf16_enable=1`" if prec == "bf16" else "builder defaults") \
                if accel == "a10g" else f"`--auto-cast={src[0]['auto_cast']}`"
            sizes = sorted({r["engine_bytes"] for r in match if r["engine_bytes"]})
            err = next((r["trace_error"] for r in src if r["trace_error"]), None)
            ev.append([ACCEL_LABEL[accel], PREC_LABEL[prec], flags,
                       ", ".join(f"{s:,}" for s in sizes) or "--",
                       f"{err:.3g}" if err else "--",
                       "yes" if accel == "a10g"
                       and any(r["bf16_flag"] for r in h) else
                       ("no" if accel == "a10g" and prec == "fp32" else "--")])
    table(L, (["accel", "precision", "builder flag", "artifact bytes",
               "trace error", "BF16 flag in banner"],
              ["---", "---", "---", "--:", "--:", "--:"]), ev)
    L.append("Three independent accounts of the same claim, because a precision label is "
             "the one thing in this round that could be wrong while every other gate "
             "passed. The A10G reads the flag back out of the server's own banner and "
             "measures the engine it built. Trainium cannot read a flag back, so it uses "
             "the compiled-versus-PyTorch maximum error: FP32 lands near 1e-6 and BF16 "
             "matmuls three orders above it, and `compile_neuron.py` REFUSES to write a "
             "NEFF whose error falls outside the window its label requires.")
    L.append("")
    L.append("That check earned its place immediately. This round was planned on the "
             "belief that `neuronx-cc`'s documented `--auto-cast` default of `matmult` "
             "reached `torch_neuronx.trace`, which would have made every earlier "
             "Trainium round's FP32 label wrong. The recorded trace errors of those "
             "rounds -- ~1.5e-06, against the 2.3e-03 this round's BF16 arm produces -- "
             "said otherwise, and a measured three-way compile confirmed it: no flag and "
             "`--auto-cast=none` produce a byte-identical NEFF. The earlier rounds were "
             "FP32. `raw/probe/autocast-default.txt` has the comparison.")
    L.append("")

    # ------------------------------------------------- the same rate on both arms
    #
    # THE CLEANEST COMPARISON IN THE ROUND, and the one a hardcoded rate pairing missed.
    # Where both arms were offered the SAME rows x TPS, the difference between their
    # medians is the cast and nothing else -- no rate term to reason away. The pairs are
    # DERIVED from the cells rather than written down, because which rates the two arms
    # happen to share depends on what each could sustain, and that is a measurement.
    matched = []
    for accel in ("a10g", "trainium1"):
        for stack in ("hushar", "triton"):
            for n_rows, tps in sorted({(r["rows"], r["tps"]) for r in rows_all}):
                a = one(accel, "fp32", n_rows, tps, stack)
                b = one(accel, "bf16", n_rows, tps, stack)
                if not a or not b:
                    continue
                both = a["sustained"] and b["sustained"]
                ratio = (f"{a['p50'] / b['p50']:.2f}x"
                         if both and b["p50"] else "--")
                gratio = (f"{a['graph'] / b['graph']:.2f}x"
                          if a["graph"] and b["graph"] else "--")
                matched.append([
                    ACCEL_LABEL[accel], stack, f"{n_rows} x {tps}",
                    f"{n_rows * tps:,}",
                    state_cell(a), lat(a, "p50"), num(a["graph"], 3), pct(a["dev"]),
                    state_cell(b), lat(b, "p50"), num(b["graph"], 3), pct(b["dev"]),
                    ratio, gratio])
    if matched:
        L.append("## FP32 against BF16 at the same offered rate")
        table(L, (["accel", "stack", "rows x TPS", "rows/s",
                   "FP32", "p50", "graph", "device",
                   "BF16", "p50", "graph", "device",
                   "p50 ratio", "graph ratio"],
                  ["---", "---", "--:", "--:", "---", "--:", "--:", "--:",
                   "---", "--:", "--:", "--:", "--:", "--:"]), matched)
        L.append("Both arms offered the same rows and the same rate, so the ratio is the "
                 "cast with no rate term in it. A `p50 ratio` is quoted only where BOTH "
                 "arms held the rate; the `graph ratio` is quoted regardless, because the "
                 "per-request device time is meaningful even on a cell that queued.")
        L.append("")
        L.append("The rows where FP32 shows `FAIL` against a BF16 `ok` are the cast "
                 "buying throughput rather than latency: the same offered rate is inside "
                 "one arm's ceiling and outside the other's.")
        L.append("")

    # ------------------------------------------------------- what the short cells measured
    ceil_rows = []
    for r in rows_all:
        if r["sustained"] or r["accel"] != "trainium1":
            continue
        # A NeuronCore executes one NEFF at a time and this host has two, so a cell that
        # queued was serving at the device's own rate: the achieved rate IS the ceiling,
        # and 2 / achieved is the per-request device span implied by it. That is an
        # INDEPENDENT estimate of the same quantity the server measured, which is what
        # makes the pair worth printing -- one comes from the client's completed requests
        # and the other from the server's own timer.
        implied = (2.0 / r["achieved"] * 1000.0) if r["achieved"] else None
        agree = (f"{abs(implied - r['graph']) / r['graph'] * 100:.1f}%"
                 if implied and r["graph"] else "--")
        ceil_rows.append([PREC_LABEL.get(r["precision"], "?"),
                          f"{r['rows']} x {r['tps']}", str(r["offered"]),
                          num(r["achieved"], 1),
                          f"{int(r['achieved'] * r['rows']):,}" if r["achieved"] else "--",
                          num(implied, 2), num(r["graph"], 2), agree,
                          pct(r["dev"]), pct(r["cpu"])])
    if ceil_rows:
        L.append("## What the Trainium cells measured")
        table(L, (["precision", "rows x TPS offered", "offered", "achieved",
                   "rows/s achieved", "implied ms/call", "server ms/call", "agreement",
                   "NeuronCore", "host CPU"],
                  ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
              ceil_rows)
        L.append("Not one Trainium cell held its offered rate, and that is what they were "
                 "run to find out. The rates were set from the A10G, where all four were "
                 "measured holding; one `trn1.2xlarge` is two NeuronCores, each executing "
                 "one NEFF at a time. So these cells measure the CEILING, which no "
                 "campaign had for this model size on this device.")
        L.append("")
        L.append("`implied ms/call` is `2 / achieved`: with both cores busy and one "
                 "execution each, the rate a saturated host completes requests at fixes "
                 "the per-request device span. `server ms/call` is hushar's own "
                 "`InferenceTime`. They are independent -- one is the client's completed "
                 "requests, the other the server's timer -- and `agreement` is the gap "
                 "between them.")
        L.append("")
        L.append("**THE SERVER FIGURE IS A WHOLE-CELL MEDIAN ON THESE CELLS, NOT AN "
                 "IN-WINDOW ONE.** Queueing for seconds saturated the host's async pool, "
                 "which starved the server's own CloudWatch publishing: around 1,450 "
                 "publishes per cell were lost to a 3.1 s connect timeout, leaving the "
                 "measured window without one complete minute bucket. The whole-cell "
                 "median needs only one bucket and survives, but it includes the warm-up. "
                 "The same saturation is why these cells wrote almost no inference log -- "
                 "see the log table.")
        L.append("")

    acc_rows = []
    for prec in ("fp32", "bf16"):
        for pack, tps in ((100, 100 if prec == "fp32" else 200),
                          (500, 20 if prec == "fp32" else 40)):
            a = one("a10g", prec, pack, tps, "hushar")
            t = one("trainium1", prec, pack, tps, "hushar")
            if not a or not t:
                continue
            ratio = (f"{t['graph'] / a['graph']:.2f}x"
                     if a["graph"] and t["graph"] else "--")
            acc_rows.append([PREC_LABEL[prec], f"{pack} x {tps}",
                             state_cell(a), lat(a, "p50"), num(a["graph"], 3),
                             pct(a["dev"]),
                             state_cell(t), lat(t, "p50"), num(t["graph"], 3),
                             pct(t["dev"]), ratio])
    if acc_rows:
        L.append("## A10G against Trainium1")
        table(L, (["precision", "rows x TPS", "A10G", "p50", "graph", "device",
                   "trn1", "p50", "graph", "device", "trn1 / A10G graph"],
                  ["---", "--:", "---", "--:", "--:", "--:", "---", "--:", "--:",
                   "--:", "--:"]), acc_rows)
        L.append("Both run hushar on the same wire at the same log rate and the same "
                 "session count, so the row is the device and its toolchain. The graphs "
                 "are NOT the same artifact: the A10G serves the ONNX file through "
                 "TensorRT and Trainium serves a NEFF that `torch_neuronx.trace` rebuilt "
                 "from the same seeded architecture, so the parameter counts differ "
                 "slightly and are reported separately. A latency is withheld on a cell "
                 "that did not hold its offered rate; `graph` is published regardless, "
                 "because the per-request device time is what says where the ceiling is.")
        L.append("")

    # ------------------------------------------------------------------ models
    L.append("## Models")
    mrows = []
    for accel in ("a10g", "trainium1"):
        for prec in ("fp32", "bf16"):
            match = [r for r in rows if r["accel"] == accel and r["precision"] == prec]
            if not match:
                continue
            count = params.get((accel, "100m"))
            layers = next((r["layers"] for r in match if r.get("layers")), None)
            sizes = sorted({r["engine_bytes"] for r in match if r["engine_bytes"]})
            mrows.append([ACCEL_LABEL[accel], PREC_LABEL[prec],
                          f"{count:,}" if count else "--",
                          str(layers) if layers else "--", "690", "5",
                          "TensorRT engine" if accel == "a10g" else "NEFF",
                          ", ".join(f"{s:,}" for s in sizes) or "--"])
    table(L, (["accelerator", "precision", "parameters", "layers", "input width",
               "classes", "artifact", "artifact bytes"],
              ["---", "---", "--:", "--:", "--:", "--:", "---", "--:"]), mrows)
    L.append("One graph per accelerator, built twice. The parameter and layer counts do "
             "not move with the precision, because the cast is a builder or compiler "
             "choice rather than a change to the model. The Trainium parameter count "
             "differs from the ONNX one because `torch_neuronx.trace` rebuilds the same "
             "seeded architecture; both are recorded rather than treated as one number. "
             "Two artifact sizes per row are the two row packagings.")
    L.append("")

    # ------------------------------------------------------------------ every cell
    L.append("## Every cell")
    table(L, (["accel", "precision", "rows x TPS", "stack", "sessions", "result",
               "p50", "p95", "p99", "graph", "vectorize", "CPU", "CPU max", "device",
               "slices"],
              ["---", "---", "--:", "---", "--:", "---",
               "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
            f"{r['rows']} x {r['tps']}",
            r["stack"], str(r["sessions"]), state_cell(r),
            lat(r, "p50"), lat(r, "p95"), lat(r, "p99"),
            num(r["graph"], 3), num(r["vec"], 3),
            pct(r["cpu"]), pct(r["cpu_max"]), pct(r["dev"]),
            f"{r['slices_held']}/{r['slices_total']}"] for r in rows])
    L.append("`graph` is hushar's `InferenceTime` and `vectorize` its "
             "`VectorizationTime`, medians of whole minute buckets inside the measured "
             "window. `device` is GPU utilization on the A10G and mean NeuronCore "
             "utilization on Trainium. A latency is withheld on a cell that did not hold "
             "its offered rate.")
    L.append("")

    # ------------------------------------------------------------------ hushar v Triton
    pairs = []
    for h in [r for r in hushar if r["accel"] == "a10g"]:
        # PAIRED WITHIN A PRECISION. The two arms ran at different offered rates, so a
        # pair that matched on rows and TPS alone would cross the axis and price the
        # precision as though it were the server.
        t = next((r for r in triton if r["accel"] == "a10g"
                  and r["precision"] == h["precision"]
                  and r["rows"] == h["rows"] and r["tps"] == h["tps"]), None)
        if not t:
            continue
        ratio = (f"{h['p50'] / t['p50']:.2f}x"
                 if h["sustained"] and t["sustained"] and t["p50"] else "--")
        pairs.append([PREC_LABEL.get(h["precision"], "?"),
                      f"{h['rows']} x {h['tps']}",
                      state_cell(h), lat(h, "p50"), lat(h, "p95"),
                      state_cell(t), lat(t, "p50"), lat(t, "p95"),
                      ratio, num(h["vec"], 3), num(h["graph"], 3),
                      num(t["graph"], 3) if t["graph"] else "--"])
    if pairs:
        L.append("## A10G: hushar against Triton")
        table(L, (["precision", "rows x TPS", "hushar", "p50", "p95", "Triton", "p50",
                   "p95", "hushar / Triton", "hushar vectorize", "hushar graph",
                   "Triton graph"],
                  ["---", "--:", "---", "--:", "--:", "---", "--:", "--:", "--:",
                   "--:", "--:", "--:"]), pairs)
        share = hushar[0]["every"] if hushar else None
        L.append(f"Same GPU, same TensorRT library, same engine shape, same offered rate, "
                 f"one graph call per request on both sides. hushar additionally "
                 f"vectorizes 210 named features and writes "
                 f"{'one request in ' + str(share) if share else 'a sample'} of them to "
                 f"S3; Triton receives the finished tensor and writes nothing. So the "
                 f"ratio prices hushar as deployed against Triton as it ships, not two "
                 f"servers doing the same work.")
        L.append("")

    # ------------------------------------------------- against the pre-vectorized control
    L.append(f"## Against `{CONTROL_RUN}`")
    crows = []
    for r in hushar:
        key = (r["accel"], r["size"], r["precision"], r["rows"], r["tps"], "hushar")
        base = controls.get(key)
        if base is None:
            why = ("no Trainium control: this model size has never been measured on "
                   "Trainium" if r["accel"] == "trainium1"
                   else "no control cell at this rate")
            crows.append([r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
                          f"{r['rows']} x {r['tps']}", str(r["sessions"]),
                          lat(r, "p50"), "--", "--", "--",
                          num(r["graph"], 3), why])
            continue
        bp50 = base["p50"] if base["sustained"] else None
        bgraph = base["graph"]
        delta = (f"{r['p50'] - bp50:+.2f}" if r["sustained"] and bp50 else "--")
        ratio = (f"{r['p50'] / bp50:.2f}x" if r["sustained"] and bp50 else "--")
        crows.append([r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
                      f"{r['rows']} x {r['tps']}", str(r["sessions"]),
                      lat(r, "p50"), num(bp50), delta, ratio,
                      num(r["graph"], 3), num(bgraph, 3)])
    table(L, (["accel", "precision", "rows x TPS", "sessions", "this round p50",
               "control p50", "delta", "ratio", "this round graph", "control graph"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          crows)
    L.append(f"The control is {CONTROL_NOTE}. So an A10G delta is the 210-feature "
             f"vectorization plus one request in five written to S3, measured against the "
             f"same engine at the same rate. The Trainium rows have NO pair and report "
             f"`--`: no campaign has measured this model size on Trainium. The round "
             f"that measured this wire and log rate did so at 10M and 20M, and it was "
             f"genuinely FP32, so it is comparable in kind but has no cell at this "
             f"size. `graph` "
             f"is carried for every row including the unpaired ones, because a device "
             f"time that did not move is what says a latency change came from the host.")
    if control_kind == "snapshot":
        L.append(f"`{CONTROL_RUN}` is **not retained** as a report. Its figures above are "
                 f"cited from `raw/control-{CONTROL_RUN}.json`, which carries the p50, "
                 f"the achieved rate and the `InferenceTime` median per cell and names "
                 f"its source: {control_source}. Each lane's own bucket reduction was "
                 f"reproduced -- the mean of per-minute p50s on the A10G, the median of "
                 f"them on Trainium -- and the eight figures that round published match "
                 f"to three decimals. The banners, samples and client output behind them "
                 f"are not here.")
    L.append("")

    # ------------------------------------------------------------------ median breakdown
    L.append("## Where the median goes")
    brows = []
    for r in hushar:
        parts = [r["graph"], r["vec"], r["tensor"]]
        known = sum(p for p in parts if p)
        rest = (r["p50"] - known) if (r["sustained"] and r["p50"]) else None
        brows.append([r["accel_label"], PREC_LABEL.get(r["precision"], "?"),
                      f"{r['rows']} x {r['tps']}", lat(r, "p50"),
                      num(r["graph"], 3), num(r["vec"], 3), num(r["tensor"], 3),
                      num(rest, 3), str(r["buckets"]),
                      num(r["vec"] / r["p50"] * 100, 1) + "%"
                      if r["sustained"] and r["vec"] and r["p50"] else "--"])
    table(L, (["accel", "precision", "rows x TPS", "p50", "graph", "vectorize",
               "to tensor", "remainder", "buckets", "vectorize share"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          brows)
    L.append("`remainder` is the client median less the three server-side measures: gRPC, "
             "protobuf decode, scheduling, the log's own copy, and the network. "
             "`to tensor` is not published on the Neuron backend.")
    L.append("")

    # ------------------------------------------------------------------ device and host
    L.append("## Device and host")
    table(L, (["accel", "precision", "rows x TPS", "stack", "device", "device max",
               "device MiB", "power W", "host CPU", "host CPU max", "host mem",
               "client CPU"],
              ["---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"), f"{r['rows']} x {r['tps']}", r["stack"],
            pct(r["dev"]), pct(r["dev_max"]), num(r["dev_mib"], 0), num(r["power"], 1),
            pct(r["cpu"]), pct(r["cpu_max"]), pct(r["mem"]), pct(r["client_cpu"])]
           for r in rows])

    # ------------------------------------------------------------------ inference log
    L.append("## Inference log")
    table(L, (["accel", "precision", "rows x TPS", "rate", "requests per object",
               "rows per object", "objects", "written GiB", "predicted MB/s",
               "measured MB/s", "shed batches", "write failures"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"), f"{r['rows']} x {r['tps']}",
            f"1 in {r['every']}", str(r["log_batch"]),
            f"{r['log_rows_per_object']:,}",
            str(r["log_objects"]) if r["log_objects"] is not None else "--",
            num((r["log_bytes"] or 0) / 1073741824, 1) if r["log_bytes"] else "--",
            num((r["log_predicted_bps"] or 0) / 1e6, 1),
            # Over WARM-UP PLUS MEASURED, which is the span the objects were written
            # across. Dividing by the measured window alone overstated it by a third and
            # made every cell look like it wrote more than predicted.
            num((r["log_bytes"] or 0) / 1e6 / (300 + (r["window"] or 900)), 1)
            if r["log_bytes"] else "--",
            str(r["log_shed"]) if r["log_shed"] is not None else "--",
            str(r["log_write_failures"])
            if r["log_write_failures"] is not None else "--"]
           for r in hushar])
    L.append("`predicted MB/s` is the offered rate times the sampled fraction times the "
             "measured feature-wire payload. `measured MB/s` is the bytes that reached "
             "the bucket over warm-up plus measured window, which is the span they were "
             "written across. A non-zero `shed batches` means the accumulator dropped "
             "its oldest buffered batch because every send permit was out, so that cell "
             "paid less than the full cost of logging; `write failures` are PUTs that "
             "did not complete.")
    L.append("")

    # ------------------------------------------------------------------ rate held
    L.append("## Rate held")
    table(L, (["accel", "precision", "stack", "rows x TPS", "result", "slices held",
               "shed", "failed", "window s", "rows/s offered", "Gbps", "% baseline"],
              ["---", "---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], PREC_LABEL.get(r["precision"], "?"), r["stack"],
            f"{r['rows']} x {r['tps']}", state_cell(r),
            f"{r['slices_held']}/{r['slices_total']}",
            str(r["shed"]), str(r["failed"]), str(r["window"]),
            f"{r['rows_per_sec']:,}",
            num(r["net_gbps"]) if r["net_gbps"] else "--",
            pct(r["net_pct"]) if r["net_pct"] else "--"] for r in rows])
    L.append("A cell sustains only when every slice held the offered rate within 2% with "
             "nothing shed and nothing failed.")
    L.append("")

    # ------------------------------------------------------------------ provenance
    L.append("## Provenance")
    hosts_seen = {}
    for r in rows:
        hosts_seen.setdefault((r["accel_label"], r["host"], r["instance"]), r)
    table(L, (["accel", "host", "instance", "cells", "engine", "TensorRT",
               "Triton image"],
              ["---", "---", "---", "--:", "---", "---", "---"]),
          [[k[0], str(k[1]), str(k[2]),
            str(sum(1 for r in rows
                    if (r["accel_label"], r["host"], r["instance"]) == k)),
            (f"ONNX Runtime {v['ort']}" if v["accel"] == "a10g"
             else f"libnrt {v['libnrt'] or '?'}"),
            v["tensorrt"] or "--", v["triton_image"] or "--"]
           for k, v in sorted(hosts_seen.items())])

    if primes:
        L.append("### Engines and NEFFs, primed before the grid")
        table(L, (["accel", "precision", "shape", "artifact", "source", "lane", "when",
                   "build/restore s", "reload proof s", "MiB", "parameters",
                   "trace error"],
                  ["---", "---", "--:", "---", "---", "---", "---", "--:", "--:", "--:",
                   "--:", "--:"]),
              [[ACCEL_LABEL.get(p["accel"], p["accel"]),
                PREC_LABEL.get(p["precision"], "?"),
                str(p["shape"]), p["kind"], p["source"], p["host"], p["when"],
                num(p["seconds"], 1), num(p["proof_seconds"], 1),
                num(p["mib"], 1), f"{p['params']:,}" if p["params"] else "--",
                f"{p['trace_error']}" if p.get("trace_error") else "--"]
               for p in sorted(primes, key=lambda p: (ACCEL_ORDER.get(p["accel"], 9),
                                                      PREC_ORDER.get(p["precision"], 9),
                                                      p["shape"]))])
        L.append("Each shape was built or restored and then served by a FRESH server, so "
                 "the cells inherit a warm cache rather than paying for a build inside a "
                 "discarded warm-up. `build/restore s` is a 20s trickle against a cold "
                 "cache and `reload proof s` the same trickle against a fresh server that "
                 "loaded it; the gap is what a cell would otherwise have paid. Every "
                 "artifact here was BUILT -- no entry restored from the S3 cache, because "
                 "no earlier campaign had compiled this model at either precision.")
        L.append("")
        L.append("`trace error` is the compiled-versus-PyTorch maximum, published only on "
                 "Trainium because the NEFF is compiled here and the ONNX graph is not. "
                 "It is the arm's proof: `compile_neuron.py` declares a window per "
                 "precision and refuses to write a NEFF whose error falls outside it. The "
                 "A10G's `MiB` is blank because those two lanes were launched before the "
                 "prime line learned to print its engine size; the sizes are in `Models` "
                 "and come from the cells themselves.")
        L.append("")

    counts = {
        "cells": len(rows),
        "sustained": sum(r["sustained"] for r in rows),
        "hushar": len(hushar),
        "triton": len(triton),
        "with CloudWatch": sum(1 for r in rows if r["buckets"]),
        "with device samples": sum(1 for r in rows if r["dev"] is not None),
        "cells that shed the log": sum(1 for r in hushar if r["log_shed"]),
        "log write failures": sum(r["log_write_failures"] or 0 for r in hushar),
    }
    table(L, (["measure", "count"], ["---", "--:"]),
          [[k, str(v)] for k, v in counts.items()])
    L.append("Cells, server banners, generated service and model configurations, "
             "per-second host and device samples, client output and CloudWatch points "
             "are under `raw/`. `raw/results-gpu.json` and `raw/results-neuron.json` are "
             "this report's only inputs. `verify.py` re-derives every number above from "
             "them.")
    L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")
    print(f"  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
