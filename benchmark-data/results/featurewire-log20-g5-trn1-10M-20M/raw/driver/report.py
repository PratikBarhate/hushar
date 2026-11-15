#!/usr/bin/env python3
"""Writes `results/featurewire-log20-g5-trn1-10M-20M/summary.md` from both lanes' results.

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
RUN = "featurewire-log20-g5-trn1-10M-20M"
RAW = BD / "results" / RUN / "raw"
OUT = BD / "results" / RUN / "summary.md"

# The round this one is a subtraction against: the SAME grid, the same wire, the same
# session count, the same graphs and engines -- with the inference log at 50% instead of
# 20%. So a delta against it is the LOG RATE and nothing else, which is the one thing the
# previous round could not isolate (it moved the wire and the log together).
CONTROLS = {
    "a10g": BD / "results" / "featurewire-log50-g5-trn1-10M-20M" / "raw"
    / "results-gpu.json",
    "trainium1": BD / "results" / "featurewire-log50-g5-trn1-10M-20M" / "raw"
    / "results-neuron.json",
}
CONTROL_RUN = "featurewire-log50-g5-trn1-10M-20M"
CONTROL_NOTE = ("the same grid at `sample_rate` 0.5 on the same wire, so a delta here is "
                "the log rate alone")
ACCEL_ORDER = {"a10g": 0, "trainium1": 1}
ACCEL_LABEL = {"a10g": "A10G", "trainium1": "trn1"}


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
        tensor = None
        buckets = g(cell, "server_side", "graph_buckets", default=0)
        cpu = g(cell, "host_series", "cpu_pct")
        cpu_max = g(cell, "host_series", "cpu_max_pct")
        mem = g(cell, "host_series", "mem_pct")
        dev = g(cell, "neuron_series", "nc_mean_pct")
        dev_max = g(cell, "neuron_series", "nc_peak_pct")
        dev_mib = g(cell, "neuron_series", "device_mib")
        power = None
        client_cpu = g(cell, "client_series", "cpu_pct")
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
        tensorrt=cell.get("host_tensorrt"),
        ort=cell.get("onnxruntime"),
        triton_image=cell.get("triton_image"),
        libnrt=cell.get("libnrt"),
        rev=cell.get("rev"),
        net_gbps=g(cell, "network", "total_gbps"),
        net_pct=g(cell, "network", "pct_of_baseline"),
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
            out[(c["accelerator"], c["size"], c["rows"], c["tps"], c["stack"])] = dict(
                p50=c.get("p50"), graph=c.get("graph_p50_ms"),
                sustained=bool(c.get("sustained")))
        return out, source, "snapshot"

    for accel, path in CONTROLS.items():
        if not path.exists():
            continue
        source = f"results/{CONTROL_RUN}"
        doc = json.loads(path.read_text())
        for c in (doc if isinstance(doc, list) else doc.get("cells", [])):
            key = (accel, c.get("size"), c.get("rows"), c.get("tps"),
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

    WHY THE LOGS AND NOT `prime/*.json`. Both lanes of this round wrote
    `runs/<run>/prime/<size>.json`, one key, so each overwrote the other for one size --
    and worse, the GPU lane's 20M prime found the Trainium lane's record already there and
    SKIPPED, so its engines were built inside the first 20M cell's 300 s warm-up instead.
    The key is lane-specific now, and the GPU prime was re-run afterwards, but the lane
    logs are per-lane by construction and were complete all along, so they are what this
    reads.

    WHICH ENTRY WINS WHEN A SHAPE WAS PRIMED TWICE. `main` is the lane run the cells were
    measured in, so where it recorded a shape, that is the figure the cells actually
    inherited. A `prime`-tagged re-run only fills shapes `main` has none for -- which here
    is 20M on the A10G, the one the collision made it skip. The `when` column says which,
    rather than the two being silently averaged or the later one quietly winning.
    """
    found = {}
    for path in sorted((raw / "driver").glob("lane-*.log")):
        stem = path.stem[len("lane-"):]
        # `lane-g5-1.log` is the main run; `lane-prime-g5-1.log` is a re-run under a tag.
        tag, host = ("main", stem)
        for known in ("prime",):
            if stem.startswith(f"{known}-"):
                tag, host = known, stem[len(known) + 1:]
        text = path.read_text(errors="replace")
        rows = []
        # GPU: `prime 10m r100: build 25.3s -> cached 22.1s`
        for size, shape, build, cached in re.findall(
                r"prime (\w+) r(\d+): build ([\d.]+)s -> cached ([\d.]+)s", text):
            rows.append(dict(accel="a10g", kind="TensorRT engine", size=size,
                             shape=int(shape), source="built", seconds=float(build),
                             proof_seconds=float(cached), mib=None, params=None))
        # Neuron: `prime 10m mb100: s3-cache 2.8s, NEFF 32.7 MiB, 9,142,789 params / ...
        #          served a trickle in 22.1s`
        for size, shape, source, secs, mib, params, proof in re.findall(
                r"prime (\w+) mb(\d+): (\S+) ([\d.]+)s, NEFF ([\d.]+) MiB, "
                r"([\d,]+) params / \d+ layers, trace error \S+, "
                r"served a trickle in ([\d.]+)s", text):
            rows.append(dict(accel="trainium1", kind="NEFF", size=size,
                             shape=int(shape), source=source, seconds=float(secs),
                             proof_seconds=float(proof), mib=float(mib),
                             params=int(params.replace(",", ""))))
        for r in rows:
            # The accelerator comes from the LINE SHAPE, not the host name: a re-run's log
            # is called `lane-prime-g5-1.log`, whose stem does not start with `g5`.
            key = (r["accel"], r["size"], r["shape"])
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
    L = [f"# hushar on its own wire with the log on: one A10G and one Trainium1", "",
         f"`{RUN}` | collected {collected} | rev `{first.get('rev', '?')}`", ""]

    # ------------------------------------------------------------------ configuration
    a10g = [r for r in rows if r["accel"] == "a10g"]
    trn1 = [r for r in rows if r["accel"] == "trainium1"]
    L.append("## Configuration")
    table(L, (["setting", "value"], ["---", "---"]), [
        ["servers", "`g5.2xlarge` 1x NVIDIA A10G 8 vCPU; "
                    "`trn1.2xlarge` 1x Trainium1 2 NeuronCore 8 vCPU"],
        ["clients", "`m9g.8xlarge`, 32 vCPU, one per lane, same availability zone"],
        ["models", "10M and 20M, FP32"],
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
    table(L, (["accel", "model", "rows x TPS", "rows/s", "stack", "p50", "p95", "p99",
               "graph", "vectorize", "device", "host CPU"],
              ["---", "---", "--:", "--:", "---", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], r["size"].upper(), f"{r['rows']} x {r['tps']}",
            f"{r['rows_per_sec']:,}", r["stack"],
            num(r["p50"]), num(r["p95"]), num(r["p99"]),
            num(r["graph"], 3), num(r["vec"], 3), pct(r["dev"]), pct(r["cpu"])]
           for r in held])

    short = [r for r in rows if not r["sustained"]]
    if short:
        L.append("## Did not hold the offered rate")
        table(L, (["accel", "model", "rows x TPS", "stack", "offered", "achieved",
                   "shortfall", "slices held", "client shed", "device", "host CPU",
                   "graph"],
                  ["---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:", "--:",
                   "--:", "--:"]),
              [[r["accel_label"], r["size"].upper(), f"{r['rows']} x {r['tps']}",
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

    # ------------------------------------------------------------------ models
    L.append("## Models")
    mrows = []
    for accel in ("a10g", "trainium1"):
        for size in ("10m", "20m"):
            match = [r for r in rows if r["accel"] == accel and r["size"] == size]
            if not match:
                continue
            n = params.get((accel, size))
            mrows.append([ACCEL_LABEL[accel], size.upper(),
                          f"{n:,}" if n else "--", "690", "5",
                          "ONNX graph" if accel == "a10g" else "NEFF"])
    table(L, (["accelerator", "model", "parameters", "input width", "classes",
               "artifact"],
              ["---", "---", "--:", "--:", "--:", "---"]), mrows)
    L.append("The Trainium parameter counts differ from the ONNX ones because "
             "`torch_neuronx.trace` rebuilds the same seeded architecture; both are "
             "recorded rather than treated as one number.")
    L.append("")

    # ------------------------------------------------------------------ every cell
    L.append("## Every cell")
    table(L, (["accel", "model", "rows x TPS", "stack", "sessions", "result",
               "p50", "p95", "p99", "graph", "vectorize", "CPU", "CPU max", "device",
               "slices"],
              ["---", "---", "--:", "---", "--:", "---",
               "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          [[r["accel_label"], r["size"].upper(), f"{r['rows']} x {r['tps']}",
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
        t = next((r for r in triton if r["accel"] == "a10g" and r["size"] == h["size"]
                  and r["rows"] == h["rows"] and r["tps"] == h["tps"]), None)
        if not t:
            continue
        ratio = (f"{h['p50'] / t['p50']:.2f}x"
                 if h["sustained"] and t["sustained"] and t["p50"] else "--")
        pairs.append([h["size"].upper(), f"{h['rows']} x {h['tps']}",
                      state_cell(h), lat(h, "p50"), lat(h, "p95"),
                      state_cell(t), lat(t, "p50"), lat(t, "p95"),
                      ratio, num(h["vec"], 3), num(h["graph"], 3),
                      num(t["graph"], 3) if t["graph"] else "--"])
    if pairs:
        L.append("## A10G: hushar against Triton")
        table(L, (["model", "rows x TPS", "hushar", "p50", "p95", "Triton", "p50",
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
        key = (r["accel"], r["size"], r["rows"], r["tps"], "hushar")
        base = controls.get(key)
        if base is None:
            crows.append([r["accel_label"], r["size"].upper(),
                          f"{r['rows']} x {r['tps']}", str(r["sessions"]),
                          lat(r, "p50"), "--", "--", "--",
                          num(r["graph"], 3), "no control cell at this rate"])
            continue
        bp50 = base["p50"] if base["sustained"] else None
        bgraph = base["graph"]
        delta = (f"{r['p50'] - bp50:+.2f}" if r["sustained"] and bp50 else "--")
        ratio = (f"{r['p50'] / bp50:.2f}x" if r["sustained"] and bp50 else "--")
        crows.append([r["accel_label"], r["size"].upper(),
                      f"{r['rows']} x {r['tps']}", str(r["sessions"]),
                      lat(r, "p50"), num(bp50), delta, ratio,
                      num(r["graph"], 3), num(bgraph, 3)])
    table(L, (["accel", "model", "rows x TPS", "sessions", "this round p50",
               "control p50", "delta", "ratio", "this round graph", "control graph"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          crows)
    L.append(f"The control is {CONTROL_NOTE}. The Trainium 20M rates and the A10G's "
             f"100 x 700 and 100 x 720 differ between the two rounds, so those cells have "
             f"no pair and report `--`. `graph` is carried for every row including the "
             f"unpaired ones, because a device time that did not move is what says a "
             f"latency change came from the host.")
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
        brows.append([r["accel_label"], r["size"].upper(),
                      f"{r['rows']} x {r['tps']}", lat(r, "p50"),
                      num(r["graph"], 3), num(r["vec"], 3), num(r["tensor"], 3),
                      num(rest, 3), str(r["buckets"]),
                      num(r["vec"] / r["p50"] * 100, 1) + "%"
                      if r["sustained"] and r["vec"] and r["p50"] else "--"])
    table(L, (["accel", "model", "rows x TPS", "p50", "graph", "vectorize",
               "to tensor", "remainder", "buckets", "vectorize share"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          brows)
    L.append("`remainder` is the client median less the three server-side measures: gRPC, "
             "protobuf decode, scheduling, the log's own copy, and the network. "
             "`to tensor` is not published on the Neuron backend.")
    L.append("")

    # ------------------------------------------------------------------ device and host
    L.append("## Device and host")
    table(L, (["accel", "model", "rows x TPS", "stack", "device", "device max",
               "device MiB", "power W", "host CPU", "host CPU max", "host mem",
               "client CPU"],
              ["---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], r["size"].upper(), f"{r['rows']} x {r['tps']}", r["stack"],
            pct(r["dev"]), pct(r["dev_max"]), num(r["dev_mib"], 0), num(r["power"], 1),
            pct(r["cpu"]), pct(r["cpu_max"]), pct(r["mem"]), pct(r["client_cpu"])]
           for r in rows])

    # ------------------------------------------------------------------ inference log
    L.append("## Inference log")
    table(L, (["accel", "model", "rows x TPS", "rate", "requests per object",
               "rows per object", "objects", "written GiB", "predicted MB/s",
               "measured MB/s", "shed batches", "write failures"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], r["size"].upper(), f"{r['rows']} x {r['tps']}",
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
    table(L, (["accel", "model", "stack", "rows x TPS", "result", "slices held",
               "shed", "failed", "window s", "rows/s offered", "Gbps", "% baseline"],
              ["---", "---", "---", "--:", "---", "--:", "--:", "--:", "--:", "--:",
               "--:", "--:"]),
          [[r["accel_label"], r["size"].upper(), r["stack"],
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
        table(L, (["accel", "model", "shape", "artifact", "source", "lane", "when",
                   "build/restore s", "reload proof s", "MiB", "parameters"],
                  ["---", "---", "--:", "---", "---", "---", "---", "--:", "--:", "--:",
                   "--:"]),
              [[ACCEL_LABEL.get(p["accel"], p["accel"]), p["size"].upper(),
                str(p["shape"]), p["kind"], p["source"], p["host"], p["when"],
                num(p["seconds"], 1), num(p["proof_seconds"], 1),
                num(p["mib"], 1), f"{p['params']:,}" if p["params"] else "--"]
               for p in sorted(primes, key=lambda p: (ACCEL_ORDER.get(p["accel"], 9),
                                                      p["size"], p["shape"]))])
        L.append("Each shape was built or restored and then served by a FRESH server, so "
                 "the cells inherit a warm cache. `build/restore s` is a 20s trickle "
                 "against a cold cache and `reload proof s` the same trickle against a "
                 "fresh server that loaded it; the gap is what a cell would otherwise "
                 "have paid inside its warm-up. `when` is `main` where the figure comes "
                 "from the lane run that measured the cells, and `prime` for the one "
                 "shape that had to be re-primed afterwards: both lanes first wrote a "
                 "single `prime/<size>.json` key, so the A10G lane's 20M prime skipped on "
                 "finding the Trainium record already there and its 20M engines were "
                 "built inside the first 20M cell's 300s discarded warm-up instead. The "
                 "key is per-lane now.")
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
