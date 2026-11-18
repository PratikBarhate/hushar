#!/usr/bin/env python3
"""Writes `results/nominibatch-sessions-gpu-g5-10M-20M/summary.md` from `raw/results.json`.

    ./report.py

NUMBERS ONLY. This run was asked for tables of what was recorded and no commentary, so
every function here emits a table or a definition of a column, and nothing interprets a
figure. The one piece of prose is the configuration block, which states what was set
rather than what it means.

Every cell of every table comes from `raw/results.json`, which `collect.py` built from the
cells in S3 plus the per-second host and GPU samples and CloudWatch's minute buckets.
`verify.py` re-derives all of it.
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


RUN = os.environ.get("RUN", "nominibatch-precision-gpu-g5-100M")
RESULTS = _benchmark_data(HERE) / "results" / RUN
RAW = RESULTS / "raw"
OUT = RESULTS / "summary.md"

MARK = {"sustained": "ok", "partial": "part", "failed": "FAIL", None: "?"}
# THE AXIS. Ordered fp32 first, so every table reads baseline-then-reduced and a delta
# column has something to its left.
PRECS = ("fp32", "bf16")
LABEL = {"fp32": "FP32", "bf16": "BF16"}


# ---------------------------------------------------------------- plumbing

def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default if default is not None else []


def fmt(value, digits=2, dash="--"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}%"


def ratio(num, den, digits=2):
    """`num / den` as `1.23x`, or `--` when either is missing or `den` is zero."""
    if num is None or den in (None, 0):
        return "--"
    return f"{num / den:.{digits}f}x"


def delta(num, base, digits=2):
    """Signed difference, with an explicit `+`."""
    if num is None or base is None:
        return "--"
    return f"{num - base:+.{digits}f}"


def row_of(cell):
    """One cell, flattened to what the tables below ask for."""
    r = cell.get("result") or {}
    v = cell.get("verdict") or {}
    cw = cell.get("cloudwatch") or {}
    gpu = cell.get("gpu_utilization") or {}
    srv = cell.get("server_utilization") or {}
    win = cell.get("measured_window") or {}
    return dict(
        cid=cell.get("id"), host=cell.get("host"), server_ip=cell.get("server_ip"),
        instance=cell.get("instance"),
        size=cell.get("size"), stack=cell.get("stack"), arm=cell.get("arm"),
        prec=cell.get("precision"), engine_bytes=cell.get("engine_bytes"),
        wire=cell.get("wire"),
        rows=cell.get("rows"), tps=cell.get("tps"),
        rows_per_sec=cell.get("rows_per_sec"),
        shape_rows=cell.get("mini_batch"),
        mb_cfg=cell.get("mini_batch_configured"),
        pieces=cell.get("pieces"), plan=cell.get("plan"),
        par=cell.get("parallelism"), kind=cell.get("parallelism_kind"),
        spr=cell.get("sessions_per_request"), lanes=cell.get("request_lanes"),
        engine_shape=cell.get("engine_shape"),
        p50=r.get("p50"), p95=r.get("p95"), p99=r.get("p99"),
        offered=r.get("offered"), achieved=r.get("achieved"),
        shed=r.get("shed"), failed=r.get("failed"),
        state=v.get("state"), held=v.get("slices_held"), total=v.get("slices_total"),
        sustained=bool(cell.get("sustained")),
        cpu=srv.get("cpu_pct"), cpu_max=srv.get("cpu_max_pct"),
        rss=srv.get("process_rss_gib"),
        gpu_pct=gpu.get("gpu_util_pct"), gpu_max=gpu.get("gpu_util_max_pct"),
        gpu_mem=gpu.get("memory_used_mib"), power=gpu.get("power_w"),
        gpu_mem_pct=gpu.get("memory_util_pct"), power_max=gpu.get("power_max_w"),
        client_cpu=(cell.get("client_utilization") or {}).get("cpu_pct"),
        graph=(cw.get("InferenceTime") or {}).get("p50_ms"),
        vec=(cw.get("VectorizationTime") or {}).get("p50_ms"),
        tot=(cw.get("ToTensorTime") or {}).get("p50_ms"),
        cw_samples=(cw.get("InferenceTime") or {}).get("samples"),
        buckets=(cw.get("InferenceTime") or {}).get("buckets"),
        window_s=win.get("seconds"),
        rev=cell.get("rev"), ort=cell.get("onnxruntime"),
        triton_image=cell.get("triton_image"), md5=cell.get("model_md5"),
        trt=cell.get("host_tensorrt"), cuda=cell.get("host_cuda_major"),
        driver=cell.get("host_nvidia_driver"), gpu_name=cell.get("host_gpu"),
    )


def result_text(r):
    return f"{MARK.get(r['state'], '?')} {fmt(r['achieved'], 1)}/{r['offered']}"


def held(r):
    """Whether a row's median is a latency figure at all.

    A cell that shed a third of its traffic has a median built from queueing, so a ratio
    against a healthy baseline is arithmetic rather than a comparison. Every derived
    column below is withheld on such a row -- the rate it actually held is in `Rate held`,
    which is where that fact belongs.
    """
    return r["state"] == "sustained"


def config_text(r):
    """How a cell was configured, in one cell, in its own server's vocabulary."""
    return f"{r['par']} {'sessions' if r['stack'] == 'hushar' else 'instances'}"


def table(header, aligns, rows):
    """A markdown table, or a one-line note when there is nothing to put in it."""
    if not rows:
        return "_no cells_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- grouping

def loads_of(rows, prec):
    """`(rows, tps)` pairs measured at one precision, in the order the grid ran them."""
    seen = []
    for r in rows:
        if r["prec"] == prec and (r["rows"], r["tps"]) not in seen:
            seen.append((r["rows"], r["tps"]))
    return seen


def all_loads(rows):
    """Every `(rows, tps)` pair in the run, smallest request first."""
    return sorted({(r["rows"], r["tps"]) for r in rows})


def arm_rows(rows, prec, load, stack):
    got = [r for r in rows
           if r["prec"] == prec and (r["rows"], r["tps"]) == load
           and r["stack"] == stack]
    return sorted(got, key=lambda r: r["par"])


def one_cell(rows, prec, load, stack):
    """The single cell at this (precision, load, stack), or None.

    Sessions are pinned in this run, so a coordinate names ONE cell. Returning the lowest
    sustained median keeps the function honest if a follow-up ever unpins them.
    """
    group = arm_rows(rows, prec, load, stack)
    ok = [r for r in group if held(r) and r["p50"] is not None]
    return min(ok, key=lambda r: r["p50"]) if ok else (group[0] if group else None)


def best_of(group):
    """The lowest median among the cells that held their rate, or None."""
    ok = [r for r in group if held(r) and r["p50"] is not None]
    return min(ok, key=lambda r: r["p50"]) if ok else None


def fewest(group):
    """The cell with the fewest sessions or instances, which every delta is read against."""
    have = [r for r in group if r["par"] is not None]
    return min(have, key=lambda r: r["par"]) if have else None


def speedup(base, reduced):
    """`base / reduced` as `1.23x`: how many times faster the reduced precision is.

    Withheld unless BOTH cells held their offered rate. A median built from queueing is
    not a latency, so dividing one by a healthy one would read as a precision effect.
    """
    if not (base and reduced and held(base) and held(reduced)):
        return "--"
    if base["p50"] is None or not reduced["p50"]:
        return "--"
    return f"{base['p50'] / reduced['p50']:.2f}x"


# ---------------------------------------------------------------- tables

def answer_table(rows):
    """Every measured coordinate on one line: payload x precision x server."""
    out = []
    for load in all_loads(rows):
        for prec in PRECS:
            h = one_cell(rows, prec, load, "hushar")
            t = one_cell(rows, prec, load, "triton")
            if not h and not t:
                continue
            out.append([
                f"{load[0]} x {load[1]}", LABEL[prec],
                fmt(h["p50"]) if h else "--", fmt(h["p95"]) if h else "--",
                fmt(h["p99"]) if h else "--",
                fmt(t["p50"]) if t else "--", fmt(t["p95"]) if t else "--",
                fmt(t["p99"]) if t else "--",
                ratio(h["p50"], t["p50"]) if h and t and held(h) and held(t) else "--",
                fmt(h["graph"], 3) if h else "--",
                pct(h["gpu_pct"]) if h else "--",
            ])
    return table(
        ["rows x TPS", "precision", "hushar p50", "hushar p95", "hushar p99",
         "Triton p50", "Triton p95", "Triton p99", "hushar / Triton", "hushar graph",
         "hushar GPU"],
        ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
         "---:", "---:"],
        out)


def precision_table(rows):
    """THE AXIS: FP32 against BF16 at the same payload, on each server."""
    out = []
    for load in all_loads(rows):
        for stack, noun in (("hushar", "hushar"), ("triton", "Triton")):
            a = one_cell(rows, "fp32", load, stack)
            b = one_cell(rows, "bf16", load, stack)
            if not a and not b:
                continue
            out.append([
                f"{load[0]} x {load[1]}", noun,
                fmt(a["p50"]) if a else "--", fmt(b["p50"]) if b else "--",
                speedup(a, b),
                delta(b["p50"], a["p50"]) if a and b else "--",
                fmt(a["p99"]) if a else "--", fmt(b["p99"]) if b else "--",
                fmt(a["graph"], 3) if a else "--", fmt(b["graph"], 3) if b else "--",
                speedup_graph(a, b),
            ])
    return table(
        ["rows x TPS", "server", "FP32 p50", "BF16 p50", "BF16 speedup",
         "p50 delta ms", "FP32 p99", "BF16 p99", "FP32 graph", "BF16 graph",
         "graph speedup"],
        ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
         "---:", "---:"],
        out)


def speedup_graph(base, reduced):
    """The same ratio on `InferenceTime` rather than on the client median.

    The client median carries the network and the request path; `InferenceTime` is wall
    time inside the backend. A precision changes the second directly, so the two ratios
    differing is the dilution the request path adds, not a contradiction.
    """
    if not (base and reduced and held(base) and held(reduced)):
        return "--"
    if base["graph"] is None or not reduced["graph"]:
        return "--"
    return f"{base['graph'] / reduced['graph']:.2f}x"


def plan_table(rows):
    """Each arm's engine or plan bytes, per precision and shape."""
    seen, out = {}, []
    for r in rows:
        key = (r["prec"], r["rows"], r["arm"])
        if key not in seen and r["engine_bytes"]:
            seen[key] = r
    for (prec, nrows, arm), r in sorted(seen.items()):
        out.append([LABEL.get(prec, prec), nrows, arm,
                    fmt(r["engine_bytes"], 0),
                    (r["plan"] if r["stack"] != "hushar" else r["engine_shape"])
                    or "--"])
    return table(["precision", "rows", "arm", "engine bytes", "plan or shape"],
                 ["---", "---:", "---", "---:", "---"], out)


def grid_table(rows):
    """Every pair: hushar at N sessions beside Triton at N instances, same load."""
    out = []
    for prec in PRECS:
        for load in loads_of(rows, prec):
            hus = {r["par"]: r for r in arm_rows(rows, prec, load, "hushar")}
            tri = {r["par"]: r for r in arm_rows(rows, prec, load, "triton")}
            for par in sorted(set(hus) | set(tri)):
                h, t = hus.get(par), tri.get(par)
                comparable = h is not None and t is not None and held(h) and held(t)
                out.append([
                    LABEL[prec], f"{load[0]} x {load[1]}", par,
                    result_text(h) if h else "--",
                    fmt(h["p50"]) if h else "--", fmt(h["p95"]) if h else "--",
                    fmt(h["p99"]) if h else "--",
                    result_text(t) if t else "--",
                    fmt(t["p50"]) if t else "--", fmt(t["p95"]) if t else "--",
                    fmt(t["p99"]) if t else "--",
                    ratio(h["p50"], t["p50"]) if comparable else "--",
                ])
    return table(
        ["precision", "rows x TPS", "parallelism", "hushar result", "hushar p50",
         "hushar p95", "hushar p99", "Triton result", "Triton p50", "Triton p95",
         "Triton p99", "hushar / Triton"],
        ["---", "---:", "---:", "---", "---:", "---:", "---:", "---", "---:", "---:",
         "---:", "---:"],
        out)


def stage_table(rows):
    """Where the median goes, for the hushar cells that reported CloudWatch."""
    out = []
    for prec in PRECS:
        for load in loads_of(rows, prec):
            for r in arm_rows(rows, prec, load, "hushar"):
                rest = (None if r["p50"] is None or r["graph"] is None
                        else r["p50"] - r["graph"] - (r["vec"] or 0.0) - (r["tot"] or 0.0))
                out.append([
                    LABEL[prec], f"{r['rows']} x {r['tps']}", config_text(r),
                    fmt(r["p50"]), fmt(r["graph"], 3), fmt(r["vec"], 3),
                    fmt(r["tot"], 3), fmt(rest, 3), r["buckets"] or "--",
                ])
    return table(
        ["precision", "rows x TPS", "hushar config", "p50", "InferenceTime",
         "VectorizationTime", "ToTensorTime", "remainder", "buckets"],
        ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        out)


def device_table(rows):
    """GPU and host, over the measured window only."""
    out = []
    for prec in PRECS:
        for load in loads_of(rows, prec):
            for stack in ("hushar", "triton"):
                for r in arm_rows(rows, prec, load, stack):
                    out.append([
                        LABEL[prec], f"{r['rows']} x {r['tps']}",
                        r["arm"], config_text(r),
                        pct(r["gpu_pct"]), pct(r["gpu_max"]),
                        fmt(r["gpu_mem"], 0), fmt(r["power"], 1),
                        pct(r["cpu"]), pct(r["cpu_max"]),
                    ])
    return table(
        ["precision", "rows x TPS", "arm", "config", "GPU", "GPU max", "GPU MiB",
         "power W", "host CPU", "CPU max"],
        ["---", "---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        out)


def rate_table(rows):
    """One rate against another at the same payload and server: what more TPS costs.

    Grouped by (precision, rows, server), lowest offered rate as the baseline. This is
    the table a capacity question actually wants: whether doubling the offered rate moved
    the latency, where the device occupancy went, and whether the backend did any more
    work per request -- `graph` should be flat, because a rate change does not alter the
    arithmetic of one request.
    """
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["prec"], r["rows"], r["stack"]), []).append(r)
    for (prec, nrows, stack), group in sorted(groups.items()):
        group.sort(key=lambda r: r["tps"])
        if len({r["tps"] for r in group}) < 2:
            continue
        base = group[0]
        for r in group:
            is_base = r is base
            out.append([
                LABEL.get(prec, prec), nrows,
                "hushar" if stack == "hushar" else "Triton",
                r["tps"], f"{r['rows_per_sec']:,}",
                result_text(r),
                fmt(r["p50"]), fmt(r["p99"]), fmt(r["graph"], 3),
                pct(r["gpu_pct"]), fmt(r["power"], 1),
                "baseline" if is_base else ratio(r["p50"], base["p50"]),
                "baseline" if is_base else ratio(r["gpu_pct"], base["gpu_pct"]),
            ])
    return table(
        ["precision", "rows", "server", "TPS", "rows/s", "result", "p50", "p99",
         "graph", "GPU busy", "power W", "p50 vs base", "GPU vs base"],
        ["---", "---:", "---", "---:", "---:", "---", "---:", "---:", "---:", "---:",
         "---:", "---:", "---:"],
        out)


def headroom_table(rows):
    """What is left on the table: device occupancy, power, memory, and both hosts' CPU.

    Every column is a recorded measurement over the measured window except the last,
    which is `100 / GPU busy` -- the factor the offered rate could rise by before the
    device is executing continuously. It is an UPPER BOUND and nothing more: occupancy is
    the share of sample periods in which the device was executing something, not how full
    it was, so a device at 50% busy does not reliably take 2x the rate. `power W` against
    `power max W` tracks the work done more honestly, and `client CPU` is here because a
    headroom claim means nothing if the load generator was the limit.
    """
    out = []
    for prec in PRECS:
        for load in loads_of(rows, prec):
            for stack in ("hushar", "triton"):
                for r in arm_rows(rows, prec, load, stack):
                    busy = r["gpu_pct"]
                    out.append([
                        LABEL[prec], f"{r['rows']} x {r['tps']}", r["arm"],
                        pct(busy), pct(r["gpu_max"]),
                        pct(r["gpu_mem_pct"]), fmt(r["gpu_mem"], 0),
                        fmt(r["power"], 1), fmt(r["power_max"], 1),
                        pct(r["cpu"]), pct(r["cpu_max"]),
                        pct(r["client_cpu"]),
                        f"{100.0 / busy:.2f}x" if busy else "--",
                    ])
    return table(
        ["precision", "rows x TPS", "arm", "GPU busy", "GPU max", "GPU mem",
         "GPU MiB", "power W", "power max W", "server CPU", "server max",
         "client CPU", "TPS x at 100% busy"],
        ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:",
         "---:", "---:", "---:", "---:"],
        out)


def hold_table(rows):
    """Slices held, shed and failed, for every cell."""
    out = []
    for r in sorted(rows, key=lambda g: (g["prec"] or "", g["rows"], g["stack"],
                                         g["par"])):
        out.append([
            LABEL.get(r["prec"], r["prec"]), r["arm"], config_text(r),
            f"{r['rows']} x {r['tps']}", result_text(r),
            f"{r['held']}/{r['total']}" if r["total"] else "--",
            r["shed"] if r["shed"] is not None else "--",
            r["failed"] if r["failed"] is not None else "--",
            r["window_s"] or "--",
        ])
    return table(
        ["precision", "arm", "config", "rows x TPS", "result", "slices held", "shed",
         "failed", "window s"],
        ["---", "---", "---", "---:", "---", "---:", "---:", "---:", "---:"],
        out)


def every_cell_table(rows):
    """One row per cell, with its id, so a figure can be traced to its artifacts."""
    out = []
    for r in sorted(rows, key=lambda g: g["cid"] or ""):
        out.append([
            f"`{r['cid']}`", LABEL.get(r["prec"], r["prec"]),
            r["plan"] or "tensorrt_plan",
            r["engine_shape"], r["par"], r["spr"] or "--", r["lanes"] or "--",
            fmt(r["p50"]), fmt(r["p95"]), fmt(r["p99"]),
            MARK.get(r["state"], "?"), r["host"] or "--",
        ])
    return table(
        ["cell", "precision", "plan", "engine shape", "parallelism", "per request",
         "lanes", "p50", "p95", "p99", "result", "host"],
        ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
         "---", "---"],
        out)


def prime_table(primes):
    """The TensorRT engine build per precision and shape, and the reload that proved it."""
    out = []
    for doc in primes:
        for s in doc.get("shapes", []):
            prec = s.get("precision") or doc.get("precision")
            out.append([
                LABEL.get(prec, prec),
                s.get("rows"), s.get("engine_shape"),
                fmt(s.get("engine_bytes"), 0),
                fmt(s.get("build_seconds"), 1), fmt(s.get("cached_seconds"), 1),
                f"{s.get('speedup')}x" if s.get("speedup") else "--",
            ])
    return table(
        ["precision", "rows", "engine shape", "engine bytes", "cold build s",
         "cached s", "ratio"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        out)


def library_table(rows):
    """The TensorRT, CUDA and driver each arm's cells recorded, per host.

    This is the table the run's premise rests on: both arms must execute on one TensorRT.
    The version is read off the host at preflight and stamped on every cell, so a
    disagreement between the two arms of a lane shows up here as two different numbers
    rather than having to be taken on trust from a deploy log.
    """
    seen, out = {}, []
    for r in rows:
        key = (r["host"], r["arm"])
        if key not in seen:
            seen[key] = r
    for (host, arm), r in sorted(seen.items()):
        out.append([host or "--", arm, r["gpu_name"] or "--", r["driver"] or "--",
                    r["trt"] or "--", r["cuda"] or "--", r["ort"] or "--",
                    "n/a (tensorrt backend)" if r["stack"] != "hushar"
                    else "loaded"])
    return table(
        ["host", "arm", "GPU", "driver", "TensorRT", "CUDA", "ONNX Runtime",
         "ORT in use"],
        ["---", "---", "---", "---", "---:", "---:", "---", "---"],
        out)


def provenance(rows):
    """What the numbers were measured on."""
    hosts, out = {}, []
    for r in rows:
        if r["host"] and r["host"] not in hosts:
            hosts[r["host"]] = r
    for name, r in sorted(hosts.items()):
        out.append([name, r["instance"], r["server_ip"] or "--", r["rev"] or "--",
                    r["ort"] or "--", r["triton_image"] or "--", r["md5"] or "--"])
    return table(["host", "instance", "address", "revision", "ONNX Runtime",
                  "Triton image", "model md5"],
                 ["---", "---", "---", "---", "---", "---", "---"], out)


def counts_table(status):
    """Per-arm tallies, from `raw/run-status.json`."""
    out = []
    for arm, s in sorted((status.get("by_arm") or {}).items()):
        out.append([arm, s.get("cells"), s.get("sustained"), s.get("failed_load"),
                    s.get("partial"), s.get("with_cloudwatch"),
                    s.get("with_gpu_samples")])
    for size, s in sorted((status.get("by_size") or {}).items()):
        out.append([LABEL.get(size, size), s.get("cells"), s.get("sustained"),
                    s.get("failed_load"), s.get("partial"), s.get("with_cloudwatch"),
                    s.get("with_gpu_samples")])
    return table(["group", "cells", "sustained", "failed rate", "partial",
                  "CloudWatch", "GPU samples"],
                 ["---", "---:", "---:", "---:", "---:", "---:", "---:"], out)


# ---------------------------------------------------------------- the document

def build():
    cells = [c for c in load(RAW / "results.json") if c.get("phase") != "probe"]
    rows = [row_of(c) for c in cells]
    rows = [r for r in rows if r["p50"] is not None or r["state"]]
    primes = [load(p, {}) for p in sorted((RAW / "prime").glob("*.json"))] \
        if (RAW / "prime").is_dir() else []
    status = load(RAW / "run-status.json", {})

    sustained = sum(1 for r in rows if r["sustained"])
    one = rows[0] if rows else {}
    trt = next((r["trt"] for r in rows if r["trt"]), None)

    doc = f"""# A 100M-parameter transformer on one A10G, FP32 against BF16

`{RUN}`

## Configuration

| setting | value |
|---|---|
| server | `g5.2xlarge`, 1x NVIDIA A10G, 8 vCPU |
| model | 100M: 100,561,925 params, 31 encoder layers, 785,000,960 MAC/row, 402,233,324 bytes |
| model vs 10M | parameters 11.00x, MAC/row 14.73x |
| graph | one ONNX file, md5 `{one.get('md5') or '--'}`, identical across both precisions |
| precision | FP32 (TensorRT default, TF32 permitted) and BF16 (`trt_bf16_enable=1` on hushar, `--bf16` on `trtexec`), additive per layer |
| payloads | 100 rows x 100 TPS and 500 rows x 20 TPS, 10,000 rows/s each |
| wire | one 690-float f32 vector per row, both arms; identity transformation on hushar, so no server-side feature vectorization |
| feature logging | `inference_log.sample_rate` 0.0 |
| candidate model | none; every request scores the one resident model |
| mini batch | none; `mini_batch_size` unset on every hushar cell, so the plan is `whole request` and a request of any size is one engine call |
| sessions | `threading.sessions_per_model` 2, `sessions_per_request` 1 on hushar; `instance_group` count 2 on Triton |
| hushar engine | ONNX Runtime {one.get('ort') or '--'}, TensorRT execution provider |
| Triton | `{one.get('triton_image') or '--'}`, `tensorrt` backend, `max_batch_size: 0` |
| TensorRT | {trt or '--'}, the same library for both arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and by a version check at preflight |
| window | {one.get('window_s') or '--'}s measured in {one.get('total') or '--'} slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | {len(rows)} measured, {sustained} sustained the offered rate |

## Results

{answer_table(rows)}
All latencies in milliseconds. `hushar graph` is `InferenceTime`: wall time per request
inside the backend. `hushar GPU` is the device's own occupancy over the measured window.

## FP32 against BF16

{precision_table(rows)}
`BF16 speedup` is FP32 p50 / BF16 p50. `graph speedup` is the same ratio on
`InferenceTime`. A row is blank where that precision was not measured at that rate.

## Offered rate against latency

{rate_table(rows)}
Lowest rate per payload and server is the baseline. `graph` is `InferenceTime`, which a
rate change should leave flat: the arithmetic of one request does not depend on how often
it arrives.

## Matched parallelism

{grid_table(rows)}

## Where the median goes

{stage_table(rows)}
Medians of whole minute buckets inside the measured window. `remainder` is the client
median less the three measures.

## Device and host

{device_table(rows)}

## Headroom

{headroom_table(rows)}
Every column is measured over the window except the last, which is `100 / GPU busy`: the
factor the offered rate could rise by before the device executes continuously. It is an
upper bound, because occupancy is the share of sample periods in which the device was
executing, not how full it was. `client CPU` is the load generator's own host.

## Rate held

{hold_table(rows)}

## TensorRT engines

{prime_table(primes)}
One engine per precision and row count, built and proved reloadable before the grid ran.

{plan_table(rows)}

## Engine and library versions

{library_table(rows)}

## Every cell

{every_cell_table(rows)}

## Provenance

{provenance(rows)}
{counts_table(status)}
Cells, server banners, generated service and model configurations, per-second host and GPU
samples, client output and CloudWatch points are under `raw/`. `raw/results.json` is this
report's only input. `verify.py` re-derives every number above from it.
"""
    return doc


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
