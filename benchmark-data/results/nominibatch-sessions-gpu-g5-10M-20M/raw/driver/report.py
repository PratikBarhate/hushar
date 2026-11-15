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


RUN = os.environ.get("RUN", "nominibatch-sessions-gpu-g5-10M-20M")
RESULTS = _benchmark_data(HERE) / "results" / RUN
RAW = RESULTS / "raw"
OUT = RESULTS / "summary.md"

MARK = {"sustained": "ok", "partial": "part", "failed": "FAIL", None: "?"}
SIZES = ("10m", "20m")
LABEL = {"10m": "10M", "20m": "20M"}


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

def loads_of(rows, size):
    """`(rows, tps)` pairs measured for one size, in the order the grid ran them."""
    seen = []
    for r in rows:
        if r["size"] == size and (r["rows"], r["tps"]) not in seen:
            seen.append((r["rows"], r["tps"]))
    return seen


def arm_rows(rows, size, load, stack):
    got = [r for r in rows
           if r["size"] == size and (r["rows"], r["tps"]) == load and r["stack"] == stack]
    return sorted(got, key=lambda r: r["par"])


def best_of(group):
    """The lowest median among the cells that held their rate, or None."""
    ok = [r for r in group if held(r) and r["p50"] is not None]
    return min(ok, key=lambda r: r["p50"]) if ok else None


def fewest(group):
    """The cell with the fewest sessions or instances, which every delta is read against."""
    have = [r for r in group if r["par"] is not None]
    return min(have, key=lambda r: r["par"]) if have else None


# ---------------------------------------------------------------- tables

def answer_table(rows):
    """Each arm at its own best sustained parallelism, and the ratio between them.

    Best rather than matched, because the two servers do not have to peak at the same
    count and pinning one would understate whichever arm did not. The matched-parallelism
    comparison is the Grid table below; this one is each server at its own setting.
    """
    out = []
    for size in SIZES:
        for load in loads_of(rows, size):
            hus = arm_rows(rows, size, load, "hushar")
            tri = arm_rows(rows, size, load, "triton")
            if not hus and not tri:
                continue
            hb, tb = best_of(hus), best_of(tri)
            out.append([
                LABEL[size], load[0], load[1],
                fmt(hb["p50"]) if hb else "--",
                hb["par"] if hb else "--",
                fmt(tb["p50"]) if tb else "--",
                tb["par"] if tb else "--",
                ratio(hb["p50"], tb["p50"]) if hb and tb else "--",
                fmt(hb["graph"], 3) if hb else "--",
                f"{sum(1 for r in hus if held(r))}/{len(hus)}",
                f"{sum(1 for r in tri if held(r))}/{len(tri)}",
            ])
    return table(
        ["model", "rows", "TPS", "hushar p50", "at sessions", "Triton p50",
         "at instances", "hushar / Triton", "hushar graph", "hushar held",
         "Triton held"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
         "---:", "---:"],
        out)


def grid_table(rows):
    """Every pair: hushar at N sessions beside Triton at N instances, same load."""
    out = []
    for size in SIZES:
        for load in loads_of(rows, size):
            hus = {r["par"]: r for r in arm_rows(rows, size, load, "hushar")}
            tri = {r["par"]: r for r in arm_rows(rows, size, load, "triton")}
            for par in sorted(set(hus) | set(tri)):
                h, t = hus.get(par), tri.get(par)
                comparable = h is not None and t is not None and held(h) and held(t)
                out.append([
                    LABEL[size], f"{load[0]} x {load[1]}", par,
                    result_text(h) if h else "--",
                    fmt(h["p50"]) if h else "--", fmt(h["p95"]) if h else "--",
                    fmt(h["p99"]) if h else "--",
                    result_text(t) if t else "--",
                    fmt(t["p50"]) if t else "--", fmt(t["p95"]) if t else "--",
                    fmt(t["p99"]) if t else "--",
                    ratio(h["p50"], t["p50"]) if comparable else "--",
                ])
    return table(
        ["model", "rows x TPS", "parallelism", "hushar result", "hushar p50",
         "hushar p95", "hushar p99", "Triton result", "Triton p50", "Triton p95",
         "Triton p99", "hushar / Triton"],
        ["---", "---:", "---:", "---", "---:", "---:", "---:", "---", "---:", "---:",
         "---:", "---:"],
        out)


def parallelism_table(rows, stack, noun):
    """One arm at each parallelism, against its own fewest."""
    out = []
    for size in SIZES:
        for load in loads_of(rows, size):
            group = arm_rows(rows, size, load, stack)
            base = fewest(group)
            for r in group:
                out.append([
                    LABEL[size], f"{r['rows']} x {r['tps']}", r["par"],
                    result_text(r), fmt(r["p50"]), fmt(r["p95"]), fmt(r["p99"]),
                    "baseline" if r is base
                    else (delta(r["p50"], base["p50"])
                          if base and held(r) and held(base) else "--"),
                ])
    return table(
        ["model", "rows x TPS", noun, "result", "p50", "p95", "p99",
         f"p50 vs fewest {noun}"],
        ["---", "---:", "---:", "---", "---:", "---:", "---:", "---:"],
        out)


def stage_table(rows):
    """Where the median goes, for the hushar cells that reported CloudWatch."""
    out = []
    for size in SIZES:
        for load in loads_of(rows, size):
            for r in arm_rows(rows, size, load, "hushar"):
                rest = (None if r["p50"] is None or r["graph"] is None
                        else r["p50"] - r["graph"] - (r["vec"] or 0.0) - (r["tot"] or 0.0))
                out.append([
                    LABEL[size], f"{r['rows']} x {r['tps']}", config_text(r),
                    fmt(r["p50"]), fmt(r["graph"], 3), fmt(r["vec"], 3),
                    fmt(r["tot"], 3), fmt(rest, 3), r["buckets"] or "--",
                ])
    return table(
        ["model", "rows x TPS", "hushar config", "p50", "InferenceTime",
         "VectorizationTime", "ToTensorTime", "remainder", "buckets"],
        ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        out)


def device_table(rows):
    """GPU and host, over the measured window only."""
    out = []
    for size in SIZES:
        for load in loads_of(rows, size):
            for stack in ("hushar", "triton"):
                for r in arm_rows(rows, size, load, stack):
                    out.append([
                        LABEL[size], f"{r['rows']} x {r['tps']}",
                        r["arm"], config_text(r),
                        pct(r["gpu_pct"]), pct(r["gpu_max"]),
                        fmt(r["gpu_mem"], 0), fmt(r["power"], 1),
                        pct(r["cpu"]), pct(r["cpu_max"]),
                    ])
    return table(
        ["model", "rows x TPS", "arm", "config", "GPU", "GPU max", "GPU MiB",
         "power W", "host CPU", "CPU max"],
        ["---", "---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        out)


def hold_table(rows):
    """Slices held, shed and failed, for every cell."""
    out = []
    for r in sorted(rows, key=lambda g: (g["size"], g["rows"], g["stack"], g["par"])):
        out.append([
            LABEL.get(r["size"], r["size"]), r["arm"], config_text(r),
            f"{r['rows']} x {r['tps']}", result_text(r),
            f"{r['held']}/{r['total']}" if r["total"] else "--",
            r["shed"] if r["shed"] is not None else "--",
            r["failed"] if r["failed"] is not None else "--",
            r["window_s"] or "--",
        ])
    return table(
        ["model", "arm", "config", "rows x TPS", "result", "slices held", "shed",
         "failed", "window s"],
        ["---", "---", "---", "---:", "---", "---:", "---:", "---:", "---:"],
        out)


def every_cell_table(rows):
    """One row per cell, with its id, so a figure can be traced to its artifacts."""
    out = []
    for r in sorted(rows, key=lambda g: g["cid"] or ""):
        out.append([
            f"`{r['cid']}`", r["plan"] or "tensorrt_plan",
            r["engine_shape"], r["par"], r["spr"] or "--", r["lanes"] or "--",
            fmt(r["p50"]), fmt(r["p95"]), fmt(r["p99"]),
            MARK.get(r["state"], "?"), r["host"] or "--",
        ])
    return table(
        ["cell", "plan", "engine shape", "parallelism", "per request", "lanes",
         "p50", "p95", "p99", "result", "host"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
         "---", "---"],
        out)


def prime_table(primes):
    """The TensorRT engine build per shape, and the reload that proved the cache."""
    out = []
    for doc in primes:
        for s in doc.get("shapes", []):
            out.append([
                LABEL.get(doc.get("size"), doc.get("size")),
                s.get("rows"), s.get("engine_shape"),
                fmt(s.get("build_seconds"), 1), fmt(s.get("cached_seconds"), 1),
                f"{s.get('speedup')}x" if s.get("speedup") else "--",
            ])
    return table(
        ["model", "rows", "engine shape", "cold build s", "cached s", "ratio"],
        ["---", "---:", "---:", "---:", "---:", "---:"],
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

    doc = f"""# hushar with no mini batch against Triton as it ships, on one A10G

`{RUN}`

## Configuration

| setting | value |
|---|---|
| server | `g5.2xlarge`, 1x NVIDIA A10G, 8 vCPU |
| models | 10M (9,138,693 params, 2 layers) and 20M (18,589,701 params, 5 layers) |
| precision | FP32, TensorRT default builder settings |
| wire | one 690-float f32 vector per row, both arms; identity transformation on hushar, so no server-side feature vectorization |
| feature logging | `inference_log.sample_rate` 0.0 |
| candidate model | none; every request scores the one resident model |
| mini batch | none; `mini_batch_size` is unset on every hushar cell, so the plan is `whole request` and a request of any size is one engine call |
| hushar engine | ONNX Runtime {one.get('ort') or '--'}, TensorRT execution provider |
| Triton | `{one.get('triton_image') or '--'}`, `tensorrt` backend, `max_batch_size: 0` |
| TensorRT | {trt or '--'}, the same library for both arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and by a version check at preflight |
| parallelism | `threading.sessions_per_model` 2, 4, 8 on hushar; `instance_group` count 2, 4, 8 on Triton |
| window | {one.get('window_s') or '--'}s measured in {one.get('total') or '--'} slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | {len(rows)} measured, {sustained} sustained the offered rate |

Every hushar cell leases one session per request (`threading.sessions_per_request` 1), so
`sessions_per_model` is also the requests in flight, which is what Triton's instance count
means. A row of the Grid is therefore a pair of cells differing in the server alone.

## Answer

{answer_table(rows)}
Each arm at its own lowest sustained median. `hushar graph` is `InferenceTime` at that
cell: wall time per request inside the backend.

## Grid

{grid_table(rows)}
`parallelism` is hushar sessions and Triton instances. A ratio is withheld where either
cell of the pair did not hold its rate.

## hushar by session count

{parallelism_table(rows, "hushar", "sessions")}

## Triton by instance count

{parallelism_table(rows, "triton", "instances")}

## Where the median goes

{stage_table(rows)}
`InferenceTime`, `VectorizationTime` and `ToTensorTime` are hushar's own CloudWatch
measures, medians of whole minute buckets inside the measured window. `remainder` is the
client median less those three.

## Device and host

{device_table(rows)}

## Rate held

{hold_table(rows)}

## TensorRT engines

{prime_table(primes)}
One engine per row count, built once before the grid ran and then proved reloadable by a
fresh server, so no cell paid for a build inside its warm-up. A session count loads more
copies of the same engine, so it adds no shape.

## Engine and library versions

{library_table(rows)}
Read off each server at preflight and stamped on every cell of both arms. Triton's
`tensorrt` backend uses no ONNX Runtime, so the ONNX Runtime column applies to hushar
only; TensorRT is the library both arms execute on.

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
