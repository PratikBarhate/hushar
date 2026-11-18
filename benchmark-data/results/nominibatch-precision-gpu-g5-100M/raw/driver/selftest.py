#!/usr/bin/env python3
"""Exercises report.py and verify.py on synthetic cells, before the real ones land.

    ./selftest.py

report.py and verify.py were rewritten for this run's axis, and the moment they are
needed is after the grid has finished -- which is the worst moment to discover a table
cannot be parsed back into the numbers that produced it. So this builds a results tree
whose cells are SHAPED exactly like the ones the sweep writes, runs both scripts over it,
and requires `verify.py` to report zero failures.

The numbers are INVENTED and are never reported as results. What is under test is the
plumbing: a cell goes in, a table comes out, and every figure in that table can be
re-derived from the cell. The fixture is built from this driver's own geometry rather
than from another campaign's records, so it stays correct when the grid changes.

It also checks the two things this run's verifier exists to catch, by constructing them
on purpose: a bf16 cell whose banner does not echo the BF16 flag, and a bf16 engine that
never narrowed its weights. Both must fail.

Writes under `benchmark-data/results/<run>-selftest/` and deletes it on success, so it
cannot be mistaken for a result.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BENCH = _benchmark_data(HERE)
RUN = "nominibatch-precision-gpu-g5-100M-selftest"
DST = BENCH / "results" / RUN
PREFIX = "g5q"
SIZE = "100m"
MD5 = "eee2644bd65cd23543ca806f102f5062"
REV = "selftest"
WIDTH = 690
GRID = [(100, 100), (500, 20)]
# BF16 only, at double the offered rate. The real grid has this asymmetry -- FP32 was
# measured at one rate and BF16 at two -- so the fixture carries it too, or the tables
# that must cope with a blank FP32 column are never exercised.
GRID_2X = [(100, 200), (500, 40)]
PRECS = ("fp32", "bf16")
PAR = 2
# Invented, but ordered the way the real thing should be: bf16 faster than fp32, the
# 500-row request several times the 100-row one, Triton a little ahead of hushar, and
# doubling the offered rate costing a little latency but not much.
BASE_P50 = {(100, "fp32"): 7.40, (100, "bf16"): 4.10,
            (500, "fp32"): 29.80, (500, "bf16"): 15.90}
# fp32 weights are about 400 MB; bf16 about half. The verifier requires the second to be
# materially smaller, so the fixture has to honour that or the selftest fails on itself.
ENGINE_BYTES = {"fp32": 403_585_164, "bf16": 202_952_572}
FACTS = dict(host_tensorrt="10.16.1", host_cuda_major="13",
             host_nvidia_driver="580.95.05", host_gpu="NVIDIA A10G",
             host_ort_so="1.28.3")


def banner(cid, prec, rows, cache):
    """A hushar banner carrying every line the gates and the verifier read back."""
    opts = ("trt_engine_cache_enable=1, "
            f"trt_engine_cache_path={cache}"
            + (", trt_bf16_enable=1" if prec == "bf16" else ""))
    return [
        f"starting transformer on tensorrt:0:cache={cache}, 1 intra-op thread(s)",
        f"  hushar: model {cid} loaded on onnxruntime/TensorRT(device=0, {opts}) "
        f"(runtime 1.28.3)",
        f"    inputs   : features FP32[{WIDTH}]",
        "    outputs  : scores FP32[5]",
        "    features : 1 binding(s)",
        f"               features FP32[{WIDTH}] <- [features] transformed and "
        f"concatenated",
        "    plan     : whole request -- a request of any size is one engine call of "
        "that many rows",
        "  hushar: metrics -> HusharBench [every 100 batches, 4 in flight]",
        "  hushar: inference logs -> /root/x/inference-logs [local filesystem, "
        "16 prefix shard(s), every 5000 batches, 4 in flight, logging no requests]",
        f"  hushar: threads -> 4 async, {PAR} concurrent request(s) x 1 intra-op "
        f"x {PAR} ONNX Runtime session(s) = {PAR} compute on 8 core(s)",
        "    admit  : 512 in flight per connection",
        f"    pool   : {PAR} session(s) per model, one blocking thread each, every "
        f"session with its own intra-op pool; a request leases an idle session while "
        f"it scores",
        "  hushar: listening on 0.0.0.0:8279",
    ]


def slices(tps, p50, p95, p99):
    return [dict(slice=i + 1, of=15, n=tps * 60, achieved=float(tps),
                 p50=p50, p95=p95, p99=p99, shed=0, failed=0) for i in range(15)]


def cell(arm, prec, rows, tps, host, rate_mult=1.0):
    hushar = arm != "triton"
    cid = f"{PREFIX}-{arm}-{SIZE}-{prec}-r{rows}t{tps}-p{PAR}"
    cache = f"/root/trt-cache/{SIZE}/{prec}/r{rows}"
    p50 = BASE_P50[(rows, prec)] * (1.0 if hushar else 0.70) * rate_mult
    p95, p99 = round(p50 * 1.08, 2), round(p50 * 1.15, 2)
    p50 = round(p50, 2)
    graph = round(p50 * 0.62, 3)
    doc = dict(
        id=cid, phase="main", arm=arm,
        stack="hushar" if hushar else "triton",
        wire="tensor", size=SIZE, model_size=SIZE, precision=prec,
        precision_confirmed=prec,
        parameter_count=100561925, layers=31, macs_per_row=785000960,
        rows=rows, tps=tps, rows_per_sec=rows * tps,
        parallelism=PAR,
        parallelism_kind="sessions" if hushar else "instances",
        intra=1 if hushar else None, wt=4 if hushar else None,
        mini_batch=rows if hushar else None, mini_batch_configured=None,
        sessions_per_request=1 if hushar else None,
        request_lanes=PAR if hushar else None,
        sessions_per_model=PAR if hushar else None,
        plan="whole request" if hushar else None,
        plan_confirmed="whole request" if hushar else None,
        pieces=1, split=False,
        max_batch=None if hushar else 0,
        warmup=300, duration=900, segments=15, settle=60,
        engine_cache=cache if hushar else None,
        engine_shape=[rows, WIDTH],
        engine_bytes=ENGINE_BYTES[prec],
        host=host, server_ip="172.31.0.1", instance="g5.2xlarge", arch="x86_64",
        cores=8, conn=512, sample_rate="0.0",
        wire_note="one 690-wide f32 vector per row, identity transformation",
        model_file="bench_model.onnx", model_md5=MD5,
        onnxruntime="1.28.3",
        triton_image="nvcr.io/nvidia/tritonserver:26.05-py3",
        rev=REV, run=RUN,
        result=dict(offered=tps, achieved=float(tps), p50=p50, p95=p95, p99=p99,
                    shed=0, failed=0),
        verdict=dict(state="sustained", sustained=True, slices_held=15,
                     slices_total=15, broke_in_slice=None, whole_window_held=True),
        sustained=True,
        slices=slices(tps, p50, p95, p99),
        measured_window=dict(start="2026-11-17T01:00:00Z", end="2026-11-17T01:15:02Z",
                             seconds=902),
        gpu_utilization=dict(samples=871, gpu_util_pct=round(58.4 * rate_mult, 2),
                             memory_util_pct=40.1,
                             memory_used_mib=1430.0,
                             power_w=round(211.5 * rate_mult, 1), sm_mhz=1710.0,
                             temperature_c=50.2,
                             gpu_util_max_pct=round(62.0 * rate_mult, 1),
                             power_max_w=round(224.0 * rate_mult, 1)),
        server_utilization=dict(samples=881, cpu_pct=14.2, process_memory_pct=4.1,
                                process_rss_gib=1.9, host_memory_pct=9.0,
                                tick_cpu_pct=13.0, cpu_max_pct=31.5),
        client_utilization=dict(samples=894, cpu_pct=0.31, process_memory_pct=0.0,
                                process_rss_gib=0.0, host_memory_pct=1.4,
                                tick_cpu_pct=0.33, cpu_max_pct=1.8),
        **FACTS,
    )
    if hushar:
        doc.update(
            provider=f"tensorrt:0:cache={cache}"
                     + (":trt_bf16_enable=1" if prec == "bf16" else ""),
            ort_version="1.28.3",
            banner=banner(cid, prec, rows, cache),
            cloudwatch={
                "InferenceTime": dict(buckets=15, samples=180000, p50_ms=graph,
                                      p95_ms=round(graph * 1.05, 3),
                                      p99_ms=round(graph * 1.08, 3),
                                      mean_ms=graph, p50_spread_ms=0.015),
                "VectorizationTime": dict(buckets=15, samples=180000, p50_ms=0.021,
                                          p95_ms=0.03, p99_ms=0.04, mean_ms=0.022,
                                          p50_spread_ms=0.002),
                "ToTensorTime": dict(buckets=15, samples=180000, p50_ms=0.0,
                                     p95_ms=0.0, p99_ms=0.0, mean_ms=0.0,
                                     p50_spread_ms=0.0),
            },
        )
    else:
        doc.update(
            platform="tensorrt_plan", backend="tensorrt",
            plan=f"/root/triton/plans/{SIZE}-{MD5[:12]}-r{rows}-{prec}.plan",
            dynamic_batching=None,
            banner=[f"reusing /root/triton/plans/{SIZE}-{MD5[:12]}-r{rows}-{prec}.plan",
                    "backend check: tensorrt", f"model md5 {MD5}",
                    "triton up on grpc"],
            cloudwatch={},
        )
    return doc


def fixture():
    cells, primes = [], []
    for prec, host in zip(PRECS, ("g5-1", "g5-2")):
        for rows, tps in GRID:
            for arm in ("hushar-tensor", "triton"):
                cells.append(cell(arm, prec, rows, tps, host))
        # The doubled rate, BF16 only, on the SAME host as its baseline -- which is how
        # the real follow-up ran it, so a rate comparison is not also a host comparison.
        if prec == "bf16":
            for rows, tps in GRID_2X:
                for arm in ("hushar-tensor", "triton"):
                    cells.append(cell(arm, prec, rows, tps, host, rate_mult=1.05))
        primes.append(dict(
            size=SIZE, precision=prec, run=RUN, rev=REV, host=host,
            cache_root=f"/root/trt-cache/{SIZE}/{prec}",
            prime_arm="hushar-tensor", prime_wire="tensor",
            shapes=[dict(rows=rows, engine_cache=f"/root/trt-cache/{SIZE}/{prec}/r{rows}",
                         engine_shape=[rows, WIDTH], precision=prec,
                         engine_bytes=ENGINE_BYTES[prec],
                         build_seconds=41.0 + rows / 100, cached_seconds=22.1,
                         speedup=round((41.0 + rows / 100) / 22.1, 1))
                    for rows, _ in GRID]))
    status = dict(
        by_arm={a: dict(cells=6, sustained=6, failed_load=0, partial=0,
                        with_cloudwatch=6 if a != "triton" else 0,
                        with_gpu_samples=6)
                for a in ("hushar-tensor", "triton")},
        by_size={})
    return cells, primes, status


def run(script, env):
    r = subprocess.run([sys.executable, str(HERE / script)],
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def main():
    cells, primes, status = fixture()
    raw = DST / "raw"
    shutil.rmtree(DST, ignore_errors=True)
    (raw / "prime").mkdir(parents=True, exist_ok=True)
    (raw / "results.json").write_text(json.dumps(cells, indent=2))
    (raw / "run-status.json").write_text(json.dumps(status, indent=2))
    for doc in primes:
        (raw / "prime" / f"{SIZE}-{doc['precision']}.json").write_text(
            json.dumps(doc, indent=2))

    env = dict(os.environ, RUN=RUN)
    rc, out = run("report.py", env)
    if rc != 0:
        print(out)
        sys.exit("report.py failed on the fixture")
    print(out.strip())

    rc, out = run("verify.py", env)
    print(out.strip())
    if rc != 0:
        sys.exit("verify.py found failures on a fixture it should accept; the report "
                 "and the verifier disagree about this run's own tables")

    # THE NEGATIVE CASES. A verifier that accepts everything is not a verifier, so the
    # two failures this run exists to rule out are constructed and must be caught.
    print("\n-- negative cases --")
    for label, mutate in (
        ("a bf16 cell whose banner omits trt_bf16_enable",
         lambda cs: [dict(c, banner=[l for l in (c.get("banner") or [])
                                     if "trt_bf16_enable=1" not in l])
                     if c["precision"] == "bf16" else c for c in cs]),
        ("a bf16 engine that did not narrow its weights",
         lambda cs: [dict(c, engine_bytes=ENGINE_BYTES["fp32"]) for c in cs]),
        ("a fabricated p50",
         lambda cs: [dict(c, result=dict(c["result"], p50=c["result"]["p50"] * 3 + 7))
                     if i == 0 else c for i, c in enumerate(cs)]),
    ):
        (raw / "results.json").write_text(json.dumps(mutate(cells), indent=2))
        rc, out = run("verify.py", env)
        if rc == 0:
            print(out)
            sys.exit(f"verify.py ACCEPTED {label}; that check is not biting")
        print(f"  caught: {label}")

    shutil.rmtree(DST, ignore_errors=True)
    print("\nselftest OK -- report.py and verify.py agree, and all three negative "
          "cases are caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
