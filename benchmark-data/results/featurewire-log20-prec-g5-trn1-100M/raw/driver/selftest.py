#!/usr/bin/env python3
"""Runs `report.py` and `verify.py` against SYNTHETIC cells, before the real ones exist.

    ./selftest.py

WHY. The report generator and the verifier are the last things to run and the first things
whose bugs are expensive: by the time the real cells land, the instances are an hour from
being terminated and a crash in `report.py` is a crash with no second chance at the data.
This synthesizes a complete 12-cell round with this round's exact geometry, runs the REAL
generator and the REAL verifier over it, and asserts both come back clean.

AND THE OTHER HALF, WHICH MATTERS MORE: it then breaks the synthetic data in ways that
would make this round a lie, and fails if the verifier does NOT catch each one. A verifier
that cannot fail is not evidence, and the faults worth testing are the ones that leave a
clean-looking report:

  * a BF16 arm whose engine did not shrink -- the flag was accepted and nothing was cast
  * a Trainium FP32 arm whose trace error is in BF16 territory -- `--auto-cast=none` did
    not take, so both "arms" are the same artifact
  * a Trainium BF16 arm whose trace error is in FP32 territory -- the cast did not apply
  * an arm compiled on the compiler's DEFAULT cast, which is `matmult` and not FP32
  * the two arms reporting different parameter counts -- not one model at two precisions
  * an unsustained cell whose latency is published anyway

It writes into `results/<RUN>-selftest/` and deletes it on the way out, so it never
touches the real tree.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for d in [start, *start.parents]:
        if (d / "results").is_dir() and d.name == "benchmark-data":
            return d
    raise SystemExit("cannot find benchmark-data/ above this script")


BD = benchmark_data(HERE)
RUN = "featurewire-log20-prec-g5-trn1-100M-selftest"
ROOT = BD / "results" / RUN
RAW = ROOT / "raw"
FAILURES = []

# ---------------------------------------------------------------- the geometry
#
# EXACTLY THIS ROUND'S GRID, so the synthetic tree exercises the same code paths the real
# one will -- including the fact that THE TWO ACCELERATORS NO LONGER RUN THE SAME RATES.
# Wave A is the four specified rates on both. Wave B is Trainium only: rates set from wave
# A's measured ceilings so the cells sustain. A selftest still carrying the wave-A grid
# alone would be rejected by the real verifier for the wrong loads, which is how this was
# caught.
GRID = {
    "a10g": {
        "fp32": [(100, 100), (500, 20)],
        "bf16": [(100, 200), (500, 40)],
    },
    "trainium1": {
        "fp32": [(100, 100), (500, 20), (100, 70), (500, 10)],
        "bf16": [(100, 200), (500, 40), (100, 70), (500, 10),
                 (100, 100), (500, 20)],
    },
}
CELLS = sum(len(v) for v in GRID["a10g"].values()) * 2 \
    + sum(len(v) for v in GRID["trainium1"].values())
# Plausible figures. The device time roughly halves at BF16 and the A10G is several times
# faster than one trn1.2xlarge, which is what the real round is expected to show -- but
# nothing here is a prediction: the point is shape, not value.
LAT = {
    ("a10g", "fp32", 100): (11.4, 0.92),
    ("a10g", "fp32", 500): (46.3, 40.4),
    ("a10g", "bf16", 100): (6.4, 0.47),
    ("a10g", "bf16", 500): (24.8, 18.5),
    ("trainium1", "fp32", 100): (48.0, 44.0),
    ("trainium1", "fp32", 500): (230.0, 220.0),
    ("trainium1", "bf16", 100): (24.0, 22.0),
    ("trainium1", "bf16", 500): (118.0, 113.0),
}
ENGINE = {
    ("a10g", "fp32", 100): 403564460, ("a10g", "fp32", 500): 403617164,
    ("a10g", "bf16", 100): 202952636, ("a10g", "bf16", 500): 206186332,
}
TRACE = {"fp32": 1.48e-06, "bf16": 2.32e-03}
AUTO_CAST = {"fp32": "none", "bf16": "matmult"}
PARAMS = {"a10g": 100498437, "trainium1": 100561925}
PREFIX = {"a10g": "g5fwp-", "trainium1": "trn1fwp-"}


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAILURES.append(what)


def cell(accel, prec, stack, rows, tps, sustained=True):
    """One synthetic cell, in the shape the collectors produce."""
    neuron = accel == "trainium1"
    hushar = stack == "hushar"
    p50, graph = LAT[(accel, prec, rows)]
    vec = 0.25 if rows == 100 else 1.12
    every, batch = 5, 50000 // rows
    cid = (f"{PREFIX[accel]}{stack}-100m-{prec}-r{rows}t{tps}-"
           f"{'p2' if not neuron else 's2'}")
    achieved = float(tps) if sustained else tps * 0.93
    doc = {
        "id": cid, "accelerator": accel,
        "accelerator_label": "NVIDIA A10G" if not neuron else "AWS Trainium1",
        "stack": stack, "wire": "features" if hushar else "tensor",
        "size": "100m", "precision": prec,
        "rows": rows, "tps": tps, "rows_per_sec": rows * tps,
        "parallelism": 2, "parallelism_kind": "sessions" if hushar else "instances",
        "sessions": 2, "sessions_per_request": 1, "request_lanes": 2,
        "pieces": 1, "split": False,
        "parameter_count": PARAMS[accel], "layers": 31,
        "macs_per_row": 784998400 if not neuron else None,
        "warmup": 300, "duration": 900, "segments": 15, "settle": 60,
        "sustained": sustained,
        "result": {"offered": tps, "achieved": achieved,
                   "p50": p50, "p95": p50 * 1.08, "p99": p50 * 1.13,
                   "shed": 0 if sustained else 1200, "failed": 0},
        "verdict": {"state": "ok" if sustained else "short",
                    "slices_held": 15 if sustained else 0, "slices_total": 15},
        "measured_window": {"seconds": 902},
        "host": "g5-1" if not neuron else "trn1-1",
        "instance": "g5.2xlarge" if not neuron else "trn1.2xlarge",
        "rev": "selftest",
        "engine_bytes": ENGINE.get((accel, prec, rows)),
    }
    if neuron:
        doc.update({
            "mini_batch": rows, "cores_per_replica": 1, "contexts": 1,
            "replicas_per_core": 1, "neuron_cores": 2, "libnrt": "2.33.1.0",
            "auto_cast": AUTO_CAST[prec], "auto_cast_type":
                None if prec == "fp32" else "bf16",
            "maximum_trace_error": TRACE[prec],
            "engine_bytes": int(PARAMS[accel] * (4 if prec == "fp32" else 2) * 0.94),
            "server_side": {"graph_ms": {"p50": graph, "p95": graph * 1.05},
                            "vectorize_ms": {"p50": vec},
                            "graph_buckets": 15},
            "host_series": {"cpu_pct": 24.0, "cpu_max_pct": 38.0, "mem_pct": 1.4},
            "neuron_series": {"nc_mean_pct": 74.0, "nc_peak_pct": 76.0,
                              "device_mib": 820},
            "client_series": {"cpu_pct": 0.4},
            "network": {"total_gbps": 0.21, "pct_of_baseline": 6.7},
        })
    else:
        doc.update({
            "mini_batch_configured": None, "mini_batch": rows,
            "plan": "whole request",
            "host_tensorrt": "10.16.1", "onnxruntime": "1.28.3",
            "triton_image": "nvcr.io/nvidia/tritonserver:26.05-py3",
            "bf16_flag_echoed": prec == "bf16",
            "cloudwatch": {"InferenceTime": {"p50_ms": graph, "p95_ms": graph * 1.05,
                                             "buckets": 15},
                           "VectorizationTime": {"p50_ms": vec},
                           "ToTensorTime": {"p50_ms": 0.0}},
            "server_utilization": {"cpu_pct": 29.0, "cpu_max_pct": 44.0,
                                   "host_memory_pct": 9.1},
            "gpu_utilization": {"gpu_util_pct": 88.0, "gpu_util_max_pct": 94.0,
                                "memory_used_mib": 1430, "power_w": 262.0},
            "client_utilization": {"cpu_pct": 0.5},
        })
    if hushar:
        doc.update({
            "sample_rate": 0.2, "sample_every": every,
            "sample_pct": 100.0 / every, "log_batch": batch,
            "log_rows_per_object": batch * rows, "log_sends": 4,
            "log_store": "Amazon S3", "log_objects": 340,
            "log_bytes_written": 18_000_000_000, "log_shed_batches": 0,
            "log_write_failures": 0,
            "log_object_bytes": batch * (5757 + 1031 * rows + 20 * rows),
            "log_bytes_per_sec": int(tps * 0.2 * (5757 + 1031 * rows + 20 * rows)),
            "banner": [
                "plan     : whole request",
                "TensorRT(device=0, trt_engine_cache_path=/root/trt-cache/100m/"
                f"{prec}/r{rows}"
                + (", trt_bf16_enable=1" if prec == "bf16" else "") + ")",
                "  pool   : 2 session(s) per model, one blocking thread each, every "
                "session with its own intra-op pool; a request leases an idle session "
                "while it scores",
                "features FP32[690] <- 210 features transformed and concatenated "
                "[a, b, c, d, e, f, g, h, ... +202 more]",
                "inference logs -> s3://b/k [Amazon S3, 16 prefix shard(s), "
                f"every {batch} batches, 4 in flight, logging 1 request in {every} "
                f"({100.0 / every:.1f}%)]",
            ],
        })
        if accel == "trainium1":
            doc["banner"] = [
                "hushar: Neuron -> model x, 2 replica(s) on NeuronCore 0-1, "
                "1 core(s) and 1 execution context(s) per replica",
                f"  ctx    : a request of N rows is ceil(N / {rows}) execution(s) of "
                f"{rows} rows on the one replica it leases; 1 can be in flight at once "
                "and any further one waits for a context",
                "  pool   : 2 replica(s) per model, one blocking thread each; a request "
                "leases an idle replica while it scores",
                "plan     : mini batches, shared session",
                f"mini batches of {rows}, scored at the same time on the leased session",
                "loaded on neuron/libnrt 2.33.1.0",
                "features FP32[690] <- 210 features transformed and concatenated "
                "[a, b, c, d, e, f, g, h, ... +202 more]",
                "inference logs -> s3://b/k [Amazon S3, 16 prefix shard(s), "
                f"every {batch} batches, 4 in flight, logging 1 request in {every} "
                f"({100.0 / every:.1f}%)]",
            ]
    else:
        doc.update({"sample_rate": None, "log_batch": None, "max_batch": 0,
                    "plan_platform": "tensorrt_plan",
                    "plan": f"/root/triton/plans/100m-abc-r{rows}-{prec}.plan"})
    return doc


def build(mutate=None):
    """Writes a complete synthetic results tree. `mutate(gpu, neu)` may break it."""
    shutil.rmtree(ROOT, ignore_errors=True)
    (RAW / "driver").mkdir(parents=True, exist_ok=True)
    gpu, neu = [], []
    for prec, loads in GRID["a10g"].items():
        for rows, tps in loads:
            for stack in ("hushar", "triton"):
                gpu.append(cell("a10g", prec, stack, rows, tps))
    for prec, loads in GRID["trainium1"].items():
        for rows, tps in loads:
            neu.append(cell("trainium1", prec, "hushar", rows, tps))
    if mutate:
        mutate(gpu, neu)
    (RAW / "results-gpu.json").write_text(json.dumps(gpu, indent=1))
    (RAW / "results-neuron.json").write_text(json.dumps(neu, indent=1))

    # THE ARCHIVED SERVER-SIDE EVIDENCE, which `verify.py` reads rather than trusting the
    # cell record. Synthesized here because a selftest that skipped it would be testing a
    # verifier running with half its checks disabled -- and those are the checks that read
    # the configuration the server actually RESOLVED, as opposed to the one the driver
    # believes it sent.
    for c in gpu + neu:
        if c.get("stack") != "hushar":
            continue
        d = RAW / "server" / str(c["host"])
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{c['id']}.service.json").write_text(json.dumps({
            "inference_log": {"sample_rate": 0.2, "uri": "s3://bucket/prefix",
                              "batch_size": c["log_batch"], "max_in_flight": 4},
            "threading": {"sessions_per_model": 2, "sessions_per_request": 1,
                           "connection_concurrency": 512, "worker_threads": 4},
        }, indent=1))
        model = {"vectorization_config": {
            "feature_order": [f"f{i:03d}" for i in range(210)]}}
        if c["accelerator"] == "trainium1":
            model.update({"mini_batch_size": c["rows"], "is_fixed": True,
                          "backend": {"type": "neuron", "start_neuron_core": 0,
                                      "neuron_cores_per_replica": 1,
                                      "execution_contexts_per_replica": 1}})
        (d / f"{c['id']}.model.json").write_text(json.dumps(model, indent=1))
        (d / f"{c['id']}.server.log").write_text("\n".join(c.get("banner") or []))
    # Lane logs, in both spellings, so `lane_prime` is exercised on each.
    (RAW / "driver" / "lane-fp32-g5-1.log").write_text(
        "prime 100m r100: build 31.1s -> cached 22.1s\n"
        "prime 100m r500: build 32.4s -> cached 22.2s\n")
    (RAW / "driver" / "lane-bf16-g5-2.log").write_text(
        "prime 100m/bf16 r100: build 55.3s -> cached 22.1s engine 202952636 bytes\n"
        "prime 100m/bf16 r500: build 62.5s -> cached 22.2s engine 206186332 bytes\n")
    for prec in ("fp32", "bf16"):
        lines = []
        for rows in (100, 500):
            mib = PARAMS["trainium1"] * (4 if prec == "fp32" else 2) * 0.94 / 1048576
            lines.append(
                f"prime 100m/{prec} mb{rows}: compiled 2{rows}.9s, NEFF {mib:.1f} MiB, "
                f"{PARAMS['trainium1']:,} params / 31 layers, "
                f"trace error {TRACE[prec]:.3g}, served a trickle in 22.1s\n")
        (RAW / "driver" / f"lane-{prec}-trn1-{1 if prec == 'fp32' else 2}.log"
         ).write_text("".join(lines))
    return gpu, neu


def run(script):
    env = dict(os.environ, REPORT_RUN=RUN,
               HUSHAR_BENCH_BUCKET=os.environ.get("HUSHAR_BENCH_BUCKET", "b"))
    r = subprocess.run([sys.executable, str(HERE / script)], capture_output=True,
                       text=True, env=env, cwd=str(HERE))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    print(f"1 - a complete, correct synthetic round ({CELLS} cells)\n")
    build()
    rc, out = run("report.py")
    check(rc == 0, f"report.py exits 0 on a clean round (rc={rc})")
    if rc != 0:
        print(out[-2500:])
    summary = ROOT / "summary.md"
    check(summary.exists(), "report.py wrote summary.md")
    if summary.exists():
        text = summary.read_text()
        for want in ("## FP32 against BF16", "## Precision evidence",
                     "## A10G against Trainium1", "## Every cell",
                     "## Sustained medians"):
            check(want in text, f"summary.md has the `{want.strip('# ')}` section")
        check(f"{CELLS} measured" in text,
              f"summary.md says {CELLS} cells were measured")
        # The precision evidence must actually carry the numbers, not just a heading.
        check("1.48e-06" in text or "1.48e-6" in text,
              "summary.md prints the FP32 trace error")
        check("202,952,636" in text, "summary.md prints a BF16 engine size")
    rc, out = run("verify.py")
    check(rc == 0, f"verify.py exits 0 on a clean round (rc={rc})")
    if rc != 0:
        print(out[-3000:])
    check("0 failures" in out, "verify.py reports 0 failures on a clean round")

    print("\n2 - faults the verifier must REFUSE to pass\n")

    def bf16_engine_did_not_shrink(gpu, neu):
        for c in gpu:
            if c["precision"] == "bf16":
                c["engine_bytes"] = ENGINE[("a10g", "fp32", c["rows"])]

    def fp32_trace_error_is_bf16(gpu, neu):
        for c in neu:
            if c["precision"] == "fp32":
                c["maximum_trace_error"] = 2.4e-03

    def bf16_trace_error_is_fp32(gpu, neu):
        for c in neu:
            if c["precision"] == "bf16":
                c["maximum_trace_error"] = 1.5e-06

    def compiled_on_the_default_cast(gpu, neu):
        for c in neu:
            if c["precision"] == "fp32":
                c["auto_cast"] = "matmult"

    def arms_disagree_about_the_model(gpu, neu):
        for c in neu:
            if c["precision"] == "bf16":
                c["parameter_count"] = 99_000_000

    def precision_missing_from_the_id(gpu, neu):
        for c in gpu:
            c["id"] = c["id"].replace(f"-{c['precision']}-", "-")

    def bf16_label_without_the_flag(gpu, neu):
        for c in gpu:
            if c["precision"] == "bf16" and c["stack"] == "hushar":
                c["bf16_flag_echoed"] = False

    def an_unsustained_cell_is_ranked(gpu, neu):
        gpu[0]["sustained"] = False
        gpu[0]["verdict"] = {"state": "short", "slices_held": 0, "slices_total": 15}
        gpu[0]["result"]["achieved"] = gpu[0]["tps"] * 0.9

    def the_log_was_switched_off(gpu, neu):
        for c in gpu + neu:
            if c["stack"] == "hushar":
                c["sample_rate"], c["sample_every"] = 0.0, 0

    def the_wire_stayed_pre_vectorized(gpu, neu):
        for c in gpu + neu:
            if c["stack"] == "hushar":
                c["wire"] = "tensor"

    for name, mutation in (
        ("a BF16 engine that did not shrink", bf16_engine_did_not_shrink),
        ("an FP32 arm whose trace error is in BF16 territory", fp32_trace_error_is_bf16),
        ("a BF16 arm whose trace error is in FP32 territory", bf16_trace_error_is_fp32),
        ("an arm whose recorded cast is not its label's", compiled_on_the_default_cast),
        ("two arms that disagree about the model", arms_disagree_about_the_model),
        ("a precision missing from the cell id", precision_missing_from_the_id),
        ("a BF16 label with no BF16 flag in the banner", bf16_label_without_the_flag),
        ("the inference log switched off", the_log_was_switched_off),
        ("the pre-vectorized wire", the_wire_stayed_pre_vectorized),
    ):
        build(mutation)
        run("report.py")
        rc, out = run("verify.py")
        caught = rc != 0 and "0 failures" not in out
        check(caught, f"verify.py REFUSES {name}")
        if not caught:
            print(f"        (verify.py said: {out.strip().splitlines()[-1:]})")

    # This one is a report-side rule, not a cell-side one: the generator must withhold the
    # latency itself, so the check is on what `report.py` printed.
    build(an_unsustained_cell_is_ranked)
    run("report.py")
    text = (ROOT / "summary.md").read_text() if (ROOT / "summary.md").exists() else ""
    rc, out = run("verify.py")
    check("0 failures" in out,
          "an unsustained cell is handled cleanly -- the generator withholds its latency "
          "rather than the verifier rejecting the round")
    check("FAIL" in text or "--" in text,
          "summary.md marks the unsustained cell rather than ranking it")

    shutil.rmtree(ROOT, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
