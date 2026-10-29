#!/usr/bin/env python3
"""Pulls the run down from S3 and enriches every cell with its measured-window data.

    collect.py                sync from S3, then enrich (reusing saved CloudWatch)
    collect.py --no-sync      enrich what is already on disk
    collect.py --refresh      fetch the CloudWatch distributions again

Reads `raw/cells/*.json` and `raw/probes/*.json`, the per-second host samples under
`raw/server/<host>/{s,t}-<id>.log` and `raw/client/<host>/<id>/host-samples.log`, the GPU
samples under `raw/server/<host>/g-<id>.log`, and -- for the hushar arm -- CloudWatch's
minute buckets for that cell's own ModelId. Writes `raw/results.json`,
`raw/probe-results.json` and `raw/run-status.json`, which are what `report.py` reads.

THE MEASURED WINDOW is taken from the cell itself (`window.start` to `window.end`), which
the sweep recorded as the instant warm-up ended and the instant the client returned. It is
not recomputed here from a start time plus a duration, because the two could disagree and
the cell is the one that was there.

CloudWatch is asked with Period=60 and only WHOLE minute buckets inside the window are
kept. The sweep starts every client on a minute boundary, so a 120 s warm-up followed by
300 s measured gives exactly five. A partial bucket at either edge would mix warm-up or
teardown into a percentile, so it is dropped rather than weighted.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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
RUN = os.environ.get("RUN", "config-sweep-gpu-g5-wire-10M-20M")
# The results tree lives beside the repository's other campaigns; the driver lives under
# benchmark-data/private, which is gitignored.
RESULTS = _benchmark_data(HERE) / "results" / RUN
RAW = RESULTS / "raw"
REGION = "us-west-2"
PROFILE = "hushar-bench"
AWS = os.environ.get("AWS_CLI", "/opt/homebrew/bin/aws")
PERIOD = 60
# InferenceTime is the graph call; VectorizationTime is the 210 transformations, which is
# what a Triton caller pays outside its own latency; ToTensorTime is the copy into the
# input buffer.
METRICS = ("InferenceTime", "VectorizationTime", "ToTensorTime")
REFRESH = "--refresh" in sys.argv
NO_SYNC = "--no-sync" in sys.argv


def sync():
    """Brings the run down from S3 into the shape the rest of this file expects.

    The bucket layout is `runs/<run>/{cells,probes,...}` for records and
    `runs/<run>/raw/{server,client,driver}` for evidence -- the sweep writes the evidence
    under its own `raw/` so a reader of the bucket can tell the two apart. Locally they
    all live under one `raw/`, the way every earlier campaign's results tree does.

    So this is TWO syncs, and that is the point: a single sync of `runs/<run>/` into `raw/`
    produces `raw/raw/server/...`, which is where this run's GPU samples went the first
    time and why every cell came back with no GPU data at all.
    """
    bucket = os.environ.get("HUSHAR_BENCH_BUCKET")
    if not bucket:
        sys.exit("set HUSHAR_BENCH_BUCKET to sync, or pass --no-sync")
    RAW.mkdir(parents=True, exist_ok=True)
    base = [AWS, "s3", "sync", "--profile", PROFILE, "--region", REGION,
            "--only-show-errors"]
    jobs = [(f"s3://{bucket}/runs/{RUN}/raw/", str(RAW))]
    for part in ("cells", "probes", "failures", "selection", "prime"):
        jobs.append((f"s3://{bucket}/runs/{RUN}/{part}/", str(RAW / part)))
    for src, dst in jobs:
        r = subprocess.run(base + [src, dst], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"sync {src} failed: {r.stderr[-800:]}")
    # A `raw/raw` is the signature of the bug above, so it is reported rather than left to
    # be discovered as missing data.
    if (RAW / "raw").exists():
        print(f"WARNING: {RAW / 'raw'} exists -- left over from a previous sync; "
              f"delete it, it shadows nothing but wastes space")
    counts = {d: len(list((RAW / d).glob('*.json')))
              for d in ("cells", "probes", "failures", "selection", "prime")
              if (RAW / d).is_dir()}
    evidence = {d: sum(1 for _ in (RAW / d).rglob('*') if _.is_file())
                for d in ("server", "client", "driver") if (RAW / d).is_dir()}
    print(f"synced records {counts}, evidence files {evidence}")


def parse_time(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def window(cell):
    return parse_time(cell["window"]["start"]), parse_time(cell["window"]["end"])


def columns(path, start, end, width):
    """Rows of `epoch v1 v2 ...` inside the window, as a list of per-column lists.

    A field that is not a number -- `-`, which `sample_gpu.sh` writes where nvidia-smi
    reported `[N/A]` -- is dropped from ITS OWN column only. The first version of this
    discarded the whole sample, so one missing power reading silently cost the utilization
    series too.
    """
    if not path.exists():
        return None, 0
    cols = [[] for _ in range(width)]
    n = 0
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            epoch = int(fields[0])
        except ValueError:
            continue
        instant = datetime.fromtimestamp(epoch, timezone.utc)
        if not (start <= instant < end):
            continue
        n += 1
        for i in range(width):
            if i + 1 >= len(fields):
                continue
            try:
                cols[i].append(float(fields[i + 1]))
            except ValueError:
                continue
    return (cols, n) if n else (None, 0)


def mean(values, digits=2):
    return round(sum(values) / len(values), digits) if values else None


def host_samples(path, start, end):
    """Means of sample_host.sh's columns: cpu%, process mem%, rss kB, host mem%, tick cpu%."""
    cols, n = columns(path, start, end, 5)
    if not cols:
        return None
    return {
        "samples": n,
        "cpu_pct": mean(cols[0]),
        "process_memory_pct": mean(cols[1], 3),
        "process_rss_gib": round(mean(cols[2], 1) / 1_048_576, 3) if cols[2] else None,
        "host_memory_pct": mean(cols[3]),
        "tick_cpu_pct": mean(cols[4]),
        "cpu_max_pct": max(cols[0]) if cols[0] else None,
    }


def gpu_samples(path, start, end):
    """Means of sample_gpu.sh's columns: util%, mem util%, mem MiB, W, MHz, degC.

    `gpu_util_pct` is nvidia-smi's utilization.gpu -- the share of each sample period in
    which at least one kernel was resident. It says the device was busy, not how full it
    was, so it is reported alongside power and clocks rather than on its own.
    """
    cols, n = columns(path, start, end, 6)
    if not cols:
        return None
    return {
        "samples": n,
        "gpu_util_pct": mean(cols[0]),
        "memory_util_pct": mean(cols[1]),
        "memory_used_mib": mean(cols[2], 1),
        "power_w": mean(cols[3], 1),
        "sm_mhz": mean(cols[4], 0),
        "temperature_c": mean(cols[5], 1),
        "gpu_util_max_pct": max(cols[0]) if cols[0] else None,
        "power_max_w": max(cols[3]) if cols[3] else None,
    }


def cloudwatch_points(cw, cell, metric, start, end):
    """Whole minute buckets inside the window, from this cell's own ModelId."""
    response = cw.get_metric_statistics(
        Namespace=cell["cloudwatch_namespace"],
        MetricName=metric,
        Dimensions=[{"Name": "ModelId", "Value": cell["cloudwatch_model_id"]}],
        StartTime=start - timedelta(seconds=PERIOD),
        EndTime=end + timedelta(seconds=PERIOD),
        Period=PERIOD,
        Statistics=["SampleCount", "Average"],
        ExtendedStatistics=["p50", "p95", "p99"],
    )
    kept = []
    for p in sorted(response.get("Datapoints", []), key=lambda p: p["Timestamp"]):
        t = p["Timestamp"].astimezone(timezone.utc)
        if t < start or t + timedelta(seconds=PERIOD) > end:
            continue
        kept.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sample_count": int(p["SampleCount"]),
            "average_us": round(p["Average"], 6),
            "p50_us": round(p["ExtendedStatistics"]["p50"], 6),
            "p95_us": round(p["ExtendedStatistics"]["p95"], 6),
            "p99_us": round(p["ExtendedStatistics"]["p99"], 6),
        })
    return kept


def summarize_points(points):
    """The mean across buckets, and the spread, because the mean hides a drifting cell.

    A mean of per-minute p50s is not the window's p50. It is close when the minutes agree,
    which is why `p50_spread_ms` is carried next to it: a large spread is the signal that
    this figure should not be read as a percentile.
    """
    if not points:
        return None
    samples = sum(p["sample_count"] for p in points)
    if not samples:
        return None
    return {
        "buckets": len(points),
        "samples": int(samples),
        "p50_ms": round(sum(p["p50_us"] for p in points) / len(points) / 1000, 3),
        "p95_ms": round(sum(p["p95_us"] for p in points) / len(points) / 1000, 3),
        "p99_ms": round(sum(p["p99_us"] for p in points) / len(points) / 1000, 3),
        "mean_ms": round(sum(p["average_us"] * p["sample_count"]
                             for p in points) / samples / 1000, 3),
        "p50_spread_ms": round((max(p["p50_us"] for p in points)
                                - min(p["p50_us"] for p in points)) / 1000, 3),
    }


def enrich(cell, cw, cw_dir):
    start, end = window(cell)
    cid, host = cell["id"], cell.get("host", "?")
    tag = "s" if cell["stack"] == "hushar" else "t"
    srv = RAW / "server" / host
    cell["measured_window"] = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "seconds": int((end - start).total_seconds())}
    cell["server_utilization"] = host_samples(srv / f"{tag}-{cid}.log", start, end)
    cell["gpu_utilization"] = gpu_samples(srv / f"g-{cid}.log", start, end)
    cell["client_utilization"] = host_samples(
        RAW / "client" / host / cid / "host-samples.log", start, end)

    if cell.get("cloudwatch_model_id"):
        saved = cw_dir / f"{cid}.json"
        if saved.exists() and not REFRESH:
            raw, source = json.loads(saved.read_text()), "saved"
        else:
            raw = {m: cloudwatch_points(cw, cell, m, start, end) for m in METRICS}
            saved.write_text(json.dumps(raw, indent=2) + "\n")
            source = "fetched"
        cell["cloudwatch"] = {m: summarize_points(pts) for m, pts in raw.items()}
        buckets = len(raw.get("InferenceTime") or [])
    else:
        cell["cloudwatch"], buckets, source = None, 0, "n/a"

    su, gu = cell["server_utilization"], cell["gpu_utilization"]
    print(f"  {cid}: host={su and su['samples']} gpu={gu and gu['samples']} "
          f"cw={buckets} ({source})")
    return cell


def main():
    if not NO_SYNC:
        sync()
    cw = None
    if any(json.loads(p.read_text()).get("cloudwatch_model_id")
           for p in (RAW / "cells").glob("*.json")) or True:
        import boto3
        cw = boto3.Session(profile_name=PROFILE,
                           region_name=REGION).client("cloudwatch")
    cw_dir = RAW / "cloudwatch"
    cw_dir.mkdir(parents=True, exist_ok=True)

    print("cells:")
    rows = [enrich(json.loads(p.read_text()), cw, cw_dir)
            for p in sorted((RAW / "cells").glob("*.json"))]
    probes = []
    if (RAW / "probes").is_dir():
        print("probes:")
        probes = [enrich(json.loads(p.read_text()), cw, cw_dir)
                  for p in sorted((RAW / "probes").glob("*.json"))]
    rows.sort(key=lambda r: (r["size"], r["arm"], r["tps"]))
    probes.sort(key=lambda r: (r["size"], r["arm"], r["parallelism"]))
    (RAW / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RAW / "probe-results.json").write_text(json.dumps(probes, indent=2) + "\n")

    def status(items):
        return {
            "cells": len(items),
            "sustained": sum(bool(r.get("sustained")) for r in items),
            "partial": sum(r.get("verdict", {}).get("state") == "partial"
                           for r in items),
            "failed_load": sum(r.get("verdict", {}).get("state") == "failed"
                               for r in items),
            "with_cloudwatch": sum(bool(r.get("cloudwatch")
                                        and r["cloudwatch"].get("InferenceTime"))
                                   for r in items),
            "with_gpu_samples": sum(bool(r.get("gpu_utilization")) for r in items),
        }

    out = {
        "run": RUN,
        "main": status(rows),
        "probes": status(probes),
        "by_arm": {arm: status([r for r in rows if r["arm"] == arm])
                   for arm in sorted({r["arm"] for r in rows})},
        "by_size": {size: status([r for r in rows if r["size"] == size])
                    for size in sorted({r["size"] for r in rows})},
        "preserved_attempt_failures": len(list((RAW / "failures").glob("*.json")))
        if (RAW / "failures").is_dir() else 0,
    }
    (RAW / "run-status.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
