#!/usr/bin/env python3
"""Pulls a run's cells out of S3 and enriches them into one results.json.

    HUSHAR_BENCH_BUCKET=... collect.py [run-name]

The cells in S3 are what the hosts wrote as they finished, one object each, and they are
the record. This adds the two things a cell cannot know about itself:

  * the CloudWatch percentiles for its measured window, which only the namespace holds --
    the server publishes one data point per scored batch, so the distribution is there and
    the server log carries only means;
  * the utilization means over the same window, re-derived from the per-second sample logs
    rather than taken from the cell's own single figure, so a report can show both the
    scheduler's reading and the tick-based one.

Nothing is computed that the report then recomputes: `results.json` is the only input the
report reads, and verify.py re-derives every displayed figure from these same originals.
"""
import json
import os
import re
import subprocess
import sys

BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET") or sys.exit("set HUSHAR_BENCH_BUCKET")
RUN = sys.argv[1] if len(sys.argv) > 1 else "minibatch-size-cpu-c9g-10M-20M"
PREFIX = f"runs/{RUN}"
PROFILE = os.environ.get("AWS_PROFILE", "hushar-bench")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
NAMESPACE = "HusharBench"


def repo_root(start):
    """The workspace root, found rather than counted.

    This file is run from two places -- the campaign's private driver directory and the
    copy committed beside the evidence under raw/driver/ -- which are at different
    depths. Counting `..` segments silently wrote a whole collection into
    `benchmark-data/benchmark-data/` the first time that happened, so the root is
    identified by what is in it instead.
    """
    here = os.path.abspath(start)
    while True:
        if all(os.path.exists(os.path.join(here, n))
               for n in ("Cargo.toml", "benchmark-data", "scripts")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise SystemExit(f"cannot find the workspace root above {start}")
        here = parent


ROOT = repo_root(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmark-data/results", RUN, "raw")
AWS = os.environ.get("AWS_CLI", "/opt/homebrew/bin/aws")


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def aws(args, timeout=900):
    return sh(f"{AWS} --profile {PROFILE} --region {REGION} {args}", timeout)


def sync():
    for sub in ("cells", "failures", "raw/server", "raw/client"):
        dest = os.path.join(OUT, sub.replace("raw/", ""))
        os.makedirs(dest, exist_ok=True)
        rc, _, err = aws(f"s3 sync s3://{BUCKET}/{PREFIX}/{sub}/ {dest}/ "
                         f"--only-show-errors")
        if rc != 0 and "NoSuchKey" not in err:
            print(f"  warning: {sub}: {err.strip()[:200]}")
        n = sum(len(f) for _, _, f in os.walk(dest))
        print(f"  {sub:<14} {n} file(s)")


def cw_percentiles(model_id, start, end, metric):
    """One data point per scored batch is in CloudWatch, so ask it for the distribution.

    MINUTE buckets clipped to the cell's own measured window, not one bucket covering it.
    CloudWatch aligns a period to multiples of itself from the epoch, so a single large
    period does not mean "this window": asking for 3600 s around a 282 s window returns
    the whole hour, which on these lanes holds four other cells, and sometimes returns
    nothing at all. Minute buckets are the finest alignment this metric supports and each
    one lies wholly inside the window.

    The mean is then exact -- a SampleCount-weighted mean of per-minute means is the mean
    over the window. The percentiles are NOT additive, so what is reported is a
    SampleCount-weighted mean of per-minute percentiles, recorded under a name that says
    so and with the buckets kept beside it so the figure can be re-derived.
    """
    query = (f"cloudwatch get-metric-statistics --namespace {NAMESPACE} "
             f"--metric-name {metric} "
             f"--dimensions Name=ModelId,Value={model_id} "
             f"--start-time {start} --end-time {end} --period 60 "
             f"--extended-statistics p50 p95 p99 --statistics Average SampleCount "
             f"--output json")
    rc, out, _ = aws(query, timeout=180)
    if rc != 0:
        return None
    try:
        points = json.loads(out).get("Datapoints", [])
    except ValueError:
        return None
    points = [p for p in points if p.get("SampleCount")]
    if not points:
        return None

    total = sum(p["SampleCount"] for p in points)

    def weighted(get):
        values = [(p["SampleCount"], get(p)) for p in points]
        values = [(w, v) for w, v in values if v is not None]
        if not values:
            return None
        return sum(w * v for w, v in values) / sum(w for w, _ in values)

    ext = lambda q: (lambda p: p.get("ExtendedStatistics", {}).get(q))  # noqa: E731
    out = {"samples": int(total),
           "buckets": len(points),
           "mean_ms": round(weighted(lambda p: p.get("Average")) / 1000.0, 3)}
    for q in ("p50", "p95", "p99"):
        value = weighted(ext(q))
        if value is not None:
            out[f"{q}_ms_bucket_mean"] = round(value / 1000.0, 3)
    # Kept so verify.py re-derives the figures above from the originals rather than
    # trusting them, and so a reader can see the spread across the window. Stored at six
    # decimals rather than the three the report prints: a bucket is an original here, and
    # re-aggregating rounded values cannot reproduce an aggregate of unrounded ones.
    #
    # A bucket with no percentile stores null, not zero. Zero is a number, and a
    # re-aggregation that believed it would quietly pull the figure down.
    def bucket_p50(point):
        value = point.get("ExtendedStatistics", {}).get("p50")
        return None if value is None else round(value / 1000.0, 6)

    out["per_minute"] = [
        {"t": p["Timestamp"], "n": int(p["SampleCount"]),
         "mean_ms": (None if p.get("Average") is None
                     else round(p["Average"] / 1000.0, 6)),
         "p50_ms": bucket_p50(p)}
        for p in sorted(points, key=lambda d: d["Timestamp"])
    ]
    return out


SAMPLE = re.compile(
    r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")


def sample_means(path):
    """Means over a sample log, with both CPU readings and both memory readings apart.

    sample_host.sh writes six columns:

        epoch  cpu_busy_%  process_RSS_%_of_host  process_RSS_kB  host_memory_%  tick_%

    The first CPU figure is the scheduler's own run-time counter and the last is the
    tick-based /proc/stat one. The two disagree on a tickless kernel, which is how a busy
    host once read as idle for a whole window, so both are carried rather than averaged
    into one number.

    Memory is kept the same way. The host figure is what a capacity dashboard reads and
    is the only one comparable across stacks, because the sampler does not own Triton's
    container PID and records its process RSS as zero.
    """
    if not os.path.isfile(path):
        return None
    sched, tick, host_mem, rss_pct, rss_kb, n = [], [], [], [], [], 0
    for line in open(path):
        m = SAMPLE.match(line.strip())
        if not m:
            continue
        n += 1
        sched.append(float(m.group(2)))
        rss_pct.append(float(m.group(3)))
        rss_kb.append(float(m.group(4)))
        host_mem.append(float(m.group(5)))
        if m.group(6):
            tick.append(float(m.group(6)))
    if not n:
        return None
    out = {"samples": n,
           "cpu_pct_sched": round(sum(sched) / len(sched), 2),
           "memory_pct": round(sum(host_mem) / len(host_mem), 2),
           "process_rss_pct": round(sum(rss_pct) / len(rss_pct), 3),
           "process_rss_gib": round(sum(rss_kb) / len(rss_kb) / (1024 * 1024), 3)}
    if tick:
        out["cpu_pct_tick"] = round(sum(tick) / len(tick), 2)
    return out


def main():
    print(f"== {RUN}")
    print(f"   s3://{BUCKET}/{PREFIX} -> {os.path.relpath(OUT, ROOT)}")
    sync()

    cells_dir = os.path.join(OUT, "cells")
    records = []
    for name in sorted(os.listdir(cells_dir)):
        if not name.endswith(".json"):
            continue
        cell = json.load(open(os.path.join(cells_dir, name)))
        window = cell.get("window", {})
        start, end = window.get("start"), window.get("end")
        model_id = cell.get("cloudwatch_model_id")
        if start and end and model_id:
            cell["cloudwatch"] = {
                metric: cw_percentiles(model_id, start, end, metric)
                for metric in ("InferenceTime", "VectorizationTime", "ToTensorTime")
            }
        tag = "s"
        host = cell.get("host", "")
        cell["server_samples"] = sample_means(
            os.path.join(OUT, "server", host, f"{tag}-{cell['id']}.log"))
        cell["client_samples"] = sample_means(
            os.path.join(OUT, "client", host, cell["id"], "host-samples.log"))
        records.append(cell)

    path = os.path.join(OUT, "results.json")
    json.dump(records, open(path, "w"), indent=2, sort_keys=True)
    ok = sum(1 for c in records if c.get("sustained"))
    print(f"\n   {len(records)} cell(s), {ok} sustained -> "
          f"{os.path.relpath(path, ROOT)}")
    status = {
        "run": RUN,
        "cells": len(records),
        "sustained": ok,
        "partial": sum(1 for c in records
                       if (c.get("verdict") or {}).get("state") == "partial"),
        "failed_load": sum(1 for c in records
                           if (c.get("verdict") or {}).get("state") == "failed"),
        "with_cloudwatch": sum(1 for c in records
                               if (c.get("cloudwatch") or {}).get("InferenceTime")),
        "with_server_samples": sum(1 for c in records if c.get("server_samples")),
        "by_session_count": {str(n): sum(1 for c in records if c["sessions"] == n)
                             for n in sorted({c["sessions"] for c in records})},
        "by_size": {z: sum(1 for c in records if c["size"] == z)
                    for z in sorted({c["size"] for c in records})},
        "phases": {ph: sum(1 for c in records if c.get("phase") == ph)
                   for ph in sorted({c.get("phase") for c in records})},
        # The log path's own outcome, which is the one thing a latency "with logging on"
        # can be quietly wrong about: a cell that shed or lost writes did not pay the
        # full cost.
        "inference_log": {
            "cells_logging": sum(1 for c in records if c.get("sample_rate")),
            "cells_that_shed": sum(1 for c in records if c.get("log_shed_batches")),
            "cells_with_write_failures": sum(1 for c in records
                                             if c.get("log_write_failures")),
            "objects_written": sum(c.get("log_objects") or 0 for c in records),
            # IN GiB, NOT BYTES, and that is a privacy constraint rather than a
            # formatting choice: a byte total crosses twelve digits at ~100 GiB, and
            # scripts/check_privacy.py treats any bare twelve-digit run as an AWS
            # account id -- correctly, since it cannot tell them apart.
            "written_gib": round(sum(c.get("log_bytes_written") or 0
                                     for c in records) / 2**30, 1),
        },
        "preserved_attempt_failures": len([
            n for n in os.listdir(os.path.join(OUT, "failures"))
            if n.endswith(".json")]) if os.path.isdir(os.path.join(OUT, "failures")) else 0,
    }
    spath = os.path.join(OUT, "run-status.json")
    json.dump(status, open(spath, "w"), indent=2, sort_keys=True)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
