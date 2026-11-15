#!/usr/bin/env python3
"""Copies the figures this round subtracts against INTO this round's own evidence.

    ./snapshot_control.py            # report what it would write
    ./snapshot_control.py --apply

WHY. `report.py` reads `featurewire-log50-g5-trn1-10M-20M/raw/results-{gpu,neuron}.json`
to build the `Against ...` table. That round is not being retained, so a report generated
after it is gone would silently lose the table -- or worse, keep the committed table while
nothing in the repository could re-derive it. A recorded run has to be regenerable from its
own `raw/`, which is the rule `documentation/benchmarking.md` states as "results.json is
this report's only input".

So this lifts the FOUR FIELDS the table actually uses, per matching cell, into
`raw/control-<run>.json` -- and nothing else. It is a citation, not a copy of a campaign:
the full evidence for those numbers lived in that round, and this file says so.

Run once, before the control round is deleted. `report.py` prefers this snapshot and falls
back to the live round only when the snapshot is absent, so the order is not load-bearing
after the first run.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BD = benchmark_data(HERE)
RUN = "featurewire-log20-g5-trn1-10M-20M"
CONTROL_RUN = "featurewire-log50-g5-trn1-10M-20M"
RAW = BD / "results" / RUN / "raw"
OUT = RAW / f"control-{CONTROL_RUN}.json"
SOURCES = {
    "a10g": BD / "results" / CONTROL_RUN / "raw" / "results-gpu.json",
    "trainium1": BD / "results" / CONTROL_RUN / "raw" / "results-neuron.json",
}
APPLY = "--apply" in sys.argv

# The fallback path, for when the control round has already been removed from `results/`.
# Its CELL RECORDS are still in the bench bucket -- only the inference-log rows were
# deleted at teardown -- and CloudWatch still holds the minute buckets inside its 15-day
# retention for one-minute data. So the snapshot can be rebuilt from the same two sources
# `collect.py` used, rather than from a local file that happens to survive.
AWS = "/opt/homebrew/bin/aws"
PROFILE = "hushar-bench"
REGION = "us-west-2"
PERIOD = 60
NAMESPACE = "HusharBench"


def cells(path):
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, list) else doc.get("cells", [])


def graph_p50(cell):
    """hushar's own InferenceTime median, wherever that lane's collector put it."""
    if cell.get("accelerator") == "trainium1":
        return (((cell.get("server_side") or {}).get("graph_ms") or {}).get("p50"))
    return (((cell.get("cloudwatch") or {}).get("InferenceTime") or {}).get("p50_ms"))


def from_s3():
    """Rebuilds the control rows from the bench bucket and CloudWatch.

    The cell records in `runs/<control>/cells/` carry the client's own numbers and the
    measured window; the device median is not in them, because `collect.py` derived it
    from CloudWatch at collection time. So this re-derives it the same way -- whole minute
    buckets inside the recorded window, Period 60, the p50 extended statistic averaged
    across them -- which is the method, not a re-interpretation of it.
    """
    import os
    bucket = os.environ.get("HUSHAR_BENCH_BUCKET")
    if not bucket:
        print("  set HUSHAR_BENCH_BUCKET to rebuild the snapshot from S3")
        return None
    try:
        import boto3
    except ImportError:
        print("  boto3 is needed to re-derive the device medians")
        return None
    cw = boto3.Session(profile_name=PROFILE, region_name=REGION).client("cloudwatch")

    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            [AWS, "s3", "sync", f"s3://{bucket}/runs/{CONTROL_RUN}/cells/", tmp,
             "--profile", PROFILE, "--region", REGION, "--only-show-errors"],
            capture_output=True, text=True)
        if done.returncode != 0:
            print(f"  could not read the control cells from S3: {done.stderr[-200:]}")
            return None
        rows = []
        for path in sorted(pathlib.Path(tmp).glob("*.json")):
            c = json.loads(path.read_text())
            result = c.get("result") or {}
            rows.append({
                "id": c.get("id"),
                "accelerator": c.get("accelerator"),
                "size": c.get("size"),
                "rows": c.get("rows"),
                "tps": c.get("tps"),
                "parallelism": c.get("parallelism") or c.get("sessions"),
                "stack": c.get("stack", "hushar"),
                "sustained": bool(c.get("sustained")),
                "state": (c.get("verdict") or {}).get("state"),
                "achieved": result.get("achieved"),
                "p50": result.get("p50"),
                "graph_p50_ms": cw_graph_p50(cw, c),
                "sample_rate": c.get("sample_rate"),
            })
    return rows


def cw_graph_p50(cw, cell):
    """`InferenceTime` p50 over the measured window, BY THE LANE'S OWN METHOD.

    The two collectors reduce the per-minute buckets differently, and reproducing the
    wrong one puts a number in the table that the deleted report never showed:

        a10g        MEAN of the per-bucket p50s. The query is padded a minute either side
                    and the buckets are then filtered to those lying wholly inside the
                    window (`collect.py`'s `summarize_points`).
        trainium1   MEDIAN of the per-bucket p50s, from a query bounded exactly by the
                    window -- a median of medians, chosen there for robustness to one odd
                    minute (`collect.py`'s `cw_median`).

    So this branches rather than picking one. Checked against the figures the control
    report published: both lanes reproduce to three decimals.
    """
    model_id = cell.get("cloudwatch_model_id")
    window = cell.get("window") or {}
    if not model_id or not window.get("start"):
        return None

    def parse(text):
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    start, end = parse(window["start"]), parse(window["end"])
    neuron = cell.get("accelerator") == "trainium1"
    pad = timedelta(seconds=0 if neuron else PERIOD)
    response = cw.get_metric_statistics(
        Namespace=NAMESPACE, MetricName="InferenceTime",
        Dimensions=[{"Name": "ModelId", "Value": model_id}],
        StartTime=start - pad, EndTime=end + pad,
        Period=PERIOD, Statistics=["SampleCount"], ExtendedStatistics=["p50"])
    points = response.get("Datapoints", [])

    if neuron:
        values = sorted(p["ExtendedStatistics"]["p50"] for p in points
                        if "p50" in p.get("ExtendedStatistics", {}))
        if not values:
            return None
        mid = len(values) // 2
        micros = (values[mid] if len(values) % 2
                  else (values[mid - 1] + values[mid]) / 2)
        return round(micros / 1000.0, 3)

    kept = [p["ExtendedStatistics"]["p50"] for p in points
            if start <= p["Timestamp"].astimezone(timezone.utc)
            and p["Timestamp"].astimezone(timezone.utc) + timedelta(seconds=PERIOD) <= end]
    if not kept:
        return None
    return round(sum(kept) / len(kept) / 1000, 3)


def main():
    rows, missing = [], []
    for accel, path in SOURCES.items():
        if not path.exists():
            missing.append(str(path.relative_to(BD)))
            continue
        for c in cells(path):
            result = c.get("result") or {}
            rows.append({
                "id": c.get("id"),
                "accelerator": accel,
                "size": c.get("size"),
                "rows": c.get("rows"),
                "tps": c.get("tps"),
                "parallelism": c.get("parallelism") or c.get("sessions"),
                "stack": c.get("stack", "hushar"),
                "sustained": bool(c.get("sustained")),
                "state": (c.get("verdict") or {}).get("state"),
                "achieved": result.get("achieved"),
                "p50": result.get("p50"),
                "graph_p50_ms": graph_p50(c),
                "sample_rate": c.get("sample_rate"),
            })
    source = f"results/{CONTROL_RUN}/raw"
    if missing:
        print(f"the control round is not in results/: {', '.join(missing)}")
        if OUT.exists():
            print(f"  {OUT.relative_to(BD)} already exists, so the report can be "
                  f"regenerated -- nothing to do")
            return 0
        print("  rebuilding from the bench bucket and CloudWatch instead")
        rows = from_s3()
        if not rows:
            return 1
        source = (f"s3://<bucket>/runs/{CONTROL_RUN}/cells/ plus CloudWatch "
                  f"{NAMESPACE} InferenceTime")

    doc = {
        "control_run": CONTROL_RUN,
        "snapshotted": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "note": "The figures this round's `Against ...` table reads, lifted so the report "
                "stays regenerable after the control round was removed from `results/`. "
                "This is a CITATION, not a reproduction: the banners, host and device "
                "samples, client output and full CloudWatch series behind these numbers "
                f"belonged to `{CONTROL_RUN}` and are not retained here. Everything "
                "differs from this round in `sample_rate` alone -- 0.5 there against 0.2 "
                "here. `graph_p50_ms` is `InferenceTime`, the median over whole minute "
                "buckets inside each cell's measured window.",
        "cells": sorted(rows, key=lambda r: (r["accelerator"] or "", r["size"] or "",
                                             r["stack"], r["rows"] or 0, r["tps"] or 0)),
    }
    print(f"{len(doc['cells'])} control cell(s) from {CONTROL_RUN}")
    for r in doc["cells"]:
        print(f"  {r['accelerator']:<10} {r['size']} {r['rows']:>4}x{r['tps']:<5} "
              f"{r['stack']:<7} {str(r['state']):<10} p50 {r['p50']:<9} "
              f"graph {r['graph_p50_ms']}")
    if APPLY:
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {OUT.relative_to(BD)}")
    else:
        print(f"\nwould write {OUT.relative_to(BD)} -- re-run with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
