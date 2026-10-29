#!/usr/bin/env python3
"""Syncs this run's evidence out of S3 and enriches every cell into `results.json`.

    ./collect.py

Four sources feed a cell, and each is read here rather than at report time so that
`results.json` is the ONE input the report reads:

    the cell record   what the orchestrator wrote the moment the cell finished --
                      settings, the client's whole-window numbers and its 15 slices,
                      the server banner, the verdict
    CloudWatch        hushar's own `InferenceTime` and `VectorizationTime`, as the median
                      over the measured window's minute buckets. This is the DEVICE and
                      FEATURE split that the client median cannot see into.
    host samples      per-second CPU and memory from sample_host.sh, cut to the window
    NeuronCore        per-second core utilization from sample_neuron.sh, cut to the window

THE WINDOW IS CUT THE SAME WAY FOR ALL THREE. A cell records `window.start` and
`window.end` as the measured span, warm-up already discarded, and every series here is
filtered to it. A sample taken while a server was starting is not a measurement of the
load, and a CloudWatch bucket that straddles the warm-up boundary carries work that was
deliberately thrown away.

ONE SUBTLETY THE REPORT HAS TO INHERIT. On a SPLIT cell hushar reports one duration per
stage per REQUEST, and under a split that duration is the MAXIMUM over the request's mini
batches, not their sum. So `graph_ms` on a split row is neither total device work nor a
clean per-piece figure, and no `everything else` subtraction is offered for it. The
convention is `minibatch-rows-cpu-arm64-10M`'s and the GPU run's; it is restated in the
cell as `graph_is_max_over_pieces` so a reader of the JSON cannot miss it.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hosts  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "results" / hosts.RUN / "raw"
BUCKET = hosts.BUCKET or sys.exit("set HUSHAR_BENCH_BUCKET")
PREFIX = hosts.PREFIX
PROFILE = hosts.PROFILE
REGION = hosts.REGION
# PINNED, because this script is run under `conda run -n local` and that environment
# carries its own awscli which fails on import with `KeyError: 'opsworkscm'`. Resolving
# `aws` through PATH therefore picks the broken one, and the S3 sync fails for a reason
# that has nothing to do with S3.
AWS = os.environ.get("AWS_CLI", "/opt/homebrew/bin/aws")


def sh(cmd, timeout=900):
    done = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def aws(args, timeout=900):
    rc, out, err = sh(f"{AWS} --profile {PROFILE} --region {REGION} {args}", timeout)
    if rc != 0:
        print(f"   ! aws {args[:60]}...: {err.strip()[-200:]}")
    return rc, out


def sync():
    """Brings the run prefix down into the layout `documentation/benchmarking.md` names.

    SUBTREE BY SUBTREE, not one recursive sync, because the two sides are shaped
    differently. The orchestrator writes cells to `runs/<run>/cells/` and per-cell
    evidence to `runs/<run>/raw/...`, while a recorded run is
    `results/<run>/raw/{cells,server,client,driver}`. A single `s3 sync` of the whole
    prefix therefore lands the evidence at `raw/raw/server/...` -- which it did, and the
    only symptom was every NeuronCore column reading `-` because the files were one
    directory deeper than the enricher looked.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    moves = [
        ("cells/", "cells/"),
        ("prime/", "prime/"),
        ("failures/", "failures/"),
        ("raw/", ""),            # raw/server/... -> server/...
    ]
    for remote, local in moves:
        target = (RAW / local) if local else RAW
        target.mkdir(parents=True, exist_ok=True)
        print(f"syncing {remote or './'} -> {target.relative_to(RAW.parent)}")
        aws(f"s3 sync s3://{BUCKET}/{PREFIX}/{remote} {target}/ --only-show-errors",
            timeout=3600)


def parse_iso(text):
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def cw_median(model_id, metric, start, end):
    """The median of a metric's per-minute p50s over the measured window.

    A MEDIAN OF MEDIANS, deliberately, and the report says so. CloudWatch holds the
    distribution per bucket; there is no way to ask it for one percentile over a span of
    buckets. Taking the median of the per-minute p50s is robust to a single odd minute in
    a way a mean is not, which matters because the first and last buckets of a window can
    be partial.
    """
    if not (start and end):
        return None, 0
    # A whole minute either side would pull in warm-up, so the window is used as recorded
    # and CloudWatch is asked for the buckets that fall inside it.
    rc, out = aws(
        f"cloudwatch get-metric-statistics --namespace {hosts.CW_NAMESPACE} "
        f"--metric-name {metric} "
        f"--dimensions Name=ModelId,Value={model_id} "
        f"--start-time {start.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"--end-time {end.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"--period 60 --extended-statistics p50 p95 p99 --output json")
    if rc != 0:
        return None, 0
    try:
        points = json.loads(out).get("Datapoints", [])
    except ValueError:
        return None, 0
    series = {}
    for stat in ("p50", "p95", "p99"):
        values = sorted(p["ExtendedStatistics"][stat] for p in points
                        if stat in p.get("ExtendedStatistics", {}))
        if values:
            mid = len(values) // 2
            micros = (values[mid] if len(values) % 2
                      else (values[mid - 1] + values[mid]) / 2)
            series[stat] = round(micros / 1000.0, 3)      # us in CloudWatch, ms here
    return (series or None), len(points)


def window_rows(path, start, end):
    """Sample lines whose epoch falls inside the measured window."""
    if not (path.exists() and start and end):
        return []
    lo, hi = start.timestamp(), end.timestamp()
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        when = int(parts[0])
        if lo <= when <= hi:
            out.append(parts)
    return out


def host_means(path, start, end):
    """CPU and memory percent over the window, from sample_host.sh's columns.

    sample_host.sh writes the SCHEDULER's own run-time counter as its CPU figure and the
    tick-based /proc/stat figure beside it, because on a tickless kernel a busy CPU whose
    tick has stopped can be accounted idle for a whole window. Both are carried.
    """
    rows = window_rows(path, start, end)
    if not rows:
        return None
    cpu, mem, cpu_stat = [], [], []
    for parts in rows:
        def at(index):
            try:
                return float(parts[index])
            except (IndexError, ValueError):
                return None
        for value, sink in ((at(1), cpu), (at(2), mem), (at(5), cpu_stat)):
            if value is not None:
                sink.append(value)
    def mean(values):
        return round(sum(values) / len(values), 1) if values else None
    return dict(samples=len(rows), cpu_pct=mean(cpu), mem_pct=mean(mem),
                cpu_pct_proc_stat=mean(cpu_stat))


def neuron_means(path, start, end):
    """Per-core and mean NeuronCore utilization over the window.

    `runtime_samples` is the count with a LIVE Neuron runtime. A window whose cores read
    `-` did not measure an idle device, it measured no server, and averaging those in as
    zero would understate every utilization figure in the report.
    """
    rows = window_rows(path, start, end)
    if not rows:
        return None
    cols = {1: [], 2: [], 3: [], 4: [], 5: []}
    live = 0
    for parts in rows:
        if len(parts) >= 7 and parts[6] != "0":
            live += 1
        for index in cols:
            if index < len(parts) and parts[index] != "-":
                try:
                    cols[index].append(float(parts[index]))
                except ValueError:
                    pass
    def mean(values):
        return round(sum(values) / len(values), 1) if values else None
    def peak(values):
        return round(max(values), 1) if values else None
    return dict(samples=len(rows), runtime_samples=live,
                nc0_pct=mean(cols[1]), nc1_pct=mean(cols[2]),
                nc_mean_pct=mean(cols[3]), nc_peak_pct=peak(cols[3]),
                device_mib=mean(cols[4]), host_mib=mean(cols[5]))


def enrich(cell):
    """Adds the server-side split, the host series and the device series to one cell."""
    start = parse_iso((cell.get("window") or {}).get("start"))
    end = parse_iso((cell.get("window") or {}).get("end"))
    host = cell.get("host")
    cid = cell["id"]

    graph, graph_points = cw_median(cell.get("cloudwatch_model_id") or cid,
                                   "InferenceTime", start, end)
    vector, vector_points = cw_median(cell.get("cloudwatch_model_id") or cid,
                                      "VectorizationTime", start, end)
    cell["server_side"] = dict(
        graph_ms=graph, vectorize_ms=vector,
        graph_buckets=graph_points, vectorize_buckets=vector_points,
        note="median over the measured window's minute-bucket percentiles; "
             "CloudWatch holds the distribution per bucket and cannot be asked for one "
             "percentile across a span of them",
    )
    # THE SPLIT CAVEAT, carried per cell rather than left to the report to remember.
    cell["graph_is_max_over_pieces"] = bool(cell.get("split"))

    cell["host_series"] = host_means(
        RAW / "server" / str(host) / f"s-{cid}.log", start, end)
    cell["client_series"] = host_means(
        RAW / "client" / str(host) / cid / "host-samples.log", start, end)
    cell["neuron_series"] = neuron_means(
        RAW / "server" / str(host) / f"n-{cid}.log", start, end)

    # Per-candidate cost, which is what makes the two packagings of one row throughput
    # comparable. Only meaningful on a cell that HELD its rate.
    result = cell.get("result") or {}
    if cell.get("sustained") and result.get("p50"):
        cell["per_candidate_us"] = round(result["p50"] * 1000.0 / cell["rows"], 1)
    else:
        cell["per_candidate_us"] = None
    return cell


def main():
    sync()
    cells_dir = RAW / "cells"
    if not cells_dir.is_dir():
        sys.exit(f"no cells under {cells_dir}; has a lane run?")
    cells = []
    for path in sorted(cells_dir.glob("*.json")):
        try:
            cells.append(json.loads(path.read_text()))
        except ValueError:
            print(f"   ! {path.name} is not readable JSON")
    print(f"{len(cells)} cell(s)")

    for index, cell in enumerate(cells, 1):
        enrich(cell)
        srv = (cell.get("server_side") or {}).get("graph_ms") or {}
        print(f"  [{index:>2}/{len(cells)}] {cell['id']:<44} "
              f"{cell.get('verdict', {}).get('state', '?'):<9} "
              f"graph={srv.get('p50', '-')} "
              f"nc={(cell.get('neuron_series') or {}).get('nc_mean_pct', '-')}")

    primes = {}
    for path in sorted((RAW / "prime").glob("*.json")) if (RAW / "prime").is_dir() else []:
        primes[path.stem] = json.loads(path.read_text())

    failures = []
    if (RAW / "failures").is_dir():
        for path in sorted((RAW / "failures").glob("*.json")):
            try:
                failures.append(json.loads(path.read_text()))
            except ValueError:
                pass

    doc = dict(
        run=hosts.RUN,
        collected=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rig=hosts.state(),
        models=hosts.MODELS,
        loads=hosts.LOADS,
        mini_batches={str(k): v for k, v in hosts.MINI_BATCHES.items()},
        replicas=hosts.REPLICAS,
        neuron_cores=hosts.NEURON_CORES,
        cores_per_replica=hosts.CORES_PER_REPLICA,
        baseline_run=hosts.BASELINE_RUN,
        gpu_counterpart_run=hosts.GPU_COUNTERPART_RUN,
        prime=primes,
        cells=cells,
        failures=failures,
        counts=dict(
            cells=len(cells),
            sustained=sum(1 for c in cells if c.get("sustained")),
            partial=sum(1 for c in cells
                        if (c.get("verdict") or {}).get("state") == "partial"),
            failed=sum(1 for c in cells
                       if (c.get("verdict") or {}).get("state") == "failed"),
            with_neuron_samples=sum(1 for c in cells if c.get("neuron_series")),
            with_server_side=sum(1 for c in cells
                                 if (c.get("server_side") or {}).get("graph_ms")),
            failures=len(failures),
        ),
    )
    out = RAW / "results.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    print(f"  {doc['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
