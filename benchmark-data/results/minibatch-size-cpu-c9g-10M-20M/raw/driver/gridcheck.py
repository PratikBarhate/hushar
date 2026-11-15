#!/usr/bin/env python3
"""Prints the grid every lane will measure, without touching AWS.

    ./gridcheck.py

One row per cell, four lanes, read straight out of `sweep.py` and `hosts.py` so what is
printed is what will run. The totals at the end are what the rig is being asked to buy.
"""
import importlib
import os
import sys

os.environ.setdefault("HUSHAR_BENCH_BUCKET", "hushar-bench-<account-id>-us-west-2")

import hosts  # noqa: E402

PER_CELL_S = 300 + 900 + 60 + 90      # warm-up, measured, settle, start/stop/collect


def lane_cells(size, loads):
    """`sweep.py`'s own grid for one lane, re-imported so module constants re-evaluate."""
    os.environ.update(REV="dryrun", MODEL_MD5="0" * 32, MODEL_SIZE=size, LOADS=loads,
                      RIG="c9g-1=10.0.0.1")
    sys.modules.pop("sweep", None)
    sweep = importlib.import_module("sweep")
    return sweep, list(sweep.main_cells())


def main():
    servers = hosts.GROUPS["servers"]
    total, grand = 0, []
    print(f"run {hosts.RUN}")
    print(f"spin {hosts.ALLOW_SPINNING}  "
          f"concurrency_per_session {hosts.CONCURRENCY_PER_SESSION}  "
          f"workers {hosts.WORKER_THREADS}")
    for name in servers:
        size, loads = hosts.SIZE_OF[name], hosts.LOADS_OF[name]
        sweep, cells = lane_cells(size, loads)
        client = hosts.LANES.get(name, "?")
        print(f"\n== {name} ({hosts.SERVER_TYPE}, {hosts.SERVER_CORES} vCPU) "
              f"<- {client}: {size.upper()} at {loads} "
              f"-- sessions {sweep.SESSION_COUNTS}, overlap "
              f"{sweep.CONCURRENCY_PER_SESSION} per session")
        print(f"   {'cell':<46} {'sess':>4} {'mb':>5} {'pieces':>7} {'flight':>7} "
              f"{'intra':>6} {'cores':>6} {'plan':<31}")
        for c in cells:
            print(f"   {c['id']:<46} {c['sessions']:>4} "
                  f"{str(c['mini_batch'] or '--'):>5} "
                  f"{c['pieces']:>7} {c['mini_batches_in_flight']:>7} "
                  f"{c['intra']:>6} {c['parallel_mini_batches']:>6} {c['plan']:<31}"
                  + ("  TIME-SLICED" if c["time_sliced"] else ""))
        hours = len(cells) * PER_CELL_S / 3600
        print(f"   {len(cells)} cell(s), {hours:.1f} h")
        total += len(cells)
        grand.append(hours)
        for c in cells:
            grand_row = (c["id"], c["log_bytes_per_sec"])
            del grand_row

    # The log volume the bucket is being asked to hold, from the measured feature-wire
    # payload rather than a guess: 5,757 B fixed + 1,031 B per row, plus 5 f32 per row.
    gib = 0.0
    for name in servers:
        _, cells = lane_cells(hosts.SIZE_OF[name], hosts.LOADS_OF[name])
        for c in cells:
            gib += c["log_bytes_per_sec"] * (c["warmup"] + c["duration"]) / 2**30
    print(f"\n{total} cell(s) over {len(servers)} lane(s); "
          f"wall clock {max(grand):.1f} h, instance time {sum(grand):.1f} h x 2 roles")
    print(f"inference log: about {gib:.0f} GiB into "
          f"s3://<bucket>/inference-logs/{hosts.RUN}/ -- delete it after collect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
