#!/usr/bin/env python3
"""Drives one c9g.16xlarge server through its lane of the mini-batch-size run, from its client.

    HUSHAR_BENCH_BUCKET=... REV=... MODEL_MD5=... MODEL_SIZE=10m RIG='c9g-1=10.0.0.1' \
      sweep.py c9g-1

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS. At a fixed offered load, which `mini_batch_size` is fastest -- on ONE
session, with idle threads ASLEEP, with ONE REQUEST IN FIVE written to S3, at two model
weights and two request sizes? `minibatch-rows-cpu-arm64-10M` answered a neighbouring
question with 4 sessions of 16 threads, spinning on, the log off, and only 4/8/16 tried,
so the optimum here is measured rather than inherited.

ONE KNOB MOVES INSIDE A COMPARISON GROUP. A group is (model, rows, tps) and its members
differ only in `mini_batch_size`. The largest member is `mini_batch_size == rows`, which
is one graph call per request and therefore the no-split baseline the rest are read
against. Every group lives entirely on one host, because the 10M CPU campaign measured a
placement offset between two instances of the same type that repeating does not reduce.

WHY ONE SESSION IS ENOUGH ON CPU, AND IS NOT ON AN ACCELERATOR. `mini_batch_size` cuts a
request and scores the pieces at the same time, one scoped thread each. On a device,
concurrent `Run()` calls on one session queue behind each other, which is why the TensorRT
campaigns had to lease a session per piece and why `resolve` refuses a split there with a
group of one. ONNX Runtime's CPU intra-op pool is shared across concurrent `Run()` calls on
the same session, so the pieces genuinely overlap and `sessions_per_request` stays 1. The
parallelism is the split, which is what this run measures.

READ `engine` CAREFULLY ON A SPLIT CELL. The service reports one duration per stage per
request, and under a split that duration is the MAXIMUM over the mini batches, because they
run at once and the slowest sets the request's span. It is therefore not the request's total
engine work, and dividing it by the request's rows would describe an efficiency nobody
gained. The report says so and offers no per-row column for split cells.

EVERY CELL GETS A CLEAN SERVER AND A CLEAN CLIENT. `clean()` runs on both hosts before every
cell and the process table is checked afterwards.
"""
import json
import os
import re
import subprocess
import sys
import time

BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET")
if not BUCKET:
    sys.exit("set HUSHAR_BENCH_BUCKET, e.g. hushar-bench-<account-id>-us-west-2")

def env(name, default=""):
    """`$name`, treating SET-BUT-EMPTY as absent.

    `os.environ.get(name, default)` returns `""` for a variable that exists and is empty,
    which is exactly what `launch.sh` passes for every knob it does not want to override.
    So the default written in this file -- where the run's configuration is documented and
    validated -- would be silently replaced by an empty string, and `int("")` raises at
    import. Every knob below reads through here instead.
    """
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


REV = env("REV") or sys.exit("set REV")
DIR = f"/root/hushar-bench-{REV}"
RUN = env("RUN", "minibatch-size-cpu-c9g-10M-20M")
PREFIX = f"runs/{RUN}"

# WHY THE CELL IDS CARRY A PREFIX OF THEIR OWN. The id is the CloudWatch `ModelId`
# dimension, and `HusharBench` already holds c9g cells from `minibatch-rows-cpu-arm64-10M`
# at overlapping geometry. Under a shared prefix the two campaigns would be one time series
# and a window query would silently mix them -- which matters most here, because that run is
# exactly the control this one is read against.
CELL_PREFIX = env("CELL_PREFIX", "c9gmb")

# The engine, checked against the server's own banner rather than trusted from the deploy.
# 1.28.0 is what `minibatch-rows-cpu-arm64-10M` ran, which is what makes that campaign a
# usable control for this one; hushar's bindings target ORT_API_VERSION 28.
ORT_VER = env("ORT_VER", "1.28.0")

MODEL_SIZE = env("MODEL_SIZE") or sys.exit("set MODEL_SIZE, 10m or 20m")
MODEL_FILE = env("MODEL_FILE", "bench_model.onnx")
MODEL_MD5 = env("MODEL_MD5") or sys.exit("set MODEL_MD5")
GEN = f"benchmark-data/generated/{MODEL_SIZE}"

WARMUP = int(env("WARMUP", "300"))       # 5 min, discarded
DURATION = int(env("DURATION", "900"))   # 15 min, measured, continuous
SEGMENTS = int(env("SEGMENTS", "15"))    # reported as 15 x 1 min
SETTLE = int(env("SETTLE", "60"))        # 1 min between cells, server down
# Start the client on a minute boundary, so the measured window is a whole number of whole
# CloudWatch buckets and nothing from warm-up shares a bucket with the measurement.
ALIGN = env("ALIGN", "1") == "1"

if SEGMENTS <= 0 or DURATION % SEGMENTS:
    sys.exit(f"DURATION={DURATION} is not a whole number of SEGMENTS={SEGMENTS}; a "
             f"part slice is not a slice the verdict can use")

# ---------------------------------------------------------------- the fixed configuration
#
# None of these is an axis. Each is pinned for a reason the report states, and each is read
# back out of the server's own banner by `start_hushar` rather than trusted from here.

# ONE SESSION PER REQUEST, AND EIGHT REQUESTS AT ONCE. `sessions_per_model` is not a
# thread-pool width -- it is how many requests the server scores AT ONCE, because a request
# leases a session group exclusively while it scores ("1 concurrent request(s)" in the
# banner). At 1 the throughput ceiling is `1 / service_time`, which a probe measured at
# 103 req/s against 250 offered, so every cell would have been a measurement of queueing.
# 8 admits the offered load on every lane with room to spare.
# A LIST, because this is a second axis: 8 and 4 are both measured, at every mini-batch
# size, so the report can say whether the best split depends on how many requests are in
# flight. Override e.g. SESSIONS='8' for one count only.
SESSION_COUNTS = [int(n) for n in env("SESSIONS", "8,4").split(",") if n.strip()]
if not SESSION_COUNTS:
    sys.exit("SESSIONS names no session count")
if len(set(SESSION_COUNTS)) != len(SESSION_COUNTS):
    sys.exit("SESSIONS repeats a count; a cell id would collide")
# The widest set, used where one number has to stand for the run.
SESSIONS = max(SESSION_COUNTS)
# Still 1: a request leases ONE session and its mini batches share it, which is the
# `mini batches, shared session` plan. Above 1 would hand each piece a distinct session,
# which is what a device provider needs and this host does not.
SESSIONS_PER_REQUEST = 1
# IDLE THREADS SLEEP. Specified, and also what `minibatch-rows-cpu-arm64-10M` measured as
# the better choice for exactly these configurations: latency equal or slightly better AND
# whole-host CPU about a third lower, because the parallelism comes from the split and a
# spinning pool competes with it.
SPIN = int(env("SPIN", "0"))
CORES = 64

# sessions x width = cores, this series' budget rule, so the whole host is spent and no
# more. With 8 sessions that is an 8-thread pool each.
def intra_for(sessions):
    """sessions x width = cores, so the whole host is spent and no more.

    Per session count rather than once for the run: 8 sessions get an 8-thread pool each
    and 4 get 16, which is the same 64 threads either way. A fixed width would make the
    two session counts also differ in how much of the host they claim, and the comparison
    between them would be measuring two things.
    """
    return max(1, int(env("INTRA", "0")) or CORES // sessions)

# THE SETTING THAT MAKES THE MINI-BATCH SIZE THE ONLY VARIABLE.
#
# `session_profile` derives a session's mini-batch overlap as `cores / intra_op_threads`,
# and `ExecutionPlan::resolve` refuses a split unless that is above 1. Left derived, this
# grid would be measuring the WIDTH as much as the mini-batch size:
#
#   intra 8 -> 8 overlap, so mb 5 on 100 rows (20 pieces) runs 3 waves deep and
#              mb 10 on 500 rows (50 pieces) runs 7 waves deep
#
# A probe at `intra 16` showed what that costs: every cell with more pieces than the
# ceiling COLLAPSED -- 107 of 250 req/s at 4.8 s, on 31% of the host -- because the
# fan-out pool is sized from the same figure and was 3 threads wide.
#
# `threading.concurrency_per_session` sets it directly, so it is held CONSTANT at the core
# count across the whole grid and every cell's pieces overlap fully whatever the width.
# Then the only thing that differs between the cells of a group is the mini-batch size,
# which is what the run is asking about. The surplus over `cores / intra` time-slices the
# session's pool rather than claiming more cores, and the banner says so.
CONCURRENCY_PER_SESSION = int(env("CONCURRENCY_PER_SESSION", str(CORES)))
# What the host will actually put on distinct cores, which is the derived figure and is
# reported beside the configured one rather than being overwritten by it.
def parallel_for(sessions):
    """How many of a request's pieces reach distinct cores, which is cores / width."""
    return max(1, CORES // intra_for(sessions))

# The unsplit reference configures NO mini batch, so `resolve` never consults the overlap
# at all. Kept as its own cell because it is the only way to price the mini-batch code
# path's own overhead: `mb == rows` is one call too, but it travels through the split path.
REF_INTRA = env("REF_INTRA", "")
# Set REFERENCE=0 to measure only the 16 specified cells.
REFERENCE = env("REFERENCE", "1") == "1"

WORKER_THREADS = int(env("WORKER_THREADS", "32"))
CONN = 512            # threading.connection_concurrency, and the client's in-flight cap
METRICS_BATCH = 100   # small, so a batch straddling the warm-up boundary carries under a
                      # second of warm-up into the first measured minute
PORT = 8279

# ---------------------------------------------------------------- the inference log

# ONE REQUEST IN FIVE, COUNTED RATHER THAN RANDOMISED. `Sampler::new(0.2)` keeps exactly one
# in five, so the rate is a property of the configuration and not of a window's luck. The
# banner prints what was chosen and `start_hushar` asserts it: a rate silently rounded to
# something else would make every number here a measurement of the wrong configuration.
SAMPLE_RATE = env("SAMPLE_RATE", "0.2")
# TO S3, NOT TO THE INSTANCE'S DISK. A deployment writes these rows to a bucket, and that is
# the path worth measuring: the sink serialises protobuf on a tokio worker, hands the object
# to the AWS SDK, and sheds rather than blocks when every permit is out. Writing to the root
# volume instead would measure gp3 and be cheaper than the real thing. One prefix per RUN so
# it can be deleted in one call.
LOG_PREFIX = env("LOG_PREFIX", f"inference-logs/{RUN}")

# ONE OBJECT PER 50,000 LOGGED ROWS, WHICH IS WHY THIS IS NOT A CONSTANT. `batch_size`
# counts REQUESTS, and a request is 100 rows in one cell and 500 in another -- so a fixed
# count would hand S3 a 56 MB object at one row count and a 266 MB object at the other, and
# the flush is what disturbs latency. Scaling it by the row count holds the object at ~53 MB
# in every cell, so the log path's disturbance is the same everywhere rather than a hidden
# variable. The payload sizes this rests on are measured, not guessed: the feature wire is
# 5,757 B fixed + 1,031 B per row from real encoded requests.
LOG_ROWS_PER_OBJECT = int(env("LOG_ROWS_PER_OBJECT", "50000"))


def log_batch_for(rows):
    """Requests buffered before one object, so that an object is a fixed number of rows."""
    return max(1, LOG_ROWS_PER_OBJECT // rows)


# Writes allowed in the air at once. If the accumulator ever has to shed, the server says so
# and `run_cell` keeps the line: shedding is the designed behaviour, not a failure, but it
# has to be visible or a cell could report a logging cost it did not actually pay.
LOG_SENDS = int(env("LOG_SENDS", "4"))

MAX_CELLS = int(env("MAX_CELLS", "0"))   # smoke testing only
LANE_TAG = env("LANE_TAG", "main")

# ---------------------------------------------------------------- the grid

WIDTH = 690

# This lane's loads. Both packagings of a size carry the same rows per second -- 100 x 250
# and 500 x 50 are both 25,000 -- so the grid also prices a bigger request against a smaller
# one at constant throughput. Overridable so a follow-up can ask one load without a new
# driver: LOADS='100x250,500x50'.
_LOADS = {
    "10m": [(100, 250), (500, 50)],       # 25,000 rows/s both
    "20m": [(100, 100), (500, 20)],       # 10,000 rows/s both
}
if env("LOADS"):
    LOADS = []
    for part in env("LOADS").split(","):
        r, _, t = part.strip().partition("x")
        LOADS.append((int(r), int(t)))
else:
    LOADS = _LOADS[MODEL_SIZE]

# THE AXIS: mini-batch size IN ROWS, per row count, largest first -- and the largest is the
# row count itself, which is one graph call and the no-split baseline. Ordering matters for a
# lane that dies part-way: what survives is then a group with its baseline rather than a set
# of ratios with nothing to divide by. Override e.g. MINI_BATCHES='100=100,50;500=500,100'.
_MINI_BATCHES = {
    100: [100, 50, 10, 5],
    500: [500, 100, 50, 10],
}
if env("MINI_BATCHES"):
    MINI_BATCHES = {}
    for group in env("MINI_BATCHES").split(";"):
        if not group.strip():
            continue
        rows, _, sizes = group.partition("=")
        MINI_BATCHES[int(rows)] = [int(s) for s in sizes.split(",") if s.strip()]
else:
    MINI_BATCHES = dict(_MINI_BATCHES)

# THE NOISE FLOOR. `minibatch-rows-cpu-arm64-10M` measured an 8.4% same-host spread on a
# mini-batch configuration -- its completion time is the slowest of several pieces running
# at once, so its variance is larger than a single graph call's. A ranking of mini-batch
# sizes is therefore not defensible without repeats: a 5% difference between two sizes would
# be inside that spread. One configuration per row count is repeated, changing nothing but
# the clock. REPEAT_MB names which size, per row count; REPEATS is how many EXTRA cells.
REPEATS = int(env("REPEATS", "2"))
_REPEAT_MB = {100: 10, 500: 50}
if env("REPEAT_MB"):
    _REPEAT_MB = {}
    for group in env("REPEAT_MB").split(";"):
        if not group.strip():
            continue
        rows, _, size = group.partition("=")
        _REPEAT_MB[int(rows)] = int(size)
REPEAT_MB = _REPEAT_MB

for _rows, _tps in LOADS:
    _sizes = MINI_BATCHES.get(_rows)
    if not _sizes:
        sys.exit(f"MINI_BATCHES names no mini-batch size for {_rows}-row requests")
    if len(set(_sizes)) != len(_sizes):
        sys.exit(f"MINI_BATCHES for {_rows} rows repeats a size; a cell id would collide")
    for _s in _sizes:
        if _s <= 0 or _s > _rows:
            sys.exit(f"mini batch {_s} is not a split of a {_rows}-row request")

# THE OVERLAP GATE. `ExecutionPlan::resolve` refuses a `mini_batch_size` unless the
# session's concurrency is above 1, and the refusal is a server that EXITS -- so this
# would abort every split cell one at a time, hours into the rig, with nothing measured.
# Checked once at import against the same figure the service will use.
if any(s < r for r, _ in LOADS for s in MINI_BATCHES[r]) and CONCURRENCY_PER_SESSION <= 1:
    sys.exit(
        f"CONCURRENCY_PER_SESSION={CONCURRENCY_PER_SESSION}, and "
        f"ExecutionPlan::resolve REFUSES a mini_batch_size unless it is above 1 -- "
        f"concurrent calls on one session would queue rather than overlap. Every split "
        f"cell would fail to start."
    )
# A cell asking for more pieces than the host can put on distinct cores is measured, not
# refused -- it is a real configuration, and the surplus time-slices the session's pool.
# The report says so rather than letting the axis look as though it stopped paying.
_time_sliced = [(n, r, s, -(-r // s))
                for n in SESSION_COUNTS for r, _ in LOADS for s in MINI_BATCHES[r]
                if -(-r // s) > parallel_for(n)]

# A command line copied from a previous campaign would silently measure something else, so
# the knobs those campaigns had are refused by name rather than ignored.
for _gone, _why in (
    ("RUNGS", "the axis here is the mini-batch SIZE; set MINI_BATCHES, e.g. "
              "MINI_BATCHES='100=100,50,10,5'"),
    ("ARMS", "this run has one arm: hushar on the feature wire. There is no Triton cell, "
             "because Triton has no mini-batch split to compare"),
    ("ONLY_ROWS", "set LOADS, e.g. LOADS='100x250'"),
    ("SHAPE", "set SESSIONS for the session counts to measure; the intra-op width follows "
              "as cores/sessions unless INTRA says otherwise"),
):
    if os.environ.get(_gone):
        sys.exit(f"{_gone} is not a knob in this run: {_why}.")

PARAMS = {"10m": 9138693, "20m": 18589701}[MODEL_SIZE]
LAYERS = {"10m": 2, "20m": 5}[MODEL_SIZE]
MACS_PER_ROW = {"10m": 53288960, "20m": 128983040}[MODEL_SIZE]

# Filled from the environment by launch.sh, which reads rig-state.json. NOT written here:
# this run launches its own hosts, so an address in this file would be a guess.
SERVERS = {}
for _entry in env("RIG").split(";"):
    if not _entry.strip():
        continue
    _name, _ip = _entry.split("=", 1)
    SERVERS[_name.strip()] = dict(ip=_ip.strip(), instance="c9g.16xlarge", arch="arm64",
                                  model_id=_name.strip())
if not SERVERS:
    sys.exit("set RIG, e.g. RIG='c9g-1=172.31.0.1' -- launch.sh does this")

# THE WIRE THIS RUN MEASURES IS `features`: the client sends 210 named features and the
# service transforms and concatenates them into the model's one 690-float input, so
# `VectorizationTime` is real work here. The tensor configuration is kept resolvable
# because it is what a Triton reference line would need, but no cell selects it.
WIRES = {
    "features": dict(config=f"{GEN}/bench_model_config.json", n_features=210,
                     note="210 named features, transformed and concatenated by hushar "
                          "into the model's one 690-float input; server-side feature "
                          "vectorization is ON"),
    "tensor": dict(config=f"{GEN}/bench_model_tensor_config.json", n_features=1,
                   note="one 690-wide vector, identity transformation"),
}


def cell_id(rows, tps, mb, sessions, repeat=None):
    stem = "whole" if mb is None else f"mb{mb}"
    base = f"{CELL_PREFIX}-{MODEL_SIZE}-r{rows}t{tps}-{stem}-s{sessions}"
    return base if repeat is None else f"{base}-rep{repeat}"


def cell(rows, tps, mb, sessions, repeat=None):
    """One measurement. `mb` is the mini-batch size IN ROWS, never a piece count.

    `mb=None` IS NOT "mini batch equal to the row count". It is the absence of the key,
    which `ExecutionPlan::resolve` answers with `WholeRequest` -- no mini-batch machinery
    at all. That distinction is load-bearing twice over: it is the only configuration at
    which the full host width is legal (`resolve` refuses a split unless
    `cores / width > 1`), and it is the unsplit server a deployer would actually run. The
    mb == rows cell is a DIFFERENT thing: one piece through the split path, at the split
    cells' width, which is what makes it the right single-variable baseline for them.

    `pieces` is derived here rather than on the host, because it is what the report's
    column means and what the banner is checked against. The last piece is however many
    rows remain: `MINI_BATCH_FIXED` is not set, so nothing is padded.

    THE LOG FIELDS ARE DERIVED HERE, not where the server is started, so every one of them
    lands in the cell record and `verify.py` can re-derive the banner assertion from the
    file rather than from this module.
    """
    whole = mb is None
    intra = int(REF_INTRA) if (whole and REF_INTRA) else intra_for(sessions)
    # Configured, not derived -- see CONCURRENCY_PER_SESSION. The reference configures no
    # mini batch, so the setting has nothing to apply to and is left off entirely.
    concurrent = None if whole else CONCURRENCY_PER_SESSION
    parallel = max(1, CORES // intra)
    pieces = 1 if whole else -(-rows // mb)          # ceil
    # Every piece overlaps, because the concurrency is set to the core count.
    in_flight = 1 if whole else min(pieces, CONCURRENCY_PER_SESSION)
    rate = float(SAMPLE_RATE)
    # What `Sampler::new` will actually do with the rate: one request in n, nearest unit
    # fraction. Carried per cell because it is what the banner prints and what the report
    # must quote -- a rate and the fraction it became are not the same statement.
    every = 0 if rate <= 0 else (1 if rate >= 1 else round(1.0 / rate))
    batch = log_batch_for(rows)
    return dict(
        id=cell_id(rows, tps, mb, sessions, repeat),
        phase="repeat" if repeat is not None else ("reference" if whole else "main"),
        stack="hushar", wire="features",
        accelerator="c9g-cpu", accelerator_label="Graviton c9g.16xlarge 64 vCPU",
        size=MODEL_SIZE, parameter_count=PARAMS, layers=LAYERS,
        macs_per_row=MACS_PER_ROW,
        rows=rows, tps=tps, rows_per_sec=rows * tps,
        # ---- the axis
        mini_batch=mb,
        # What is actually written into the model configuration. `None` means the key is
        # NOT exported, which is how `whole request` is spelled -- not by setting it to the
        # row count. On the split cells it equals `mini_batch`, so the configuration and
        # the axis are the same number.
        mini_batch_configured=mb,
        pieces=pieces,
        split=pieces > 1,
        # The no-split member of a comparison group: one piece, through the same plan and
        # at the same width as the splits it is read against.
        baseline=(not whole) and mb == rows,
        reference=whole,
        last_piece_rows=rows if whole else rows - mb * (pieces - 1),
        repeat=repeat,
        # ---- parallelism
        sessions=sessions,
        sessions_per_request=SESSIONS_PER_REQUEST,
        intra=intra,
        # cores / width, which is `ExecutionProfile::concurrent_mini_batches` and the
        # ceiling on how many of a request's pieces are ever in flight. Recorded per cell
        # because a cell whose pieces exceed the cores reaches them by time-slicing the
        # session's pool, and the report must say so rather than let the axis appear to
        # bottom out.
        concurrency_per_session=concurrent,
        # THE SESSION'S CAPABILITY, NOT THE PLAN'S USE OF IT. `session_profile` computes
        # this from the provider and the pool width alone -- it does not know whether a
        # mini batch was configured -- so the reference cell's session reports the same
        # overlap as any other session of that width. Recording 1 here because the plan
        # uses one piece conflated the two and failed all eight reference cells on an
        # assertion the server could never satisfy.
        concurrent_mini_batches=parallel if whole else CONCURRENCY_PER_SESSION,
        parallel_mini_batches=parallel,
        mini_batches_in_flight=in_flight,
        # More pieces in flight than the host has cores for them, so the surplus
        # time-slices the session's pool instead of claiming more cores.
        time_sliced=(not whole) and pieces > parallel,
        spin=SPIN,
        wt=WORKER_THREADS,
        threads=sessions * intra,
        plan=("whole request" if whole else "mini batches, shared session"),
        plan_detail=(
            "a request of any size is one engine call of that many rows" if whole
            else f"rows cut into mini batches of {mb}, scored at the same time on the "
                 f"leased session"),
        # ---- the inference log
        sample_rate=rate,
        sample_every=every,
        sample_pct=(100.0 / every if every else 0.0),
        log_batch=batch,
        log_sends=LOG_SENDS,
        log_rows_per_object=batch * rows,
        # ~53 MB at either packaging, from the measured feature-wire payload plus 5 f32
        # scores per row.
        log_object_bytes=batch * (5757 + 1031 * rows + 20 * rows),
        log_bytes_per_sec=int(tps * rate * (5757 + 1031 * rows + 20 * rows)),
        log_store="Amazon S3",
        # ---- the window
        warmup=WARMUP, duration=DURATION, segments=SEGMENTS, settle=SETTLE,
        precision="fp32",
        engine_shape=[rows if whole else mb, WIDTH],
        split_note=(
            f"no mini batch is configured, so the request is one engine call of {rows} "
            f"rows on a {intra}-thread pool -- the unsplit server a deployer would run, "
            f"and the reference the split cells are priced against"
            if whole else
            f"{rows} rows in {pieces} mini batch(es) of up to {mb}, scored at the same "
            f"time on one session's shared intra-op pool, {in_flight} in flight at once; "
            f"the request's reported stage durations are the MAXIMUM over the pieces, not "
            f"their sum"
            if pieces > 1 else
            f"mini_batch_size equals the row count, so the request is one engine call of "
            f"{rows} rows through the split path at the split cells' {intra}-thread "
            f"width -- the no-split baseline they are read against"),
    )


def main_cells():
    """The grid: every load x every mini-batch size, largest split first.

    Cells are ordered so the NO-SPLIT baseline of a load runs before the splits it is the
    baseline for. If the lane dies part-way, what survives is then a group with a baseline
    rather than a set of ratios with nothing to divide by. The unsplit reference runs with
    its group; the repeats come last, because they measure the noise floor rather than a
    point on the axis.
    """
    for sessions in SESSION_COUNTS:
        for rows, tps in LOADS:
            for mb in MINI_BATCHES[rows]:
                yield cell(rows, tps, mb, sessions)
            if REFERENCE:
                yield cell(rows, tps, None, sessions)
    for sessions in SESSION_COUNTS:
        for rows, tps in LOADS:
            mb = REPEAT_MB.get(rows)
            if not mb or mb not in MINI_BATCHES[rows]:
                continue
            for n in range(2, REPEATS + 2):
                yield cell(rows, tps, mb, sessions, repeat=n)


# ---------------------------------------------------------------- plumbing

def sh(cmd, timeout=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"local command timed out after {timeout}s"


def ssh(ip, script, timeout=900):
    q = script.replace("'", "'\"'\"'")
    # IdentitiesOnly, because a host carrying a stale sweep.pub will otherwise offer that
    # key instead of this one and be refused without ever trying the right file.
    return sh(
        f"ssh -i /root/.ssh/sweep -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "
        f"-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 root@{ip} '{q}'",
        timeout=timeout,
    )


def s3_has(key):
    # HeadObject needs GetObject on the exact key; ListBucket is the permission every role
    # here has, so the key is asked for as a prefix and compared exactly.
    rc, out, _ = sh(
        f"aws s3api list-objects-v2 --bucket {BUCKET} --prefix {key} "
        f"--max-items 1 --query 'Contents[0].Key' --output text", timeout=120)
    return rc == 0 and out.strip() == key


def s3_put(local, key):
    for _ in range(3):
        if sh(f"aws s3 cp {local} s3://{BUCKET}/{key} --only-show-errors",
              timeout=600)[0] == 0:
            return True
        time.sleep(5)
    return False


def s3_put_tree(local, key):
    for _ in range(3):
        if sh(f"aws s3 cp {local} s3://{BUCKET}/{key} --recursive --only-show-errors",
              timeout=900)[0] == 0:
            return True
        time.sleep(5)
    return False


# A BUCKET NAME EMBEDS AN AWS ACCOUNT ID, AND THIS RUN PUTS ONE IN THE SERVER'S BANNER.
# Earlier CPU campaigns logged inference rows to a local path, so the only place the bucket
# appeared was a command line. This one logs to S3, so the banner carries
# `inference logs -> s3://hushar-bench-<account-id>-us-west-2/...` -- and the banner is
# stored in every cell record, and the lane log is committed beside the evidence. So the id
# is masked at the two boundaries where text becomes a committed artifact: `log` and
# `record`. Masked rather than dropped, because the destination IS a measurement.
ACCOUNT_RE = re.compile(r"\d{12}")


def redact(value):
    """Masks the bucket name and any 12-digit account id, recursing through records.

    THE BUCKET LITERAL IS REPLACED FIRST, and that ordering is the whole point. Masking
    only the twelve-digit run is not enough once a string has been CUT: an S3 error line
    truncated to its last 300 characters can land mid-account-id, leaving an eight-digit
    tail that no `\\d{12}` pattern can see and that `check_privacy.py` -- which looks for
    the id exactly -- also steps over. Replacing the whole bucket name removes the digits
    before any cut can split them.
    """
    if isinstance(value, str):
        if BUCKET:
            value = value.replace(BUCKET, "<bucket>")
        return ACCOUNT_RE.sub("<account-id>", value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def redacted_tail(text, n=300):
    """The last `n` characters of `text`, REDACTED FIRST so a cut cannot split an id.

    NOT named `tail`. A local named `tail` in `run_cell` once shadowed a module function of
    that name and cost five cells to an UnboundLocalError; `dryrun.py` now scans for that
    shape, and the distinct name is the other half of the fix.
    """
    return (redact(text or "").strip() or "")[-n:] or None


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {redact(msg)}",
          flush=True)


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# run_client.sh writes one row per rate through summarise.py, tagged with TAG, which is
# pinned to `cell` below so this pattern does not depend on the model name.
SUMMARY_RE = re.compile(
    r"^cell\s+(\d+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)",
    re.M)

# One line per slice of the measured window. A five-minute window cannot tell a rate that
# is sustained from a rate that is sustained for five minutes, which is what these are for.
SLICE_RE = re.compile(
    r"^\s*slice (\d+)/(\d+)\s*:\s*n=(\d+)\s+achieved=([\d.]+|nan)\s+p50=([\d.]+|nan)\s+"
    r"p95=([\d.]+|nan)\s+p99=([\d.]+|nan)\s+shed=(\d+)\s+failed=(\d+)", re.M)


def num(text):
    return None if text == "nan" else float(text)


def parse_slices(text):
    out = []
    for m in SLICE_RE.finditer(text or ""):
        idx, of, n, achieved, p50, p95, p99, shed, failed = m.groups()
        out.append(dict(slice=int(idx), of=int(of), n=int(n), achieved=num(achieved),
                        p50=num(p50), p95=num(p95), p99=num(p99),
                        shed=int(shed), failed=int(failed)))
    return out


def parse_summary(text):
    """The client's own table row, as numbers. None when it wrote no row."""
    m = SUMMARY_RE.search(text or "")
    if not m:
        return None
    offered, achieved, p50, p95, p99, shed, failed = m.groups()
    return dict(offered=int(offered), achieved=float(achieved), p50=float(p50),
                p95=float(p95), p99=float(p99), shed=int(shed), failed=int(failed))


def verdict(res, slices, offered):
    """A rate is a result only if the server kept up with it IN EVERY SLICE.

    Offered and achieved have to agree within 2% with nothing shed and nothing failed, and
    that has to hold for every slice rather than for the window's average -- a cell that
    served 15 minutes at the rate and 1 minute at half of it averages to something that
    looks fine. A cell that held some slices is `partial` and names the first it lost, which
    is a different statement from one that never held the rate; neither may have its latency
    ranked.
    """
    if not res:
        return dict(state="failed", sustained=False, slices_held=0,
                    slices_total=len(slices), first_lost=None,
                    why="the client wrote no parsable summary")
    held, first_lost = 0, None
    for s in slices:
        good = (s["shed"] == 0 and s["failed"] == 0 and s["achieved"] is not None
                and s["achieved"] >= 0.98 * offered)
        if good:
            held += 1
        elif first_lost is None:
            first_lost = s["slice"]
    window_ok = (res["shed"] == 0 and res["failed"] == 0
                 and res["achieved"] >= 0.98 * offered)
    total = len(slices)
    if total and held == total and window_ok:
        return dict(state="sustained", sustained=True, slices_held=held,
                    slices_total=total, first_lost=None, why=None)
    if held:
        return dict(state="partial", sustained=False, slices_held=held,
                    slices_total=total, first_lost=first_lost,
                    why=f"held {held} of {total} slice(s), first lost slice {first_lost}")
    return dict(state="failed", sustained=False, slices_held=held, slices_total=total,
                first_lost=first_lost,
                why=f"held no slice of {total}; achieved {res['achieved']} of {offered}")


def cpu_percent(report):
    m = re.search(r"^cpu\s+([\d.]+)", report or "", re.M)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- clean slate

def clean(ip, cid):
    """Leaves both hosts with nothing of this benchmark running.

    Every pattern is written `[t]arget/...` so the remote shell carrying the pattern does
    not match itself.
    """
    # client side: this host
    sh("pkill -f '[t]arget/release/benchmark-client' >/dev/null 2>&1; true")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    sh(f"rm -rf /tmp/hushar-client/{cid}")

    # server side
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true; "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            f"pkill -f '[t]arget/release/hushar' >/dev/null 2>&1 || true; "
            f"sleep 3; pkill -9 -f '[t]arget/release/hushar' >/dev/null 2>&1 || true")

    # and prove it, rather than assume the kill worked
    for _ in range(20):
        rc, out, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true; "
                             f"(ss -ltn 2>/dev/null || netstat -ltn) "
                             f"| grep -cE ':{PORT} ' || true")
        nums = [n for n in out.split() if n.isdigit()]
        if len(nums) >= 2 and nums[0] == "0" and nums[1] == "0":
            return True
        time.sleep(3)
    log(f"{cid} WARNING: server host still shows a server process or a bound port "
        f"after cleaning")
    return False


# ---------------------------------------------------------------- starting a server

def hushar_env(cell_, host):
    env = [
        f"MODEL_CONFIG={WIRES[cell_['wire']]['config']}",
        "PROVIDER=cpu",
        # One ONNX Runtime session. The blocking pool is sized to it, one thread per
        # session, and with sessions_per_request 1 the pieces of a split request overlap
        # inside that session's shared intra-op pool -- see the module docstring.
        f"SESSIONS={cell_['sessions']}",
        f"SESSIONS_PER_REQUEST={cell_['sessions_per_request']}",
        f"THREADS={cell_['intra']}",
        f"WORKER_THREADS={cell_['wt']}",
        f"ALLOW_SPINNING={cell_['spin']}",
        f"CONN_LIMIT={CONN}",
        "CW_NAMESPACE=HusharBench",
        # THE CELL ID IS THE DIMENSION. Unique across the run and already carrying the
        # campaign prefix, so no two cells and no two campaigns can become one time series
        # in `HusharBench`. The previous CPU campaign keyed this per HOST instead, which
        # worked only because its cells never overlapped in time.
        f"MODEL_ID={cell_['id']}",
        f"METRICS_BATCH={METRICS_BATCH}",
        # THE LOG PATH THIS RUN MEASURES. The rate, the destination and the object size all
        # travel per cell and are asserted back out of the banner -- see `start_hushar`.
        # The URI carries the cell id so one cell's objects can be counted and so two cells
        # never share a prefix.
        f"SAMPLE_RATE={SAMPLE_RATE}",
        f"LOG_URI=s3://{BUCKET}/{LOG_PREFIX}/{cell_['id']}",
        f"LOG_BATCH={cell_['log_batch']}",
        f"LOG_SENDS={LOG_SENDS}",
        f"LABEL={cell_['id']}",
    ]
    # THE AXIS, AND THE ONE PLACE THE REFERENCE CELL DIFFERS. `resolve` answers the ABSENCE
    # of this key with `WholeRequest`; setting it equal to the row count instead gives
    # `SharedSession` with one piece, which is a different code path and a different plan
    # name. So the reference omits it rather than setting it to `rows`.
    #
    # `MINI_BATCH_FIXED` is never set either way: it pads the last piece up to the
    # mini-batch size, which only a graph with a pinned leading dimension needs. Padding
    # here would make a 100-row request at mb 30 score 120 rows and report a latency for
    # work nobody asked for.
    if cell_["mini_batch_configured"] is not None:
        env.append(f"MINI_BATCH_SIZE={cell_['mini_batch_configured']}")
        # Held constant across the grid so the mini-batch size is the only variable;
        # see CONCURRENCY_PER_SESSION. Omitted on the reference, which configures no
        # mini batch for it to apply to.
        env.append(f"CONCURRENCY_PER_SESSION={cell_['concurrency_per_session']}")
    return " ".join(env)


def start_hushar(cell_, host, ip):
    """Starts hushar and returns its banner, or None having recorded why not."""
    rc, out, err = ssh(
        ip,
        f"cd {DIR} && export HOME=/root && . ./env.sh && {hushar_env(cell_, host)} "
        f"scripts/benchmark/run_server.sh start 2>&1 | tail -60",
        timeout=1800,
    )
    if rc != 0 or "listening on" not in out:
        log(f"{cell_['id']} SERVER FAILED rc={rc}\n{out[-1500:]}\n{err[-500:]}")
        return None, {"failed": "server did not start", "server_out": out[-3000:]}
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]

    # The engine the graph was actually scored by, read from the server's own banner.
    m = re.search(r"\(runtime ([0-9]+\.[0-9]+\.[0-9]+)\)", out)
    if not m or m.group(1) != ORT_VER:
        got = m.group(1) if m else "unreported"
        log(f"{cell_['id']} ABORT: server loaded ONNX Runtime {got}, expected {ORT_VER}")
        return None, {"failed": f"ONNX Runtime {got}, expected {ORT_VER}",
                      "banner": banner}

    # THE AXIS OF THIS RUN, read back out of the server's own execution plan rather than
    # trusted from the environment this function just built. A cell that silently ran
    # UNSPLIT -- or split differently than asked -- would be reported under the wrong
    # mini-batch size, which is the one way this run could lie and be believed.
    #
    # The plan NAME comes from `ExecutionPlan::name()` and the detail from `describe()`,
    # and both are checked: the name alone cannot tell mb 50 from mb 10, and the detail
    # alone would pass a `SessionPerMiniBatch` cell that leased a group this run does not
    # configure.
    if f"plan     : {cell_['plan']} -- " not in out:
        log(f"{cell_['id']} ABORT: the banner does not report the {cell_['plan']!r} plan")
        return None, {"failed": f"banner does not report plan {cell_['plan']!r}",
                      "banner": banner}
    if cell_["plan_detail"] not in out:
        log(f"{cell_['id']} ABORT: the banner does not describe the plan as "
            f"{cell_['plan_detail']!r}")
        return None, {"failed": f"banner does not describe the plan as "
                                f"{cell_['plan_detail']!r}", "banner": banner}
    if cell_["reference"] and "mini batch" in out.split("plan     :")[1].split("\n")[0]:
        log(f"{cell_['id']} ABORT: the banner reports a split on the unsplit reference")
        return None, {"failed": "banner reports a mini batch on the unsplit reference",
                      "banner": banner}
    if "one session each" in out:
        log(f"{cell_['id']} ABORT: the banner reports a session per mini batch")
        return None, {"failed": "banner reports SessionPerMiniBatch; this run scores "
                                "every piece on one shared session", "banner": banner}

    # THE OVERLAP, which is the setting that makes the mini-batch size the only variable in
    # a comparison group. `concurrent` is what one session accepts in flight and `parallel`
    # is how many of those the host puts on distinct cores; the service reports both, and a
    # width that silently differed would change how much of a split is real.
    expect_overlap = (f"overlap  : {cell_['concurrent_mini_batches']} mini batch(es) at "
                      f"once per session, {cell_['parallel_mini_batches']} on distinct "
                      f"compute unit(s)")
    if expect_overlap not in out:
        log(f"{cell_['id']} ABORT: the banner does not report {expect_overlap!r}")
        return None, {"failed": f"banner does not report '{expect_overlap}'",
                      "banner": banner}
    # And that the figure was CONFIGURED rather than derived, on every split cell. Without
    # this a server built before `threading.concurrency_per_session` existed would silently
    # derive `cores / intra`, which for this grid's widths is 8 -- and the deeper splits
    # would run in waves while the report claimed they overlapped.
    if cell_["concurrency_per_session"] is not None:
        expect_cfg = (f"{cell_['concurrency_per_session']} configured by "
                      f'"threading.concurrency_per_session"')
        if expect_cfg not in out:
            log(f"{cell_['id']} ABORT: the banner does not report {expect_cfg!r}")
            return None, {"failed": f"banner does not report '{expect_cfg}'",
                          "banner": banner}

    # The spinning policy the server says it applied. The runtime accepts a setting it does
    # not know without a word, so the banner is the only evidence.
    want_spin = ("spin   : idle intra-op and inter-op threads "
                 + ("spin" if cell_["spin"] else "sleep"))
    if want_spin not in out:
        log(f"{cell_['id']} ABORT: banner does not say {want_spin!r}")
        return None, {"failed": f"banner missing '{want_spin}'", "banner": banner}

    # The session arithmetic, from the line the server derived it on.
    if f"x {cell_['sessions']} ONNX Runtime session(s)" not in out:
        log(f"{cell_['id']} ABORT: the banner does not report "
            f"{cell_['sessions']} session(s)")
        return None, {"failed": f"banner does not report 'x {cell_['sessions']} ONNX "
                                f"Runtime session(s)'", "banner": banner}
    if "in groups of" in out:
        log(f"{cell_['id']} ABORT: the banner reports a session group")
        return None, {"failed": "banner reports a session group; this run leases one "
                                "session per request", "banner": banner}

    # The wire, from the banner's own feature line. Every cell here is the FEATURE wire, so
    # a banner showing one binding means the deploy staged the tensor config and the cell
    # would be measuring a pre-vectorized wire under this run's name.
    n = WIRES[cell_["wire"]]["n_features"]
    expect = ("<- [features] transformed and concatenated" if n == 1
              else f"<- {n} features transformed and concatenated")
    if expect not in out:
        log(f"{cell_['id']} ABORT: banner does not show the {cell_['wire']} wire")
        return None, {"failed": f"banner does not show '{expect}'", "banner": banner}

    # No candidate model: every request scores the one resident model.
    if "A/B" in out:
        log(f"{cell_['id']} ABORT: the banner reports a candidate model")
        return None, {"failed": "banner reports a candidate model", "banner": banner}

    # THE LOG CONFIGURATION, READ BACK OUT OF THE BANNER. Every one of these could be wrong
    # while the server still served happily:
    #
    #   the rate     `Sampler::new` turns a rate into one-request-in-n, so a value that is
    #                not a unit fraction becomes the NEAREST one. A cell handed 0.22 would
    #                log 20% and report 0.22, and nothing downstream would catch it.
    #   the store    an s3:// URI that failed to parse would fall back to a local path,
    #                which writes to gp3 instead of to a bucket -- a different cost, and a
    #                cheaper one.
    #   the object   `batch_size` is per REQUEST, so the wrong one at 500 rows is a 2.6 GB
    #                object. The flush is what disturbs latency, so this is the number the
    #                cell's latency is most sensitive to.
    expect_log = (f"logging 1 request in {cell_['sample_every']} "
                  f"({cell_['sample_pct']:.1f}%)")
    if expect_log not in out:
        log(f"{cell_['id']} ABORT: the banner does not report {expect_log!r}")
        return None, {"failed": f"banner does not report '{expect_log}'",
                      "banner": banner}
    if "inference logs -> s3://" not in out or "Amazon S3" not in out:
        log(f"{cell_['id']} ABORT: the inference log is not going to S3")
        return None, {"failed": "banner does not report an Amazon S3 inference log",
                      "banner": banner}
    if f"every {cell_['log_batch']} batches" not in out:
        log(f"{cell_['id']} ABORT: the log batch size is not {cell_['log_batch']}")
        return None, {"failed": f"banner does not report 'every "
                                f"{cell_['log_batch']} batches'", "banner": banner}

    # Exactly one engine, so the cell is not sharing the host with a survivor.
    _, procs, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true")
    n_procs = procs.strip().split()[0] if procs.strip() else "?"
    if n_procs != "1":
        log(f"{cell_['id']} ABORT: {n_procs} hushar process(es), expected 1")
        return None, {"failed": f"{n_procs} hushar processes running", "banner": banner}
    return banner, {"ort_version": m.group(1),
                    "provider": "cpu",
                    "plan_confirmed": cell_["plan"],
                    "spin_confirmed": want_spin,
                    "mini_batch_confirmed": cell_["mini_batch_configured"],
                    "pieces": cell_["pieces"],
                    "sessions_per_model": cell_["sessions"],
                    "sessions_per_request": cell_["sessions_per_request"],
                    "intra_op_threads": cell_["intra"],
                    # Confirmed from the banner above, so the report quotes what the server
                    # said rather than what the driver asked for.
                    "log_confirmed": expect_log,
                    "log_uri": f"s3://<bucket>/{LOG_PREFIX}/{cell_['id']}",
                    "log_store": "Amazon S3"}


# ---------------------------------------------------------------- running one cell

def client_command(cell_, ip, label):
    """The one client invocation: warm-up then the measured window, on this host."""
    return (
        f"cd {DIR} && export HOME=/root && "
        f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={WIRES[cell_['wire']]['config']} "
        f"TAG=cell ROWS={cell_['rows']} TPS={cell_['tps']} "
        f"WARMUP={cell_['warmup']} DURATION={cell_['duration']} "
        f"SEGMENTS={cell_['segments']} CONCURRENCY={CONN} "
        f"LABEL={label} "
        f"scripts/benchmark/run_client.sh > /tmp/{label}.client.out 2>&1"
    )


def run_cell(cell_, host):
    ip = SERVERS[host]["ip"]
    cid = cell_["id"]
    key = f"{PREFIX}/cells/{cid}.json"
    if s3_has(key):
        log(f"{cid} already measured, skipping")
        return True

    log(f"{cid} start  {'whole request' if cell_['reference'] else 'mb' + str(cell_['mini_batch'])} "
        f"({cell_['pieces']} piece(s)"
        f"{'' if cell_['pieces'] == 1 else ', last ' + str(cell_['last_piece_rows'])}"
        f"{', ' + str(cell_['mini_batches_in_flight']) + ' in flight' if cell_['split'] else ''}"
        f"{' TIME-SLICED' if cell_['time_sliced'] else ''}) "
        f"s{cell_['sessions']}x{cell_['intra']} spin{cell_['spin']} "
        f"overlap{cell_['concurrent_mini_batches']} "
        f"{cell_['threads']}thr/{CORES}cpu "
        f"{cell_['rows']}rows x {cell_['tps']}tps = {cell_['rows_per_sec']} rows/s "
        f"log=1in{cell_['sample_every']}/batch{cell_['log_batch']}")

    # 1 · a clean server and a clean client, verified
    clean_ok = clean(ip, cid)
    banner, extra = start_hushar(cell_, host, ip)
    if banner is None:
        record(cell_, host, extra, ok=False)
        clean(ip, cid)
        return False

    # 2 · one client invocation, on this host. Started on a minute boundary so the measured
    # window is a whole number of whole CloudWatch buckets.
    label = cid
    started = time.time()
    if ALIGN:
        time.sleep(max(0, 60 - time.time() % 60))
    client_start = time.time()
    ssh(ip, "true")   # keep the control channel warm while the client runs
    proc = subprocess.Popen(client_command(cell_, ip, label), shell=True)

    # 3 · sample both hosts over the MEASURED window only
    time.sleep(cell_["warmup"])
    win_start = time.time()
    csample = f"/tmp/hushar-clientsample/{cid}.log"
    sh(f"mkdir -p /tmp/hushar-clientsample && cd {DIR} && "
       f"RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; true")
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/s-{cid}.log "
            f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1")

    try:
        client_rc = proc.wait(timeout=cell_["warmup"] + cell_["duration"] + 900)
    except subprocess.TimeoutExpired:
        proc.kill()
        client_rc = 124
    win_end = time.time()

    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/s-{cid}.log "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true")
    _, cpu, _ = ssh(ip, f"cd {DIR} && export HOME=/root && "
                        f"OUT=/tmp/hushar-server/s-{cid}.log "
                        f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    _, ccpu, _ = sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
                    f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")

    # 4 · the client's own numbers
    #
    # TWO FILES, AND THE SLICES ARE ONLY IN THE SECOND. `run_client.sh` pipes the client's
    # stdout through `summarise.py`, which appends ONE aggregate row per rate to
    # summary.txt -- the per-slice lines stay in the client's own log. Reading slices from
    # summary.txt therefore finds none, and the verdict silently becomes `0/0 slice(s)
    # held`, which is indistinguishable from a cell that held nothing. The smoke cell
    # caught exactly that.
    summary, slice_text, res = "", "", None
    try:
        summary = open(f"/tmp/hushar-client/{label}/summary.txt").read()
    except OSError:
        pass
    try:
        slice_text = open(
            f"/tmp/hushar-client/{label}/logs/{cell_['tps']}.client.log").read()
    except OSError:
        pass
    res = parse_summary(summary)
    slices = parse_slices(slice_text)
    if res and not slices:
        log(f"{cid} WARNING: no per-slice lines found in "
            f"logs/{cell_['tps']}.client.log; the verdict cannot say which slices held")

    if client_rc != 0 or not res:
        failure_tail = redacted_tail(summary, 2000)
        try:
            failure_tail = redacted_tail(open(f"/tmp/{label}.client.out").read(), 2000)
        except OSError:
            pass
        log(f"{cid} CLIENT FAILED rc={client_rc} summary={len(summary)}B\n"
            f"{(failure_tail or '')[-900:]}")
        record(cell_, host, {"failed": f"client rc={client_rc}, no parsable summary",
                             "client_out": failure_tail, "banner": banner}, ok=False)
        clean(ip, cid)
        return False

    # 5 · collect this cell's raw output NOW, in the session that measured it
    #
    # WHAT THE LOG PATH ACTUALLY DID, taken before the host is cleaned. Two numbers, and
    # neither is inferable from the configuration:
    #
    #   shed   the accumulator drops the OLDEST buffered batch when every send permit is
    #          out, printing one `logs:` line each time. Shedding is the designed behaviour
    #          -- a slow bucket must not become slow inference -- but a cell that shed was
    #          not paying the full cost of logging, so a latency read as "with logging on"
    #          would be understated.
    #   objects  what reached the bucket. Multiplied by the configured rows per object it
    #          should land near rows x tps x rate x window, and a large shortfall with no
    #          shed line means the sends were failing rather than queueing.
    _, shed_out, _ = ssh(ip, f"grep -h 'dropped to keep the newest' "
                             f"/tmp/hushar-server/{cid}.server.log 2>/dev/null "
                             f"| tail -1 || true")
    m_shed = re.search(r"\((\d+) dropped since start\)", shed_out or "")
    # A PUT that FAILED is a different fault from a send that queued, and the server
    # distinguishes them: `logs: lost N rows, could not write ...`. Counted, because a cell
    # whose writes were all rejected -- a missing permission, say -- would otherwise read as
    # a cell that logged for free.
    _, lost_out, _ = ssh(ip, f"grep -hc 'could not write' "
                             f"/tmp/hushar-server/{cid}.server.log 2>/dev/null || true")
    _, lost_line, _ = ssh(ip, f"grep -h 'could not write' "
                              f"/tmp/hushar-server/{cid}.server.log 2>/dev/null "
                              f"| tail -1 || true")
    _, obj_out, _ = ssh(ip, f"export HOME=/root; aws s3 ls "
                            f"s3://{BUCKET}/{LOG_PREFIX}/{cid}/ --recursive "
                            f"--summarize 2>/dev/null | tail -3 || true")
    m_objs = re.search(r"Total Objects:\s*(\d+)", obj_out or "")
    m_size = re.search(r"Total Size:\s*(\d+)", obj_out or "")
    log_facts = {
        "log_shed_batches": int(m_shed.group(1)) if m_shed else 0,
        "log_shed_line": redacted_tail(shed_out),
        "log_write_failures": int((lost_out or "0").strip() or 0),
        "log_write_failure_line": redacted_tail(lost_line),
        "log_objects": int(m_objs.group(1)) if m_objs else None,
        "log_bytes_written": int(m_size.group(1)) if m_size else None,
    }
    if log_facts["log_shed_batches"]:
        log(f"{cid} NOTE: the inference log shed {log_facts['log_shed_batches']} "
            f"batch(es) -- S3 did not keep up, so this cell paid less than the full "
            f"logging cost")
    if log_facts["log_write_failures"]:
        log(f"{cid} NOTE: {log_facts['log_write_failures']} inference-log write(s) "
            f"FAILED: {log_facts['log_write_failure_line']}")

    ssh(ip, f"export HOME=/root; cd /tmp/hushar-server && "
            f"for f in {cid}.server.log {cid}.service.json {cid}.model.json "
            f"s-{cid}.log; do "
            f"aws s3 cp $f s3://{BUCKET}/{PREFIX}/raw/server/{host}/ "
            f"--only-show-errors 2>/dev/null; done")
    s3_put_tree(f"/tmp/hushar-client/{label}", f"{PREFIX}/raw/client/{host}/{label}")
    s3_put(csample, f"{PREFIX}/raw/client/{host}/{label}/host-samples.log")

    clean(ip, cid)

    v = verdict(res, slices, cell_["tps"])
    doc = {
        "banner": banner,
        "summary": summary,
        # The per-slice lines as the client wrote them, so the verdict can be re-derived
        # from the evidence rather than trusted from `slices`.
        "slice_report": redacted_tail(slice_text, 4000),
        "result": res,
        "slices": slices,
        "verdict": v,
        "sustained": v["sustained"],
        "cpu_report": cpu,
        "client_cpu_report": ccpu,
        "server_cpu_pct": cpu_percent(cpu),
        "client_cpu_pct": cpu_percent(ccpu),
        "clean_slate_verified": clean_ok,
        "cloudwatch_namespace": "HusharBench",
        "cloudwatch_model_id": cid,
        "window": {"client_start": iso(client_start), "start": iso(win_start),
                   "end": iso(win_end),
                   "warmup_seconds": cell_["warmup"],
                   "duration_seconds": cell_["duration"],
                   "minute_aligned": ALIGN,
                   "note": f"one continuous {cell_['duration']}s load reported as "
                           f"{cell_['segments']} slice(s); host samples and CloudWatch "
                           f"both cover {int(win_end - win_start)}s from the end of "
                           f"warm-up"},
        "wall": {"cell_start": iso(started), "cell_end": iso(win_end)},
    }
    doc.update(extra)
    doc.update(log_facts)
    record(cell_, host, doc)
    log(f"{cid} done   p50={res['p50']} p95={res['p95']} p99={res['p99']} "
        f"achieved={res['achieved']}/{res['offered']} shed={res['shed']} "
        f"srvcpu={cpu_percent(cpu)} {v['state']} "
        f"({v['slices_held']}/{v['slices_total']} slice(s) held) "
        f"-- settling {cell_['settle']}s")
    time.sleep(cell_["settle"])
    return True


def record(cell_, host, extra, ok=True, category="cells"):
    """A measured cell goes to cells/, which is what "already done" means.

    A failed attempt goes to failures/ under its own timestamp, so it is kept for diagnosis
    but a resume retries the cell rather than treating the failure as a result.
    """
    doc = dict(cell_)
    doc.update(extra)
    # `host` is the PLAN name, so every rig calls its first server `c9g-1` and two different
    # machines record the same value. The address discriminates them, which is what lets a
    # report say that a row added in a later session sits on another instance.
    doc.update(host=host, server_ip=SERVERS[host]["ip"],
               instance=SERVERS[host]["instance"],
               arch=SERVERS[host]["arch"], cores=CORES, conn=CONN,
               run_sample_rate=SAMPLE_RATE,
               run_log_rows_per_object=LOG_ROWS_PER_OBJECT,
               run_log_sends=LOG_SENDS,
               run_log_prefix=f"s3://<bucket>/{LOG_PREFIX}",
               wire_note=WIRES[cell_["wire"]]["note"],
               model_file=MODEL_FILE, model_md5=MODEL_MD5, model_size=MODEL_SIZE,
               onnxruntime=ORT_VER,
               rev=REV, run=RUN)
    for field in ("ort_so", "kernel", "nproc"):
        if HOST_FACTS.get(field) is not None:
            doc[f"host_{field}"] = HOST_FACTS[field]
    p = f"/tmp/{cell_['id']}.json"
    # Masked here rather than at every site that builds a string: this is the one place a
    # cell becomes a file, and the banner it carries names the S3 log destination, whose
    # bucket embeds an account id.
    json.dump(redact(doc), open(p, "w"), indent=2)
    if ok:
        s3_put(p, f"{PREFIX}/{category}/{cell_['id']}.json")
    else:
        s3_put(p, f"{PREFIX}/failures/{cell_['id']}-{int(time.time())}.json")


def put_lane_log(host):
    """This orchestrator's own log, beside the cells.

    A cell carries its own timestamps, so losing this is survivable, but the log is the only
    record of what a lane SKIPPED and why. The path comes from the launcher through LANE_LOG
    rather than being rebuilt here: a second guess at the filename is a second chance to get
    it wrong.
    """
    candidates = [p for p in (os.environ.get("LANE_LOG"),
                              f"/root/orchestrator-{LANE_TAG}-{host}.log",
                              f"/root/orchestrator-{host}.log") if p]
    for path in candidates:
        if os.path.exists(path):
            s3_put(path, f"{PREFIX}/raw/driver/lane-{host}.log")
            return
    log(f"WARNING: no lane log found at any of {candidates}")


# ---------------------------------------------------------------- preflight

HOST_FACTS = {}


def preflight(host):
    """Refuses to measure a machine that is not the one the report will name.

    Every figure is attributed to an instance type, an engine version, a graph and a server
    build, so each is checked against the host itself rather than trusted from this file.
    """
    ip = SERVERS[host]["ip"]
    want = SERVERS[host]["instance"]
    gen = f"{DIR}/{GEN}"
    rc, out, err = ssh(ip, "set -e; "
                       "T=$(curl -sf -X PUT http://169.254.169.254/latest/api/token "
                       "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); "
                       "echo type=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/instance-type); "
                       "echo nproc=$(nproc); "
                       "echo kernel=$(uname -r); "
                       f"echo providers=$(ls -1 /root/ort-cpu-{ORT_VER}/lib/ 2>/dev/null "
                       "| grep providers_ | grep -vc providers_shared || true); "
                       f"echo ortso=$(readlink -f /root/ort-cpu-{ORT_VER}/lib/"
                       "libonnxruntime.so | sed 's/.*libonnxruntime\\.so\\.//'); "
                       f"echo md5=$(md5sum {gen}/{MODEL_FILE} | cut -c1-32); "
                       f"echo featcfg=$(test -f {gen}/bench_model_config.json "
                       "&& echo yes || echo no)")
    if rc != 0:
        sys.exit(f"preflight: cannot reach {host} at {ip}: {err[-300:]}")
    f = dict(l.split("=", 1) for l in out.split() if "=" in l)
    log(f"preflight {host}: {f}")

    if f.get("type") != want:
        sys.exit(f"preflight: {host} is {f.get('type')} but the run claims {want}")
    if f.get("nproc") != str(CORES):
        sys.exit(f"preflight: {host} has {f.get('nproc')} CPUs, expected {CORES}")
    # This run is CPU only. A library that merely COULD register an accelerator
    # disqualifies the host rather than being trusted not to.
    if f.get("providers") != "0":
        sys.exit(f"preflight: {host} ships {f.get('providers')} accelerator provider "
                 f"library/libraries; this run is CPU only")
    if f.get("ortso") != ORT_VER:
        sys.exit(f"preflight: {host} has libonnxruntime.so.{f.get('ortso')} in "
                 f"/root/ort-cpu-{ORT_VER}/lib, expected {ORT_VER}")
    if f.get("md5") != MODEL_MD5:
        sys.exit(f"preflight: {host} model md5 {f.get('md5')}, expected {MODEL_MD5}")
    if f.get("featcfg") != "yes":
        sys.exit(f"preflight: {host} is missing {gen}/bench_model_config.json; "
                 f"the feature wire is what this run measures")

    HOST_FACTS.update(ort_so=f.get("ortso"), kernel=f.get("kernel"),
                      nproc=f.get("nproc"))
    log(f"preflight {host}: OK, {want}, {CORES} CPUs, CPU-only runtime, "
        f"model md5 {f.get('md5')}, feature wire staged")


# ---------------------------------------------------------------- main

def drive(cells, host, category="cells"):
    if MAX_CELLS:
        cells = cells[:MAX_CELLS]
        log(f"MAX_CELLS={MAX_CELLS}: driving only {len(cells)} cell(s)")
    for c in cells:
        for attempt in (1, 2):
            try:
                if run_cell(c, host):
                    break
                log(f"{c['id']} attempt {attempt} did not measure")
            except Exception as e:      # a cell must never take the run down
                log(f"{c['id']} attempt {attempt} EXCEPTION {type(e).__name__}: {e}")
                try:
                    record(c, host, {"failed": f"{type(e).__name__}: {e}"}, ok=False)
                except Exception:
                    pass
            try:
                clean(SERVERS[host]["ip"], c["id"])
            except Exception:
                pass
            time.sleep(30)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: sweep.py <server-host>")
    host = sys.argv[1]
    if host not in SERVERS:
        sys.exit(f"{host} is not in RIG={sorted(SERVERS)}")

    preflight(host)
    cells = list(main_cells())
    per_cell_min = (WARMUP + DURATION + SETTLE + 90) / 60.0
    log(f"run={RUN} rev={REV} host={host} size={MODEL_SIZE} "
        f"loads={LOADS} mini_batches={MINI_BATCHES} "
        f"reference={'whole request' if REFERENCE else 'none'} "
        f"repeats={REPEATS}x{REPEAT_MB} "
        f"sessions={SESSION_COUNTS} spr={SESSIONS_PER_REQUEST} "
        f"intra={{n: intra_for(n) for n in SESSION_COUNTS}} spin={SPIN} "
        f"concurrency_per_session={CONCURRENCY_PER_SESSION} "
        f"(on distinct cores: {{n: parallel_for(n) for n in SESSION_COUNTS}}) "
        f"wire=features/210 vectorize=server candidate=none "
        f"sample_rate={SAMPLE_RATE} log=s3://<bucket>/{LOG_PREFIX} "
        f"log_rows_per_object={LOG_ROWS_PER_OBJECT} log_sends={LOG_SENDS} "
        f"warmup={WARMUP}s measured={DURATION}s/{SEGMENTS}slices settle={SETTLE}s")
    if _time_sliced:
        log("NOTE: these cells put more pieces in flight than the host has cores for "
            "them, so the surplus time-slices the session pool: "
            + ", ".join(f"s{n} {r}rows mb{s}={p} pieces > {parallel_for(n)} core(s)"
                        for n, r, s, p in _time_sliced))
    log(f"{len(cells)} cell(s): " + ", ".join(
        ("whole" if c["reference"] else f"mb{c['mini_batch']}")
        + f"@{c['rows']}x{c['tps']}/s{c['sessions']}"
        + (f" rep{c['repeat']}" if c["repeat"] else "") for c in cells))
    log(f"about {len(cells) * per_cell_min / 60:.1f} h at "
        f"{per_cell_min:.1f} min/cell")

    # A selection record, so a reader of the bucket can tell what this lane INTENDED to
    # measure from what it managed to.
    p = f"/tmp/selection-{MODEL_SIZE}.json"
    json.dump(dict(run=RUN, size=MODEL_SIZE, host=host, rev=REV,
                   axis="mini_batch_size", pinned=False,
                   loads=[list(l) for l in LOADS],
                   mini_batches={str(k): v for k, v in MINI_BATCHES.items()},
                   reference=dict(enabled=REFERENCE, plan="whole request"),
                   repeats=dict(count=REPEATS,
                                at={str(k): v for k, v in REPEAT_MB.items()}),
                   session_counts=SESSION_COUNTS,
                   sessions_per_request=SESSIONS_PER_REQUEST,
                   intra_op_threads={str(n): intra_for(n) for n in SESSION_COUNTS},
                   allow_spinning=SPIN,
                   concurrency_per_session=CONCURRENCY_PER_SESSION,
                   parallel_mini_batches={str(n): parallel_for(n)
                                          for n in SESSION_COUNTS},
                   time_sliced=[dict(sessions=n, rows=r, mini_batch=s, pieces=p)
                                for n, r, s, p in _time_sliced],
                   rule="one cell per mini-batch size per load per session count. Inside a "
                        "group only the mini-batch size differs: the session count, the "
                        "intra-op width and the per-session concurrency are all held. The "
                        "size equal to the row count is the no-split baseline and runs "
                        "first; one `whole request` cell per group configures no mini batch "
                        "at all and prices the split path's own overhead",
                   cells=[c["id"] for c in cells]), open(p, "w"), indent=2)
    s3_put(p, f"{PREFIX}/selection/{MODEL_SIZE}.json")

    drive(cells, host)
    put_lane_log(host)
    log(f"host={host} lane complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
