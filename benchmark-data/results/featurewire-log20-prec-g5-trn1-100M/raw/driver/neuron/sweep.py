#!/usr/bin/env python3
"""Drives one server host through its lane of the trn1 no-mini-batch session sweep.

    HUSHAR_BENCH_BUCKET=... REV=... MODEL_SIZE=10m sweep.py <server-host>

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS. Every Trainium campaign in this series measured hushar with its
mini-batch split engaged, which is a scheduling decision Triton has no counterpart for.
A hushar-against-Triton column therefore priced two servers AND that decision at once.
This run removes it:

    WITH THE REQUEST SCORED IN ONE CALL, WHAT DOES ONE trn1.2xlarge DO,
    AND HOW MANY MODEL REPLICAS DOES IT WANT?

So the only knob that moves is `threading.sessions_per_model`, and the numbers are
readable against `nominibatch-sessions-gpu-g5-10M-20M`, which asked this of one A10G with
a Triton arm beside it.

ONE AXIS. `sessions` is Neuron model replicas, one complete NEFF copy each, placed one
per core until every core has one. `threading.sessions_per_request` is 1, so a request
leases ONE replica and `sessions_per_model` IS the requests in flight -- the same thing
Triton's instance count means, which is what makes the comparison a pair.

    sessions   replicas/core   requests in flight
    2                      1                    2
    4                      2                    4
    8                      4                    8

NO MINI BATCH, AND ON THIS PROVIDER THAT IS `mini_batch_size == rows`. A NEFF has a
FIXED leading dimension, so hushar's Neuron backend requires the row count to be
declared and `run_server.sh` refuses `PROVIDER=neuron` without it. Scoring 100 rows in
one call of 100 is therefore spelled `mini_batch_size = 100`, which yields `pieces = 1`
and one execution per request. That differs from the GPU counterpart, where the unsplit
configuration is the ABSENCE of the key, and the difference is in the configuration only:
both are one engine call per request. `cell()` derives the mini batch FROM the row count
and `SPLIT_FORBIDDEN` refuses any environment variable that would reintroduce a split,
so a command line copied from the previous campaign fails loudly.

WHAT BOUNDS THE ANSWER. `backend.neuron_cores_per_replica` is pinned at 1 and
`backend.execution_contexts_per_replica` at 1, both by instruction: these are small
tabular models, nothing here needs tensor parallelism, and with one piece per request a
second context per replica would be loaded and never leased.

A NeuronCore executes ONE NEFF at a time, and a trn1.2xlarge has two. So device capacity
is fixed at two concurrent executions no matter how many replicas are resident, and
sessions above 2 buy queue lanes and host-side overlap rather than device throughput.
`config-sweep-neuron-trn1-10M-50M` saw exactly that: 4 replicas held a rate 2 could not,
at 16.48 ms against 10.5.

THE WIRE IS PRE-VECTORIZED, BY INSTRUCTION. The client sends one feature per row that
already IS the finished [rows, 690] FP32 vector, and the model configuration transforms
it with `identity`, so the server copies it into the input and derives nothing. It is
2.54x the bytes of the 210-named-feature wire -- 100,000 rows/s is 276 MB/s inbound
against a `trn1.2xlarge` baseline of 3.125 Gbps -- so each cell measures its own rx and
tx bytes over the window, and a network ceiling appears as a number rather than as a
device limit.

THE RATES WERE SPECIFIED, NOT PROBED, AND THE 20M ONES ARE EXPECTED TO BE OVER THE
CEILING. `minibatch-neuron-trn1-10M-20M` measured 20M on this box as device-bound: no 20M
cell held either specified rate across its whole 20-cell grid, the best being 610.0 of
750 req/s at 100 rows and 112.5 of 150 at 500 rows, with NeuronCore occupancy at 90-99%.
A cell that does not hold its rate is reported by ACHIEVED RATE and enters no latency
comparison, because the median of a rate that was not held is queue depth. Those cells
still measure the ceiling.

TWO PHASES. There is no `control` phase: contexts are pinned at 1 for every cell, so
there is nothing to price.

    prime    one NEFF per row count, per model -- two shapes, b100 and b500. Restored
             from the S3 cache when an earlier campaign built it, compiled otherwise,
             then PROVED loadable and servable by a fresh server before any cell runs.
             This phase also derives the tensor-wire configuration from the compiled one,
             so the width the client sends is read from the model rather than written
             down here.
    main     the 6 cells per size. 5 min warm-up, 15 min measured as one continuous load
             in 15 slices, a fresh server per cell, 1 min idle after it stops.

FIFTEEN ONE-MINUTE SLICES, NOT ONE FIFTEEN-MINUTE NUMBER. A window reported as a single
figure cannot tell a rate that is sustained from a rate that is sustained for a while. The
client reports the measured window as consecutive slices, so a configuration that falls
behind part-way through is visible instead of averaged away. A cell is `sustained` only if
EVERY slice held it; one that held some is `partial`, with the slice it broke in. One
minute per slice keeps each slice a whole CloudWatch bucket.
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

REV = os.environ.get("REV") or sys.exit("set REV")
DIR = f"/root/hushar-bench-{REV}"
RUN = os.environ.get("RUN", "featurewire-log20-prec-g5-trn1-100M")
PREFIX = f"runs/{RUN}"

MODEL_SIZE = os.environ.get("MODEL_SIZE") or sys.exit("set MODEL_SIZE, e.g. 100m")
# The compile target, as compile_neuron.py wants it. The model is pinned by SEED and
# TARGET rather than by a file hash: there is no graph to hash before the host builds it.
MODEL_TARGET = {"10m": "10M", "20m": "20M", "100m": "100M"}.get(MODEL_SIZE) \
    or sys.exit(f"MODEL_SIZE={MODEL_SIZE!r}: one of 10m, 20m, 100m")
# Continuity with the campaigns that got this from the same code path. A different number
# means the generator moved and a Trainium figure can no longer be read against them, so
# prime refuses rather than quietly measuring another model. `.get` rather than `[]`: a
# size nobody has measured on Trainium has nothing to check against, and the KeyError a
# subscript would raise here fires at IMPORT time -- before preflight, before any log
# line, with a traceback that names a dict rather than the thing that is missing.
EXPECT_PARAMS = {"10m": 9_142_789, "20m": 18_599_941, "100m": None}.get(MODEL_SIZE)
ALLOW_PARAM_DRIFT = os.environ.get("ALLOW_PARAM_DRIFT", "") == "1"

# ---------------------------------------------------------------- the precision
#
# THE AXIS OF THIS ROUND, PASSED AS AN EXPLICIT FLAG ON BOTH ARMS.
#
# `neuronx-cc`'s CLI reference gives `--auto-cast` a default of `matmult`, which would mean
# a compile passing no flag casts FP32 matrix multiplies to BF16. THAT DEFAULT DOES NOT
# REACH THIS CODE PATH: `torch_neuronx.trace` builds its own argument list, and a measured
# three-way compile on this host type -- no flag, `--auto-cast=none`, `--auto-cast=matmult`
# -- produced a BYTE-IDENTICAL NEFF and an identical trace error for the first two, against
# a 0.40x NEFF and a three-orders-larger error for the third. See
# `results/featurewire-log20-prec-g5-trn1-100M/raw/probe/autocast-default.txt`.
#
# So the earlier Trainium campaigns in this series, which passed no flag, were GENUINELY
# FP32 and their `"precision": "fp32"` was right. This round's FP32 arm IS comparable with
# `featurewire-log20-g5-trn1-10M-20M`'s Trainium rows -- same wire, same log rate, same
# session count, same cast, different model size.
#
# The flag is passed anyway, for a narrower reason that still holds: a precision label
# must not rest on an undocumented framework default that a release can change without
# saying so. compile_neuron.py owns the table and PROVES each arm from the
# compiled-versus-PyTorch error -- ~1e-6 at FP32, three orders above it once the matmuls
# are cast -- refusing to write a NEFF whose error falls outside the window its label
# requires. That check is what caught the mistaken belief described above.
PRECISION = os.environ.get("PRECISION", "fp32")
PRECISIONS = {
    "fp32": dict(auto_cast="none", auto_cast_type=None,
                 note="--auto-cast=none, so every operation keeps the declared dtype"),
    "bf16": dict(auto_cast="matmult", auto_cast_type="bf16",
                 note="--auto-cast=matmult --auto-cast-type=bf16, so the matrix-multiply "
                      "engine runs in BF16 and normalization and accumulation stay FP32 "
                      "-- the same division TensorRT's --bf16 makes on the GPU"),
}
if PRECISION not in PRECISIONS:
    sys.exit(f"PRECISION={PRECISION!r}: one of {sorted(PRECISIONS)}")
PREC_TAG = PRECISION
# BYTES PER STORED WEIGHT, for the "is this even the model graph" floor in prime_shape.
# A BF16 NEFF holds its cast constants at half the width, so an FP32-derived floor would
# REFUSE a perfectly good BF16 artifact -- the gate would fire on the arm it was never
# written for.
WEIGHT_BYTES = {"fp32": 4, "bf16": 2}[PRECISION]

# PER PRECISION, NOT JUST PER SIZE. The NEFF cache is keyed by path, so two arms sharing
# one key would mean the second restores the first's bytes, passes every gate, and
# publishes a cell whose precision column is a lie. Earlier campaigns wrote
# `models/neuron/<size>/`; this layout nests a precision under it, which is disjoint from
# those keys and so cannot collide with what they cached.
GEN = f"benchmark-data/generated/neuron/{MODEL_SIZE}/{PREC_TAG}"
MODELS_S3 = f"models/neuron/{MODEL_SIZE}/{PREC_TAG}"

WARMUP = int(os.environ.get("WARMUP", "300"))       # 5 min, discarded
DURATION = int(os.environ.get("DURATION", "900"))   # 15 min, measured, continuous
SEGMENTS = int(os.environ.get("SEGMENTS", "15"))    # reported as 15 x 1 min
SETTLE = int(os.environ.get("SETTLE", "60"))        # 1 min between cells
# Start the client on a minute boundary, so the measured window is a whole number of whole
# CloudWatch buckets and nothing from warm-up shares a bucket with the measurement.
ALIGN = os.environ.get("ALIGN", "1") == "1"

CONN = 512            # threading.connection_concurrency, and the client's in-flight cap

# ---------------------------------------------------------------- the inference log
#
# FEATURE LOGGING AT 50%, WHICH NO EARLIER TRAINIUM RUN MEASURED. The sampler is COUNTED
# rather than randomised -- `Sampler::new(0.5)` keeps one request in two exactly -- so the
# rate is a property of the configuration and not of a window's luck. The banner prints
# what was chosen and `start_hushar` asserts it.
SAMPLE_RATE = os.environ.get("SAMPLE_RATE", "0.2")
# TO S3, NOT TO THE INSTANCE'S DISK, because that is the path a deployment uses and the
# cost this run is measuring. One prefix per RUN so it can be deleted in one call.
LOG_PREFIX = os.environ.get("LOG_PREFIX", f"inference-logs/{RUN}")
# ONE OBJECT PER 50,000 LOGGED ROWS. `batch_size` counts REQUESTS, and a request is 100
# rows in one cell and 500 in another, so a fixed count would hand S3 a 56 MB object at
# one packaging and a 266 MB object at the other -- and the flush is what disturbs
# latency. Scaling by the row count holds the object at ~53 MB and the cadence at ~1/s on
# the heaviest cell of each packaging, so the log path disturbs every cell the same way.
# The same rule as the GPU lane, so the two accelerators are comparable on this axis too.
LOG_ROWS_PER_OBJECT = int(os.environ.get("LOG_ROWS_PER_OBJECT", "50000"))


def log_batch_for(rows):
    """Requests buffered before one object, so that an object is a fixed number of rows."""
    return max(1, LOG_ROWS_PER_OBJECT // rows)


LOG_SENDS = int(os.environ.get("LOG_SENDS", "4"))
# 4 async workers on 8 vCPUs, which is what every Neuron run in this series used on the
# same host, held here so this run is comparable with the tensor-wire one at the same
# load. On THIS wire the pool does real work -- 210 transformations per row, and the
# log's serialisation -- where on the tensor wire the transformation was a copy.
WORKER_THREADS = int(os.environ.get("WORKER_THREADS", "4"))
# Small batches, so a metric batch that straddles the warm-up boundary carries under a
# second of warm-up into the first measured minute rather than several.
METRICS_BATCH = 100
PORT = 8279
CORES = 8                 # vCPUs on trn1.2xlarge
NEURON_CORES = int(os.environ.get("NEURON_CORES", "2"))
# `trn1.2xlarge`, from DescribeInstanceTypes: 3.125 Gbps sustained, 12.5 Gbps burst. The
# SUSTAINED figure is the one a 15-minute window is measured against.
NET_BASELINE_GBPS = float(os.environ.get("NET_BASELINE_GBPS", "3.125"))
# PINNED BY INSTRUCTION. See the module docstring for what they cost. One core per
# replica bounds the split's device term: a NeuronCore runs one NEFF at a time and a
# request's whole group of replicas sits on one core, so the pieces serialize there. One
# context per replica is then exactly sufficient rather than a handicap -- a request
# leases `pieces` replicas and puts ONE piece on each, so a second context per replica
# would be loaded and never leased.
CORES_PER_REPLICA = int(os.environ.get("CORES_PER_REPLICA", "1"))
CONTEXTS_PER_REPLICA = int(os.environ.get("CONTEXTS_PER_REPLICA", "1"))

MAX_CELLS = int(os.environ.get("MAX_CELLS", "0"))   # smoke testing only
LANE_TAG = os.environ.get("LANE_TAG", "main")

if SEGMENTS <= 0 or DURATION % SEGMENTS:
    sys.exit(f"DURATION={DURATION} is not a whole number of SEGMENTS={SEGMENTS}; a "
             f"part slice is not a slice the verdict can use")

WIDTH = 690

# ---------------------------------------------------------------- the loads
#
# TWO PACKAGINGS OF THE SAME ROW THROUGHPUT per size, so the grid also answers -- for
# free -- whether a bigger request is cheaper per candidate on a NeuronCore. That is the
# lever the GPU run found to be the real one there, and the opposite of the CPU's.
#
# THE 20M RATES BRACKET A CEILING THIS SERIES HAS NOW MEASURED TWICE, which is why there
# are four of them rather than two. `featurewire-log50-g5-trn1-10M-20M` recorded the
# device span per request on this exact configuration: 3.861 ms at 100 rows and 19.012 ms
# at 500. A NeuronCore executes one NEFF at a time and this host has two, so the device
# ceiling is
#
#     100 rows   2 / 0.003861 = 518 req/s
#     500 rows   2 / 0.019012 = 105 req/s
#
# and that round's 500 and 100 req/s offered -- 97% and 95% of those -- came back at 490.6
# (0/15 slices) and 99.4 (5/15). Dropping the log from 50% to 20% removes host work, not
# device work, so the ceiling barely moves. Each packaging therefore gets a pair that
# straddles the knee, at 87%/77% and 76%/71% device occupancy, so at least one of each
# reports a LATENCY and the two together say where the knee is. Four cells on this lane is
# 1.5 h, which still finishes well inside the A10G lane.
_LOADS = {
    ("10m", "fp32"): [(100, 1000), (500, 200)],       # 100,000 rows/s both
    ("20m", "fp32"): [(100, 450), (100, 400), (500, 80), (500, 75)],
    # ---- WAVE A, AS ORIGINALLY SPECIFIED. Set from the A10G, where the same four cells
    # were MEASURED holding in `nominibatch-precision-gpu-g5-100M` at 93.5%, 79.8%, 89.1%
    # and 72.7% device occupancy. One trn1.2xlarge is two NeuronCores, each executing one
    # NEFF at a time, and none of the four held. What they measured instead is the CEILING,
    # which no campaign had for this model size on this device:
    #
    #     fp32  100 rows   90.9 req/s achieved,  22.00 ms per call
    #     fp32  500 rows   17.4 req/s achieved, 114.89 ms per call
    #     bf16  100 rows  127.1 req/s achieved,  16.02 ms per call
    #     bf16  500 rows   27.4 req/s achieved,  73.10 ms per call
    #
    # ---- WAVE B, SET FROM THOSE MEASUREMENTS RATHER THAN FROM A PROJECTION, AND CHOSEN
    # TO BE SUSTAINABLE. The point of wave B is a SUSTAINED latency on this device, which
    # wave A could not give: a median taken while requests queue measures the queue. So
    # every wave-B rate sits at or below 80% device occupancy, `offered x ms-per-call /
    # 2 cores`, and every one is a round number:
    #
    #     rows x TPS   fp32 occupancy   bf16 occupancy
    #     100 x  70         77%              56%
    #     500 x  10         57%              36%
    #     100 x 100        110%  (refused)   80%
    #     500 x  20        115%  (refused)   73%
    #
    # WHERE 80% COMES FROM. This series has sustained a trn1 cell at 87% computed
    # occupancy -- `featurewire-log20-g5-trn1-10M-20M`'s 20M 100 x 450, which held 15/15
    # slices -- and has failed one at 97%, the control round's 100 x 500 at 490.6 achieved.
    # So the knee is somewhere above 87%, and 80% is inside it with room for the 2% rate
    # tolerance a cell has to hold across all fifteen slices.
    #
    # Rates at 88% (fp32 100 x 80) and 96% (bf16 100 x 120) were considered and dropped:
    # both are close enough to the knee that they are probes of where it is rather than
    # measurements of a latency, and this wave is for the latency.
    #
    # THE FIRST TWO RATES ARE A MATCHED PAIR, which is what makes this a precision
    # comparison: both arms are offered the SAME rate, so the difference between their
    # medians is the cast and nothing else. Wave A could not do that -- it gave BF16 twice
    # the rate -- so its arms were only comparable on device time. The last two are BF16's
    # own higher sustainable rates, where FP32 cannot follow; they price the THROUGHPUT the
    # cast buys rather than the latency.
    #
    # Wave A's rates stay in the list so a resumed lane is a no-op over them rather than a
    # re-measurement: `run_cell` skips any cell whose record S3 already holds.
    ("100m", "fp32"): [(100, 100), (500, 20), (100, 70), (500, 10)],
    ("100m", "bf16"): [(100, 200), (500, 40), (100, 70), (500, 10),
                       (100, 100), (500, 20)],
}


def _parse_loads(spec):
    """`100x200,100x750`, or `100m=100x200;20m=100x200,100x750` to differ by size."""
    if "=" in spec:
        for part in spec.split(";"):
            size, _, rates = part.partition("=")
            if size.strip() == MODEL_SIZE:
                spec = rates
                break
        else:
            sys.exit(f"LOADS={spec!r} names no rates for {MODEL_SIZE}")
    return [tuple(int(x) for x in pair.split("x")) for pair in spec.split(",") if pair]


if os.environ.get("LOADS"):
    LOADS = _parse_loads(os.environ["LOADS"])
elif (MODEL_SIZE, PRECISION) in _LOADS:
    LOADS = _LOADS[(MODEL_SIZE, PRECISION)]
else:
    sys.exit(f"no rates for {MODEL_SIZE}/{PRECISION}; set LOADS, e.g. LOADS=100x200")

# ---------------------------------------------------------------- no mini batch
#
# THE REQUEST IS NEVER CUT, AND THAT IS NOT A SETTING HERE. The mini batch is DERIVED
# from the row count -- `mini_batch == rows` is one NEFF call of exactly the rows the
# request carries -- so there is no axis, no baseline to divide by and nothing to sweep.
#
# It is spelled as a size rather than as an absence because of the provider: a NEFF has a
# FIXED leading dimension, hushar's Neuron backend takes the row count from the
# configuration, and `run_server.sh` refuses `PROVIDER=neuron` without `MINI_BATCH_SIZE`.
# The GPU counterpart omits the key entirely; both mean one engine call per request.
MINI_BATCH_FOR_ROWS = {rows: rows for rows, _ in LOADS}

# A command line copied from `minibatch-sessions-neuron-trn1-10M-20M` would otherwise
# quietly measure a different run. Refused BY NAME, so it fails at startup saying what to
# set instead.
SPLIT_FORBIDDEN = [
    ("MINI_BATCHES", "this run scores every request in one call; set SESSIONS"),
    ("MINI_BATCH_SIZE", "this run scores every request in one call; set SESSIONS"),
    ("PIECES", "this run scores every request in one call; set SESSIONS"),
]
for _name, _why in SPLIT_FORBIDDEN:
    if os.environ.get(_name):
        sys.exit(f"{_name} is not an axis of {RUN}: {_why}")

# ---------------------------------------------------------------- the sessions
#
# PINNED AT 2, NOT SWEPT. `threading.sessions_per_model` -- Neuron model replicas, one
# complete NEFF copy each. `nominibatch-sessions-neuron-trn1-10M-20M` swept 2, 4 and 8 at
# the 10M loads and 2 was the lowest sustained median in both; above one replica per core
# a count buys queue lanes rather than device throughput, because a NeuronCore executes
# one NEFF at a time and this host has two. So the count is held at 2 -- one replica per
# core, the whole device and nothing oversubscribed -- and the cells go to the wire.
#
# `threading.sessions_per_request` is 1 for every cell, so a request leases ONE replica
# and `sessions_per_model` IS the requests in flight, the same quantity Triton's
# `instance_group` count sets. That equivalence is what makes a trn1 row readable against
# the A10G lane of this run.
SESSIONS = [int(x) for x in os.environ.get("SESSIONS", "2").split(",") if x.strip()]
if not SESSIONS:
    sys.exit("SESSIONS names no session counts")
for _s in SESSIONS:
    if _s <= 0:
        sys.exit(f"SESSIONS={_s} is not a session count")
    # An uneven spread would put more replicas on one core than another, and the cell
    # would not mean what its id says. The service places replica `r` on core
    # `r / replicas_per_core`, so a whole number of cores is the condition.
    if _s % NEURON_CORES:
        sys.exit(f"SESSIONS={_s} is not a whole number of {NEURON_CORES} NeuronCores; "
                 f"an uneven spread would put more replicas on one core than another "
                 f"and the cell would not mean what its id says")

# Every distinct NEFF shape this lane needs, which is what `prime` iterates. A NEFF's
# leading dimension is fixed and a request is one call of its own row count, so a shape
# is a ROW COUNT here -- two per size, b100 and b500, shared by all three session counts
# because a session count loads more copies of the same NEFF rather than another shape.
SHAPES = sorted({MINI_BATCH_FOR_ROWS[rows] for rows, _ in LOADS})

# Filled from the environment by launch.sh, which reads rig-state.json. NOT written here:
# this run launches its own hosts, so an address in this file would be a guess.
SERVERS = {}
for _entry in os.environ.get("RIG", "").split(";"):
    if not _entry.strip():
        continue
    _name, _ip = _entry.split("=", 1)
    SERVERS[_name.strip()] = dict(ip=_ip.strip(), instance="trn1.2xlarge", arch="x86_64",
                                  model_id=_name.strip())
if not SERVERS:
    sys.exit("set RIG, e.g. RIG='trn1-1=172.31.0.1' -- launch.sh does this")

N_FEATURES = 210
TENSOR_INPUT = "features"
WIRE_NOTE = ("210 named features per row, transformed and concatenated by hushar into the "
             "model's one [rows, 690] FP32 input; server-side feature vectorization is ON. "
             "2.54x FEWER bytes on the wire than the pre-vectorized 690-float tensor the "
             "earlier Trainium runs sent, at the cost of 210 transformations per row")

PHASE = os.environ.get("PHASE", "all")
if PHASE not in ("all", "prime", "main"):
    sys.exit(f"PHASE={PHASE!r}: use all, prime or main")


# ---------------------------------------------------------------- the artifacts

def artifact(mb):
    """Where one mini-batch size's NEFF, configuration and metadata live.

    `s3` is the CACHE. A NEFF is expensive to build and is a pure function of the seeded
    architecture, the target parameter count and the batch size, so the first lane to
    build one publishes it and every later lane, re-deploy and resume restores it. That
    is also what makes `prime` resumable, which matters because it is the long phase.

    `config` is the configuration `compile_neuron.py` WROTE beside the NEFF, and it is
    what this run serves unmodified: it names the 210 features, so the server transforms
    and concatenates them into the NEFF's one input. The cache is keyed on size and batch
    alone, shared with every earlier Trainium campaign's entries, because the NEFF is the
    same bytes whichever wire serves it -- the vectorization block is a server setting,
    not part of the graph. Which is exactly why the earlier runs could derive a
    one-feature pre-vectorized configuration from this same file; this run does not.
    """
    stem = f"bench_model_neuron_{MODEL_SIZE}_b{mb}"
    return dict(
        stem=stem,
        neff=f"{GEN}/{stem}.neff",
        config=f"{GEN}/{stem}_config.json",
        metadata=f"{GEN}/{stem}_metadata.json",
        workdir=f"{GEN}/{stem}.work",
        s3_neff=f"{MODELS_S3}/{stem}.neff",
        s3_config=f"{MODELS_S3}/{stem}_config.json",
        s3_metadata=f"{MODELS_S3}/{stem}_metadata.json",
    )


# ---------------------------------------------------------------- the cells

def cell_id(rows, tps, sessions, probe=False):
    """The cell id, and the dimension CloudWatch sees.

    `trn1fwp-`: the `HusharBench` namespace already carries ids from five Trainium
    campaigns, and `trn1l20-` -- the 10M/20M round on this same feature wire at this same
    log rate and session count -- would share everything but the model size. `fwp` is the
    feature wire plus the precision axis.

    THE PRECISION IS IN THE ID, and that is load-bearing twice over. `run_cell` skips a
    cell whose record S3 already holds, so two arms sharing an id would make the whole
    second arm a silent no-op; and the id IS the CloudWatch `ModelId`, so a shared one
    would average two precisions into one time series.

    The mini batch is not in the id because it is not a variable: it is the row count,
    which is already there.
    """
    return (f"trn1fwp-{MODEL_SIZE}-{PREC_TAG}-r{rows}t{tps}-s{sessions}"
            + ("-probe" if probe else ""))


def cell(rows, tps, sessions, phase="main", probe=False, mb=None):
    """One measurement.

    `mb` is DERIVED from the row count, so a request is one NEFF call of exactly the rows
    it carries: `pieces` is 1, `sessions_per_request` is 1, and the engine shape is
    `[rows, 690]`. That makes `request_lanes` equal to `sessions`, which is also what
    Triton's instance count means, and the pair comparable. Only the prime probe passes
    `mb`, and only because a probe is a shape test rather than a measured rate.

    THE LOG FIELDS ARE DERIVED HERE so every one of them lands in the cell record and the
    verifier can re-derive the banner assertion from the file rather than from this module.
    """
    mb = MINI_BATCH_FOR_ROWS[rows] if mb is None else mb
    pieces = rows // mb
    if not probe and pieces != 1:
        sys.exit(f"cell {rows}x{tps} came out as {pieces} pieces; this run scores every "
                 f"request in one call and has no split")
    replicas_per_core = sessions // NEURON_CORES
    # What `Sampler::new` will actually do with the rate: one request in n, nearest unit
    # fraction. Carried per cell because it is what the banner prints and what the report
    # must quote -- a rate and the fraction it became are not the same statement.
    rate = float(SAMPLE_RATE)
    every = 0 if rate <= 0 else (1 if rate >= 1 else round(1.0 / rate))
    batch = log_batch_for(rows)
    return dict(
        id=cell_id(rows, tps, sessions, probe),
        phase=phase,
        stack="hushar", wire="features",
        # WHICH DEVICE SCORED IT. This run measures two accelerators into ONE S3 cell
        # prefix, so the field is what separates them: each collector enriches only its
        # own, and the combined report groups on it.
        accelerator="trainium1", accelerator_label="AWS Trainium1",
        size=MODEL_SIZE, target=MODEL_TARGET,
        rows=rows, tps=tps, rows_per_sec=rows * tps,
        # ---- the inference log
        sample_rate=rate,
        sample_every=every,
        sample_pct=(100.0 / every if every else 0.0),
        log_batch=batch,
        log_sends=LOG_SENDS,
        log_rows_per_object=batch * rows,
        # ~53 MB at either packaging, from the measured feature-wire payload: 5,757 B
        # fixed + 1,031 B per row, plus 5 f32 scores per row.
        log_object_bytes=batch * (5757 + 1031 * rows + 20 * rows),
        log_bytes_per_sec=int(tps * rate * (5757 + 1031 * rows + 20 * rows)),
        log_store="Amazon S3",
        # NOT AN AXIS. Recorded because it is a configured value the banner is checked
        # against, and because `mini_batch == rows` is how "no mini batch" is spelled on a
        # fixed-shape NEFF. Every cell of this run has pieces == 1.
        mini_batch=mb,
        pieces=pieces,
        split=False,
        # THE AXIS.
        sessions=sessions,
        # One replica per request, so `sessions` is the requests in flight.
        sessions_per_request=1,
        request_lanes=sessions,
        replicas=sessions,                 # the name the shared report helpers use
        replicas_per_core=replicas_per_core,
        cores_per_replica=CORES_PER_REPLICA,
        contexts=CONTEXTS_PER_REPLICA,
        neuron_cores=NEURON_CORES,
        parallelism=sessions,
        parallelism_kind="sessions",
        wt=WORKER_THREADS,
        warmup=WARMUP, duration=DURATION, segments=SEGMENTS, settle=SETTLE,
        # THE AXIS OF THIS ROUND. Carried per cell, with the flags that produced it, so
        # the report's precision column is read off the record rather than off the lane's
        # environment -- and so `verify.py` can refuse a cell whose NEFF metadata
        # disagrees with the label its id carries.
        precision=PRECISION,
        auto_cast=PRECISIONS[PRECISION]["auto_cast"],
        auto_cast_type=PRECISIONS[PRECISION]["auto_cast_type"],
        precision_note=PRECISIONS[PRECISION]["note"],
        neff=artifact(mb)["neff"],
        neff_shape=[mb, WIDTH],
        split_note=(
            "no mini batch: a request is one execution of its own row count, so the "
            "replica it leases runs the whole request in one NEFF call. A NeuronCore "
            "executes one NEFF at a time and this host has two, so device capacity is "
            "two concurrent executions at every session count; replicas above one per "
            "core buy queue lanes and host-side overlap, not device throughput."
        ),
    )


def main_cells():
    """The grid: every load x every session count. Six cells per size.

    ORDERED SO A TRUNCATED LANE IS STILL READABLE. Session count ascending within a load,
    so the fewest-sessions baseline is measured first and a lane that dies part-way
    leaves comparisons with something to divide by.
    """
    for rows, tps in LOADS:
        for sessions in sorted(SESSIONS):
            yield cell(rows, tps, sessions)


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


def s3_get_json(key):
    rc, out, _ = sh(f"aws s3 cp s3://{BUCKET}/{key} - 2>/dev/null", timeout=180)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


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


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {redact(msg)}",
          flush=True)


# A BUCKET NAME EMBEDS AN AWS ACCOUNT ID, AND THIS RUN PUTS ONE IN THE SERVER'S BANNER.
# Every earlier Trainium campaign logged inference rows to a local path, so the only place
# the bucket appeared was a command line. This one logs to S3, so the banner carries
# `inference logs -> s3://hushar-bench-<account-id>-us-west-2/...` -- and the banner is
# stored in every cell record, and the lane log is committed beside the evidence. So the
# id is masked at the two boundaries where text becomes a committed artifact: `log` and
# `record`. Masked rather than dropped, because the destination IS a measurement -- the
# report has to be able to say the rows went to S3.
ACCOUNT_RE = re.compile(r"\d{12}")


def redact(value):
    """Masks the bucket name and any 12-digit account id, recursing through records.

    THE BUCKET LITERAL IS REPLACED FIRST, and that ordering is the whole point. Masking
    only the twelve-digit run is not enough once a string has been CUT: an S3 error line
    truncated to its last 300 characters can land mid-account-id, leaving an eight-digit
    tail that no `\d{12}` pattern can see and that `check_privacy.py` -- which looks for
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
    """The last `n` characters of `text`, REDACTED FIRST so a cut cannot split an id."""
    return (redact(text or "").strip() or "")[-n:] or None


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# run_client.sh writes one row per rate through summarise.py, tagged with TAG, which is
# pinned to `cell` below so this pattern does not depend on the model name.
SUMMARY_RE = re.compile(
    r"^cell\s+(\d+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)",
    re.M)

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


def slice_held(slice_, offered_tps):
    if not slice_:
        return False
    return (slice_["shed"] == 0 and slice_["failed"] == 0
            and slice_["achieved"] is not None
            and slice_["achieved"] >= 0.98 * offered_tps)


def verdict(res, slices, offered_tps):
    """Whether the rate was sustained, and if not, how far it got.

    The verdict is over the SLICES, and a cell that held some of them is its own state --
    `partial` -- rather than being rounded to either answer. The whole-window figures
    still have to agree, because a slice counts only the requests that started in it and
    a request that began in the last slice and finished after the window is counted
    nowhere.
    """
    held = [slice_held(s, offered_tps) for s in slices]
    whole_ok = bool(res) and res["shed"] == 0 and res["failed"] == 0 \
        and res["achieved"] >= 0.98 * res["offered"]
    if slices and all(held) and whole_ok:
        state = "sustained"
    elif any(held):
        state = "partial"
    else:
        state = "failed"
    return dict(state=state, sustained=state == "sustained",
                slices_held=sum(held), slices_total=len(slices),
                broke_in_slice=next((i + 1 for i, h in enumerate(held) if not h), None),
                whole_window_held=whole_ok)


def cpu_percent(report):
    m = re.search(r"^cpu\s+([\d.]+)", report or "", re.M)
    return float(m.group(1)) if m else None


def net_counters(ip):
    """Total rx/tx bytes across the server's real interfaces, from /proc/net/dev.

    MEASURED BECAUSE THIS WIRE COULD BE THE BOTTLENECK. 690 floats per row is 2.54x the
    bytes of the named-feature wire, so 100,000 rows/s is 276 MB/s inbound against a
    `trn1.2xlarge` baseline of 3.125 Gbps. If a cell is shedding, the question "device or
    network?" has to be answerable from the record rather than argued about afterwards,
    and a counter delta over the measured window answers it in one number.

    `lo` is excluded: the client is a different host, so loopback is noise.
    """
    rc, out, _ = ssh(
        ip,
        "awk 'NR>2 { gsub(/:/, \"\", $1); if ($1 != \"lo\") { rx += $2; tx += $10 } } "
        "END { print rx, tx }' /proc/net/dev",
        timeout=120)
    if rc != 0:
        return None
    parts = out.split()
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return dict(rx=int(parts[0]), tx=int(parts[1]))


def net_window(before, after, seconds):
    """Bytes per second over the measured window, and what fraction of the baseline.

    Returns None when either counter could not be read, so a missing figure is absent
    rather than silently zero.
    """
    if not before or not after or seconds <= 0:
        return None
    rx = (after["rx"] - before["rx"]) / seconds
    tx = (after["tx"] - before["tx"]) / seconds
    # Counters are monotonic unless they wrapped, which at 64 bits they will not have.
    if rx < 0 or tx < 0:
        return None
    gbps = (rx + tx) * 8 / 1e9
    return dict(
        rx_bytes_per_sec=round(rx, 1),
        tx_bytes_per_sec=round(tx, 1),
        rx_mib_per_sec=round(rx / 1048576.0, 1),
        total_gbps=round(gbps, 3),
        baseline_gbps=NET_BASELINE_GBPS,
        pct_of_baseline=round(gbps / NET_BASELINE_GBPS * 100, 1),
        seconds=round(seconds, 1),
        note="rx+tx over every non-loopback interface, from /proc/net/dev deltas across "
             "the measured window only. The baseline is the instance type's documented "
             "sustained figure from DescribeInstanceTypes, not its burst ceiling.",
    )


# ---------------------------------------------------------------- clean slate

def clean(ip, cid):
    """Leaves both hosts with nothing of this benchmark running, and PROVES it.

    The Neuron-specific counter is the last one: a process that died without releasing
    its NeuronCores leaves the next cell sharing a device it believes it owns, and unlike
    a busy CPU that does not show up as contention -- it shows up as a NEFF that will not
    load, several minutes into a cell. Every pattern is written `[t]arget/...` so the
    remote shell carrying the pattern does not match itself.
    """
    sh("pkill -f '[t]arget/release/benchmark-client' >/dev/null 2>&1; true")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    sh(f"rm -rf /tmp/hushar-client/{cid}")

    ssh(ip, f"cd {DIR} && export HOME=/root && . ./env.sh && "
            f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true; "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            # Any orphaned Neuron sampler from an abandoned cell, found by its pid file
            # rather than by name, so a cell id this function does not know is still
            # swept. neuron-monitor attaches to the device, so a survivor would be
            # counted by the device check below and block the next cell for ever.
            f"for n in /tmp/hushar-server/n-*.log.pid; do [ -f \"$n\" ] && "
            f"OUT=${{n%.pid}} /root/neuron-driver/sample_neuron.sh stop "
            f">/dev/null 2>&1; done; "
            f"pkill -f '[n]euron-monitor' >/dev/null 2>&1 || true; "
            f"pkill -f '[t]arget/release/hushar' >/dev/null 2>&1 || true; "
            f"sleep 3; pkill -9 -f '[t]arget/release/hushar' >/dev/null 2>&1 || true; "
            f"sleep 2")

    for _ in range(20):
        rc, out, _ = ssh(
            ip,
            "pgrep -cf '[t]arget/release/hushar' || true; "
            f"(ss -ltn 2>/dev/null || netstat -ltn) | grep -cE ':{PORT} ' || true; "
            # Processes holding a Neuron device, counted out of /proc rather than with
            # lsof: lsof is not on the DLAMI and a missing tool would read as zero.
            "c=0; for p in /proc/[0-9]*; do "
            "ls -l $p/fd 2>/dev/null | grep -q /dev/neuron && c=$((c+1)); "
            "done; echo $c")
        nums = [n for n in out.split() if n.isdigit()]
        if len(nums) >= 3 and all(n == "0" for n in nums[:3]):
            return True
        time.sleep(3)
    log(f"{cid} WARNING: server host still shows a hushar process, a bound port or a "
        f"process holding a NeuronCore after cleaning")
    return False


# ---------------------------------------------------------------- starting a server

def hushar_env(cell_):
    art = artifact(cell_["mini_batch"])
    return " ".join([
        # THE WIRE. `config` is the configuration `compile_neuron.py` WROTE beside the
        # NEFF, and it carries the 210 named features -- so the client sends names and
        # values and the server transforms and concatenates them into the NEFF's one
        # input. The earlier Trainium runs derived a one-feature `tensor_config` from this
        # same file to bypass that work; this run does not, which is the point of it.
        f"MODEL=vectorized MODEL_CONFIG={art['config']}",
        "PROVIDER=neuron",
        f"NEFF_PATH={art['neff']}",
        # NO MINI BATCH, SPELLED AS A SIZE. `MINI_BATCH_FIXED=1` pads the last piece up to
        # the size, which pads nothing here because the size IS the row count -- so the
        # NEFF is presented exactly the one shape it was compiled for, in one call.
        f"MINI_BATCH_SIZE={cell_['mini_batch']} MINI_BATCH_FIXED=1",
        # For Neuron, SESSIONS is model REPLICAS rather than ORT sessions: each loads one
        # complete NEFF copy per core it occupies.
        f"SESSIONS={cell_['sessions']}",
        # 1 here, so a request leases ONE replica and `sessions` is the requests in flight.
        f"SESSIONS_PER_REQUEST={cell_['sessions_per_request']}",
        f"NEURON_CORES_PER_REPLICA={cell_['cores_per_replica']}",
        f"NEURON_REPLICAS_PER_CORE={cell_['replicas_per_core']}",
        "NEURON_START_CORE=0",
        # Pinned at 1 by instruction: these are tabular models with no long sequence, so
        # there is nothing for a second concurrent tensor set to overlap.
        f"NEURON_EXECUTION_CONTEXTS={cell_['contexts']}",
        f"WORKER_THREADS={cell_['wt']}",
        f"CONN_LIMIT={CONN}",
        "CW_NAMESPACE=HusharBench",
        # The cell id IS the dimension: it is unique across the run and carries the
        # campaign prefix, so no two cells and no two campaigns can become one series.
        f"MODEL_ID={cell_['id']}",
        f"METRICS_BATCH={METRICS_BATCH}",
        # THE LOG PATH THIS RUN MEASURES. Asserted back out of the banner -- see
        # `start_hushar`. The URI carries the cell id so one cell's objects can be counted.
        f"SAMPLE_RATE={SAMPLE_RATE}",
        f"LOG_URI=s3://{BUCKET}/{LOG_PREFIX}/{cell_['id']}",
        f"LOG_BATCH={cell_['log_batch']}",
        f"LOG_SENDS={LOG_SENDS}",
        f"LABEL={cell_['id']}",
    ])


def start_hushar(cell_, ip, timeout=2400):
    """Starts hushar and returns its banner, or None having recorded why not.

    EVERY AXIS IS READ BACK OUT OF THE SERVER'S OWN BANNER rather than trusted from the
    environment this function just built. A cell that silently ran unsplit, or on one
    context, or on the wrong replica count, would still serve and still produce a
    latency -- and that latency would be a lie about a configuration. These checks are
    the one defence against the failure mode this run is most able to publish and have
    believed, because its expected answer is "the split bought little".
    """
    cid = cell_["id"]
    rc, out, err = ssh(
        ip,
        f"cd {DIR} && export HOME=/root && . ./env.sh && {hushar_env(cell_)} "
        f"scripts/benchmark/run_server.sh start 2>&1 | tail -60",
        timeout=timeout,
    )
    if rc != 0 or "listening on" not in out:
        log(f"{cid} SERVER FAILED rc={rc}\n{out[-1500:]}\n{err[-500:]}")
        return None, {"failed": "server did not start", "server_out": out[-3000:]}

    # THE FULL LOG, NOT JUST run_server.sh's FILTERED ECHO. That filter greps `features `
    # with a trailing space, which matched the old wire's `210 features transformed and
    # concatenated` and does NOT match this run's `[features] transformed and
    # concatenated` -- one feature is named in brackets rather than counted. The wire is
    # the central claim of this run, so the assertion reads the server's own log instead
    # of a grep that was written for a different wire.
    _, raw, _ = ssh(ip, f"head -120 /tmp/hushar-server/{cid}.server.log 2>/dev/null || true",
                    timeout=300)
    hay = out + "\n" + (raw or "")
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]
    wire_lines = [l.strip() for l in (raw or "").splitlines()
                  if "transformed and concatenated" in l]
    banner += wire_lines

    # The engine. A model that fell back to ONNX Runtime would still serve and would not
    # be measuring Trainium at all.
    m = re.search(r"loaded on neuron/libnrt (\S+)", hay)
    if not m:
        log(f"{cid} ABORT: the banner does not name the Neuron runtime")
        return None, {"failed": "banner does not name neuron/libnrt", "banner": banner}
    libnrt = m.group(1)

    checks = []
    # NO SPLIT, FROM THE LINE THE SERVICE PRINTS ABOUT ITS OWN SPLITTING. The plan is
    # the only place the server says whether it cut the request up, so the whole claim of
    # this run rests on this check. Asserted UNCONDITIONALLY -- there is no split branch,
    # so a cell that somehow acquired pieces cannot pass by taking the other path.
    mb = cell_["mini_batch"]
    checks.append((
        "plan     : mini batches, shared session",
        "banner does not report the shared-session plan, so the request may have been "
        "cut into pieces on distinct replicas"))
    checks.append((
        f"mini batches of {mb}, scored at the same time on the leased session",
        f"banner does not report mini batches of {mb}, which is the row count and so "
        f"one call per request"))
    # THE AXIS: sessions, cores per replica and contexts, all from the one
    # authoritative line:
    #   hushar: Neuron -> model <id>, N replica(s) on NeuronCore 0-1, K core(s) and
    #           C execution context(s) per replica
    checks.append((
        f"{cell_['sessions']} replica(s) on NeuronCore ",
        f"banner does not report {cell_['sessions']} replica(s)"))
    checks.append((
        f"{cell_['cores_per_replica']} core(s) and {cell_['contexts']} "
        f"execution context(s) per replica",
        f"banner does not report {cell_['cores_per_replica']} core(s) and "
        f"{cell_['contexts']} context(s) per replica"))
    checks.append((
        f"{cell_['contexts']} can be in flight at once",
        f"banner does not report {cell_['contexts']} execution(s) in flight"))
    # THE GROUP WIDTH IS CHECKED FOR ITS ABSENCE, below. A request leases ONE replica
    # here, and the service prints the pool's group only when it is above 1, so a
    # `in groups of` line would mean a request held several replicas and the cell was
    # serving fewer requests in flight than `sessions` claims.
    # THE WIRE. 210 named features transformed and concatenated into the model's one
    # 690-float input: this is the whole of the server-side vectorization contract, and a
    # banner showing `[features] transformed and concatenated` -- one feature named in
    # brackets rather than 210 counted -- would mean a cell was serving the earlier
    # campaigns' pre-vectorized wire under this run's name.
    checks.append((
        f"FP32[{WIDTH}] <- {N_FEATURES} features transformed and concatenated",
        f"banner does not show the {N_FEATURES}-named-feature wire; the server may be "
        f"serving a pre-vectorized tensor instead of vectorizing on the host"))
    # THE LOG CONFIGURATION, READ BACK OUT OF THE BANNER. Half the requests are written to
    # S3 and every one of these could be wrong while the server still served happily:
    # `Sampler::new` rounds a rate to the nearest one-in-n, an s3:// URI that failed to
    # parse would write somewhere cheaper, and the wrong `batch_size` at 500 rows is a
    # 2.6 GB object -- which is the number the cell's latency is most sensitive to.
    checks.append((
        f"logging 1 request in {cell_['sample_every']} "
        f"({cell_['sample_pct']:.1f}%)",
        f"banner does not report logging 1 request in {cell_['sample_every']}"))
    checks.append((
        "inference logs -> s3://",
        "banner does not report an S3 inference log destination"))
    checks.append((
        f"every {cell_['log_batch']} batches",
        f"banner does not report an object every {cell_['log_batch']} batches"))
    if cell_["replicas_per_core"] > 1:
        checks.append((
            f"{cell_['replicas_per_core']} replicas per core",
            f"banner does not report {cell_['replicas_per_core']} replicas per core"))

    for needle, why in checks:
        if needle not in hay:
            log(f"{cid} ABORT: {why}")
            return None, {"failed": why, "banner": banner}

    # NO CANDIDATE MODEL, by instruction: every request scores the same model. The server
    # prints an `A/B` line when a candidate is resident, so its presence is the refusal.
    if "A/B" in hay:
        log(f"{cid} ABORT: the banner reports an A/B split; this run serves one model")
        return None, {"failed": "a candidate model is resident", "banner": banner}
    # ONE REPLICA PER REQUEST, PROVED BY A LINE THE SERVER PRINTS rather than by the
    # absence of one. This gate used to look for `in groups of` and refuse if it appeared
    # -- but that string no longer exists anywhere in the service, so the check could
    # never fire and proved nothing. A GPU-lane gate in this same harness went stale the
    # same way and aborted a whole round; an absence check is exactly the kind that rots
    # silently, because a gate that cannot fail looks identical to one that passes.
    #
    # Both of these are positive. `on the one replica it leases` comes from the ctx line
    # and `leases an idle replica` from the pool line, and together they say a request
    # holds ONE replica for as long as it scores -- which is what makes `sessions` the
    # requests in flight, the quantity Triton's instance count sets and the reason a trn1
    # row is readable against a g5 one.
    for needle, why in (
        ("on the one replica it leases",
         "banner does not state that a request leases ONE replica, so `sessions` may not "
         "be the requests in flight"),
        (f"pool   : {cell_['sessions']} replica(s) per model, one blocking thread each",
         f"banner does not report {cell_['sessions']} replica(s) per model with one "
         f"blocking thread each"),
        ("leases an idle replica while it scores",
         "banner does not state that a request leases an idle replica while it scores"),
    ):
        if needle not in hay:
            log(f"{cid} ABORT: {why}")
            return None, {"failed": why, "banner": banner}

    # Exactly one server, so the cell is not sharing the device with a survivor.
    _, procs, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true")
    n_procs = procs.strip().split()[0] if procs.strip() else "?"
    if n_procs != "1":
        log(f"{cid} ABORT: {n_procs} hushar process(es), expected 1")
        return None, {"failed": f"{n_procs} hushar processes running", "banner": banner}

    return banner, {
        "libnrt": libnrt,
        "provider": "neuron",
        "mini_batch": mb,
        "pieces": cell_["pieces"],
        "contexts_confirmed": cell_["contexts"],
        "sessions_confirmed": cell_["sessions"],
        "group_width_confirmed": cell_["sessions_per_request"],
        # THE FEATURE WIRE, not the tensor one. The gate above requires
        # `FP32[690] <- 210 features transformed and concatenated`, which is the server
        # vectorizing 210 named features on the host; the earlier campaigns' one-feature
        # pre-vectorized configuration is what `tensor` would mean, and it is refused.
        "wire_confirmed": "features",
    }


# ---------------------------------------------------------------- running one cell

def client_command(cell_, ip, label):
    """The one client invocation: warm-up, then one continuous window in slices."""
    art = artifact(cell_["mini_batch"])
    return (
        f"cd {DIR} && export HOME=/root && "
        f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={art['config']} "
        f"TAG=cell ROWS={cell_['rows']} TPS={cell_['tps']} "
        f"WARMUP={cell_['warmup']} DURATION={cell_['duration']} "
        f"SEGMENTS={cell_['segments']} LABEL={label} "
        f"scripts/benchmark/run_client.sh > /tmp/{label}.client.out 2>&1"
    )


def run_cell(cell_, host, category="cells"):
    ip = SERVERS[host]["ip"]
    cid = cell_["id"]
    key = f"{PREFIX}/{category}/{cid}.json"
    if s3_has(key):
        log(f"{cid} already measured, skipping")
        return s3_get_json(key)

    log(f"{cid} start  mb={cell_['mini_batch']}/{cell_['pieces']}p "
        f"sessions={cell_['sessions']} ({cell_['replicas_per_core']}/core) "
        f"groups_of={cell_['sessions_per_request']} lanes={cell_['request_lanes']} "
        f"ctx={cell_['contexts']} "
        f"{cell_['rows']}rows x {cell_['tps']}tps = {cell_['rows_per_sec']} rows/s "
        f"{cell_['warmup']}+{cell_['duration']}s")

    # 1 · a clean server and a clean client, verified
    clean_ok = clean(ip, cid)
    banner, extra = start_hushar(cell_, ip)
    if banner is None:
        record(cell_, host, extra, ok=False, category=category)
        clean(ip, cid)
        return None

    # 2 · one client invocation, started on a minute boundary so the measured window is a
    #     whole number of whole CloudWatch buckets
    label = cid
    started = time.time()
    if ALIGN:
        wait = 60 - (time.time() % 60)
        time.sleep(wait if wait > 3 else wait + 60)
    ssh(ip, "true")   # keep the control channel warm while the client runs
    client_start = time.time()
    proc = subprocess.Popen(client_command(cell_, ip, label), shell=True)

    # 3 · sample both hosts, and the NeuronCores, over the MEASURED window only
    time.sleep(cell_["warmup"])
    win_start = time.time()
    csample = f"/tmp/hushar-clientsample/{cid}.log"
    sh(f"mkdir -p /tmp/hushar-clientsample && cd {DIR} && "
       f"RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; true")
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/s-{cid}.log "
            f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; "
            f"OUT=/tmp/hushar-server/n-{cid}.log "
            f"/root/neuron-driver/sample_neuron.sh start >/dev/null 2>&1")
    net_before = net_counters(ip)

    try:
        client_rc = proc.wait(timeout=cell_["warmup"] + cell_["duration"] + 1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        client_rc = 124
    win_end = time.time()
    net_after = net_counters(ip)
    net = net_window(net_before, net_after, win_end - win_start)

    _, neuron_report, _ = ssh(
        ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/s-{cid}.log "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            f"OUT=/tmp/hushar-server/n-{cid}.log "
            f"/root/neuron-driver/sample_neuron.sh stop 2>&1 || true")
    _, cpu, _ = ssh(ip, f"cd {DIR} && export HOME=/root && "
                        f"OUT=/tmp/hushar-server/s-{cid}.log "
                        f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    _, ccpu, _ = sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
                    f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")

    # 4 · the client's own numbers, whole window and per slice
    summary = ""
    try:
        summary = open(f"/tmp/hushar-client/{label}/summary.txt").read()
    except OSError:
        pass
    res = parse_summary(summary)
    try:
        client_log = open(
            f"/tmp/hushar-client/{label}/logs/{cell_['tps']}.client.log").read()
    except OSError:
        client_log = ""
    slices = parse_slices(client_log)

    if client_rc != 0 or not res:
        tail = summary[-2000:]
        try:
            tail = open(f"/tmp/{label}.client.out").read()[-2000:]
        except OSError:
            pass
        log(f"{cid} CLIENT FAILED rc={client_rc} summary={len(summary)}B\n{tail[-900:]}")
        record(cell_, host, {"failed": f"client rc={client_rc}, no parsable summary",
                             "client_out": tail, "banner": banner},
               ok=False, category=category)
        clean(ip, cid)
        return None
    if len(slices) != cell_["segments"]:
        # Not fatal -- the whole-window result is still a measurement -- but the point of
        # slicing is the hold check, so a cell without them says so.
        log(f"{cid} WARNING: {len(slices)} slice line(s), expected {cell_['segments']}")

    # 5 · collect this cell's raw output NOW, in the session that measured it
    #
    # WHAT THE LOG PATH ACTUALLY DID, taken before the host is cleaned. Neither number is
    # inferable from the configuration: a cell that shed, or whose writes failed, did not
    # pay the full cost of logging, so a latency read as "with logging on" would be
    # understated. Same capture as the GPU lane, so the two are comparable on this axis.
    _, shed_out, _ = ssh(ip, f"grep -h 'dropped to keep the newest' "
                             f"/tmp/hushar-server/{cid}.server.log 2>/dev/null "
                             f"| tail -1 || true")
    m_shed = re.search(r"\((\d+) dropped since start\)", shed_out or "")
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
            f"s-{cid}.log n-{cid}.log; do "
            f"aws s3 cp $f s3://{BUCKET}/{PREFIX}/raw/server/{host}/ "
            f"--only-show-errors 2>/dev/null; done")
    s3_put_tree(f"/tmp/hushar-client/{label}", f"{PREFIX}/raw/client/{host}/{label}")
    s3_put(csample, f"{PREFIX}/raw/client/{host}/{label}/host-samples.log")

    clean(ip, cid)

    v = verdict(res, slices, cell_["tps"])
    doc = {
        "banner": banner,
        "summary": summary,
        "result": res,
        "slices": slices,
        "verdict": v,
        "sustained": v["sustained"],
        "cpu_report": cpu,
        "client_cpu_report": ccpu,
        "server_cpu_pct": cpu_percent(cpu),
        "client_cpu_pct": cpu_percent(ccpu),
        "neuron_sampler_report": (neuron_report or "").strip()[-400:],
        "neuron_sample_log": f"raw/server/{host}/n-{cid}.log",
        "network": net,
        "clean_slate_verified": clean_ok,
        "cloudwatch_namespace": "HusharBench",
        "cloudwatch_model_id": cid,
        "window": {"client_start": iso(client_start), "start": iso(win_start),
                   "end": iso(win_end),
                   "warmup_seconds": cell_["warmup"],
                   "duration_seconds": cell_["duration"],
                   "minute_aligned": ALIGN,
                   "note": f"one continuous {cell_['duration']}s load reported as "
                           f"{cell_['segments']} slice(s); host, NeuronCore and "
                           f"CloudWatch all cover {int(win_end - win_start)}s from the "
                           f"end of warm-up"},
        "wall": {"cell_start": iso(started), "cell_end": iso(win_end)},
    }
    doc.update(extra)
    doc.update(log_facts)
    record(cell_, host, doc, category=category)
    log(f"{cid} done   p50={res['p50']} p95={res['p95']} p99={res['p99']} "
        f"achieved={res['achieved']}/{res['offered']} shed={res['shed']} "
        f"srvcpu={cpu_percent(cpu)} {v['state']} "
        f"({v['slices_held']}/{v['slices_total']} slice(s) held) "
        f"-- settling {cell_['settle']}s")
    time.sleep(cell_["settle"])
    return {**cell_, **doc}


def record(cell_, host, extra, ok=True, category="cells"):
    """A measured cell goes to cells/, which is what "already done" means.

    A failed attempt goes to failures/ under its own timestamp, so it is kept for
    diagnosis but a resume retries the cell rather than treating the failure as a result.
    """
    doc = dict(cell_)
    doc.update(extra)
    # `host` is the PLAN name, so every rig calls its first server `trn1-1` and two
    # different machines record the same value. The address discriminates them, which is
    # what lets a report say that a row added in a later session sits on another instance
    # instead of silently mixing it in.
    doc.update(host=host, server_ip=SERVERS[host]["ip"],
               instance=SERVERS[host]["instance"],
               arch=SERVERS[host]["arch"], cores=CORES,
               neuron_cores=NEURON_CORES,
               conn=CONN,
               # NOT `sample_rate`: that is per CELL, as a float, and overwriting it here
               # with the run's string would make the report format it as text.
               run_sample_rate=SAMPLE_RATE,
               run_log_rows_per_object=LOG_ROWS_PER_OBJECT,
               run_log_sends=LOG_SENDS,
               run_log_prefix=f"s3://<bucket>/{LOG_PREFIX}",
               wire_note=WIRE_NOTE,
               model_size=MODEL_SIZE, target=MODEL_TARGET,
               rev=REV, run=RUN)
    p = f"/tmp/{cell_['id']}.json"
    # Masked here rather than at every site that builds a string: this is the
    # one place a cell becomes a file, and the banner it carries names the S3
    # log destination, whose bucket embeds an account id.
    json.dump(redact(doc), open(p, "w"), indent=2)
    if ok:
        s3_put(p, f"{PREFIX}/{category}/{cell_['id']}.json")
    else:
        s3_put(p, f"{PREFIX}/failures/{cell_['id']}-{int(time.time())}.json")


def put_lane_log(host):
    """This orchestrator's own log, beside the cells.

    A cell carries its own timestamps, so losing this is survivable, but the log is the
    only record of what a lane SKIPPED and why. The path comes from the launcher through
    LANE_LOG rather than being rebuilt here: a second guess at the filename is a second
    chance to get it wrong.
    """
    candidates = [p for p in (os.environ.get("LANE_LOG"),
                              f"/root/orchestrator-{LANE_TAG}-{host}.log",
                              f"/root/orchestrator-{host}.log") if p]
    # THE KEY CARRIES THE LANE TAG, and `LANE_KEY` from the launcher wins when it is set.
    # This used to upload `lane-<host>.log` unconditionally while launch.sh ALSO uploaded
    # `lane-<tag>-<host>.log`, so every lane published its log TWICE under two names --
    # and `report.py`, which parses the lane logs for the prime figures, then counted
    # each primed shape twice and printed duplicate rows.
    key = os.environ.get("LANE_KEY") or (
        f"{PREFIX}/raw/driver/lane-{host}.log" if LANE_TAG == "main"
        else f"{PREFIX}/raw/driver/lane-{LANE_TAG}-{host}.log")
    for path in candidates:
        if os.path.exists(path):
            s3_put(path, key)
            return
    log(f"WARNING: no lane log found at any of {candidates}")


# ---------------------------------------------------------------- prime

def stage_config(ip, mb):
    """Carries the COMPILED configuration to the client, and checks its width.

    hushar's own wire is named features: a caller sends 210 of them and the service
    transforms and concatenates them into the model's one `[rows, 690]` input. THIS RUN
    SERVES EXACTLY THAT, so there is nothing to derive -- the configuration the server
    serves is the one `compile_neuron.py` wrote beside the NEFF, unmodified.

    What still has to happen is getting that file to the CLIENT. `benchmark-client` reads
    the same configuration to build its generators, one per named feature, so one file
    settles what the client sends AND what the server expects and they cannot disagree.
    The NEFF lives only on the server, so the config is read back off the server rather
    than fetched separately: the client then generates from the same bytes that were
    compiled against.

    The earlier Trainium campaigns derived a ONE-feature `identity` configuration here, so
    that the client sent the finished 690-float vector and the server copied it. That is
    the control this run reads against and is deliberately not done.

    THE WIDTH IS READ, NOT WRITTEN DOWN. It is the sum of the configuration's own
    transformation widths, which is the number `compile_neuron.py` built the NEFF's input
    from, so a configuration that would make the client send the wrong row is refused here
    rather than discovered as a shape error inside a cell.
    """
    art = artifact(mb)
    rc, out, err = ssh(ip, f"cat {DIR}/{art['config']}", timeout=300)
    if rc != 0 or not out.strip():
        return False, f"could not read {art['config']} from the server: {err[-300:]}"
    try:
        source = json.loads(out)
    except ValueError as error:
        return False, f"{art['config']} is not JSON: {error}"

    vc = (source.get("vectorization_config") or {})
    order = vc.get("feature_order") or []
    transforms = vc.get("feature_transformations") or {}
    if not order:
        return False, f"{art['config']} names no features to take a width from"
    if len(order) != N_FEATURES:
        return False, (f"{art['config']} names {len(order)} feature(s), but this run "
                       f"serves the {N_FEATURES}-named-feature wire; a one-feature "
                       f"configuration is the PRE-VECTORIZED control, not this run")

    def width_of(name):
        """The values one feature contributes to the concatenated row."""
        spec = transforms[name]
        kind = spec.get("type")
        if kind == "embedding_lookup":
            table = spec.get("embeddings") or {}
            if not table:
                raise KeyError(f"{name} has an empty embedding table")
            return len(next(iter(table.values())))
        if kind == "one_hot_encoding":
            return len(spec.get("categories") or spec.get("vocabulary") or [])
        default = spec.get("default_val")
        return len(default) if isinstance(default, list) else 1

    try:
        width = sum(width_of(name) for name in order)
    except (KeyError, TypeError) as error:
        return False, (f"{art['config']} has a feature whose width cannot be read: "
                       f"{error}")
    if width != WIDTH:
        return False, (f"{art['config']} concatenates to {width} values per row, but this "
                       f"run's NEFFs take {WIDTH}; the client would send the wrong row")

    # Onto the client, the same bytes the server is serving. ~96 KB of JSON, so it is
    # written from the text that was read rather than re-serialised: a round trip through
    # json.dumps would reorder nothing that matters but would make the two files differ
    # byte for byte, and "the same file" is the claim being made.
    local = f"{DIR}/{art['config']}"
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "w") as handle:
        handle.write(out)
    return True, None


def prime_shape(host, ip, mb):
    """Builds or restores ONE NEFF, then PROVES a fresh server loads and serves it.

    Neuron does not compile ONNX and a NEFF has a FIXED leading dimension, so every mini
    batch in the grid is a separate compile. Three things happen here, in this order,
    and each exists because of a way the phase can otherwise lie:

      1 RESTORE OR BUILD. The S3 copy is authoritative when present, which makes the
        phase resumable and makes a second lane free. A build retains its compiler work
        directory and records the SHA-256, the parameter count, the layer count and the
        compiled-versus-PyTorch error that `compile_neuron.py` checks before writing.
      2 CARRY THE CONFIGURATION TO THE CLIENT, from the server's own copy.
      3 PROVE IT. A fresh server loads the NEFF and serves a short trickle of exactly
        `mb` rows -- one execution at the compiled shape. Without this, a NEFF that
        loads but cannot serve would be discovered 2 minutes into a 18-minute cell, and
        a NEFF whose shape disagrees with its configuration would be discovered later
        still.
    """
    art = artifact(mb)
    clean(ip, f"prime-{MODEL_SIZE}-mb{mb}")

    have = all(s3_has(art[k]) for k in ("s3_neff", "s3_config", "s3_metadata"))
    t0 = time.time()
    if have:
        # The bucket name is NOT logged: it embeds an AWS account id, and a lane log is
        # committed beside the evidence. The prefix alone says what was restored.
        log(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: restoring from {MODELS_S3} in the bench bucket")
        rc, out, err = ssh(
            ip,
            f"set -e; cd {DIR}; mkdir -p {GEN}; "
            f"aws s3 cp s3://{BUCKET}/{art['s3_neff']} {art['neff']} --only-show-errors; "
            f"aws s3 cp s3://{BUCKET}/{art['s3_config']} {art['config']} "
            f"--only-show-errors; "
            f"aws s3 cp s3://{BUCKET}/{art['s3_metadata']} {art['metadata']} "
            f"--only-show-errors; "
            f"ls -l {art['neff']}",
            timeout=1800)
        if rc != 0:
            sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: restore failed: "
                     f"{err[-500:]} {out[-500:]}")
        source = "s3-cache"
    else:
        log(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: compiling (target {MODEL_TARGET}, "
            f"--auto-cast={PRECISIONS[PRECISION]['auto_cast']})")
        rc, out, err = ssh(
            ip,
            f"set -e; cd {DIR}; export HOME=/root; . ./env.sh; "
            # compile_neuron.py refuses a non-empty work directory, which is the right
            # behaviour and means a retry has to clear it rather than append to it.
            f"rm -rf {art['workdir']}; mkdir -p {GEN}; "
            f"\"$NEURON_VENV/bin/python\" scripts/benchmark/compile_neuron.py "
            f"--params {MODEL_TARGET} --batch-size {mb} --out-dir {GEN} "
            # THE PRECISION, PASSED EXPLICITLY ON BOTH ARMS -- not because the default
            # is wrong (it is FP32 through this code path, measured) but so the label
            # rests on the artifact rather than on a framework default. compile_neuron.py
            # proves the result from the trace error and refuses to write a NEFF whose
            # error does not match its label.
            f"--precision {PRECISION} 2>&1 | tail -40",
            # A 100M graph at 500 rows is a far larger compile than the 10M and 20M
            # models this phase was written for, and no campaign has recorded how long
            # one takes -- every prior entry was a cache restore. 4 h rather than the
            # previous 90 min, because a compile killed at the timeout costs the whole
            # phase and leaves nothing cached for the retry.
            timeout=int(os.environ.get("COMPILE_TIMEOUT", "14400")))
        if rc != 0 or "maximum compiled-versus-PyTorch error" not in out:
            sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: compile failed rc={rc}\n"
                     f"{out[-3000:]}\n{err[-500:]}")
        source = "compiled"
    build_seconds = round(time.time() - t0, 1)

    rc, meta_text, _ = ssh(ip, f"cat {DIR}/{art['metadata']}", timeout=300)
    try:
        meta = json.loads(meta_text)
    except ValueError:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: {art['metadata']} is not readable JSON")

    # The model is pinned by target, so the parameter count the generator lands on is the
    # only evidence it is the same model the baseline campaign measured.
    if EXPECT_PARAMS and meta.get("parameter_count") != EXPECT_PARAMS:
        message = (f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: the generator produced "
                   f"{meta.get('parameter_count'):,} parameters, but "
                   f"config-sweep-neuron-trn1-10M-50M measured {EXPECT_PARAMS:,} from "
                   f"this same code path. A Trainium figure can no longer be read "
                   f"against that campaign. Set ALLOW_PARAM_DRIFT=1 to measure anyway "
                   f"and say so in the report.")
        if not ALLOW_PARAM_DRIFT:
            sys.exit(message)
        log(f"WARNING {message}")
    if meta.get("batch_size") != mb:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: the metadata says batch "
                 f"{meta.get('batch_size')}; a NEFF's leading dimension is fixed, so "
                 f"this NEFF cannot serve this cell")

    # IS IT THE PRECISION THIS ARM ASKED FOR? The cache is keyed by path, so a restored
    # NEFF is only as trustworthy as the key it came from -- and a key that was ever
    # wrong, or an arm pointed at the other's prefix, would hand back a graph that loads,
    # serves, passes every banner gate and reports the wrong precision column. The
    # metadata travels WITH the artifact and was written by the compile that produced it,
    # so it is the one witness that survives the cache.
    #
    # `auto_cast` is checked rather than `precision` alone because `precision` was a
    # LITERAL in compile_neuron.py until this round: every NEFF built before it says
    # "fp32" whatever the compiler did. A restored artifact with no `auto_cast` key is
    # therefore pre-dating the fix and cannot be used to make a precision claim.
    want_cast = PRECISIONS[PRECISION]["auto_cast"]
    if "auto_cast" not in meta:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: the NEFF metadata has no "
                 f"`auto_cast` key, so it was compiled before the precision flag was "
                 f"passed explicitly and its `precision` field is a literal rather than "
                 f"a record of what the compiler did. It cannot support this round's "
                 f"precision column. Delete s3://<bucket>/{art['s3_neff']} and let this "
                 f"phase rebuild it.")
    if meta.get("auto_cast") != want_cast or meta.get("precision") != PRECISION:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: this arm asked for "
                 f"{PRECISION} (--auto-cast={want_cast}) but the NEFF reports "
                 f"precision={meta.get('precision')!r} "
                 f"auto_cast={meta.get('auto_cast')!r}. The two precision arms have "
                 f"collided on one cache key; a measurement from this NEFF would label "
                 f"the wrong arm.")

    # IS THIS EVEN THE MODEL? A NEFF carries the weights as constants, so the parameters
    # alone put a floor under its size -- and the first attempt at this phase published a
    # 338 KiB artifact that was an NKI KERNEL rather than the graph, which only hushar's
    # load caught. The size test is decisive, costs nothing, and names what went wrong.
    #
    # THE FLOOR IS PER PRECISION. A BF16 NEFF holds its cast constants at half the width,
    # so an FP32-derived floor would REFUSE a perfectly good BF16 artifact -- the gate
    # firing on the arm it was never written for, after a compile measured in hours.
    weight_mib = meta.get("parameter_count", 0) * WEIGHT_BYTES / 1048576.0
    neff_mib = round(meta.get("size_bytes", 0) / 1048576.0, 1)
    if neff_mib < 0.5 * weight_mib:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: the NEFF is {neff_mib} MiB but "
                 f"{meta.get('parameter_count'):,} {PREC_TAG} parameters are "
                 f"{weight_mib:.1f} MiB on their own. This is not the model graph -- a "
                 f"compile route that lowers a STATELESS graph produces a small NEFF "
                 f"that takes weights as inputs, which hushar cannot serve.")

    ok, why = stage_config(ip, mb)
    if not ok:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: {why}")

    # THE PROOF. A fresh server, the lowest parallelism, one context, and a trickle of
    # exactly `mb` rows so the NEFF is presented its one compiled shape.
    probe = cell(mb, 2, min(SESSIONS), phase="prime", probe=True, mb=mb)
    probe["id"] = f"trn1fwp-prime-{MODEL_SIZE}-{PREC_TAG}-b{mb}"
    probe.update(warmup=0, duration=20, segments=1, settle=0)
    banner, extra = start_hushar(probe, ip, timeout=1800)
    if banner is None:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: a fresh server could not serve the NEFF: "
                 f"{extra.get('failed')}")
    sh(f"rm -rf /tmp/hushar-client/{probe['id']}")
    t1 = time.time()
    rc, out, _ = sh(
        f"cd {DIR} && export HOME=/root && "
        f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={art['config']} "
        f"TAG=cell ROWS={mb} TPS=2 WARMUP=0 DURATION=20 SEGMENTS=1 "
        f"LABEL={probe['id']} scripts/benchmark/run_client.sh 2>&1", timeout=1200)
    load_seconds = round(time.time() - t1, 1)
    try:
        res = parse_summary(open(f"/tmp/hushar-client/{probe['id']}/summary.txt").read())
    except OSError:
        res = parse_summary(out)
    ssh(ip, f"cd {DIR} && export HOME=/root && . ./env.sh && "
            f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true")
    clean(ip, probe["id"])
    if not res or res["failed"] or res["achieved"] <= 0:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: the NEFF loaded but did not serve a "
                 f"trickle cleanly: {res}")

    # PUBLISHED ONLY NOW, AFTER A SERVER HAS ACTUALLY SERVED IT. The first version of this
    # phase published straight after the compile, and a NEFF hushar could not load was
    # cached in S3 where every later lane and every resume would restore it -- a failure
    # that outlives the host that caused it. A cache entry now means "this served".
    if source == "compiled":
        rc2, _, err2 = ssh(
            ip,
            f"set -e; cd {DIR}; "
            f"aws s3 cp {art['neff']} s3://{BUCKET}/{art['s3_neff']} --only-show-errors; "
            f"aws s3 cp {art['config']} s3://{BUCKET}/{art['s3_config']} "
            f"--only-show-errors; "
            f"aws s3 cp {art['metadata']} s3://{BUCKET}/{art['s3_metadata']} "
            f"--only-show-errors",
            timeout=1800)
        if rc2 != 0:
            # Not fatal: the NEFF is on the host and the cells can run. It only means the
            # next lane rebuilds it.
            log(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb} WARNING: could not publish: {err2[-300:]}")

    # A projection, not a gate: the real check is a server starting. 6 replicas each load
    # a complete NEFF copy, so the heaviest cell asks one device for `max(replicas) *
    # bytes`. Logged at prime so a later load failure has an explanation already in the
    # record rather than being diagnosed from scratch.
    worst = max(SESSIONS)
    log(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb}: {source} {build_seconds}s, NEFF {neff_mib} MiB, "
        f"{meta.get('parameter_count'):,} params / {meta.get('layers')} layers, "
        f"trace error {meta.get('maximum_trace_error'):.3g}, "
        f"served a trickle in {load_seconds}s "
        f"(at {worst} replicas the device is asked for ~{round(neff_mib * worst, 1)} MiB "
        f"of NEFF copies)")
    return dict(mini_batch=mb, neff=art["neff"], neff_shape=[mb, WIDTH],
                source=source, build_seconds=build_seconds,
                load_proof_seconds=load_seconds, load_proof_result=res,
                sha256=meta.get("sha256"), size_bytes=meta.get("size_bytes"),
                neff_mib=neff_mib,
                # THE PRECISION AND ITS PROOF, carried out of the artifact's own
                # metadata. `maximum_trace_error` is the evidence the cast applied: FP32
                # lands ~1e-6 on this architecture and casting the matmuls puts it three
                # orders above that, so the two arms are distinguishable from this number
                # alone without trusting any flag or label.
                precision=meta.get("precision"),
                auto_cast=meta.get("auto_cast"),
                auto_cast_type=meta.get("auto_cast_type"),
                compiler_args=meta.get("compiler_args"),
                neuronx_cc=meta.get("neuronx_cc"),
                parameter_count=meta.get("parameter_count"),
                layers=meta.get("layers"), input_width=meta.get("input_width"),
                maximum_trace_error=meta.get("maximum_trace_error"),
                torch=meta.get("torch"), torch_neuronx=meta.get("torch_neuronx"),
                projected_device_mib_at_max_replicas=round(neff_mib * worst, 1),
                metadata=meta)


def prime(host):
    """One NEFF per mini-batch shape this lane will measure, built or restored and proved.

    Failure here is fatal to the lane, and that is deliberate: a NEFF has a fixed leading
    dimension, so a missing shape is not a slow cell, it is a cell that cannot run.
    """
    ip = SERVERS[host]["ip"]
    # PER PRECISION. This record is what makes a resume free, so a key shared between the
    # two arms would have the second arm short-circuit on the first's record and never
    # compile its own NEFFs -- then fail in its first cell on a missing artifact.
    key = f"{PREFIX}/prime/neuron-{MODEL_SIZE}-{PREC_TAG}.json"
    done = s3_get_json(key) if s3_has(key) else None
    if done:
        log(f"prime {MODEL_SIZE}/{PREC_TAG} already done; carrying the configurations to the client")
        # The NEFFs and configurations live on the HOST, and a resumed lane may be running
        # on a host that was replaced since. Restoring them is cheap and a resume that
        # skipped it would fail in the first cell.
        for shape in done.get("shapes", []):
            mb = shape["mini_batch"]
            art = artifact(mb)
            ssh(ip, f"cd {DIR} && mkdir -p {GEN} && "
                    f"for f in {art['s3_neff']} {art['s3_config']} "
                    f"{art['s3_metadata']}; do "
                    f"test -r \"benchmark-data/generated/neuron/{MODEL_SIZE}/"
                    f"$(basename $f)\" || aws s3 cp s3://{BUCKET}/$f "
                    f"{GEN}/$(basename $f) --only-show-errors; done", timeout=1800)
            ok, why = stage_config(ip, mb)
            if not ok:
                sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG} mb{mb} (resume): {why}")
        return done

    log(f"prime {MODEL_SIZE}/{PREC_TAG}: {len(SHAPES)} NEFF shape(s) {SHAPES}")
    shapes = [prime_shape(host, ip, mb) for mb in SHAPES]

    params = {s["parameter_count"] for s in shapes}
    layers = {s["layers"] for s in shapes}
    if len(params) != 1 or len(layers) != 1:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG}: the shapes disagree about the model -- "
                 f"parameters {params}, layers {layers}. Every NEFF in a lane must be "
                 f"the same architecture at a different batch, or the mini-batch column "
                 f"is comparing two models.")

    # EVERY NEFF IN A LANE MUST ALSO AGREE ABOUT ITS PRECISION, for the same reason they
    # must agree about the architecture: the two shapes are read as one arm, so a lane
    # holding one FP32 and one BF16 NEFF would publish a precision column that is true of
    # neither packaging.
    casts = {s["auto_cast"] for s in shapes}
    if len(casts) != 1:
        sys.exit(f"prime {MODEL_SIZE}/{PREC_TAG}: the shapes disagree about the "
                 f"precision -- auto_cast {casts}. Both NEFFs in an arm must be the same "
                 f"cast, or the precision column describes neither.")

    doc = dict(size=MODEL_SIZE, target=MODEL_TARGET, run=RUN, rev=REV, host=host,
               precision=PRECISION,
               auto_cast=PRECISIONS[PRECISION]["auto_cast"],
               auto_cast_type=PRECISIONS[PRECISION]["auto_cast_type"],
               neuronx_cc=shapes[0].get("neuronx_cc"),
               maximum_trace_error=max(s["maximum_trace_error"] for s in shapes),
               parameter_count=shapes[0]["parameter_count"],
               layers=shapes[0]["layers"], input_width=shapes[0]["input_width"],
               shapes=shapes,
               note="One entry per NEFF SHAPE, which is the mini batch: a NEFF's leading "
                    "dimension is FIXED, so every mini-batch size in the grid is a "
                    "separate compile and a cell can only run a shape that exists. "
                    "`source` is `compiled` when this lane built it and `s3-cache` when "
                    "an earlier lane did -- a NEFF is a pure function of the seeded "
                    "architecture, the target parameter count, the batch size AND the "
                    "cast, so reusing one is reusing the same bytes, which the SHA-256 "
                    "records. `load_proof_seconds` is a 20 s trickle of exactly "
                    "`mini_batch` rows served by a FRESH server, so a NEFF that loads "
                    "but cannot serve is found here rather than inside a cell's warm-up. "
                    "`maximum_trace_error` is the compiled-versus-PyTorch error and is "
                    "the PROOF of the cast: ~1e-6 at FP32 and three orders above it once "
                    "the matmuls are BF16, so the arms are distinguishable from this "
                    "number alone.")
    p = f"/tmp/prime-{MODEL_SIZE}-{PREC_TAG}.json"
    json.dump(doc, open(p, "w"), indent=2)
    s3_put(p, key)
    log(f"prime {MODEL_SIZE}/{PREC_TAG} complete: {doc['parameter_count']:,} parameters, "
        f"{doc['layers']} layers, shapes "
        + ", ".join(f"b{s['mini_batch']} {s['neff_mib']}MiB {s['source']}"
                    for s in shapes))
    return doc


# ---------------------------------------------------------------- preflight

def preflight(host):
    """Refuses to measure a machine that is not the one the report will name.

    Every figure is attributed to an instance type, a device, a runtime version and a
    server build, so each is checked against the host itself rather than trusted from
    this file. The Neuron checks are the ones a CPU rig does not need: a trn1 whose driver
    did not load, or whose core count is not 2, produces numbers that look fine and
    describe a different machine -- and the core count in particular is load-bearing here,
    because 6 replicas over 2 cores is the whole meaning of `3 per core`.
    """
    h = SERVERS[host]
    ip, want = h["ip"], h["instance"]
    rc, out, err = ssh(ip, "set -e; "
                       "T=$(curl -sf -X PUT http://169.254.169.254/latest/api/token "
                       "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); "
                       "echo type=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/instance-type); "
                       "echo ami=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/ami-id); "
                       "echo kernel=$(uname -r); "
                       "echo nproc=$(nproc); "
                       "echo dev=$(ls /dev/neuron* 2>/dev/null | wc -l); "
                       # The core count, from neuron-ls's JSON rather than its table. The
                       # table has ONE row per DEVICE and reports the core count in a
                       # column, so counting rows reads 1 on a two-core trn1.2xlarge --
                       # which it did, and warned about a machine that was correct.
                       "echo cores=$(/opt/aws/neuron/bin/neuron-ls -j 2>/dev/null "
                       "| python3 -c 'import json,sys\n"
                       "try:\n"
                       "    d = json.load(sys.stdin)\n"
                       "except Exception:\n"
                       "    print(\"\"); raise SystemExit\n"
                       "if isinstance(d, dict):\n"
                       "    d = d.get(\"neuron_devices\") or d.get(\"devices\") or []\n"
                       "print(sum(int(x.get(\"nc_count\") or 0) for x in d) or \"\")' "
                       "2>/dev/null || true); "
                       "echo nrt=$(test -r /opt/aws/neuron/lib/libnrt.so.1 "
                       "&& echo yes || echo no); "
                       "echo monitor=$(test -x /opt/aws/neuron/bin/neuron-monitor "
                       "&& echo yes || echo no); "
                       "echo sampler=$(test -x /root/neuron-driver/sample_neuron.sh "
                       "&& echo yes || echo no); "
                       f"echo hushar=$(test -x {DIR}/target/release/hushar "
                       "&& echo yes || echo no); "
                       f"cd {DIR}; . ./env.sh; "
                       "echo provider=$PROVIDER; "
                       "echo venv=$(test -x \"$NEURON_VENV/bin/python\" "
                       "&& echo yes || echo no); "
                       "echo nrtpath=$(basename \"$NRT_DYLIB_PATH\")",
                       timeout=600)
    if rc != 0:
        sys.exit(f"preflight: cannot reach {host} at {ip}: {err[-300:]} {out[-300:]}")
    f = dict(l.split("=", 1) for l in out.split() if "=" in l)
    log(f"preflight {host}: {f}")

    checks = [
        (f.get("type") == want, f"{host} is {f.get('type')}, the run claims {want}"),
        (f.get("nproc") == str(CORES), f"{f.get('nproc')} vCPUs, expected {CORES}"),
        (f.get("dev", "0") != "0", "no /dev/neuron device; the driver is not loaded"),
        (f.get("nrt") == "yes", "libnrt.so.1 is missing"),
        (f.get("monitor") == "yes", "neuron-monitor is missing"),
        (f.get("sampler") == "yes", "the NeuronCore sampler is missing"),
        (f.get("hushar") == "yes", "no hushar binary on the server"),
        (f.get("provider") == "neuron",
         f"env.sh defaults PROVIDER={f.get('provider')}, expected neuron"),
        (f.get("venv") == "yes",
         "env.sh does not point NEURON_VENV at a usable python; prime cannot compile"),
    ]
    for ok, why in checks:
        if not ok:
            sys.exit(f"preflight: {why}")
    # The core count is parsed from a table and a format change would silently read 0, so
    # a disagreement is a warning that names both numbers rather than a fatal check on a
    # brittle parse. The banner assertions in start_hushar are the load-bearing ones.
    if f.get("cores") not in (str(NEURON_CORES), "", None):
        log(f"preflight {host} WARNING: neuron-ls suggests {f.get('cores')} NeuronCore "
            f"row(s), the grid assumes {NEURON_CORES}; every replica placement in this "
            f"lane is derived from {NEURON_CORES}")
    log(f"preflight {host}: OK -- {want}, {f.get('dev')} neuron device(s), "
        f"{f.get('nrtpath')}, kernel {f.get('kernel')}, ami {f.get('ami')}")
    return f


# ---------------------------------------------------------------- main

def drive(cells, host, category="cells"):
    done = []
    for c in cells:
        if MAX_CELLS and len(done) >= MAX_CELLS:
            log(f"MAX_CELLS={MAX_CELLS} reached")
            break
        for attempt in (1, 2):
            try:
                rec = run_cell(c, host, category)
                if rec:
                    done.append(rec)
                    break
                log(f"{c['id']} attempt {attempt} did not measure")
            except Exception as e:      # a cell must never take the lane down
                log(f"{c['id']} attempt {attempt} EXCEPTION {type(e).__name__}: {e}")
                try:
                    record(c, host, {"failed": f"{type(e).__name__}: {e}"},
                           ok=False, category=category)
                except Exception:
                    pass
            try:
                clean(SERVERS[host]["ip"], c["id"])
            except Exception:
                pass
            time.sleep(30)
        put_lane_log(host)
    return done


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: sweep.py <server-host>")
    host = sys.argv[1]
    if host not in SERVERS:
        sys.exit(f"{host} is not in RIG={sorted(SERVERS)}")

    preflight(host)
    log(f"run={RUN} rev={REV} host={host} size={MODEL_SIZE} target={MODEL_TARGET} "
        f"phase={PHASE} loads={LOADS} mini_batch=rows (no split) "
        f"sessions={SESSIONS} shapes={SHAPES} "
        f"wire=features/{N_FEATURES} vectorize=server "
        f"cores_per_replica={CORES_PER_REPLICA} "
        f"contexts={CONTEXTS_PER_REPLICA} sample_rate={SAMPLE_RATE} "
        f"log=s3://<bucket>/{LOG_PREFIX} "
        f"log_rows_per_object={LOG_ROWS_PER_OBJECT} log_sends={LOG_SENDS} "
        f"candidate=none "
        f"warmup={WARMUP}s measured={DURATION}s/{SEGMENTS}slices settle={SETTLE}s")

    if PHASE in ("all", "prime"):
        prime(host)
        if PHASE == "prime":
            return 0

    if PHASE in ("all", "main"):
        cells = list(main_cells())
        log(f"main: {len(cells)} cell(s) -- "
            + ", ".join(f"{c['rows']}x{c['tps']} mb{c['mini_batch']}/{c['pieces']}p "
                        f"s{c['sessions']}g{c['sessions_per_request']}"
                        f"/{c['request_lanes']}lane" for c in cells))
        drive(cells, host, "cells")

    put_lane_log(host)
    log("lane complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
