#!/usr/bin/env python3
"""Drives one server host through its lane of the trn1 mini-batch and replica sweep.

    HUSHAR_BENCH_BUCKET=... REV=... MODEL_SIZE=10m sweep.py <server-host>

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS. `minibatch-gpu-g5-10M-20M` asked whether cutting a request into mini
batches lowers its latency on one NVIDIA A10G and found it did not -- for a reason that
belonged to the SERVER rather than the device. hushar leases ONE ONNX Runtime session per
request, every mini batch of that request calls `run_named` on it, and the TensorRT
provider serializes concurrent `Run()` on one session. The pieces fanned out for the
feature work and then queued on the device. That report named the missing knob:
`execution_contexts_per_replica`, which exists only on the Neuron backend.

    SO: ON THE DEVICE THAT HAS PER-PIECE CONTEXTS, DOES THE SPLIT PAY?
    AND HOW MANY REPLICAS DOES ONE trn1.2xlarge WANT AT THESE LOADS?

TWO AXES, ONE CORE PER REPLICA THROUGHOUT.

    mini batch   rows per NEFF call, `is_fixed` on. `mini_batch == rows` is one call and
                 is the baseline; anything smaller cuts the request into exact pieces.
    replicas     `threading.sessions_per_model`. Two NeuronCores, one core per replica,
                 so 2 is one replica per core, 4 is two and 6 is three.

WHAT BOUNDS THE ANSWER, READ OUT OF THE SOURCE BEFORE ANY CELL RAN. `cores_per_replica`
is pinned at 1 by instruction -- these are small tabular models and nothing here needs
tensor parallelism. A NeuronCore executes ONE NEFF at a time, so with one core per replica
a split's DEVICE term cannot improve: the pieces of a request all land on the one core
that request's replica owns. Two terms can still move:

    device     WORSE with splitting. N executions instead of one, at a smaller M, so the
               arithmetic is the same and the per-call overhead is N times over.
    vectorize  BETTER with splitting. The 210 transformations are host work and the
               pieces really do run at once, on a rayon pool sized to `num_cpus` (8).
    dma        can improve, but only with contexts > 1: a second tensor set lets one
               piece's host-device copy overlap another's execution.

`config-sweep-neuron-trn1-10M-50M` is the evidence for the bound. Its 41% latency cut on
50M came from `s1 · k2 · c2` -- ONE replica spanning TWO cores -- not from the split on
its own. The pieces won there because they reached two DIFFERENT cores. This run keeps
one core per replica, so it measures whether the host-side saving plus DMA overlap can
outrun N times the per-call overhead. A negative answer is a measurement, not a wasted
lane: it closes the series the GPU run opened, on the one device whose knob the GPU run
said it lacked.

CONTEXTS TRACK THE PIECE COUNT, WHICH IS THE SERVER'S OWN RULE. hushar prints it:

    ctx    : a request of N rows is ceil(N / S) execution(s) of S rows on the one replica
             it leases; C can be in flight at once and any further one waits for a context
    note   : a replica serves one request at a time, so
             "backend.execution_contexts_per_replica" is worth exactly the executions one
             request produces -- fewer serialize the request, more stay idle

So `contexts = pieces`. For every UNSPLIT cell that is 1, so the baselines are the setting
they would have had anyway and the mini-batch column stays a one-variable subtraction. The
`control` phase then re-runs the largest split of each load at `contexts = 1` to PRICE
that choice rather than assume it -- without it the report would be asserting that
per-piece contexts matter on the strength of a docstring.

THE RATES WERE SPECIFIED, NOT PROBED, AND TWO OF FOUR ARE EXPECTED TO BE OVER THE CEILING.
Projecting from the two measured points in `config-sweep-neuron-trn1-10M-50M` -- 1.644 ms
per 100-row call at 10M, 10.99 ms at 50M, linear in parameters -- 20M lands near 4.1 ms,
so two serial cores cap near 485 req/s against the 750 asked and near 97 against the 150
asked at 500 rows. The 10M rungs sit inside their measured 1.10k TPS ceiling but not far
inside it. A cell that does not hold its rate is reported by ACHIEVED RATE and enters no
latency comparison, because the median of a rate that was not held is queue depth. Those
cells still measure the ceiling, which is the figure nobody had for 20M on this device.

THREE PHASES.

    prime    one NEFF per mini-batch size, per model. Neuron does not compile ONNX, so
             `compile_neuron.py` rebuilds the seeded architecture and keeps what
             `torch_neuronx.trace` emits, for ONE fixed batch. Each is cached in S3, so a
             re-deploy or a second lane costs nothing, and each is PROVED loadable by a
             fresh server before any cell runs. A NEFF has a fixed leading dimension, so
             this phase is not optional: a cell whose shape was never compiled cannot run.
    main     the grid. 2 min warm-up, 15 min measured as one continuous load in 15 slices,
             a fresh server per cell, 1 min idle after it stops.
    control  the largest split of each load at `contexts = 1`, so the contexts decision
             above is a measured one.

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
RUN = os.environ.get("RUN", "minibatch-neuron-trn1-10M-20M")
PREFIX = f"runs/{RUN}"

MODEL_SIZE = os.environ.get("MODEL_SIZE") or sys.exit("set MODEL_SIZE, 10m or 20m")
# The compile target, as compile_neuron.py wants it. The model is pinned by SEED and
# TARGET rather than by a file hash: there is no graph to hash before the host builds it.
MODEL_TARGET = {"10m": "10M", "20m": "20M"}.get(MODEL_SIZE) \
    or sys.exit(f"MODEL_SIZE={MODEL_SIZE!r}: one of 10m, 20m")
# Continuity with config-sweep-neuron-trn1-10M-50M, which got this from the same code
# path. A different number means the generator moved and a Trainium figure can no longer
# be read against that campaign, so prime refuses rather than quietly measuring another
# model. 20M was never built for Neuron, so it has nothing to check against.
EXPECT_PARAMS = {"10m": 9_142_789, "20m": None}[MODEL_SIZE]
ALLOW_PARAM_DRIFT = os.environ.get("ALLOW_PARAM_DRIFT", "") == "1"

GEN = f"benchmark-data/generated/neuron/{MODEL_SIZE}"
MODELS_S3 = f"models/neuron/{MODEL_SIZE}"

# FP32 THROUGHOUT, AND THAT IS A CHOICE. Trainium1's systolic array is bf16-native, so
# `neuronx-cc --auto-cast` is the obvious lever the moment a load turns out to be
# device-bound -- and 20M at 100x750 is device-bound by 45%. It is deliberately NOT taken:
# every other run in this series is FP32, so a cast arm would not be comparable with any
# of them and could not earn a place in the report. It stays an open follow-up with a
# known mechanism rather than a number mixed into an FP32 table.

WARMUP = int(os.environ.get("WARMUP", "120"))       # 2 min, discarded
DURATION = int(os.environ.get("DURATION", "900"))   # 15 min, measured, continuous
SEGMENTS = int(os.environ.get("SEGMENTS", "15"))    # reported as 15 x 1 min
SETTLE = int(os.environ.get("SETTLE", "60"))        # 1 min between cells
# Start the client on a minute boundary, so the measured window is a whole number of whole
# CloudWatch buckets and nothing from warm-up shares a bucket with the measurement.
ALIGN = os.environ.get("ALIGN", "1") == "1"

CONN = 512            # threading.connection_concurrency, and the client's in-flight cap
SAMPLE_RATE = "0.0"   # feature logging off; it is not what this run is measuring
# 4 async workers on 8 vCPUs, which is what config-sweep-neuron-trn1-10M-50M used on the
# same host. These do the vectorization; the model is on the device, so this pool is the
# feature path's budget and holding it fixed keeps the mini batch the variable.
WORKER_THREADS = int(os.environ.get("WORKER_THREADS", "4"))
# Small batches, so a metric batch that straddles the warm-up boundary carries under a
# second of warm-up into the first measured minute rather than several.
METRICS_BATCH = 100
PORT = 8279
CORES = 8                 # vCPUs on trn1.2xlarge
NEURON_CORES = int(os.environ.get("NEURON_CORES", "2"))
# PINNED BY INSTRUCTION. See the module docstring for what it costs: with one core per
# replica a split cannot improve the device term, because a NeuronCore runs one NEFF at
# a time and all of a request's pieces are on the core its replica owns.
CORES_PER_REPLICA = int(os.environ.get("CORES_PER_REPLICA", "1"))

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
_LOADS = {
    "10m": [(100, 1000), (500, 200)],     # 100,000 rows/s both
    "20m": [(100, 750), (500, 150)],      #  75,000 rows/s both
}


def _parse_loads(spec):
    """`100x200,100x750`, or `10m=100x200;20m=100x200,100x750` to differ by size."""
    if "=" in spec:
        for part in spec.split(";"):
            size, _, rungs = part.partition("=")
            if size.strip() == MODEL_SIZE:
                spec = rungs
                break
        else:
            sys.exit(f"LOADS={spec!r} names no rungs for {MODEL_SIZE}")
    return [tuple(int(x) for x in pair.split("x")) for pair in spec.split(",") if pair]


LOADS = _parse_loads(os.environ["LOADS"]) if os.environ.get("LOADS") \
    else _LOADS[MODEL_SIZE]

# ---------------------------------------------------------------- the mini batches
#
# Per ROW COUNT, because a mini batch only means anything relative to the request it cuts:
# `{100: [100, 50, 25]}` reads "a 100-row request, scored whole, then in halves, then in
# quarters". The UNSPLIT size must be present for each row count or the gain column has no
# baseline. Every entry must DIVIDE its row count exactly: a short last batch would be
# padded by `is_fixed`, which costs a second NEFF and makes the latency a blend of two
# compiled shapes rather than a measurement of one.
_MINI_BATCHES = {
    100: [100, 50, 25],
    500: [500, 250, 100],
}


def _parse_mini_batches(spec):
    """`100=100,50,25;500=500,250,100`, or a bare `100,50,25` for every row count."""
    if "=" not in spec:
        shared = [int(x) for x in spec.split(",") if x.strip()]
        return {rows: list(shared) for rows, _ in LOADS}
    out = {}
    for part in spec.split(";"):
        rows, _, sizes = part.partition("=")
        out[int(rows)] = [int(x) for x in sizes.split(",") if x.strip()]
    return out


MINI_BATCHES = (_parse_mini_batches(os.environ["MINI_BATCHES"])
                if os.environ.get("MINI_BATCHES") else _MINI_BATCHES)

for _rows, _tps in LOADS:
    _sizes = MINI_BATCHES.get(_rows)
    if not _sizes:
        sys.exit(f"MINI_BATCHES names no sizes for {_rows}-row requests")
    _bad = [m for m in _sizes if m <= 0 or _rows % m]
    if _bad:
        sys.exit(f"mini batch {_bad} does not divide {_rows} rows exactly; this run "
                 f"only measures exact splits, so each cell presents one NEFF shape")
    if _rows not in _sizes:
        sys.exit(f"MINI_BATCHES for {_rows} rows omits the unsplit size {_rows}; "
                 f"without it the gain column has no baseline")

# ---------------------------------------------------------------- the replicas
#
# The second axis. On two NeuronCores with one core per replica these are 1, 2 and 3
# replicas per core. Above one per core a core serves several requests at once but still
# EXECUTES one at a time, so this axis buys host-side overlap and queue lanes rather than
# device capacity -- `config-sweep-neuron-trn1-10M-50M` saw exactly that, where 4 replicas
# held a rate 2 could not but at 16.48 ms against 10.5.
REPLICAS = [int(x) for x in os.environ.get("REPLICAS", "2,4,6").split(",") if x.strip()]
for _r in REPLICAS:
    if _r <= 0 or _r % NEURON_CORES:
        sys.exit(f"REPLICAS={_r} is not a whole number of {NEURON_CORES} NeuronCores; "
                 f"an uneven spread would put more replicas on one core than another and "
                 f"the cell would not mean what its id says")

# Every distinct NEFF shape this lane needs, which is what `prime` iterates. A shape is
# the mini batch and nothing else, so a 500-row request at mini batch 100 and a 100-row
# request scored whole share ONE NEFF.
SHAPES = sorted({m for rows, _ in LOADS for m in MINI_BATCHES[rows]})

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
WIRE_NOTE = ("210 named features, transformed and concatenated by hushar into one "
             "[rows, 690] FP32 input; the same request path "
             "config-sweep-neuron-trn1-10M-50M measured")

PHASE = os.environ.get("PHASE", "all")
if PHASE not in ("all", "prime", "main", "control"):
    sys.exit(f"PHASE={PHASE!r}: use all, prime, main or control")


# ---------------------------------------------------------------- the artifacts

def artifact(mb):
    """Where one mini-batch size's NEFF, configuration and metadata live.

    `s3` is the CACHE. A NEFF is expensive to build and is a pure function of the seeded
    architecture, the target parameter count and the batch size, so the first lane to
    build one publishes it and every later lane, re-deploy and resume restores it. That
    is also what makes `prime` resumable, which matters because it is the long phase.
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

def cell_id(rows, tps, mb, replicas, contexts, pieces, probe=False):
    """The cell id, and the dimension CloudWatch sees.

    `-ctx1` is appended only when the contexts do NOT track the piece count, which is the
    `control` phase and nothing else. So a main cell's id stays readable, and a control
    cell cannot be mistaken for one in a table.
    """
    tail = "" if contexts == pieces else f"-ctx{contexts}"
    return (f"trn1-{MODEL_SIZE}-r{rows}t{tps}-mb{mb}-s{replicas}{tail}"
            + ("-probe" if probe else ""))


def cell(rows, tps, mb, replicas, contexts=None, phase="main", probe=False):
    """One measurement.

    `contexts` defaults to the piece count, which is the server's own rule for what a
    context is worth: a replica serves one request at a time, so the useful number is the
    executions one request produces. Passing it explicitly is how the control phase
    prices that default.
    """
    pieces = rows // mb
    contexts = pieces if contexts is None else contexts
    replicas_per_core = replicas // NEURON_CORES
    return dict(
        id=cell_id(rows, tps, mb, replicas, contexts, pieces, probe),
        phase=phase,
        stack="hushar", wire="features",
        size=MODEL_SIZE, target=MODEL_TARGET,
        rows=rows, tps=tps, rows_per_sec=rows * tps,
        # THE AXES.
        mini_batch=mb,
        pieces=pieces,
        split=pieces > 1,
        replicas=replicas,
        replicas_per_core=replicas_per_core,
        cores_per_replica=CORES_PER_REPLICA,
        contexts=contexts,
        # Recorded rather than derived, because this is the flag a verifier checks the
        # contexts decision by and the report groups the control cells on.
        contexts_track_pieces=(contexts == pieces),
        neuron_cores=NEURON_CORES,
        parallelism=replicas,              # the name the shared report helpers use
        parallelism_kind="replicas",
        wt=WORKER_THREADS,
        warmup=WARMUP, duration=DURATION, segments=SEGMENTS, settle=SETTLE,
        precision="fp32",
        neff=artifact(mb)["neff"],
        neff_shape=[mb, WIDTH],
        split_note=(
            "one core per replica, so all of a request's pieces execute on the one "
            "NeuronCore its replica owns and a NeuronCore runs one NEFF at a time; the "
            "split's device term therefore cannot improve. What it can improve is the "
            "210 host-side transformations, which fan out on the rayon pool, and -- with "
            "contexts above one -- the host-device copies, which can overlap another "
            "piece's execution."
        ),
    )


def main_cells():
    """The grid: every load x every replica count x every mini batch.

    ORDERED SO A TRUNCATED LANE IS STILL READABLE. Replica count ascending on the outside
    and mini batch DESCENDING on the inside, so each replica count completes as a whole
    mini-batch group with its unsplit baseline first. If the lane dies part-way, what
    survives is groups that can be compared rather than gains with nothing to divide by.
    """
    for rows, tps in LOADS:
        for replicas in REPLICAS:
            for mb in sorted(MINI_BATCHES[rows], reverse=True):
                yield cell(rows, tps, mb, replicas)


def control_cells():
    """The contexts control: the largest split of each load, at `contexts = 1`.

    The main grid sets `contexts = pieces` on the server's own advice. That advice is a
    comment until something measures it, so this re-runs the most-split cell of each load
    with ONE context -- the setting that forces a request's pieces through a single tensor
    set -- at the baseline replica count. The pair differs in exactly one field, so the
    subtraction prices what per-piece contexts are worth on this device.

    Deliberately LAST and deliberately small: it is two cells per lane, and if the lane
    runs out of time the main grid is already complete.
    """
    for rows, tps in LOADS:
        mb = min(MINI_BATCHES[rows])
        if rows // mb <= 1:
            continue
        yield cell(rows, tps, mb, REPLICAS[0], contexts=1, phase="control")


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
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


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
        f"MODEL=vectorized MODEL_CONFIG={art['config']}",
        "PROVIDER=neuron",
        f"NEFF_PATH={art['neff']}",
        # THE FIRST AXIS. `MINI_BATCH_FIXED=1` pads the last piece up to the size, which
        # pads nothing here because every split in the grid divides exactly -- so the
        # NEFF is presented exactly the one shape it was compiled for.
        f"MINI_BATCH_SIZE={cell_['mini_batch']} MINI_BATCH_FIXED=1",
        # THE SECOND AXIS. For Neuron, SESSIONS is model REPLICAS rather than ORT
        # sessions: each loads one complete NEFF copy per core it occupies.
        f"SESSIONS={cell_['replicas']}",
        f"NEURON_CORES_PER_REPLICA={cell_['cores_per_replica']}",
        f"NEURON_REPLICAS_PER_CORE={cell_['replicas_per_core']}",
        "NEURON_START_CORE=0",
        # Tracks the piece count by default, which is the server's own rule. The control
        # phase is the only thing that passes anything else.
        f"NEURON_EXECUTION_CONTEXTS={cell_['contexts']}",
        f"WORKER_THREADS={cell_['wt']}",
        f"CONN_LIMIT={CONN}",
        "CW_NAMESPACE=HusharBench",
        # The cell id IS the dimension: it is unique across the run and carries the
        # campaign prefix, so no two cells and no two campaigns can become one series.
        f"MODEL_ID={cell_['id']}",
        f"METRICS_BATCH={METRICS_BATCH}",
        f"SAMPLE_RATE={SAMPLE_RATE}",
        f"LOG_URI={DIR}/inference-logs",
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
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]

    # The engine. A model that fell back to ONNX Runtime would still serve and would not
    # be measuring Trainium at all.
    m = re.search(r"loaded on neuron/libnrt (\S+)", out)
    if not m:
        log(f"{cid} ABORT: the banner does not name the Neuron runtime")
        return None, {"failed": "banner does not name neuron/libnrt", "banner": banner}
    libnrt = m.group(1)

    checks = []
    # The mini batch, from the line the service prints about its own splitting.
    mb = cell_["mini_batch"]
    checks.append((
        f"mini batches of {mb}, scored together, the last is padded up to it",
        f"banner does not report mini batches of {mb}"))
    # Replicas, cores per replica and contexts, all from the one authoritative line:
    #   hushar: Neuron -> model <id>, N replica(s) on NeuronCore 0-1, K core(s) and
    #           C execution context(s) per replica
    checks.append((
        f"{cell_['replicas']} replica(s) on NeuronCore ",
        f"banner does not report {cell_['replicas']} replica(s)"))
    checks.append((
        f"{cell_['cores_per_replica']} core(s) and {cell_['contexts']} "
        f"execution context(s) per replica",
        f"banner does not report {cell_['cores_per_replica']} core(s) and "
        f"{cell_['contexts']} context(s) per replica"))
    # The contexts-in-flight line, which is the service restating the rule this run's
    # contexts choice is based on. Checked separately from the count above because it is
    # the one that ties contexts to the PIECES rather than to a configuration field.
    checks.append((
        f"{cell_['contexts']} can be in flight at once",
        f"banner does not report {cell_['contexts']} execution(s) in flight"))
    # The wire. A banner showing one feature means the deploy staged the wrong config.
    checks.append((
        f"<- {N_FEATURES} features transformed and concatenated",
        f"banner does not show the {N_FEATURES}-feature wire"))
    if cell_["replicas_per_core"] > 1:
        checks.append((
            f"{cell_['replicas_per_core']} replicas per core",
            f"banner does not report {cell_['replicas_per_core']} replicas per core"))

    for needle, why in checks:
        if needle not in out:
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
        "replicas_confirmed": cell_["replicas"],
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
        f"replicas={cell_['replicas']} ({cell_['replicas_per_core']}/core) "
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

    try:
        client_rc = proc.wait(timeout=cell_["warmup"] + cell_["duration"] + 1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        client_rc = 124
    win_end = time.time()

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
               conn=CONN, sample_rate=SAMPLE_RATE,
               wire_note=WIRE_NOTE,
               model_size=MODEL_SIZE, target=MODEL_TARGET,
               rev=REV, run=RUN)
    p = f"/tmp/{cell_['id']}.json"
    json.dump(doc, open(p, "w"), indent=2)
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
    for path in candidates:
        if os.path.exists(path):
            s3_put(path, f"{PREFIX}/raw/driver/lane-{host}.log")
            return
    log(f"WARNING: no lane log found at any of {candidates}")


# ---------------------------------------------------------------- prime

def fetch_config(ip, mb):
    """Copies one mini-batch size's model configuration onto the CLIENT.

    The client reads it to know which 210 features to send. It is taken from the SERVER
    rather than generated here, because the server's copy is the one its own model loaded
    and a second generation is a second chance for the two to disagree about a width. The
    client never needs the NEFF -- benchmark-client reads the configuration as plain JSON
    and touches only the vectorization block.
    """
    art = artifact(mb)
    rc, out, err = ssh(ip, f"cat {DIR}/{art['config']}", timeout=300)
    if rc != 0 or not out.strip():
        return False, f"could not read {art['config']} from the server: {err[-300:]}"
    try:
        doc = json.loads(out)
    except ValueError as error:
        return False, f"{art['config']} is not JSON: {error}"
    order = (doc.get("vectorization_config") or {}).get("feature_order") or []
    if len(order) != N_FEATURES:
        return False, (f"{art['config']} declares {len(order)} features, "
                       f"expected {N_FEATURES}")
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
        log(f"prime {MODEL_SIZE} mb{mb}: restoring from {MODELS_S3} in the bench bucket")
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
            sys.exit(f"prime {MODEL_SIZE} mb{mb}: restore failed: "
                     f"{err[-500:]} {out[-500:]}")
        source = "s3-cache"
    else:
        log(f"prime {MODEL_SIZE} mb{mb}: compiling (target {MODEL_TARGET})")
        rc, out, err = ssh(
            ip,
            f"set -e; cd {DIR}; export HOME=/root; . ./env.sh; "
            # compile_neuron.py refuses a non-empty work directory, which is the right
            # behaviour and means a retry has to clear it rather than append to it.
            f"rm -rf {art['workdir']}; mkdir -p {GEN}; "
            f"\"$NEURON_VENV/bin/python\" scripts/benchmark/compile_neuron.py "
            f"--params {MODEL_TARGET} --batch-size {mb} --out-dir {GEN} 2>&1 | tail -30",
            timeout=5400)
        if rc != 0 or "maximum compiled-versus-PyTorch error" not in out:
            sys.exit(f"prime {MODEL_SIZE} mb{mb}: compile failed rc={rc}\n"
                     f"{out[-2500:]}\n{err[-500:]}")
        source = "compiled"
    build_seconds = round(time.time() - t0, 1)

    rc, meta_text, _ = ssh(ip, f"cat {DIR}/{art['metadata']}", timeout=300)
    try:
        meta = json.loads(meta_text)
    except ValueError:
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: {art['metadata']} is not readable JSON")

    # The model is pinned by target, so the parameter count the generator lands on is the
    # only evidence it is the same model the baseline campaign measured.
    if EXPECT_PARAMS and meta.get("parameter_count") != EXPECT_PARAMS:
        message = (f"prime {MODEL_SIZE} mb{mb}: the generator produced "
                   f"{meta.get('parameter_count'):,} parameters, but "
                   f"config-sweep-neuron-trn1-10M-50M measured {EXPECT_PARAMS:,} from "
                   f"this same code path. A Trainium figure can no longer be read "
                   f"against that campaign. Set ALLOW_PARAM_DRIFT=1 to measure anyway "
                   f"and say so in the report.")
        if not ALLOW_PARAM_DRIFT:
            sys.exit(message)
        log(f"WARNING {message}")
    if meta.get("batch_size") != mb:
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: the metadata says batch "
                 f"{meta.get('batch_size')}; a NEFF's leading dimension is fixed, so "
                 f"this NEFF cannot serve this cell")

    # IS THIS EVEN THE MODEL? A NEFF carries the weights as constants, so FP32 parameters
    # alone put a floor under its size -- and the first attempt at this phase published a
    # 338 KiB artifact that was an NKI KERNEL rather than the graph, which only hushar's
    # load caught. The size test is decisive, costs nothing, and names what went wrong.
    weight_mib = meta.get("parameter_count", 0) * 4 / 1048576.0
    neff_mib = round(meta.get("size_bytes", 0) / 1048576.0, 1)
    if neff_mib < 0.5 * weight_mib:
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: the NEFF is {neff_mib} MiB but "
                 f"{meta.get('parameter_count'):,} FP32 parameters are "
                 f"{weight_mib:.1f} MiB on their own. This is not the model graph -- a "
                 f"compile route that lowers a STATELESS graph produces a small NEFF "
                 f"that takes weights as inputs, which hushar cannot serve.")

    ok, why = fetch_config(ip, mb)
    if not ok:
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: {why}")

    # THE PROOF. A fresh server, the lowest parallelism, one context, and a trickle of
    # exactly `mb` rows so the NEFF is presented its one compiled shape.
    probe = cell(mb, 2, mb, REPLICAS[0], contexts=1, phase="prime", probe=True)
    probe["id"] = f"trn1-prime-{MODEL_SIZE}-mb{mb}"
    probe.update(warmup=0, duration=20, segments=1, settle=0)
    banner, extra = start_hushar(probe, ip, timeout=1800)
    if banner is None:
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: a fresh server could not serve the NEFF: "
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
        sys.exit(f"prime {MODEL_SIZE} mb{mb}: the NEFF loaded but did not serve a "
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
            log(f"prime {MODEL_SIZE} mb{mb} WARNING: could not publish: {err2[-300:]}")

    # A projection, not a gate: the real check is a server starting. 6 replicas each load
    # a complete NEFF copy, so the heaviest cell asks one device for `max(replicas) *
    # bytes`. Logged at prime so a later load failure has an explanation already in the
    # record rather than being diagnosed from scratch.
    worst = max(REPLICAS)
    log(f"prime {MODEL_SIZE} mb{mb}: {source} {build_seconds}s, NEFF {neff_mib} MiB, "
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
    key = f"{PREFIX}/prime/{MODEL_SIZE}.json"
    done = s3_get_json(key) if s3_has(key) else None
    if done:
        log(f"prime {MODEL_SIZE} already done; carrying the configurations to the client")
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
            ok, why = fetch_config(ip, mb)
            if not ok:
                sys.exit(f"prime {MODEL_SIZE} mb{mb} (resume): {why}")
        return done

    log(f"prime {MODEL_SIZE}: {len(SHAPES)} NEFF shape(s) {SHAPES}")
    shapes = [prime_shape(host, ip, mb) for mb in SHAPES]

    params = {s["parameter_count"] for s in shapes}
    layers = {s["layers"] for s in shapes}
    if len(params) != 1 or len(layers) != 1:
        sys.exit(f"prime {MODEL_SIZE}: the shapes disagree about the model -- "
                 f"parameters {params}, layers {layers}. Every NEFF in a lane must be "
                 f"the same architecture at a different batch, or the mini-batch column "
                 f"is comparing two models.")

    doc = dict(size=MODEL_SIZE, target=MODEL_TARGET, run=RUN, rev=REV, host=host,
               parameter_count=shapes[0]["parameter_count"],
               layers=shapes[0]["layers"], input_width=shapes[0]["input_width"],
               shapes=shapes,
               note="One entry per NEFF SHAPE, which is the mini batch: a NEFF's leading "
                    "dimension is FIXED, so every mini-batch size in the grid is a "
                    "separate compile and a cell can only run a shape that exists. "
                    "`source` is `compiled` when this lane built it and `s3-cache` when "
                    "an earlier lane did -- a NEFF is a pure function of the seeded "
                    "architecture, the target parameter count and the batch size, so "
                    "reusing one is reusing the same bytes, which the SHA-256 records. "
                    "`load_proof_seconds` is a 20 s trickle of exactly `mini_batch` rows "
                    "served by a FRESH server, so a NEFF that loads but cannot serve is "
                    "found here rather than inside a cell's warm-up.")
    p = f"/tmp/prime-{MODEL_SIZE}.json"
    json.dump(doc, open(p, "w"), indent=2)
    s3_put(p, key)
    log(f"prime {MODEL_SIZE} complete: {doc['parameter_count']:,} parameters, "
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
        f"phase={PHASE} loads={LOADS} mini_batches={MINI_BATCHES} "
        f"replicas={REPLICAS} shapes={SHAPES} "
        f"cores_per_replica={CORES_PER_REPLICA} contexts=pieces "
        f"warmup={WARMUP}s measured={DURATION}s/{SEGMENTS}slices settle={SETTLE}s")

    if PHASE in ("all", "prime"):
        prime(host)
        if PHASE == "prime":
            return 0

    if PHASE in ("all", "main"):
        cells = list(main_cells())
        log(f"main: {len(cells)} cell(s) -- "
            + ", ".join(f"{c['rows']}x{c['tps']} mb{c['mini_batch']}/{c['pieces']}p "
                        f"s{c['replicas']}c{c['contexts']}" for c in cells))
        drive(cells, host, "cells")
        if PHASE == "main":
            put_lane_log(host)
            return 0

    if PHASE in ("all", "control"):
        cells = list(control_cells())
        log(f"control: {len(cells)} cell(s) at contexts=1 -- "
            + ", ".join(c["id"] for c in cells))
        drive(cells, host, "cells")

    put_lane_log(host)
    log("lane complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
