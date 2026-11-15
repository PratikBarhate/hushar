#!/usr/bin/env python3
"""Drives one server host through its lane of the g5 GPU run.

    HUSHAR_BENCH_BUCKET=... REV=... MODEL_SIZE=10m MODEL_MD5=... sweep.py <server-host>

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS, AND WHY IT IS A DIFFERENT QUESTION FROM THE LAST ONE. Every GPU
campaign in this series measured hushar with its mini-batch split engaged -- a scheduling
decision Triton has no counterpart for. A hushar-against-Triton column therefore priced
two servers AND that decision at once. This run removes it:

    WITH NO MINI BATCH, HOW DOES hushar COMPARE WITH TRITON AS IT SHIPS?

So every hushar cell scores the whole request in ONE engine call, and the only knob that
moves is how many model copies the server holds:

    hushar   threading.sessions_per_model = 2, 4, 8   (sessions_per_request stays 1)
    Triton   instance_group count         = 2, 4, 8

A row of the grid is then a pair of cells differing in the SERVER and nothing else: same
GPU, same TensorRT library, same engine shape, same bytes on the wire, same offered rate,
one graph call per request on both sides.

THE SPLIT IS NOT A KNOB HERE, AND CANNOT BE SET. There is no mini-batch axis in this
driver and no environment variable that reintroduces one -- `cell()` always builds the
`whole request` plan. That is deliberate: the previous campaign
(`minibatch-sessions-gpu-g5-10M-20M`) is the run that measured the split, and a driver that
could quietly configure one would make this run's claim unverifiable from its own records.

WHAT MOVES WITH THE SESSION COUNT, AND WHAT DOES NOT. With one session per request,
`sessions_per_model` IS the requests in flight -- 2, 4, 8 -- which is what Triton's
instance count also means. That is the comparison this run wants, and it differs from the
previous campaign, which leased a GROUP of sessions per request and so held requests in
flight at 2 on every rung. The two runs share cell geometry and session numbers but not
their meaning, which is why the cell ids carry a different prefix.

    10M   100 rows x 1000/s   sessions 2, 4, 8     100,000 rows/s
          500 rows x  200/s   sessions 2, 4, 8     100,000 rows/s
    20M   100 rows x  750/s   sessions 2, 4, 8      75,000 rows/s
          500 rows x  150/s   sessions 2, 4, 8      75,000 rows/s

Holding rows/s constant across the two packagings also answers, for free, whether a BIGGER
request is cheaper per candidate than a smaller one, on both servers.

    triton   Triton Inference Server, `tensorrt` backend, prebuilt `model.plan`, KIND_GPU,
             `max_batch_size: 0`, the 690-float vector in. Its ONNX Runtime version is NOT
             matched to hushar's and is not claimed to be: the `tensorrt` backend does not
             use ONNX Runtime at all. TensorRT -- the library that actually executes the
             graph -- IS the same on both sides, because hushar links the copies extracted
             from this very image and `deploy.sh` refuses on any unresolved dependency.
             The version is read off each host and recorded on every cell, so the claim is
             checkable from the records rather than resting on the deploy.

TWO PHASES.

    prime  compiles hushar's TensorRT engine into a cache ONCE PER SHAPE -- and a shape
           here is a ROW COUNT, because the whole request is the call -- then PROVES each
           is picked up by a fresh server rather than rebuilt. ONNX Runtime's provider
           compiles lazily at the first Run() with whatever shape it sees, so without this
           the first cell at each row count would spend its warm-up inside the TensorRT
           builder.
    main   the grid. 5 min warm-up discarded, 15 min measured, a fresh server every cell,
           1 min idle after it stops.

ONE GRAPH CALL PER REQUEST, ON BOTH STACKS. Each cell presents the engine exactly one
shape -- `[rows, 690]`. Triton's plan is pinned the same way with
`min = opt = max = [rows, 690]` and `max_batch_size: 0`, one plan per row count, built at
deploy. Neither stack batches across requests, so a per-request latency stays comparable.

FIFTEEN ONE-MINUTE SLICES, NOT ONE FIFTEEN-MINUTE NUMBER. A window reported as a single
figure cannot tell a rate that is sustained from a rate that is sustained for a while. Both
clients report the measured window as consecutive slices, so a configuration that falls
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
HARNESS = "/root/triton-harness"
RUN = os.environ.get("RUN", "nominibatch-sessions-gpu-g5-10M-20M")
PREFIX = f"runs/{RUN}"

# WHY THE CELL IDS CARRY A PREFIX OF THEIR OWN. The id is the CloudWatch `ModelId`
# dimension, and this grid's geometry is the previous campaign's: `10m-r100t1000` at 2
# sessions exists in both. Under a shared prefix the two campaigns would be one time
# series in the `HusharBench` namespace, and a window query would silently mix them. `nb`
# is "no mini batch", which is what distinguishes this run from that one.
CELL_PREFIX = os.environ.get("CELL_PREFIX", "g5nb")

# The engine both stacks must run. hushar's bindings target ORT_API_VERSION 28, so the
# 1.28 line has no FFI skew. Checked against the server's own banner rather than trusted.
ORT_VER = os.environ.get("ORT_VER", "1.28.3")
# 26.05 is the NEWEST Triton whose TensorRT major pairs with ONNX Runtime: Triton moved to
# TensorRT 11 at 26.06, and every ORT GPU release from 1.28.3 to 1.31.0 still links
# `libnvinfer.so.10`. See deploy.sh for the version table this was read from.
TRITON_IMAGE = os.environ.get("TRITON_IMAGE",
                              "nvcr.io/nvidia/tritonserver:26.05-py3")

MODEL_SIZE = os.environ.get("MODEL_SIZE") or sys.exit("set MODEL_SIZE, 10m or 20m")
MODEL_FILE = os.environ.get("MODEL_FILE", "bench_model.onnx")
MODEL_MD5 = os.environ.get("MODEL_MD5") or sys.exit("set MODEL_MD5")
TRITON_MODEL = "bench_vectorized"
GEN = f"benchmark-data/generated/{MODEL_SIZE}"
CACHE_ROOT = f"/root/trt-cache/{MODEL_SIZE}"


def cache_for(rows):
    """One TensorRT engine cache directory per engine SHAPE, which here is a row count.

    The cache is keyed by graph, shape and builder options, so one shared directory would
    work -- and would also make a cell's provenance rest on cache-key semantics rather
    than on something a reader can list. A directory per shape means `prime` can prove
    each engine independently and a cell can be checked against the exact path it used.
    """
    return f"{CACHE_ROOT}/r{rows}"


WARMUP = int(os.environ.get("WARMUP", "300"))       # 5 min, discarded
DURATION = int(os.environ.get("DURATION", "900"))   # 15 min, measured, continuous
SEGMENTS = int(os.environ.get("SEGMENTS", "15"))    # reported as 15 x 1 min
SETTLE = int(os.environ.get("SETTLE", "60"))        # 1 min between cells, server down
# The probe window, kept because `prime` builds its trickle cells with `probe=True`.
# There is no probe PHASE in this run: every rung states its own session count.
PROBE_WARMUP = int(os.environ.get("PROBE_WARMUP", "45"))
PROBE_DURATION = int(os.environ.get("PROBE_DURATION", "90"))
PROBE_SEGMENTS = int(os.environ.get("PROBE_SEGMENTS", "3"))
PROBE_SETTLE = int(os.environ.get("PROBE_SETTLE", "20"))
# Start the client on a minute boundary, so the measured window is a whole number of whole
# CloudWatch buckets and nothing from warm-up shares a bucket with the measurement.
ALIGN = os.environ.get("ALIGN", "1") == "1"

CONN = 512            # threading.connection_concurrency, and the client's in-flight cap
SAMPLE_RATE = "0.0"   # feature logging off, as this run was specified
# 4 async workers on 8 vCPUs, which is what every GPU campaign in this series used. The
# graph is on the device, so this pool is the request path's budget, not the engine's.
WORKER_THREADS = int(os.environ.get("WORKER_THREADS", "4"))
# Small batches, so a metric batch that straddles the warm-up boundary carries under a
# second of warm-up into the first measured minute rather than several.
METRICS_BATCH = 100
# The graph runs on the GPU, so ONNX Runtime's intra-op pool only shepherds it. One thread
# per session; the parallelism this run moves is the session count.
INTRA = 1
PORT = 8279
GRPC = 8001
METRICS_PORT = 8002
CORES = 8

MAX_CELLS = int(os.environ.get("MAX_CELLS", "0"))   # smoke testing only
# THIS RUN'S ARMS. `hushar-tensor`, not `hushar`: the run was specified with the client
# sending every feature as one vector of f32 and NO feature vectorization on the server,
# which is exactly what the tensor wire is. Both arms therefore receive the same bytes per
# row and call the graph once, so a hushar-against-Triton difference is the two SERVERS.
ARMS = [a for a in os.environ.get("ARMS", "hushar-tensor,triton").split(",") if a]
# WHICH SERVER, AND ON WHICH WIRE. The two are independent, and conflating them is what
# made the first GPU report unable to answer its own question. An arm names both:
#
#   hushar          hushar, 210 named features in       -- the real request path
#   hushar-tensor   hushar, one 690-float vector in      -- Triton's contract, our server
#   triton          Triton, one 690-float vector in      -- their contract, their server
#
# `hushar-tensor` is the control: it shares the GRAPH and the DEVICE PATH with `hushar`
# and the WIRE with `triton`, so subtracting it from `hushar` prices the feature API and
# subtracting it from `triton` prices the server. Nothing else in the cell changes.
ARM_SPEC = {
    "hushar": dict(stack="hushar", wire="features"),
    "hushar-tensor": dict(stack="hushar", wire="tensor"),
    "triton": dict(stack="triton", wire="tensor"),
}
for _a in ARMS:
    if _a not in ARM_SPEC:
        sys.exit(f"ARMS={_a!r}: one of {sorted(ARM_SPEC)}")
LANE_TAG = os.environ.get("LANE_TAG", "main")

if SEGMENTS <= 0 or DURATION % SEGMENTS:
    sys.exit(f"DURATION={DURATION} is not a whole number of SEGMENTS={SEGMENTS}; a "
             f"part slice is not a slice the verdict can use")

WIDTH = 690
# This lane's grid. Overridable so a follow-up can ask one cell without a new driver.
# TWO PACKAGINGS OF THE SAME ROW THROUGHPUT per size -- see the module docstring.
_LOADS = {
    "10m": [(100, 1000), (500, 200)],     # 100,000 rows/s both
    "20m": [(100, 750), (500, 150)],      #  75,000 rows/s both
}


def _parse_loads(spec):
    """`100x200,100x750`, or `10m=100x200;20m=100x200,100x750` to differ by size.

    The per-size form exists because the two models do not share a ceiling: a rung that
    stresses 20M is unremarkable for 10M, and spending a cell on it buys nothing.
    """
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

# ---------------------------------------------------------------- the rungs
#
# THE AXIS OF THIS RUN, AND THE ONLY ONE: the session count. Per row count, the model
# copies the server holds -- `threading.sessions_per_model` on hushar, `instance_group`
# count on Triton. Both arms run the SAME list, which is what makes the grid pairable.
#
# There is no mini-batch entry because there is no mini batch: `cell()` builds the
# `whole request` plan unconditionally, so a request of any size is one engine call of
# that many rows. `sessions_per_request` therefore stays 1 and requests in flight equals
# the session count, which is also what a Triton instance count means.
_SESSIONS = {
    100: [2, 4, 8],
    500: [2, 4, 8],
}


def _parse_sessions(spec):
    """`100=2,4,8;500=2,4,8`, or a bare `2,4,8` applied to every row count."""
    def counts(text):
        out = []
        for item in text.split(","):
            if not item.strip():
                continue
            value = int(item)
            if value <= 0:
                sys.exit(f"SESSIONS entry {item!r} is not a positive session count")
            out.append(value)
        return out
    if "=" not in spec:
        shared = counts(spec)
        return {rows: list(shared) for rows, _ in LOADS}
    out = {}
    for part in spec.split(";"):
        rows, _, rest = part.partition("=")
        out[int(rows)] = counts(rest)
    return out


SESSIONS = _parse_sessions(os.environ["SESSIONS"]) if os.environ.get("SESSIONS") \
    else _SESSIONS

# A command line copied from the previous campaign would silently measure something else,
# so the knobs that campaign had are refused by name rather than ignored.
for _gone, _why in (
    ("RUNGS", "the mini batch travelled with the session count in that run; this run "
              "has no mini batch at all. Set SESSIONS, e.g. SESSIONS='100=2,4,8'"),
    ("MINI_BATCHES", "this run configures no mini batch; set SESSIONS"),
    ("MINI_BATCH_SIZE", "this run configures no mini batch; set SESSIONS"),
    ("PIN_PARALLELISM", "the session count IS the axis here; set SESSIONS"),
):
    if os.environ.get(_gone):
        sys.exit(f"{_gone} is not a knob in this run: {_why}.")

for _rows, _tps in LOADS:
    _counts = SESSIONS.get(_rows)
    if not _counts:
        sys.exit(f"SESSIONS names no session counts for {_rows}-row requests")
    if len(set(_counts)) != len(_counts):
        sys.exit(f"SESSIONS for {_rows} rows repeats a count; a cell id would collide")

# Every distinct engine shape this lane will ask the provider to compile, which is what
# `prime` iterates. With the whole request as the call, a shape IS a row count, so both
# arms compile the same two engines and one cache directory serves every session count at
# that row count.
SHAPES = sorted({rows for rows, _ in LOADS})

# Triton's engine shape is the row count too. `deploy.sh` builds one `trtexec` plan per
# entry, and `preflight` counts them: a cell whose plan was not already built is abandoned
# rather than measured, because building it inside the cell would make the measurement
# include `trtexec`.
TRITON_ROWS = sorted({rows for rows, _ in LOADS})

# Triton runs one cell per session count hushar runs, at the same offered load. Each is
# the matched-parallelism reference line for the hushar cell carrying the same number.
TRITON_PARALLELISM = {rows: sorted(counts) for rows, counts in SESSIONS.items()}

# The heaviest load, kept for the record.
TOP = max(LOADS, key=lambda rt: rt[0] * rt[1])

PARAMS = {"10m": 9138693, "20m": 18589701}[MODEL_SIZE]
LAYERS = {"10m": 2, "20m": 5}[MODEL_SIZE]
MACS_PER_ROW = {"10m": 53288960, "20m": 128983040}[MODEL_SIZE]

# Filled from the environment by launch.sh, which reads rig-state.json. NOT written here:
# this run launches its own hosts, so an address in this file would be a guess.
SERVERS = {}
for _entry in os.environ.get("RIG", "").split(";"):
    if not _entry.strip():
        continue
    _name, _ip = _entry.split("=", 1)
    SERVERS[_name.strip()] = dict(ip=_ip.strip(), instance="g5.2xlarge", arch="x86_64",
                                  model_id=_name.strip())
if not SERVERS:
    sys.exit("set RIG, e.g. RIG='g5-1=172.31.0.1' -- launch.sh does this")

# THE WIRE THIS RUN MEASURES IS `tensor`. The client sends one 690-float vector per row and
# the service's single feature is an identity transformation over it, so there is no feature
# vectorization on the server -- `VectorizationTime` is near zero by construction rather
# than by being made cheaper. The 210-feature configuration is kept resolvable so a
# follow-up can price the feature API without a new driver, but no cell here selects it.
WIRES = {
    "features": dict(config=f"{GEN}/bench_model_config.json", n_features=210,
                     note="210 named features, transformed and concatenated by hushar"),
    "tensor": dict(config=f"{GEN}/bench_model_tensor_config.json", n_features=1,
                   note="one 690-wide f32 vector per row, identity transformation; no "
                        "feature vectorization on the server"),
}
TRITON_SPEC = f"{GEN}/bench_tensor_feature_spec.json"

PHASE = os.environ.get("PHASE", "all")
if PHASE not in ("all", "prime", "main"):
    sys.exit(f"PHASE={PHASE!r}: use all, prime or main. There is no probe phase in this "
             f"run -- every rung states its own session count; see SESSIONS above.")

# What `preflight` read off the host, attached to every cell this lane records. The
# TensorRT version is the load-bearing one: this run claims both arms execute on the same
# library, and a claim a record cannot support is not a measurement.
HOST_FACTS = {}


# ---------------------------------------------------------------- the cells

def cell_id(arm, rows, tps, par, probe=False):
    return (f"{CELL_PREFIX}-{arm}-{MODEL_SIZE}-r{rows}t{tps}-p{par}"
            + ("-probe" if probe else ""))


def cell(arm, rows, tps, par, probe=False):
    """One measurement. `par` is sessions_per_model for hushar, instances for Triton.

    THE REQUEST IS NEVER CUT. `ExecutionPlan::resolve` returns `whole request` when no
    `mini_batch_size` is configured, which is what this driver does -- so `pieces` is 1,
    `sessions_per_request` is 1, and the engine shape is `[rows, 690]` on both arms. That
    makes `request_lanes` equal to `par`, which is also what Triton's instance count
    means, and the pair comparable.

    Configuring `mini_batch_size == rows` would be the same single call arithmetically,
    but it is REFUSED on this provider: `resolve` sees a mini batch on a serializing unit
    with a group of one and cannot know the request will be exactly that size. So the
    unsplit configuration is the absence of the key, not the key set to the row count.
    """
    hushar = ARM_SPEC[arm]["stack"] == "hushar"
    return dict(
        id=cell_id(arm, rows, tps, par, probe),
        phase="probe" if probe else "main",
        arm=arm, stack=ARM_SPEC[arm]["stack"], wire=ARM_SPEC[arm]["wire"],
        size=MODEL_SIZE, parameter_count=PARAMS, layers=LAYERS,
        macs_per_row=MACS_PER_ROW,
        rows=rows, tps=tps, rows_per_sec=rows * tps,
        parallelism=par,
        parallelism_kind=("sessions" if hushar else "instances"),
        intra=INTRA if hushar else None,
        wt=WORKER_THREADS if hushar else None,
        # The engine shape, recorded for every hushar cell: it is what the cache path, the
        # prime phase and the report's column all key on. It equals the row count because
        # the whole request is the call.
        mini_batch=rows if hushar else None,
        # What is actually written into the model configuration, which is NOT the same
        # thing: nothing is. Kept explicit rather than implied, so a verifier can check
        # the configuration against the server's own plan line.
        mini_batch_configured=None,
        sessions_per_request=1 if hushar else None,
        # Requests that can score at once. With one session per request this IS the
        # session count, so it moves with the axis -- unlike the previous campaign, which
        # leased a group per request and held this at 2 on every rung.
        request_lanes=par if hushar else None,
        plan="whole request" if hushar else None,
        pieces=1,
        split=False,
        max_batch=None if hushar else 0,
        warmup=PROBE_WARMUP if probe else WARMUP,
        duration=PROBE_DURATION if probe else DURATION,
        segments=PROBE_SEGMENTS if probe else SEGMENTS,
        settle=PROBE_SETTLE if probe else SETTLE,
        precision="fp32",
        engine_cache=cache_for(rows) if hushar else None,
        engine_shape=[rows, WIDTH],
        split_note="no mini batch is configured, so a request of any size is one engine "
                   "call of that many rows; sessions_per_request stays 1 and "
                   "sessions_per_model is therefore the requests in flight, which is "
                   "what a Triton instance count also means",
    )


def main_cells(selection):
    """The grid: every load x every session count, on both arms.

    Cells are ordered so the FEWEST sessions of a row count runs first, before the
    counts it is the baseline for. If the lane dies part-way, what survives is then a
    group with a baseline rather than a set of deltas with nothing to subtract from.
    """
    for arm in ARMS:
        for rows, tps in LOADS:
            counts = (SESSIONS[rows] if ARM_SPEC[arm]["stack"] == "hushar"
                      else TRITON_PARALLELISM[rows])
            for par in sorted(counts):
                yield cell(arm, rows, tps, par)


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

# Both clients print this line, one per slice of the measured window, in the same format.
# One parser therefore reads both stacks and the comparison cannot drift between them.
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
    """The hushar client's own table row, as numbers. None when it wrote no row."""
    m = SUMMARY_RE.search(text or "")
    if not m:
        return None
    offered, achieved, p50, p95, p99, shed, failed = m.groups()
    return dict(offered=int(offered), achieved=float(achieved), p50=float(p50),
                p95=float(p95), p99=float(p99), shed=int(shed), failed=int(failed))


def parse_triton_client(text):
    """Triton's Python client prints one JSON object as its last act; that is the result.

    Found from the first `{` on the line rather than by requiring the line to begin with
    one, because the client prints it as `JSON {...}`.
    """
    for line in reversed((text or "").splitlines()):
        start = line.find("{")
        if start == -1:
            continue
        frag = line[start:].strip()
        if not frag.endswith("}"):
            continue
        try:
            d = json.loads(frag)
        except ValueError:
            continue
        if "p50" in d and "achieved" in d:
            return dict(offered=int(d["offered"]), achieved=float(d["achieved"]),
                        p50=float(d["p50"]), p95=float(d["p95"]), p99=float(d["p99"]),
                        shed=int(d["shed"]), failed=int(d["failed"]),
                        client_cpu=d.get("client_cpu"))
    return None


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


# ---------------------------------------------------------------- Triton metrics

FAMILIES = ["nv_inference_request", "nv_inference_queue", "nv_inference_compute_input",
            "nv_inference_compute_infer", "nv_inference_compute_output"]


def scrape_metrics(ip):
    rc, out, _ = ssh(ip, f"curl -sf localhost:{METRICS_PORT}/metrics || true",
                     timeout=120)
    return out if rc == 0 else ""


def metric_families(text):
    """`nv_inference_*` sums, counts and quantiles, keyed by family.

    A fresh container serves each cell, so its cumulative counters start at zero and the
    sum/count DIFFERENCE across the measured window is an exact mean for that window.
    Quantiles cannot be differenced, so the ones reported are cumulative over warm-up plus
    the window and the cell records them as such. This is why the headline comparison is
    taken from the CLIENT percentiles: those are windowed, and sliced, on both stacks.
    """
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        for fam in FAMILIES:
            if not name.startswith(fam):
                continue
            rec = out.setdefault(fam, {"quantiles": {}})
            if name.endswith("_summary_us_sum"):
                rec["sum_us"] = rec.get("sum_us", 0.0) + value
            elif name.endswith("_summary_us_count"):
                rec["count"] = rec.get("count", 0.0) + value
            elif name.endswith("_summary_us"):
                q = re.search(r'quantile="([\d.]+)"', line)
                if q:
                    rec["quantiles"].setdefault(q.group(1), []).append(value)
            elif name.endswith("_duration_us"):
                rec["duration_us"] = rec.get("duration_us", 0.0) + value
    return out


def server_side(before, after):
    """The measured window's own mean per family, plus the end-of-run quantiles."""
    b, a = metric_families(before), metric_families(after)
    out = {}
    for fam, rec in a.items():
        entry = {}
        d_sum = rec.get("sum_us", 0.0) - b.get(fam, {}).get("sum_us", 0.0)
        d_cnt = rec.get("count", 0.0) - b.get(fam, {}).get("count", 0.0)
        if d_cnt > 0:
            entry["window_mean_ms"] = round(d_sum / d_cnt / 1000.0, 3)
            entry["window_count"] = int(d_cnt)
        d_dur = rec.get("duration_us", 0.0) - b.get(fam, {}).get("duration_us", 0.0)
        if d_dur:
            # Milliseconds, not microseconds: the cumulative figure runs to twelve digits
            # in microseconds, which every AWS-account-id scanner reads as an account id.
            entry["window_duration_ms"] = round(d_dur / 1000.0, 3)
        qs = {q: round(sum(v) / len(v) / 1000.0, 3)
              for q, v in rec.get("quantiles", {}).items() if v}
        if qs:
            entry["cumulative_quantiles_ms"] = qs
        if entry:
            out[fam.replace("nv_inference_", "")] = entry
    return out


# ---------------------------------------------------------------- clean slate

def clean(ip, cid):
    """Leaves both hosts with nothing of this benchmark running.

    Both stacks every time, not just the one this cell needs: a lane alternates between
    them, and a surviving Triton container would share the GPU with the next hushar cell
    without either noticing. Every pattern is written `[t]arget/...` so the remote shell
    carrying the pattern does not match itself.
    """
    sh("pkill -f '[t]arget/release/benchmark-client' >/dev/null 2>&1; true")
    sh("pkill -f '[t]riton_client.py' >/dev/null 2>&1; true")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    sh(f"rm -rf /tmp/hushar-client/{cid} /tmp/triton-client/{cid}")

    ssh(ip, f"cd {DIR} && export HOME=/root && . ./env.sh && "
            f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true; "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            # Any orphaned GPU sampler from an abandoned cell, found by its pid file
            # rather than by name, so a cell id this function does not know is still swept.
            f"for g in /tmp/hushar-server/g-*.log.pid; do [ -f \"$g\" ] && "
            f"OUT=${{g%.pid}} /root/gpu-driver/sample_gpu.sh stop >/dev/null 2>&1; done; "
            f"pkill -f '[t]arget/release/hushar' >/dev/null 2>&1 || true; "
            f"docker rm -f triton >/dev/null 2>&1 || true; "
            f"sleep 3; pkill -9 -f '[t]arget/release/hushar' >/dev/null 2>&1 || true")

    # And prove it, rather than assume the kill worked. The fourth counter is the one a
    # CPU rig does not need: a process that died without releasing the device leaves the
    # next cell sharing a GPU it believes it owns.
    for _ in range(20):
        rc, out, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true; "
                             f"(ss -ltn 2>/dev/null || netstat -ltn) "
                             f"| grep -cE ':({PORT}|{GRPC}) ' || true; "
                             f"docker ps -q --filter name=triton | wc -l; "
                             f"nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                             f"2>/dev/null | grep -c . || true")
        nums = [n for n in out.split() if n.isdigit()]
        if len(nums) >= 4 and all(n == "0" for n in nums[:4]):
            return True
        time.sleep(3)
    log(f"{cid} WARNING: server host still shows a server process, a bound port, a "
        f"Triton container or a process holding the GPU after cleaning")
    return False


# ---------------------------------------------------------------- starting a server

def hushar_env(cell_, host):
    env = [
        f"MODEL_CONFIG={WIRES[cell_['wire']]['config']}",
        # The engine cache is what makes a cell's warm-up a warm-up rather than a
        # TensorRT build. `prime` filled one per shape and proved a fresh server picks
        # each up. The path is the cell's own, so a cell cannot silently borrow another
        # shape's engine.
        f"PROVIDER=tensorrt:0:cache={cell_['engine_cache']}",
        # sessions_per_model: the backend replicas loaded, so the model copies on the
        # device. With one session per request it is also the requests in flight, which
        # is what makes it the counterpart of Triton's instance count.
        f"SESSIONS={cell_['parallelism']}",
        # Stated rather than left to default. 1 is one session per request, which is what
        # `whole request` needs: there are no pieces to spread over a group.
        f"SESSIONS_PER_REQUEST={cell_['sessions_per_request']}",
        f"THREADS={INTRA}",
        f"WORKER_THREADS={cell_['wt']}",
        f"CONN_LIMIT={CONN}",
        "CW_NAMESPACE=HusharBench",
        # The cell id IS the dimension: it is unique across the run and already
        # carries the campaign prefix, so no two cells and no two campaigns can
        # become one time series in the HusharBench namespace.
        f"MODEL_ID={cell_['id']}",
        f"METRICS_BATCH={METRICS_BATCH}",
        f"SAMPLE_RATE={SAMPLE_RATE}",
        f"LOG_URI={DIR}/inference-logs",
        f"LABEL={cell_['id']}",
    ]
    # NO MINI_BATCH_SIZE IS EVER EXPORTED. That absence is the run: `resolve` returns
    # `whole request` when the key is missing, and setting it equal to the row count would
    # be refused on this provider anyway -- see `cell()`.
    return " ".join(env)


def start_hushar(cell_, host, ip):
    """Starts hushar and returns its banner, or None having recorded why not."""
    rc, out, err = ssh(
        ip,
        f"cd {DIR} && export HOME=/root && . ./env.sh && {hushar_env(cell_, host)} "
        f"scripts/benchmark/run_server.sh start 2>&1 | tail -60",
        timeout=2400,
    )
    if rc != 0 or "listening on" not in out:
        log(f"{cell_['id']} SERVER FAILED rc={rc}\n{out[-1500:]}\n{err[-500:]}")
        return None, {"failed": "server did not start", "server_out": out[-3000:]}
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]

    # THE check of this run. A graph that fell back to the CPU provider would still serve,
    # and would still produce a latency, and that latency would be a lie about a GPU.
    if "TensorRT(device=0" not in out:
        log(f"{cell_['id']} ABORT: the banner does not name the TensorRT provider")
        return None, {"failed": "banner does not name TensorRT(device=0",
                      "banner": banner}
    if f"trt_engine_cache_path={cell_['engine_cache']}" not in out:
        log(f"{cell_['id']} ABORT: the banner does not echo the engine cache path")
        return None, {"failed": f"banner does not echo "
                                f"trt_engine_cache_path={cell_['engine_cache']}",
                      "banner": banner}

    # THE INVARIANT OF THIS RUN, read back out of the server's own banner rather than
    # trusted from the environment this function just built. A cell that silently ran a
    # SPLIT would be comparing a hushar feature against a Triton that has none, which is
    # the one way this run could lie and be believed.
    if f"plan     : {cell_['plan']}" not in out:
        log(f"{cell_['id']} ABORT: the banner does not report the {cell_['plan']!r} plan")
        return None, {"failed": f"banner does not report plan {cell_['plan']!r}",
                      "banner": banner}
    if "mini batches of" in out:
        log(f"{cell_['id']} ABORT: the banner reports a mini batch")
        return None, {"failed": "banner reports a mini batch; this run configures none",
                      "banner": banner}

    # The session arithmetic, from the same line the server derived it on: the model
    # copies, the group width, and therefore the requests in flight. Checked because the
    # grid's whole claim is that a hushar cell and its Triton pair differ in the server.
    spr, lanes, spm = (cell_["sessions_per_request"], cell_["request_lanes"],
                       cell_["parallelism"])
    expect_threads = f"{lanes} concurrent request(s) x {spr} session(s) each"
    if expect_threads not in out:
        log(f"{cell_['id']} ABORT: the banner does not report {expect_threads!r}")
        return None, {"failed": f"banner does not report '{expect_threads}'",
                      "banner": banner}
    if f"x {spm} ONNX Runtime session(s)" not in out:
        log(f"{cell_['id']} ABORT: the banner does not report {spm} sessions")
        return None, {"failed": f"banner does not report 'x {spm} ONNX Runtime "
                                f"session(s)'", "banner": banner}

    # The engine the graph was actually scored by, read from the server's own banner.
    m = re.search(r"\(runtime ([0-9]+\.[0-9]+\.[0-9]+)\)", out)
    if not m or m.group(1) != ORT_VER:
        got = m.group(1) if m else "unreported"
        log(f"{cell_['id']} ABORT: server loaded ONNX Runtime {got}, expected {ORT_VER}")
        return None, {"failed": f"ONNX Runtime {got}, expected {ORT_VER}",
                      "banner": banner}

    # The wire, from the banner's own feature line. Every hushar cell here is the tensor
    # wire, so a banner showing 210 features means the deploy staged the wrong config.
    n = WIRES[cell_["wire"]]["n_features"]
    expect = ("<- [features] transformed and concatenated" if n == 1
              else f"<- {n} features transformed and concatenated")
    if expect not in out:
        log(f"{cell_['id']} ABORT: banner does not show the {cell_['wire']} wire")
        return None, {"failed": f"banner does not show '{expect}'", "banner": banner}

    # Exactly one engine, so the cell is not sharing the GPU with a survivor.
    _, procs, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true")
    n_procs = procs.strip().split()[0] if procs.strip() else "?"
    if n_procs != "1":
        log(f"{cell_['id']} ABORT: {n_procs} hushar process(es), expected 1")
        return None, {"failed": f"{n_procs} hushar processes running", "banner": banner}
    return banner, {"ort_version": m.group(1),
                    "provider": f"tensorrt:0:cache={cell_['engine_cache']}",
                    "mini_batch": cell_["mini_batch"], "pieces": 1,
                    "plan_confirmed": cell_["plan"],
                    "sessions_per_request": spr, "request_lanes": lanes,
                    "sessions_per_model": spm}


def start_triton(cell_, ip):
    """Regenerates the repository, starts the container, and checks what it serves.

    The plan itself was compiled at deploy. A cell that rebuilt it would be measuring
    `trtexec`, so `make_trt_repo.py` reporting a reuse is an assertion here and not a log
    line: the plan is keyed by model md5 and row count, and a cell whose plan did not
    already exist is abandoned rather than measured.
    """
    rc, out, err = ssh(
        ip,
        f"export HOME=/root && cd {HARNESS} && "
        f"BUNDLE={DIR} REPO=/root/triton/models MODEL_SIZE={MODEL_SIZE} "
        f"MODEL_MD5={MODEL_MD5} MODEL_NAME={TRITON_MODEL} "
        f"INSTANCES={cell_['parallelism']} MAX_BATCH=0 ROWS={cell_['rows']} "
        f"TRITON_IMAGE={TRITON_IMAGE} python3 make_trt_repo.py 2>&1 | tail -20 && "
        f"GPUS=all MODEL_NAME={TRITON_MODEL} EXPECT_BACKEND=tensorrt "
        f"EXPECT_MAX_BATCH=0 EXPECT_INSTANCES={cell_['parallelism']} "
        f"TRITON_IMAGE={TRITON_IMAGE} REPO=/root/triton/models "
        f"GRPC_PORT={GRPC} METRICS_PORT={METRICS_PORT} ./run_triton.sh start 2>&1 | tail -50",
        timeout=2400,
    )
    if rc != 0 or "triton up on grpc" not in out:
        log(f"{cell_['id']} TRITON FAILED rc={rc}\n{out[-1500:]}\n{err[-500:]}")
        return None, {"failed": "Triton did not start", "server_out": out[-3000:]}
    # run_triton.sh aborts by itself on the wrong backend or a disagreeing config; these
    # lines are the evidence, kept in the cell so the report can point at them.
    if "backend check: tensorrt" not in out:
        log(f"{cell_['id']} ABORT: not served by the tensorrt backend")
        return None, {"failed": "not the tensorrt backend", "server_out": out[-3000:]}
    if f"model md5 {MODEL_MD5}" not in out:
        log(f"{cell_['id']} ABORT: Triton did not confirm the model md5")
        return None, {"failed": "model md5 unconfirmed", "server_out": out[-3000:]}
    if "reusing /root/triton/plans" not in out:
        log(f"{cell_['id']} ABORT: the plan was built inside the cell, not reused")
        return None, {"failed": "plan rebuilt inside the cell rather than reused "
                                "from deploy", "server_out": out[-3000:]}
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]
    plan = re.search(r"reusing (\S+\.plan)", out)
    return banner, {"platform": "tensorrt_plan", "backend": "tensorrt",
                    "plan": plan.group(1) if plan else None,
                    "triton_image": TRITON_IMAGE,
                    "dynamic_batching": None,
                    "batching_note": f"max_batch_size 0, so one graph call per request at "
                                     f"the engine's static [{cell_['rows']}, {WIDTH}] "
                                     f"profile -- the same shape hushar's cell presents, "
                                     f"and the same number of calls per request. Triton "
                                     f"runs as it ships; nothing here is a hushar-only "
                                     f"feature."}


# ---------------------------------------------------------------- running one cell

def client_command(cell_, ip, label):
    """The one client invocation: warm-up, then one continuous window reported in slices."""
    if cell_["stack"] == "hushar":
        return (
            f"cd {DIR} && export HOME=/root && "
            f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={WIRES[cell_['wire']]['config']} "
            f"TAG=cell ROWS={cell_['rows']} TPS={cell_['tps']} "
            f"WARMUP={cell_['warmup']} DURATION={cell_['duration']} "
            f"SEGMENTS={cell_['segments']} LABEL={label} "
            f"scripts/benchmark/run_client.sh > /tmp/{label}.client.out 2>&1"
        )
    return (
        f"cd {HARNESS} && export HOME=/root && "
        f"mkdir -p /tmp/triton-client/{label} && "
        f"python3 triton_client.py --server {ip}:{GRPC} --model {TRITON_MODEL} "
        f"--feature-spec {DIR}/{TRITON_SPEC} "
        f"--rows {cell_['rows']} --tps {cell_['tps']} "
        f"--warmup-secs {cell_['warmup']} --duration-secs {cell_['duration']} "
        f"--segments {cell_['segments']} --concurrency {CONN} "
        f"--label '{label}' --out /tmp/triton-client/{label}/result.json "
        f"> /tmp/triton-client/{label}/client.log 2>&1"
    )


def run_cell(cell_, host, category="cells"):
    ip = SERVERS[host]["ip"]
    cid = cell_["id"]
    key = f"{PREFIX}/{category}/{cid}.json"
    if s3_has(key):
        log(f"{cid} already measured, skipping")
        return s3_get_json(key)

    log(f"{cid} start  {cell_['stack']}/{cell_['wire']} "
        f"{cell_['parallelism_kind']}={cell_['parallelism']} "
        f"{cell_['rows']}rows x {cell_['tps']}tps = {cell_['rows_per_sec']} rows/s "
        f"{cell_['warmup']}+{cell_['duration']}s")

    # 1 · a clean server and a clean client, verified
    clean_ok = clean(ip, cid)
    if cell_["stack"] == "hushar":
        banner, extra = start_hushar(cell_, host, ip)
    else:
        banner, extra = start_triton(cell_, ip)
    if banner is None:
        record(cell_, host, extra, ok=False, category=category)
        clean(ip, cid)
        return None

    # 2 · one client invocation, on this host, started on a minute boundary so the
    #     measured window is a whole number of whole CloudWatch buckets
    label = cid
    started = time.time()
    if ALIGN:
        wait = 60 - (time.time() % 60)
        time.sleep(wait if wait > 3 else wait + 60)
    ssh(ip, "true")   # keep the control channel warm while the client runs
    client_start = time.time()
    proc = subprocess.Popen(client_command(cell_, ip, label), shell=True)

    # 3 · sample both hosts, and the GPU, over the MEASURED window only
    time.sleep(cell_["warmup"])
    win_start = time.time()
    metrics_before = scrape_metrics(ip) if cell_["stack"] == "triton" else ""
    csample = f"/tmp/hushar-clientsample/{cid}.log"
    tag = "s" if cell_["stack"] == "hushar" else "t"
    sh(f"mkdir -p /tmp/hushar-clientsample && cd {DIR} && "
       f"RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; true")
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/{tag}-{cid}.log "
            f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; "
            f"OUT=/tmp/hushar-server/g-{cid}.log "
            f"/root/gpu-driver/sample_gpu.sh start >/dev/null 2>&1")

    try:
        client_rc = proc.wait(timeout=cell_["warmup"] + cell_["duration"] + 1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        client_rc = 124
    win_end = time.time()
    # Before the container is stopped and before the samplers, so the delta covers the
    # measured window and nothing after it.
    metrics_after = scrape_metrics(ip) if cell_["stack"] == "triton" else ""

    _, gpu_report, _ = ssh(
        ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/{tag}-{cid}.log "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            f"OUT=/tmp/hushar-server/g-{cid}.log /root/gpu-driver/sample_gpu.sh stop "
            f"2>&1 || true")
    _, cpu, _ = ssh(ip, f"cd {DIR} && export HOME=/root && "
                        f"OUT=/tmp/hushar-server/{tag}-{cid}.log "
                        f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    _, ccpu, _ = sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
                    f"scripts/benchmark/sample_host.sh report 2>&1 | tail -8")

    # 4 · the client's own numbers, whole window and per slice
    summary, res = "", None
    if cell_["stack"] == "hushar":
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
    else:
        try:
            summary = open(f"/tmp/triton-client/{label}/client.log").read()
        except OSError:
            pass
        res = parse_triton_client(summary)
        client_log = summary
    slices = parse_slices(client_log)

    if client_rc != 0 or not res:
        tail = summary[-2000:]
        if cell_["stack"] == "hushar":
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
    if cell_["stack"] == "hushar":
        ssh(ip, f"export HOME=/root; cd /tmp/hushar-server && "
                f"for f in {cid}.server.log {cid}.service.json {cid}.model.json "
                f"s-{cid}.log g-{cid}.log; do "
                f"aws s3 cp $f s3://{BUCKET}/{PREFIX}/raw/server/{host}/ "
                f"--only-show-errors 2>/dev/null; done")
        s3_put_tree(f"/tmp/hushar-client/{label}", f"{PREFIX}/raw/client/{host}/{label}")
    else:
        ssh(ip, f"export HOME=/root; "
                f"cd /root/triton/models/{TRITON_MODEL} && "
                f"aws s3 cp config.pbtxt "
                f"s3://{BUCKET}/{PREFIX}/raw/server/{host}/{cid}.config.pbtxt "
                f"--only-show-errors 2>/dev/null; "
                f"cd /tmp/hushar-server && for f in t-{cid}.log g-{cid}.log; do "
                f"aws s3 cp $f s3://{BUCKET}/{PREFIX}/raw/server/{host}/ "
                f"--only-show-errors 2>/dev/null; done; "
                f"docker logs triton 2>&1 | tail -200 > /tmp/{cid}.triton.log; "
                f"aws s3 cp /tmp/{cid}.triton.log "
                f"s3://{BUCKET}/{PREFIX}/raw/server/{host}/ --only-show-errors 2>/dev/null")
        s3_put_tree(f"/tmp/triton-client/{label}", f"{PREFIX}/raw/client/{host}/{label}")
    s3_put(csample, f"{PREFIX}/raw/client/{host}/{label}/host-samples.log")

    clean(ip, cid)

    v = verdict(res, slices, cell_["tps"])
    doc = {
        "banner": banner,
        "summary": summary if cell_["stack"] == "hushar" else summary[-4000:],
        "result": res,
        "slices": slices,
        "verdict": v,
        "sustained": v["sustained"],
        "cpu_report": cpu,
        "client_cpu_report": ccpu,
        "server_cpu_pct": cpu_percent(cpu),
        "client_cpu_pct": cpu_percent(ccpu),
        "gpu_sampler_report": (gpu_report or "").strip()[-200:],
        "gpu_sample_log": f"raw/server/{host}/g-{cid}.log",
        "clean_slate_verified": clean_ok,
        "cloudwatch_namespace": "HusharBench" if cell_["stack"] == "hushar" else None,
        "cloudwatch_model_id": cid if cell_["stack"] == "hushar" else None,
        "window": {"client_start": iso(client_start), "start": iso(win_start),
                   "end": iso(win_end),
                   "warmup_seconds": cell_["warmup"],
                   "duration_seconds": cell_["duration"],
                   "minute_aligned": ALIGN,
                   "note": f"one continuous {cell_['duration']}s load reported as "
                           f"{cell_['segments']} slice(s); host, GPU and CloudWatch all "
                           f"cover {int(win_end - win_start)}s from the end of warm-up"},
        "wall": {"cell_start": iso(started), "cell_end": iso(win_end)},
    }
    doc.update(extra)
    if cell_["stack"] == "triton":
        doc["triton_server_side"] = server_side(metrics_before, metrics_after)
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
    # `host` is the PLAN name, so every rig calls its first server `g5-1` and two
    # different machines record the same value. The address discriminates them, which is
    # what lets a report say that a row added in a later session sits on another
    # instance instead of silently mixing it in.
    doc.update(host=host, server_ip=SERVERS[host]["ip"],
               instance=SERVERS[host]["instance"],
               arch=SERVERS[host]["arch"], cores=CORES,
               conn=CONN, sample_rate=SAMPLE_RATE,
               wire_note=WIRES[cell_["wire"]]["note"],
               model_file=MODEL_FILE, model_md5=MODEL_MD5, model_size=MODEL_SIZE,
               onnxruntime=ORT_VER, triton_image=TRITON_IMAGE,
               rev=REV, run=RUN)
    # THE SHARED-LIBRARY CLAIM, ON EVERY CELL OF BOTH ARMS. This run asserts that Triton's
    # TensorRT and hushar's are the same, and `deploy.sh` is what makes it so -- but a
    # deploy-time gate leaves nothing in the record. `preflight` reads the version off the
    # host and it is stamped here, so the two arms' cells can be compared for it directly.
    for field in ("tensorrt", "cuda_major", "nvidia_driver", "gpu", "ort_so"):
        if HOST_FACTS.get(field) is not None:
            doc[f"host_{field}"] = HOST_FACTS[field]
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

# The hushar arm `prime` builds its engines with. It must be one of THIS run's arms rather
# than a fixed choice: the engine cache is keyed by the graph and the shape, and both wires
# name the same `model_path`, so a features-wire prime would PROBABLY serve a tensor-wire
# cell -- but "probably" here means the first cell of each shape silently spends its
# warm-up inside the TensorRT builder, which is the one thing this phase exists to prevent.
PRIME_ARM = next((a for a in ARMS if ARM_SPEC[a]["stack"] == "hushar"), None)


def prime_shape(host, ip, rows):
    """Builds ONE engine shape into its own cache, and proves that shape is reused.

    A one-session server is started and sent a trickle of `rows`-row requests, so the
    provider observes exactly the `[rows, 690]` shape this cache is for; the server is
    stopped, and a SECOND fresh server serves the same trickle. If the cache works the
    second pass is fast. Both timings are recorded, because the difference between them is
    the evidence rather than the claim.
    """
    cache = cache_for(rows)
    c = cell(PRIME_ARM, rows, 2, 1, probe=True)
    c.update(phase="prime", warmup=0, duration=20, segments=1, settle=0)
    clean(ip, c["id"])
    ssh(ip, f"rm -rf {cache} && mkdir -p {cache}")

    passes = []
    for label in ("first", "cached"):
        c["id"] = f"{CELL_PREFIX}-prime-{MODEL_SIZE}-r{rows}-{label}"
        banner, extra = start_hushar(c, host, ip)
        if banner is None:
            sys.exit(f"prime {MODEL_SIZE} r{rows} ({label} pass): {extra.get('failed')}")
        sh(f"rm -rf /tmp/hushar-client/{c['id']}")
        trickle = (f"cd {DIR} && export HOME=/root && "
                   f"SERVER=http://{ip}:{PORT} "
                   f"MODEL_CONFIG={WIRES[c['wire']]['config']} "
                   f"TAG=cell ROWS={rows} TPS=2 WARMUP=0 DURATION=20 SEGMENTS=1 "
                   f"LABEL={c['id']} scripts/benchmark/run_client.sh 2>&1")
        t0 = time.time()
        # An hour: a heavy graph's first TensorRT build is minutes, and the point of this
        # phase is to pay for it exactly once per shape.
        rc, out, _ = sh(trickle, timeout=3600)
        took = time.time() - t0
        # From the file run_client.sh writes, not from stdout: the summary is appended to
        # summary.txt and reading it is how run_cell does it too.
        try:
            res = parse_summary(open(f"/tmp/hushar-client/{c['id']}/summary.txt").read())
        except OSError:
            res = parse_summary(out)
        _, listing, _ = ssh(ip, f"ls -l {cache}/ 2>&1 | tail -20")
        ssh(ip, f"cd {DIR} && export HOME=/root && . ./env.sh && "
                f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true")
        passes.append(dict(pass_=label, seconds=round(took, 1), rc=rc, result=res,
                           cache_listing=listing.strip()[-1500:]))
        log(f"prime {MODEL_SIZE} r{rows} {label} pass: rc={rc} {took:.0f}s {res}")
        clean(ip, c["id"])
        time.sleep(10)

    first, cached = passes[0], passes[1]
    if not cached["result"] or cached["result"]["failed"]:
        sys.exit(f"prime {MODEL_SIZE} r{rows}: the cached engine did not serve a trickle "
                 f"cleanly: {cached['result']}")
    if "no engine" in (cached["cache_listing"] or "") or not cached["cache_listing"]:
        sys.exit(f"prime {MODEL_SIZE} r{rows}: the cache directory is empty after a build")
    return dict(rows=rows, engine_cache=cache, engine_shape=[rows, WIDTH],
                passes=passes,
                build_seconds=first["seconds"], cached_seconds=cached["seconds"],
                speedup=(round(first["seconds"] / cached["seconds"], 1)
                         if cached["seconds"] else None))


def prime(host):
    """Builds and proves one TensorRT engine PER SHAPE this lane will measure.

    ONNX Runtime's TensorRT provider compiles at the first Run() with the shape it sees,
    so without this the first cell at each row count would spend its warm-up inside the
    TensorRT builder. With the whole request as the call there are exactly as many shapes
    as row counts -- two -- and one cache serves every session count at that row count,
    because a session count does not change the engine.

    Failure here is fatal to the lane. A cell measured on a server that is still compiling
    is not a slow measurement, it is a wrong one.
    """
    ip = SERVERS[host]["ip"]
    key = f"{PREFIX}/prime/{MODEL_SIZE}.json"
    if s3_has(key):
        log(f"prime {MODEL_SIZE} already done")
        return s3_get_json(key)
    if PRIME_ARM is None:
        log(f"prime {MODEL_SIZE}: no hushar arm in ARMS={ARMS}, nothing to prime "
            f"(Triton's plans are built by deploy.sh)")
        return None

    log(f"prime {MODEL_SIZE}: {len(SHAPES)} engine shape(s) {SHAPES} "
        f"on the {ARM_SPEC[PRIME_ARM]['wire']} wire")
    shapes = [prime_shape(host, ip, rows) for rows in SHAPES]

    doc = dict(size=MODEL_SIZE, run=RUN, rev=REV, host=host,
               cache_root=CACHE_ROOT, shapes=shapes,
               prime_arm=PRIME_ARM, prime_wire=ARM_SPEC[PRIME_ARM]["wire"],
               note="One entry per engine SHAPE. With no mini batch the whole request is "
                    "the call, so a shape is a ROW COUNT and one cache directory serves "
                    "every session count at that row count -- a session count loads more "
                    "copies of the same engine, it does not change the engine. Primed on "
                    "the SAME WIRE the cells use, so no cell can pay for a TensorRT build "
                    "inside its warm-up. build_seconds is a 20 s trickle against a cold "
                    "cache, so it carries the TensorRT engine build; cached_seconds is "
                    "the same trickle against a FRESH server that loaded the cache. The "
                    "gap is the cost every cell would otherwise have paid inside its "
                    "warm-up.")
    p = f"/tmp/prime-{MODEL_SIZE}.json"
    json.dump(doc, open(p, "w"), indent=2)
    s3_put(p, key)
    for s in shapes:
        log(f"prime {MODEL_SIZE} r{s['rows']}: build {s['build_seconds']}s "
            f"-> cached {s['cached_seconds']}s")
    return doc


# ---------------------------------------------------------------- preflight

def preflight(host):
    """Refuses to measure a machine that is not the one the report will name.

    Every figure is attributed to an instance type, a GPU, an engine version, a graph and
    a server build, so each is checked against the host itself rather than trusted from
    this file. The GPU checks are the ones a CPU rig does not need: a `g5.2xlarge` with no
    visible A10G, or a TensorRT provider library with an unresolved dependency, produces
    numbers that look fine and mean nothing.

    What it reads is also KEPT, in `HOST_FACTS`, and stamped on every cell: this run's
    central claim is that both arms execute on one TensorRT, and that has to be checkable
    from the records rather than only from the deploy log.
    """
    h = SERVERS[host]
    ip, want = h["ip"], h["instance"]
    gen = f"{DIR}/{GEN}"
    rc, out, err = ssh(ip, "set -e; "
                       "T=$(curl -sf -X PUT http://169.254.169.254/latest/api/token "
                       "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); "
                       "echo type=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/instance-type); "
                       "echo ami=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/ami-id); "
                       "echo kernel=$(uname -r); "
                       "echo nproc=$(nproc); "
                       "echo gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader "
                       "| head -1 | tr ' ' '_'); "
                       "echo gpucount=$(nvidia-smi -L | grep -c GPU); "
                       "echo driver=$(nvidia-smi --query-gpu=driver_version "
                       "--format=csv,noheader | head -1); "
                       f"echo md5=$(md5sum {gen}/{MODEL_FILE} | cut -c1-32); "
                       f"echo hushar=$(test -x {DIR}/target/release/hushar "
                       "&& echo yes || echo no); "
                       f"echo plans=$(ls /root/triton/plans/{MODEL_SIZE}-*.plan "
                       "2>/dev/null | sed 's/.*-r\\([0-9]*\\)-.*/\\1/' | sort -n "
                       "| paste -sd, -); "
                       f"echo image=$(docker image inspect {TRITON_IMAGE} "
                       ">/dev/null 2>&1 && echo yes || echo no); "
                       "echo gpusampler=$(test -x /root/gpu-driver/sample_gpu.sh "
                       "&& echo yes || echo no); "
                       f"cd {DIR}; . ./env.sh; "
                       "echo ortso=$(readlink -f $ORT_DYLIB_PATH "
                       "| sed 's/.*libonnxruntime\\.so\\.//'); "
                       "echo trtso=$IMAGE_TENSORRT; "
                       "echo cudamaj=$IMAGE_CUDA_MAJOR; "
                       # The library hushar's provider will actually load, read off the
                       # path rather than off env.sh, so the recorded version is the file
                       # and not the variable that was supposed to describe it.
                       "echo trtlinked=$(readlink -f /root/nvlibs/libnvinfer.so.10 "
                       "2>/dev/null | sed 's/.*libnvinfer\\.so\\.//'); "
                       "echo trtmissing=$(ldd $(dirname $ORT_DYLIB_PATH)/"
                       "libonnxruntime_providers_tensorrt.so | grep -c 'not found' "
                       "|| true)",
                       timeout=600)
    if rc != 0:
        sys.exit(f"preflight: cannot reach {host} at {ip}: {err[-300:]} {out[-300:]}")
    f = dict(l.split("=", 1) for l in out.split() if "=" in l)
    log(f"preflight {host}: {f}")

    # Which row counts have a prebuilt plan, as a set. A deploy may have built MORE than
    # this lane needs -- staging both row counts on both servers is what makes a lane
    # interchangeable -- so the check is that every row count this lane will ASK for has
    # one, not that the counts match.
    have_plans = {int(p) for p in (f.get("plans") or "").split(",") if p.strip().isdigit()}
    need_plans = set(TRITON_ROWS) if any(
        ARM_SPEC[a]["stack"] != "hushar" for a in ARMS) else set()

    checks = [
        (f.get("type") == want, f"{host} is {f.get('type')}, the run claims {want}"),
        (f.get("nproc") == str(CORES), f"{f.get('nproc')} vCPUs, expected {CORES}"),
        (f.get("gpucount") == "1", f"{f.get('gpucount')} GPU(s) visible, expected 1"),
        ("A10G" in (f.get("gpu") or ""), f"GPU is {f.get('gpu')}, expected an A10G"),
        (f.get("md5") == MODEL_MD5, f"model md5 {f.get('md5')}, expected {MODEL_MD5}"),
        (f.get("hushar") == "yes", "no hushar binary on the server"),
        (need_plans <= have_plans,
         f"TensorRT plan(s) for {MODEL_SIZE} exist for rows {sorted(have_plans)}, but "
         f"this lane needs {sorted(need_plans)}; the missing one(s) must be prebuilt at "
         f"deploy, because building a plan inside a cell would make the measurement "
         f"include trtexec"),
        (f.get("image") == "yes", f"{TRITON_IMAGE} is not present"),
        (f.get("gpusampler") == "yes", "the GPU sampler is missing"),
        (f.get("ortso") == ORT_VER,
         f"libonnxruntime.so.{f.get('ortso')}, expected {ORT_VER}"),
        (f.get("trtmissing") == "0",
         "the TensorRT provider library has unresolved dependencies"),
        (bool(f.get("trtso")), "env.sh does not record the image's TensorRT version"),
        # The claim, checked rather than asserted: the libnvinfer hushar will dlopen is
        # the one extracted from the Triton image, so both arms run one library.
        (f.get("trtlinked") == f.get("trtso"),
         f"hushar links libnvinfer {f.get('trtlinked')} but the Triton image ships "
         f"{f.get('trtso')}; this run claims both arms execute on the same TensorRT"),
    ]
    for ok, why in checks:
        if not ok:
            sys.exit(f"preflight: {why}")
    HOST_FACTS.update(tensorrt=f.get("trtso"), cuda_major=f.get("cudamaj"),
                      nvidia_driver=f.get("driver"), gpu=(f.get("gpu") or "").replace(
                          "_", " "),
                      ort_so=f.get("ortso"))
    log(f"preflight {host}: OK -- {want}, {f.get('gpu')}, driver {f.get('driver')}, "
        f"TensorRT {f.get('trtso')} on CUDA {f.get('cudamaj')} (hushar links "
        f"{f.get('trtlinked')}), ONNX Runtime {f.get('ortso')}, "
        f"{MODEL_SIZE} md5 {f.get('md5')}")
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
    log(f"run={RUN} rev={REV} host={host} size={MODEL_SIZE} arms={ARMS} "
        f"phase={PHASE} loads={LOADS} sessions={SESSIONS} shapes={SHAPES} "
        f"triton_parallelism={TRITON_PARALLELISM} "
        f"warmup={WARMUP}s measured={DURATION}s/{SEGMENTS}slices settle={SETTLE}s "
        f"plan=whole request (no mini batch)")

    if PHASE in ("all", "prime"):
        prime(host)
        if PHASE == "prime":
            return 0

    selection = {}
    if PHASE in ("all", "main"):
        # Recorded as its own rule so a reader of selection/ can tell a run whose session
        # count was the axis from one that ranked something.
        for arm in ARMS:
            hushar = ARM_SPEC[arm]["stack"] == "hushar"
            kind = "sessions" if hushar else "instances"
            plan = {rows: (SESSIONS[rows] if hushar else TRITON_PARALLELISM[rows])
                    for rows, _ in LOADS}
            rule = ("one cell per session count; no mini batch, so sessions_per_request "
                    "is 1 and the session count is the requests in flight"
                    if hushar else
                    "one cell per instance_group count, matching hushar's session counts")
            selection[arm] = dict(
                parallelism=None, parallelism_kind=kind, per_rung=True,
                plan={str(k): v for k, v in plan.items()},
                rule=rule, probes=[])
            log(f"selection {arm}: {kind} {plan} (no probe)")
        p = f"/tmp/selection-{MODEL_SIZE}.json"
        json.dump(dict(run=RUN, size=MODEL_SIZE, host=host, rev=REV,
                       pinned=False, per_rung=True,
                       mini_batch=None,
                       sessions={str(k): v for k, v in SESSIONS.items()},
                       selection=selection), open(p, "w"), indent=2)
        s3_put(p, f"{PREFIX}/selection/{MODEL_SIZE}.json")

    if PHASE in ("all", "main"):
        cells = list(main_cells(selection))
        log(f"main: {len(cells)} cell(s) -- "
            + ", ".join(f"{c['rows']}x{c['tps']}"
                        + f" {c['parallelism_kind'][0]}{c['parallelism']}"
                        + f" [{c['arm']}]" for c in cells))
        drive(cells, host, "cells")

    put_lane_log(host)
    log("lane complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
