#!/usr/bin/env python3
"""Drives one server host through its lane of the design-against-Triton run, from its client.

    HUSHAR_BENCH_BUCKET=... REV=... MODEL_MD5=... sweep.py <server-host>

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS. The rows-and-mini-batch campaign already made the LIKE-FOR-LIKE
comparison -- identical bytes, identical graph, one graph call each -- and put hushar within
6% of Triton. That question is closed. This run asks the one a reader actually has:

    GIVEN THE SAME TRAFFIC, WHICH DESIGN SERVES IT BETTER, EACH AT ITS OWN BEST?

The traffic is fixed and real: N candidates, each described by 210 named features. What the
two designs ASK OF THE CALLER differs, and that difference is the design:

    hushar  the caller sends 210 named features. The server transforms and concatenates
            them into the 690-wide input, and splits the request into mini batches scored
            at once. VECTORIZATION AND MINI-BATCHING ARE THE DESIGN.
    Triton  the caller sends a finished 690-float tensor, because Triton's contract has no
            other form. Triton's own throughput lever is the opposite of a split: the
            dynamic batcher AGGREGATES queued requests into bigger graph calls.

So this is not the same bytes on the wire, and it is not meant to be. It is the same work,
arriving the way each design asks for it.

THREE ARMS, BECAUSE TWO CANNOT SEPARATE THE WIRE FROM THE SERVER.

    H   hushar, 210 named features, mini batch -- OUR DESIGN, end to end
    T   Triton, 690-float tensor, its best configuration -- THEIR DESIGN
    X   hushar, 690-float tensor, mini batch -- the control

Without X a reader cannot tell whether a gap between H and T is hushar's server or hushar's
contract. H - X is the cost of the feature API; X - T is the two servers on identical bytes.
Every rung carries all three, measured back to back on one host.

BEST OF BOTH, AND THE BEST IS MEASURED RATHER THAN ASSERTED. `PHASE=tune` ranks
configurations of one stack at a single load with a short window -- long enough to rank, and
a ranking does not need fifteen minutes. Triton gets its instance and thread split, its
execution mode, and its dynamic batcher at three queue delays; hushar gets its mini-batch
width, its spinning policy and its session split. The winners are then PINNED into
`PHASE=head` through BEST_HUSHAR and BEST_TRITON, so the head-to-head names the
configurations it ran and neither side is handicapped by a setting chosen for the other.

WHAT THE CALLER PAYS, AND WHY IT IS NOT IN THE LATENCY COLUMN. Triton's contract does not
remove the 210 transformations; it moves them to the caller, usually per candidate rather
than per batch and in whatever language the caller happens to be. That cost is reported
SEPARATELY and as a LOWER BOUND, from two things this run already collects: the
`VectorizationTime` hushar publishes on every H cell, which is that work done in batched
Rust and is the cheapest it can possibly be, and the serialized payload size each client
prints. Charging Triton for a strawman client would be the easier claim and the weaker one.

AND IT ASKS FOR FIFTEEN MINUTES, NOT FIVE. A five-minute window cannot tell a rate that is
sustained from a rate that is sustained for five minutes. Each head-to-head cell offers its
rate under one CONTINUOUS fifteen-minute load and both clients report it as three
consecutive five-minute slices, so a configuration that falls behind part-way through is
visible instead of having its shortfall averaged away. A cell is `sustained` only if EVERY
slice held it; one that held some is `partial`, with the slice it broke in.

SPINNING. Triton's onnxruntime backend forwards no `session.*.allow_spinning` key, so Triton
always runs ONNX Runtime's default spin. hushar's policy is therefore not assumed: it is one
of the things `PHASE=tune` decides, and the head-to-head records which policy each stack ran.

COMPARISON GROUPS DO NOT CROSS HOSTS. The 10M campaign measured a placement offset between
two instances of the same type that repeating does not reduce. Each table is placed entirely
on one host, and arm P re-measures the floor on this build.
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
RUN = os.environ.get("RUN", "design-vs-triton-cpu-arm64-10M")
PREFIX = f"runs/{RUN}"

# The engine both stacks must run. Triton 26.08 bundles 1.28.0 and hushar's bindings
# target ORT_API_VERSION 28, so the engine is the same build on both sides of every ratio.
# Checked against the server's own banner on hushar and against the image on Triton,
# rather than trusted from the deploy.
ORT_VER = os.environ.get("ORT_VER", "1.28.0")
TRITON_IMAGE = os.environ.get("TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:26.08-py3")

MODEL_SIZE = os.environ.get("MODEL_SIZE", "10m")
MODEL_FILE = os.environ.get("MODEL_FILE", "bench_model.onnx")
MODEL_MD5 = os.environ.get("MODEL_MD5") or sys.exit("set MODEL_MD5")
TRITON_MODEL = "bench_vectorized"

WARMUP = int(os.environ.get("WARMUP", "120"))       # 2 min, discarded
DURATION = int(os.environ.get("DURATION", "900"))   # 15 min, measured, continuous
SEGMENTS = int(os.environ.get("SEGMENTS", "3"))     # reported as 3 x 5 min
SETTLE = int(os.environ.get("SETTLE", "60"))        # 1 min between cells
GUARD = int(os.environ.get("GUARD", "20"))   # after warmup before sampling, so warmup's
                                             # last metric batch lands outside the window
CONN = 512            # threading.connection_concurrency, fixed for the whole run
SAMPLE_RATE = "0.0"   # feature logging off; it is not what this run is measuring
WORKER_THREADS = 32   # the 10M campaign's phase W showed this is not a lever
METRICS_BATCH = 500
PORT = 8279
GRPC = 8001
METRICS_PORT = 8002
CORES = 64
# Smoke testing only: stop after this many cells.
MAX_CELLS = int(os.environ.get("MAX_CELLS", "0"))
# Which arms this invocation runs. A probe passes a short list and short windows; the main
# sweep passes nothing and gets the lane's full order.
ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a]
# Narrows to these row counts, so a probe can ask one question. Empty means all.
ONLY_ROWS = {int(n) for n in os.environ.get("ONLY_ROWS", "").split(",") if n.strip()}
# Set to 0 to disable the ladder's early stop and measure every rung regardless.
EARLY_STOP = os.environ.get("EARLY_STOP", "1") != "0"
# Which lane this is, used only to find this orchestrator's own log.
LANE_TAG = os.environ.get("LANE_TAG", "main")

# Filled from the environment by launch.sh, which reads rig-state.json. NOT written here:
# this run launches its own hosts, so an address in this file would be a guess.
SERVERS = {}
for _entry in os.environ.get("RIG", "").split(";"):
    if not _entry.strip():
        continue
    _name, _ip = _entry.split("=", 1)
    SERVERS[_name.strip()] = dict(ip=_ip.strip(), instance="c9g.16xlarge", arch="arm64",
                                  model_id=f"ceil-{_name.strip()}")
if not SERVERS:
    sys.exit("set RIG, e.g. RIG='c9g-1=172.31.0.1;c9g-2=172.31.0.2' -- launch.sh does this")

# The wire every cell runs. `features` is kept resolvable so a follow-up probe can price
# it again without a new driver, but no cell in this run selects it.
WIRES = {
    "tensor": dict(config=f"benchmark-data/generated/{MODEL_SIZE}/bench_model_tensor_config.json",
                   n_features=1,
                   note="one 690-wide vector, identity transformation; the form Triton's "
                        "contract requires, so both stacks receive the same bytes"),
    "features": dict(config=f"benchmark-data/generated/{MODEL_SIZE}/bench_model_config.json",
                     n_features=210,
                     note="210 named features, transformed and concatenated by hushar"),
}
TRITON_SPEC = f"benchmark-data/generated/{MODEL_SIZE}/bench_tensor_feature_spec.json"

# ---------------------------------------------------------------- the cells
#
# Everything below is a table rather than a loop body, so a probe can retune a rung
# without the cell definitions moving.

# Which phase this invocation runs. `tune` ranks configurations of one stack at a single
# load with a short window; `head` runs the pinned winners over the ladder.
PHASE = os.environ.get("PHASE", "head")
if PHASE not in ("tune", "head"):
    sys.exit(f"PHASE={PHASE!r}: use tune or head")

# The default shape both stacks start from: 4 sessions/instances of 16 intra-op threads,
# 64 threads on 64 vCPUs. Two prior campaigns settled that the product should be the vCPU
# count; `PHASE=tune` re-checks the split.
SHAPE = tuple(int(n) for n in os.environ.get("SHAPE", "4x16").split("x"))


def hushar_cfg(mb=4, spin=1, sessions=None, width=None):
    """One hushar configuration. `mb=None` means mini_batch = rows, i.e. no split."""
    return dict(mb=mb, spin=spin,
                sessions=SHAPE[0] if sessions is None else sessions,
                width=SHAPE[1] if width is None else width)


def triton_cfg(instances=None, width=None, exec_mode="0", max_batch=0, delay_us=0):
    """One Triton configuration.

    `max_batch=0` is one graph call per request, which is what the like-for-like campaign
    measured. Above zero it turns on the DYNAMIC BATCHER, which is Triton's own throughput
    lever and the mirror image of hushar's mini-batch split: it aggregates queued requests
    into one bigger graph call instead of splitting one request into smaller concurrent
    ones. `delay_us` is how long it will wait to find company.
    """
    return dict(instances=SHAPE[0] if instances is None else instances,
                width=SHAPE[1] if width is None else width,
                exec_mode=exec_mode, max_batch=max_batch, delay_us=delay_us)


def describe(stack, cfg):
    if stack == "hushar":
        return (f"{cfg['sessions']}x{cfg['width']} "
                f"mb{'rows' if cfg['mb'] is None else cfg['mb']} "
                f"spin{cfg['spin']}")
    pool = "inter" if cfg["exec_mode"] == "1" else "intra"
    batch = ("one call per request" if cfg["max_batch"] == 0
             else f"dynamic batching, max {cfg['max_batch']}, "
                  f"delay {cfg['delay_us']}us")
    return f"{cfg['instances']}x{cfg['width']} on {pool}, {batch}"


# ---------------------------------------------------------------- PHASE=tune
#
# One stack per lane, a short window, and TWO load points. A ranking does not need fifteen
# minutes: it needs every candidate to meet the same traffic on the same host, minutes
# apart.
#
# Two loads, not one, because of Triton's dynamic batcher specifically. At 10,000 rows/s
# the previous campaign used about half the host, so the queue is shallow and the batcher
# rarely finds company -- every delay setting would look like no batching at all and
# Triton's own best lever would be ranked on a load where it cannot work. 24,000 rows/s is
# inside the bracket the previous campaign left for the ceiling, which is the regime the
# head-to-head is actually about. A configuration has to be read at both: the winner at
# 10,000 may not be the winner at 24,000, and if they disagree the report says which load
# the pinned configuration was chosen for.
#
# TUNE_ROWS is a setting because a winner at one row count is not a winner at another. A
# mini batch of 4 is a 25-piece fan-out on a 100-row request and a 50-piece one on a
# 200-row request, and the 100-row pass showed the fan-out's width against the session
# count is exactly what decides the ranking -- two sessions could not absorb 25 pieces at
# all. So each row count the head-to-head measures gets its own tuning pass.
TUNE_ROWS = int(os.environ.get("TUNE_ROWS", "100"))
TUNE_RATES = [int(n) for n in os.environ.get("TUNE_RATES", "10000,24000").split(",")]

# The mini-batch widths worth trying at a row count. Wider requests get a wider candidate:
# at 200 rows a mini batch of 16 is still 13 pieces, which is a plausible fit for 4
# sessions, while at 100 rows it would be 7 and is not where the 10M campaign's optimum sat.
TUNE_MB = {1: [None], 100: [None, 2, 4, 8], 200: [None, 2, 4, 8, 16]}


def tune_hushar(rows):
    """hushar's candidates at this row count.

    The 100-row pass settled the spinning policy decisively -- `mb4 spin0` held its rate on
    69.8% of the host while `mb4 spin1` shed 32,172 requests -- so spinning is no longer the
    axis being swept. It keeps ONE anchor cell so each pass re-confirms the finding on its
    own row count rather than inheriting it, and spends the rest of its cells on the
    mini-batch width and the session split, which is where a wider fan-out could move the
    answer.
    """
    out = [(f"mb{'rows' if mb is None else mb}-spin0", hushar_cfg(mb=mb, spin=0))
           for mb in TUNE_MB.get(rows, [None, 2, 4, 8])]
    out.append(("mb4-spin1", hushar_cfg(mb=4, spin=1)))          # the anchor
    out.append(("mb4-s8x8", hushar_cfg(mb=4, spin=0, sessions=8, width=8)))
    out.append(("mb4-s2x32", hushar_cfg(mb=4, spin=0, sessions=2, width=32)))
    return out


def tune_triton(rows):
    """Triton's candidates: the instance and thread split, the execution mode, and the
    dynamic batcher at three queue delays.

    Unchanged by row count, because none of these levers is about the request's width --
    which is itself the finding at 100 rows, and worth re-testing at 200 rather than
    assumed. max_batch is set well above the largest request so the batcher's behaviour is
    governed by the delay rather than by a ceiling on the batch.
    """
    return [
        ("i4x16",     triton_cfg(instances=4, width=16)),
        ("i8x8",      triton_cfg(instances=8, width=8)),
        ("i2x32",     triton_cfg(instances=2, width=32)),
        ("i16x4",     triton_cfg(instances=16, width=4)),
        ("i4x16-par", triton_cfg(instances=4, width=16, exec_mode="1")),
        ("dyn0",      triton_cfg(max_batch=4000, delay_us=0)),
        ("dyn200",    triton_cfg(max_batch=4000, delay_us=200)),
        ("dyn1000",   triton_cfg(max_batch=4000, delay_us=1000)),
    ]

# ---------------------------------------------------------------- PHASE=head
#
# The winners, pinned. Passed in as JSON by launch.sh so the head-to-head is driven by what
# the tuning MEASURED rather than by what this file guessed; the defaults are the previous
# campaign's winners, so an un-tuned run is still a defensible comparison and says which it
# was.
#
# An empty string is "not set", not malformed JSON: launch.sh exports every pass-through
# variable whether or not it was given one, so `BEST_HUSHAR=` arrives on every lane.


def cfg_from_env(name):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        sys.exit(f"{name} is not JSON: {e}\n  got: {raw[:200]}")


BEST_HUSHAR = cfg_from_env("BEST_HUSHAR") or hushar_cfg(mb=4, spin=1)
BEST_TRITON = cfg_from_env("BEST_TRITON") or triton_cfg()
TUNED = bool(cfg_from_env("BEST_HUSHAR") or cfg_from_env("BEST_TRITON"))

# ONE CONFIGURATION CANNOT BE BEST FOR ALL TRAFFIC, and pretending otherwise would
# handicap this run's own worst case on purpose.
#
# The 100-row tuning pass chose spinning OFF, because there the split supplies the
# parallelism and a spinning pool competes with it: at 24,000 rows/s `mb4 spin0` held
# 240/240 req/s at 69.8% CPU while `mb4 spin1` shed 32,172 and achieved 160.8. None of that
# reasoning survives at ONE row. A one-row request has no split, so the intra-op pool is its
# only parallelism, and the previous campaign measured spinning off shedding 684,719
# requests at 8,000 rows/s because the server could not wake its threads fast enough.
#
# Nor does a width chosen at 100 rows necessarily hold at 200, where the same mini batch is
# twice the fan-out. So every row count the head-to-head measures may carry its own pinned
# configuration, passed in as BEST_HUSHAR_BY_ROWS, e.g.
#
#     BEST_HUSHAR_BY_ROWS='{"200": {"mb":8,"spin":0,"sessions":4,"width":16}}'
#
# Both settings are "hushar at its best" -- for different traffic -- and every cell records
# the configuration it ran, so a table that puts them side by side has to say so. Triton has
# no equivalent choice to make: its onnxruntime backend forwards no spinning key at all.
BEST_HUSHAR_BY_ROWS = {1: dict(BEST_HUSHAR, spin=1)}
for _rows, _cfg in (cfg_from_env("BEST_HUSHAR_BY_ROWS") or {}).items():
    BEST_HUSHAR_BY_ROWS[int(_rows)] = dict(BEST_HUSHAR, **_cfg)
BEST_TRITON_BY_ROWS = {int(k): dict(BEST_TRITON, **v)
                       for k, v in (cfg_from_env("BEST_TRITON_BY_ROWS") or {}).items()}


def best_hushar_for(rows):
    return BEST_HUSHAR_BY_ROWS.get(rows, BEST_HUSHAR)


def best_triton_for(rows):
    return BEST_TRITON_BY_ROWS.get(rows, BEST_TRITON)

# THE LADDER, in rows per second offered. Steps of 4,000, and it reaches to 40,000 because
# the tuning pass moved the target: hushar held 24,000 rows/s at 69.8% CPU, which puts a
# linear estimate of its ceiling near 34,000, while Triton held the same rate at 97.5% and
# so cannot be far above it. A ladder that stopped at 32,000 would have bracketed Triton
# and merely bounded hushar from below. The lowest rung is 20,000 rather than 16,000 for
# the same reason -- both stacks are comfortable there, and the tuning pass already
# measured both at 10,000 and 24,000 if a low-load latency figure is wanted.
#
# Single-row requests keep their own ladder. One row is the WORST case for our design --
# nothing to split, nothing to amortize 210 transformations over -- so it is measured
# deliberately rather than omitted.
LADDER = {
    1:   [4000, 6000, 8000, 10000, 12000],
    100: [20000, 24000, 28000, 32000, 36000, 40000],
    200: [20000, 24000, 28000, 32000, 36000, 40000],
}

# Arm P: one configuration repeated, changing nothing but the clock, so the report has a
# floor to read its gaps against. Deliberately below the ceiling, because a median taken at
# saturation measures how long the queue was. Our design, as the report quotes it.
P_REPEATS = 3
P_ROWS, P_RATE = 100, 10000

ARMS_HEAD = {
    # arm: stack, wire, configuration for a row count, what it is for
    "H": ("hushar", "features", best_hushar_for,
          "OUR DESIGN: 210 named features, vectorized and mini-batched by the server"),
    "T": ("triton", "tensor", best_triton_for,
          "THEIR DESIGN: a finished 690-float tensor, Triton at its best configuration"),
    "X": ("hushar", "tensor", best_hushar_for,
          "CONTROL: hushar on Triton's contract, so H-X is the feature API and X-T is the "
          "two servers on identical bytes"),
    "P": ("hushar", "features", best_hushar_for,
          "precision repeat of our design, below the ceiling"),
}

# Which host carries which arm, and at which row counts. An arm's cells never split across
# hosts, because its table compares them with each other.
#
#   c9g-1  the three-way ladder at the two row counts where candidates are packed, which is
#          the regime the design is for. Every rung is measured by H, T and X back to back,
#          so a rung is a self-contained comparison.
#   c9g-2  the single-row ladder -- the design's worst case, measured on purpose -- and the
#          precision repeats.
LANES_HEAD = {
    "c9g-1": [("H", 100), ("T", 100), ("X", 100),
              ("H", 200), ("T", 200), ("X", 200)],
    "c9g-2": [("H", 1), ("T", 1), ("X", 1),
              ("P", P_ROWS)],
}

# PHASE=tune: one stack per lane, so both are ranked at once and neither ranking is
# interleaved with the other's load. The second pair of hosts is a SECOND RIG, launched to
# tune 200 rows while the head-to-head was already measuring on the first -- tuning on a
# busy host would have contaminated both.
LANES_TUNE = {"c9g-1": "triton", "c9g-2": "hushar",
              "c9g-3": "triton", "c9g-4": "hushar"}


def cell(arm, stack, wire, rows, tps, rate, cfg, note, repeat=None, key=None,
         ladder=None, rung=None):
    """One measurement. Everything a report needs to name this cell is in it."""
    c = dict(arm=arm, stack=stack, wire=wire, rows=rows, tps=tps,
             rows_per_sec=rows * tps, offered_rows_per_sec=rate,
             repeat=repeat, key=key, ladder=ladder, rung=rung,
             config=dict(cfg), config_text=describe(stack, cfg),
             note=f"{note}; {rows} row(s) x {tps} req/s = {rows * tps} rows/s offered")
    if stack == "hushar":
        # A split at least as wide as the request is not a split. One-row requests cannot
        # be split at all, so the cell records that it ran without one rather than
        # claiming a mini batch it did not have.
        mb = cfg["mb"]
        if mb is not None and mb >= rows:
            mb = None
            c["mb_clamped"] = (f"mini batch {cfg['mb']} is not narrower than {rows} "
                               f"row(s), so this cell ran one graph call per request")
        c.update(mb=mb, mb_is_rows=mb is None, spin=cfg["spin"],
                 sessions=cfg["sessions"], width=cfg["width"],
                 threads=cfg["sessions"] * cfg["width"], wt=WORKER_THREADS)
    else:
        c.update(mb=None, mb_is_rows=True, spin=1, sessions=cfg["instances"],
                 width=cfg["width"], threads=cfg["instances"] * cfg["width"], wt=None)
    return c


def cells_tune(host):
    """One stack's candidates, at each load point, in a fixed order.

    Grouped by load rather than by candidate, so every candidate at one load is measured
    close together and the comparison within a load does not carry the host's drift.
    """
    stack = LANES_TUNE[host]
    table = tune_hushar(TUNE_ROWS) if stack == "hushar" else tune_triton(TUNE_ROWS)
    wire = "features" if stack == "hushar" else "tensor"
    out = []
    for rate in TUNE_RATES:
        tps = max(rate // TUNE_ROWS, 1)
        for name, cfg in table:
            out.append(cell("U", stack, wire, TUNE_ROWS, tps, rate, cfg,
                            f"tuning {stack} at {describe(stack, cfg)}",
                            key=f"U-{stack}-{name}-t{tps}"))
    return out


def cells_head(host):
    """This host's lane, in the order the cells should be measured.

    Ordered by (row count, rung, arm) rather than arm by arm, so the arms that a table puts
    side by side are measured minutes apart at the same rung. Adjacent in time is the
    closest two cells on one host can be, and a rung whose three cells were hours apart
    would carry the host's drift as part of its comparison.
    """
    wanted = LANES_HEAD[host]
    if ARMS:
        wanted = [(a, r) for (a, r) in wanted if a in ARMS]
    if ONLY_ROWS:
        wanted = [(a, r) for (a, r) in wanted if r in ONLY_ROWS]

    out = []
    for rows in sorted({r for _, r in wanted}):
        arms_here = [a for a, r in wanted if r == rows]
        if "P" in arms_here:
            stack, wire, cfg, note = ARMS_HEAD["P"]
            tps = max(P_RATE // P_ROWS, 1)
            out += [cell("P", stack, wire, P_ROWS, tps, P_RATE, cfg(P_ROWS),
                         f"{note}, repeat {n} of {P_REPEATS}", repeat=n,
                         key=f"P-r{P_ROWS}t{tps}-rep{n}", ladder=f"P-r{P_ROWS}")
                    for n in range(1, P_REPEATS + 1)]
            arms_here = [a for a in arms_here if a != "P"]
        for rung, rate in enumerate(LADDER[rows]):
            tps = max(rate // rows, 1)
            for arm in arms_here:
                stack, wire, cfg, note = ARMS_HEAD[arm]
                out.append(cell(arm, stack, wire, rows, tps, rate, cfg(rows), note,
                                key=f"{arm}-r{rows}t{tps}", ladder=f"{arm}-r{rows}",
                                rung=rung))
    return out


def cells_for(host):
    out = cells_tune(host) if PHASE == "tune" else cells_head(host)
    for c in out:
        c["host"] = host
        c["id"] = f"{host}-{c['key']}"
    return out


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
    # HeadObject needs GetObject on the exact key; ListBucket is the permission every
    # role here has, so the key is asked for as a prefix and compared exactly.
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


# run_client.sh writes one row per rate through summarise.py, tagged with TAG. TAG is
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
    """The measured window's slices, in order, as numbers."""
    out = []
    for m in SLICE_RE.finditer(text or ""):
        idx, of, n, achieved, p50, p95, p99, shed, failed = m.groups()
        out.append(dict(slice=int(idx), of=int(of), n=int(n),
                        achieved=num(achieved), p50=num(p50), p95=num(p95),
                        p99=num(p99), shed=int(shed), failed=int(failed)))
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
    """Did this five minutes serve the rate it was offered?"""
    if not slice_:
        return False
    return (slice_["shed"] == 0 and slice_["failed"] == 0
            and slice_["achieved"] is not None
            and slice_["achieved"] >= 0.98 * offered_tps)


def verdict(res, slices, offered_tps):
    """Whether the rate was sustained, and if not, how far it got.

    A fifteen-minute window reported as one number cannot tell a rate that is sustained
    from a rate that is sustained for five minutes, which is the whole reason the window
    is this long. So the verdict is over the SLICES, and a cell that held some of them is
    its own state -- `partial` -- rather than being rounded to either answer.

    The whole-window figures still have to agree, because a slice is only the requests
    that started in it and a request that began in slice three and finished after the
    window is counted nowhere.
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
    """Triton's Prometheus text, for the server-side breakdown."""
    rc, out, _ = ssh(ip, f"curl -sf localhost:{METRICS_PORT}/metrics || true",
                     timeout=120)
    return out if rc == 0 else ""


def metric_families(text):
    """`nv_inference_*` sums, counts and quantiles, keyed by family.

    Both a cumulative sum/count and the summary quantiles are taken, because only one of
    them can be made window-correct. A fresh container serves each cell, so its cumulative
    counters start at zero and the sum/count DIFFERENCE across the measured window is an
    exact mean for that window. Quantiles cannot be differenced, so the ones reported are
    cumulative over warmup plus the window and the cell records them as such.

    This is why the hushar-versus-Triton headline is taken from the CLIENT percentiles:
    those are windowed, and now sliced, on both stacks.
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
    them, and a surviving Triton container would share the host with the next hushar cell
    without either noticing. Every pattern is written `[t]arget/...` so the remote shell
    carrying the pattern does not match itself.
    """
    # client side: this host
    sh("pkill -f '[t]arget/release/benchmark-client' >/dev/null 2>&1; true")
    sh("pkill -f '[t]riton_client.py' >/dev/null 2>&1; true")
    sh(f"cd {DIR} && RUN_DIR=/tmp/hushar-clientsample "
       f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1; true")
    sh(f"rm -rf /tmp/hushar-client/{cid} /tmp/triton-client/{cid}")

    # server side
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"scripts/benchmark/run_server.sh stop >/dev/null 2>&1 || true; "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true; "
            f"pkill -f '[t]arget/release/hushar' >/dev/null 2>&1 || true; "
            f"docker rm -f triton >/dev/null 2>&1 || true; "
            f"sleep 3; pkill -9 -f '[t]arget/release/hushar' >/dev/null 2>&1 || true")

    # and prove it, rather than assume the kill worked
    for _ in range(20):
        rc, out, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true; "
                             f"(ss -ltn 2>/dev/null || netstat -ltn) "
                             f"| grep -cE ':({PORT}|{GRPC}) ' || true; "
                             f"docker ps -q --filter name=triton | wc -l")
        nums = [n for n in out.split() if n.isdigit()]
        if len(nums) >= 3 and nums[0] == "0" and nums[1] == "0" and nums[2] == "0":
            return True
        time.sleep(3)
    log(f"{cid} WARNING: server host still shows a server process, a bound port or a "
        f"Triton container after cleaning")
    return False


# ---------------------------------------------------------------- starting a server

def hushar_env(cell_, host):
    h = SERVERS[host]
    env = [
        f"MODEL_CONFIG={WIRES[cell_['wire']]['config']}",
        "PROVIDER=cpu",
        # SESSIONS is the concurrency as well as the session count: the blocking pool is
        # sized to it, one thread per session.
        f"SESSIONS={cell_['sessions']}",
        f"THREADS={cell_['width']}",
        f"WORKER_THREADS={cell_['wt']}",
        f"ALLOW_SPINNING={cell_['spin']}",
        f"CONN_LIMIT={CONN}",
        "CW_NAMESPACE=HusharBench",
        # One ModelId per host, so both lanes can publish to one namespace at the same
        # time without becoming a single time series.
        f"MODEL_ID={h['model_id']}-{cell_['wire']}",
        f"METRICS_BATCH={METRICS_BATCH}",
        f"SAMPLE_RATE={SAMPLE_RATE}",
        f"LOG_URI={DIR}/inference-logs",
        f"LABEL={cell_['id']}",
    ]
    if cell_["mb"] is not None:
        env.append(f"MINI_BATCH_SIZE={cell_['mb']}")
    else:
        # mini_batch = ROWS is one graph call per request, which is Triton's
        # max_batch_size 0. Stated explicitly rather than left unset, so the cell and its
        # Triton twin are the same shape.
        env.append(f"MINI_BATCH_SIZE={cell_['rows']}")
    return " ".join(env)


def start_hushar(cell_, host, ip):
    """Starts hushar and returns its banner, or None having recorded why not."""
    rc, out, err = ssh(
        ip,
        f"cd {DIR} && export HOME=/root && {hushar_env(cell_, host)} "
        f"scripts/benchmark/run_server.sh start 2>&1 | tail -40",
        timeout=1200,
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

    # The spinning policy the server says it applied. The runtime accepts a setting it
    # does not know without a word, so the banner is the only evidence.
    want = ("spin   : idle intra-op and inter-op threads "
            + ("spin" if cell_["spin"] else "sleep"))
    if want not in out:
        log(f"{cell_['id']} ABORT: banner does not say '{want}'")
        return None, {"failed": f"banner missing '{want}'", "banner": banner}

    # The wire, from the banner's own feature line. Every cell here is the tensor wire, so
    # a banner showing 210 features means the deploy staged the wrong configuration.
    n = WIRES[cell_["wire"]]["n_features"]
    expect = ("<- [features] transformed and concatenated" if n == 1
              else f"<- {n} features transformed and concatenated")
    if expect not in out:
        log(f"{cell_['id']} ABORT: banner does not show the {cell_['wire']} wire")
        return None, {"failed": f"banner does not show '{expect}'", "banner": banner}

    # Exactly one engine, so the cell is not sharing the host with a survivor.
    _, procs, _ = ssh(ip, "pgrep -cf '[t]arget/release/hushar' || true")
    n_procs = procs.strip().split()[0] if procs.strip() else "?"
    if n_procs != "1":
        log(f"{cell_['id']} ABORT: {n_procs} hushar process(es), expected 1")
        return None, {"failed": f"{n_procs} hushar processes running", "banner": banner}
    return banner, {"ort_version": m.group(1)}


def start_triton(cell_, ip):
    """Builds the repository, starts the container, and checks what it actually serves.

    The configuration comes from the cell, not from this function: `PHASE=tune` varies the
    instance and thread split, the execution mode and the dynamic batcher, and `PHASE=head`
    pins whichever won. A Triton tuned only to match hushar's shape would not be Triton at
    its best, which is what this run claims to compare against.
    """
    cfg = cell_["config"]
    rc, out, err = ssh(
        ip,
        f"export HOME=/root && cd {HARNESS} && "
        f"BUNDLE={DIR} REPO=/root/triton/models MODEL_SIZE={MODEL_SIZE} "
        f"MODEL_FILE={MODEL_FILE} MODEL_NAME={TRITON_MODEL} MODEL_MD5={MODEL_MD5} "
        f"INSTANCES={cfg['instances']} "
        f"INTRA={cfg['width'] if cfg['exec_mode'] == '0' else 1} "
        f"INTER={cfg['width'] if cfg['exec_mode'] == '1' else 1} "
        f"EXEC_MODE={cfg['exec_mode']} "
        f"MAX_BATCH={cfg['max_batch']} DELAY_US={cfg['delay_us']} "
        f"SPINNING=default python3 make_repo.py 2>&1 && "
        f"MODEL_NAME={TRITON_MODEL} EXPECT_BACKEND=onnxruntime "
        f"TRITON_IMAGE={TRITON_IMAGE} REPO=/root/triton/models "
        f"GRPC_PORT={GRPC} METRICS_PORT={METRICS_PORT} ./run_triton.sh start 2>&1 | tail -40",
        timeout=1200,
    )
    if rc != 0:
        log(f"{cell_['id']} TRITON FAILED rc={rc}\n{out[-1500:]}\n{err[-500:]}")
        return None, {"failed": "Triton did not start", "server_out": out[-3000:]}
    # run_triton.sh aborts by itself on the wrong backend or a disagreeing config; this
    # line is the evidence, kept in the cell so the report can point at it.
    if "backend check: onnxruntime" not in out:
        log(f"{cell_['id']} ABORT: not on the C++ onnxruntime backend")
        return None, {"failed": "not the onnxruntime backend",
                      "server_out": out[-3000:]}
    if f"model md5 {MODEL_MD5}" not in out:
        log(f"{cell_['id']} ABORT: Triton did not confirm the model md5")
        return None, {"failed": "model md5 unconfirmed", "server_out": out[-3000:]}
    # The batcher is the one setting whose absence would be invisible: a config.pbtxt that
    # failed to declare it still serves, just without the lever this cell is measuring.
    want_batch = f"max_batch_size {cfg['max_batch']}"
    if want_batch not in out:
        log(f"{cell_['id']} ABORT: make_repo did not report '{want_batch}'")
        return None, {"failed": f"config does not report '{want_batch}'",
                      "server_out": out[-3000:]}
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]
    return banner, {"ort_version": ORT_VER,
                    "spinning": "ort-default-on",
                    "spinning_note": "no spinning key in config.pbtxt; the onnxruntime "
                                     "backend in 26.08 forwards no "
                                     "session.*.allow_spinning entry, so ONNX Runtime's "
                                     "default spin applies",
                    "dynamic_batching": (None if cfg["max_batch"] == 0
                                         else {"max_batch_size": cfg["max_batch"],
                                               "max_queue_delay_us": cfg["delay_us"]})}


# ---------------------------------------------------------------- running one cell

def client_command(cell_, ip, label):
    """The one client invocation: warmup, then one continuous window reported in slices."""
    if cell_["stack"] == "hushar":
        return (
            f"cd {DIR} && export HOME=/root && "
            f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={WIRES[cell_['wire']]['config']} "
            f"TAG=cell ROWS={cell_['rows']} TPS={cell_['tps']} "
            f"WARMUP={WARMUP} DURATION={DURATION} SEGMENTS={SEGMENTS} LABEL={label} "
            f"scripts/benchmark/run_client.sh > /tmp/{label}.client.out 2>&1"
        )
    return (
        f"cd {HARNESS} && export HOME=/root && "
        f"mkdir -p /tmp/triton-client/{label} && "
        f"python3 triton_client.py --server {ip}:{GRPC} --model {TRITON_MODEL} "
        f"--feature-spec {DIR}/{TRITON_SPEC} "
        f"--rows {cell_['rows']} --tps {cell_['tps']} "
        f"--warmup-secs {WARMUP} --duration-secs {DURATION} --segments {SEGMENTS} "
        f"--label '{label}' --out /tmp/triton-client/{label}/result.json "
        f"> /tmp/triton-client/{label}/client.log 2>&1"
    )


def run_cell(cell_, host):
    ip = SERVERS[host]["ip"]
    cid = cell_["id"]
    key = f"{PREFIX}/cells/{cid}.json"
    if s3_has(key):
        log(f"{cid} already measured, skipping")
        return True

    mb = "rows" if cell_["mb"] is None else cell_["mb"]
    log(f"{cid} start  {cell_['stack']}/{cell_['wire']} "
        f"s{cell_['sessions']}x{cell_['width']} mb{mb} spin{cell_['spin']} "
        f"{cell_['threads']}thr/{CORES}cpu "
        f"{cell_['rows']}rows x {cell_['tps']}tps = {cell_['rows_per_sec']} rows/s")

    # 1 · a clean server and a clean client, verified
    clean_ok = clean(ip, cid)
    if cell_["stack"] == "hushar":
        banner, extra = start_hushar(cell_, host, ip)
    else:
        banner, extra = start_triton(cell_, ip)
    if banner is None:
        record(cell_, host, extra, ok=False)
        clean(ip, cid)
        return False

    # 2 · one client invocation, on this host
    label = cid
    started = time.time()
    ssh(ip, "true")   # keep the control channel warm while the client runs
    proc = subprocess.Popen(client_command(cell_, ip, label), shell=True)

    # 3 · sample both hosts over the MEASURED window only
    time.sleep(WARMUP + GUARD)
    win_start = time.time()
    metrics_before = scrape_metrics(ip) if cell_["stack"] == "triton" else ""
    csample = f"/tmp/hushar-clientsample/{cid}.log"
    tag = "s" if cell_["stack"] == "hushar" else "t"
    sh(f"mkdir -p /tmp/hushar-clientsample && cd {DIR} && "
       f"RUN_DIR=/tmp/hushar-clientsample OUT={csample} "
       f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1; true")
    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/{tag}-{cid}.log "
            f"scripts/benchmark/sample_host.sh start >/dev/null 2>&1")

    try:
        client_rc = proc.wait(timeout=WARMUP + DURATION + 1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        client_rc = 124
    win_end = time.time()
    # Before the container is stopped and before the samplers, so the delta covers the
    # measured window and nothing after it.
    metrics_after = scrape_metrics(ip) if cell_["stack"] == "triton" else ""

    ssh(ip, f"cd {DIR} && export HOME=/root && "
            f"OUT=/tmp/hushar-server/{tag}-{cid}.log "
            f"scripts/benchmark/sample_host.sh stop >/dev/null 2>&1 || true")
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
                             "client_out": tail, "banner": banner}, ok=False)
        clean(ip, cid)
        return False
    if len(slices) != SEGMENTS:
        # Not fatal -- the whole-window result is still a measurement -- but the point of
        # a fifteen-minute window is the slices, so a cell without them says so.
        log(f"{cid} WARNING: {len(slices)} slice line(s), expected {SEGMENTS}")

    # 5 · collect this cell's raw output NOW, in the session that measured it.
    if cell_["stack"] == "hushar":
        ssh(ip, f"export HOME=/root; cd /tmp/hushar-server && "
                f"for f in {cid}.server.log {cid}.service.json {cid}.model.json "
                f"s-{cid}.log; do "
                f"aws s3 cp $f s3://{BUCKET}/{PREFIX}/raw/server/{host}/ "
                f"--only-show-errors 2>/dev/null; done")
        s3_put_tree(f"/tmp/hushar-client/{label}", f"{PREFIX}/raw/client/{host}/{label}")
    else:
        ssh(ip, f"export HOME=/root; "
                f"cd /root/triton/models/{TRITON_MODEL} && "
                f"aws s3 cp config.pbtxt "
                f"s3://{BUCKET}/{PREFIX}/raw/server/{host}/{cid}.config.pbtxt "
                f"--only-show-errors 2>/dev/null; "
                f"cd /tmp/hushar-server && aws s3 cp t-{cid}.log "
                f"s3://{BUCKET}/{PREFIX}/raw/server/{host}/ --only-show-errors "
                f"2>/dev/null; "
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
        "clean_slate_verified": clean_ok,
        "window": {"start": iso(win_start), "end": iso(win_end),
                   "measured_from": iso(win_start),
                   "note": f"one continuous {DURATION}s load reported as {SEGMENTS} "
                           f"slice(s); sampling and srv percentiles cover "
                           f"{int(win_end - win_start)}s, starting {GUARD}s after warmup "
                           f"ended so warmup's last metric batch falls outside it"},
        "wall": {"cell_start": iso(started), "cell_end": iso(win_end)},
    }
    doc.update(extra)
    if cell_["stack"] == "triton":
        doc["triton_server_side"] = server_side(metrics_before, metrics_after)
        doc["triton_image"] = TRITON_IMAGE
    record(cell_, host, doc)
    log(f"{cid} done   p50={res['p50']} p95={res['p95']} "
        f"achieved={res['achieved']}/{res['offered']} shed={res['shed']} "
        f"srvcpu={cpu_percent(cpu)} clicpu={cpu_percent(ccpu)} "
        f"{v['state']} ({v['slices_held']}/{v['slices_total']} slice(s) held) "
        f"-- settling {SETTLE}s")
    time.sleep(SETTLE)
    return True


def record(cell_, host, extra, ok=True):
    """A measured cell goes to cells/, which is what "already done" means.

    A failed attempt goes to failures/ under its own timestamp, so it is kept for
    diagnosis but a resume retries the cell rather than treating the failure as a result.
    """
    doc = dict(cell_)
    doc.update(extra)
    doc.update(instance=SERVERS[host]["instance"], arch=SERVERS[host]["arch"],
               phase=PHASE, tuned=TUNED,
               warmup_s=WARMUP, duration_s=DURATION, segments=SEGMENTS,
               settle_s=SETTLE, guard_s=GUARD,
               conn=CONN, sample_rate=SAMPLE_RATE, cores=CORES,
               wire_note=WIRES[cell_["wire"]]["note"],
               model_file=MODEL_FILE, model_md5=MODEL_MD5, model_size=MODEL_SIZE,
               rev=REV, run=RUN)
    p = f"/tmp/{cell_['id']}.json"
    json.dump(doc, open(p, "w"), indent=2)
    if ok:
        s3_put(p, f"{PREFIX}/cells/{cell_['id']}.json")
    else:
        s3_put(p, f"{PREFIX}/failures/{cell_['id']}-{int(time.time())}.json")


def put_lane_log(host):
    """This orchestrator's own log, beside the cells.

    The previous campaign lost its lane logs when the rig was terminated. A cell carries
    its own timestamps, so the loss was survivable, but the log is the only record of what
    a lane SKIPPED and why -- which this run needs, because its ladders stop early.

    The path comes from the launcher through LANE_LOG rather than being rebuilt here: the
    lane tag is the launcher's to choose, and a second guess at the filename is a second
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


# ---------------------------------------------------------------- preflight

def preflight(host, need_triton):
    """Refuses to measure a machine that is not the one the report will name.

    Every figure is attributed to an instance type, an engine version, a graph and a
    server build, so each is checked against the host itself rather than trusted from
    this file.
    """
    h = SERVERS[host]
    ip, want = h["ip"], h["instance"]
    gen = f"{DIR}/benchmark-data/generated/{MODEL_SIZE}"
    rc, out, err = ssh(ip, "set -e; "
                       "T=$(curl -sf -X PUT http://169.254.169.254/latest/api/token "
                       "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); "
                       "echo type=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/instance-type); "
                       "echo ami=$(curl -sf -H \"X-aws-ec2-metadata-token: $T\" "
                       "http://169.254.169.254/latest/meta-data/ami-id); "
                       "echo kernel=$(uname -r); "
                       "echo nproc=$(nproc); "
                       f"echo providers=$(ls -1 /root/ort-cpu-{ORT_VER}/lib/ 2>/dev/null "
                       "| grep providers_ | grep -vc providers_shared || true); "
                       f"echo ortso=$(readlink -f /root/ort-cpu-{ORT_VER}/lib/"
                       "libonnxruntime.so | sed 's/.*libonnxruntime\\.so\\.//'); "
                       f"echo md5=$(md5sum {gen}/{MODEL_FILE} | cut -c1-32); "
                       f"echo tensorcfg=$(test -f {gen}/bench_model_tensor_config.json "
                       "&& echo yes || echo no); "
                       f"echo tensorspec=$(test -f {gen}/bench_tensor_feature_spec.json "
                       "&& echo yes || echo no); "
                       "echo onnx=$(python3 -c 'import onnx' >/dev/null 2>&1 "
                       "&& echo yes || echo no); "
                       f"echo docker=$(command -v docker >/dev/null && echo yes || echo no)")
    if rc != 0:
        sys.exit(f"preflight: cannot reach {host} at {ip}: {err[-300:]}")
    f = dict(l.split("=", 1) for l in out.split() if "=" in l)
    log(f"preflight {host}: {f}")

    if f.get("type") != want:
        sys.exit(f"preflight: {host} is {f.get('type')} but the run claims {want}")
    if f.get("nproc") != str(CORES):
        sys.exit(f"preflight: {host} has {f.get('nproc')} CPUs, expected {CORES}")
    if f.get("providers") != "0":
        sys.exit(f"preflight: {host} ships {f.get('providers')} accelerator provider "
                 f"library/libraries; this run is CPU only")
    if f.get("ortso") != ORT_VER:
        sys.exit(f"preflight: {host} has libonnxruntime.so.{f.get('ortso')} in "
                 f"/root/ort-cpu-{ORT_VER}/lib, expected {ORT_VER}")
    if f.get("md5") != MODEL_MD5:
        sys.exit(f"preflight: {host} model md5 {f.get('md5')}, expected {MODEL_MD5}")
    # The tensor wire IS this run's wire, so a missing file is not a degraded run, it is
    # no run at all.
    for name in ("tensorcfg", "tensorspec"):
        if f.get(name) != "yes":
            sys.exit(f"preflight: {host} is missing the tensor wire's {name}; "
                     f"run make_tensor_config.py and re-deploy")
    if need_triton and f.get("docker") != "yes":
        sys.exit(f"preflight: {host} has no docker, and this lane has Triton cells")
    # make_repo.py derives the config.pbtxt shapes from the graph itself, so a Triton lane
    # needs onnx importable on the SERVER. A freshly launched host has neither onnx nor
    # pip, and discovering that one cell at a time costs the lane its whole Triton arm.
    if need_triton and f.get("onnx") != "yes":
        sys.exit(f"preflight: {host} cannot import onnx, which make_repo.py needs to read "
                 f"the graph's signature; re-run deploy_triton.sh")

    # Both clients print one line per slice, which is how a fifteen-minute cell says WHEN
    # it stopped holding. An older binary or an older harness would run the cell and
    # silently produce nothing to slice, so both are checked before any cell is measured.
    rc, out, _ = sh(f"cd {DIR} && ./target/release/benchmark-client --help 2>&1", 120)
    if rc != 0 or "--segments" not in out:
        sys.exit("preflight: this benchmark-client has no --segments; the bundle predates "
                 "the per-slice reporting this run depends on. Re-package and re-deploy.")
    if need_triton:
        rc, out, _ = sh(f"cd {HARNESS} && python3 triton_client.py --help 2>&1", 120)
        if rc != 0 or "--segments" not in out:
            sys.exit("preflight: this triton_client.py has no --segments; the harness "
                     "predates the per-slice reporting. Re-package and re-deploy.")
    log(f"preflight {host}: both clients report slices")

    log(f"preflight {host}: OK, {want}, {CORES} CPUs, CPU-only runtime, "
        f"ami {f.get('ami')}, kernel {f.get('kernel')}, model md5 {f.get('md5')}"
        + (", docker present" if need_triton else ""))
    return f


# ---------------------------------------------------------------- main

def drive(cells, host):
    """Measures the lane, skipping the rungs a failed rung has already answered.

    THE EARLY STOP. A ladder exists to bracket a ceiling, and once a rung has not been
    sustained every higher rung on the same ladder is known to fail too -- measuring them
    costs twenty minutes each and adds nothing. So a ladder stops at its first unsustained
    rung, and the skipped cells are recorded as skipped rather than silently absent.

    `partial` does NOT stop a ladder on its own the way `failed` does: a rate that held
    for ten minutes and then broke is the most interesting result this run can produce,
    and the rung above it is worth measuring to see whether it breaks sooner. A ladder
    stops on `failed`, or on a second consecutive `partial`.
    """
    if MAX_CELLS:
        cells = cells[:MAX_CELLS]
        log(f"MAX_CELLS={MAX_CELLS}: driving only {len(cells)} cell(s)")

    stopped = {}        # ladder -> why it stopped
    partials = {}       # ladder -> consecutive partial count
    for c in cells:
        ladder = c["ladder"]
        if EARLY_STOP and ladder in stopped:
            log(f"{c['id']} SKIPPED: {stopped[ladder]}")
            record(c, host, {"skipped": stopped[ladder]}, ok=False)
            continue

        measured = False
        for attempt in (1, 2):
            try:
                if run_cell(c, host):
                    measured = True
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
        put_lane_log(host)

        # The verdict comes from S3 rather than from run_cell, so a resumed lane reaches
        # the same stop as an uninterrupted one: a cell skipped because S3 already holds
        # it still governs the rungs above it.
        if not measured or c["rung"] is None:
            continue
        doc = s3_get_json(f"{PREFIX}/cells/{c['id']}.json") or {}
        state = (doc.get("verdict") or {}).get("state", "unknown")
        if state == "failed":
            stopped[ladder] = (f"{c['key']} did not hold any of its "
                               f"{SEGMENTS} slice(s); every higher rung would fail too")
            log(f"ladder {ladder} STOPS: {stopped[ladder]}")
        elif state == "partial":
            partials[ladder] = partials.get(ladder, 0) + 1
            if partials[ladder] >= 2:
                stopped[ladder] = (f"{c['key']} and the rung below it both held only "
                                   f"part of the window")
                log(f"ladder {ladder} STOPS: {stopped[ladder]}")
        else:
            partials[ladder] = 0


def main():
    host = sys.argv[1]
    if host not in SERVERS:
        sys.exit(f"unknown host {host}; RIG names {sorted(SERVERS)}")
    lanes = LANES_TUNE if PHASE == "tune" else LANES_HEAD
    if host not in lanes:
        sys.exit(f"no {PHASE} lane defined for {host}; lanes are {sorted(lanes)}")
    mine = cells_for(host)
    preflight(host, need_triton=any(c["stack"] == "triton" for c in mine))

    per_cell_min = (WARMUP + GUARD + DURATION + SETTLE + 150) / 60.0
    by_arm = {}
    for c in mine:
        by_arm[c["arm"], c["stack"], c["wire"]] = \
            by_arm.get((c["arm"], c["stack"], c["wire"]), 0) + 1
    log(f"host={host} rev={REV} run={RUN} phase={PHASE} bucket set")
    if PHASE == "head":
        log(f"pinned hushar: {describe('hushar', BEST_HUSHAR)}")
        log(f"pinned triton: {describe('triton', BEST_TRITON)}")
        if TUNED:
            log("configurations were TUNED by PHASE=tune")
        else:
            log("configurations were NOT tuned: these are the previous campaign's "
                "winners, and the report must say so")
    for (arm, stack, wire), n in sorted(by_arm.items()):
        log(f"  arm {arm}: {stack}/{wire}, {n} cell(s)")
    log(f"{len(mine)} cell(s) at most: " + ", ".join(c["key"] for c in mine))
    log(f"warmup={WARMUP}s measured={DURATION}s in {SEGMENTS} slice(s) settle={SETTLE}s "
        f"-> {per_cell_min:.1f} min/cell, at most "
        f"{len(mine) * per_cell_min / 60:.1f} h"
        + ("; ladders stop early, so fewer" if PHASE == "head" else ""))
    drive(mine, host)
    put_lane_log(host)
    log(f"host={host} SWEEP COMPLETE")


if __name__ == "__main__":
    main()
