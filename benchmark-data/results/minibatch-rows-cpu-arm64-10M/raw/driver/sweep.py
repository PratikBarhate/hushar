#!/usr/bin/env python3
"""Drives one server host through its lane of the rows-and-mini-batch run, from its client.

    HUSHAR_BENCH_BUCKET=... sweep.py <server-host>

Runs detached. Every cell is written to S3 the moment it finishes, so a lost laptop, a
dropped session or a restarted orchestrator loses at most the cell in flight; re-running
skips whatever S3 already holds.

WHAT THIS RUN ASKS. Mini-batching is supposed to pay when a request carries many
candidates to score. The previous campaigns swept threading and measured that claim only
in passing, at two row counts. This one makes ROWS PER REQUEST the primary axis and asks
three questions in order:

    R   same rows per second, different packaging. Does the host spend less CPU per
        candidate when candidates arrive many to a request?
    M   given a multi-candidate request, does splitting it into mini batches scored at
        the same time cut latency, and does the gain grow with the row count?
    T   is any of this a property of hushar rather than of the workload? Triton serving
        the same graph, the same bytes and the same engine answers that.

Plus three supporting arms: C puts a capacity number on each packaging so a shed cell
becomes a measurement, V prices hushar's feature API against the tensor wire, and P
re-measures the noise floor and the between-host offset on this build.

TWO WIRES, ONE GRAPH. hushar takes named features and builds the tensor; Triton takes the
tensor. Comparing them as they stand would measure the servers and 210 transformations at
once, so the parity arm gives hushar a configuration whose single feature IS the finished
690-wide vector, with an identity transformation. Both stacks then receive the same bytes
and call the graph once each, and arm V measures what the real 210-feature wire costs on
top. See driver/make_tensor_config.py.

SPINNING IS SYMMETRIC HERE, FOR THE FIRST TIME. Triton's onnxruntime backend forwards no
`session.*.allow_spinning` key, so Triton always runs ONNX Runtime's default spin; every
previous campaign compared it against a hushar that slept. Arm T runs hushar with spinning
ON so the two match. It is not assumed safe elsewhere: the 10M campaign found spinning ON
collapsed the best mini-batch configuration at its top load, so arms R and M measure both
policies.

COMPARISON GROUPS DO NOT CROSS HOSTS. The 10M campaign measured a 1.7% offset between the
two c9g hosts, which repeating does not reduce. Every arm whose table puts cells side by
side is placed entirely on one host; only the choice of host differs between arms.

EVERY CELL GETS A CLEAN SERVER AND A CLEAN CLIENT. `clean()` runs on both hosts before
every cell and the process table is checked afterwards.
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
RUN = os.environ.get("RUN", "minibatch-rows-cpu-arm64-10M")
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

WARMUP = int(os.environ.get("WARMUP", "120"))      # 2 min, discarded
DURATION = int(os.environ.get("DURATION", "300"))  # 5 min, measured
SETTLE = int(os.environ.get("SETTLE", "60"))       # 1 min between cells
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
# Which arms this invocation runs, in order. The probe passes a short list and short
# windows; the main sweep passes nothing and gets the lane's full order.
ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a]
# Narrows an arm to these row counts, so a probe can ask one question. Empty means all.
ONLY_ROWS = {int(n) for n in os.environ.get("ONLY_ROWS", "").split(",") if n.strip()}

SERVERS = {
    "c9g-1": dict(ip="172.31.46.43", instance="c9g.16xlarge", arch="arm64",
                  model_id="vec-c9g-1"),
    "c9g-2": dict(ip="172.31.33.183", instance="c9g.16xlarge", arch="arm64",
                  model_id="vec-c9g-2"),
}

# The two wires, as model configurations the deploy staged. `features` is what a caller
# actually sends; `tensor` is the already-vectorized form Triton's contract requires, so
# it is the only wire on which the two stacks can be compared.
WIRES = {
    "features": dict(config=f"benchmark-data/generated/{MODEL_SIZE}/bench_model_config.json",
                     n_features=210,
                     note="210 named features, transformed and concatenated by hushar"),
    "tensor": dict(config=f"benchmark-data/generated/{MODEL_SIZE}/bench_model_tensor_config.json",
                   n_features=1,
                   note="one 690-wide vector, identity transformation"),
}
TRITON_SPEC = f"benchmark-data/generated/{MODEL_SIZE}/bench_tensor_feature_spec.json"

# ---------------------------------------------------------------- the cells
#
# Everything below is a table rather than a loop body, so the probe can retune a rate
# without the cell definitions moving.

# sessions x intra-op width. The product is the vCPU count, which is the budget rule the
# earlier campaigns established; the probe picks between candidates at one load point.
SHAPE = tuple(int(n) for n in os.environ.get("SHAPE", "4x16").split("x"))

# Arm R holds ROWS PER SECOND constant and varies the packaging, so a column of cells is
# the same work arriving differently. Two levels, both probe-checked to be reachable by
# single-row requests: 2000 rows/s costs 79% of the host when packed one per request, and
# 8000 rows/s saturates it at 99.9%, which is the ceiling that packaging has.
R_ROWS = [1, 5, 10, 25, 50, 100, 200]
R_LEVELS = [int(n) for n in os.environ.get("R_LEVELS", "2000,8000").split(",")]
# Spinning off is measured at BOTH levels, at the two ends of the axis and one middle.
# Both, not just the heavy one: the probe found spinning off collapses at rows=1 and
# 8000 rows/s (6419 of 8000 served, 204k shed at 36% CPU), so a light-level pair is what
# gives the spin comparison a cell that sustained at each end.
R_SPIN_OFF_ROWS = [1, 10, 100]

# Arms M, T and V compare cells WITHIN one row count, so their rate only has to be a
# realistic load for that row count rather than equal across rows. 10000 rows/s wherever
# it is reachable, which the probe measured at 49% of the host for 100-row requests;
# single-row requests cannot reach it -- 8000 rows/s already saturates -- so rows=1 is
# offered 1000 and the table says so. A cell that sheds is kept and marked, never
# replaced by a slower rate.
LOAD = {1: 1000, 10: 1000, 25: 400, 50: 200, 100: 100, 200: 50}

# Arm M: the mini-batch split. `None` means mini_batch = ROWS, which is one graph call per
# request and the symmetric setting Triton runs. A split wider than the request is not a
# split, so cells_for drops those.
M_ROWS = [10, 50, 100, 200]
M_BATCHES = [None, 4, 8, 16]
M_SPIN_OFF_BATCHES = [4]        # the winner from the 10M campaign, re-measured sleeping

# Arms T and V: the parity comparison and the price of the feature API.
TV_ROWS = [1, 10, 50, 100, 200]

# Arm C: the capacity ladder, in rows per second, so a packaging gets a number rather
# than a verdict. Each rate is a separate cell; the highest sustained one is the answer.
# The single-row ladder reaches past the 8000 rows/s the probe sustained, so the ceiling
# is bracketed rather than merely touched.
C_ROWS = [1, 10, 100]
C_LADDER = {1: [2000, 4000, 8000, 12000],
            10: [5000, 10000, 20000, 40000],
            100: [5000, 10000, 20000, 40000]}

# Arm P: one configuration repeated, changing nothing but the clock. Within-host spread is
# run-to-run noise; the gap between the hosts is a placement offset.
P_REPEATS = 3

# Which host carries which arm. An arm's cells never split across hosts, because its table
# compares them with each other.
LANES = {
    "c9g-1": ["R", "M", "P"],
    "c9g-2": ["T", "V", "P", "C"],
}


def cell(arm, stack, wire, rows, tps, mb, spin, key, note, sessions=None, width=None,
         repeat=None):
    sessions = SHAPE[0] if sessions is None else sessions
    width = SHAPE[1] if width is None else width
    return dict(arm=arm, stack=stack, wire=wire, rows=rows, tps=tps, mb=mb,
                mb_is_rows=mb is None, spin=spin, sessions=sessions, width=width,
                threads=sessions * width, wt=WORKER_THREADS if stack == "hushar" else None,
                rows_per_sec=rows * tps, repeat=repeat, key=key, note=note)


def arm_R():
    """Same rows per second, different packaging."""
    out = []
    for level in R_LEVELS:
        for rows in R_ROWS:
            tps = max(level // rows, 1)
            for spin in (1, 0):
                if spin == 0 and rows not in R_SPIN_OFF_ROWS:
                    continue
                out.append(cell(
                    "R", "hushar", "features", rows, tps, None, spin,
                    f"R-l{level}-r{rows}t{tps}-spin{spin}",
                    f"{level} rows/s offered as {rows} row(s) x {tps} req/s, "
                    f"one graph call per request, spinning "
                    f"{'on' if spin else 'off'}"))
    return out


def arm_M():
    """The mini-batch split, against the row count."""
    out = []
    for rows in M_ROWS:
        tps = LOAD[rows]
        for mb in M_BATCHES:
            # A mini batch at least as wide as the request is the no-split cell, which
            # `None` already is, so a duplicate under another name is dropped.
            if mb is not None and mb >= rows:
                continue
            for spin in (1, 0):
                if spin == 0 and mb not in M_SPIN_OFF_BATCHES:
                    continue
                name = "rows" if mb is None else str(mb)
                out.append(cell(
                    "M", "hushar", "features", rows, tps, mb, spin,
                    f"M-r{rows}t{tps}-mb{name}-spin{spin}",
                    f"mini batch {name} on {rows} row(s) x {tps} req/s, spinning "
                    f"{'on' if spin else 'off'}"))
    return out


def arm_T():
    """Triton on the tensor wire: the parity arm. Spinning on, because Triton's is."""
    return [cell("T", "triton", "tensor", rows, LOAD[rows], None, 1,
                 f"T-r{rows}t{LOAD[rows]}",
                 f"Triton, one graph call per request, {rows} row(s) x {LOAD[rows]} req/s")
            for rows in TV_ROWS]


def arm_V():
    """hushar on both wires at Triton's load: parity against T, and the feature price."""
    out = []
    for rows in TV_ROWS:
        tps = LOAD[rows]
        for wire in ("tensor", "features"):
            out.append(cell(
                "V", "hushar", wire, rows, tps, None, 1,
                f"V-{wire}-r{rows}t{tps}",
                f"hushar on the {wire} wire, one graph call per request, "
                f"{rows} row(s) x {tps} req/s"))
    return out


def arm_C():
    """A capacity ladder per packaging, so a ceiling is a number."""
    out = []
    for rows in C_ROWS:
        for level in C_LADDER[rows]:
            tps = max(level // rows, 1)
            out.append(cell(
                "C", "hushar", "features", rows, tps, None, 1,
                f"C-r{rows}t{tps}",
                f"capacity ladder: {rows} row(s) x {tps} req/s = {rows * tps} rows/s"))
    return out


def arm_P():
    """One configuration repeated, for the noise floor and the between-host offset."""
    rows, tps = 100, LOAD[100]
    return [cell("P", "hushar", "features", rows, tps, 4, 1,
                 f"P-r{rows}t{tps}-rep{n}",
                 f"precision repeat {n} of {P_REPEATS}, mini batch 4", repeat=n)
            for n in range(1, P_REPEATS + 1)]


BUILDERS = {"R": arm_R, "M": arm_M, "T": arm_T, "V": arm_V, "C": arm_C, "P": arm_P}


def cells_for(host):
    """This host's lane, in the order the arms should be measured.

    `ARMS` set explicitly is taken as given rather than intersected with the lane: the
    probe deliberately runs a lane's arm on the other host to compare two shapes at once,
    and a filter that silently dropped those cells would look like a lane that finished.
    `ONLY_ROWS` narrows further, which is how a probe asks one question without measuring
    the whole arm.

    Arms T and V are interleaved by row count rather than run one after the other: the
    parity comparison is between two cells, and adjacent in time is the closest two cells
    on one host can be.
    """
    wanted = ARMS or LANES[host]
    out = []
    if "T" in wanted and "V" in wanted:
        by_rows = {}
        for c in arm_T() + arm_V():
            by_rows.setdefault(c["rows"], []).append(c)
        for rows in TV_ROWS:
            out += sorted(by_rows.get(rows, []), key=lambda c: (c["stack"], c["wire"]))
        wanted = [a for a in wanted if a not in ("T", "V")]
    for arm in wanted:
        out += BUILDERS[arm]()
    if ONLY_ROWS:
        out = [c for c in out if c["rows"] in ONLY_ROWS]
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


def sustained(res):
    """A rate is a result only if the server kept up with it.

    Offered and achieved have to agree with nothing shed and nothing failed. A cell that
    shed is not a slower version of the same measurement -- it is a different one, taken
    at whatever rate the server was actually able to serve -- so it must never rank.
    """
    if not res:
        return False
    return (res["shed"] == 0 and res["failed"] == 0
            and res["achieved"] >= 0.98 * res["offered"])


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
    those are windowed on both stacks, on the same host, for both.
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
        # time without becoming a single time series. The wire is in the id too: the
        # parity and feature cells are different measurements of the same graph.
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

    # The wire, from the banner's own feature line: the parity cell must bind exactly one
    # feature and the real cell all 210, or the two arms have been swapped.
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
    """Builds the repository, starts the container, and checks what it actually serves."""
    rc, out, err = ssh(
        ip,
        f"export HOME=/root && cd {HARNESS} && "
        f"BUNDLE={DIR} REPO=/root/triton/models MODEL_SIZE={MODEL_SIZE} "
        f"MODEL_FILE={MODEL_FILE} MODEL_NAME={TRITON_MODEL} MODEL_MD5={MODEL_MD5} "
        f"INSTANCES={cell_['sessions']} INTRA={cell_['width']} EXEC_MODE=0 "
        f"MAX_BATCH=0 SPINNING=default python3 make_repo.py 2>&1 && "
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
    banner = [l.rstrip() for l in out.splitlines() if l.strip()]
    return banner, {"ort_version": ORT_VER,
                    "spinning": "ort-default-on",
                    "spinning_note": "no spinning key in config.pbtxt; the onnxruntime "
                                     "backend in 26.08 forwards no "
                                     "session.*.allow_spinning entry, so ONNX Runtime's "
                                     "default spin applies"}


# ---------------------------------------------------------------- running one cell

def client_command(cell_, ip, label):
    """The one client invocation: warmup then the measured window, on this host."""
    if cell_["stack"] == "hushar":
        return (
            f"cd {DIR} && export HOME=/root && "
            f"SERVER=http://{ip}:{PORT} MODEL_CONFIG={WIRES[cell_['wire']]['config']} "
            f"TAG=cell ROWS={cell_['rows']} TPS={cell_['tps']} "
            f"WARMUP={WARMUP} DURATION={DURATION} LABEL={label} "
            f"scripts/benchmark/run_client.sh > /tmp/{label}.client.out 2>&1"
        )
    return (
        f"cd {HARNESS} && export HOME=/root && "
        f"mkdir -p /tmp/triton-client/{label} && "
        f"python3 triton_client.py --server {ip}:{GRPC} --model {TRITON_MODEL} "
        f"--feature-spec {DIR}/{TRITON_SPEC} "
        f"--rows {cell_['rows']} --tps {cell_['tps']} "
        f"--warmup-secs {WARMUP} --duration-secs {DURATION} "
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
        client_rc = proc.wait(timeout=WARMUP + DURATION + 900)
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

    # 4 · the client's own numbers
    summary, res = "", None
    if cell_["stack"] == "hushar":
        try:
            summary = open(f"/tmp/hushar-client/{label}/summary.txt").read()
        except OSError:
            pass
        res = parse_summary(summary)
    else:
        try:
            summary = open(f"/tmp/triton-client/{label}/client.log").read()
        except OSError:
            pass
        res = parse_triton_client(summary)

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

    doc = {
        "banner": banner,
        "summary": summary if cell_["stack"] == "hushar" else summary[-4000:],
        "result": res,
        "sustained": sustained(res),
        "cpu_report": cpu,
        "client_cpu_report": ccpu,
        "server_cpu_pct": cpu_percent(cpu),
        "client_cpu_pct": cpu_percent(ccpu),
        "clean_slate_verified": clean_ok,
        "window": {"start": iso(win_start), "end": iso(win_end),
                   "measured_from": iso(win_start),
                   "note": f"sampling and srv percentiles cover "
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
        f"sustained={sustained(res)} -- settling {SETTLE}s")
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
               warmup_s=WARMUP, duration_s=DURATION, settle_s=SETTLE, guard_s=GUARD,
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
    # The parity arm cannot be improvised on the host: both files are derived on the
    # laptop from the real configuration, so a missing one means a stale deploy.
    for name in ("tensorcfg", "tensorspec"):
        if f.get(name) != "yes":
            sys.exit(f"preflight: {host} is missing the parity arm's {name}; "
                     f"run make_tensor_config.py and re-deploy")
    if need_triton and f.get("docker") != "yes":
        sys.exit(f"preflight: {host} has no docker, and this lane has Triton cells")

    log(f"preflight {host}: OK, {want}, {CORES} CPUs, CPU-only runtime, "
        f"model md5 {f.get('md5')}, both wires staged"
        + (", docker present" if need_triton else ""))


# ---------------------------------------------------------------- main

def drive(cells, host):
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
    host = sys.argv[1]
    if host not in SERVERS:
        sys.exit(f"unknown host {host}; one of {sorted(SERVERS)}")
    mine = cells_for(host)
    preflight(host, need_triton=any(c["stack"] == "triton" for c in mine))

    per_cell_min = (WARMUP + GUARD + DURATION + SETTLE + 90) / 60.0
    by_arm = {}
    for c in mine:
        by_arm[c["arm"]] = by_arm.get(c["arm"], 0) + 1
    log(f"host={host} rev={REV} run={RUN} shape={SHAPE[0]}x{SHAPE[1]} bucket set")
    log(f"arms: " + ", ".join(f"{a}={n}" for a, n in sorted(by_arm.items())))
    log(f"{len(mine)} cell(s): " + ", ".join(c["key"] for c in mine))
    log(f"warmup={WARMUP}s measured={DURATION}s settle={SETTLE}s "
        f"-> {per_cell_min:.1f} min/cell, about {len(mine) * per_cell_min / 60:.1f} h")
    drive(mine, host)
    log(f"host={host} SWEEP COMPLETE")


if __name__ == "__main__":
    main()
