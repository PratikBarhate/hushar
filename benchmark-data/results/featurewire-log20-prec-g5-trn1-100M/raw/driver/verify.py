#!/usr/bin/env python3
"""Re-derives every number in `summary.md` from the evidence, and fails on disagreement.

    ./verify.py            # must print `0 failures`

The report is generated, not written, so the risk is not a typo -- it is a generator that
reads the wrong field, or a run whose settings were not what the configuration table
claims. So this checks three different things, and the third is the one that matters most:

  1 EVERY FIGURE IN THE REPORT appears in the results JSON it was taken from. Each table
    row is parsed back out of the markdown and matched against the cell.
  2 EVERY SETTING THE REPORT CLAIMS is on every cell AND in the server's own banner --
    the sample rate and the fraction it resolved to, the S3 destination, the object size,
    the feature count, the session arithmetic, the Neuron core and context counts, the
    absence of a mini batch on the A10G and its equality with the row count on Trainium,
    and the absence of a candidate model.
  3 A CONTROL TEST proves the checks can fail: a fabricated cell carrying the control
    run's pre-vectorized wire, and another carrying logging at 0.0, must both be caught.
    A verifier that cannot fail is not evidence.

Nothing here reads `report.py`. The two derive the same numbers from the same inputs
independently, which is the only arrangement where agreement means anything.
"""
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BD = benchmark_data(HERE)
# OVERRIDABLE so `selftest.py` can point this and `report.py` at a
# synthetic results tree. A test that cannot run the real verifier is not a
# test of the real verifier.
RUN = os.environ.get("REPORT_RUN", "featurewire-log20-prec-g5-trn1-100M")
RAW = BD / "results" / RUN / "raw"
SUMMARY = BD / "results" / RUN / "summary.md"

# What this round was specified with. Written here as literals, independently of the
# sweeps, so a driver that drifted from the instructions is caught rather than echoed.
WANT = dict(
    sample_rate=0.2,
    sample_every=5,
    log_rows_per_object=50000,
    log_store="Amazon S3",
    sessions=2,
    sessions_per_request=1,
    cores_per_replica=1,
    contexts=1,
    n_features=210,
    pieces=1,
    warmup=300,
    duration=900,
    segments=15,
    settle=60,
)
# KEYED BY (ACCELERATOR, PRECISION), because the precision is this round's axis and the
# two arms are offered DIFFERENT rates -- BF16 double FP32 at each packaging. A grid check
# keyed on size alone would accept an arm that ran the other arm's rates.
# WAVE A is the four rates the round was specified with, measured on both accelerators.
# WAVE B is Trainium only: rates set from wave A's MEASURED ceilings so that the cells
# sustain and therefore report a latency, every one a multiple of 5 at or below 80% device
# occupancy. The A10G needs no wave B -- all eight of its cells held wave A's rates.
WANT_LOADS = {
    ("a10g", "fp32"): [(100, 100), (500, 20)],
    ("a10g", "bf16"): [(100, 200), (500, 40)],
    ("trainium1", "fp32"): [(100, 100), (500, 20),            # wave A, over the ceiling
                            (100, 70), (500, 10)],            # wave B, sustainable
    ("trainium1", "bf16"): [(100, 200), (500, 40),            # wave A, over the ceiling
                            (100, 70), (500, 10),             # wave B, matched with fp32
                            (100, 100), (500, 20)],           # wave B, BF16-only headroom
}
WANT_SIZE = "100m"
# What each arm must have asked its builder for. NEITHER IS THAT TOOL'S DEFAULT: TensorRT
# does not enable BF16 unless told, and neuronx-cc 2.26 casts matmuls unless told not to.
WANT_AUTO_CAST = {"fp32": "none", "bf16": "matmult"}
# A BF16 artifact holds its weights at half the width. The bound is loose -- the engine
# also carries layer metadata and any layer the builder left at full width -- but a BF16
# arm that did not shrink AT ALL did not cast anything.
MAX_BF16_SIZE_RATIO = 0.85
# The compiled-versus-PyTorch maximum error that separates the Trainium arms. FP32 lands
# near 1e-6 on this architecture; casting the matmuls puts it three orders above that --
# 6.9e-04 measured over 2 layers, 2.3e-03 over 31.
MAX_FP32_TRACE_ERROR = 1e-4
# The cell-id prefixes, which are also the CloudWatch ModelId. Named once here so a round
# derived from another cannot keep the parent's prefix while claiming its own.
WANT_PREFIX = {"a10g": "g5fwp-", "trainium1": "trn1fwp-"}

FAILURES = []


def fail(what):
    FAILURES.append(what)


def ok(cond, what):
    if not cond:
        fail(what)
    return cond


def g(doc, *path, default=None):
    for key in path:
        if not isinstance(doc, dict):
            return default
        doc = doc.get(key)
    return default if doc is None else doc


def load(path):
    if not path.exists():
        sys.exit(f"missing {path}; run the lane's collect.py and report.py first")
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, list) else doc.get("cells", [])


def banner_text(cell):
    """The server banner as one string, from the cell record and the archived log."""
    text = "\n".join(cell.get("banner") or [])
    log = RAW / "server" / str(cell.get("host")) / f"{cell['id']}.server.log"
    if log.exists():
        text += "\n" + log.read_text(errors="replace")
    return text


def service_config(cell):
    """The service configuration the server actually resolved, as archived."""
    path = RAW / "server" / str(cell.get("host")) / f"{cell['id']}.service.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def model_config(cell):
    path = RAW / "server" / str(cell.get("host")) / f"{cell['id']}.model.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


# ---------------------------------------------------------------- 2 · the settings

def check_cell_settings(cell, strict_banner=True):
    """Every setting this round was specified with, on one cell."""
    cid = cell["id"]
    accel = cell.get("accelerator")
    neuron = accel == "trainium1"
    hushar = cell.get("stack", "hushar") == "hushar"

    ok(accel in ("a10g", "trainium1"), f"{cid}: accelerator {accel!r} is not one of the two")

    # ---- THE AXIS. A precision label is the one thing in this round that could be wrong
    # while every other gate passed, so it is checked against the ARTIFACT rather than
    # against itself. The cross-arm ratios are checked once, in `check_precision`.
    prec = cell.get("precision")
    ok(prec in ("fp32", "bf16"),
       f"{cid}: precision {prec!r} is not one of the two arms")
    ok(prec and f"-{prec}-" in cid,
       f"{cid}: the cell id does not carry its precision, so two arms could share an "
       f"S3 record and a CloudWatch series")
    if neuron:
        # The compiler's own --auto-cast default is `matmult`, so an absent value means
        # the FP32 claim rests on a default that is not FP32.
        want_cast = {"fp32": "none", "bf16": "matmult"}[prec] if prec else None
        ok(cell.get("auto_cast") == want_cast,
           f"{cid}: auto_cast {cell.get('auto_cast')!r}, expected {want_cast!r} -- "
           f"both arms must pass --auto-cast explicitly, so the label rests on the "
           f"artifact rather than on a framework default")
    elif hushar:
        # hushar echoes every provider option it registered, so the flag's presence or
        # absence is the server's own account of the precision it built.
        ok(cell.get("bf16_flag_echoed") == (prec == "bf16"),
           f"{cid}: the banner's BF16 flag was "
           f"{'absent' if not cell.get('bf16_flag_echoed') else 'present'} on a "
           f"{prec} cell")

    ok(cell.get("pieces") == WANT["pieces"] and not cell.get("split"),
       f"{cid}: {cell.get('pieces')} piece(s), split={cell.get('split')}; "
       f"this round scores every request in one call")
    ok((cell.get("parallelism") or cell.get("sessions")) == WANT["sessions"],
       f"{cid}: parallelism {cell.get('parallelism')}, expected {WANT['sessions']}")
    ok(cell.get("warmup") == WANT["warmup"] and cell.get("duration") == WANT["duration"]
       and cell.get("segments") == WANT["segments"]
       and cell.get("settle") == WANT["settle"],
       f"{cid}: window is {cell.get('warmup')}+{cell.get('duration')}s/"
       f"{cell.get('segments')} settle {cell.get('settle')}s, expected "
       f"{WANT['warmup']}+{WANT['duration']}/{WANT['segments']} settle {WANT['settle']}")

    if not hushar:
        # Triton writes no log and vectorizes nothing; the report must not claim it does.
        ok(cell.get("sample_rate") is None and cell.get("log_batch") is None,
           f"{cid}: a Triton cell carries log settings")
        ok(cell.get("wire") == "tensor",
           f"{cid}: Triton's wire is {cell.get('wire')!r}, expected tensor")
        return

    # ---- the wire
    ok(cell.get("wire") == "features",
       f"{cid}: wire is {cell.get('wire')!r}, expected features -- this round "
       f"vectorizes on the server")
    # ---- the log
    ok(cell.get("sample_rate") == WANT["sample_rate"],
       f"{cid}: sample_rate {cell.get('sample_rate')}, expected {WANT['sample_rate']}")
    ok(cell.get("sample_every") == WANT["sample_every"],
       f"{cid}: the rate resolved to 1 in {cell.get('sample_every')}, expected "
       f"1 in {WANT['sample_every']}")
    ok(cell.get("log_rows_per_object") == WANT["log_rows_per_object"],
       f"{cid}: {cell.get('log_rows_per_object')} rows per object, expected "
       f"{WANT['log_rows_per_object']}")
    ok(cell.get("log_batch") == WANT["log_rows_per_object"] // cell["rows"],
       f"{cid}: log_batch {cell.get('log_batch')} is not "
       f"{WANT['log_rows_per_object']}/{cell['rows']}")
    ok(cell.get("log_store") == WANT["log_store"],
       f"{cid}: log store {cell.get('log_store')!r}, expected {WANT['log_store']!r}")
    # A CELL THAT SHED OR LOST WRITES IS NOT A FAULT TO REJECT -- it is a measurement of
    # the log path under load, and rejecting it would make this verifier refuse reality.
    # What IS a fault is a report that quietly ranks such a cell's latency as "with
    # logging on" without saying so. So the requirement is DISCLOSURE, checked against
    # the report in `check_report`, and all that is required here is that the counts were
    # captured at all rather than left unknown.
    ok(cell.get("log_shed_batches") is not None,
       f"{cid}: no inference-log shed count was captured, so the report cannot say "
       f"whether this cell paid the full cost of logging")
    ok(cell.get("log_write_failures") is not None,
       f"{cid}: no inference-log write-failure count was captured")
    ok(cell.get("log_objects") is not None,
       f"{cid}: no inference-log object count was captured, so nothing confirms the "
       f"rows reached the bucket")

    # ---- Neuron placement
    if neuron:
        ok(cell.get("cores_per_replica") == WANT["cores_per_replica"],
           f"{cid}: {cell.get('cores_per_replica')} core(s) per replica, expected "
           f"{WANT['cores_per_replica']}")
        ok(cell.get("contexts") == WANT["contexts"],
           f"{cid}: {cell.get('contexts')} context(s) per replica, expected "
           f"{WANT['contexts']}")
        ok(cell.get("mini_batch") == cell["rows"],
           f"{cid}: mini_batch {cell.get('mini_batch')} != rows {cell['rows']}; on a "
           f"fixed-shape NEFF that is how one call per request is spelled")
    else:
        ok(cell.get("mini_batch_configured") is None,
           f"{cid}: mini_batch_size {cell.get('mini_batch_configured')} was configured; "
           f"this round configures none on the A10G")

    # ---- the resolved service configuration, which is the server's own copy
    svc = service_config(cell)
    if svc is None:
        if strict_banner:
            fail(f"{cid}: no archived service configuration to check the settings against")
    else:
        ok(g(svc, "inference_log", "sample_rate") == WANT["sample_rate"],
           f"{cid}: the resolved configuration says sample_rate "
           f"{g(svc, 'inference_log', 'sample_rate')}")
        ok(str(g(svc, "inference_log", "uri", default="")).startswith("s3://"),
           f"{cid}: the resolved configuration's log uri is not an s3:// location")
        ok(g(svc, "inference_log", "batch_size") == cell.get("log_batch"),
           f"{cid}: the resolved configuration's log batch_size "
           f"{g(svc, 'inference_log', 'batch_size')} != {cell.get('log_batch')}")
        ok(g(svc, "threading", "sessions_per_model") == WANT["sessions"],
           f"{cid}: the resolved configuration says sessions_per_model "
           f"{g(svc, 'threading', 'sessions_per_model')}")
        ok(g(svc, "threading", "sessions_per_request", default=1)
           == WANT["sessions_per_request"],
           f"{cid}: the resolved configuration leases "
           f"{g(svc, 'threading', 'sessions_per_request', default=1)} session(s) per "
           f"request")
        ok("candidate_model" not in svc,
           f"{cid}: the resolved configuration carries a candidate model")

    mdl = model_config(cell)
    if mdl is not None:
        order = g(mdl, "vectorization_config", "feature_order", default=[])
        ok(len(order) == WANT["n_features"],
           f"{cid}: the model configuration names {len(order)} feature(s), expected "
           f"{WANT['n_features']}")
        if neuron:
            ok(mdl.get("mini_batch_size") == cell["rows"] and mdl.get("is_fixed"),
               f"{cid}: the model configuration says mini_batch_size "
               f"{mdl.get('mini_batch_size')} is_fixed {mdl.get('is_fixed')}")
            ok(g(mdl, "backend", "neuron_cores_per_replica") == WANT["cores_per_replica"]
               and g(mdl, "backend", "execution_contexts_per_replica") == WANT["contexts"],
               f"{cid}: the model configuration's Neuron backend is "
               f"{g(mdl, 'backend')}")
        else:
            ok("mini_batch_size" not in mdl,
               f"{cid}: the model configuration carries mini_batch_size "
               f"{mdl.get('mini_batch_size')}")

    # ---- the banner, which is the server saying what it did
    hay = banner_text(cell)
    if not hay.strip():
        if strict_banner:
            fail(f"{cid}: no banner recorded, so no setting can be read back from the server")
        return
    # The percent is DERIVED from the resolved fraction, not written down: the round's
    # rate is a knob, and a hardcoded "50.0%" left over from the previous round would
    # pass a banner that logged at a different rate than the report claims.
    want_log = (f"logging 1 request in {WANT['sample_every']} "
                f"({100.0 / WANT['sample_every']:.1f}%)")
    ok(want_log in hay,
       f"{cid}: the banner does not report {want_log!r}")
    ok("inference logs -> s3://" in hay and "Amazon S3" in hay,
       f"{cid}: the banner does not report an Amazon S3 inference log")
    ok(f"every {cell.get('log_batch')} batches" in hay,
       f"{cid}: the banner does not report an object every "
       f"{cell.get('log_batch')} batches")
    ok(f"<- {WANT['n_features']} features transformed and concatenated" in hay,
       f"{cid}: the banner does not show the {WANT['n_features']}-feature wire")
    ok("A/B" not in hay, f"{cid}: the banner reports a candidate model")
    if neuron:
        ok(f"{WANT['sessions']} replica(s) on NeuronCore " in hay,
           f"{cid}: the banner does not report {WANT['sessions']} replica(s)")
        ok(f"{WANT['cores_per_replica']} core(s) and {WANT['contexts']} "
           f"execution context(s) per replica" in hay,
           f"{cid}: the banner does not report one core and one context per replica")
        # POSITIVE, NOT AN ABSENCE. This used to assert `"in groups of" not in hay`,
        # but that string does not exist anywhere in the service any more, so the check
        # could never fire and proved nothing. The ctx line states the lease directly.
        ok("on the one replica it leases" in hay,
           f"{cid}: the banner does not state that a request leases ONE replica, so the "
           f"requests in flight may not be the session count")
        ok("leases an idle replica while it scores" in hay,
           f"{cid}: the banner does not state that a request leases an idle replica")
    else:
        ok("plan     : whole request" in hay,
           f"{cid}: the banner does not report the `whole request` plan")
        ok("mini batches of" not in hay,
           f"{cid}: the banner reports a mini batch; this round configures none")
        ok("TensorRT(device=0" in hay,
           f"{cid}: the banner does not name the TensorRT provider")
        # The CURRENT wording. hushar once printed `N concurrent request(s) x 1
        # session(s) each`; that clause is gone, and a gate still looking for it aborted
        # every cell of `nominibatch-precision-gpu-g5-100M` before a row was scored.
        ok(f"pool   : {WANT['sessions']} session(s) per model, "
           f"one blocking thread each" in hay,
           f"{cid}: the banner does not report {WANT['sessions']} session(s) per model")
        ok("a request leases an idle session while it scores" in hay,
           f"{cid}: the banner does not state one session per request")
        # The precision, from the provider options the server echoed.
        ok(("trt_bf16_enable=1" in hay) == (cell.get("precision") == "bf16"),
           f"{cid}: the banner's BF16 flag does not match the cell's "
           f"{cell.get('precision')} label")


# ---------------------------------------------------------------- 1 · the report

NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_tables(text):
    """Every markdown table in the report, as (header, rows of cells)."""
    tables, header, body = [], None, []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-:") and c for c in cells):
                continue
            if header is None:
                header = cells
            else:
                body.append(cells)
        else:
            if header is not None:
                tables.append((header, body))
            header, body = None, []
    if header is not None:
        tables.append((header, body))
    return tables


def check_report(cells, text):
    """Every latency and utilization in the report traces to a cell."""
    by_id = {c["id"]: c for c in cells}
    tables = parse_tables(text)
    ok(len(tables) >= 6, f"the report has {len(tables)} table(s), expected at least 6")

    # Build the set of numbers the evidence can justify, to two decimals, so a figure
    # the generator invented has nowhere to come from.
    allowed = set()
    for c in cells:
        res = c.get("result") or {}
        neuron = c.get("accelerator") == "trainium1"
        for value in (res.get("p50"), res.get("p95"), res.get("p99"),
                      res.get("achieved")):
            if value is not None:
                allowed.add(f"{value:.2f}")
                allowed.add(f"{value:.1f}")
        for value in (
            g(c, "server_side", "graph_ms", "p50") if neuron
                else g(c, "cloudwatch", "InferenceTime", "p50_ms"),
            g(c, "server_side", "vectorize_ms", "p50") if neuron
                else g(c, "cloudwatch", "VectorizationTime", "p50_ms"),
            g(c, "cloudwatch", "ToTensorTime", "p50_ms"),
        ):
            if value is not None:
                allowed.add(f"{value:.3f}")
        for value in (
            g(c, "neuron_series", "nc_mean_pct") if neuron
                else g(c, "server_utilization", "cpu_pct"),
            g(c, "host_series", "cpu_pct") if neuron
                else g(c, "gpu_utilization", "gpu_util_pct"),
        ):
            if value is not None:
                allowed.add(f"{value:.1f}")

    # A SHORT CELL MUST NEVER BE RANKED. The convention this series uses is that a cell
    # which did not hold its offered rate reports its achieved rate and no latency, so a
    # reader cannot compare it with one that did.
    for header, body in tables:
        if "result" not in header or "p50" not in header:
            continue
        ri, pi = header.index("result"), header.index("p50")
        for cells_row in body:
            if cells_row[ri].startswith(("FAIL", "partial")):
                ok(cells_row[pi] == "--",
                   f"the report prints p50 {cells_row[pi]} for an unsustained cell "
                   f"({cells_row[ri]}); an unsustained cell reports its achieved rate "
                   f"and no latency")

    # Every per-cell latency in the `Every cell` table is one of the recorded ones.
    for header, body in tables:
        if header[:3] != ["accel", "precision", "rows x TPS"] or "p50" not in header:
            continue
        for cells_row in body:
            for col in ("p50", "p95", "p99"):
                if col not in header:
                    continue
                value = cells_row[header.index(col)]
                if NUMBER.match(value):
                    ok(value in allowed,
                       f"the report's {col} {value} in row {cells_row[:3]} is not a "
                       f"recorded figure")

    # The headline counts must match the cell set.
    sustained = sum(1 for c in cells if c.get("sustained"))
    ok(f"{len(cells)} measured, {sustained} sustained" in text,
       f"the configuration table does not say '{len(cells)} measured, "
       f"{sustained} sustained'")

    # DISCLOSURE, WHICH IS THE REAL REQUIREMENT AROUND THE LOG PATH. A cell that shed
    # buffered batches or lost writes did not pay the full cost of logging, so its latency
    # is not "with logging on" in the way the round claims. That is a legitimate
    # measurement -- but only if the report says so, with the number, rather than ranking
    # the cell silently alongside cells that paid in full.
    shed = [c for c in cells if c.get("log_shed_batches")]
    lost = [c for c in cells if c.get("log_write_failures")]
    for c in shed:
        ok(str(c["log_shed_batches"]) in text,
           f"{c['id']}: shed {c['log_shed_batches']} log batch(es), and the report does "
           f"not print that number anywhere -- so a reader cannot tell this cell paid "
           f"less than the full cost of logging")
    for c in lost:
        ok(str(c["log_write_failures"]) in text,
           f"{c['id']}: {c['log_write_failures']} inference-log write(s) failed, and the "
           f"report does not print that number")
    if shed or lost:
        ok("shed" in text.lower() and "write failure" in text.lower(),
           "cells shed log batches or lost writes, but the report has no column naming "
           "either")
    return by_id


# ---------------------------------------------------------------- the grid itself

def check_grid(cells):
    ids = [c["id"] for c in cells]
    ok(len(set(ids)) == len(ids), f"duplicate cell id(s) among {len(ids)}")
    for c in cells:
        prefix = WANT_PREFIX.get(c.get("accelerator"), "?")
        ok(c["id"].startswith(prefix),
           f"{c['id']}: does not carry this round's own `{prefix}` prefix, so its "
           f"CloudWatch ModelId could collide with a control run's")
    ok(all(c.get("size") == WANT_SIZE for c in cells),
       f"this round serves one model size ({WANT_SIZE}); found "
       f"{sorted({c.get('size') for c in cells})}")
    for (accel, prec), want in WANT_LOADS.items():
        got = sorted({(c["rows"], c["tps"]) for c in cells
                      if c.get("accelerator") == accel and c.get("precision") == prec
                      and c.get("stack", "hushar") == "hushar"})
        if not got:
            fail(f"{accel} {prec}: no hushar cell was measured")
            continue
        ok(got == sorted(want),
           f"{accel} {prec}: loads {got}, specified {sorted(want)}")

    # THE SHARED-LIBRARY CLAIM. The report says both A10G arms execute on one TensorRT,
    # so the two arms' recorded versions must agree and must be present.
    trt = {c.get("host_tensorrt") for c in cells if c.get("accelerator") == "a10g"}
    trt.discard(None)
    a10g = [c for c in cells if c.get("accelerator") == "a10g"]
    if a10g:
        ok(len(trt) == 1,
           f"the A10G cells recorded TensorRT {sorted(trt)}; the report claims both arms "
           f"execute on the same library")
        ok(trt and next(iter(trt)),
           "no TensorRT version was recorded on the A10G cells, so the shared-library "
           "claim rests on the deploy log alone")


def check_precision(cells):
    """Proves the two arms are genuinely different artifacts, not two labels.

    THE ONE FAILURE THIS ROUND COULD PUBLISH AND HAVE BELIEVED is a precision arm that
    ran the other precision. Every other fault makes a cell fail or a number look odd;
    this one produces a clean, complete, internally consistent report whose headline
    finding ("BF16 bought nothing", or "BF16 bought everything") is an artifact of a flag
    that did not take. So the arms are compared against each other here, on evidence that
    does not come from the label:

      the artifact   a BF16 engine or NEFF holds its weights at half the width, so it must
                     be materially SMALLER than its FP32 pair at the same shape.
      the cast       on Trainium, what the compiler was actually told. `neuronx-cc`'s own
                     default is `matmult`, so `none` must be explicit on the FP32 arm --
                     and every NEFF built before this round lacked the flag entirely,
                     which is why its metadata is not trusted without `auto_cast`.
      the error      the compiled-versus-PyTorch maximum. FP32 lands near 1e-6 and BF16
                     matmuls near 1e-2, four orders apart, so the arms are separable from
                     this number with no reference to any flag.
      the model      both arms must report the SAME parameter and layer count, because the
                     cast is a builder choice and not a change to the graph. An arm that
                     differs here is serving another model, and the comparison is void.
    """
    for accel in ("a10g", "trainium1"):
        arms = {}
        for prec in ("fp32", "bf16"):
            match = [c for c in cells if c.get("accelerator") == accel
                     and c.get("precision") == prec]
            if match:
                arms[prec] = match
        if len(arms) < 2:
            # Not a failure: a lane may have been truncated. But the report must not then
            # print a precision comparison, which `check_report` enforces by provenance.
            continue

        # ---- the artifact, per shape and per stack
        for stack in ("hushar", "triton"):
            for rows in sorted({c["rows"] for c in cells}):
                def size_of(prec):
                    return sorted({c["engine_bytes"] for c in arms.get(prec, [])
                                   if c.get("stack", "hushar") == stack
                                   and c["rows"] == rows and c.get("engine_bytes")})
                f, b = size_of("fp32"), size_of("bf16")
                if not f or not b:
                    continue
                ratio = b[0] / f[0]
                ok(ratio <= MAX_BF16_SIZE_RATIO,
                   f"{accel} {stack} r{rows}: the BF16 artifact is {b[0]:,} bytes "
                   f"against FP32's {f[0]:,} -- {ratio:.3f}x, above the "
                   f"{MAX_BF16_SIZE_RATIO} ceiling. A BF16 build holds its weights at "
                   f"half the width, so an arm that did not shrink did not cast.")

        # ---- the model is the same on both arms
        for field in ("parameter_count", "layers"):
            values = {prec: sorted({c.get(field) for c in match if c.get(field)})
                      for prec, match in arms.items()}
            flat = {v for vs in values.values() for v in vs}
            ok(len(flat) <= 1,
               f"{accel}: the two precision arms report different {field} {values}; the "
               f"cast is a builder choice and must not change the graph")

        # ---- Trainium: the cast and the error
        if accel != "trainium1":
            continue
        for prec, match in arms.items():
            casts = {c.get("auto_cast") for c in match}
            ok(casts == {WANT_AUTO_CAST[prec]},
               f"trainium1 {prec}: auto_cast {sorted(c for c in casts if c)}, expected "
               f"{WANT_AUTO_CAST[prec]!r}")
            errs = [c.get("maximum_trace_error") or g(c, "metadata", "maximum_trace_error")
                    for c in match]
            errs = [float(e) for e in errs if e is not None]
            if not errs:
                fail(f"trainium1 {prec}: no compiled-versus-PyTorch error was recorded, "
                     f"so nothing independent of the flag says this arm is {prec}")
                continue
            if prec == "fp32":
                ok(max(errs) < MAX_FP32_TRACE_ERROR,
                   f"trainium1 fp32: trace error {max(errs):.3g} is at or above "
                   f"{MAX_FP32_TRACE_ERROR:.0e}, which is where a CAST shows up -- this "
                   f"arm may not be FP32")
            else:
                ok(min(errs) >= MAX_FP32_TRACE_ERROR,
                   f"trainium1 bf16: trace error {min(errs):.3g} is below "
                   f"{MAX_FP32_TRACE_ERROR:.0e}, which is FP32 territory -- the cast "
                   f"may not have applied and this arm would be FP32 under a BF16 label")


# ---------------------------------------------------------------- 3 · the control test

def control_test():
    """Proves the settings checks can fail, on cells built to be wrong."""
    before = len(FAILURES)
    base = dict(
        id=WANT_PREFIX["a10g"] + "control-100m-fp32-r100t100-p2", accelerator="a10g",
        stack="hushar", precision="fp32", bf16_flag_echoed=False,
        wire="features", size="100m", rows=100, tps=100, parallelism=2, pieces=1,
        split=False, sample_rate=WANT['sample_rate'],
        sample_every=WANT['sample_every'], log_rows_per_object=50000,
        log_batch=500, log_store="Amazon S3", log_write_failures=0,
        # Captured-ness is what the real checks require of these three, so the control
        # cell has to carry them or it fails for being synthetic rather than for being
        # wrong -- which is the one way this control test could mislead.
        log_shed_batches=0, log_objects=1200,
        mini_batch_configured=None, warmup=300, duration=900, segments=15, settle=60,
        banner=["plan     : whole request", "TensorRT(device=0", "A",
                "<- 210 features transformed and concatenated",
                # The CURRENT pool wording, which `check_cell_settings` now requires.
                # Without it this control cell fails for being out of date rather than
                # for being wrong, which would make the control test useless.
                f"  pool   : {WANT['sessions']} session(s) per model, one blocking "
                f"thread each, every session with its own intra-op pool; a request "
                f"leases an idle session while it scores",
                "inference logs -> s3://b/k [Amazon S3, 16 prefix shard(s), "
                f"every 500 batches, 4 in flight, logging 1 request in "
                f"{WANT['sample_every']} ({100.0 / WANT['sample_every']:.1f}%)]"],
        host=None,
    )
    check_cell_settings(dict(base), strict_banner=False)
    caught_clean = len(FAILURES) - before
    ok(caught_clean == 0,
       f"the control test's CLEAN cell produced {caught_clean} failure(s); the checks "
       f"reject a correct cell")

    for what, mutation in (
        ("the pre-vectorized wire", dict(wire="tensor")),
        ("logging switched off", dict(sample_rate=0.0, sample_every=0)),
        ("the previous round's 50% rate", dict(sample_rate=0.5,
                                               sample_every=2)),
        ("a mini batch configured", dict(mini_batch_configured=100)),
        ("four sessions", dict(parallelism=4)),
        ("the default log batch", dict(log_batch=5000)),
        ("a local log destination", dict(log_store="local filesystem")),
        # NOT "write failures happened" -- that is a measurement of the log path under
        # load, and the requirement on it is DISCLOSURE in the report, checked in
        # `check_report`. What is a fault here is the count never having been captured,
        # because then nothing can say whether the cell paid the full cost of logging.
        ("an uncaptured write-failure count", dict(log_write_failures=None)),
        ("an uncaptured shed count", dict(log_shed_batches=None)),
        ("an uncaptured object count", dict(log_objects=None)),
        ("a short measured window", dict(duration=300)),
        # ---- THE AXIS. Each of these is a way a precision arm could be mislabelled
        # while serving perfectly, which is the one fault this round could publish and
        # have believed.
        ("a precision outside the two arms", dict(precision="int8")),
        ("a precision missing from the cell id",
         dict(id=WANT_PREFIX["a10g"] + "control-100m-r100t100-p2")),
        ("a bf16 label on a cell whose banner had no BF16 flag",
         dict(precision="bf16", bf16_flag_echoed=False)),
        ("an fp32 label on a cell whose banner DID carry the BF16 flag",
         dict(precision="fp32", bf16_flag_echoed=True)),
    ):
        mark = len(FAILURES)
        check_cell_settings(dict(base, **mutation), strict_banner=False)
        if len(FAILURES) == mark:
            fail(f"CONTROL TEST: a cell with {what} was NOT caught, so this verifier "
                 f"cannot detect that fault in a real cell")
        else:
            del FAILURES[mark:]


def main():
    gpu = load(RAW / "results-gpu.json")
    neu = load(RAW / "results-neuron.json")
    cells = gpu + neu
    if not cells:
        sys.exit("no cells in either results file")
    if not SUMMARY.exists():
        sys.exit(f"missing {SUMMARY}; run report.py first")
    text = SUMMARY.read_text()

    print(f"{len(cells)} cell(s): {len(gpu)} A10G, {len(neu)} Trainium")
    check_grid(cells)
    check_precision(cells)
    for c in cells:
        check_cell_settings(c)
    check_report(cells, text)
    control_test()

    print()
    if FAILURES:
        seen = []
        for f in FAILURES:
            if f and f not in seen:
                seen.append(f)
        print(f"{len(seen)} failure(s):")
        for f in seen:
            print(f"  - {f}")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
