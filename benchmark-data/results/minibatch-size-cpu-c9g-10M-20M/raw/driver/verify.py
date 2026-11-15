#!/usr/bin/env python3
"""Re-derives every number in `summary.md` from the evidence, and fails on disagreement.

    ./verify.py            # must print `0 failures`

The report is generated, not written, so the risk is not a typo -- it is a generator that
reads the wrong field, or a run whose settings were not what the configuration table
claims. So this checks four different things, and the last is the one that matters most:

  1 EVERY FIGURE IN THE REPORT appears in the results JSON it was taken from. Each table
    row is parsed back out of the markdown and matched against the cells.
  2 EVERY SETTING THE REPORT CLAIMS is on every cell, in the server's own BANNER, and in
    the resolved service and model configurations the server wrote -- the sample rate and
    the fraction it resolved to, the S3 destination, the object size, the feature count,
    the session count, the intra-op width, the per-session concurrency, the spinning
    policy, the mini-batch size and the piece arithmetic, and the absence of a candidate
    model.
  3 THE AGGREGATES ARE RE-COMPUTED from the per-minute CloudWatch buckets the collector
    kept, so a weighted mean cannot be wrong in the same direction twice.
  4 A CONTROL TEST proves the checks can fail: fabricated cells carrying a derived
    concurrency, a spinning pool, the wrong session count and logging switched off must
    all be caught. A verifier that cannot fail is not evidence.

Nothing here reads `report.py`. The two derive the same numbers from the same inputs
independently, which is the only arrangement where agreement means anything.
"""
import json
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
RUN = "minibatch-size-cpu-c9g-10M-20M"
RAW = BD / "results" / RUN / "raw"
SUMMARY = BD / "results" / RUN / "summary.md"

# What this round was specified with. Written here as literals, independently of the
# sweep, so a driver that drifted from the instructions is caught rather than echoed.
WANT = dict(
    sample_rate=0.2,
    sample_every=5,
    log_rows_per_object=50000,
    log_store="Amazon S3",
    sessions_per_request=1,
    concurrency_per_session=64,
    cores=64,
    n_features=210,
    allow_spinning=False,
    ort="1.28.0",
    instance="c9g.16xlarge",
    warmup=300,
    duration=900,
    segments=15,
    settle=60,
)
SESSION_COUNTS = {4, 8}
# rows -> the mini-batch sizes specified for that packaging.
WANT_MINI_BATCHES = {100: [100, 50, 10, 5], 500: [500, 100, 50, 10]}
WANT_LOADS = {
    "10m": [(100, 250), (500, 50)],
    "20m": [(100, 100), (500, 20)],
}
WANT_PREFIX = "c9gmb-"

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


def banner_text(cell):
    """The server banner as one string, from the cell record and the archived log."""
    text = "\n".join(cell.get("banner") or [])
    log = RAW / "server" / str(cell.get("host")) / f"{cell['id']}.server.log"
    if log.exists():
        text += "\n" + log.read_text(errors="replace")
    return text


def service_config(cell):
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

def check_cell_settings(cell, strict_files=True):
    """Every setting this round was specified with, on one cell."""
    cid = cell["id"]
    reference = bool(cell.get("reference"))
    rows, mb = cell["rows"], cell.get("mini_batch")

    ok(cid.startswith(WANT_PREFIX),
       f"{cid}: does not carry this round's own `{WANT_PREFIX}` prefix, so its "
       f"CloudWatch ModelId could collide with another campaign's")
    ok(cell.get("instance") == WANT["instance"],
       f"{cid}: instance {cell.get('instance')!r}, expected {WANT['instance']!r}")
    ok(cell.get("cores") == WANT["cores"],
       f"{cid}: {cell.get('cores')} core(s), expected {WANT['cores']}")
    ok(cell.get("onnxruntime") == WANT["ort"],
       f"{cid}: ONNX Runtime {cell.get('onnxruntime')}, expected {WANT['ort']}")
    ok(cell.get("wire") == "features",
       f"{cid}: wire is {cell.get('wire')!r}, expected features -- this round "
       f"vectorizes on the server")
    ok(cell.get("warmup") == WANT["warmup"] and cell.get("duration") == WANT["duration"]
       and cell.get("segments") == WANT["segments"]
       and cell.get("settle") == WANT["settle"],
       f"{cid}: window is {cell.get('warmup')}+{cell.get('duration')}s/"
       f"{cell.get('segments')} settle {cell.get('settle')}s, expected "
       f"{WANT['warmup']}+{WANT['duration']}/{WANT['segments']} settle {WANT['settle']}")

    # ---- the parallelism
    ok(cell.get("sessions") in SESSION_COUNTS,
       f"{cid}: {cell.get('sessions')} session(s), expected one of {sorted(SESSION_COUNTS)}")
    ok(cell.get("sessions_per_request") == WANT["sessions_per_request"],
       f"{cid}: leases {cell.get('sessions_per_request')} session(s) per request, "
       f"expected {WANT['sessions_per_request']} -- a request's mini batches share one")
    ok(cell.get("sessions") and cell.get("intra")
       and cell["sessions"] * cell["intra"] == WANT["cores"],
       f"{cid}: {cell.get('sessions')} session(s) x {cell.get('intra')} thread(s) is not "
       f"the {WANT['cores']}-core budget")
    ok(cell.get("spin") == 0,
       f"{cid}: spin {cell.get('spin')}, expected 0 -- idle threads sleep in this round")
    ok(cell.get("parallel_mini_batches") == WANT["cores"] // cell["intra"],
       f"{cid}: parallel_mini_batches {cell.get('parallel_mini_batches')} is not "
       f"{WANT['cores']}/{cell['intra']}")

    # ---- the axis
    if reference:
        ok(mb is None and cell.get("mini_batch_configured") is None,
           f"{cid}: the unsplit reference carries mini_batch {mb!r}; it must configure "
           f"none at all, which is what makes its plan `whole request`")
        ok(cell.get("concurrency_per_session") is None,
           f"{cid}: the reference carries concurrency_per_session "
           f"{cell.get('concurrency_per_session')}; with no mini batch there is nothing "
           f"for it to apply to")
        ok(cell.get("plan") == "whole request",
           f"{cid}: plan {cell.get('plan')!r}, expected 'whole request'")
        ok(cell.get("pieces") == 1 and not cell.get("split"),
           f"{cid}: the reference reports {cell.get('pieces')} piece(s)")
    else:
        ok(mb in WANT_MINI_BATCHES.get(rows, []),
           f"{cid}: mini batch {mb} is not one of the sizes specified for {rows} rows, "
           f"{WANT_MINI_BATCHES.get(rows)}")
        ok(cell.get("mini_batch_configured") == mb,
           f"{cid}: configured mini_batch_size {cell.get('mini_batch_configured')} "
           f"!= the axis value {mb}")
        ok(cell.get("pieces") == -(-rows // mb),
           f"{cid}: {cell.get('pieces')} piece(s) is not ceil({rows}/{mb})")
        ok(cell.get("last_piece_rows") == rows - mb * (cell["pieces"] - 1),
           f"{cid}: last piece {cell.get('last_piece_rows')} rows is wrong for "
           f"{rows} rows at mini batch {mb}")
        ok(cell.get("concurrency_per_session") == WANT["concurrency_per_session"],
           f"{cid}: concurrency_per_session {cell.get('concurrency_per_session')}, "
           f"expected {WANT['concurrency_per_session']} -- held constant so the "
           f"mini-batch size is the only variable in a group")
        ok(cell.get("concurrent_mini_batches") == WANT["concurrency_per_session"],
           f"{cid}: the session reports {cell.get('concurrent_mini_batches')} concurrent "
           f"mini batches, expected the configured "
           f"{WANT['concurrency_per_session']}")
        ok(cell.get("plan") == "mini batches, shared session",
           f"{cid}: plan {cell.get('plan')!r}, expected 'mini batches, shared session'")
        ok(cell.get("mini_batches_in_flight")
           == min(cell["pieces"], WANT["concurrency_per_session"]),
           f"{cid}: {cell.get('mini_batches_in_flight')} in flight is not "
           f"min({cell['pieces']}, {WANT['concurrency_per_session']})")

    # ---- the log
    ok(cell.get("sample_rate") == WANT["sample_rate"],
       f"{cid}: sample_rate {cell.get('sample_rate')}, expected {WANT['sample_rate']}")
    ok(cell.get("sample_every") == WANT["sample_every"],
       f"{cid}: the rate resolved to 1 in {cell.get('sample_every')}, expected "
       f"1 in {WANT['sample_every']}")
    ok(cell.get("log_rows_per_object") == WANT["log_rows_per_object"],
       f"{cid}: {cell.get('log_rows_per_object')} rows per object, expected "
       f"{WANT['log_rows_per_object']}")
    ok(cell.get("log_batch") == WANT["log_rows_per_object"] // rows,
       f"{cid}: log_batch {cell.get('log_batch')} is not "
       f"{WANT['log_rows_per_object']}/{rows}")
    ok(cell.get("log_store") == WANT["log_store"],
       f"{cid}: log store {cell.get('log_store')!r}, expected {WANT['log_store']!r}")
    # A CELL THAT SHED IS NOT A FAULT TO REJECT -- it is a measurement of the log path
    # under load. What IS a fault is a report that ranks such a cell silently, which is
    # checked against the report in `check_report`. Here all that is required is that the
    # counts were captured at all rather than left unknown.
    for field in ("log_shed_batches", "log_write_failures", "log_objects"):
        ok(cell.get(field) is not None,
           f"{cid}: no {field} was captured, so the report cannot say whether this cell "
           f"paid the full cost of logging")

    # ---- the resolved service configuration, which is the server's own copy
    svc = service_config(cell)
    if svc is None:
        if strict_files:
            fail(f"{cid}: no archived service configuration to check the settings against")
    else:
        ok(g(svc, "inference_log", "sample_rate") == WANT["sample_rate"],
           f"{cid}: the resolved configuration says sample_rate "
           f"{g(svc, 'inference_log', 'sample_rate')}")
        ok(str(g(svc, "inference_log", "uri", default="")).startswith("s3://"),
           f"{cid}: the resolved configuration's log uri is not an s3:// location")
        ok(g(svc, "inference_log", "batch_size") == cell.get("log_batch"),
           f"{cid}: the resolved log batch_size "
           f"{g(svc, 'inference_log', 'batch_size')} != {cell.get('log_batch')}")
        ok(g(svc, "threading", "sessions_per_model") == cell["sessions"],
           f"{cid}: the resolved configuration says sessions_per_model "
           f"{g(svc, 'threading', 'sessions_per_model')}, cell says {cell['sessions']}")
        ok(g(svc, "threading", "intra_op_threads") == cell["intra"],
           f"{cid}: the resolved configuration says intra_op_threads "
           f"{g(svc, 'threading', 'intra_op_threads')}, cell says {cell['intra']}")
        ok(g(svc, "threading", "allow_spinning") is False,
           f"{cid}: the resolved configuration says allow_spinning "
           f"{g(svc, 'threading', 'allow_spinning')}, expected false")
        ok(g(svc, "threading", "sessions_per_request", default=1)
           == WANT["sessions_per_request"],
           f"{cid}: the resolved configuration leases "
           f"{g(svc, 'threading', 'sessions_per_request', default=1)} session(s) per "
           f"request")
        if reference:
            ok("concurrency_per_session" not in (svc.get("threading") or {}),
               f"{cid}: the reference's resolved configuration carries "
               f"concurrency_per_session")
        else:
            ok(g(svc, "threading", "concurrency_per_session")
               == WANT["concurrency_per_session"],
               f"{cid}: the resolved configuration says concurrency_per_session "
               f"{g(svc, 'threading', 'concurrency_per_session')}, expected "
               f"{WANT['concurrency_per_session']}")
        ok("candidate_model" not in svc,
           f"{cid}: the resolved configuration carries a candidate model")

    mdl = model_config(cell)
    if mdl is None:
        if strict_files:
            fail(f"{cid}: no archived model configuration")
    else:
        order = g(mdl, "vectorization_config", "feature_order", default=[])
        ok(len(order) == WANT["n_features"],
           f"{cid}: the model configuration names {len(order)} feature(s), expected "
           f"{WANT['n_features']}")
        if reference:
            ok("mini_batch_size" not in mdl,
               f"{cid}: the reference's model configuration carries mini_batch_size "
               f"{mdl.get('mini_batch_size')}")
        else:
            ok(mdl.get("mini_batch_size") == mb,
               f"{cid}: the model configuration says mini_batch_size "
               f"{mdl.get('mini_batch_size')}, expected {mb}")
            ok(not mdl.get("is_fixed"),
               f"{cid}: the model configuration pads the last piece (is_fixed), which "
               f"would score rows nobody asked for")

    # ---- the banner, which is the server saying what it did
    hay = banner_text(cell)
    if not hay.strip():
        if strict_files:
            fail(f"{cid}: no banner recorded, so no setting can be read back from the server")
        return
    want_log = (f"logging 1 request in {WANT['sample_every']} "
                f"({100.0 / WANT['sample_every']:.1f}%)")
    ok(want_log in hay, f"{cid}: the banner does not report {want_log!r}")
    ok("inference logs -> s3://" in hay and "Amazon S3" in hay,
       f"{cid}: the banner does not report an Amazon S3 inference log")
    ok(f"every {cell.get('log_batch')} batches" in hay,
       f"{cid}: the banner does not report an object every "
       f"{cell.get('log_batch')} batches")
    ok(f"<- {WANT['n_features']} features transformed and concatenated" in hay,
       f"{cid}: the banner does not show the {WANT['n_features']}-feature wire")
    ok("A/B" not in hay, f"{cid}: the banner reports a candidate model")
    ok("spin   : idle intra-op and inter-op threads sleep" in hay,
       f"{cid}: the banner does not report sleeping threads")
    ok(f"x {cell['sessions']} ONNX Runtime session(s)" in hay,
       f"{cid}: the banner does not report {cell['sessions']} session(s)")
    ok("in groups of" not in hay,
       f"{cid}: the banner reports a session group, so a request leased more than one "
       f"session and its mini batches did not share one")
    ok("one session each" not in hay,
       f"{cid}: the banner reports SessionPerMiniBatch; this round shares one session")
    ok(f"plan     : {cell.get('plan')} -- " in hay,
       f"{cid}: the banner does not report the {cell.get('plan')!r} plan")
    want_overlap = (f"overlap  : {cell.get('concurrent_mini_batches')} mini batch(es) at "
                    f"once per session, {cell.get('parallel_mini_batches')} on distinct "
                    f"compute unit(s)")
    ok(want_overlap in hay, f"{cid}: the banner does not report {want_overlap!r}")
    if not reference:
        want_cfg = (f"{WANT['concurrency_per_session']} configured by "
                    f'"threading.concurrency_per_session"')
        ok(want_cfg in hay,
           f"{cid}: the banner does not say the concurrency was CONFIGURED -- a server "
           f"that derived it would have overlapped only "
           f"{cell.get('parallel_mini_batches')} pieces while the report claims "
           f"{WANT['concurrency_per_session']}")
        ok(f"rows cut into mini batches of {mb}," in hay,
           f"{cid}: the banner does not describe mini batches of {mb}")


# ---------------------------------------------------------------- 3 · the aggregates

def check_cloudwatch(cell):
    """Re-computes the weighted means from the per-minute buckets the collector kept."""
    cid = cell["id"]
    for metric in ("InferenceTime", "VectorizationTime"):
        block = g(cell, "cloudwatch", metric)
        if not block:
            fail(f"{cid}: no CloudWatch {metric}")
            continue
        buckets = block.get("per_minute") or []
        ok(len(buckets) == block.get("buckets"),
           f"{cid} {metric}: {len(buckets)} bucket(s) kept, {block.get('buckets')} claimed")
        ok(len(buckets) == WANT["duration"] // 60,
           f"{cid} {metric}: {len(buckets)} whole-minute bucket(s) inside a "
           f"{WANT['duration']}s window, expected {WANT['duration'] // 60}")
        ok(sum(b["n"] for b in buckets) == block.get("samples"),
           f"{cid} {metric}: bucket sample counts do not sum to {block.get('samples')}")
        pairs = [(b["n"], b["p50_ms"]) for b in buckets if b.get("p50_ms") is not None]
        if pairs:
            mine = sum(n * v for n, v in pairs) / sum(n for n, _ in pairs)
            theirs = block.get("p50_ms_bucket_mean")
            ok(theirs is not None and abs(mine - theirs) < 5e-4,
               f"{cid} {metric}: re-derived weighted p50 {mine:.6f} != reported "
               f"{theirs}")


# ---------------------------------------------------------------- 1 · the report

NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_tables(text):
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
    tables = parse_tables(text)
    ok(len(tables) >= 10, f"the report has {len(tables)} table(s), expected at least 10")

    allowed = set()
    for c in cells:
        res = c.get("result") or {}
        for value in (res.get("p50"), res.get("p95"), res.get("p99"),
                      res.get("achieved")):
            if value is not None:
                allowed.add(f"{value:.2f}")
                allowed.add(f"{value:.1f}")
        for value in (g(c, "cloudwatch", "InferenceTime", "p50_ms_bucket_mean"),
                      g(c, "cloudwatch", "VectorizationTime", "p50_ms_bucket_mean"),
                      g(c, "cloudwatch", "ToTensorTime", "p50_ms_bucket_mean")):
            if value is not None:
                allowed.add(f"{value:.3f}")
        for value in (g(c, "server_samples", "cpu_pct_sched"),
                      g(c, "server_samples", "cpu_pct_tick"),
                      g(c, "server_samples", "memory_pct")):
            if value is not None:
                allowed.add(f"{value:.1f}")

    # A SHORT CELL MUST NEVER BE RANKED.
    for header, body in tables:
        if "result" not in header or "p50" not in header:
            continue
        ri, pi = header.index("result"), header.index("p50")
        for r in body:
            if r[ri].startswith(("FAIL", "partial")):
                ok(r[pi] == "--",
                   f"the report prints p50 {r[pi]} for an unsustained cell ({r[ri]}); an "
                   f"unsustained cell reports its achieved rate and no latency")

    # Every per-cell latency in `Every cell` is one of the recorded ones.
    for header, body in tables:
        if header[:2] != ["cell", "model"] or "p50" not in header:
            continue
        for r in body:
            for col in ("p50", "p95", "p99"):
                if col not in header:
                    continue
                value = r[header.index(col)]
                if NUMBER.match(value):
                    ok(value in allowed,
                       f"the report's {col} {value} in row {r[:3]} is not a recorded "
                       f"figure")

    sustained = sum(1 for c in cells if c.get("sustained"))
    ok(f"{len(cells)} measured, {sustained} sustained" in text,
       f"the configuration table does not say '{len(cells)} measured, "
       f"{sustained} sustained'")

    # DISCLOSURE around the log path.
    shed = [c for c in cells if c.get("log_shed_batches")]
    lost = [c for c in cells if c.get("log_write_failures")]
    for c in shed:
        ok(str(c["log_shed_batches"]) in text,
           f"{c['id']}: shed {c['log_shed_batches']} log batch(es), and the report does "
           f"not print that number anywhere")
    for c in lost:
        ok(str(c["log_write_failures"]) in text,
           f"{c['id']}: {c['log_write_failures']} inference-log write(s) failed, and the "
           f"report does not print that number")
    if shed or lost:
        ok("shed" in text.lower(),
           "cells shed log batches, but the report has no column naming it")

    # THE GAIN COLUMN, re-divided. A ratio is the report's main claim, so it is not taken
    # on trust: each is recomputed from the two cells it came from.
    by = {}
    for c in cells:
        if c.get("repeat"):
            continue
        by[(c["size"], c["rows"], c["tps"], c["sessions"],
            "none" if c.get("reference") else str(c.get("mini_batch")))] = c
    for header, body in tables:
        if "gain" not in header or "mini batch" not in header:
            continue
        gi, mi = header.index("gain"), header.index("mini batch")
        for r in body:
            if not r[gi].endswith("x"):
                continue
            # The group is whichever one holds a baseline matching this row's cells; the
            # table is per group, so find the cell by its mini-batch label and latency.
            want = float(r[gi][:-1])
            pi = header.index("p50")
            if not NUMBER.match(r[pi]):
                continue
            here = float(r[pi])
            match = [c for k, c in by.items()
                     if k[4] == r[mi] and (c.get("result") or {}).get("p50") == here]
            if not match:
                continue
            c = match[0]
            base = by.get((c["size"], c["rows"], c["tps"], c["sessions"],
                           str(c["rows"])))
            if not (base and base.get("sustained")):
                continue
            got = (base["result"]["p50"] / here)
            ok(abs(got - want) < 0.015,
               f"the report's gain {want}x for {c['id']} re-divides to {got:.2f}x")


# ---------------------------------------------------------------- the grid

def check_grid(cells):
    ids = [c["id"] for c in cells]
    ok(len(set(ids)) == len(ids), f"duplicate cell id(s) among {len(ids)}")
    for size, want in WANT_LOADS.items():
        got = sorted({(c["rows"], c["tps"]) for c in cells if c["size"] == size})
        ok(got == sorted(want), f"{size}: loads {got}, specified {sorted(want)}")
    # Every specified cell was measured, at both session counts.
    for size, loads in WANT_LOADS.items():
        for rows, tps in loads:
            for sess in sorted(SESSION_COUNTS):
                got = sorted({c["mini_batch"] for c in cells
                              if c["size"] == size and c["rows"] == rows
                              and c["tps"] == tps and c["sessions"] == sess
                              and not c.get("reference") and not c.get("repeat")},
                             reverse=True)
                ok(got == sorted(WANT_MINI_BATCHES[rows], reverse=True),
                   f"{size} {rows}x{tps} s{sess}: mini batches {got}, specified "
                   f"{sorted(WANT_MINI_BATCHES[rows], reverse=True)}")
                ok(any(c for c in cells
                       if c["size"] == size and c["rows"] == rows and c["tps"] == tps
                       and c["sessions"] == sess and c.get("reference")),
                   f"{size} {rows}x{tps} s{sess}: no unsplit reference cell")
    # One comparison group never spans two machines.
    for size, loads in WANT_LOADS.items():
        for rows, tps in loads:
            hosts = {c["host"] for c in cells if c["size"] == size
                     and c["rows"] == rows and c["tps"] == tps}
            ok(len(hosts) == 1,
               f"{size} {rows}x{tps} was measured on {sorted(hosts)}; a mini-batch "
               f"comparison must not span two instances")


# ---------------------------------------------------------------- 4 · the control test

def control_test():
    """Proves the settings checks can fail, on cells built to be wrong."""
    before = len(FAILURES)
    base = dict(
        id=WANT_PREFIX + "control-10m-r100t250-mb5-s8", size="10m", wire="features",
        rows=100, tps=250, sessions=8, intra=8, sessions_per_request=1, spin=0,
        mini_batch=5, mini_batch_configured=5, pieces=20, split=True, last_piece_rows=5,
        concurrency_per_session=64, concurrent_mini_batches=64,
        parallel_mini_batches=8, mini_batches_in_flight=20,
        plan="mini batches, shared session", reference=False,
        instance="c9g.16xlarge", cores=64, onnxruntime="1.28.0",
        sample_rate=0.2, sample_every=5, log_rows_per_object=50000, log_batch=500,
        log_store="Amazon S3", log_shed_batches=0, log_write_failures=0, log_objects=10,
        warmup=300, duration=900, segments=15, settle=60,
        banner=[
            "  spin   : idle intra-op and inter-op threads sleep",
            "  plan     : mini batches, shared session -- rows cut into mini batches of 5,"
            " scored at the same time on the leased session, x",
            "  overlap  : 64 mini batch(es) at once per session, 8 on distinct compute unit(s)",
            '  overlap  : 64 configured by "threading.concurrency_per_session", x',
            "  <- 210 features transformed and concatenated",
            "hushar: threads -> 32 async, 8 concurrent request(s) x 1 session(s) each x 8"
            " intra-op x 8 ONNX Runtime session(s) = 64 compute on 64 core(s)",
            "inference logs -> s3://b/k [Amazon S3, 16 prefix shard(s), every 500 batches,"
            " 4 in flight, logging 1 request in 5 (20.0%)]",
        ],
        host=None,
    )
    check_cell_settings(dict(base), strict_files=False)
    caught = len(FAILURES) - before
    ok(caught == 0,
       f"the control test's CLEAN cell produced {caught} failure(s); the checks reject a "
       f"correct cell: {FAILURES[before:]}")
    del FAILURES[before:]

    for what, mutation in (
        ("a derived rather than configured concurrency",
         dict(concurrency_per_session=8, concurrent_mini_batches=8)),
        ("the overlap silently capped at the pool width",
         dict(concurrent_mini_batches=8)),
        ("spinning left on", dict(spin=1)),
        ("a session group leased per request", dict(sessions_per_request=2)),
        ("a session count outside the specified set", dict(sessions=2, intra=32)),
        ("a budget that does not spend the host", dict(intra=4)),
        ("logging switched off", dict(sample_rate=0.0, sample_every=0)),
        ("the previous round's 50% rate", dict(sample_rate=0.5, sample_every=2)),
        ("the default log batch", dict(log_batch=5000)),
        ("a local log destination", dict(log_store="local filesystem")),
        ("a mini-batch size not in the specification", dict(mini_batch=7)),
        ("piece arithmetic that does not follow", dict(pieces=4)),
        ("the pre-vectorized wire", dict(wire="tensor")),
        ("a short measured window", dict(duration=300)),
        ("an uncaptured shed count", dict(log_shed_batches=None)),
        ("an uncaptured object count", dict(log_objects=None)),
        ("the wrong engine", dict(onnxruntime="1.28.3")),
        ("the wrong instance type", dict(instance="c8g.16xlarge")),
    ):
        mark = len(FAILURES)
        check_cell_settings(dict(base, **mutation), strict_files=False)
        if len(FAILURES) == mark:
            fail(f"CONTROL TEST: a cell with {what} was NOT caught, so this verifier "
                 f"cannot detect that fault in a real cell")
        else:
            del FAILURES[mark:]

    # And the reference cell's own shape.
    ref = dict(base, id=WANT_PREFIX + "control-ref", reference=True, mini_batch=None,
               mini_batch_configured=None, concurrency_per_session=None,
               concurrent_mini_batches=8, pieces=1, split=False, last_piece_rows=100,
               mini_batches_in_flight=1, plan="whole request",
               banner=[l for l in base["banner"]
                       if "plan     :" not in l and "configured by" not in l]
               + ["  plan     : whole request -- a request of any size is one engine call"
                  " of that many rows"])
    mark = len(FAILURES)
    check_cell_settings(dict(ref), strict_files=False)
    ok(len(FAILURES) == mark,
       f"the control test's CLEAN REFERENCE cell produced {len(FAILURES) - mark} "
       f"failure(s): {FAILURES[mark:]}")
    del FAILURES[mark:]
    for what, mutation in (
        ("a mini batch on the unsplit reference", dict(mini_batch=100,
                                                       mini_batch_configured=100)),
        ("a concurrency on the unsplit reference", dict(concurrency_per_session=64)),
    ):
        mark = len(FAILURES)
        check_cell_settings(dict(ref, **mutation), strict_files=False)
        if len(FAILURES) == mark:
            fail(f"CONTROL TEST: a reference cell with {what} was NOT caught")
        else:
            del FAILURES[mark:]


def main():
    path = RAW / "results.json"
    if not path.exists():
        sys.exit(f"missing {path}; run collect.py first")
    cells = json.loads(path.read_text())
    if not cells:
        sys.exit("no cells in the results file")
    if not SUMMARY.exists():
        sys.exit(f"missing {SUMMARY}; run report.py first")
    text = SUMMARY.read_text()

    print(f"{len(cells)} cell(s) on {len({c['host'] for c in cells})} host(s)")
    check_grid(cells)
    for c in cells:
        check_cell_settings(c)
        check_cloudwatch(c)
    check_report(cells, text)
    control_test()

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
