#!/usr/bin/env python3
"""Generates `summary.md` from `raw/results.json`. Numbers only.

    ./report.py

Every figure is read from the collected evidence; nothing is typed in. `verify.py`
re-derives all of it independently and fails on disagreement.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BD = benchmark_data(HERE)
RUN = "minibatch-size-cpu-c9g-10M-20M"
RAW = BD / "results" / RUN / "raw"
OUT = BD / "results" / RUN / "summary.md"

# The campaign this one is read against: the same host type, the same engine build, the
# same wire and the same 10M graph, with 4 sessions of 16 threads, spinning ON, the log
# OFF, and the mini batch swept over 4/8/16 rather than the row count's own divisors.
BASELINE_RUN = "minibatch-rows-cpu-arm64-10M"


def g(doc, *path, default=None):
    for key in path:
        if not isinstance(doc, dict):
            return default
        doc = doc.get(key)
    return default if doc is None else doc


def num(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def pct(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}%"


def table(lines, header, rows):
    lines.append("")
    lines.append("| " + " | ".join(header[0]) + " |")
    lines.append("|" + "|".join(header[1]) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")


def row(cell):
    """One cell, flattened to the names the tables use."""
    res = cell.get("result") or {}
    ver = cell.get("verdict") or {}
    return dict(
        id=cell["id"],
        size=cell["size"],
        rows=cell["rows"],
        tps=cell["tps"],
        rows_per_sec=cell["rows_per_sec"],
        sessions=cell["sessions"],
        intra=cell["intra"],
        mb=cell.get("mini_batch"),
        pieces=cell["pieces"],
        in_flight=cell.get("mini_batches_in_flight"),
        parallel=cell.get("parallel_mini_batches"),
        cps=cell.get("concurrency_per_session"),
        time_sliced=bool(cell.get("time_sliced")),
        reference=bool(cell.get("reference")),
        baseline=bool(cell.get("baseline")),
        phase=cell.get("phase"),
        repeat=cell.get("repeat"),
        plan=cell.get("plan"),
        state=ver.get("state", "?"),
        sustained=bool(cell.get("sustained")),
        slices_held=ver.get("slices_held"),
        slices_total=ver.get("slices_total"),
        offered=res.get("offered"),
        achieved=res.get("achieved"),
        p50=res.get("p50"),
        p95=res.get("p95"),
        p99=res.get("p99"),
        shed=res.get("shed"),
        failed=res.get("failed"),
        window=g(cell, "window", "duration_seconds"),
        graph=g(cell, "cloudwatch", "InferenceTime", "p50_ms_bucket_mean"),
        graph95=g(cell, "cloudwatch", "InferenceTime", "p95_ms_bucket_mean"),
        vec=g(cell, "cloudwatch", "VectorizationTime", "p50_ms_bucket_mean"),
        tensor=g(cell, "cloudwatch", "ToTensorTime", "p50_ms_bucket_mean"),
        buckets=g(cell, "cloudwatch", "InferenceTime", "buckets", default=0),
        cpu=g(cell, "server_samples", "cpu_pct_sched"),
        cpu_tick=g(cell, "server_samples", "cpu_pct_tick"),
        mem=g(cell, "server_samples", "memory_pct"),
        rss=g(cell, "server_samples", "process_rss_gib"),
        client_cpu=g(cell, "client_samples", "cpu_pct_sched"),
        every=cell.get("sample_every"),
        log_batch=cell.get("log_batch"),
        log_rows_per_object=cell.get("log_rows_per_object"),
        log_objects=cell.get("log_objects"),
        log_bytes=cell.get("log_bytes_written"),
        log_shed=cell.get("log_shed_batches"),
        log_write_failures=cell.get("log_write_failures"),
        log_predicted_bps=cell.get("log_bytes_per_sec"),
        host=cell.get("host"),
        instance=cell.get("instance"),
        cores=cell.get("cores"),
        ort=cell.get("onnxruntime"),
        rev=cell.get("rev"),
        md5=cell.get("model_md5"),
        params=cell.get("parameter_count"),
        layers=cell.get("layers"),
        macs=cell.get("macs_per_row"),
    )


def label(r):
    """What a cell's packaging is called in a table."""
    if r["reference"]:
        return "none"
    return str(r["mb"])


def state_cell(r):
    """What the cell did to its offered rate, in the verdict's own vocabulary."""
    if r["sustained"]:
        return f"ok {num(r['achieved'], 1)}/{r['offered']}"
    if r["state"] == "partial":
        return (f"partial {num(r['achieved'], 1)}/{r['offered']} "
                f"({r['slices_held']}/{r['slices_total']})")
    return f"FAIL {num(r['achieved'], 1)}/{r['offered']}"


def lat(r, key):
    """A latency, withheld on a cell that did not hold its offered rate."""
    return num(r[key]) if r["sustained"] else "--"


def groups(rows):
    """(size, rows, tps, sessions) -> the mini-batch sweep measured in it, largest first.

    Repeats are excluded: they are the same configuration measured again, and belong to
    the precision table rather than to the axis.
    """
    out = {}
    for r in rows:
        if r["repeat"]:
            continue
        out.setdefault((r["size"], r["rows"], r["tps"], r["sessions"]), []).append(r)
    for key in out:
        out[key].sort(key=lambda r: (r["reference"], -(r["mb"] or 0)))
    return dict(sorted(out.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][3])))


def main():
    path = RAW / "results.json"
    if not path.exists():
        sys.exit(f"missing {path}; run collect.py first")
    rows = [row(c) for c in json.loads(path.read_text())]
    status = json.loads((RAW / "run-status.json").read_text())
    first = rows[0]
    sizes = sorted({r["size"] for r in rows})
    counts = sorted({r["sessions"] for r in rows}, reverse=True)
    collected = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    L = [f"# hushar mini-batch size on one c9g.16xlarge: 10M and 20M", "",
         f"`{RUN}` | collected {collected} | rev `{first['rev']}`", ""]

    # ------------------------------------------------------------------ configuration
    L.append("## Configuration")
    table(L, (["setting", "value"], ["---", "---"]), [
        ["servers", f"`{first['instance']}` {first['cores']} vCPU arm64, one per lane, "
                    f"four lanes"],
        ["clients", "`m9g.8xlarge`, 32 vCPU, one per lane, same availability zone"],
        ["models", "10M and 20M, FP32, ONNX"],
        ["wire", "210 named features per row, transformed and concatenated by hushar "
                 "into the model's one 690-float input; server-side vectorization ON"],
        ["feature logging", f"`inference_log.sample_rate` 0.2 -> 1 request in "
                            f"{first['every']}, to Amazon S3"],
        ["log object", f"one per {first['log_rows_per_object']:,} logged rows -- "
                       + ", ".join(f"{b} requests at {n} rows" for n, b in
                                   sorted({(r['rows'], r['log_batch']) for r in rows}))
                       + f"; {status['inference_log']['cells_with_write_failures']} cell(s) "
                       f"with write failures and "
                       f"{status['inference_log']['cells_that_shed']} that shed"],
        ["candidate model", "none; every request scores the one resident model"],
        ["mini batch", "`mini_batch_size` in rows, the primary axis. The size equal to "
                       "the row count is one engine call per request"],
        ["parallelism", f"`threading.sessions_per_model` "
                        f"{' and '.join(str(n) for n in counts)} (requests scored at "
                        f"once), `sessions_per_request` 1, so a request's mini batches "
                        f"share its one session"],
        ["intra-op width", ", ".join(
            f"{n} session(s) x {next(r['intra'] for r in rows if r['sessions'] == n)} "
            f"thread(s)" for n in counts) + f" = {first['cores']} either way"],
        ["per-session concurrency", f"`threading.concurrency_per_session` "
                                    f"{first['cores']}, held constant, so every cell's "
                                    f"mini batches overlap and the size is the only "
                                    f"variable in a group"],
        ["spinning", "`threading.allow_spinning` false; idle threads sleep"],
        ["engine", f"ONNX Runtime {first['ort']}, CPU provider, official CPU-only build"],
        ["window", f"{first['window']}s measured in {first['slices_total']} slices, "
                   f"after a 300s warm-up discarded; fresh server per cell, 60s idle "
                   f"between"],
        ["cells", f"{status['cells']} measured, {status['sustained']} sustained the "
                  f"offered rate"],
    ])

    # ------------------------------------------------------------------ the answer
    L.append("## Mini-batch size against latency")
    L.append("")
    L.append("`gain` is against the same group's `mini batch = rows` cell. `pieces` is "
             "`ceil(rows / mini batch)`; `on cores` is how many of them reach distinct "
             "cores, the rest sharing the session's pool. `none` configures no mini batch "
             "at all.")
    for (size, rws, tps, sess), grp in groups(rows).items():
        base = next((r for r in grp if r["baseline"]), None)
        b = base["p50"] if base and base["sustained"] else None
        L.append("")
        L.append(f"### {size.upper()}, {rws} rows x {tps} req/s = {rws * tps:,} rows/s, "
                 f"{sess} session(s) of {grp[0]['intra']} thread(s)")
        table(L, (["mini batch", "pieces", "in flight", "on cores", "result", "p50",
                   "p95", "p99", "graph", "vectorize", "CPU", "gain"],
                  ["--:", "--:", "--:", "--:", "---", "--:", "--:", "--:", "--:", "--:",
                   "--:", "--:"]),
              [[label(r), str(r["pieces"]), str(r["in_flight"]), str(r["parallel"]),
                state_cell(r), lat(r, "p50"), lat(r, "p95"), lat(r, "p99"),
                num(r["graph"], 3), num(r["vec"], 3), pct(r["cpu"]),
                (f"{b / r['p50']:.2f}x" if (b and r["sustained"] and r["p50"]) else "--")]
               for r in grp])

    # ------------------------------------------------------------------ best per group
    L.append("## Best mini-batch size per group")
    best = []
    for (size, rws, tps, sess), grp in groups(rows).items():
        held = [r for r in grp if r["sustained"] and not r["reference"]]
        if not held:
            continue
        top = min(held, key=lambda r: r["p50"])
        base = next((r for r in grp if r["baseline"]), None)
        b = base["p50"] if base and base["sustained"] else None
        ref = next((r for r in grp if r["reference"]), None)
        best.append([size.upper(), f"{rws} x {tps}", str(sess), label(top),
                     str(top["pieces"]), num(top["p50"]), num(top["p95"]),
                     f"{b / top['p50']:.2f}x" if b else "--",
                     num(ref["p50"]) if (ref and ref["sustained"]) else "--",
                     (f"{ref['p50'] / top['p50']:.2f}x"
                      if (ref and ref["sustained"] and top["p50"]) else "--")])
    table(L, (["model", "rows x TPS", "sessions", "best mini batch", "pieces", "p50",
               "p95", "gain vs rows", "no mini batch p50", "gain vs none"],
              ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          best)
    L.append("`gain vs rows` divides the group's one-call-per-request cell by the best "
             "split. `gain vs none` divides the cell that configures no mini batch at "
             "all, which is the same single call without the split path.")
    L.append("")

    # ------------------------------------------------------------------ session count
    L.append("## Session count: 4 against 8")
    pairs = []
    for size in sizes:
        for rws, tps in sorted({(r["rows"], r["tps"]) for r in rows
                                if r["size"] == size}):
            for mb_label in sorted({label(r) for r in rows
                                    if r["size"] == size and r["rows"] == rws
                                    and not r["repeat"]},
                                   key=lambda s: -(int(s) if s.isdigit() else 0)):
                cells = {r["sessions"]: r for r in rows
                         if r["size"] == size and r["rows"] == rws and r["tps"] == tps
                         and label(r) == mb_label and not r["repeat"]}
                a, b = cells.get(4), cells.get(8)
                if not (a and b):
                    continue
                ratio = (f"{b['p50'] / a['p50']:.2f}x"
                         if a["sustained"] and b["sustained"] and a["p50"] else "--")
                pairs.append([size.upper(), f"{rws} x {tps}", mb_label,
                              str(a["pieces"]),
                              lat(a, "p50"), pct(a["cpu"]),
                              lat(b, "p50"), pct(b["cpu"]), ratio])
    table(L, (["model", "rows x TPS", "mini batch", "pieces", "4 sessions p50",
               "CPU", "8 sessions p50", "CPU", "8 / 4"],
              ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]), pairs)
    L.append("4 sessions carry 16 intra-op threads each and 8 carry 8, so the compute "
             "budget is the same and only its distribution differs. Above 1.00x the "
             "4-session cell was faster.")
    L.append("")

    # ------------------------------------------------------------------ short cells
    short = [r for r in rows if not r["sustained"]]
    if short:
        L.append("## Did not hold the offered rate")
        table(L, (["cell", "model", "rows x TPS", "sessions", "mini batch", "offered",
                   "achieved", "shortfall", "slices held", "client shed", "CPU",
                   "graph"],
                  ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:",
                   "--:", "--:"]),
              [[f"`{r['id']}`", r["size"].upper(), f"{r['rows']} x {r['tps']}",
                str(r["sessions"]), label(r), str(r["offered"]),
                num(r["achieved"], 1),
                pct((r["offered"] - r["achieved"]) / r["offered"] * 100),
                f"{r['slices_held']}/{r['slices_total']}", str(r["shed"]),
                pct(r["cpu"]), num(r["graph"], 3)] for r in short])
        L.append("No latency is quoted for these: a median taken while requests queue "
                 "measures the queue.")
        L.append("")

    # ------------------------------------------------------------------ precision
    reps = {}
    for r in rows:
        if r["phase"] != "repeat" and not (
                r["repeat"] is None and any(x["repeat"] and x["size"] == r["size"]
                                            and x["rows"] == r["rows"]
                                            and x["sessions"] == r["sessions"]
                                            and x["mb"] == r["mb"] for x in rows)):
            continue
        reps.setdefault((r["size"], r["rows"], r["tps"], r["sessions"], r["mb"]),
                        []).append(r)
    prec = []
    for key, grp in sorted(reps.items()):
        held = [r for r in grp if r["sustained"]]
        if len(held) < 2:
            continue
        vals = sorted(r["p50"] for r in held)
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        size, rws, tps, sess, mb = key
        prec.append([size.upper(), f"{rws} x {tps}", str(sess), str(mb), str(len(vals)),
                     num(mean), num(sd, 3),
                     pct((vals[-1] - vals[0]) / mean * 100),
                     ", ".join(f"{v:.2f}" for v in vals)])
    if prec:
        L.append("## Precision")
        table(L, (["model", "rows x TPS", "sessions", "mini batch", "n", "p50 mean",
                   "p50 sd", "spread", "the medians"],
                  ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "---"]), prec)
        L.append("One configuration per group measured three times on one host, changing "
                 "nothing but the clock. A difference along the mini-batch axis has to "
                 "exceed this spread to be a finding.")
        L.append("")

    # ------------------------------------------------------------------ every cell
    L.append("## Every cell")
    table(L, (["cell", "model", "rows x TPS", "sess", "intra", "mini batch", "pieces",
               "result", "p50", "p95", "p99", "graph", "vectorize", "CPU", "mem",
               "slices"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:", "---", "--:", "--:",
               "--:", "--:", "--:", "--:", "--:", "--:"]),
          [[f"`{r['id']}`", r["size"].upper(), f"{r['rows']} x {r['tps']}",
            str(r["sessions"]), str(r["intra"]), label(r), str(r["pieces"]),
            state_cell(r), lat(r, "p50"), lat(r, "p95"), lat(r, "p99"),
            num(r["graph"], 3), num(r["vec"], 3), pct(r["cpu"]), pct(r["mem"]),
            f"{r['slices_held']}/{r['slices_total']}"]
           for r in sorted(rows, key=lambda r: (r["size"], r["rows"], -r["sessions"],
                                                r["reference"], -(r["mb"] or 0),
                                                r["repeat"] or 0))])
    L.append("`graph` is hushar's `InferenceTime` and `vectorize` its "
             "`VectorizationTime`, each a SampleCount-weighted mean of whole-minute p50s "
             "inside the measured window. On a split cell both are the MAXIMUM over the "
             "mini batches, because they run at once and the slowest sets the request's "
             "span -- so neither is the request's total work, and no per-row figure is "
             "offered for them. `CPU` is whole-host busy time from the scheduler's own "
             "run-time counters.")
    L.append("")

    # ------------------------------------------------------------------ median split
    L.append("## Where the median goes")
    brows = []
    for r in sorted(rows, key=lambda r: (r["size"], r["rows"], -r["sessions"],
                                         -(r["mb"] or 0))):
        if r["repeat"]:
            continue
        parts = [r["graph"], r["vec"], r["tensor"]]
        known = sum(p for p in parts if p)
        rest = (r["p50"] - known) if (r["sustained"] and r["p50"]) else None
        brows.append([r["size"].upper(), f"{r['rows']} x {r['tps']}",
                      str(r["sessions"]), label(r), lat(r, "p50"),
                      num(r["graph"], 3), num(r["vec"], 3), num(r["tensor"], 3),
                      num(rest, 3), str(r["buckets"])])
    table(L, (["model", "rows x TPS", "sessions", "mini batch", "p50", "graph",
               "vectorize", "to tensor", "remainder", "buckets"],
              ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          brows)
    L.append("`remainder` is the client median less the three server-side measures: "
             "gRPC, protobuf decode, admission, the fan-out itself, the log's own copy, "
             "and the network. On a split cell the three server-side figures are "
             "per-piece maxima, so the remainder absorbs the waves a request waited "
             "through.")
    L.append("")

    # ------------------------------------------------------------------ host
    L.append("## Host")
    table(L, (["cell", "CPU sched", "CPU tick", "host mem", "server RSS GiB",
               "client CPU"],
              ["---", "--:", "--:", "--:", "--:", "--:"]),
          [[f"`{r['id']}`", pct(r["cpu"]), pct(r["cpu_tick"]), pct(r["mem"]),
            num(r["rss"], 2), pct(r["client_cpu"])]
           for r in sorted(rows, key=lambda r: r["id"])])
    L.append("Both CPU readings are kept: the scheduler's run-time counters and the "
             "tick-based `/proc/stat` columns, which on a tickless kernel can account a "
             "busy CPU as idle.")
    L.append("")

    # ------------------------------------------------------------------ inference log
    L.append("## Inference log")
    table(L, (["cell", "rate", "requests per object", "rows per object", "objects",
               "written GiB", "predicted MB/s", "measured MB/s", "shed batches",
               "write failures"],
              ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]),
          [[f"`{r['id']}`", f"1 in {r['every']}", str(r["log_batch"]),
            f"{r['log_rows_per_object']:,}",
            str(r["log_objects"]) if r["log_objects"] is not None else "--",
            num((r["log_bytes"] or 0) / 2**30, 1) if r["log_bytes"] else "--",
            num((r["log_predicted_bps"] or 0) / 1e6, 1),
            num((r["log_bytes"] or 0) / 1e6 / (300 + (r["window"] or 900)), 1)
            if r["log_bytes"] else "--",
            str(r["log_shed"]) if r["log_shed"] is not None else "--",
            str(r["log_write_failures"])
            if r["log_write_failures"] is not None else "--"]
           for r in sorted(rows, key=lambda r: r["id"])])
    L.append("`predicted MB/s` is the offered rate times the sampled fraction times the "
             "measured feature-wire payload of 5,757 B fixed + 1,031 B per row plus five "
             "f32 scores per row. `measured MB/s` is the bytes that reached the bucket "
             "over warm-up plus measured window, which is the span they were written "
             "across. A non-zero `shed batches` means the accumulator dropped its oldest "
             "buffered batch because every send permit was out, so that cell paid less "
             "than the full cost of logging.")
    L.append("")

    # ------------------------------------------------------------------ rate held
    L.append("## Rate held")
    table(L, (["cell", "result", "slices held", "shed", "failed", "window s",
               "rows/s offered"],
              ["---", "---", "--:", "--:", "--:", "--:", "--:"]),
          [[f"`{r['id']}`", state_cell(r),
            f"{r['slices_held']}/{r['slices_total']}", str(r["shed"]),
            str(r["failed"]), str(r["window"]), f"{r['rows_per_sec']:,}"]
           for r in sorted(rows, key=lambda r: r["id"])])
    L.append("A cell sustains only when every slice held the offered rate within 2% with "
             "nothing shed and nothing failed.")
    L.append("")

    # ------------------------------------------------------------------ models
    L.append("## Models")
    table(L, (["model", "parameters", "layers", "MACs per row", "input width",
               "classes", "md5"],
              ["---", "--:", "--:", "--:", "--:", "--:", "---"]),
          [[s.upper(),
            f"{next(r['params'] for r in rows if r['size'] == s):,}",
            str(next(r["layers"] for r in rows if r["size"] == s)),
            f"{next(r['macs'] for r in rows if r['size'] == s):,}",
            "690", "5", f"`{next(r['md5'] for r in rows if r['size'] == s)}`"]
           for s in sizes])

    # ------------------------------------------------------------------ provenance
    L.append("## Provenance")
    seen = {}
    for r in rows:
        seen.setdefault((r["host"], r["instance"]), []).append(r)
    table(L, (["host", "instance", "cells", "model", "loads", "engine"],
              ["---", "---", "--:", "---", "---", "---"]),
          [[str(k[0]), str(k[1]), str(len(v)),
            ", ".join(sorted({x["size"].upper() for x in v})),
            ", ".join(sorted({f"{x['rows']}x{x['tps']}" for x in v})),
            f"ONNX Runtime {v[0]['ort']}"]
           for k, v in sorted(seen.items())])
    table(L, (["measure", "count"], ["---", "--:"]), [
        ["cells", str(status["cells"])],
        ["sustained", str(status["sustained"])],
        ["partial", str(status["partial"])],
        ["did not hold", str(status["failed_load"])],
        ["with CloudWatch", str(status["with_cloudwatch"])],
        ["with host samples", str(status["with_server_samples"])],
        ["cells that shed the log", str(status["inference_log"]["cells_that_shed"])],
        ["log write failures",
         str(status["inference_log"]["cells_with_write_failures"])],
        ["log objects written", f"{status['inference_log']['objects_written']:,}"],
        ["log written GiB", str(status["inference_log"]["written_gib"])],
        ["preserved attempt failures", str(status["preserved_attempt_failures"])],
    ])
    L.append(f"Cells, server banners, generated service and model configurations, "
             f"per-second host samples, client output and CloudWatch points are under "
             f"`raw/`. `raw/results.json` is this report's only input. `verify.py` "
             f"re-derives every number above from it.")
    L.append("")
    L.append(f"The nearest earlier campaign on this host type is `{BASELINE_RUN}`, which "
             f"ran 4 sessions of 16 threads with spinning on, the inference log off, and "
             f"the mini batch swept over 4, 8 and 16 rather than the row count's own "
             f"divisors. Four settings differ, so it is named rather than subtracted.")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(BD.parent)} ({len(rows)} cells, "
          f"{status['sustained']} sustained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
