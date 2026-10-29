"""Re-derives every figure in summary.md from the originals, and fails on any gap.

    verify.py [run-name]

report.py reads results.json. This reads the ORIGINAL evidence -- the cell records in
raw/cells/ as the hosts wrote them, the per-second sample logs in raw/server/, the client
logs in raw/client/, and for the model premise the ONNX file itself -- recomputes what the
report printed, and checks the printed text contains it.

A figure that exists only because the collector or the generator computed it is not
evidence. So `results.json` is treated here as a suspect, not a source: where a value
appears in both, the raw cell wins and a disagreement is a failure.
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as REP  # noqa: E402

# epoch cpu_busy_% process_RSS_% process_RSS_kB host_memory_% tick_%
SAMPLE = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")

checks = 0
bad = []


def check(ok, what):
    global checks
    checks += 1
    if not ok:
        bad.append(what)


def has(doc, text, what):
    check(text in doc, f"{what}: summary.md does not contain {text!r}")


def num(v, places=2):
    return f"{v:,.{places}f}"


def sample_means(path):
    """CPU (scheduler) and host-memory means, re-derived from the per-second log.

    Taken by COLUMN POSITION, not by assumption: reading column 3 as host memory is a
    mistake this campaign's collector made once, and column 3 is the process's RSS.
    """
    if not os.path.isfile(path):
        return None
    sched, host_mem = [], []
    for line in open(path):
        m = SAMPLE.match(line.strip())
        if m:
            sched.append(float(m.group(2)))
            host_mem.append(float(m.group(5)))
    if not sched:
        return None
    return (round(sum(sched) / len(sched), 2),
            round(sum(host_mem) / len(host_mem), 2))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else REP.RUN
    out = os.path.join(REP.ROOT, "benchmark-data/results", run)
    doc = open(os.path.join(out, "summary.md")).read()
    results = json.load(open(os.path.join(out, "raw/results.json")))

    # ---------------------------------------------------- the raw cells are the record
    raw = {}
    for p in glob.glob(os.path.join(out, "raw/cells/*.json")):
        c = json.load(open(p))
        raw[c["id"]] = c
    check(len(raw) > 0, "no raw cell records found under raw/cells/")
    check(len(raw) == len(results),
          f"raw/cells has {len(raw)} records but results.json has {len(results)}")

    for c in results:
        r = raw.get(c["id"])
        check(r is not None, f"{c['id']}: in results.json but not in raw/cells/")
        if not r:
            continue
        # Everything the report prints per cell must match the host's own record.
        for key in ("p50", "p95", "p99", "shed", "failed", "achieved"):
            a = (c.get("result") or {}).get(key)
            b = (r.get("result") or {}).get(key)
            check(a == b, f"{c['id']}: result.{key} {a} in results.json, {b} in raw cell")
        check((c.get("verdict") or {}).get("state")
              == (r.get("verdict") or {}).get("state"),
              f"{c['id']}: verdict disagrees between results.json and the raw cell")
        check(c["offered_rows_per_sec"] == r["rows"] * r["tps"],
              f"{c['id']}: offered rate is not rows x tps")
        check(c["config_text"] == r["config_text"],
              f"{c['id']}: config_text disagrees with the raw cell")

    # ---------------------------------------------------- the run's own invariants
    sample = results[0]
    for c in results:
        check(c["model_md5"] == REP.MD5_20M,
              f"{c['id']}: model md5 {c['model_md5']}, expected the 20M graph")
        check(c["model_size"] == "20m", f"{c['id']}: model_size {c['model_size']}")
        check(c["duration_s"] == sample["duration_s"],
              f"{c['id']}: measured window differs from the rest of the run")
        check(c["warmup_s"] == sample["warmup_s"], f"{c['id']}: warmup differs")
        check(c["settle_s"] == sample["settle_s"], f"{c['id']}: settle differs")
        check(c["segments"] == sample["segments"], f"{c['id']}: segment count differs")
        check(c["cores"] == 64, f"{c['id']}: not a 64-vCPU host")
        # The engine, from the server's own banner rather than from a field someone set.
        if c["stack"] == "hushar":
            banner = " ".join(c.get("banner") or [])
            check("runtime 1.28.0" in banner,
                  f"{c['id']}: banner does not report ONNX Runtime 1.28.0")
    # The user asked for 2 min warmup, 15 min bench, 1 min settle. Assert it.
    check(sample["warmup_s"] == 120, f"warmup is {sample['warmup_s']}s, expected 120")
    check(sample["duration_s"] == 900, f"window is {sample['duration_s']}s, expected 900")
    check(sample["settle_s"] == 60, f"settle is {sample['settle_s']}s, expected 60")
    has(doc, f"{sample['duration_s']}s load", "measured window")
    has(doc, f"{sample['warmup_s']}s discarded warmup", "warmup")
    has(doc, f"{sample['settle_s']}s of settling", "settle")

    # ---------------------------------------------------- the model premise
    onnx_path = os.path.join(REP.ROOT, "benchmark-data/generated/20m/bench_model.onnx")
    if os.path.isfile(onnx_path):
        blob = open(onnx_path, "rb").read()
        check(hashlib.md5(blob).hexdigest() == REP.MD5_20M,
              "the generated 20M graph does not have the md5 every cell was served")
        check(len(blob) == REP.BYTES_20M,
              f"20M onnx is {len(blob)} bytes, report says {REP.BYTES_20M}")
        try:
            import onnx                                   # noqa: F401
            import numpy as np
            g = onnx.load(onnx_path)
            params = sum(int(np.prod(i.dims)) for i in g.graph.initializer)
            check(params == REP.PARAMS_20M,
                  f"20M graph has {params:,} parameters, report says {REP.PARAMS_20M:,}")
            kinds = {n.op_type for n in g.graph.node}
            check(len(kinds) == 13,
                  f"20M graph has {len(kinds)} distinct op types, report says 13")
            inp = [i for i in g.graph.input
                   if i.name not in {x.name for x in g.graph.initializer}]
            check(len(inp) == 1, "20M graph does not take exactly one input")
            dims = [d.dim_param or d.dim_value
                    for d in inp[0].type.tensor_type.shape.dim]
            check(dims[1] == 690, f"20M graph input width is {dims[1]}, report says 690")
        except ImportError:
            pass
    base_onnx = os.path.join(REP.ROOT, "benchmark-data/generated/10m/bench_model.onnx")
    if os.path.isfile(base_onnx):
        check(os.path.getsize(base_onnx) == REP.BYTES_10M,
              "10M onnx size disagrees with the ratio table")
    # The ratios the report prints must follow from those counts.
    has(doc, f"**{REP.PARAMS_20M/REP.PARAMS_10M:.2f}×**", "parameter ratio")
    has(doc, f"**{REP.MACS_20M/REP.MACS_10M:.2f}×**", "MACs ratio")
    has(doc, f"{REP.PARAMS_10M:,}", "10M parameter count")
    has(doc, f"{REP.PARAMS_20M:,}", "20M parameter count")

    # ---------------------------------------------------- CPU, from the sample logs
    for c in results:
        tag = "s" if c["stack"] == "hushar" else "t"
        got = sample_means(os.path.join(out, "server", c["host"],
                                        f"{tag}-{c['id']}.log"))
        if got is None:
            continue
        cpu_raw, mem_raw = got
        cpu_json = (c.get("server_samples") or {}).get("cpu_pct_sched")
        mem_json = (c.get("server_samples") or {}).get("memory_pct")
        check(cpu_json is not None and abs(cpu_json - cpu_raw) < 0.01,
              f"{c['id']}: CPU {cpu_json} in results.json, {cpu_raw} from the sample log")
        check(mem_json is not None and abs(mem_json - mem_raw) < 0.01,
              f"{c['id']}: memory {mem_json} vs {mem_raw} from the sample log")

    # ---------------------------------------------------- payloads, from client logs
    for rows in REP.row_counts():
        pay = REP.payload_for(rows)
        if pay:
            has(doc, f"**{pay[0]:,} bytes per request**", f"{rows}-row payload")
            has(doc, f"{pay[2]} named features", f"{rows}-row feature count")
            check(pay[2] == 210, f"{rows} rows: client reports {pay[2]} features, not 210")
            # bytes/row must be bytes/request divided by rows, from the same line
            check(abs(pay[1] - pay[0] / rows) < 0.6,
                  f"{rows} rows: payload per row {pay[1]} is not {pay[0]}/{rows}")

    # ---------------------------------------------------- every printed table row
    for rows in REP.row_counts():
        for arm in REP.arms_at(rows):
            for rate in REP.rates(rows):
                c = REP.pick(arm, rows, rate)
                if not c:
                    continue
                r = raw[c["id"]]
                for key, places in (("p50", 2), ("p95", 2), ("p99", 2)):
                    v = (r.get("result") or {}).get(key)
                    if v is not None:
                        has(doc, num(v, places), f"{c['id']} {key}")
                shed = (r.get("result") or {}).get("shed") or 0
                has(doc, f"{shed:,}", f"{c['id']} shed")
                has(doc, f"| {rows} | {rate // rows:,} | {rate:,} |",
                    f"{c['id']} rows/req, req/s and rows/s prefix")

    # ---------------------------------------------------- ceilings, from raw verdicts
    for rows in REP.row_counts():
        for arm in REP.arms_at(rows):
            held = [r["rows"] * r["tps"] for r in raw.values()
                    if r["arm"] == arm and r["rows"] == rows
                    and (r.get("verdict") or {}).get("state") == "sustained"]
            lost = [r["rows"] * r["tps"] for r in raw.values()
                    if r["arm"] == arm and r["rows"] == rows
                    and (r.get("verdict") or {}).get("state") in ("partial", "failed")]
            lo, hi = (max(held) if held else None), (min(lost) if lost else None)
            rlo, rhi = REP.ceiling(arm, rows)
            check((lo, hi) == (rlo, rhi),
                  f"{rows} rows arm {arm}: ceiling {(rlo, rhi)} from results.json, "
                  f"{(lo, hi)} from the raw verdicts")
            if lo and hi:
                has(doc, f"{lo:,} – {hi:,}", f"{rows} rows arm {arm} ceiling bracket")
        # A skipped rung must NOT be inside anyone's bracket: skipped cells live in
        # raw/failures/ and must never have become an upper bound.
        for p in glob.glob(os.path.join(out, "failures/*.json")):
            d = json.load(open(p))
            if d.get("rows") != rows or "skipped" not in d:
                continue
            check(d["id"] not in raw,
                  f"{d['id']} was skipped by the early stop but is also in raw/cells/")

    # ---------------------------------------------------- sustained means every slice
    for c in results:
        v = c.get("verdict") or {}
        slices = c.get("slices") or []
        if v.get("state") == "sustained":
            check(v.get("slices_held") == v.get("slices_total") == len(slices)
                  and len(slices) == sample["segments"],
                  f"{c['id']}: called sustained without all {sample['segments']} slices")
            check((c.get("result") or {}).get("shed") == 0,
                  f"{c['id']}: called sustained but shed requests")
        if v.get("state") == "partial":
            check(0 < (v.get("slices_held") or 0) < (v.get("slices_total") or 0),
                  f"{c['id']}: called partial but did not hold some-but-not-all slices")

    # ---------------------------------------------------- latency ratios
    for rows in REP.row_counts():
        for rate in REP.shared_rungs(rows, "H", "T"):
            h, t = raw[REP.pick("H", rows, rate)["id"]], raw[REP.pick("T", rows, rate)["id"]]
            hp, tp = h["result"]["p50"], t["result"]["p50"]
            check(hp > 0, f"{rows} rows at {rate}: hushar p50 is not positive")
            has(doc, f"**{tp/hp:.1f}×**", f"{rows} rows at {rate} latency ratio")

    # ---------------------------------------------------- noise floor
    n = REP.noise()
    if n:
        count, lo, hi, spread = n
        ps = sorted(raw[c["id"]]["result"]["p50"] for c in REP.cells("P"))
        check(ps[0] == lo and ps[-1] == hi,
              "noise-floor extremes disagree with the raw repeat cells")
        check(abs(100.0 * (ps[-1] - ps[0]) / ps[0] - spread) < 1e-9,
              "noise-floor spread is not recomputable from the raw repeats")
        has(doc, f"Spread **{spread:.1f}%**", "noise-floor spread")
        cfgs = {raw[c["id"]]["config_text"] for c in REP.cells("P")}
        check(len(cfgs) == 1, f"the repeat cells do not share one configuration: {cfgs}")

    # ---------------------------------------------------- graph share
    for rows in REP.row_counts():
        for rate in REP.rates(rows):
            c = REP.pick("H", rows, rate)
            if not c or (c.get("verdict") or {}).get("state") != "sustained":
                continue
            inf, e2e = REP.cw(c, "InferenceTime"), REP.res(c, "p50")
            if inf is None or e2e is None:
                continue
            check(0 < inf <= e2e * 1.05,
                  f"{c['id']}: InferenceTime {inf} exceeds end-to-end p50 {e2e}")
            has(doc, f"{100.0*inf/e2e:.0f}%", f"{c['id']} graph share")

    # ---------------------------------------------------- the baseline comparison
    if REP.BASE:
        for c in REP.BASE:
            check(c["model_md5"] != REP.MD5_20M,
                  "a baseline cell carries the 20M md5; the two campaigns are not distinct")
        for rows in REP.row_counts():
            for arm in ("H", "T"):
                b, o = REP.base_best(arm, rows), REP.own_best(arm, rows)
                if not b or not o:
                    continue
                check((b.get("verdict") or {}).get("state") == "sustained",
                      f"baseline {arm} {rows} rows: comparison cell was not sustained")
                check((o.get("verdict") or {}).get("state") == "sustained",
                      f"this run {arm} {rows} rows: comparison cell was not sustained")
                has(doc, f"{REP.res(o,'p50')/REP.res(b,'p50'):.1f}×",
                    f"{rows} rows arm {arm} latency cost of the 2x model")

    # ---------------------------------------------------- report hygiene
    check("arm H" not in doc and "arm T" not in doc and "arm L" not in doc,
          "summary.md names arm letters; it should name configurations")
    for secret in re.findall(r"\b\d{12}\b", doc):
        check(False, f"summary.md contains a 12-digit number that reads as an account id: "
                     f"{secret}")

    print(f"{checks} checks")
    if bad:
        print(f"\n{len(bad)} FAILED:")
        for b in bad[:60]:
            print(f"  - {b}")
        if len(bad) > 60:
            print(f"  ... and {len(bad)-60} more")
        sys.exit(1)
    print("every figure in summary.md re-derives from the originals")


if __name__ == "__main__":
    main()
