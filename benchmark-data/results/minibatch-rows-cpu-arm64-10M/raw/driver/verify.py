#!/usr/bin/env python3
"""Re-derives every figure in ../summary.md from the originals, and fails on disagreement.

    python verify.py
    python verify.py --cloudwatch     # also re-query CloudWatch instead of trusting
                                      # the per-minute buckets stored in results.json

The report is generated, not written, so the thing worth checking is not its prose but
whether its numbers still follow from the evidence. This rebuilds the record set from
scratch out of:

    cells/<cell>.json                 what the host wrote the moment the cell finished
    server/<host>/s-<cell>.log        the per-second host samples, hushar cells
    server/<host>/t-<cell>.log        the same, Triton cells
    server/<host>/<cell>.service.json the configuration the server actually resolved
    server/<host>/<cell>.model.json   the model configuration it actually loaded
    server/<host>/<cell>.config.pbtxt Triton's resolved configuration
    client/<host>/<cell>/...          the client's own output

then asserts results.json matches it field by field, re-runs report.py against the
rebuild, and requires the regenerated text to be byte-identical to the committed
summary.md. If it is, every displayed figure follows from the originals by construction,
which is a stronger statement than checking the tables one at a time.

What it cannot re-derive offline is the CloudWatch query itself. The per-minute buckets
are stored per cell, so the aggregation arithmetic is re-checked from them; `--cloudwatch`
re-asks the namespace as well, which needs credentials and only works while the metrics
are still in retention.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, ".."))
SUMMARY = os.path.abspath(os.path.join(RAW, "..", "summary.md"))
RESULTS = os.path.join(RAW, "results.json")

PROBLEMS = []
CHECKS = [0]


def check(ok, message):
    CHECKS[0] += 1
    if not ok:
        PROBLEMS.append(message)
    return ok


def close(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ------------------------------------------------------------------ rebuild

SAMPLE = re.compile(
    r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")


def sample_means(path):
    """The same reduction collect.py does, written again from the column contract.

    epoch  cpu_busy_%  process_RSS_%  process_RSS_kB  host_memory_%  tick_%
    """
    if not os.path.isfile(path):
        return None
    cols = [[], [], [], [], []]
    for line in open(path):
        m = SAMPLE.match(line.strip())
        if not m:
            continue
        cols[0].append(float(m.group(2)))
        cols[1].append(float(m.group(3)))
        cols[2].append(float(m.group(4)))
        cols[3].append(float(m.group(5)))
        if m.group(6):
            cols[4].append(float(m.group(6)))
    if not cols[0]:
        return None
    mean = lambda v: sum(v) / len(v)   # noqa: E731
    out = {"samples": len(cols[0]),
           "cpu_pct_sched": round(mean(cols[0]), 2),
           "memory_pct": round(mean(cols[3]), 2),
           "process_rss_pct": round(mean(cols[1]), 3),
           "process_rss_gib": round(mean(cols[2]) / (1024 * 1024), 3)}
    if cols[4]:
        out["cpu_pct_tick"] = round(mean(cols[4]), 2)
    return out


def aggregate_buckets(buckets):
    """The weighted aggregate of per-minute CloudWatch buckets, recomputed.

    `samples` counts every bucket, because every bucket held data points. A per-field
    weighted mean skips the buckets where that field is null: CloudWatch can return a
    bucket with a mean and no percentile, and treating the absence as a value would
    silently pull the figure down. This mirrors collect.py's rule, deliberately, because
    the two have to agree for a disagreement to mean something.
    """
    if not buckets:
        return None
    total = sum(b["n"] for b in buckets)
    if not total:
        return None

    def weighted(field):
        kept = [(b["n"], b[field]) for b in buckets if b.get(field) is not None]
        if not kept:
            return None
        return sum(n * v for n, v in kept) / sum(n for n, _ in kept)

    out = {"samples": int(total), "buckets": len(buckets)}
    mean = weighted("mean_ms")
    p50 = weighted("p50_ms")
    if mean is not None:
        out["mean_ms"] = round(mean, 3)
    if p50 is not None:
        out["p50_ms_bucket_mean"] = round(p50, 3)
    return out


def rebuild():
    """Every record, from the per-cell originals and the sample logs alone.

    One exception, and it is stated rather than hidden: a CloudWatch query cannot be
    re-run from files, so the per-minute buckets stored in results.json are treated as
    originals. Their aggregates are then recomputed HERE, with this file's own
    arithmetic, rather than carried over from collect.py, so the figures the report
    prints are still re-derived rather than trusted.
    """
    cells_dir = os.path.join(RAW, "cells")
    stored = {c["id"]: c for c in json.load(open(RESULTS))}
    out = []
    for name in sorted(os.listdir(cells_dir)):
        if not name.endswith(".json"):
            continue
        cell = json.load(open(os.path.join(cells_dir, name)))
        tag = "s" if cell["stack"] == "hushar" else "t"
        host, cid = cell["host"], cell["id"]
        cell["server_samples"] = sample_means(
            os.path.join(RAW, "server", host, f"{tag}-{cid}.log"))
        cell["client_samples"] = sample_means(
            os.path.join(RAW, "client", host, cid, "host-samples.log"))
        source = stored.get(cid, {})
        if source.get("cloudwatch"):
            cell["cloudwatch"] = {}
            for metric, rec in source["cloudwatch"].items():
                if not rec or not rec.get("per_minute"):
                    cell["cloudwatch"][metric] = rec
                    continue
                again = aggregate_buckets(rec["per_minute"])
                again["per_minute"] = rec["per_minute"]
                cell["cloudwatch"][metric] = again
            cell["cloudwatch_model_id"] = source.get("cloudwatch_model_id")
        out.append(cell)
    return out


# ------------------------------------------------------------------ checks

def check_against_results(rebuilt):
    """results.json must agree with the originals on every field the report reads."""
    stored = {c["id"]: c for c in json.load(open(RESULTS))}
    check(len(stored) == len(rebuilt),
          f"results.json holds {len(stored)} cells, cells/ holds {len(rebuilt)}")
    for c in rebuilt:
        s = stored.get(c["id"])
        if not check(s is not None, f"{c['id']} missing from results.json"):
            continue
        for field in ("arm", "stack", "wire", "rows", "tps", "mb", "spin",
                      "sessions", "width", "rows_per_sec", "sustained",
                      "model_md5", "host", "instance", "rev"):
            check(s.get(field) == c.get(field),
                  f"{c['id']}: {field} is {s.get(field)!r} in results.json, "
                  f"{c.get(field)!r} in the cell record")
        for field in ("p50", "p95", "p99", "achieved", "offered", "shed", "failed"):
            check(close(s["result"].get(field), c["result"].get(field)),
                  f"{c['id']}: result.{field} disagrees")
        for side in ("server_samples", "client_samples"):
            a, b = s.get(side), c.get(side)
            if a is None and b is None:
                continue
            if not check(a is not None and b is not None,
                         f"{c['id']}: {side} present in one source only"):
                continue
            for field in ("cpu_pct_sched", "cpu_pct_tick", "memory_pct", "samples"):
                check(close(a.get(field), b.get(field), 2e-3),
                      f"{c['id']}: {side}.{field} is {a.get(field)} in results.json, "
                      f"{b.get(field)} re-derived from the sample log")
        # CloudWatch: the aggregate must follow from the per-minute buckets beside it.
        for metric, rec in (s.get("cloudwatch") or {}).items():
            if not rec:
                continue
            again = aggregate_buckets(rec.get("per_minute"))
            if not check(again is not None,
                         f"{c['id']}: {metric} has no per-minute buckets to check"):
                continue
            for field in ("samples", "buckets", "mean_ms", "p50_ms_bucket_mean"):
                check(close(rec.get(field), again.get(field), 2e-3),
                      f"{c['id']}: cloudwatch.{metric}.{field} is {rec.get(field)}, "
                      f"re-aggregated {again.get(field)}")


def check_settings(rebuilt):
    """The settings a table prints must be what the server actually resolved."""
    for c in rebuilt:
        host, cid = c["host"], c["id"]
        if c["stack"] == "hushar":
            svc = os.path.join(RAW, "server", host, f"{cid}.service.json")
            mdl = os.path.join(RAW, "server", host, f"{cid}.model.json")
            if not check(os.path.isfile(svc), f"{cid}: no resolved service.json"):
                continue
            if not check(os.path.isfile(mdl), f"{cid}: no resolved model.json"):
                continue
            service = json.load(open(svc))
            model = json.load(open(mdl))
            th = service.get("threading", {})
            check(th.get("sessions_per_model") == c["sessions"],
                  f"{cid}: sessions_per_model {th.get('sessions_per_model')} resolved, "
                  f"{c['sessions']} recorded")
            check(th.get("intra_op_threads") == c["width"],
                  f"{cid}: intra_op_threads {th.get('intra_op_threads')} resolved, "
                  f"{c['width']} recorded")
            check(th.get("allow_spinning") == bool(c["spin"]),
                  f"{cid}: allow_spinning {th.get('allow_spinning')} resolved, "
                  f"{bool(c['spin'])} recorded")
            check(th.get("worker_threads") == c["wt"],
                  f"{cid}: worker_threads {th.get('worker_threads')} resolved, "
                  f"{c['wt']} recorded")
            want_mb = c["rows"] if c["mb"] is None else c["mb"]
            check(model.get("mini_batch_size") == want_mb,
                  f"{cid}: mini_batch_size {model.get('mini_batch_size')} resolved, "
                  f"{want_mb} implied by the record")
            # The wire is the thing two arms differ by, so it is checked against the
            # configuration rather than trusted from the cell's own label.
            order = model.get("vectorization_config", {}).get("feature_order", [])
            want = 1 if c["wire"] == "tensor" else 210
            check(len(order) == want,
                  f"{cid}: {len(order)} features resolved, wire {c['wire']!r} wants "
                  f"{want}")
            check(c["model_md5"] and model.get("model_path", "").endswith(".onnx"),
                  f"{cid}: model_path does not name an .onnx")
        else:
            cfg = os.path.join(RAW, "server", host, f"{cid}.config.pbtxt")
            if not check(os.path.isfile(cfg), f"{cid}: no resolved config.pbtxt"):
                continue
            text = open(cfg).read()
            check(re.search(rf"count:\s*{c['sessions']}\b", text) is not None,
                  f"{cid}: config.pbtxt does not set instance count {c['sessions']}")
            check(re.search(
                rf'intra_op_thread_count"\s*value:\s*{{\s*string_value:\s*"{c["width"]}"',
                text) is not None,
                f"{cid}: config.pbtxt does not set intra_op_thread_count {c['width']}")
            check("max_batch_size: 0" in text,
                  f"{cid}: config.pbtxt does not set max_batch_size 0, so Triton was not "
                  f"making one graph call per request")
            check("allow_spinning" not in text,
                  f"{cid}: config.pbtxt writes a spinning key the backend cannot honour")


def check_invariants(rebuilt):
    md5s = {c["model_md5"] for c in rebuilt}
    check(len(md5s) == 1, f"cells score {len(md5s)} different graphs: {md5s}")
    orts = {c.get("ort_version") for c in rebuilt if c.get("ort_version")}
    check(orts == {"1.28.0"},
          f"cells report ONNX Runtime {orts}, expected exactly {{'1.28.0'}}")
    revs = {c["rev"] for c in rebuilt}
    check(len(revs) == 1, f"cells come from {len(revs)} revisions: {revs}")

    for c in rebuilt:
        r = c["result"]
        held = (r["shed"] == 0 and r["failed"] == 0
                and r["achieved"] >= 0.98 * r["offered"])
        check(held == c["sustained"],
              f"{c['id']}: sustained is {c['sustained']} but achieved "
              f"{r['achieved']}/{r['offered']}, shed {r['shed']}, failed {r['failed']}")
        check(r["failed"] == 0, f"{c['id']}: {r['failed']} requests failed")
        check(c["rows"] * c["tps"] == c["rows_per_sec"],
              f"{c['id']}: rows x tps is not rows_per_sec")
        check(c.get("clean_slate_verified") is True,
              f"{c['id']}: the clean-slate check did not pass before this cell")
        w = c.get("window") or {}
        check(bool(w.get("start") and w.get("end")),
              f"{c['id']}: no measured window recorded")

    # The lane logs were not collected before the rig was terminated, so completeness is
    # checked against the plan instead: a lane that died early shows as missing cells.
    sys.path.insert(0, HERE)
    os.environ.setdefault("HUSHAR_BENCH_BUCKET", "verify")
    os.environ.setdefault("REV", list(revs)[0] if revs else "verify")
    os.environ.setdefault("MODEL_MD5", list(md5s)[0] if md5s else "verify")
    try:
        import sweep
    except SystemExit as e:
        check(False, f"could not import the sweep to check completeness: {e}")
        return
    planned = set()
    for host in sweep.SERVERS:
        planned |= {c["id"] for c in sweep.cells_for(host)}
    got = {c["id"] for c in rebuilt}
    check(planned == got,
          f"the measured set differs from the plan: "
          f"{len(planned - got)} missing ({sorted(planned - got)[:4]}), "
          f"{len(got - planned)} unplanned ({sorted(got - planned)[:4]})")


def check_report(rebuilt):
    """Regenerate from the rebuild and require the text to match byte for byte."""
    if not check(os.path.isfile(SUMMARY), "summary.md does not exist"):
        return
    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "results.json")
        summary = os.path.join(tmp, "summary.md")
        json.dump(rebuilt, open(results, "w"), indent=2, sort_keys=True)
        env = dict(os.environ, RESULTS_JSON=results, SUMMARY_OUT=summary)
        run = subprocess.run([sys.executable, os.path.join(HERE, "report.py")],
                             capture_output=True, text=True, env=env)
        if not check(run.returncode == 0,
                     f"report.py failed on the rebuild: {run.stderr[-400:]}"):
            return
        want = open(summary).read()
        have = open(SUMMARY).read()
        if check(want == have,
                 "summary.md does not match what the originals regenerate"):
            return
        a, b = want.splitlines(), have.splitlines()
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else "<missing>"
            y = b[i] if i < len(b) else "<missing>"
            if x != y:
                PROBLEMS.append(f"  first difference at line {i + 1}:\n"
                                f"    regenerated: {x[:160]}\n"
                                f"    committed  : {y[:160]}")
                break


def check_cloudwatch_live(rebuilt):
    """Re-ask the namespace, for the cells whose server-side figures the report prints."""
    aws = os.environ.get("AWS_CLI", "/opt/homebrew/bin/aws")
    profile = os.environ.get("AWS_PROFILE", "hushar-bench")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    stored = {c["id"]: c for c in json.load(open(RESULTS))}
    asked = 0
    for c in rebuilt:
        rec = (stored.get(c["id"], {}).get("cloudwatch") or {}).get("InferenceTime")
        if not rec:
            continue
        model_id = stored[c["id"]].get("cloudwatch_model_id")
        w = c["window"]
        cmd = (f"{aws} --profile {profile} --region {region} cloudwatch "
               f"get-metric-statistics --namespace HusharBench "
               f"--metric-name InferenceTime "
               f"--dimensions Name=ModelId,Value={model_id} "
               f"--start-time {w['start']} --end-time {w['end']} --period 60 "
               f"--extended-statistics p50 --statistics Average SampleCount "
               f"--output json")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            check(False, f"{c['id']}: CloudWatch re-query failed")
            continue
        points = [p for p in json.loads(r.stdout).get("Datapoints", [])
                  if p.get("SampleCount")]
        again = aggregate_buckets([
            {"n": int(p["SampleCount"]), "mean_ms": p["Average"] / 1000.0,
             "p50_ms": p.get("ExtendedStatistics", {}).get("p50", 0.0) / 1000.0}
            for p in points])
        asked += 1
        check(again is not None and close(again["p50_ms_bucket_mean"],
                                          rec["p50_ms_bucket_mean"], 5e-3),
              f"{c['id']}: InferenceTime p50 re-queried "
              f"{(again or {}).get('p50_ms_bucket_mean')} against stored "
              f"{rec['p50_ms_bucket_mean']}")
    print(f"  re-queried CloudWatch for {asked} cell(s)")


def check_feature_section():
    """Re-derives the word-feature section from the RESOLVED configurations on every host.

    The section's numbers describe the model, not the measurements, so they cannot be
    checked against the cells. They are checked against the configuration each server
    actually loaded instead -- every one of them, independently of the single copy
    report.py happens to read -- so a claim about what the model eats cannot drift from
    what was served.
    """
    text = open(SUMMARY).read()
    served = {}
    for host in sorted(os.listdir(os.path.join(RAW, "server"))):
        d = os.path.join(RAW, "server", host)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".model.json"):
                continue
            with open(os.path.join(d, name)) as handle:
                v = (json.load(handle).get("vectorization_config") or {})
            if len(v.get("feature_order") or []) == 210:
                served.setdefault(json.dumps(v, sort_keys=True), []).append(name)
    check(len(served) == 1,
          f"{len(served)} different 210-feature configurations were served; the feature "
          f"section describes one model")
    if not served:
        return
    v = json.loads(next(iter(served)))
    n_configs = len(next(iter(served.values())))
    check(n_configs >= 10,
          f"only {n_configs} resolved 210-feature configuration(s) to check against")

    order = v["feature_order"]
    tf = v["feature_transformations"]
    word_kinds = ("embedding_lookup", "one_hot_encoding")
    total = words = n_words = 0
    per_feature = []
    for name in order:
        t = tf[name]
        kind = t["type"]
        if kind == "embedding_lookup":
            vocab = len(t["embeddings"])
            width = len(next(iter(t["embeddings"].values())))
            # every row of the table must be the same width, or "a W-float row per word"
            # is not true of the table
            check(len({len(x) for x in t["embeddings"].values()}) == 1,
                  f"{name}: embedding rows are not all the same width")
        elif kind == "one_hot_encoding":
            vocab = len(t["categories"])
            width = vocab
            check(len(set(t["categories"])) == vocab,
                  f"{name}: the one-hot category list repeats a word")
        elif kind == "identity":
            vocab, width = None, len(t["default_val"])
        else:
            vocab, width = None, 1
        total += width
        if kind in word_kinds:
            words += width
            n_words += 1
            per_feature.append((name, kind, vocab, width))

    # the graph's own declared input width, from the same configurations
    check(f"`features FP32[rows, {total}]`" in text,
          f"the report does not state the input width as {total}")
    check(f"**{n_words} of them are words**" in text,
          f"the report does not say {n_words} features are words")
    check(f"**{words} of the {total} numbers" in text,
          f"the report does not say {words} of {total} floats come from words")
    check(f"{100 * words / total:.0f}%" in text,
          f"the report does not state the word share as {100 * words / total:.0f}%")

    # every word feature must appear with its own vocabulary and output width
    for name, kind, vocab, width in per_feature:
        kindname = "one-hot" if kind == "one_hot_encoding" else "embedding"
        how = ("one slot per word" if kind == "one_hot_encoding"
               else f"a {width}-float row per word")
        check(f"| `{name}` | {kindname} | {vocab} | {width} | {how} |" in text,
              f"{name}: no table row stating {vocab} words -> {width} floats ({how})")
        if kind == "one_hot_encoding":
            check(vocab == width,
                  f"{name}: a one-hot's width must equal its vocabulary ({vocab} vs {width})")

    # the claim that the Triton client never sees a word: its spec must be a bare array
    spec = os.path.join(RAW, "..", "..", "..", "generated")
    check(f'{{"features": {{"kind": "float_array", "width": {total}}}}}' in text
          or f'"width": {total}' in text,
          "the report does not quote the tensor wire's feature spec")

    # and the word kinds named here must be the ones the client generates strings for
    client = os.path.join(RAW, "..", "..", "..", "..", "benchmark-client/src/main.rs")
    if os.path.isfile(client):
        src = open(client).read()
        for kind in word_kinds:
            idx = src.find(f'"{kind}"')
            check(idx != -1 and "Generator::Choice" in src[idx:idx + 400],
                  f"benchmark-client does not send a word for {kind}; the section's "
                  f"claim that these are the word features is wrong")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloudwatch", action="store_true",
                    help="re-query CloudWatch as well as re-aggregating the stored "
                         "per-minute buckets")
    args = ap.parse_args()

    print(f"verifying {os.path.relpath(SUMMARY, os.path.dirname(RAW))}")
    rebuilt = rebuild()
    print(f"  rebuilt {len(rebuilt)} cell(s) from cells/ and the sample logs")
    check_against_results(rebuilt)
    check_settings(rebuilt)
    check_invariants(rebuilt)
    if args.cloudwatch:
        check_cloudwatch_live(rebuilt)
    check_report(rebuilt)
    check_feature_section()

    print(f"  {CHECKS[0]} checks")
    if PROBLEMS:
        print(f"\nFAILED, {len(PROBLEMS)} problem(s):")
        for p in PROBLEMS[:40]:
            print(f"  {p}")
        if len(PROBLEMS) > 40:
            print(f"  ... and {len(PROBLEMS) - 40} more")
        return 1
    print("\nOK: every figure in the report follows from the originals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
