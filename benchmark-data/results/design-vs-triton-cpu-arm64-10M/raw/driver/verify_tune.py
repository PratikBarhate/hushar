#!/usr/bin/env python3
"""Re-derives every figure in the tuning summary from the originals, and fails on any gap.

    verify_tune.py <out-run> <in-run> [<in-run> …]

The report reads each pass's results.json. This reads the ORIGINAL cell records in
raw/cells/ and the per-second sample logs in raw/server/, re-computes what the report
printed, and checks that every number in summary.md appears where it should. A figure that
only exists because the collector or the generator computed it is not evidence.

It imports the report's own ranking so the two cannot drift apart -- but it re-derives the
INPUTS to that ranking from the originals, which is the part that matters. The ranking rule
has been wrong twice; both times this file is what said so.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_tune as R  # noqa: E402

SAMPLE = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")

checks = 0
bad = []


def check(ok, what):
    global checks
    checks += 1
    if not ok:
        bad.append(what)


def originals(run):
    d = os.path.join(R.RESULTS, run, "raw/cells")
    out = {}
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as handle:
                cell = json.load(handle)
            out[cell["id"]] = (run, cell)
    return out


def sample_cpu(path):
    """The scheduler CPU mean, re-derived from the per-second log."""
    if not os.path.isfile(path):
        return None
    values = []
    for line in open(path):
        m = SAMPLE.match(line.strip())
        if m:
            values.append(float(m.group(2)))
    return round(sum(values) / len(values), 2) if values else None


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: verify_tune.py <out-run> <in-run> [<in-run> …]")
    out_run, in_runs = sys.argv[1], sys.argv[2:]

    results, cells = [], {}
    for run in in_runs:
        results += R.load(run)
        cells.update(originals(run))
    with open(os.path.join(R.RESULTS, out_run, "summary.md")) as handle:
        doc = handle.read()

    check(len(results) == len(cells),
          f"results.json files hold {len(results)} cells, raw/cells holds {len(cells)}")

    for r in results:
        cid = r["id"]
        if cid not in cells:
            bad.append(f"{cid} is in results.json but not raw/cells")
            continue
        run, o = cells[cid]

        # 1 - the collector must not have altered what the host recorded
        for field in ("result", "slices", "verdict", "config", "config_text", "wire",
                      "stack", "rows", "tps", "offered_rows_per_sec", "model_md5", "rev"):
            check(r.get(field) == o.get(field), f"{cid}: {field} differs from the original")

        # 2 - the verdict must FOLLOW from the slices, not be asserted
        v, res = o["verdict"], o["result"]
        held = [s["shed"] == 0 and s["failed"] == 0
                and s["achieved"] is not None and s["achieved"] >= 0.98 * res["offered"]
                for s in o["slices"]]
        check(v["slices_held"] == sum(held), f"{cid}: slices_held disagrees with the slices")
        whole = (res["shed"] == 0 and res["failed"] == 0
                 and res["achieved"] >= 0.98 * res["offered"])
        want = "sustained" if (held and all(held) and whole) else (
            "partial" if any(held) else "failed")
        check(v["state"] == want, f"{cid}: state {v['state']}, re-derived {want}")
        check(len(o["slices"]) == o["segments"],
              f"{cid}: {len(o['slices'])} slices, {o['segments']} configured")

        # 3 - CPU must come from the sample log, not the cell's own one-off figure
        tag = "s" if o["stack"] == "hushar" else "t"
        path = os.path.join(R.RESULTS, run, f"raw/server/{o['host']}/{tag}-{cid}.log")
        got = sample_cpu(path)
        check(got is not None, f"{cid}: no server sample log at {os.path.basename(path)}")
        if got is not None:
            check((r.get("server_samples") or {}).get("cpu_pct_sched") == got,
                  f"{cid}: cpu_pct_sched disagrees with the sample log")

        # 4 - the configuration the report names must be the one the server CONFIRMED
        banner = "\n".join(o.get("banner") or [])
        if o["stack"] == "hushar":
            want_spin = "spin" if o["config"]["spin"] else "sleep"
            check(f"idle intra-op and inter-op threads {want_spin}" in banner,
                  f"{cid}: banner does not confirm spinning {want_spin}")
            n = 210 if o["wire"] == "features" else 1
            expect = ("<- [features] transformed and concatenated" if n == 1
                      else f"<- {n} features transformed and concatenated")
            check(expect in banner, f"{cid}: banner does not confirm the {o['wire']} wire")
        else:
            check("backend check: onnxruntime" in banner,
                  f"{cid}: banner does not confirm the onnxruntime backend")
            check(f"max_batch_size {o['config']['max_batch']}" in banner,
                  f"{cid}: banner does not confirm max_batch_size")
            check(f"model md5 {o['model_md5']}" in banner,
                  f"{cid}: Triton did not confirm the model md5")

        # 5 - the row must appear with the traffic it actually offered, as one contiguous
        #     prefix, so a rows/req or req/s borrowed from another cell cannot pass
        check(f"| `{o['config_text']}` | {o['rows']} | {o['tps']} |" in doc,
              f"{cid}: no table row stating {o['rows']} rows/req at {o['tps']} req/s")
        check(f"{res['p50']:.2f}" in doc, f"{cid}: p50 {res['p50']:.2f} missing")
        check(o["rows"] * o["tps"] == o["offered_rows_per_sec"],
              f"{cid}: {o['rows']} x {o['tps']} != {o['offered_rows_per_sec']} rows/s")

    # 6 - the named winners must be what the report's OWN rules produce, and the rules must
    #     be the ones the prose claims. Both have been wrong; this is the check that said so.
    high = max(c["offered_rows_per_sec"] for c in results)
    row_counts = sorted({c["rows"] for c in results})
    for rows in row_counts:
        for stack, label in (("hushar", "hushar"), ("triton", "Triton")):
            pool = [c for c in results
                    if c["stack"] == stack and c["rows"] == rows
                    and c["offered_rows_per_sec"] == high]
            ordered = R.rank(pool)
            top = ordered[0]
            tied = [c for c in ordered
                    if c["verdict"]["sustained"] and not R.pathological(c)
                    and c["result"]["p50"] <= top["result"]["p50"] * (1 + R.TIE_BAND)]
            winner = min(tied, key=R.complexity)
            check(f"| {rows} | {winner['tps']} | `{winner['config_text']}` |" in doc
                  or f"`{winner['config_text']}` | {winner['result']['p50']:.2f} |" in doc,
                  f"the answer table does not name {label}'s winner at {rows} rows "
                  f"({winner['config_text']!r})")
            check(winner["verdict"]["sustained"],
                  f"{label}'s winner at {rows} rows did not sustain its rate")
            # a tie must be reported as a tie, never as a victory
            if len(tied) > 1:
                check(f"{len(tied)} hushar and" in doc or "tie band" in doc,
                      f"{label} at {rows} rows has {len(tied)} tied configurations and "
                      f"the report does not say so")

    # 7 - the tail guard must actually exclude something, or it is decoration
    flagged = [c for c in results if c["verdict"]["sustained"] and R.pathological(c)]
    check(flagged, "no cell trips the tail guard; the rule is untested on this data")
    check("**tail**" in doc, "the report never marks a cell as tail-limited")

    # 8 - the stated counts must match
    held_total = sum(1 for c in results if c["verdict"]["sustained"])
    check(f"{held_total} of {len(results)} held" in doc,
          f"the report does not state {held_total} of {len(results)} held")

    # 9 - no account identifier anywhere in the report
    check(not re.search(r"\b\d{12}\b", doc), "the report contains a 12-digit number")

    print(f"{checks} checks over {len(in_runs)} pass(es), {len(results)} cells")
    if bad:
        print(f"\n{len(bad)} FAILED:")
        for b in bad[:40]:
            print(f"  - {b}")
        return 1
    print("every displayed figure re-derives from the original cell records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
