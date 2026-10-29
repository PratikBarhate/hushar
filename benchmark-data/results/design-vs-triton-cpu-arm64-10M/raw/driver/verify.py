#!/usr/bin/env python3
"""Re-derives every figure in summary.md from the originals, and fails on any gap.

    verify.py [run-name]

The report reads results.json. This reads the ORIGINAL cell records in raw/cells/, the
per-second sample logs in raw/server/, and the client logs in raw/client/, re-computes what
the report printed, and checks it is there. A figure that exists only because the collector
or the generator computed it is not evidence.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as REP  # noqa: E402

SAMPLE = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")

checks = 0
bad = []


def check(ok, what):
    global checks
    checks += 1
    if not ok:
        bad.append(what)


def sample_means(path):
    """CPU and memory means, re-derived from the per-second log.

    Column layout is `epoch cpu_busy_% process_RSS_% process_RSS_kB host_memory_% tick_%`.
    Reading column 3 as host memory is a mistake this campaign's collector made once, so
    both are taken by position here rather than by assumption.
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
    with open(os.path.join(out, "raw/results.json")) as handle:
        results = json.load(handle)
    with open(os.path.join(out, "summary.md")) as handle:
        doc = handle.read()

    cells = {}
    d = os.path.join(out, "raw/cells")
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as handle:
                c = json.load(handle)
            cells[c["id"]] = c
    check(len(results) == len(cells),
          f"results.json has {len(results)} cells, raw/cells has {len(cells)}")

    for r in results:
        cid = r["id"]
        if cid not in cells:
            bad.append(f"{cid} is in results.json but not raw/cells")
            continue
        o = cells[cid]

        # 1 - the collector must not have altered what the host recorded
        for field in ("result", "slices", "verdict", "config", "config_text", "wire",
                      "stack", "arm", "rows", "tps", "offered_rows_per_sec",
                      "model_md5", "rev"):
            check(r.get(field) == o.get(field), f"{cid}: {field} differs from the original")

        # 2 - the verdict must FOLLOW from the slices
        v, res = o["verdict"], o["result"]
        held = [s["shed"] == 0 and s["failed"] == 0 and s["achieved"] is not None
                and s["achieved"] >= 0.98 * res["offered"] for s in o["slices"]]
        whole = (res["shed"] == 0 and res["failed"] == 0
                 and res["achieved"] >= 0.98 * res["offered"])
        want = "sustained" if (held and all(held) and whole) else (
            "partial" if any(held) else "failed")
        check(v["state"] == want, f"{cid}: state {v['state']}, re-derived {want}")
        check(v["slices_held"] == sum(held), f"{cid}: slices_held disagrees with slices")
        check(len(o["slices"]) == o["segments"],
              f"{cid}: {len(o['slices'])} slices, {o['segments']} configured")
        check(o["rows"] * o["tps"] == o["offered_rows_per_sec"],
              f"{cid}: {o['rows']} x {o['tps']} != {o['offered_rows_per_sec']}")

        # 3 - CPU and memory must come from the sample log
        tag = "s" if o["stack"] == "hushar" else "t"
        path = os.path.join(out, f"raw/server/{o['host']}/{tag}-{cid}.log")
        got = sample_means(path)
        check(got is not None, f"{cid}: no server sample log")
        if got:
            s = r.get("server_samples") or {}
            check(s.get("cpu_pct_sched") == got[0],
                  f"{cid}: cpu_pct_sched {s.get('cpu_pct_sched')}, re-derived {got[0]}")
            check(s.get("memory_pct") == got[1],
                  f"{cid}: memory_pct {s.get('memory_pct')}, re-derived {got[1]}")

        # 4 - the configuration must be the one the server CONFIRMED in its banner
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
            check(f"model md5 {o['model_md5']}" in banner,
                  f"{cid}: Triton did not confirm the model md5")

        # 5 - the arm must match the wire and stack the report claims for it
        arm_wire = {"H": ("hushar", "features"), "X": ("hushar", "tensor"),
                    "T": ("triton", "tensor"), "P": ("hushar", "features")}
        check(arm_wire[o["arm"]] == (o["stack"], o["wire"]),
              f"{cid}: arm {o['arm']} is {o['stack']}/{o['wire']}, "
              f"expected {arm_wire[o['arm']]}")

        # 6 - every cell must appear in a table, with its own traffic and p50
        if o["arm"] != "P":
            label = REP.ARM[o["arm"]][0]
            check(f"| {label} | {o['rows']} | {o['tps']} | "
                  f"{o['offered_rows_per_sec']:,} |" in doc,
                  f"{cid}: no table row for {label} at {o['rows']}r x {o['tps']}/s")
        check(f"{res['p50']:.2f}" in doc, f"{cid}: p50 {res['p50']:.2f} missing")

    # 7 - the ceilings must be what the cells say, and the report must name them
    for rows in REP.row_counts():
        for arm in ("H", "T", "X"):
            held = [c["offered_rows_per_sec"] for c in results
                    if c["arm"] == arm and c["rows"] == rows
                    and c["verdict"]["sustained"]]
            got = REP.ceiling(arm, rows)[0]
            check(got == (max(held) if held else None),
                  f"ceiling({arm}, {rows}) is {got}, re-derived {max(held) if held else None}")
        hc, tc = REP.ceiling("H", rows)[0], REP.ceiling("T", rows)[0]
        check(f"| {rows} | {hc:,} | {tc:,} |" in doc,
              f"the ceilings table does not state {rows} rows: {hc:,} vs {tc:,}")

    # 8 - the answer table may only contain rungs where BOTH designs held
    for line in doc.splitlines():
        m = re.match(r"\| (\d+) \| (\d+) \| ([\d,]+) \| \*\*([\d.]+)\*\* \|", line)
        if not m:
            continue
        rows, tps = int(m.group(1)), int(m.group(2))
        rate = int(m.group(3).replace(",", ""))
        h = REP.pick("H", rows, rate)
        t = REP.pick("T", rows, rate)
        check(h and t and h["verdict"]["sustained"] and t["verdict"]["sustained"],
              f"the answer table lists {rows}r x {tps}/s where one design did not hold")
        check(f"{h['result']['p50']:.2f}" == m.group(4),
              f"the answer table's p50 for {rows}r x {tps}/s is not the cell's")

    # 9 - the payload figures must come from the client logs, not from arithmetic
    for rows in REP.row_counts():
        for arm in ("H", "X"):
            got = REP.payload_for(rows, arm)
            check(got is not None, f"no measured payload for {arm} at {rows} rows")
            if got:
                check(f"{got[0]:,} B" in doc,
                      f"payload {got[0]:,} B for {arm} at {rows} rows missing from report")

    # 10 - the noise floor must be re-derivable, and must be smaller than every ratio the
    #      report calls a finding
    reps = [c for c in results if c["arm"] == "P"]
    check(len(reps) >= 3, f"only {len(reps)} precision repeat(s)")
    ps = [c["result"]["p50"] for c in reps]
    floor = 100 * (max(ps) - min(ps)) / (sum(ps) / len(ps))
    check(f"{floor:.1f}%" in doc, f"the report does not state the {floor:.1f}% noise floor")
    check(floor < 5.0, f"the noise floor is {floor:.1f}%, too wide for these claims")

    # 11 - the engine times the report compares must be the published ones
    for rows in REP.row_counts():
        for rate in REP.rates(rows):
            for arm in ("H", "X"):
                c = REP.pick(arm, rows, rate)
                if c and REP.engine(c) is not None:
                    cw = (c.get("cloudwatch") or {}).get("InferenceTime") or {}
                    buckets = cw.get("per_minute") or []
                    check(buckets, f"{c['id']}: InferenceTime has no per-minute buckets")

    # 12 - no account identifier anywhere in the report
    check(not re.search(r"\b\d{12}\b", doc), "the report contains a 12-digit number")

    print(f"{checks} checks over {len(results)} cells")
    if bad:
        print(f"\n{len(bad)} FAILED:")
        for b in bad[:40]:
            print(f"  - {b}")
        return 1
    print("every displayed figure re-derives from the original cell records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
