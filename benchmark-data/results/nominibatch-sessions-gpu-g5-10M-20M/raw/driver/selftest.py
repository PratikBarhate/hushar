#!/usr/bin/env python3
"""Exercises report.py and verify.py on realistic cells, before any GPU hour is spent.

    ./selftest.py

report.py and verify.py are new code for this run, and the moment they are needed is
after a nine-hour grid has finished -- which is the worst moment to find out a table
cannot be parsed back. So this builds a results tree whose cells are SHAPED exactly like
the ones the sweep will write, runs both scripts over it, and requires `verify.py` to
report zero failures.

The cells come from the PREVIOUS campaign's real records -- its unsplit hushar rung and
its Triton cells at 2, 4 and 8 -- rewritten into this run's geometry: the id prefix, the
`r<rows>` engine cache path, the session count, and the banner lines the session count
changes. The hushar cells at 4 and 8 sessions are clones of the 2-session one with those
fields moved, so the NUMBERS are not this run's and are never reported as such; what is
being tested is the plumbing from a cell to a table and back.

Writes under `benchmark-data/results/<run>-selftest/` and deletes it on success, so it
cannot be mistaken for a result.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BENCH = _benchmark_data(HERE)
SOURCE_RUN = "minibatch-sessions-gpu-g5-10M-20M"
RUN = "nominibatch-sessions-gpu-g5-10M-20M-selftest"
SRC = BENCH / "results" / SOURCE_RUN / "raw" / "results.json"
DST = BENCH / "results" / RUN
PREFIX = "g5nb"
# This run's grid, which is what the synthetic tree must cover.
GRID = {"10m": [(100, 1000), (500, 200)], "20m": [(100, 750), (500, 150)]}
COUNTS = (2, 4, 8)
FACTS = dict(host_tensorrt="10.16.1", host_cuda_major="13",
             host_nvidia_driver="580.95.05", host_gpu="NVIDIA A10G")


def rewrite_banner(lines, old_cache, new_cache, old_par, new_par):
    out = []
    for line in lines:
        line = line.replace(old_cache, new_cache)
        line = re.sub(r"\b\d+ concurrent request\(s\) x 1 session\(s\) each",
                      f"{new_par} concurrent request(s) x 1 session(s) each", line)
        line = re.sub(r"x \d+ ONNX Runtime session\(s\)",
                      f"x {new_par} ONNX Runtime session(s)", line)
        out.append(line)
    return out


def main():
    if not SRC.is_file():
        sys.exit(f"no {SRC}; the previous campaign's results are the input here")
    src = json.loads(SRC.read_text())

    cells = []
    for size, loads in GRID.items():
        for rows, tps in loads:
            base_h = next((c for c in src
                           if c["size"] == size and c["rows"] == rows
                           and c["tps"] == tps and c["stack"] == "hushar"
                           and c.get("mini_batch_configured") is None), None)
            if base_h is None:
                sys.exit(f"the source run has no unsplit hushar cell for "
                         f"{size} {rows}x{tps}; cannot build the fixture")
            for par in COUNTS:
                c = json.loads(json.dumps(base_h))
                old_cache, old_par = c["engine_cache"], c["parallelism"]
                new_cache = f"/root/trt-cache/{size}/r{rows}"
                c.update(
                    id=f"{PREFIX}-{c['arm']}-{size}-r{rows}t{tps}-p{par}",
                    run=RUN,
                    parallelism=par, request_lanes=par, sessions_per_request=1,
                    sessions_per_model=par,
                    mini_batch=rows, mini_batch_configured=None, pieces=1, split=False,
                    plan="whole request", plan_confirmed="whole request",
                    engine_cache=new_cache, engine_shape=[rows, 690],
                    provider=f"tensorrt:0:cache={new_cache}",
                    banner=rewrite_banner(c["banner"], old_cache, new_cache,
                                          old_par, par),
                    **FACTS)
                c["cloudwatch_model_id"] = c["id"]
                cells.append(c)

            for par in COUNTS:
                base_t = next((c for c in src
                               if c["size"] == size and c["rows"] == rows
                               and c["tps"] == tps and c["stack"] != "hushar"
                               and c["parallelism"] == par), None)
                if base_t is None:
                    continue
                c = json.loads(json.dumps(base_t))
                c.update(id=f"{PREFIX}-{c['arm']}-{size}-r{rows}t{tps}-p{par}",
                         run=RUN,
                         mini_batch=None, mini_batch_configured=None,
                         pieces=1, split=False, **FACTS)
                cells.append(c)

    if not any(c["stack"] != "hushar" for c in cells):
        sys.exit("the fixture has no Triton cell, so the pairing is untested")

    raw = DST / "raw"
    (raw / "prime").mkdir(parents=True, exist_ok=True)
    (raw / "results.json").write_text(json.dumps(cells, indent=2) + "\n")
    for size in GRID:
        (raw / "prime" / f"{size}.json").write_text(json.dumps(dict(
            size=size, run=RUN,
            shapes=[dict(rows=rows, engine_cache=f"/root/trt-cache/{size}/r{rows}",
                         engine_shape=[rows, 690], build_seconds=25.0 + rows / 1000,
                         cached_seconds=22.1, speedup=1.1)
                    for rows, _ in GRID[size]]), indent=2) + "\n")

    def status(items):
        return dict(cells=len(items),
                    sustained=sum(bool(c.get("sustained")) for c in items),
                    partial=sum((c.get("verdict") or {}).get("state") == "partial"
                                for c in items),
                    failed_load=sum((c.get("verdict") or {}).get("state") == "failed"
                                    for c in items),
                    with_cloudwatch=sum(bool((c.get("cloudwatch") or {})
                                             .get("InferenceTime")) for c in items),
                    with_gpu_samples=sum(bool(c.get("gpu_utilization"))
                                         for c in items))
    (raw / "run-status.json").write_text(json.dumps(dict(
        run=RUN, main=status(cells), probes=status([]),
        by_arm={a: status([c for c in cells if c["arm"] == a])
                for a in sorted({c["arm"] for c in cells})},
        by_size={s: status([c for c in cells if c["size"] == s])
                 for s in sorted({c["size"] for c in cells})}), indent=2) + "\n")

    env = dict(os.environ, RUN=RUN)
    rc = 0
    for script in ("report.py", "verify.py"):
        print(f"\n===== {script} =====")
        r = subprocess.run([sys.executable, str(HERE / script)], env=env,
                           capture_output=True, text=True)
        print(r.stdout.strip() or "(no output)")
        if r.stderr.strip():
            print(r.stderr.strip()[-3000:])
        rc |= r.returncode

    if rc == 0:
        print(f"\nSELFTEST OK over {len(cells)} fixture cell(s); removing {DST}")
        shutil.rmtree(DST)
    else:
        print(f"\nSELFTEST FAILED; the tree is left at {DST} for inspection")
    return rc


if __name__ == "__main__":
    sys.exit(main())
