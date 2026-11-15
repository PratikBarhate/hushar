#!/usr/bin/env python3
"""Proves the grid and the per-cell configuration BEFORE any instance is launched.

    ./dryrun.py

Imports both sweeps with a fake rig, prints every cell each lane would measure, and
asserts the properties this round was specified with. It launches nothing and reaches no
network: `sweep.py` does all of its grid arithmetic at import time, so importing it with
the environment set is enough to see what it would run.

WHY THIS EXISTS. The cheapest moment to find out that a driver is going to measure the
wrong thing is before it has eight instances and four hours. Every earlier campaign in
this series found at least one of these the expensive way -- a wire that silently stayed
pre-vectorized, a session count copied from the previous grid, a log batch left at the
default. Each of those is a single assertion here.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILURES = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAILURES.append(what)


def load(name, path, env):
    """Imports a sweep under a given environment, isolated from the other lane's module."""
    for key, value in env.items():
        os.environ[key] = value
    # Both lanes define a module called `hosts`; the Neuron sweep imports it by name from
    # its own directory, so the path has to lead rather than follow.
    sys.path.insert(0, str(path.parent))
    for stale in ("hosts",):
        sys.modules.pop(stale, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


BASE = dict(
    HUSHAR_BENCH_BUCKET="hushar-bench-REDACTED-us-west-2",
    REV="dryrun",
    RUN="featurewire-log20-g5-trn1-10M-20M",
)


def gpu_lane(size):
    env = dict(BASE, MODEL_SIZE=size, RIG=f"g5-1=10.0.0.1",
               MODEL_MD5={"10m": "5539c97131c896db6c58088d9f9b9ee3",
                          "20m": "30f9c5330b808a34b55fe88048a20da6"}[size])
    return load(f"gpu_sweep_{size}", HERE / "gpu" / "sweep.py", env)


def neuron_lane(size):
    env = dict(BASE, MODEL_SIZE=size, RIG=f"trn1-1=10.0.0.2")
    return load(f"neuron_sweep_{size}", HERE / "neuron" / "sweep.py", env)


# ---------------------------------------------------------------- what was specified

# Straight from the round's instructions, so the assertions below compare the driver with
# the request rather than with itself.
WANT_GPU = {
    "10m": [(100, 1000), (500, 200)],
    "20m": [(100, 750), (100, 720), (100, 700), (500, 150)],
}
WANT_TRN1 = {
    "10m": [(100, 1000), (500, 200)],
    "20m": [(100, 450), (100, 400), (500, 80), (500, 75)],
}
WANT_SESSIONS = 2
WANT_RATE = 0.2
WANT_EVERY = 5
# The cell-id prefixes, which are also the CloudWatch ModelId. They must not
# match the 50% round's, because that round is the control this one subtracts
# against and a shared prefix would average the two into one series.
WANT_PREFIX = {"a10g": "g5l20-", "trainium1": "trn1l20-"}


def shadow_scan():
    """Refuses a sweep where a local variable shadows a module-level function.

    WHY THIS IS A GATE AND NOT A STYLE NOTE. `run_cell` assigns a local `tail` in its
    client-failure path. When a module-level `tail()` helper was added for redaction,
    Python bound the name local for the WHOLE function, so the call earlier in the body
    raised `UnboundLocalError` -- after the cell had already served its full 20-minute
    window. Four lanes lost a cell each to it. Importing the module cannot catch it,
    because the line only runs at the end of a real cell; this reads the AST instead.
    """
    import ast
    bad = []
    for lane in ("gpu", "neuron"):
        src = HERE / lane / "sweep.py"
        tree = ast.parse(src.read_text())
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            assigned = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(sub, (ast.For, ast.comprehension)):
                    target = getattr(sub, "target", None)
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id in funcs and sub.func.id in assigned):
                    bad.append(f"{lane}/sweep.py: {node.name}() calls {sub.func.id}() "
                               f"and also assigns a local named {sub.func.id}")
    for message in sorted(set(bad)):
        check(False, message)
    if not bad:
        check(True, "no module-level function is shadowed by a local in either sweep")


def main():
    shadow_scan()
    print()
    print("GPU lane (g5.2xlarge)\n")
    for size in ("10m", "20m"):
        m = gpu_lane(size)
        cells = list(m.main_cells({}))
        hushar = [c for c in cells if c["stack"] == "hushar"]
        triton = [c for c in cells if c["stack"] == "triton"]
        print(f"{size}: {len(cells)} cell(s) "
              f"({len(hushar)} hushar, {len(triton)} triton)")
        for c in cells:
            extra = (f"mb={c['mini_batch_configured']} log_batch={c['log_batch']} "
                     f"rate={c['sample_rate']} obj={c['log_object_bytes'] // 1048576}MiB"
                     if c["stack"] == "hushar" else "no log, pre-vectorized tensor")
            print(f"    {c['id']:<34} {c['rows']:>4}r x {c['tps']:>4}/s  "
                  f"{c['parallelism_kind'][:3]}={c['parallelism']}  "
                  f"wire={c['wire']:<8} {extra}")
        print()
        check([(c["rows"], c["tps"]) for c in hushar] == WANT_GPU[size],
              f"g5 {size} loads are {WANT_GPU[size]}")
        check(all(c["wire"] == "features" for c in hushar),
              f"g5 {size} hushar cells use the 210-feature wire")
        check(all(c["parallelism"] == WANT_SESSIONS for c in cells),
              f"g5 {size} every cell is at {WANT_SESSIONS} sessions/instances")
        check(all(c["sessions_per_request"] == 1 for c in hushar),
              f"g5 {size} leases one session per request")
        check(all(c["pieces"] == 1 and not c["split"] for c in cells),
              f"g5 {size} scores every request in one call")
        check(all(c["mini_batch_configured"] is None for c in hushar),
              f"g5 {size} configures no mini_batch_size at all")
        check(all(c["sample_rate"] == WANT_RATE for c in hushar),
              f"g5 {size} logs at {WANT_RATE}")
        check(all(c["sample_every"] == WANT_EVERY
                  and abs(c["sample_pct"] - 100.0 / WANT_EVERY) < 1e-9
                  for c in hushar),
              f"g5 {size} resolves to one request in {WANT_EVERY}, "
              f"{100.0 / WANT_EVERY:.1f}%")
        check(all(c["log_rows_per_object"] == 50000 for c in hushar),
              f"g5 {size} writes one object per 50,000 rows")
        check(all(40 <= c["log_object_bytes"] / 1048576 <= 70 for c in hushar),
              f"g5 {size} objects are 40-70 MiB")
        check(all(c["log_store"] == "Amazon S3" for c in hushar),
              f"g5 {size} logs to S3")
        check(all(c["accelerator"] == "a10g" for c in cells),
              f"g5 {size} cells are tagged a10g")
        check(all(c["log_batch"] is None and c["sample_rate"] is None
                  for c in triton),
              f"g5 {size} Triton cells carry no log settings")
        check(len({c["id"] for c in cells}) == len(cells),
              f"g5 {size} cell ids are unique")
        check(all(c["id"].startswith(WANT_PREFIX["a10g"]) for c in cells),
              f"g5 {size} cell ids start with {WANT_PREFIX['a10g']}")

    print("\nTrainium lane (trn1.2xlarge)\n")
    for size in ("10m", "20m"):
        m = neuron_lane(size)
        cells = list(m.main_cells())
        print(f"{size}: {len(cells)} cell(s)")
        for c in cells:
            print(f"    {c['id']:<30} {c['rows']:>4}r x {c['tps']:>4}/s  "
                  f"s={c['sessions']} rpc={c['replicas_per_core']} "
                  f"cpr={c['cores_per_replica']} ctx={c['contexts']} "
                  f"mb={c['mini_batch']} wire={c['wire']} "
                  f"log_batch={c['log_batch']} rate={c['sample_rate']} "
                  f"obj={c['log_object_bytes'] // 1048576}MiB")
        print()
        check([(c["rows"], c["tps"]) for c in cells] == WANT_TRN1[size],
              f"trn1 {size} loads are {WANT_TRN1[size]}")
        check(all(c["wire"] == "features" for c in cells),
              f"trn1 {size} uses the 210-feature wire")
        check(m.N_FEATURES == 210, f"trn1 {size} expects 210 features in the banner")
        check(all(c["sessions"] == WANT_SESSIONS for c in cells),
              f"trn1 {size} every cell is at {WANT_SESSIONS} replicas")
        check(all(c["cores_per_replica"] == 1 for c in cells),
              f"trn1 {size} uses one core per replica")
        check(all(c["contexts"] == 1 for c in cells),
              f"trn1 {size} uses one execution context per replica")
        check(all(c["replicas_per_core"] == 1 for c in cells),
              f"trn1 {size} places one replica per core, so both cores are busy")
        check(all(c["mini_batch"] == c["rows"] and c["pieces"] == 1 for c in cells),
              f"trn1 {size} scores every request in one NEFF call")
        check(all(c["sessions_per_request"] == 1 for c in cells),
              f"trn1 {size} leases one replica per request")
        check(all(c["sample_rate"] == WANT_RATE for c in cells),
              f"trn1 {size} logs at {WANT_RATE}")
        check(all(c["sample_every"] == WANT_EVERY
                  and abs(c["sample_pct"] - 100.0 / WANT_EVERY) < 1e-9
                  for c in cells),
              f"trn1 {size} resolves to one request in {WANT_EVERY}, "
              f"{100.0 / WANT_EVERY:.1f}%")
        check(all(c["log_rows_per_object"] == 50000 for c in cells),
              f"trn1 {size} writes one object per 50,000 rows")
        check(all(c["accelerator"] == "trainium1" for c in cells),
              f"trn1 {size} cells are tagged trainium1")
        check(all(c["id"].startswith(WANT_PREFIX["trainium1"]) for c in cells),
              f"trn1 {size} cell ids start with {WANT_PREFIX['trainium1']}")
        check(m.SHAPES == sorted({r for r, _ in WANT_TRN1[size]}),
              f"trn1 {size} needs NEFF shapes {sorted({r for r, _ in WANT_TRN1[size]})}")

    # No cell id may collide ACROSS the two lanes: both write into one S3 prefix, so a
    # collision would silently overwrite a measurement with another accelerator's.
    every = []
    for size in ("10m", "20m"):
        every += [c["id"] for c in gpu_lane(size).main_cells({})]
        every += [c["id"] for c in neuron_lane(size).main_cells()]
    print()
    check(len(set(every)) == len(every),
          f"all {len(every)} cell ids are unique across both lanes and both sizes")
    check(len(every) == 18, f"the round is 18 cells, not {len(every)}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
