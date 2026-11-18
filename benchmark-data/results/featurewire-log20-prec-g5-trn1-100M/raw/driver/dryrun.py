#!/usr/bin/env python3
"""Proves the grid and the per-cell configuration BEFORE any instance is launched.

    ./dryrun.py

Imports both sweeps at both precisions with a fake rig, prints every cell each lane would
measure, and asserts the properties this round was specified with. It launches nothing and
reaches no network: `sweep.py` does all of its grid arithmetic at import time, so
importing it with the environment set is enough to see what it would run.

WHY THIS EXISTS. The cheapest moment to find out that a driver is going to measure the
wrong thing is before it has eight instances and six hours. Every earlier campaign in this
series found at least one of these the expensive way -- a wire that silently stayed
pre-vectorized, a session count copied from the previous grid, a log batch left at the
default, a banner gate that could no longer fire. Each of those is a single assertion here.

WHAT THIS ROUND ADDS: THE PRECISION, WHICH IS THE AXIS AND SO THE THING MOST WORTH
REFUSING TO GUESS AT. Three separate ways a precision arm can be a lie are checked:

  the flag      the GPU arm's provider string must carry `trt_bf16_enable=1` on bf16 and
                must NOT carry it on fp32; the Trainium arm's `--auto-cast` must be
                `matmult` on bf16 and `none` on fp32. Neither default is the one wanted:
                TensorRT's default is no BF16, and neuronx-cc 2.26's default is matmult.
  the key       the engine cache, the NEFF cache prefix and the prime record must be
                DISJOINT between the two arms. Sharing any of them means the second arm
                restores the first's artifact, passes every other gate, and publishes a
                precision column that is wrong.
  the id        the cell id must carry the precision, because `run_cell` skips a cell S3
                already holds -- so a shared id makes the entire second arm a silent
                no-op -- and because the id IS the CloudWatch `ModelId`.
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


MD5 = "eee2644bd65cd23543ca806f102f5062"
SIZE = "100m"

BASE = dict(
    HUSHAR_BENCH_BUCKET="hushar-bench-REDACTED-us-west-2",
    REV="dryrun",
    RUN="featurewire-log20-prec-g5-trn1-100M",
)


def gpu_lane(prec):
    env = dict(BASE, MODEL_SIZE=SIZE, PRECISION=prec, RIG="g5-1=10.0.0.1",
               MODEL_MD5=MD5)
    return load(f"gpu_sweep_{prec}", HERE / "gpu" / "sweep.py", env)


def neuron_lane(prec):
    env = dict(BASE, MODEL_SIZE=SIZE, PRECISION=prec, RIG="trn1-1=10.0.0.2")
    return load(f"neuron_sweep_{prec}", HERE / "neuron" / "sweep.py", env)


# ---------------------------------------------------------------- what was specified

# Straight from the round's instructions, so the assertions below compare the driver with
# the request rather than with itself.
#
#   100M parameter f32    100 x 100   sessions 2
#                         500 x  20   sessions 2
#   100M parameter bf16   100 x 200   sessions 2
#                         500 x  40   sessions 2
#
# Both accelerators run the same four. The A10G measured all four holding in
# `nominibatch-precision-gpu-g5-100M`; one trn1.2xlarge is projected to hold none of them,
# which the round runs anyway to measure the ceiling. See neuron/hosts.py.
# KEYED BY (ACCELERATOR, PRECISION), because the two accelerators no longer run the same
# grid. Wave A is the four specified rates on both. Wave B is Trainium only: rates set from
# wave A's MEASURED ceilings so the cells sustain and report a latency, each a multiple of
# 5 at or below 80% device occupancy. The A10G held all of wave A and needs no wave B.
WANT_LOADS = {
    ("a10g", "fp32"): [(100, 100), (500, 20)],
    ("a10g", "bf16"): [(100, 200), (500, 40)],
    ("trainium1", "fp32"): [(100, 100), (500, 20), (100, 70), (500, 10)],
    ("trainium1", "bf16"): [(100, 200), (500, 40), (100, 70), (500, 10),
                            (100, 100), (500, 20)],
}
WANT_SESSIONS = 2
WANT_RATE = 0.2
WANT_EVERY = 5
WANT_PARAMS = 100_498_437
WANT_LAYERS = 31
PRECS = ("fp32", "bf16")
# The cell-id prefixes, which are also the CloudWatch ModelId. They must not match the
# 10M/20M feature-wire round's (`g5l20-`, `trn1l20-`) nor the tensor-wire precision
# round's (`g5q-`), because both are controls this round reads against and a shared prefix
# would average two configurations into one series.
WANT_PREFIX = {"a10g": "g5fwp-", "trainium1": "trn1fwp-"}
# What each arm must ask its builder for. NEITHER IS A DEFAULT.
WANT_TRT_OPTION = {"fp32": None, "bf16": "trt_bf16_enable=1"}
WANT_AUTO_CAST = {"fp32": "none", "bf16": "matmult"}


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


def compile_scan():
    """Compiles every driver module, which `ast.parse` is not enough to do.

    A REPEATED KEYWORD ARGUMENT IS NOT A PARSE ERROR. `ast.parse` accepted
    `dict(..., precision=PRECISION, ..., precision="fp32")` in gpu/sweep.py -- the
    duplicate is rejected later, when the compiler builds the call -- so a syntax check
    that stops at the AST reports a clean file and the import then fails on the host.
    """
    import py_compile
    for rel in ("gpu/sweep.py", "neuron/sweep.py", "gpu/hosts.py", "neuron/hosts.py",
                "gpu/collect.py", "neuron/collect.py", "report.py", "verify.py"):
        try:
            py_compile.compile(str(HERE / rel), doraise=True)
            ok, why = True, rel
        except py_compile.PyCompileError as error:
            ok, why = False, f"{rel}: {str(error).splitlines()[-1][:120]}"
        check(ok, f"{why} compiles")


def main():
    shadow_scan()
    compile_scan()

    gpu, trn1 = {}, {}

    print("\nA10G lane (g5.2xlarge), one precision per host\n")
    for prec in PRECS:
        m = gpu_lane(prec)
        gpu[prec] = m
        cells = list(m.main_cells({}))
        hushar = [c for c in cells if c["stack"] == "hushar"]
        triton = [c for c in cells if c["stack"] == "triton"]
        print(f"{prec}: {len(cells)} cell(s) "
              f"({len(hushar)} hushar, {len(triton)} triton)  cache={m.CACHE_ROOT}")
        for c in cells:
            extra = (f"mb={c['mini_batch_configured']} log_batch={c['log_batch']} "
                     f"rate={c['sample_rate']} obj={c['log_object_bytes'] // 1048576}MiB"
                     if c["stack"] == "hushar" else "no log, pre-vectorized tensor")
            print(f"    {c['id']:<46} {c['rows']:>4}r x {c['tps']:>4}/s  "
                  f"{c['parallelism_kind'][:3]}={c['parallelism']}  "
                  f"wire={c['wire']:<8} {extra}")
        print()
        want = WANT_LOADS[("a10g", prec)]
        check([(c["rows"], c["tps"]) for c in hushar] == want,
              f"g5 {prec} loads are {want}")
        check(all(c["wire"] == "features" for c in hushar),
              f"g5 {prec} hushar cells use the 210-feature wire")
        check(all(c["parallelism"] == WANT_SESSIONS for c in cells),
              f"g5 {prec} every cell is at {WANT_SESSIONS} sessions/instances")
        check(all(c["sessions_per_request"] == 1 for c in hushar),
              f"g5 {prec} leases one session per request")
        check(all(c["pieces"] == 1 and not c["split"] for c in cells),
              f"g5 {prec} scores every request in one call")
        check(all(c["mini_batch_configured"] is None for c in hushar),
              f"g5 {prec} configures no mini_batch_size at all")
        check(all(c["sample_rate"] == WANT_RATE for c in hushar),
              f"g5 {prec} logs at {WANT_RATE}")
        check(all(c["sample_every"] == WANT_EVERY
                  and abs(c["sample_pct"] - 100.0 / WANT_EVERY) < 1e-9
                  for c in hushar),
              f"g5 {prec} resolves to one request in {WANT_EVERY}, "
              f"{100.0 / WANT_EVERY:.1f}%")
        check(all(c["log_rows_per_object"] == 50000 for c in hushar),
              f"g5 {prec} writes one object per 50,000 rows")
        check(all(40 <= c["log_object_bytes"] / 1048576 <= 70 for c in hushar),
              f"g5 {prec} objects are 40-70 MiB")
        check(all(c["log_store"] == "Amazon S3" for c in hushar),
              f"g5 {prec} logs to S3")
        check(all(c["accelerator"] == "a10g" for c in cells),
              f"g5 {prec} cells are tagged a10g")
        check(all(c["log_batch"] is None and c["sample_rate"] is None for c in triton),
              f"g5 {prec} Triton cells carry no log settings")
        check(len({c["id"] for c in cells}) == len(cells),
              f"g5 {prec} cell ids are unique")
        check(all(c["id"].startswith(WANT_PREFIX["a10g"]) for c in cells),
              f"g5 {prec} cell ids start with {WANT_PREFIX['a10g']}")
        # ---- the model, which must be the same on both arms
        check(all(c["parameter_count"] == WANT_PARAMS and c["layers"] == WANT_LAYERS
                  for c in cells),
              f"g5 {prec} serves {WANT_PARAMS:,} parameters over {WANT_LAYERS} layers")
        # ---- the precision
        check(all(c["precision"] == prec for c in cells),
              f"g5 {prec} every cell is labelled {prec}")
        check(all(c["id"].count(f"-{prec}-") == 1 for c in cells),
              f"g5 {prec} the precision is in every cell id")
        check(m.PRECISIONS[prec]["trtexec_flag"]
              == (None if prec == "fp32" else "--bf16"),
              f"g5 {prec} trtexec flag is {None if prec == 'fp32' else '--bf16'}")
        spec = m.provider_spec(hushar[0])
        want_opt = WANT_TRT_OPTION[prec]
        check(("trt_bf16_enable=1" in spec) == (want_opt is not None),
              f"g5 {prec} provider string "
              f"{'carries' if want_opt else 'does NOT carry'} trt_bf16_enable=1")
        check(m.CACHE_ROOT.endswith(f"/{prec}"),
              f"g5 {prec} engine cache is namespaced by precision ({m.CACHE_ROOT})")

    print("\nTrainium lane (trn1.2xlarge), one precision per host\n")
    for prec in PRECS:
        m = neuron_lane(prec)
        trn1[prec] = m
        cells = list(m.main_cells())
        print(f"{prec}: {len(cells)} cell(s)  neff={m.MODELS_S3}  "
              f"auto-cast={m.PRECISIONS[prec]['auto_cast']}")
        for c in cells:
            print(f"    {c['id']:<46} {c['rows']:>4}r x {c['tps']:>4}/s  "
                  f"s={c['sessions']} rpc={c['replicas_per_core']} "
                  f"cpr={c['cores_per_replica']} ctx={c['contexts']} "
                  f"mb={c['mini_batch']} wire={c['wire']} "
                  f"log_batch={c['log_batch']} rate={c['sample_rate']} "
                  f"obj={c['log_object_bytes'] // 1048576}MiB")
        print()
        want = WANT_LOADS[("trainium1", prec)]
        check([(c["rows"], c["tps"]) for c in cells] == want,
              f"trn1 {prec} loads are {want}")
        check(all(c["wire"] == "features" for c in cells),
              f"trn1 {prec} uses the 210-feature wire")
        check(m.N_FEATURES == 210, f"trn1 {prec} expects 210 features in the banner")
        check(all(c["sessions"] == WANT_SESSIONS for c in cells),
              f"trn1 {prec} every cell is at {WANT_SESSIONS} replicas")
        check(all(c["cores_per_replica"] == 1 for c in cells),
              f"trn1 {prec} uses one core per replica")
        check(all(c["contexts"] == 1 for c in cells),
              f"trn1 {prec} uses one execution context per replica")
        check(all(c["replicas_per_core"] == 1 for c in cells),
              f"trn1 {prec} places one replica per core, so both cores are busy")
        check(all(c["mini_batch"] == c["rows"] and c["pieces"] == 1 for c in cells),
              f"trn1 {prec} scores every request in one NEFF call")
        check(all(c["sessions_per_request"] == 1 for c in cells),
              f"trn1 {prec} leases one replica per request")
        check(all(c["sample_rate"] == WANT_RATE for c in cells),
              f"trn1 {prec} logs at {WANT_RATE}")
        check(all(c["sample_every"] == WANT_EVERY
                  and abs(c["sample_pct"] - 100.0 / WANT_EVERY) < 1e-9
                  for c in cells),
              f"trn1 {prec} resolves to one request in {WANT_EVERY}, "
              f"{100.0 / WANT_EVERY:.1f}%")
        check(all(c["log_rows_per_object"] == 50000 for c in cells),
              f"trn1 {prec} writes one object per 50,000 rows")
        check(all(c["accelerator"] == "trainium1" for c in cells),
              f"trn1 {prec} cells are tagged trainium1")
        check(all(c["id"].startswith(WANT_PREFIX["trainium1"]) for c in cells),
              f"trn1 {prec} cell ids start with {WANT_PREFIX['trainium1']}")
        check(m.SHAPES == sorted({r for r, _ in want}),
              f"trn1 {prec} needs NEFF shapes {sorted({r for r, _ in want})}")
        # ---- the precision
        check(all(c["precision"] == prec for c in cells),
              f"trn1 {prec} every cell is labelled {prec}")
        check(all(c["id"].count(f"-{prec}-") == 1 for c in cells),
              f"trn1 {prec} the precision is in every cell id")
        check(m.PRECISIONS[prec]["auto_cast"] == WANT_AUTO_CAST[prec],
              f"trn1 {prec} compiles with --auto-cast={WANT_AUTO_CAST[prec]}")
        check(all(c["auto_cast"] == WANT_AUTO_CAST[prec] for c in cells),
              f"trn1 {prec} every cell records auto_cast={WANT_AUTO_CAST[prec]}")
        check(f"/{prec}" in m.MODELS_S3 and f"/{prec}" in m.GEN,
              f"trn1 {prec} NEFF cache is namespaced by precision ({m.MODELS_S3})")
        check(m.WEIGHT_BYTES == (4 if prec == "fp32" else 2),
              f"trn1 {prec} sizes its NEFF floor at "
              f"{4 if prec == 'fp32' else 2} bytes per weight")

    # ---- THE KEYS MUST BE DISJOINT BETWEEN THE ARMS. Each of these is a path or prefix
    # that, if shared, lets the second arm restore the first's artifact and publish a
    # precision column that is wrong while every other gate passes.
    print()
    check(gpu["fp32"].CACHE_ROOT != gpu["bf16"].CACHE_ROOT,
          "the two A10G arms use different TensorRT engine caches")
    check(trn1["fp32"].MODELS_S3 != trn1["bf16"].MODELS_S3,
          "the two Trainium arms use different NEFF cache prefixes")
    check(trn1["fp32"].GEN != trn1["bf16"].GEN,
          "the two Trainium arms write their NEFFs to different directories")
    check(trn1["fp32"].artifact(100)["s3_neff"] != trn1["bf16"].artifact(100)["s3_neff"],
          "the b100 NEFF has a different S3 key on each arm")
    check(trn1["fp32"].artifact(500)["s3_neff"] != trn1["bf16"].artifact(500)["s3_neff"],
          "the b500 NEFF has a different S3 key on each arm")

    # No cell id may collide ACROSS the lanes or the arms: all four write into one S3
    # prefix, so a collision silently overwrites a measurement -- or, worse, makes
    # `run_cell` skip the second one and report a complete round that never ran it.
    every = []
    for prec in PRECS:
        every += [c["id"] for c in gpu[prec].main_cells({})]
        every += [c["id"] for c in trn1[prec].main_cells()]
    print()
    check(len(set(every)) == len(every),
          f"all {len(every)} cell ids are unique across both lanes and both precisions")
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
