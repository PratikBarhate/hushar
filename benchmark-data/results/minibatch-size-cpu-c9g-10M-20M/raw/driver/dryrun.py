#!/usr/bin/env python3
"""Static checks on the driver, before anything is launched. Touches no AWS.

    ./dryrun.py            # must print `0 failures`

Each of these corresponds to a fault that has actually cost measured cells in this series.
Running it is cheaper than a lane.
"""
import ast
import importlib
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
FAILURES = []

# The four lanes, as hosts.py defines them.
sys.path.insert(0, str(HERE))
os.environ.setdefault("HUSHAR_BENCH_BUCKET", "hushar-bench-<account-id>-us-west-2")
import hosts  # noqa: E402


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAILURES.append(what)
    return ok


def lane_sweep(size, loads, **extra):
    """`sweep.py` re-imported for one lane, so module-level constants re-evaluate."""
    os.environ.update(REV="dryrun", MODEL_MD5="0" * 32, MODEL_SIZE=size, LOADS=loads,
                      RIG="c9g-1=10.0.0.1", **extra)
    sys.modules.pop("sweep", None)
    return importlib.import_module("sweep")


def shadow_scan():
    """Refuses a sweep where a local variable shadows a module-level function.

    WHY THIS IS A GATE AND NOT A STYLE NOTE. A previous round's `run_cell` assigned a
    local `tail` in its client-failure path. When a module-level `tail()` helper was added
    for redaction, Python bound the name local for the WHOLE function, so the call earlier
    in the body raised `UnboundLocalError` -- after the cell had already served its full
    measured window. Five cells were lost to it. Importing the module cannot catch it,
    because the line only runs at the end of a real cell; this reads the AST instead.
    """
    bad = []
    for name in ("sweep.py", "collect.py", "report.py", "verify.py"):
        tree = ast.parse((HERE / name).read_text())
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
                    bad.append(f"{name}: {node.name}() calls {sub.func.id}() and also "
                               f"assigns a local named {sub.func.id}")
    for message in sorted(set(bad)):
        check(False, message)
    if not bad:
        check(True, "no module-level function is shadowed by a local")


def env_shadow_scan():
    """Every knob launch.sh forwards must be read through sweep.py's `env()`.

    WHY. `launch.sh` deliberately passes an EMPTY value for a knob it does not want to
    override, so that sweep.py's own default -- where the run is documented and validated
    -- applies. But `os.environ.get(name, default)` returns `""` for a variable that is
    set and empty, not the default, and `int("")` raises at import. That would abort every
    lane before it measured anything.
    """
    launch = (HERE / "launch.sh").read_text()
    forwarded = set(re.findall(r"^\s*([A-Z_]+)=\"\$\{\1:-\}\"", launch, re.M))
    src = (HERE / "sweep.py").read_text()
    raw = set(re.findall(r'os\.environ\.get\(\s*"([A-Z_]+)"', src))
    # `env()` reads os.environ.get itself, and the refusal list is meant to see a raw
    # value, so both are expected here.
    allowed = {"HUSHAR_BENCH_BUCKET", "LANE_LOG"}
    leaked = sorted((forwarded & raw) - allowed)
    check(not leaked,
          f"knobs launch.sh forwards as empty are read with a bare os.environ.get, so "
          f"their sweep.py default is shadowed by '': {leaked}"
          if leaked else
          f"all {len(forwarded)} knob(s) launch.sh forwards as empty are read through "
          f"env(), which treats set-but-empty as absent")


def assertion_consistency():
    """The banner strings sweep.py asserts must be the ones verify.py re-checks.

    A round where the two drifted would pass both gates and prove nothing: the sweep would
    accept a server the verifier never looks at, or the verifier would demand a line the
    sweep never required.

    ADJACENT STRING LITERALS ARE JOINED FIRST. Both files wrap long assertions across
    lines, so a phrase that is one string at run time is two in the source and a plain
    substring search finds neither. The join is undone here rather than the assertions
    being rewritten to fit the checker.
    """
    def source(name):
        src = (HERE / name).read_text()
        # `"... " \n f"..."` and `"... " \n f'...'` both become one literal at run time.
        return re.sub(r"[\"']\s*\n\s*f?[\"']", "", src)

    sweep_src, verify_src = source("sweep.py"), source("verify.py")
    shared = [
        "logging 1 request in ",
        "inference logs -> s3://",
        "Amazon S3",
        " features transformed and concatenated",
        "spin   : idle intra-op and inter-op threads ",
        " ONNX Runtime session(s)",
        "in groups of",
        "one session each",
        "overlap  : ",
        'configured by "threading.concurrency_per_session"',
        "plan     : ",
        "A/B",
    ]
    missing = [s for s in shared if s not in sweep_src or s not in verify_src]
    check(not missing,
          f"banner assertions present in only one of sweep.py/verify.py: {missing}"
          if missing else
          f"all {len(shared)} banner assertion(s) appear in both sweep.py and verify.py")


def grid():
    """Every lane's grid enumerates, and the arithmetic holds."""
    total = 0
    for name in hosts.GROUPS["servers"]:
        size, loads = hosts.SIZE_OF[name], hosts.LOADS_OF[name]
        sw = lane_sweep(size, loads)
        cells = list(sw.main_cells())
        total += len(cells)
        rows, tps = cells[0]["rows"], cells[0]["tps"]
        check(all(c["rows"] == rows and c["tps"] == tps for c in cells),
              f"{name}: every cell is one load, {rows}x{tps}")
        check(all(c["pieces"] == (1 if c["reference"]
                                  else -(-c["rows"] // c["mini_batch"]))
                  for c in cells),
              f"{name}: pieces is ceil(rows / mini batch) on every cell")
        check(all(c["sessions"] * c["intra"] == sw.CORES for c in cells),
              f"{name}: sessions x intra-op width spends exactly {sw.CORES} core(s) on "
              f"every cell")
        check(all(c["concurrency_per_session"] == sw.CONCURRENCY_PER_SESSION
                  for c in cells if not c["reference"]),
              f"{name}: the per-session concurrency is held at "
              f"{sw.CONCURRENCY_PER_SESSION} across the lane, so the mini-batch size is "
              f"the only variable in a group")
        check(all(c["concurrency_per_session"] is None and c["mini_batch"] is None
                  for c in cells if c["reference"]),
              f"{name}: the unsplit reference configures no mini batch and no "
              f"concurrency, which is what makes its plan `whole request`")
        check(len({c["id"] for c in cells}) == len(cells),
              f"{name}: all {len(cells)} cell id(s) are distinct")
        groups = {}
        for c in cells:
            if not c["repeat"]:
                groups.setdefault(c["sessions"], []).append(c)
        check(all(sum(1 for c in g if c["baseline"]) == 1 for g in groups.values()),
              f"{name}: every session count's group has exactly one "
              f"`mini batch = rows` baseline")
        check(all(g[0]["baseline"] for g in groups.values()),
              f"{name}: the baseline runs FIRST in its group, so a lane that dies "
              f"part-way leaves ratios with something to divide by")
    check(total == 56, f"{total} cell(s) across {len(hosts.GROUPS['servers'])} lane(s)")


def overlap_gate():
    """A concurrency of 1 with a split configured must be refused at import."""
    try:
        lane_sweep("10m", "100x250", CONCURRENCY_PER_SESSION="1")
        check(False, "a per-session concurrency of 1 with a split configured was "
                     "ACCEPTED; every split cell would fail to start")
    except SystemExit as e:
        check("REFUSES a mini_batch_size" in str(e),
              "a per-session concurrency of 1 is refused at import, with the reason")
    finally:
        os.environ.pop("CONCURRENCY_PER_SESSION", None)


def no_identifiers():
    """No account id or bucket literal in any driver file."""
    ids = set(re.findall(r"\d{12}",
                         " ".join((HERE / n).read_text()
                                  for n in os.listdir(HERE)
                                  if (HERE / n).is_file()
                                  and n.endswith((".py", ".sh", ".md")))))
    check(not ids, f"a 12-digit run appears in a driver file: {sorted(ids)}"
          if ids else "no 12-digit identifier in any driver file")


def main():
    print("shadowing")
    shadow_scan()
    print("\nenvironment")
    env_shadow_scan()
    print("\nassertions")
    assertion_consistency()
    print("\nthe overlap gate")
    overlap_gate()
    print("\nprivacy")
    no_identifiers()
    print("\nthe grid")
    grid()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        return 1
    print("0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
