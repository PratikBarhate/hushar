#!/usr/bin/env bash
# Settles, by MEASUREMENT, what `torch_neuronx.trace` does about `--auto-cast` when it is
# told nothing -- which is what every Trainium campaign before
# `featurewire-log20-prec-g5-trn1-100M` compiled with.
#
# WHY THIS EXISTS. `neuronx-cc`'s CLI reference says `--auto-cast` defaults to `matmult`,
# so a compile that passes no flag should cast FP32 matrix multiplies to BF16. From that
# alone it was CLAIMED that the earlier rounds' "FP32" Trainium NEFFs were BF16-matmul
# wearing an FP32 label. But those rounds' own prime records disagree: their NEFFs are
# 0.94x of their FP32 weight bytes and their compiled-versus-PyTorch error is ~1.5e-06,
# which is what `--auto-cast=none` produces and nothing like the ~2.3e-03 that `matmult`
# produces. A documented default and a recorded artifact cannot both be right.
#
# So this compiles ONE model THREE ways on one host in one session and prints the two
# numbers that distinguish them. The framework may well not inherit the CLI's default --
# that is the thing being tested, rather than assumed in either direction.
set -euo pipefail

REV="${REV:?REV required}"
TARGET="${TARGET:-10M}"
BATCH="${BATCH:-100}"
DIR="/root/hushar-bench-$REV"
export HOME=/root
cd "$DIR"
. ./env.sh

"$NEURON_VENV/bin/python" - <<'PY'
import importlib.util, inspect, os, shutil, sys, time
from pathlib import Path

ROOT = Path.cwd()
TARGET = os.environ.get("TARGET", "10M")
BATCH = int(os.environ.get("BATCH", "100"))

# PATH before the import: `import torch_neuronx` spawns `libneuronpjrt-path` from this
# interpreter's bin, so the order is load-bearing.
binaries = Path(sys.executable).parent
os.environ["PATH"] = os.pathsep.join([str(binaries), os.environ.get("PATH", "")])

import numpy as np
import torch
import torch_neuronx

spec = importlib.util.spec_from_file_location(
    "gen", ROOT / "scripts" / "generate_benchmark_models.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

target = gen.parameter_count(TARGET)
template = gen.vectorized_config(np.random.default_rng(0), model_id="", model_path="",
                                 provider="cpu")
vc = template["vectorization_config"]
width = sum(len(vc["feature_transformations"][n]["default_val"])
            for n in vc["feature_order"])
layers, params = gen.depth_for_target(
    lambda d: gen.VectorizedModel(width, d), target)

print(f"model: target {TARGET} -> {params:,} parameters, {layers} layers, "
      f"input [{BATCH}, {width}]")
print(f"torch_neuronx {getattr(torch_neuronx, '__version__', '?')}, "
      f"neuronx-cc on PATH: {shutil.which('neuronx-cc')}")
sig = inspect.signature(torch_neuronx.trace).parameters
print(f"trace() accepts: {sorted(sig)}")
print()

# EXACTLY the three argument lists that matter. The first is what every campaign before
# this one passed -- the old `compile_model` body, verbatim.
VARIANTS = [
    ("no auto-cast flag  (the OLD code path)", ["--model-type=transformer"]),
    ("--auto-cast=none", ["--model-type=transformer", "--auto-cast=none"]),
    ("--auto-cast=matmult --auto-cast-type=bf16",
     ["--model-type=transformer", "--auto-cast=matmult", "--auto-cast-type=bf16"]),
]

fp32_bytes = params * 4
results = []
for label, args in VARIANTS:
    work = ROOT / f"probe-autocast/{abs(hash(label)) % 10**8}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    # Isolated per variant, so "the NEFF this compile produced" is unambiguous.
    os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(work / "neff-cache")
    (work / "neff-cache").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = gen.VectorizedModel(width, layers).eval()
    example = torch.randn(BATCH, width)
    t0 = time.time()
    compiled = torch_neuronx.trace(model, example,
                                   compiler_workdir=str(work),
                                   compiler_args=args)
    with torch.no_grad():
        actual = compiled(example).to("cpu")
        expected = model(example)
    took = time.time() - t0
    err = float((expected - actual).abs().max())
    neffs = sorted(work.rglob("*.neff"))
    size = max((p.stat().st_size for p in neffs), default=0)
    results.append((label, args, err, size, took, len(neffs)))
    print(f"{label}")
    print(f"    args              {' '.join(args)}")
    print(f"    trace error       {err:.3g}")
    print(f"    NEFF bytes        {size:,}  ({size / fp32_bytes:.3f}x of "
          f"{fp32_bytes:,} FP32 weight bytes)")
    print(f"    compile seconds   {took:.1f}   NEFFs emitted {len(neffs)}")
    print()

print("=" * 72)
base = results[0]
none_ = results[1]
cast = results[2]
same_as_none = abs(base[2] - none_[2]) / max(none_[2], 1e-30) < 0.5
same_as_cast = abs(base[2] - cast[2]) / max(cast[2], 1e-30) < 0.5
print(f"no-flag trace error {base[2]:.3g} vs none {none_[2]:.3g} vs matmult {cast[2]:.3g}")
if same_as_none and not same_as_cast:
    print("VERDICT: passing no --auto-cast behaves as --auto-cast=none.")
    print("         torch_neuronx.trace does NOT inherit the neuronx-cc CLI default of")
    print("         `matmult`, so the earlier campaigns' NEFFs were genuinely FP32.")
elif same_as_cast and not same_as_none:
    print("VERDICT: passing no --auto-cast behaves as --auto-cast=matmult.")
    print("         The CLI default DOES reach the framework, so the earlier campaigns'")
    print("         NEFFs were BF16-matmul under an FP32 label.")
else:
    print("VERDICT: inconclusive -- the no-flag error matches neither arm cleanly.")
    print("         Report the three numbers and draw no conclusion.")
shutil.rmtree(ROOT / "probe-autocast", ignore_errors=True)
PY
