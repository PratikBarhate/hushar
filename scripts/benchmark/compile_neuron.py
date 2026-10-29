#!/usr/bin/env python3
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Compiles the deterministic vectorized benchmark model into one NEFF.

Run this inside the PyTorch/Neuron virtual environment on a Neuron DLAMI. Neuron does not
compile ONNX directly, so the script rebuilds the same seeded PyTorch model used by
``generate_benchmark_models.py`` and compiles it for one fixed batch size.

Exactly one NEFF must be produced, because hushar's direct libnrt backend loads a
complete graph rather than something that coordinates several compiled partitions. The
NEFF cache directory is therefore isolated per invocation and asserted to hold one file.

**The compilation route is** ``torch_neuronx.trace``, **and it has to be.** ``trace``
compiles the model with its parameters BAKED IN as constants, so the NEFF takes exactly
one input -- the ``[batch, width]`` feature tensor -- which is the contract hushar's
backend serves.

The Neuron dynamo backend in torch-neuronx 2.14 cannot replace it here, and the reason is
worth recording because the API docs suggest otherwise. ``torch.compile(model,
backend="neuron")`` lowers this model correctly and runs it on the device to 2.7e-07 of
CPU PyTorch, but:

* it writes no model NEFF to disk at all -- only an NKI kernel -- because the compiled
  graph is held in memory for the backend's own execution; and
* the graph it lowers is **stateless**: ``aot_module_simplified`` turns every parameter
  into a graph INPUT, so routing its StableHLO through ``neuronx-cc`` yields a NEFF whose
  first input is a weight tensor. hushar refuses it, correctly, with
  ``NEFF input "input0" fixes its batch at 4096``.

Baking 9M parameters back into that graph is a different piece of work from compiling a
model, so this script requires a torch-neuronx that still has ``trace`` and says so
plainly when it does not. ``config-sweep-neuron-trn1-10M-50M`` compiled with
torch_neuronx 2.9, which has it.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "generate_benchmark_models.py"

DYNAMO_REFUSAL = """\
This torch-neuronx ({version}) has no `trace`, and the dynamo backend cannot stand in
for it for this purpose.

`torch.compile(model, backend="neuron")` does lower and run this model correctly, but it
produces no NEFF file, and the graph it lowers is STATELESS -- `aot_module_simplified`
lifts every parameter to a graph input. A NEFF built from that StableHLO takes weights as
inputs, so hushar's backend refuses it with a batch size read off a weight tensor rather
than off the feature input.

Use a Neuron DLAMI whose torch-neuronx still provides `trace`, which compiles the
parameters in as constants:

    Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)

That is also the toolchain `config-sweep-neuron-trn1-10M-50M` compiled with, so a NEFF
built there is comparable with that campaign's.\
"""


def load_generator():
    """Loads the benchmark architecture from its single source of truth."""
    spec = importlib.util.spec_from_file_location("hushar_benchmark_models", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    """Returns one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiler_version():
    """The `neuronx-cc` version string, or None when it cannot be asked.

    Both streams are read: `neuronx-cc --version` writes its banner to STDERR, so a
    stdout-only read reports "no version" on a host where the compiler is present and
    working.
    """
    try:
        done = subprocess.run(["neuronx-cc", "--version"], capture_output=True,
                              text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    for stream in (done.stdout, done.stderr):
        for line in (stream or "").splitlines():
            if "version" in line.lower() and any(ch.isdigit() for ch in line):
                return line.strip()
    return None


def put_interpreter_bin_on_path():
    """Prepends this interpreter's `bin` to PATH, and returns it.

    This must run BEFORE ``import torch_neuronx``, not just before compiling. Importing
    the module spawns ``libneuronpjrt-path``, a helper that lives in the virtual
    environment's ``bin``, so an import with the wrong PATH fails with
    ``FileNotFoundError: 'libneuronpjrt-path'`` -- which looks like a missing file rather
    than a missing PATH entry. The compiler, ``neuronx-cc``, is found the same way.
    """
    binaries = Path(sys.executable).parent
    path = os.environ.get("PATH", "")
    if str(binaries) not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([str(binaries), path]) if path else str(binaries)
    return binaries


def isolate_neff_cache(cache_dir):
    """Points the NKI kernel NEFF cache at `cache_dir`, away from the shared default.

    Not where the model NEFF goes -- ``trace`` writes that into its compiler work
    directory -- but keeping it out of the shared ``/tmp/neff_cache`` means the work
    directory holds exactly one NEFF and ``only_neff`` can say so.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(cache_dir)


def compile_model(torch_neuronx, model, example, workdir):
    """Compiles with the trace options the installed Neuron release supports.

    `trace` compiles the parameters in as constants and emits the NEFF into
    `compiler_workdir`, which is why that directory is retained: it is both the
    diagnostic trail and where the artifact is found.
    """
    if not hasattr(torch_neuronx, "trace"):
        raise RuntimeError(DYNAMO_REFUSAL.format(
            version=getattr(torch_neuronx, "__version__", "unknown")))
    parameters = inspect.signature(torch_neuronx.trace).parameters
    options = {}
    if "compiler_workdir" in parameters:
        options["compiler_workdir"] = str(workdir)
    if "compiler_args" in parameters:
        options["compiler_args"] = ["--model-type=transformer"]
    compiled = torch_neuronx.trace(model, example, **options)
    with torch.no_grad():
        return compiled, compiled(example)


def only_neff(workdir):
    """Returns the one complete NEFF emitted into the compiler work directory."""
    found = sorted(workdir.rglob("*.neff"))
    if len(found) != 1:
        rendered = ", ".join(str(path.relative_to(workdir)) for path in found) or "none"
        raise RuntimeError(
            f"expected one NEFF under {workdir}, found {len(found)}: {rendered}. "
            f"hushar's libnrt backend loads one complete graph, so a model that compiled "
            f"into several partitions cannot be served."
        )
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="model size, such as 10M or 50M")
    parser.add_argument("--batch-size", required=True, type=int, help="fixed rows per NEFF call")
    parser.add_argument(
        "--out-dir",
        default="benchmark-data/generated",
        help="output directory relative to the repository root",
    )
    parser.add_argument(
        "--work-dir",
        help="compiler work directory; defaults beside the output NEFF",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    # PATH FIRST, THEN THE IMPORT. `import torch_neuronx` spawns `libneuronpjrt-path`
    # from this interpreter's bin, so the order here is load-bearing.
    binaries = put_interpreter_bin_on_path()
    try:
        import torch_neuronx
    except ImportError:
        parser.error(
            "torch_neuronx is unavailable; use the PyTorch/Neuron virtual environment "
            "on a Neuron DLAMI"
        )

    generated = load_generator()
    target = generated.parameter_count(args.params)
    # Only the widths are wanted here, so the identity fields are left empty. The
    # argument list is spelled out in full rather than relying on defaults: this call
    # drifted out of step with the generator once already, and a positional call that
    # silently takes `model_path` as `provider` would produce a configuration that loads
    # and is wrong, rather than a TypeError.
    template = generated.vectorized_config(
        np.random.default_rng(0), model_id="", model_path="", provider="cpu"
    )
    vectorization = template["vectorization_config"]
    width = sum(
        len(vectorization["feature_transformations"][name]["default_val"])
        for name in vectorization["feature_order"]
    )
    layers, parameter_count = generated.depth_for_target(
        lambda depth: generated.VectorizedModel(width, depth), target
    )

    torch.manual_seed(0)
    model = generated.VectorizedModel(width, layers).eval()
    example = torch.randn(args.batch_size, width)

    out_dir = ROOT / args.out_dir.strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bench_model_neuron_{args.params.lower()}_b{args.batch_size}"
    workdir = Path(args.work_dir) if args.work_dir else out_dir / f"{stem}.work"
    if workdir.exists() and any(workdir.iterdir()):
        raise RuntimeError(f"compiler work directory is not empty: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)

    # The NEFF cache is isolated under this invocation's work directory, so "exactly one
    # NEFF" is a property of this compile rather than of whatever the shared default
    # `/tmp/neff_cache` happened to be holding.
    # The NEFF cache is isolated under this invocation's work directory so that nothing a
    # previous compile left behind can be mistaken for this one's output.
    cache_dir = workdir / "neff-cache"
    isolate_neff_cache(cache_dir)
    # THE PRECONDITION IS THAT THE COMPILER CAN BE SPAWNED. The compiler runs as a
    # subprocess found through PATH, so `which` is exactly the question that will be
    # asked; the version string is best-effort provenance and must not gate the compile.
    executable = shutil.which("neuronx-cc")
    if executable is None:
        raise RuntimeError(
            f"neuronx-cc is not on PATH (interpreter's bin is {binaries}); the compiler "
            f"is spawned by name and would fail with 'posix_spawn failed: No such file "
            f"or directory'"
        )
    found_compiler = compiler_version()
    print(
        f"compiling {parameter_count:,} parameters, {layers} layers, "
        f"input [{args.batch_size}, {width}]"
    )
    print(f"compiler: {executable}")
    print(f"compiler version: {found_compiler or 'not reported'}")
    print(f"torch_neuronx: {getattr(torch_neuronx, '__version__', 'unknown')}")

    compiled, actual = compile_model(torch_neuronx, model, example, workdir)
    compiled.save(str(workdir / f"{stem}.pt"))
    with torch.no_grad():
        expected = model(example)
    actual = actual.to("cpu")
    maximum_error = float((expected - actual).abs().max())
    if not torch.isfinite(actual).all():
        raise RuntimeError("compiled model returned a non-finite value")
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"compiled model returned {tuple(actual.shape)}, expected {tuple(expected.shape)}"
        )

    source = only_neff(workdir)
    neff_path = out_dir / f"{stem}.neff"
    shutil.copy2(source, neff_path)

    model_path = str(neff_path.relative_to(ROOT))
    config = generated.vectorized_config(
        np.random.default_rng(0),
        model_id=stem.replace("_", "-"),
        model_path=model_path,
        provider="cpu",
        rows=args.batch_size,
    )
    config.pop("execution_provider", None)
    config["backend"] = {
        "type": "neuron",
        "start_neuron_core": 0,
        "neuron_cores_per_replica": 1,
        "execution_contexts_per_replica": 1,
    }
    config_path = out_dir / f"{stem}_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    metadata = {
        "model": model_path,
        "sha256": sha256(neff_path),
        "size_bytes": neff_path.stat().st_size,
        "parameter_count": parameter_count,
        "layers": layers,
        "input_width": width,
        "batch_size": args.batch_size,
        "maximum_trace_error": maximum_error,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "unknown"),
        "neuronx_cc": found_compiler,
        "precision": "fp32",
        # The route is recorded because it CHANGED, and a NEFF compiled one way is not
        # evidently the same artifact as one compiled another. `config-sweep-neuron-trn1-
        # 10M-50M` used torch_neuronx.trace, which no longer exists in torch-neuronx 2.14.
        "compile_route": "torch_neuronx.trace",
        "compile_route_note": (
            "torch_neuronx.trace compiles the parameters in as CONSTANTS, so the NEFF "
            "takes one input -- the [batch, width] feature tensor -- which is the "
            "contract hushar's backend serves. The torch-neuronx 2.14 dynamo backend was "
            "evaluated and cannot replace it: it writes no model NEFF to disk, and the "
            "graph it lowers is stateless, so a NEFF built from its StableHLO takes "
            "weights as inputs and hushar refuses it. Same route as "
            "config-sweep-neuron-trn1-10M-50M."
        ),
        "compiler_workdir": str(workdir.relative_to(ROOT)),
        "neff_cache_dir": str(cache_dir.relative_to(ROOT)),
    }
    metadata_path = out_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {neff_path} ({neff_path.stat().st_size:,} bytes)")
    print(f"wrote {config_path}")
    print(f"wrote {metadata_path}")
    print(f"maximum compiled-versus-PyTorch error: {maximum_error:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
