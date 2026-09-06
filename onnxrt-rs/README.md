# onnxrt-rs

A safe Rust front end over the [ONNX Runtime](https://onnxruntime.ai/) C API,
generated with `bindgen` from the real upstream header.

This is the inference engine `hushar` runs on. ONNX Runtime was chosen over a
pure-Rust engine for two reasons: far broader operator coverage, and execution
providers, which are the only way to reach CoreML on Apple Silicon, CUDA or
TensorRT on NVIDIA, and ROCm or MIGraphX on AMD. The provider is chosen from
configuration at startup, so one binary covers every target.

## Design

See [documentation/bindgen.md](../documentation/bindgen.md) for the diagrams and
the general reasoning; the summary is:

**Bindings are checked in; `bindgen` is opt-in.** `src/bindings/ort.rs`
is committed `bindgen` output, so building needs neither libclang nor an ONNX
Runtime installation. `--features bindgen` regenerates from
`headers/onnxruntime_c_api.h`.

**The library is loaded at run time, not linked.** The entire C API hangs off one
exported symbol, `OrtGetApiBase`, which returns a table of function pointers, so
run-time loading costs a single `dlsym`. In return:

- the workspace builds on a laptop with no ONNX Runtime installed,
- CI needs no native dependency,
- a deployment can swap runtime builds without recompiling,
- one binary runs on hosts with and without the runtime, and a missing runtime
  is a clear error rather than a refusal to start.

**Every handle owns its release.** `Environment`, `Session`, `SessionOptions`,
`MemoryInfo`, `TensorView`, `OwnedTensor`, and the type/shape info handle all
implement `Drop` calling the matching `Release*`. Every `unsafe` block carries
the reasoning for why it is sound.

## Vendored header

| | |
|---|---|
| Version | ONNX Runtime v1.29.0, `ORT_API_VERSION` 29 |
| Files | `headers/onnxruntime_c_api.h`, `onnxruntime_ep_c_api.h`, `onnxruntime_error_code.h` |
| Licence | MIT, see `headers/LICENSE.onnxruntime` |

The header is vendored so `--features bindgen` works offline and reproducibly.

## Finding the runtime

Set `ORT_DYLIB_PATH` to a specific shared library, or install ONNX Runtime where
the dynamic loader already looks. Without either, these names are tried:

| Platform | Names |
|---|---|
| Linux | `libonnxruntime.so.1`, `libonnxruntime.so` |
| macOS | `libonnxruntime.1.dylib`, `libonnxruntime.dylib` |
| Windows | `onnxruntime.dll` |

The versioned name is tried first deliberately. `libonnxruntime.so.1` is the
SONAME recorded in the library's own `DT_SONAME`, so it is the name the dynamic
loader can always resolve on a host where the runtime is installed;
`libonnxruntime.so` is the *linker* name, conventionally shipped only in a
`-dev`/`-devel` package. Preferring the SONAME also picks the right library when
several ONNX Runtime major versions are installed side by side.

Get a build for your platform from the
[releases page](https://github.com/microsoft/onnxruntime/releases). Assets are named
`onnxruntime-<os>-<arch>-<version>` and each carries the library under `lib/`:

| Platform | Asset for 1.29.0 |
|---|---|
| Linux x64 | `onnxruntime-linux-x64-1.29.0.tgz` |
| Linux arm64 | `onnxruntime-linux-aarch64-1.29.0.tgz` |
| macOS, Apple Silicon | `onnxruntime-osx-arm64-1.29.0.tgz` |
| Windows x64 | `onnxruntime-win-x64-1.29.0.zip` |
| NVIDIA, CUDA 12 | `onnxruntime-linux-x64-gpu_cuda12-1.29.0.tgz` |

```bash
# Linux
export ORT_DYLIB_PATH=$PWD/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so
# macOS
export ORT_DYLIB_PATH=$PWD/onnxruntime-osx-arm64-1.29.0/lib/libonnxruntime.dylib
```

Note that `onnxruntime.ai/docs/install` covers the language packages — pip, NuGet, npm —
rather than the standalone shared library this crate loads, which is why the releases page
is the link to follow.

A runtime older than the vendored header cannot supply a compatible API table;
`Api::load` reports that as `Error::UnsupportedApiVersion` rather than crashing.

## Usage

```rust
use onnxrt_rs::{Environment, SessionBuilder};
use std::sync::Arc;

let env = Arc::new(Environment::new("hushar")?);
let model = std::fs::read("model.onnx")?;

let session = Arc::new(
    SessionBuilder::new(&env)?
        .intra_op_threads(1)?          // the caller already saturates the CPU
        .build_from_memory(&model)?,   // no filesystem round trip
);

let mut features = vec![1.0f32, 2.0, 3.0, 0.5, 1.0, 1.5];  // 2 rows x 3 features
let output = session.run_single(&mut features, &[2, 3])?;

println!("{:?} -> {:?}", output.shape()?, output.as_slice::<f32>()?);
```

`Session` is `Send + Sync`, so an `Arc<Session>` serves concurrent requests
directly: ONNX Runtime documents `Run` on a shared session as needing no external
synchronisation
([microsoft/onnxruntime#114](https://github.com/microsoft/onnxruntime/issues/114)).

## Testing

Unit tests always run. The integration tests in `tests/onnxruntime_smoke.rs` need
a real runtime and skip with an explanation when none is available, so the suite
stays green without ONNX Runtime installed.

```bash
cargo test -p onnxrt-rs     # unit tests, integration skipped
export ORT_DYLIB_PATH=…     # the library for your platform, see above
cargo test -p onnxrt-rs     # everything
```

The models the integration tests load live in this crate's own `test-data`, so the
suite needs nothing but this crate and a `libonnxruntime`. They are generated by
`scripts/generate_test_model.py` and `scripts/generate_typed_test_models.py` in the
repository root, which write a copy into every crate that tests one.

## Regenerating the bindings

```bash
cargo build -p onnxrt-rs --features bindgen
cp "$(find target -name bindings.rs -path '*onnxrt-rs*' | head -1)" \
   src/bindings/ort.rs   # then restore the provenance header comment
```

To move to a different ONNX Runtime release, replace the files in `headers/` (or
point `ORT_HEADER_DIR` elsewhere) and regenerate. `bindgen` emits compile-time
layout assertions for every struct, so a size, alignment, or field-offset
mismatch is a build error rather than a memory-safety bug.

## Coverage

Wrapped: environment and logging, session options (thread counts, optimisation
level), loading from memory, input/output name discovery, CPU tensors over
borrowed buffers, running inference, reading typed outputs with shapes, and
execution provider registration for CPU, CoreML, XNNPACK, TensorRT and MIGraphX
— see [documentation/hardware.md](../documentation/hardware.md) for the provider
strings and how availability is checked.

Not wrapped yet: IO binding, run options, and custom operators. `Api::raw()` is
the escape hatch to the full 425-entry table until those are added.
