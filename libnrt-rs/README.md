# libnrt-rs

A safe Rust front end over the [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/)
runtime (`libnrt`) C API, generated with `bindgen` from the real SDK headers.

Neuron is the runtime behind Inferentia and Trainium instances. Models are
compiled ahead of time by the Neuron compiler into a **NEFF**; this crate loads
and executes a NEFF. It does not compile ONNX. The pipeline is:

```
model.onnx  ──neuronx-cc──▶  model.neff  ──libnrt-rs──▶  inference
```

## Design

See [documentation/bindgen.md](../documentation/bindgen.md) for the diagrams and
the general reasoning; the summary is:

**Bindings are checked in; `bindgen` is opt-in.** The Neuron SDK only installs on
Linux hosts with Neuron devices, so requiring it at build time would make the
whole workspace unbuildable anywhere else. `src/bindings/nrt.rs` is committed
`bindgen` output; `--features bindgen` regenerates it from the SDK.

**`libnrt.so.1` is opened at run time, never linked.** One binary therefore runs
on Neuron and non-Neuron hosts alike — it only fails when something actually asks
for a device. A service can carry a Neuron backend and a CPU backend together and
choose at startup.

**Shutdown ordering is enforced by the type system.** `nrt_init`/`nrt_close` are
process-global, and closing the runtime with a model still loaded is undefined
behaviour. `Runtime` is handed out as an `Arc` and `Model`, `Tensor`, and
`TensorSet` each hold a clone, so reference counting makes the correct teardown
order the only possible one. A second `nrt_init` in one process is reported as
`Error::AlreadyInitialized` instead of corrupting shared state.

## The headers are not vendored

Unlike ONNX Runtime's MIT licence, the Neuron SDK headers ship as
`Copyright Amazon.com, Inc. All Rights Reserved`, so **they are deliberately not
committed to this repository**. Instead `headers/wrapper.h` is our own file that
`#include`s them from the SDK on the build host.

| | |
|---|---|
| Source | `aws-neuronx-runtime-lib` 2.34.10.0-ac18d186d |
| Repository | <https://apt.repos.neuron.amazonaws.com> |
| Install path | `/opt/aws/neuron/include`, override with `NEURON_INCLUDE_DIR` |
| Library | `libnrt.so.1`, override with `NRT_DYLIB_PATH` |

Only the mechanically derived `bindgen` output is checked in, and the allowlist is
narrowed to `nrt_*` / `NRT_*` so nothing beyond the API this crate wraps is
reproduced.

### About `headers/shim/linux/types.h`

`ndl/neuron_driver_shared.h` includes `<linux/types.h>` because it shares structs
with the neuron kernel driver, which would otherwise make it impossible to
regenerate the bindings anywhere but Linux. The shim supplies the six fixed-width
typedefs the SDK headers actually use (`__u8`, `__u16`, `__u32`, `__u64`, `__s32`,
`__s64`), and `build.rs` adds it with `-idirafter` so it is searched *after* every
real include path — on Linux the genuine kernel header always wins.

### Cross-platform note

The committed bindings are generated on `x86_64-unknown-linux-gnu` — a platform
`libnrt` actually runs on — so they use the real `<linux/types.h>` rather than the
shim.

They can also be regenerated off Linux, which is what the shim is for, and that
was checked rather than assumed. Generating on macOS arm64 and on Linux x86-64
and diffing the two gives **4 differing lines**, all of them kernel typedefs:

| | macOS arm64 (shim) | Linux x86-64 (real headers) |
|---|---|---|
| `__u32` | `u32` | `::std::os::raw::c_uint` |
| `__u64` | `u64` | `::std::os::raw::c_ulonglong` |

Same size, alignment, and signedness on both, so either spelling is
layout-correct — but the Linux form is the faithful one, which is why it is what
gets committed. Everything else, including every struct layout and every
function signature, is byte-identical.

That result is expected rather than lucky: this API is built only from `char`,
`int`, `int32_t`, `uint32_t`, `uint64_t`, `size_t`, pointers, and C enums. It is
*not* a general guarantee. If AWS ever adds a `long`, a bitfield, or a packed
struct, regenerate on the target. To check for drift on a Neuron host:

```bash
rustup component add rustfmt          # see the caveat below
cargo build -p libnrt-rs --features bindgen
diff <(tail -n +15 src/bindings/nrt.rs) \
     "$(find target -name bindings.rs -path '*libnrt-rs*' | head -1)"
```

**`rustfmt` must be installed**, or this diff is worthless. bindgen shells out to
`rustfmt` to format its output and only warns when it is missing — it then emits
the entire binding on two enormous lines, which is valid Rust but makes a
line-based diff report the whole file as changed. Minimal toolchains, including
the official `rust:*-bookworm` container images, do not ship it.

## Platform support

Linux only, with a Neuron device. Everywhere else `Runtime::init` returns
`Error::UnsupportedPlatform` rather than failing to compile. Use
`libnrt_rs::is_supported_platform()` to branch at startup.

## Usage

```rust
use libnrt_rs::{Model, Runtime};
use std::sync::Arc;

let runtime = Runtime::init()?;                       // once per process
println!("libnrt {} / {} cores", runtime.version()?, runtime.visible_neuron_cores()?);

let neff = std::fs::read("model.neff")?;
let model = Model::load(Arc::clone(&runtime), &neff, 0, 1)?;   // core 0, 1 core

// Buffers sized straight from the NEFF's own tensor table.
let (mut inputs, mut outputs) = model.allocate_io(0)?;

let input = model.inputs().next().expect("model has an input").name.clone();
inputs.get_mut(&input).expect("just allocated").write_f32(&[1.0, 2.0, 3.0])?;

model.execute(&inputs, &mut outputs)?;

for tensor in outputs.tensors() {
    println!("{} -> {:?}", tensor.name(), tensor.read_f32()?);
}
```

`NEURON_RT_VISIBLE_CORES` restricts which cores the process can see; compare
`total_neuron_cores()` with `visible_neuron_cores()` to detect that.

## Testing

Unit tests cover the parts that do not need hardware: `NRT_STATUS` mapping
round-trips, the non-contiguous `NRT_DTYPE_*` values (where a careless `as` cast
would be wrong), element sizes, and the off-Linux failure path. They run
everywhere:

```bash
cargo test -p libnrt-rs
```

**There is no hardware test coverage.** Executing a NEFF requires an Inf/Trn
instance, so the load-and-execute path in `Model` and `Tensor` is verified by
review against the SDK headers, not by a passing test. Run the example above on an
Inf2 or Trn1 instance before trusting it in production.

## Coverage

Wrapped: runtime init/close with lifecycle guards, version and NeuronCore counts,
NEFF load/unload, tensor info discovery (names, usages, dtypes, shapes, sizes),
device and host tensor allocation, tensor read/write including `f32` helpers,
tensor sets, and execution.

Not wrapped yet: collectives (`nrt_load_collectives`, all-gather, barrier),
async execution and callbacks, profiling and tracing, pinned host allocation,
tensor slices and attached buffers, and the batch tensor operations. Those
declarations are present in `sys` for anyone who needs them.

Note that `nrt_get_model_tensor_info` — the only way to learn a NEFF's tensor
names and shapes — lives in the SDK's `nrt_experimental.h`, so this crate inherits
that header's experimental status for tensor discovery.
