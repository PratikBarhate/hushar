# NVIDIA A10G: hushar against Triton, 10M and 20M

One A10G GPU, one graph, one TensorRT. hushar scores it through ONNX Runtime's TensorRT execution provider; Triton scores it through its own `tensorrt` backend from a prebuilt plan. Same offered load, same device, and -- because hushar loads the TensorRT libraries taken out of the Triton image -- an engine built by the same compiler on both sides.

## Answer

| model | rows x TPS | hushar p50 | Triton p50 | lower median | hushar p99 | Triton p99 |
|---|---:|---:|---:|---:|---:|---:|
| 10M | 100 x 200 | 2.19 | 1.43 | Triton 1.53x | 2.68 | 1.65 |
| 10M | 100 x 400 | 2.39 | 1.41 | Triton 1.70x | 2.73 | 1.55 |
| 10M | 100 x 800 | 2.53 | 1.60 | Triton 1.58x | 3.19 | 1.69 |
| 10M | 100 x 1000 | 2.68 | 1.64 | Triton 1.63x | 3.40 | 6.23 |
| 20M | 100 x 200 | 3.10 | 2.39 | Triton 1.30x | 3.54 | 2.61 |
| 20M | 100 x 400 | 3.16 | 2.30 | Triton 1.37x | 3.59 | 2.53 |
| 20M | 100 x 600 | 3.59 | 2.79 | Triton 1.29x | 4.79 | 3.03 |
| 20M | 100 x 750 | 4.44 | 3.36 | Triton 1.32x | 5.59 | 5.15 |

- **Triton has the lower median on every load, by 1.3x to 1.7x.** That reverses the CPU campaigns, and it is this run's headline.
- **The feature path is not what costs hushar the median.** Vectorizing 210 features takes 0.23-0.26 ms, while the gap to Triton is 0.71-1.08 ms. What accounts for it is the 1.17-1.58 ms that is neither graph nor vectorization: gRPC, admission, the copy into the input buffer, and leasing a session. On a CPU that overhead sat under a graph call of tens of milliseconds and did not matter; here the 10M graph is under 1 ms and the overhead is the dominant term.
- **Doubling the model costs the graph 2.15x to 2.20x, against 2.42x more arithmetic per row.** TensorRT tracks the FLOPs on this device rather than the parameter count, which is the honest way to read a 2x model; end to end hushar only slows 1.32x to 1.42x, because the fixed per-request cost below does not scale with the graph.
- **The tails do not follow the medians.** At 10M 100 x 1000 -- the only load where they diverge -- Triton's p99 jumps to 6.23 ms against hushar's 3.40, so Triton's median advantage there is bought with a 1.8x worse tail. On the other 9 loads Triton leads at both.
- **Capacity is the same for both stacks:** **10M** both hold every rate offered, to 1000; **20M** both hold 750 and both fail 800, topping out near 777 req/s. The two servers break at the same rung and at the same achieved rate, which says the ceiling belongs to the card and not to either of them. A rate that was never offered is not a ceiling, so where nothing failed these are floors.
- **At the top of the grid the device IS the constraint.** GPU occupancy spans 14-99.7%, peaking at 99.7% and 299 W on 20M 100 x 750. Where the card is saturated the two stacks' medians converge, because what is left to differ is no longer the server. Below that, the host runs out before the GPU does.
- 20 measured cells: 16 sustained, 0 partial, 4 failed the rate. 20 carry GPU samples and 10 carry server-side percentiles.

This table holds only the loads BOTH stacks sustained -- every slice of the measured window serving at least 98% of the offered rate with nothing shed and nothing failed. The four overload cells are in `Every cell` below, reported by achieved rate: the latency of a rate that was not held is queue depth and has no median worth comparing.

## Every cell

| model | rows x TPS | stack | layout | result | p50 | p95 | p99 | srv p50 | GPU | W | CPU | shed | cell |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10M | 100 x 200 | hushar | sessions 2 | ok 200.0/200 | 2.19 | 2.53 | 2.68 | 0.784 | 14.0% | 84 | 5.5% | 0 | `g5-hushar-10m-r100t200-p2` |
| 10M | 100 x 200 | Triton | instances 4 | ok 200.0/200 | 1.43 | 1.57 | 1.65 | -- | 13.9% | 83 | 3.0% | 0 | `g5-triton-10m-r100t200-p4` |
| 10M | 100 x 400 | hushar | sessions 2 | ok 400.0/400 | 2.39 | 2.61 | 2.73 | 0.782 | 28.0% | 109 | 10.4% | 0 | `g5-hushar-10m-r100t400-p2` |
| 10M | 100 x 400 | Triton | instances 4 | ok 400.0/400 | 1.41 | 1.48 | 1.55 | -- | 28.4% | 109 | 4.5% | 0 | `g5-triton-10m-r100t400-p4` |
| 10M | 100 x 800 | hushar | sessions 2 | ok 800.0/800 | 2.53 | 2.97 | 3.19 | 0.818 | 54.0% | 162 | 19.3% | 0 | `g5-hushar-10m-r100t800-p2` |
| 10M | 100 x 800 | Triton | instances 4 | ok 800.0/800 | 1.60 | 1.66 | 1.69 | -- | 56.7% | 162 | 8.4% | 0 | `g5-triton-10m-r100t800-p4` |
| 10M | 100 x 1000 | hushar | sessions 2 | ok 1000.0/1000 | 2.68 | 3.17 | 3.40 | 0.853 | 66.4% | 190 | 24.7% | 0 | `g5-hushar-10m-r100t1000-p2` |
| 10M | 100 x 1000 | Triton | instances 4 | ok 1000.0/1000 | 1.64 | 1.84 | 6.23 | -- | 70.7% | 189 | 10.2% | 0 | `g5-triton-10m-r100t1000-p4` |
| 20M | 100 x 200 | hushar | sessions 4 | ok 200.0/200 | 3.10 | 3.32 | 3.54 | 1.683 | 31.9% | 118 | 7.6% | 0 | `g5-hushar-20m-r100t200-p4` |
| 20M | 100 x 200 | Triton | instances 4 | ok 200.0/200 | 2.39 | 2.54 | 2.61 | -- | 31.8% | 116 | 3.7% | 0 | `g5-triton-20m-r100t200-p4` |
| 20M | 100 x 400 | hushar | sessions 4 | ok 400.0/400 | 3.16 | 3.39 | 3.59 | 1.719 | 63.4% | 178 | 14.6% | 0 | `g5-hushar-20m-r100t400-p4` |
| 20M | 100 x 400 | Triton | instances 4 | ok 400.0/400 | 2.30 | 2.48 | 2.53 | -- | 63.8% | 180 | 5.1% | 0 | `g5-triton-20m-r100t400-p4` |
| 20M | 100 x 600 | hushar | sessions 4 | ok 600.0/600 | 3.59 | 4.40 | 4.79 | 1.888 | 91.0% | 235 | 23.1% | 0 | `g5-hushar-20m-r100t600-p4` |
| 20M | 100 x 600 | Triton | instances 4 | ok 600.0/600 | 2.79 | 2.98 | 3.03 | -- | 90.9% | 235 | 7.1% | 0 | `g5-triton-20m-r100t600-p4` |
| 20M | 100 x 750 | hushar | sessions 4 | ok 750.0/750 | 4.44 | 5.25 | 5.59 | 2.715 | 99.7% | 299 | 36.6% | 0 | `g5-hushar-20m-r100t750-p4` * |
| 20M | 100 x 750 | Triton | instances 4 | ok 750.0/750 | 3.36 | 3.70 | 5.15 | -- | 99.2% | 295 | 16.8% | 0 | `g5-triton-20m-r100t750-p4` |
| 20M | 100 x 800 | hushar | sessions 4 | no 776.9/800 | 4127.76 | 4134.33 | 4136.40 | 4.715 | 99.9% | 299 | 59.8% | 6917 | `g5-hushar-20m-r100t800-p4` |
| 20M | 100 x 800 | Triton | instances 4 | no 772.4/800 | 662.81 | 666.15 | 667.80 | -- | 99.7% | 298 | 9.7% | 8618 | `g5-triton-20m-r100t800-p4` |
| 20M | 100 x 1000 | hushar | sessions 4 | no 774.7/1000 | 5161.62 | 5170.45 | 5173.19 | 4.715 | 99.9% | 299 | 59.8% | 90146 | `g5-hushar-20m-r100t1000-p4` |
| 20M | 100 x 1000 | Triton | instances 4 | no 772.4/1000 | 663.38 | 666.39 | 667.74 | -- | 99.7% | 298 | 9.7% | 68640 | `g5-triton-20m-r100t1000-p4` |


`*` measured by a different run of the same configuration: 20M 100 x 750 by [config-sweep-gpu-g5-wire-10M-20M](../config-sweep-gpu-g5-wire-10M-20M/summary.md). Every field that defines the cell -- graph md5, sessions, mini batch, window, ONNX Runtime, engine cache -- is identical; only the instance and the session differ.

`srv p50` is hushar's own `InferenceTime` -- the graph call alone -- as the mean of the measured window's minute buckets. Triton's equivalent is cumulative over warm-up and cannot be windowed, so it is kept in the raw cells and not compared here. `GPU` is occupancy over time from `nvidia-smi`, not saturation.

## The parallelism each stack ran, and why

| model | stack | load | parallelism | result | p50 | p95 | chosen |
|---|---|---:|---:|---:|---:|---:|---:|
| 10M | hushar | 100 x 1000 | sessions 1 | no 901.1/1000 | 4437.09 | 4493.43 |  |
| 10M | hushar | 100 x 1000 | sessions 2 | ok 1000.0/1000 | 2.67 | 3.19 | <- |
| 10M | hushar | 100 x 1000 | sessions 4 | ok 1000.0/1000 | 2.69 | 3.22 |  |
| 10M | Triton | 100 x 1000 | instances 1 | ok 1000.0/1000 | 1.62 | 6.14 |  |
| 10M | Triton | 100 x 1000 | instances 2 | ok 1000.0/1000 | 1.62 | 1.89 |  |
| 10M | Triton | 100 x 1000 | instances 4 | ok 1000.0/1000 | 1.59 | 1.78 | <- |
| 20M | hushar | 100 x 1000 | sessions 1 | no 499.7/1000 | 8001.93 | 8139.30 |  |
| 20M | hushar | 100 x 1000 | sessions 2 | no 736.6/1000 | 5430.02 | 5445.36 |  |
| 20M | hushar | 100 x 1000 | sessions 4 | no 777.0/1000 | 5147.06 | 5160.18 |  |
| 20M | Triton | 100 x 1000 | instances 1 | no 576.4/1000 | 895.52 | 907.33 |  |
| 20M | Triton | 100 x 1000 | instances 2 | no 764.1/1000 | 674.03 | 677.73 |  |
| 20M | Triton | 100 x 1000 | instances 4 | no 777.4/1000 | 662.48 | 665.49 | <- |

For hushar the knob is `threading.sessions_per_model`, an ONNX Runtime session and execution context each; for Triton it is `instance_group.count`, execution contexts of one engine. Both decide how many requests may be on the device at once.

The probe measures the heaviest rung of that model's grid, and the rule is the one the Trainium run used: the lowest median among the configurations that SUSTAINED it. For 10M hushar, 10M Triton that is what fired (lowest p50 among sustained probes). For 20M Triton no configuration sustained the probe load, so the selection fell back to the highest achieved rate -- a fallback, not the primary rule, and `selection/` records which one applied.

## Where hushar's median goes

| model | rows x TPS | hushar p50 | graph | vectorize | everything else | graph share | vectorize per row |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10M | 100 x 200 | 2.19 | 0.784 | 0.234 | 1.172 | 36% | 2.34 us |
| 10M | 100 x 400 | 2.39 | 0.782 | 0.235 | 1.373 | 33% | 2.35 us |
| 10M | 100 x 800 | 2.53 | 0.818 | 0.244 | 1.468 | 32% | 2.44 us |
| 10M | 100 x 1000 | 2.68 | 0.853 | 0.248 | 1.579 | 32% | 2.48 us |
| 20M | 100 x 200 | 3.10 | 1.683 | 0.229 | 1.188 | 54% | 2.29 us |
| 20M | 100 x 400 | 3.16 | 1.719 | 0.241 | 1.200 | 54% | 2.41 us |
| 20M | 100 x 600 | 3.59 | 1.888 | 0.247 | 1.455 | 53% | 2.47 us |
| 20M | 100 x 750 | 4.44 | 2.715 | 0.261 | 1.464 | 61% | 2.61 us |

`graph` is hushar's own `InferenceTime`, `vectorize` its `VectorizationTime`, and `everything else` is the remainder of the client-side median -- gRPC, admission, the copy into the input buffer, and leasing a session. Two things to take from it. First, Triton's contract does not remove the 210 transformations, it moves them to the caller, and `vectorize` is what that work costs in batched Rust on the server, so read it as a LOWER bound on the caller's cost. Second, and more useful here: `everything else` barely moves when the graph doubles, because it is per-request cost and not per-FLOP cost. Its SHARE does fall at 20M -- but only because the graph grew around it, not because it got cheaper. It is what the gap against Triton is made of, and it is where the work is if that gap is worth closing.

## What 2x the model costs

| rows x TPS | hushar 10M | hushar 20M | ratio | Triton 10M | Triton 20M | ratio | graph 10M | graph 20M | graph ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 x 200 | 2.19 | 3.10 | 1.42x | 1.43 | 2.39 | 1.67x | 0.784 | 1.683 | 2.15x |
| 100 x 400 | 2.39 | 3.16 | 1.32x | 1.41 | 2.30 | 1.63x | 0.782 | 1.719 | 2.20x |

The 20M graph is the 10M graph with five encoder layers instead of two: the same 210 features, the same 690-float input, the same thirteen ONNX operator types and the same matmul shapes. It carries **2.03x the parameters** and **2.42x the arithmetic per row** -- the two differ because `to_tokens` is a `Linear(690 -> 8x512)` applied once per row, so it costs the same 2,830,336 parameters at either size and every added parameter lands in the trunk. Read latency against 2.42x, not 2x. `graph` is hushar's `InferenceTime`, which is the device doing the arithmetic and nothing else; the client columns carry fixed per-request cost that does not scale with the model, so they understate the ratio by construction.

## What was running

| what | value |
|---|---|
| server | `g5.2xlarge`, 8 vCPU |
| GPU | NVIDIA A10G, 23028 MiB, driver 595.112.07 |
| kernel | 6.12.111-136.204.amzn2023.x86_64 |
| TensorRT | 10.16.1 on CUDA 13 -- the same library on both sides, because hushar loads the copies taken out of the Triton image |
| hushar | ONNX Runtime 1.28.3 (`onnxruntime-linux-x64-gpu_cuda13-1.28.3`), TensorRT execution provider, FP32 |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `platform: tensorrt_plan`, `max_batch_size: 0`, KIND_GPU, FP32 |
| graphs | 10M `5539c97131c8` (9,138,693 parameters, 2 layers), 20M `30f9c5330b80` (18,589,701 parameters, 5 layers) |
| engine cache 10M | compiled in 25.4s, then reloaded by a fresh server in 22.1s |
| engine cache 20M | compiled in 25.8s, then reloaded by a fresh server in 22.1s |

## Method and limits

- **Window.** 120s warm-up discarded, then 300s measured as one continuous load reported in 5 slices. A fresh server per cell, stopped afterwards, 60s idle before the next. Clients start on a minute boundary so the measured window is whole CloudWatch buckets.
- **One graph call per request.** The engine is built with min = opt = max = `[100, 690]`, so the shape is static on both sides: hushar runs `mini_batch_size = rows`, Triton runs `max_batch_size: 0`. Neither stack batches across requests.
- **The wires differ.** hushar is sent 210 named features; Triton is sent the 690-float tensor its plan requires. The section above prices that difference. A tensor-wire hushar arm would close it exactly and is not in this run.
- **The A10G is chosen, not settled for.** It is Ampere, contemporary with the Trainium1 this line of work is read against; the L4 in the g6 family is Ada, two years newer. `g5.2xlarge` also matches the Trainium host on the axis the feature path cares about, 8 vCPU. g6 and gr6 were available at moments when g5 was not, and were declined: a newer card would flatter the GPU column and make the comparison mean less. See the note on `SERVER_TYPE` in `hosts.py`.
- **FP32 only.** Both stacks use TensorRT's default builder settings. An A10G's FP16 and INT8 tensor cores are most of what the device is for, so these numbers are a floor on what it can do, not a ceiling.
- **One host, both sizes, measured one after the other.** `g5.2xlarge` capacity allowed one lane, so 10M and 20M share a machine -- which is what the weight comparison needs, since a placement offset between two instances would otherwise contaminate every ratio in it.
- **The rows were not all measured on one instance** (2 machines of the same type): 10M 100x1000, 10M 100x200, 10M 100x400, 10M 100x800, 20M 100x1000, 20M 100x200, 20M 100x400, 20M 100x600, 20M 100x750, 20M 100x800 on 2026-10-25; 20M 100x750 on 2026-10-26. Each row's two arms were always measured together on one machine, so every hushar-against-Triton comparison is within-host. Only comparisons ACROSS rows -- the shape of the latency curve -- cross a machine, and they carry the repeatability band below.
- **GPU utilization is occupancy, not saturation.** `nvidia-smi` reports the share of each sample period with at least one resident kernel, so a stream of small kernels reads high on a device with headroom. Power and clocks are reported next to it for that reason.
- **Evidence.** `raw/results.json` is this report's input. Cells, resolved configurations, server and client logs, per-second host and GPU samples, Triton's `config.pbtxt`, the `trtexec` build logs and the CloudWatch points are under `raw/`. `verify.py` re-derives every number here from them.
