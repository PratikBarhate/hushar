# Benchmark results

Latency in milliseconds, client-observed round trips. A cell is sustained only when all 15
slices of the window served the offered rate within 2% with nothing shed or failed; only
sustained cells carry a latency. Each run is a 900s window after a 300s warm-up, a fresh
server per cell, 60s idle between, unless marked. Model structure:
[`scripts/benchmark-models.md`](../../scripts/benchmark-models.md).

---

## 1. Evaluation on small models

10M (2 encoder layers) and 20M (5 layers), FP32, at the highest rate each host sustained.
hushar as deployed: 210 named features per row vectorized on the server, one request in
five written to Amazon S3. Accelerator rows are 2 sessions scoring a whole request; CPU
rows are 4 sessions splitting it into mini batches of 5 rows at 100 and 10 rows at 500
(appendix A). Sources:
[`minibatch-size-cpu-c9g-10M-20M`](minibatch-size-cpu-c9g-10M-20M/summary.md),
[`featurewire-log20-g5-trn1-10M-20M`](featurewire-log20-g5-trn1-10M-20M/summary.md).

| device | instance | model | rows x TPS | rows/s | p50 | p95 | p99 | device busy | host CPU |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| Graviton, 64 vCPU | c9g.16xlarge | 10M | 100 x 250 | 25,000 | 7.35 | 9.82 | 11.67 | -- | 71.0% |
| Graviton, 64 vCPU | c9g.16xlarge | 10M | 500 x 50 | 25,000 | 17.64 | 21.57 | 22.72 | -- | 66.6% |
| Graviton, 64 vCPU | c9g.16xlarge | 20M | 100 x 100 | 10,000 | 14.16 | 16.50 | 22.30 | -- | 65.0% |
| Graviton, 64 vCPU | c9g.16xlarge | 20M | 500 x 20 | 10,000 | 38.02 | 42.79 | 45.42 | -- | 62.6% |
| NVIDIA A10G | g5.2xlarge | 10M | 100 x 1000 | 100,000 | 2.63 | 3.14 | 3.50 | 66.2% | 25.5% |
| NVIDIA A10G | g5.2xlarge | 10M | 500 x 200 | 100,000 | 7.37 | 8.32 | 8.79 | 59.5% | 17.1% |
| NVIDIA A10G | g5.2xlarge | 20M | 100 x 720 | 72,000 | 4.17 | 5.41 | 7.45 | 97.6% | 31.9% |
| NVIDIA A10G | g5.2xlarge | 20M | 500 x 150 | 75,000 | 11.86 | 13.44 | 15.66 | 98.9% | 22.2% |
| Trainium1, 2 cores | trn1.2xlarge | 10M | 100 x 1000 | 100,000 | 3.48 | 4.30 | 5.07 | 75.2% | 33.3% |
| Trainium1, 2 cores | trn1.2xlarge | 10M | 500 x 200 | 100,000 | 11.98 | 13.39 | 14.12 | 76.5% | 28.1% |
| Trainium1, 2 cores | trn1.2xlarge | 20M | 100 x 450 | 45,000 | 5.10 | 5.73 | 6.18 | 81.8% | 26.8% |
| Trainium1, 2 cores | trn1.2xlarge | 20M | 500 x 80 | 40,000 | 22.20 | 23.21 | 23.46 | 74.5% | 22.1% |

Only the 20M rates were swept to failure: the A10G sustained 100 x 720 and shed 100 x 750
at 732.5 req/s. So 100,000 and 25,000 rows/s are the highest measured, not limits.
`device busy` is occupancy, the share of sample periods with the device executing, not how
full it was.

---

## 2. Evaluation on a 100M model: FP32 against BF16

One graph of 100,561,925 parameters in 31 encoder layers, built twice. The exported ONNX
file stores 100,498,437 values, lower because untrained LayerNorms all hold the same
all-ones scale and all-zeros bias and the export deduplicates them. Structure and
arithmetic: [`scripts/benchmark-models.md`](../../scripts/benchmark-models.md).
On the A10G both arms build the same ONNX bytes into a TensorRT engine, BF16 enabled
additively per layer through `trt_bf16_enable=1`. On Trainium each arm is a separate
`neuronx-cc` compile, `--auto-cast=none` against `--auto-cast=matmult --auto-cast-type=bf16`,
which casts the matmul engine and leaves the rest at FP32. Same wire and log rate as
section 1. One row per device, precision and request shape, at the highest rate that arm
sustained.
Source: [`featurewire-log20-prec-g5-trn1-100M`](featurewire-log20-prec-g5-trn1-100M/summary.md).

| device | precision | rows x TPS | rows/s | p50 | p95 | p99 | graph | vectorize | device busy | host CPU |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | 10,000 | 10.97 | 11.34 | 11.57 | 9.817 | 0.249 | 93.3% | 14.4% |
| A10G | BF16 | 100 x 200 | 20,000 | **5.85** | 6.30 | 6.65 | 4.685 | 0.236 | 89.1% | 15.6% |
| A10G | FP32 | 500 x 20 | 10,000 | 44.00 | 44.52 | 44.82 | 40.757 | 1.112 | 79.8% | 11.9% |
| A10G | BF16 | 500 x 40 | 20,000 | **22.10** | 22.57 | 22.94 | 18.311 | 1.106 | 72.7% | 12.0% |
| Trainium1 | FP32 | 100 x 70 | 7,000 | 22.74 | 23.02 | 23.10 | 21.734 | 0.228 | 75.4% | 20.2% |
| Trainium1 | BF16 | 100 x 100 | 10,000 | **16.44** | 16.72 | 16.80 | 15.473 | 0.225 | 76.4% | 20.7% |
| Trainium1 | FP32 | 500 x 10 | 5,000 | 117.22 | 118.16 | 118.36 | 113.650 | 0.976 | 56.6% | 15.0% |
| Trainium1 | BF16 | 500 x 20 | 10,000 | **75.08** | 76.03 | 76.22 | 71.866 | 0.926 | 71.5% | 19.2% |

`graph` is hushar's `InferenceTime` and `vectorize` its `VectorizationTime`, medians of
whole minute buckets inside the window. BF16 cuts device time 2.10x at 100 rows and 2.23x at
500 on the A10G, 1.40x and 1.58x on Trainium. The paired arms sustained different rates,
which does not enter the ratio: device time is flat with the offered rate below saturation.

**Trainium's rates come from a measured ceiling.** Offered the A10G's rates, all four
Trainium cells shed at 97.2-97.9% NeuronCore occupancy: 90.9 of 100 req/s and 127.1 of 200
at 100 rows, 17.4 of 20 and 27.4 of 40 at 500. One `trn1.2xlarge` is two NeuronCores, each
executing one NEFF at a time.

**The precision labels rest on artifacts.** The BF16 engine and NEFF are 0.43x to 0.51x of
their FP32 pair, and Trainium's compiled-versus-PyTorch maximum error is 1.48e-06 at
`--auto-cast=none` against 2.32e-03 at `--auto-cast=matmult`. Compiling with no flag
produced a NEFF byte-identical to `--auto-cast=none`, so the earlier Trainium rounds passed
no flag and were still FP32, which makes section 1's Trainium rows comparable with this
section's FP32 arm.

---

## 3. hushar against Triton

`g5.2xlarge`, A10G, TensorRT 10.16.1, driver 595.112.07, the same library for both stacks.
Both take the finished 690-float tensor on the wire and write no inference log, so this is
the inference path alone. Each stack runs 2 sessions or instances except where marked.
`gap` is `hushar p50 - Triton p50`; `device busy` and `host CPU` are hushar / Triton. Sources:
[`nominibatch-sessions-gpu-g5-10M-20M`](nominibatch-sessions-gpu-g5-10M-20M/summary.md),
[`nominibatch-precision-gpu-g5-100M`](nominibatch-precision-gpu-g5-100M/summary.md).

| model | precision | rows x TPS | hushar p50 | Triton p50 | gap | ratio | hushar p99 | Triton p99 | device busy | host CPU |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10M | FP32 | 100 x 1000 | 2.48 | **1.41** | 1.07 | 1.76x | 3.27 | 8.52 | 67.1% / 70.9% | 21.9% / 9.7% |
| 10M | FP32 | 500 x 200 | 8.75 | **5.09** | 3.66 | 1.72x | 10.61 | 5.32 | 59.5% / 59.9% | 17.1% / 4.2% |
| 20M | FP32 | 100 x 750 | 4.39 | **3.38** \* | 1.01 | 1.30x | 5.40 | 4.82 | 100.0% / 99.7% | 32.1% / 8.6% |
| 20M | FP32 | 500 x 150 | 14.19 | **9.29** | 4.90 | 1.53x | 16.25 | 9.95 | 99.3% / 99.9% | 22.1% / 4.0% |
| 100M | FP32 | 100 x 100 | 11.05 | **10.10** | 0.95 | 1.09x | 11.55 | 10.35 | 93.5% / 93.5% | 14.2% / 3.6% |
| 100M | FP32 | 500 x 20 | 45.90 | **42.11** | 3.79 | 1.09x | 47.09 | 42.36 | 79.9% / 79.9% | 12.1% / 2.1% |
| 100M | BF16 | 100 x 200 | 6.31 | **5.30** | 1.01 | 1.19x | 6.96 | 5.69 | 89.4% / 89.8% | 15.1% / 5.1% |
| 100M | BF16 | 500 x 40 | 24.20 | **20.25** | 3.95 | 1.20x | 26.19 | 20.65 | 72.6% / 72.2% | 12.1% / 2.7% |

Both stacks held every offered rate, so the difference is latency and not capacity. The
device is equally busy under both stacks, within 0.6 points in every cell but 10M 100 x 1000
(67.1% against 70.9%), so the gap is not the engine. It tracks the request shape rather than
the model: 0.95-1.07 ms at 100 rows and 3.66-4.90 ms at 500, across a 10x range of parameter
count and both precisions. Host CPU is 12-32% on hushar against 2-10% on Triton.

\* Triton at **4 instances**, the only cell not at 2, where it is unusually slow:
5.08 ms against 3.38 at 4 and 3.44 at 8. The count makes no difference at 20M 500 x 150
(9.29 / 9.28 / 9.30 at 2 / 4 / 8), so that cell stays at 2.

---

## Appendix A. Mini batch: worth having on CPU, a loss on GPU

`mini_batch_size` cuts a request into pieces scored at the same time on a thread pool. A
size equal to the request's rows is one execution, the baseline here; `mini batch` is the
lowest-latency split that sustained the rate in that group. Both runs are on the feature
wire; the CPU run logs at 20%, the GPU run has logging off and a 120s warm-up. CPU rows are
the 4-session groups. `device busy` and `host CPU` are one execution / split. Sources:
[`minibatch-size-cpu-c9g-10M-20M`](minibatch-size-cpu-c9g-10M-20M/summary.md),
`minibatch-gpu-g5-10M-20M` (git `635c57f4`).

| instance | device | model | rows x TPS | one execution | mini batch | pieces | split p50 | effect | device busy | host CPU |
|---|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| c9g.16xlarge | Graviton, 64 vCPU | 10M | 100 x 250 | 17.08 | 5 | 20 | **7.35** | 2.32x faster | -- | 70.7% / 71.0% |
| c9g.16xlarge | Graviton, 64 vCPU | 10M | 500 x 50 | 71.15 | 10 | 50 | **17.64** | 4.03x faster | -- | 69.8% / 66.6% |
| c9g.16xlarge | Graviton, 64 vCPU | 20M | 100 x 100 | 35.63 | 5 | 20 | **14.16** | 2.52x faster | -- | 67.4% / 65.0% |
| c9g.16xlarge | Graviton, 64 vCPU | 20M | 500 x 20 | 163.87 | 10 | 50 | **38.02** | 4.31x faster | -- | 66.9% / 62.6% |
| g5.2xlarge | NVIDIA A10G | 10M | 100 x 1000 | **2.64** | 50 | 2 | 3.09 | 1.17x slower | 67.0% / 86.5% | 24.1% / 33.1% |
| g5.2xlarge | NVIDIA A10G | 10M | 500 x 200 | 7.48 | 250 | 2 | 7.20 | level | 59.5% / 62.4% | 16.3% / 17.7% |
| g5.2xlarge | NVIDIA A10G | 20M | 100 x 750 | **4.74** | -- | -- | -- | no split held the rate | 99.6% / -- | 37.9% / -- |
| g5.2xlarge | NVIDIA A10G | 20M | 500 x 150 | **12.34** | 250 | 2 | 15.38 | 1.25x slower | 99.6% / 99.0% | 21.9% / 29.1% |

On 64 cores the pieces reach distinct cores at the same host CPU, 67-71% for one execution
against 63-71% split. On one A10G they do not: a request leases one ONNX Runtime session and
shares it across its mini batches, and the TensorRT provider serializes concurrent `Run()`
on one session, so the pieces queue on one device context and the host does more work for
it, +1.4 to +9.0 points of CPU. `graph` rose by up to +4.723 ms against a largest `vectorize`
saving of -0.709 ms. Three GPU split cells failed a rate their own baseline held, the worst
serving 324.8 of 750 req/s at 26.6% host CPU.

The lever that works on the A10G is the opposite one, merging. At equal rows/s the 500-row
request cost 15.0 us per row against the 100-row request's 26.4 at 10M, and 24.7 against
47.4 at 20M, because row parallelism is consumed inside the kernel.
