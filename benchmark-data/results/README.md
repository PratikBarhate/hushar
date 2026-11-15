# Benchmark results

Latency in milliseconds. Every cell held its offered rate for 15 slices of a 15-minute
window after a 5-minute warm-up, unless marked. Model structure:
[`scripts/benchmark-models.md`](../../scripts/benchmark-models.md).

---

## 1. Mini-batch size on CPU

`c9g.16xlarge`, 64 vCPU Graviton, ONNX Runtime 1.28.0, fp32. 4 sessions, 64 threads.
Mini-batch size is the only variable within a group. `vs 1 execution` is against that
group's `mini-batch size = rows` cell, which is one execution per request.
Source: [`minibatch-size-cpu-c9g-10M-20M`](minibatch-size-cpu-c9g-10M-20M/summary.md).

### 10M

| rows x TPS | mini-batch size | mini-batches | p50 | p95 | p99 | CPU | vs 1 execution |
|---|--:|--:|--:|--:|--:|--:|--:|
| 100 x 250 | none | 1 | 16.55 \* | 2254.80 | 2273.94 | 70% | -- |
| 100 x 250 | 100 | 1 | 17.08 | 24.52 | 121.57 | 71% | 1.00x |
| 100 x 250 | 50 | 2 | 14.90 | 15.79 | 16.35 | 70% | 1.15x |
| 100 x 250 | 10 | 10 | 8.25 | 8.56 | 9.13 | 66% | 2.07x |
| 100 x 250 | **5** | 20 | **7.35** | 9.82 | 11.67 | 71% | **2.32x** |
| 500 x 50 | none | 1 | 70.30 | 78.12 | 87.02 | 69% | -- |
| 500 x 50 | 500 | 1 | 71.15 | 135.02 | 1197.10 | 70% | 1.00x |
| 500 x 50 | 100 | 5 | 47.64 | 49.26 | 50.18 | 66% | 1.49x |
| 500 x 50 | 50 | 10 | 36.77 | 37.54 | 37.96 | 65% | 1.94x |
| 500 x 50 | **10** | 50 | **17.64** | 21.57 | 22.72 | 67% | **4.03x** |

### 20M

| rows x TPS | mini-batch size | mini-batches | p50 | p95 | p99 | CPU | vs 1 execution |
|---|--:|--:|--:|--:|--:|--:|--:|
| 100 x 100 | none | 1 | 36.36 | 57.30 | 63.59 | 68% | -- |
| 100 x 100 | 100 | 1 | 35.63 | 56.85 | 57.82 | 67% | 1.00x |
| 100 x 100 | 50 | 2 | 32.75 | 34.03 | 34.75 | 66% | 1.09x |
| 100 x 100 | 10 | 10 | 17.16 | 17.51 | 17.73 | 62% | 2.08x |
| 100 x 100 | **5** | 20 | **14.16** | 16.50 | 22.30 | 65% | **2.52x** |
| 500 x 20 | none | 1 | 163.23 | 176.31 | 269.62 | 67% | -- |
| 500 x 20 | 500 | 1 | 163.87 | 179.01 | 274.43 | 67% | 1.00x |
| 500 x 20 | 100 | 5 | 107.85 | 110.92 | 112.46 | 62% | 1.52x |
| 500 x 20 | 50 | 10 | 82.94 | 84.09 | 84.73 | 62% | 1.98x |
| 500 x 20 | **10** | 50 | **38.02** | 42.79 | 45.42 | 63% | **4.31x** |

Latency falls monotonically as the mini-batch size shrinks, in all four groups. The axis
never turned over, so 5 and 10 are the best measured rather than optima. 500-row requests
gain more than 100-row ones because they divide into more mini-batches. `none` and
`mini-batch size = rows` land within 0.4-3.1% of each other, so the split path costs
nothing measurable when it yields a single mini-batch.

\* `none` at 10M 100 x 250 held 12 of 15 slices at 247.7 req/s. That load is at the host's
edge without a split, which is why its p95 is what it is.

---

## 2. Hushar across hardware

Feature processing on the wire (210 features, 690 floats per row), inference log at 20%.
Accelerator rows are 2 sessions scoring a whole request; CPU rows are 4 sessions splitting
it. Each row is at that hardware's own sustainable rate, so the rates differ by design.
Source: [`featurewire-log20-g5-trn1-10M-20M`](featurewire-log20-g5-trn1-10M-20M/summary.md)
for GPU and Trainium, section 1's run for CPU.

| accelerator | chip | instance | model | rows x TPS | p50 | p95 | p99 | host CPU | device busy |
|---|---|---|---|--:|--:|--:|--:|--:|--:|
| CPU | Graviton, 64 vCPU | c9g.16xlarge | 10M | 100 x 250 | 7.35 | 9.82 | 11.67 | 71% | -- |
| CPU | Graviton, 64 vCPU | c9g.16xlarge | 10M | 500 x 50 | 17.64 | 21.57 | 22.72 | 67% | -- |
| CPU | Graviton, 64 vCPU | c9g.16xlarge | 20M | 100 x 100 | 14.16 | 16.50 | 22.30 | 65% | -- |
| CPU | Graviton, 64 vCPU | c9g.16xlarge | 20M | 500 x 20 | 38.02 | 42.79 | 45.42 | 63% | -- |
| GPU | NVIDIA A10G | g5.2xlarge | 10M | 100 x 1000 | 2.63 | 3.14 | 3.50 | 26% | 66.3% |
| GPU | NVIDIA A10G | g5.2xlarge | 10M | 500 x 200 | 7.37 | 8.32 | 8.79 | 17% | 59.5% |
| GPU | NVIDIA A10G | g5.2xlarge | 20M | 100 x 700 | 4.14 | 5.20 | 5.73 | 31% | 96.0% |
| GPU | NVIDIA A10G | g5.2xlarge | 20M | 500 x 150 | 11.86 | 13.44 | 15.66 | 22% | 98.9% |
| Trainium | Trainium1, 2 NeuronCores | trn1.2xlarge | 10M | 100 x 1000 | 3.48 | 4.30 | 5.07 | 33% | 75.2% |
| Trainium | Trainium1, 2 NeuronCores | trn1.2xlarge | 10M | 500 x 200 | 11.98 | 13.39 | 14.12 | 28% | 76.5% |
| Trainium | Trainium1, 2 NeuronCores | trn1.2xlarge | 20M | 100 x 450 | 5.10 | 5.73 | 6.18 | 27% | 81.8% |
| Trainium | Trainium1, 2 NeuronCores | trn1.2xlarge | 20M | 500 x 80 | 22.20 | 23.21 | 23.46 | 22% | 74.5% |

Read the rate with the latency. The GPU serves 100 x 1000 at 2.63 ms where the CPU serves
100 x 250 at 7.35 ms: four times the rate at a third of the latency. The CPU rows are the
best of their mini-batch sweep (sessions 4, mini batch 5 at 100 rows and 10 at 500); the
accelerator rows take whole requests, since splitting cannot overlap on them.

Host CPU is 17-33% on the accelerators against 63-71% on CPU, and that remainder is the
feature path and the log rather than inference. `device busy` is the accelerator's own
occupancy; on CPU the host **is** the device, so the preceding column is that figure.

**`device busy` is occupancy, not saturation.** It is nvidia-smi's `utilization.gpu` and
Neuron's `neuroncore_utilization`: the share of each sample period in which the device was
executing something. A stream of small calls reads high while the device still has
arithmetic headroom, so a higher number here does not mean more work done. Power draw is
the better saturation hint, and it moves with the model rather than the request shape: the
A10G drew 185-191 W at 10M against 277-289 W at 20M, where the occupancy readings of 96%
and 99% are near their ceiling either way.

---

## 3. Hushar against Triton

`g5.2xlarge`, A10G, TensorRT 10.16.1, driver 595.112.07, fp32. Both stacks take tensors on
the wire and write no inference log, so this is the inference path alone. Each stack runs
at its own best parallelism, which is 2 everywhere except where marked.
Source: [`nominibatch-sessions-gpu-g5-10M-20M`](nominibatch-sessions-gpu-g5-10M-20M/summary.md).

| model | rows x TPS | stack | p50 | p95 | p99 | host CPU | GPU busy |
|---|--:|---|--:|--:|--:|--:|--:|
| 10M | 100 x 1000 | hushar | 2.48 | 2.93 | 3.27 | 22% | 67.1% |
| 10M | 100 x 1000 | triton | **1.41** | 1.72 | 8.52 | 10% | 70.9% |
| 10M | 500 x 200 | hushar | 8.75 | 9.17 | 10.61 | 17% | 59.5% |
| 10M | 500 x 200 | triton | **5.09** | 5.31 | 5.32 | 4% | 59.9% |
| 20M | 100 x 750 | hushar | 4.39 | 5.12 | 5.40 | 32% | 100.0% |
| 20M | 100 x 750 | triton | **3.38** \*\* | 3.71 | 4.82 | 9% | 99.7% |
| 20M | 500 x 150 | hushar | 14.19 | 15.68 | 16.25 | 22% | 99.3% |
| 20M | 500 x 150 | triton | **9.29** | 9.80 | 9.95 | 4% | 99.9% |

Triton is ahead on all four, by 1.0 to 4.9 ms. Both stacks held every offered rate, so the
gap is latency and not capacity.

At 20M the GPU is at its occupancy ceiling for both stacks (99-100%, drawing 298-299 W), so
neither is leaving the device idle; the difference is in how fast each gets a request onto
it. At 10M the device is 59-71% busy at 179-186 W, so both have headroom there.

\*\* Triton at **4 instances**, the only row not at 2. Two is Hushar's best in all four
cells and Triton's in the 10M pair, but Triton is unusually slow at 2 here: 5.08 ms against
3.38 at 4 and 3.44 at 8. At 20M 500 x 150 the count makes no difference to Triton
(9.29 / 9.28 / 9.30 at 2 / 4 / 8), so that row stays at 2.

---

## Reading these

- One host per run, so figures within a run compare cleanly and figures across runs carry a
  placement offset. Sections 1 and 2 cross hosts for the CPU-against-accelerator reading.
- p50/p95/p99 are client-observed round trips, including the network.
- `host CPU` is the server instance's own utilisation during the measurement window.
- Each run directory holds its full grid, per-cell JSON, server banners, and the driver that
  produced it.
