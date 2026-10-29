# Trainium `trn1.2xlarge`: does a mini batch lower hushar's latency?

One Trainium1, two NeuronCores, **one core per replica throughout**. Two things change down the grid: how many rows go into one NEFF call, and how many replicas serve. `minibatch-gpu-g5-10M-20M` asked the first question on one NVIDIA A10G, found the split bought nothing, and named the knob it lacked -- `execution_contexts_per_replica`, which exists only on the Neuron backend. This run asks it on the device that has it.

## Answer

| model | rows x TPS | unsplit p50 | best p50 | replicas | verdict |
|---|---:|---:|---:|---:|---|
| 10M | 100 x 1000 | 3.46 | 2.94 | 2 | **1.18x faster** at mini batch 50 |
| 10M | 500 x 200 | 12.35 | 10.96 | 4 | **1.13x faster** at mini batch 100 |
| 20M | 100 x 750 | -- | -- | -- | **no cell held the rate** (best 610.0, 81% of target) |
| 20M | 500 x 150 | -- | -- | -- | **no cell held the rate** (best 112.5, 75% of target) |

Both figures in a row are at the SAME replica count, so the mini batch is the only variable. 20M has no row because no 20M cell held its rate.

**1 · Splitting a request LOWERS its latency here, which is the opposite of the GPU result.** At 10M, 100 rows, 1000 req/s, cutting the request into two 50-row pieces takes the median from 3.46 ms to 2.94 ms (1.18x) and the p99 from 10.15 ms to 3.87 ms (**2.6x**). The tail moves further than the median. On Neuron each piece leases its own execution context, so one piece's host-device copy overlaps another's compute; on the GPU both pieces queued on one ONNX Runtime session.

**2 · Two pieces is the optimum, not a trend.** Four pieces is worse than two on every load and both models -- per-call overhead overtakes the overlap. At 10M, 100 rows, four pieces does not even hold the rate at one replica per core.

**3 · Replicas beyond one per core never help.** They add no capacity and cost latency:

| 10M 100 x 1000, mini batch 50 | p50 | p99 | CPU |
|---|---:|---:|---:|
| 2 replicas (1/core) | 2.94 | 3.87 | 36.2% |
| 4 replicas (2/core) | 3.23 | 4.29 | 37.9% |
| 6 replicas (3/core) | 3.29 | 4.62 | 37.9% |

The second replica per core only raises core occupancy, which matters when the device is the bottleneck and is wasted when it is not.

**4 · 20M does not fit on this box at either specified rate, and that is structural.** A NeuronCore executes one NEFF at a time and there are two, so the sustainable rate is bounded by `cores / device span`. At 20M that span is 3.859 ms unsplit, so 100 x 750 needs 2.89 lanes and the box has 2. Best achieved: **610.0 of 750** at 100 rows (81%) and **112.5 of 150** at 500 rows (75%).

Every saturated one-replica-per-core cell lands at 1.77-1.91 lane-seconds per second of the 2 available -- the device tops out near 90% core occupancy, and `achieved = occupancy x cores / device span` across both models. That is a description of six cells, not a fit to one.

**5 · A wider request is cheaper per candidate at 10M, and does not rescue 20M.** At the same 100,000 rows/s, 100-row requests cost 29.4 us per candidate against 21.8 for 500-row (1.35x). But at 20M the wider packaging delivers FEWER rows/s than the narrow one (56,250 against 61,000), so it is not a way around the lane wall.

## Every cell that held its rate

`device span` is hushar's own `InferenceTime`, the median over the measured window's minute buckets. On a split row it is the maximum over the request's pieces, which with one core per replica is the whole request's span on the device. `NC` is mean NeuronCore occupancy.

| model | rows x TPS | mini batch | pieces | replicas | p50 | p95 | p99 | device span | vectorize | NC | CPU | us/candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 100 x 1000 | 100 | 1 | 2 | 3.46 | 4.50 | 10.15 | 1.644 | 0.274 | 75.2% | 33.0% | 34.6 |
| 10M | 100 x 1000 | 100 | 1 | 4 | 3.40 | 3.95 | 4.44 | 1.862 | 0.267 | 75.2% | 33.2% | 34.0 |
| 10M | 100 x 1000 | 100 | 1 | 6 | 3.41 | 3.97 | 4.55 | 1.859 | 0.269 | 75.1% | 33.4% | 34.1 |
| 10M | 100 x 1000 | 50 | 2 | 2 | 2.94 | 3.51 | 3.87 | 1.549 | 0.208 | 72.6% | 36.2% | 29.4 |
| 10M | 100 x 1000 | 50 | 2 | 4 | 3.23 | 3.79 | 4.29 | 1.878 | 0.211 | 72.6% | 37.9% | 32.3 |
| 10M | 100 x 1000 | 50 | 2 | 6 | 3.29 | 3.86 | 4.62 | 1.858 | 0.213 | 72.6% | 37.9% | 32.9 |
| 10M | 100 x 1000 | 25 | 4 | 4 | 3.84 | 4.57 | 5.10 | 2.345 | 0.129 | 82.5% | 58.5% | 38.4 |
| 10M | 100 x 1000 | 25 | 4 | 6 | 3.84 | 4.70 | 5.87 | 2.350 | 0.132 | 82.5% | 58.6% | 38.4 |
| 10M | 500 x 200 | 500 | 1 | 4 | 12.35 | 16.32 | 17.96 | 8.271 | 1.009 | 76.5% | 28.1% | 24.7 |
| 10M | 500 x 200 | 500 | 1 | 6 | 12.42 | 16.82 | 18.01 | 8.285 | 1.020 | 76.5% | 28.3% | 24.8 |
| 10M | 500 x 200 | 250 | 2 | 2 | 11.18 | 12.65 | 13.42 | 7.846 | 0.811 | 76.7% | 29.0% | 22.4 |
| 10M | 500 x 200 | 250 | 2 | 4 | 11.21 | 14.69 | 17.56 | 7.889 | 0.816 | 76.7% | 29.4% | 22.4 |
| 10M | 500 x 200 | 250 | 2 | 6 | 11.25 | 15.14 | 17.74 | 7.911 | 0.805 | 76.7% | 29.5% | 22.5 |
| 10M | 500 x 200 | 100 | 5 | 2 | 10.90 | 11.92 | 12.53 | 7.505 | 0.440 | 75.1% | 37.3% | 21.8 |
| 10M | 500 x 200 | 100 | 5 | 4 | 10.96 | 12.13 | 13.63 | 7.511 | 0.439 | 75.2% | 37.5% | 21.9 |
| 10M | 500 x 200 | 100 | 5 | 6 | 10.98 | 12.20 | 14.64 | 7.513 | 0.439 | 75.2% | 37.6% | 22.0 |

## Cells that did not hold their rate

Reported by achieved rate and deliberately not placed beside a sustained cell. `partial` held some slices of the window and `failed` held none -- the distinction matters for the one 10M cell that missed by 0.1 req/s.

| model | rows x TPS | mini batch | pieces | replicas | state | achieved | shed | slices held | NC | CPU |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10M | 100 x 1000 | 25 | 4 | 2 | failed | 961.7 | 35,029 | 0/15 | 79.3% | 54.8% |
| 10M | 500 x 200 | 500 | 1 | 2 | partial | 199.9 | 60 | 14/15 | 76.5% | 28.1% |
| 20M | 100 x 750 | 100 | 1 | 2 | failed | 492.1 | 260,126 | 0/15 | 89.4% | 29.3% |
| 20M | 100 x 750 | 100 | 1 | 4 | failed | 538.1 | 213,129 | 0/15 | 97.8% | 33.5% |
| 20M | 100 x 750 | 100 | 1 | 6 | failed | 538.1 | 213,105 | 0/15 | 97.8% | 37.3% |
| 20M | 100 x 750 | 50 | 2 | 2 | failed | 566.3 | 184,434 | 0/15 | 90.3% | 33.2% |
| 20M | 100 x 750 | 50 | 2 | 4 | failed | 609.9 | 139,950 | 0/15 | 97.3% | 43.5% |
| 20M | 100 x 750 | 50 | 2 | 6 | failed | 610.0 | 139,754 | 0/15 | 97.3% | 52.3% |
| 20M | 100 x 750 | 25 | 4 | 2 | failed | 506.1 | 245,815 | 0/15 | 89.4% | 44.8% |
| 20M | 100 x 750 | 25 | 4 | 4 | failed | 550.3 | 200,706 | 0/15 | 97.3% | 52.7% |
| 20M | 100 x 750 | 25 | 4 | 6 | failed | 541.0 | 210,156 | 0/15 | 95.7% | 52.0% |
| 20M | 500 x 150 | 500 | 1 | 2 | failed | 99.1 | 51,388 | 0/15 | 92.2% | 29.5% |
| 20M | 500 x 150 | 500 | 1 | 4 | failed | 106.6 | 43,631 | 0/15 | 99.1% | 30.4% |
| 20M | 500 x 150 | 500 | 1 | 6 | failed | 106.6 | 43,632 | 0/15 | 99.1% | 34.3% |
| 20M | 500 x 150 | 250 | 2 | 2 | failed | 107.6 | 42,663 | 0/15 | 94.9% | 29.8% |
| 20M | 500 x 150 | 250 | 2 | 4 | failed | 112.5 | 37,655 | 0/15 | 99.3% | 38.9% |
| 20M | 500 x 150 | 250 | 2 | 6 | failed | 112.5 | 37,685 | 0/15 | 99.2% | 47.8% |
| 20M | 500 x 150 | 100 | 5 | 2 | failed | 106.1 | 44,227 | 0/15 | 96.4% | 40.2% |
| 20M | 500 x 150 | 100 | 5 | 4 | failed | 109.2 | 41,028 | 0/15 | 99.3% | 45.0% |
| 20M | 500 x 150 | 100 | 5 | 6 | failed | 108.8 | 41,447 | 0/15 | 98.9% | 46.0% |

## The contexts control

The grid sets `execution_contexts_per_replica = pieces`, on the service's own advice. These four cells re-run the most-split configuration of each load with **one** context, so that choice is priced rather than assumed. Each pair differs in exactly one field.

| model | rows x TPS | mini batch | pieces | contexts = pieces | contexts = 1 | worth |
|---|---:|---:|---:|---|---|---|
| 10M | 100 x 1000 | 25 | 4 | 961.7 req/s | 902.1 req/s | +6.6% throughput |
| 10M | 500 x 200 | 100 | 5 | 10.90 ms | 11.22 ms | 1.03x on the median |
| 20M | 100 x 750 | 25 | 4 | 506.1 req/s | 485.5 req/s | +4.2% throughput |
| 20M | 500 x 150 | 100 | 5 | 106.1 req/s | 103.9 req/s | +2.1% throughput |

Per-piece contexts are worth a few percent -- real, consistent in sign, and far smaller than the split itself. The split's gain is therefore mostly the shape of the call, not the overlap the contexts buy.

## Against the A10G, on the same graphs and the same loads

From `minibatch-gpu-g5-10M-20M`'s own report. The devices are contemporaries and the wire is identical; the graph is the same design at the same weight but not the same file, since Neuron compiles from PyTorch and the GPU run from ONNX.

| model | rows x TPS | A10G: split verdict | trn1: split verdict |
|---|---:|---|---|
| 10M | 100 x 1000 | 1.17x slower | 1.18x faster at mini batch 50 |
| 10M | 500 x 200 | level | 1.13x faster at mini batch 100 |
| 20M | 100 x 750 | no split held the rate | no cell held the rate (best 610.0, 81% of target) |
| 20M | 500 x 150 | 1.25x slower | no cell held the rate (best 112.5, 75% of target) |

The split pays where the pieces reach something that can run them at once. On one A10G reached through a single ONNX Runtime session they could not. On a NeuronCore with per-piece contexts they can -- and the same hardware fact that makes the split work is what caps 20M: two cores, each running one NEFF at a time.

## What was measured

| | |
|---|---|
| server | `trn1.2xlarge` -- 1 Trainium1, 2 NeuronCores, 32 GB device memory, 8 vCPU |
| client | `m9g.8xlarge`, same availability zone |
| precision | FP32 throughout |
| 10M model | 9,142,789 parameters, 2 layers, `[batch, 690]` -> `[batch, 5]`; NEFFs b25, b50, b100, b250, b500 |
| 20M model | 18,599,941 parameters, 5 layers, `[batch, 690]` -> `[batch, 5]`; NEFFs b25, b50, b100, b250, b500 |
| toolchain | torch_neuronx 2.9.0.2.15.32035, libnrt 2.33.1.0, `Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)` |
| window | 120 s warm-up discarded, then 900 s measured as one continuous load in 15 slices; fresh server per cell, stopped afterwards, 60 s idle |
| cells | 40 measured: 17 sustained, 1 partial, 22 failed the rate; 0 harness failures |

### Limits

- **One core per replica, by instruction.** So this run does not say what a split does when the pieces reach DIFFERENT cores, which is the configuration `config-sweep-neuron-trn1-10M-50M` found worth 41% on 50M. That is a separate experiment.
- **The rates were specified, not probed.** Both 20M rungs turned out to be above the ceiling, so 20M is a capacity result rather than a latency one.
- **10 cells carry no server-side timings.** All are 20M 500-row cells, and the service says why in its own log: `metrics: 4 sends still in flight, so 300 buffered records were dropped to keep the newest; the destination is not keeping up`. CloudWatch could not absorb the metric rate under the heaviest overload. Every affected cell is unsustained and so enters no latency comparison; their achieved rates and NeuronCore occupancy are independent of CloudWatch and stand.
- **`device span` on a split row is the maximum over the pieces**, not their sum, because the service reports one duration per stage per request. It is the request's span on the device, which is what the lane arithmetic needs, and is not a per-piece figure.
- **NeuronCore occupancy is occupancy, not saturation** -- the share of a sample period with work resident. A split produces a stream of smaller executions and so reads high on a core with headroom.
- **No cell was repeated**, so no same-host spread was measured here. Two medians within 8.4% are reported as `level`, inherited from `minibatch-rows-cpu-arm64-10M`, which measured that spread on a split configuration.

Evidence is under `raw/`: `results.json` is this report's only input, with cell records, resolved server configurations, banners, per-second host and NeuronCore samples, and the compiled-artifact provenance. `raw/driver/` carries the scripts that ran.
