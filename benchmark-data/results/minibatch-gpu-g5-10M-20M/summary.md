# NVIDIA A10G: does a mini batch lower hushar's latency?

One A10G, one ONNX graph, one TensorRT. hushar scores it through ONNX Runtime's TensorRT execution provider, and the only thing that changes down each group of cells is **how many rows go into one graph call**. `mini_batch_size` equal to the request's rows is one call and is the baseline; anything smaller cuts the request into pieces that are handed to a thread pool and scored at the same time. `minibatch-rows-cpu-arm64-10M` found that knob worth up to 2.64x on a 64-core CPU. This run asks what it is worth on a GPU.

## Answer

| model | rows x TPS | rows/s | unsplit p50 | best split | its p50 | verdict |
|---|---:|---:|---:|---:|---:|---|
| 10M | 100 x 1000 | 100,000 | 2.64 | 50 (2 pieces) | 3.09 | **1.17x slower** |
| 10M | 500 x 200 | 100,000 | 7.48 | 250 (2 pieces) | 7.20 | **level** |
| 20M | 100 x 750 | 75,000 | 4.74 | -- | -- | no split sustained the rate |
| 20M | 500 x 150 | 75,000 | 12.34 | 250 (2 pieces) | 15.38 | **1.25x slower** |

**1 · Splitting a request did not lower its latency at any load measured.** Of 4 loads, 2 were slower than their own unsplit baseline, 1 was inside this run's 8.4% tie band, and on 1 no split held the offered rate at all. That is the opposite of `minibatch-rows-cpu-arm64-10M`, where the same knob was worth 2.10x at 100 rows and 2.64x at 200 -- and the reason is the next finding rather than the device.

**2 · The term a split can improve is too small to pay for the term it makes worse.** Across the unsplit cells vectorization is 6% to 15% of the client median, so even driving it to zero cannot buy what the extra kernel launches and host-device copies cost. One ONNX Runtime session is leased per REQUEST and shared by all of its mini batches, and the TensorRT provider serializes concurrent `Run()` on one session, so the pieces fan out for the feature work and then queue on the device. `execution_contexts_per_replica`, the knob that would give each piece its own device context, exists only on the Neuron backend.

**3 · The two terms really do move in opposite directions, and the device term is the larger by an order of magnitude.** On the sustained splits `graph` rose by as much as +4.723 ms while the largest `vectorize` saving anywhere was -0.709 ms -- both on 20M 500 x 150 at mini batch 100, the same cell. So the split does exactly what it was designed to do to the feature path, and the graph then eats roughly 7x the gain. That ratio, not the mini-batch size, is the result.

**4 · At enough pieces the split stops being a latency question and breaks capacity.** 3 split cells failed the rate its own unsplit baseline held: worst was 20M 100 x 750 at mini batch 25, which served 324.8 of 750 req/s and shed 430,622 at only 26.6% CPU. **Shedding at low CPU is the serialization signature** -- nothing is saturated, the pieces are queued behind one session's execution context. A split multiplies the graph calls a request makes, so it multiplies the rate the device is asked to serve.

**5 · Even where the split is harmless it is not free.** Against its own unsplit baseline a sustained split cell ran +1.4 to +11.7 percentage points of whole-host CPU, because the fan-out pool is `num_cpus` wide and a request now occupies several of its threads instead of one. On a CPU host that cost bought cores; here there is nothing to buy.

**6 · The lever that works on this device is the opposite one.** 10M at 100,000 rows/s, 100 rows cost 26.4 us per candidate against 500 rows at 15.0, so the 500-row request is 1.76x cheaper per candidate; 20M at 75,000 rows/s, 100 rows cost 47.4 us per candidate against 500 rows at 24.7, so the 500-row request is 1.92x cheaper per candidate. Row parallelism is consumed INSIDE the kernel, so filling the device with a wider batch is what uses its headroom -- and a split is a move in the other direction. Note the CPU column while reading it: the larger request is cheaper per candidate AND cheaper on the host.

**7 · Triton moves the same way under the packaging change, which is what puts that result on the device rather than on hushar.** 10M 100 rows 16.0 us per candidate against 500 rows at 10.6; 20M 100 rows 36.0 us per candidate against 500 rows at 19.3. It has no counterpart to the split, so it is a fixed line here and not a variable -- and its ONNX Runtime version is NOT matched to hushar's and is not claimed to be, because the `tensorrt` backend does not use ONNX Runtime at all. What is matched is TensorRT, the library that executes the graph.

**8 · On one load Triton's median advantage is bought with a worse tail.** 10M 100 x 1000, where Triton's p99 is 8.52 ms against hushar's 3.32. Everywhere else it leads at both. Worth stating because a median comparison is the wrong one to ship against a tail objective.

**9 · This is a result about one GPU, not about mini batching.** The split pays where the pieces reach DISTINCT compute units: `minibatch-rows-cpu-arm64-10M` measured 2.10x at 100 rows on 64 cores, and the Trainium campaign cut device time 41-49% by spanning two NeuronCores with one replica. Both had somewhere for the second piece to go. One A10G reached through a single ONNX Runtime session does not, and no mini-batch size can create one. The change that would make this experiment different is per-piece context leasing, not a different mini batch.

## Every mini batch, against its own baseline

One group per load. The mini batch is the only thing that changes inside a group: same graph, same offered rate, same session count, same host, and the unsplit cell at the top is what the column divides by. Only cells that **sustained** their offered rate appear here -- every slice of the measured window serving at least 98% of it with nothing shed and nothing failed -- because the median of a rate that was not held is queue depth and not a latency.

| model | rows x TPS | mini batch | pieces | p50 ms | p95 ms | p99 ms | against unsplit | GPU | CPU | cell |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 10M | 100 x 1000 | 100 | 1 | 2.64 | 3.11 | 3.32 | baseline | 67.0% | 24.1% | `g5-hushar-10m-r100t1000-mb100-p2` |
| 10M | 100 x 1000 | 50 | 2 | 3.09 | 3.59 | 3.83 | 1.17x slower | 86.5% | 33.1% | `g5-hushar-10m-r100t1000-mb50-p2` |
| 10M | 500 x 200 | 500 | 1 | 7.48 | 8.36 | 8.72 | baseline | 59.5% | 16.3% | `g5-hushar-10m-r500t200-mb500-p2` |
| 10M | 500 x 200 | 250 | 2 | 7.20 | 8.13 | 8.43 | level | 62.4% | 17.7% | `g5-hushar-10m-r500t200-mb250-p2` |
| 10M | 500 x 200 | 100 | 5 | 7.35 | 8.32 | 8.58 | level | 68.8% | 21.8% | `g5-hushar-10m-r500t200-mb100-p2` |
| 20M | 100 x 750 | 100 | 1 | 4.74 | 5.50 | 5.89 | baseline | 99.6% | 37.9% | `g5-hushar-20m-r100t750-mb100-p4` |
| 20M | 500 x 150 | 500 | 1 | 12.34 | 13.66 | 14.17 | baseline | 99.6% | 21.9% | `g5-hushar-20m-r500t150-mb500-p4` |
| 20M | 500 x 150 | 250 | 2 | 15.38 | 16.63 | 17.22 | 1.25x slower | 99.0% | 29.1% | `g5-hushar-20m-r500t150-mb250-p4` |
| 20M | 500 x 150 | 100 | 5 | 16.63 | 17.53 | 17.94 | 1.35x slower | 99.2% | 33.6% | `g5-hushar-20m-r500t150-mb100-p4` |

## Where the median goes

`graph` is hushar's own `InferenceTime` and `vectorize` its `VectorizationTime`, both as the median over the measured window's minute buckets. Read the two columns in opposite directions: the split is what makes `graph` worse and `vectorize` better, and this table is the whole of the answer above.

**On a split cell both figures are the MAXIMUM over that request's mini batches**, not their sum, because the service reports one duration per stage per request and the slowest piece sets the request's span. With the pieces serialized on the device that is neither total device work nor a clean per-piece figure, so `everything else` -- which is a subtraction -- is left blank there and computed only for the unsplit cells, where it is exact.

| model | rows x TPS | mini batch | pieces | p50 ms | graph ms | vectorize ms | everything else | graph share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 100 x 1000 | 100 | 1 | 2.64 | 0.852 | 0.248 | 1.540 | 32% |
| 10M | 100 x 1000 | 50 | 2 | 3.09 | 1.274 | 0.187 | n/a (split) | 41% (max piece) |
| 10M | 500 x 200 | 500 | 1 | 7.48 | 3.171 | 1.113 | 3.196 | 42% |
| 10M | 500 x 200 | 250 | 2 | 7.20 | 3.156 | 0.591 | n/a (split) | 44% (max piece) |
| 10M | 500 x 200 | 100 | 5 | 7.35 | 3.719 | 0.426 | n/a (split) | 51% (max piece) |
| 20M | 100 x 750 | 100 | 1 | 4.74 | 2.879 | 0.262 | 1.599 | 61% |
| 20M | 500 x 150 | 500 | 1 | 12.34 | 7.890 | 1.122 | 3.328 | 64% |
| 20M | 500 x 150 | 250 | 2 | 15.38 | 10.934 | 0.663 | n/a (split) | 71% (max piece) |
| 20M | 500 x 150 | 100 | 5 | 16.63 | 12.613 | 0.413 | n/a (split) | 76% (max piece) |

## The other lever: 100 rows against 500, at the same rows per second

Both cells unsplit, both offering the same rows per second, so only the packaging differs. This is the comparison a GPU actually rewards and it falls out of the grid for free: on a CPU the lever is to SPLIT a request to win cores, on a GPU it is to MERGE requests to fill the device. `per candidate` is the client median divided by the rows in the request.

| model | rows/s | packaging | req/s | p50 ms | per candidate us | graph ms | GPU | CPU | per candidate, against the other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10M | 100,000 | 100 rows | 1000 | 2.64 | 26.4 | 0.852 | 67.0% | 24.1% | 1.76x slower than 500 rows |
| 10M | 100,000 | 500 rows | 200 | 7.48 | 15.0 | 3.171 | 59.5% | 16.3% | 1.76x faster than 100 rows |
| 20M | 75,000 | 100 rows | 750 | 4.74 | 47.4 | 2.879 | 99.6% | 37.9% | 1.92x slower than 500 rows |
| 20M | 75,000 | 500 rows | 150 | 12.34 | 24.7 | 7.890 | 99.6% | 21.9% | 1.92x faster than 100 rows |

## The reference line: Triton on the same packaging

Triton is **not a variable in this run** -- it has no counterpart to the mini batch, so it runs one unsplit cell per load. It is here to separate *the split did nothing* from *the packaging did nothing*: a row count that is worse on both stacks is the device, not hushar. The wires differ, as in every report in this series: Triton's plan takes the finished 690-float tensor, so the 210 transformations move to its caller, and hushar's `vectorize` column above is a lower bound on what that costs outside Triton's number.

| model | rows x TPS | hushar unsplit p50 | Triton p50 | lower median | hushar p99 | Triton p99 | Triton cell |
|---|---:|---:|---:|---|---:|---:|---|
| 10M | 100 x 1000 | 2.64 | 1.60 | Triton 1.65x | 3.32 | 8.52 | `g5-triton-10m-r100t1000-mb100-p2` |
| 10M | 500 x 200 | 7.48 | 5.29 | Triton 1.41x | 8.72 | 5.48 | `g5-triton-10m-r500t200-mb500-p2` |
| 20M | 100 x 750 | 4.74 | 3.60 | Triton 1.32x | 5.89 | 5.28 | `g5-triton-20m-r100t750-mb100-p4` |
| 20M | 500 x 150 | 12.34 | 9.66 | Triton 1.28x | 14.17 | 10.45 | `g5-triton-20m-r500t150-mb500-p4` |

## Cells that did not hold their rate

Reported by achieved rate and deliberately not placed beside a sustained cell: the latency of a rate that was not held is queue depth and has no median worth comparing.

| model | rows x TPS | stack | mini batch | result | achieved | shed | failed | slices held | GPU | CPU | cell |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 10M | 100 x 1000 | hushar | 25 | no | 823.6 | 175,592 | 0 | 0/15 | 84.9% | 39.8% | `g5-hushar-10m-r100t1000-mb25-p2` |
| 20M | 100 x 750 | hushar | 50 | no | 571.6 | 178,092 | 0 | 0/15 | 100.0% | 57.2% | `g5-hushar-20m-r100t750-mb50-p4` |
| 20M | 100 x 750 | hushar | 25 | no | 324.8 | 430,622 | 0 | 0/15 | 77.2% | 26.6% | `g5-hushar-20m-r100t750-mb25-p4` |

## Every cell

| model | rows x TPS | stack | mini batch | pieces | shape | result | p50 | p95 | p99 | graph | vectorize | GPU | W | CPU | shed | cell |
|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10M | 100 x 1000 | hushar | 100 | 1 | [100, 690] | ok 1000.0/1000 | 2.64 | 3.11 | 3.32 | 0.852 | 0.248 | 67.0% | 188 | 24.1% | 0 | `g5-hushar-10m-r100t1000-mb100-p2` |
| 10M | 100 x 1000 | hushar | 50 | 2 | [50, 690] | ok 1000.0/1000 | 3.09 | 3.59 | 3.83 | 1.274 | 0.187 | 86.5% | 250 | 33.1% | 0 | `g5-hushar-10m-r100t1000-mb50-p2` |
| 10M | 100 x 1000 | hushar | 25 | 4 | [25, 690] | no 823.6/1000 | 4851.65 | 4905.81 | 4986.71 | 2.145 | 0.158 | 84.9% | 198 | 39.8% | 175,592 | `g5-hushar-10m-r100t1000-mb25-p2` |
| 10M | 100 x 1000 | triton | n/a | 1 | [100, 690] | ok 1000.0/1000 | 1.60 | 1.89 | 8.52 | -- | -- | 71.2% | 188 | 9.7% | 0 | `g5-triton-10m-r100t1000-mb100-p2` |
| 10M | 500 x 200 | hushar | 500 | 1 | [500, 690] | ok 200.0/200 | 7.48 | 8.36 | 8.72 | 3.171 | 1.113 | 59.5% | 184 | 16.3% | 0 | `g5-hushar-10m-r500t200-mb500-p2` |
| 10M | 500 x 200 | hushar | 250 | 2 | [250, 690] | ok 200.0/200 | 7.20 | 8.13 | 8.43 | 3.156 | 0.591 | 62.4% | 171 | 17.7% | 0 | `g5-hushar-10m-r500t200-mb250-p2` |
| 10M | 500 x 200 | hushar | 100 | 5 | [100, 690] | ok 200.0/200 | 7.35 | 8.32 | 8.58 | 3.719 | 0.426 | 68.8% | 186 | 21.8% | 0 | `g5-hushar-10m-r500t200-mb100-p2` |
| 10M | 500 x 200 | triton | n/a | 1 | [500, 690] | ok 200.0/200 | 5.29 | 5.42 | 5.48 | -- | -- | 59.9% | 189 | 4.4% | 0 | `g5-triton-10m-r500t200-mb500-p2` |
| 20M | 100 x 750 | hushar | 100 | 1 | [100, 690] | ok 750.0/750 | 4.74 | 5.50 | 5.89 | 2.879 | 0.262 | 99.6% | 298 | 37.9% | 0 | `g5-hushar-20m-r100t750-mb100-p4` |
| 20M | 100 x 750 | hushar | 50 | 2 | [50, 690] | no 571.6/750 | 5246.94 | 5251.78 | 5253.87 | 6.713 | 0.258 | 100.0% | 299 | 57.2% | 178,092 | `g5-hushar-20m-r100t750-mb50-p4` |
| 20M | 100 x 750 | hushar | 25 | 4 | [25, 690] | no 324.8/750 | 9227.84 | 9295.01 | 9514.42 | 11.114 | 0.157 | 77.2% | 185 | 26.6% | 430,622 | `g5-hushar-20m-r100t750-mb25-p4` |
| 20M | 100 x 750 | triton | n/a | 1 | [100, 690] | ok 750.0/750 | 3.60 | 3.90 | 5.28 | -- | -- | 99.7% | 299 | 8.4% | 0 | `g5-triton-20m-r100t750-mb100-p4` |
| 20M | 500 x 150 | hushar | 500 | 1 | [500, 690] | ok 150.0/150 | 12.34 | 13.66 | 14.17 | 7.890 | 1.122 | 99.6% | 299 | 21.9% | 0 | `g5-hushar-20m-r500t150-mb500-p4` |
| 20M | 500 x 150 | hushar | 250 | 2 | [250, 690] | ok 150.0/150 | 15.38 | 16.63 | 17.22 | 10.934 | 0.663 | 99.0% | 292 | 29.1% | 0 | `g5-hushar-20m-r500t150-mb250-p4` |
| 20M | 500 x 150 | hushar | 100 | 5 | [100, 690] | ok 150.0/150 | 16.63 | 17.53 | 17.94 | 12.613 | 0.413 | 99.2% | 299 | 33.6% | 0 | `g5-hushar-20m-r500t150-mb100-p4` |
| 20M | 500 x 150 | triton | n/a | 1 | [500, 690] | ok 150.0/150 | 9.66 | 10.25 | 10.45 | -- | -- | 99.9% | 298 | 4.1% | 0 | `g5-triton-20m-r500t150-mb500-p4` |

## The engines, built once each

ONNX Runtime's TensorRT provider compiles lazily at the first `Run()` with whatever shape it sees, so each mini-batch size is a separate engine and each was built and then PROVED reloadable by a fresh server before any cell ran. Without this the first cell at each shape would have spent its warm-up inside the TensorRT builder. A shape is the mini batch and nothing else, so a 500-row request at mini batch 100 and a 100-row request scored whole share one engine.

| model | mini batch | engine shape | first build | fresh server reload | cache |
|---|---:|---|---:|---:|---|
| 10M | 25 | [25, 690] | 25.2s | 22.1s | `/root/trt-cache/10m/mb25` |
| 10M | 50 | [50, 690] | 25.2s | 22.1s | `/root/trt-cache/10m/mb50` |
| 10M | 100 | [100, 690] | 25.3s | 22.1s | `/root/trt-cache/10m/mb100` |
| 10M | 250 | [250, 690] | 25.5s | 22.1s | `/root/trt-cache/10m/mb250` |
| 10M | 500 | [500, 690] | 26.0s | 22.1s | `/root/trt-cache/10m/mb500` |
| 20M | 25 | [25, 690] | 25.6s | 22.1s | `/root/trt-cache/20m/mb25` |
| 20M | 50 | [50, 690] | 25.7s | 22.1s | `/root/trt-cache/20m/mb50` |
| 20M | 100 | [100, 690] | 25.7s | 22.1s | `/root/trt-cache/20m/mb100` |
| 20M | 250 | [250, 690] | 25.9s | 22.1s | `/root/trt-cache/20m/mb250` |
| 20M | 500 | [500, 690] | 26.5s | 22.1s | `/root/trt-cache/20m/mb500` |

## What was running

| what | value |
|---|---|
| server | `g5.2xlarge`, 8 vCPU |
| GPU | NVIDIA A10G |
| TensorRT | the library extracted from the Triton image, so both stacks compile with the same one; `deploy.sh` refuses on any unresolved dependency in the provider |
| hushar | ONNX Runtime 1.28.3, TensorRT execution provider, FP32 |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `max_batch_size: 0`, KIND_GPU, FP32 |
| graphs | 10M `5539c97131c8`, 20M `30f9c5330b80` |
| sessions | 10M 2, 20M 4 -- pinned from `config-sweep-gpu-g5-10M-20M`, not re-probed |
| window | 120s warm-up discarded, then 900s measured as one continuous load in 15 slices |
| engine shapes | 10M [25, 50, 100, 250, 500], 20M [25, 50, 100, 250, 500] |
| revision | `7af1bf77` |

## Method and limits

- **Window.** 120s warm-up discarded, then 900s measured as one continuous load reported in 15 slices. A fresh server per cell, stopped afterwards, 60s idle before the next. Clients start on a minute boundary so the measured window is whole CloudWatch buckets.
- **Nothing is imported from `config-sweep-gpu-g5-10M-20M`.** Its unsplit cells measured the same loads on the same graphs, but over a 300s window against this run's 900s, and a window is one of the fields that DEFINES a cell. Every baseline here was therefore re-measured rather than copied.
- **Parallelism is pinned, not probed.** `sessions_per_model` is fixed per model size at the value `config-sweep-gpu-g5-10M-20M` selected by the lowest median among the configurations that sustained these loads. Holding it fixed is what makes the mini-batch column a one-variable subtraction; re-ranking it would let it move between the cells this report subtracts. The same value is used at 500 rows as at 100, so the packaging comparison is also at equal concurrency.
- **The pin gives Triton hushar's number, which at 10M is not Triton's own.** One value per model size is applied to both stacks, so each cell runs the two servers at equal concurrency. At 20M that is `instances 4`, which is also what `config-sweep-gpu-g5-10M-20M` selected for Triton. At 10M it is `instances 2` where that campaign selected 4 -- and its own probes priced the difference at **1.62 ms against 1.59 ms at the median and 1.89 against 1.78 at p95**, both inside this report's 8.4% tie band. Disclosed rather than corrected, because Triton is a fixed reference line here and not a variable: nothing in the answer above is computed from it.
- **Every split divides exactly**, so `is_fixed` padded nothing and each cell presented the engine exactly one shape. A short last batch would have meant two shapes, two compiles and a median blended from two tactic selections.
- **No cell was repeated, so the tie band is inherited.** Two medians within 8.4% are reported as `level`. That figure is the same-host spread `minibatch-rows-cpu-arm64-10M` measured by repeating a SPLIT configuration, which is the noisiest thing in this kind of run because its completion time is the slowest of several pieces. Using it rather than the 1.1% single-call figure is the conservative choice.
- **GPU utilization is occupancy, not saturation.** `nvidia-smi` reports the share of each sample period with at least one resident kernel, so a stream of small kernels reads high on a device with headroom -- and a split produces exactly such a stream. Power and clocks are reported beside it for that reason. Once the device is saturated hushar's `InferenceTime` also stops being compute, because it brackets the whole provider call and so absorbs queueing for the device.
- **FP32 only.** Both stacks take TensorRT's default builder settings, so these numbers are a floor on what the card can do rather than a ceiling.
- **Evidence.** `raw/results.json` is this report's input. Cells, resolved configurations, server and client logs, per-second host and GPU samples, the engine-cache proofs, Triton's `config.pbtxt`, the `trtexec` build logs and the CloudWatch points are under `raw/`. `verify.py` re-derives every number here from them.

16 measured cells: 13 sustained, 0 partial, 3 failed the rate. 16 carry GPU samples. 9 appear in the gain table.
