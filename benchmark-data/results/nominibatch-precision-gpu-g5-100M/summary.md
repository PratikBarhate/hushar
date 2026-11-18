# A 100M-parameter transformer on one A10G, FP32 against BF16

`nominibatch-precision-gpu-g5-100M`

## Configuration

| setting | value |
|---|---|
| server | `g5.2xlarge`, 1x NVIDIA A10G, 8 vCPU |
| model | 100M: 100,498,437 params, 31 encoder layers, 784,998,400 MAC/row, 402,233,324 bytes |
| model vs 10M | parameters 11.00x, MAC/row 14.73x |
| graph | one ONNX file, md5 `eee2644bd65cd23543ca806f102f5062`, identical across both precisions |
| precision | FP32 (TensorRT default, TF32 permitted) and BF16 (`trt_bf16_enable=1` on hushar, `--bf16` on `trtexec`), additive per layer |
| payloads | 100 rows x 100 TPS and 500 rows x 20 TPS, 10,000 rows/s each |
| wire | one 690-float f32 vector per row, both arms; identity transformation on hushar, so no server-side feature vectorization |
| feature logging | `inference_log.sample_rate` 0.0 |
| candidate model | none; every request scores the one resident model |
| mini batch | none; `mini_batch_size` unset on every hushar cell, so the plan is `whole request` and a request of any size is one engine call |
| sessions | `threading.sessions_per_model` 2, `sessions_per_request` 1 on hushar; `instance_group` count 2 on Triton |
| hushar engine | ONNX Runtime 1.28.3, TensorRT execution provider |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `max_batch_size: 0` |
| TensorRT | 10.16.1, the same library for both arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and by a version check at preflight |
| window | 902s measured in 15 slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | 12 measured, 12 sustained the offered rate |

## Results

| rows x TPS | precision | hushar p50 | hushar p95 | hushar p99 | Triton p50 | Triton p95 | Triton p99 | hushar / Triton | hushar graph | hushar GPU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 x 100 | FP32 | 11.05 | 11.38 | 11.55 | 10.10 | 10.26 | 10.35 | 1.09x | 9.652 | 93.5% |
| 100 x 100 | BF16 | 6.05 | 6.35 | 6.54 | 5.28 | 5.45 | 5.55 | 1.15x | 4.697 | 44.9% |
| 100 x 200 | BF16 | 6.31 | 6.69 | 6.96 | 5.30 | 5.63 | 5.69 | 1.19x | 4.696 | 89.4% |
| 500 x 20 | FP32 | 45.90 | 46.30 | 47.09 | 42.11 | 42.27 | 42.36 | 1.09x | 40.176 | 79.9% |
| 500 x 20 | BF16 | 24.30 | 24.65 | 24.97 | 20.21 | 20.44 | 20.55 | 1.20x | 18.336 | 36.4% |
| 500 x 40 | BF16 | 24.20 | 25.65 | 26.19 | 20.25 | 20.51 | 20.65 | 1.20x | 18.320 | 72.6% |

All latencies in milliseconds. `hushar graph` is `InferenceTime`: wall time per request
inside the backend. `hushar GPU` is the device's own occupancy over the measured window.

## FP32 against BF16

| rows x TPS | server | FP32 p50 | BF16 p50 | BF16 speedup | p50 delta ms | FP32 p99 | BF16 p99 | FP32 graph | BF16 graph | graph speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 x 100 | hushar | 11.05 | 6.05 | 1.83x | -5.00 | 11.55 | 6.54 | 9.652 | 4.697 | 2.05x |
| 100 x 100 | Triton | 10.10 | 5.28 | 1.91x | -4.82 | 10.35 | 5.55 | -- | -- | -- |
| 100 x 200 | hushar | -- | 6.31 | -- | -- | -- | 6.96 | -- | 4.696 | -- |
| 100 x 200 | Triton | -- | 5.30 | -- | -- | -- | 5.69 | -- | -- | -- |
| 500 x 20 | hushar | 45.90 | 24.30 | 1.89x | -21.60 | 47.09 | 24.97 | 40.176 | 18.336 | 2.19x |
| 500 x 20 | Triton | 42.11 | 20.21 | 2.08x | -21.90 | 42.36 | 20.55 | -- | -- | -- |
| 500 x 40 | hushar | -- | 24.20 | -- | -- | -- | 26.19 | -- | 18.320 | -- |
| 500 x 40 | Triton | -- | 20.25 | -- | -- | -- | 20.65 | -- | -- | -- |

`BF16 speedup` is FP32 p50 / BF16 p50. `graph speedup` is the same ratio on
`InferenceTime`. A row is blank where that precision was not measured at that rate.

## Offered rate against latency

| precision | rows | server | TPS | rows/s | result | p50 | p99 | graph | GPU busy | power W | p50 vs base | GPU vs base |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 100 | hushar | 100 | 10,000 | ok 100.0/100 | 6.05 | 6.54 | 4.697 | 44.9% | 126.0 | baseline | baseline |
| BF16 | 100 | hushar | 200 | 20,000 | ok 200.0/200 | 6.31 | 6.96 | 4.696 | 89.4% | 196.2 | 1.04x | 1.99x |
| BF16 | 100 | Triton | 100 | 10,000 | ok 100.0/100 | 5.28 | 5.55 | -- | 44.6% | 128.3 | baseline | baseline |
| BF16 | 100 | Triton | 200 | 20,000 | ok 200.0/200 | 5.30 | 5.69 | -- | 89.8% | 198.1 | 1.00x | 2.01x |
| BF16 | 500 | hushar | 20 | 10,000 | ok 20.0/20 | 24.30 | 24.97 | 18.336 | 36.4% | 129.8 | baseline | baseline |
| BF16 | 500 | hushar | 40 | 20,000 | ok 40.0/40 | 24.20 | 26.19 | 18.320 | 72.6% | 209.5 | 1.00x | 1.99x |
| BF16 | 500 | Triton | 20 | 10,000 | ok 20.0/20 | 20.21 | 20.55 | -- | 36.1% | 131.2 | baseline | baseline |
| BF16 | 500 | Triton | 40 | 20,000 | ok 40.0/40 | 20.25 | 20.65 | -- | 72.2% | 210.6 | 1.00x | 2.00x |

Lowest rate per payload and server is the baseline. `graph` is `InferenceTime`, which a
rate change should leave flat: the arithmetic of one request does not depend on how often
it arrives.

## Matched parallelism

| precision | rows x TPS | parallelism | hushar result | hushar p50 | hushar p95 | hushar p99 | Triton result | Triton p50 | Triton p95 | Triton p99 | hushar / Triton |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| FP32 | 500 x 20 | 2 | ok 20.0/20 | 45.90 | 46.30 | 47.09 | ok 20.0/20 | 42.11 | 42.27 | 42.36 | 1.09x |
| FP32 | 100 x 100 | 2 | ok 100.0/100 | 11.05 | 11.38 | 11.55 | ok 100.0/100 | 10.10 | 10.26 | 10.35 | 1.09x |
| BF16 | 500 x 20 | 2 | ok 20.0/20 | 24.30 | 24.65 | 24.97 | ok 20.0/20 | 20.21 | 20.44 | 20.55 | 1.20x |
| BF16 | 500 x 40 | 2 | ok 40.0/40 | 24.20 | 25.65 | 26.19 | ok 40.0/40 | 20.25 | 20.51 | 20.65 | 1.20x |
| BF16 | 100 x 100 | 2 | ok 100.0/100 | 6.05 | 6.35 | 6.54 | ok 100.0/100 | 5.28 | 5.45 | 5.55 | 1.15x |
| BF16 | 100 x 200 | 2 | ok 200.0/200 | 6.31 | 6.69 | 6.96 | ok 200.0/200 | 5.30 | 5.63 | 5.69 | 1.19x |


## Where the median goes

| precision | rows x TPS | hushar config | p50 | InferenceTime | VectorizationTime | ToTensorTime | remainder | buckets |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| FP32 | 500 x 20 | 2 sessions | 45.90 | 40.176 | 0.105 | 0.000 | 5.619 | 15 |
| FP32 | 100 x 100 | 2 sessions | 11.05 | 9.652 | 0.015 | 0.000 | 1.383 | 15 |
| BF16 | 500 x 20 | 2 sessions | 24.30 | 18.336 | 0.110 | 0.000 | 5.854 | 15 |
| BF16 | 500 x 40 | 2 sessions | 24.20 | 18.320 | 0.104 | 0.000 | 5.776 | 15 |
| BF16 | 100 x 100 | 2 sessions | 6.05 | 4.697 | 0.014 | 0.000 | 1.339 | 15 |
| BF16 | 100 x 200 | 2 sessions | 6.31 | 4.696 | 0.015 | 0.000 | 1.599 | 15 |

Medians of whole minute buckets inside the measured window. `remainder` is the client
median less the three measures.

## Device and host

| precision | rows x TPS | arm | config | GPU | GPU max | GPU MiB | power W | host CPU | CPU max |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | 500 x 20 | hushar-tensor | 2 sessions | 79.9% | 82.0% | 1259 | 245.7 | 12.1% | 26.3% |
| FP32 | 500 x 20 | triton | 2 instances | 79.9% | 81.0% | 903 | 243.4 | 2.1% | 20.2% |
| FP32 | 100 x 100 | hushar-tensor | 2 sessions | 93.5% | 95.0% | 1117 | 231.8 | 14.2% | 29.5% |
| FP32 | 100 x 100 | triton | 2 instances | 93.5% | 94.0% | 753 | 233.5 | 3.6% | 20.4% |
| BF16 | 500 x 20 | hushar-tensor | 2 sessions | 36.4% | 38.0% | 799 | 129.8 | 6.7% | 20.2% |
| BF16 | 500 x 20 | triton | 2 instances | 36.1% | 37.0% | 635 | 131.2 | 2.0% | 20.6% |
| BF16 | 500 x 40 | hushar-tensor | 2 sessions | 72.6% | 74.0% | 806 | 209.5 | 12.1% | 25.4% |
| BF16 | 500 x 40 | triton | 2 instances | 72.2% | 73.0% | 635 | 210.6 | 2.7% | 21.3% |
| BF16 | 100 x 100 | hushar-tensor | 2 sessions | 44.9% | 46.0% | 710 | 126.0 | 8.0% | 23.2% |
| BF16 | 100 x 100 | triton | 2 instances | 44.6% | 49.0% | 541 | 128.3 | 3.2% | 20.9% |
| BF16 | 100 x 200 | hushar-tensor | 2 sessions | 89.4% | 91.0% | 713 | 196.2 | 15.1% | 30.0% |
| BF16 | 100 x 200 | triton | 2 instances | 89.8% | 90.0% | 541 | 198.1 | 5.1% | 23.9% |


## Headroom

| precision | rows x TPS | arm | GPU busy | GPU max | GPU mem | GPU MiB | power W | power max W | server CPU | server max | client CPU | TPS x at 100% busy |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 500 x 20 | hushar-tensor | 79.9% | 82.0% | 65.9% | 1259 | 245.7 | 281.6 | 12.1% | 26.3% | 0.3% | 1.25x |
| FP32 | 500 x 20 | triton | 79.9% | 81.0% | 65.9% | 903 | 243.4 | 281.7 | 2.1% | 20.2% | 0.1% | 1.25x |
| FP32 | 100 x 100 | hushar-tensor | 93.5% | 95.0% | 47.4% | 1117 | 231.8 | 237.2 | 14.2% | 29.5% | 0.3% | 1.07x |
| FP32 | 100 x 100 | triton | 93.5% | 94.0% | 47.0% | 753 | 233.5 | 237.2 | 3.6% | 20.4% | 0.1% | 1.07x |
| BF16 | 500 x 20 | hushar-tensor | 36.4% | 38.0% | 21.0% | 799 | 129.8 | 212.9 | 6.7% | 20.2% | 0.3% | 2.75x |
| BF16 | 500 x 20 | triton | 36.1% | 37.0% | 21.0% | 635 | 131.2 | 212.9 | 2.0% | 20.6% | 0.1% | 2.77x |
| BF16 | 500 x 40 | hushar-tensor | 72.6% | 74.0% | 42.3% | 806 | 209.5 | 239.9 | 12.1% | 25.4% | 0.5% | 1.38x |
| BF16 | 500 x 40 | triton | 72.2% | 73.0% | 42.0% | 635 | 210.6 | 244.2 | 2.7% | 21.3% | 0.1% | 1.38x |
| BF16 | 100 x 100 | hushar-tensor | 44.9% | 46.0% | 16.0% | 710 | 126.0 | 138.9 | 8.0% | 23.2% | 0.3% | 2.23x |
| BF16 | 100 x 100 | triton | 44.6% | 49.0% | 16.9% | 541 | 128.3 | 141.0 | 3.2% | 20.9% | 0.1% | 2.24x |
| BF16 | 100 x 200 | hushar-tensor | 89.4% | 91.0% | 33.0% | 713 | 196.2 | 200.9 | 15.1% | 30.0% | 0.5% | 1.12x |
| BF16 | 100 x 200 | triton | 89.8% | 90.0% | 33.0% | 541 | 198.1 | 203.0 | 5.1% | 23.9% | 0.1% | 1.11x |

Every column is measured over the window except the last, which is `100 / GPU busy`: the
factor the offered rate could rise by before the device executes continuously. It is an
upper bound, because occupancy is the share of sample periods in which the device was
executing, not how full it was. `client CPU` is the load generator's own host.

## Rate held

| precision | arm | config | rows x TPS | result | slices held | shed | failed | window s |
|---|---|---|---:|---|---:|---:|---:|---:|
| BF16 | hushar-tensor | 2 sessions | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 902 |
| BF16 | hushar-tensor | 2 sessions | 100 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 |
| BF16 | triton | 2 instances | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 903 |
| BF16 | triton | 2 instances | 100 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 903 |
| BF16 | hushar-tensor | 2 sessions | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 902 |
| BF16 | hushar-tensor | 2 sessions | 500 x 40 | ok 40.0/40 | 15/15 | 0 | 0 | 902 |
| BF16 | triton | 2 instances | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 905 |
| BF16 | triton | 2 instances | 500 x 40 | ok 40.0/40 | 15/15 | 0 | 0 | 905 |
| FP32 | hushar-tensor | 2 sessions | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 902 |
| FP32 | triton | 2 instances | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 903 |
| FP32 | hushar-tensor | 2 sessions | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 902 |
| FP32 | triton | 2 instances | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 905 |


## TensorRT engines

| precision | rows | engine shape | engine bytes | cold build s | cached s | ratio |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 100 | [100, 690] | 202955140 | 53.2 | 22.1 | 2.4x |
| BF16 | 500 | [500, 690] | 206186060 | 62.4 | 22.2 | 2.8x |
| FP32 | 100 | [100, 690] | 403566716 | 31.2 | 22.1 | 1.4x |
| FP32 | 500 | [500, 690] | 403617164 | 32.4 | 22.2 | 1.5x |

One engine per precision and row count, built and proved reloadable before the grid ran.

| precision | rows | arm | engine bytes | plan or shape |
|---|---:|---|---:|---|
| BF16 | 100 | hushar-tensor | 202955140 | [100, 690] |
| BF16 | 100 | triton | 202952572 | /root/triton/plans/100m-eee2644bd65c-r100-bf16.plan |
| BF16 | 500 | hushar-tensor | 206186060 | [500, 690] |
| BF16 | 500 | triton | 206196372 | /root/triton/plans/100m-eee2644bd65c-r500-bf16.plan |
| FP32 | 100 | hushar-tensor | 403566716 | [100, 690] |
| FP32 | 100 | triton | 403567372 | /root/triton/plans/100m-eee2644bd65c-r100-fp32.plan |
| FP32 | 500 | hushar-tensor | 403617164 | [500, 690] |
| FP32 | 500 | triton | 403617812 | /root/triton/plans/100m-eee2644bd65c-r500-fp32.plan |


## Engine and library versions

| host | arm | GPU | driver | TensorRT | CUDA | ONNX Runtime | ORT in use |
|---|---|---|---|---:|---:|---|---|
| g5-1 | hushar-tensor | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | loaded |
| g5-1 | triton | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | n/a (tensorrt backend) |
| g5-2 | hushar-tensor | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | loaded |
| g5-2 | triton | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | n/a (tensorrt backend) |


## Every cell

| cell | precision | plan | engine shape | parallelism | per request | lanes | p50 | p95 | p99 | result | host |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `g5q-hushar-tensor-100m-bf16-r100t100-p2` | BF16 | whole request | [100, 690] | 2 | 1 | 2 | 6.05 | 6.35 | 6.54 | ok | g5-2 |
| `g5q-hushar-tensor-100m-bf16-r100t200-p2` | BF16 | whole request | [100, 690] | 2 | 1 | 2 | 6.31 | 6.69 | 6.96 | ok | g5-2 |
| `g5q-hushar-tensor-100m-bf16-r500t20-p2` | BF16 | whole request | [500, 690] | 2 | 1 | 2 | 24.30 | 24.65 | 24.97 | ok | g5-2 |
| `g5q-hushar-tensor-100m-bf16-r500t40-p2` | BF16 | whole request | [500, 690] | 2 | 1 | 2 | 24.20 | 25.65 | 26.19 | ok | g5-2 |
| `g5q-hushar-tensor-100m-fp32-r100t100-p2` | FP32 | whole request | [100, 690] | 2 | 1 | 2 | 11.05 | 11.38 | 11.55 | ok | g5-1 |
| `g5q-hushar-tensor-100m-fp32-r500t20-p2` | FP32 | whole request | [500, 690] | 2 | 1 | 2 | 45.90 | 46.30 | 47.09 | ok | g5-1 |
| `g5q-triton-100m-bf16-r100t100-p2` | BF16 | /root/triton/plans/100m-eee2644bd65c-r100-bf16.plan | [100, 690] | 2 | -- | -- | 5.28 | 5.45 | 5.55 | ok | g5-2 |
| `g5q-triton-100m-bf16-r100t200-p2` | BF16 | /root/triton/plans/100m-eee2644bd65c-r100-bf16.plan | [100, 690] | 2 | -- | -- | 5.30 | 5.63 | 5.69 | ok | g5-2 |
| `g5q-triton-100m-bf16-r500t20-p2` | BF16 | /root/triton/plans/100m-eee2644bd65c-r500-bf16.plan | [500, 690] | 2 | -- | -- | 20.21 | 20.44 | 20.55 | ok | g5-2 |
| `g5q-triton-100m-bf16-r500t40-p2` | BF16 | /root/triton/plans/100m-eee2644bd65c-r500-bf16.plan | [500, 690] | 2 | -- | -- | 20.25 | 20.51 | 20.65 | ok | g5-2 |
| `g5q-triton-100m-fp32-r100t100-p2` | FP32 | /root/triton/plans/100m-eee2644bd65c-r100-fp32.plan | [100, 690] | 2 | -- | -- | 10.10 | 10.26 | 10.35 | ok | g5-1 |
| `g5q-triton-100m-fp32-r500t20-p2` | FP32 | /root/triton/plans/100m-eee2644bd65c-r500-fp32.plan | [500, 690] | 2 | -- | -- | 42.11 | 42.27 | 42.36 | ok | g5-1 |


## Correction

| figure | configuration line | architecture | source |
|---|--:|--:|---|
| parameters | 100,498,437 | 100,561,925 | sum over `graph.initializer` of `bench_model.onnx` |

The configuration line reports the values **stored** in the exported graph, which is 1,024
lower for every LayerNorm after the first than the architecture's own count of 100,561,925:
untrained, every LayerNorm still holds its initial all-ones scale and all-zeros bias, so the
export deduplicates them behind `Identity` nodes. The same gap is present at 10M and 20M.
Both counts are right about different things, and no measurement here depends on either.

An earlier revision of this section gave 3,150,336 as an "ONNX per-layer constant" against
3,152,384 for the architecture. There is one architecture; the 2,048 is two deduplicated
LayerNorms, not a different layer. That revision also restated MAC per row as 784,998,400,
which is withdrawn: the figure omits the classifier's 512 x 5, and 785,000,960 is the count
consistent with the 10M and 20M models. The decomposition for all three sizes is in
[`scripts/benchmark-models.md`](../../../scripts/benchmark-models.md).

`raw/results.json` carries the driver's values on every cell, as the record of what the
driver wrote rather than a document edited after the fact.


## Provenance

| host | instance | address | revision | ONNX Runtime | Triton image | model md5 |
|---|---|---|---|---|---|---|
| g5-1 | g5.2xlarge | 172.31.36.95 | e6265e6e | 1.28.3 | nvcr.io/nvidia/tritonserver:26.05-py3 | eee2644bd65cd23543ca806f102f5062 |
| g5-2 | g5.2xlarge | 172.31.36.161 | e6265e6e | 1.28.3 | nvcr.io/nvidia/tritonserver:26.05-py3 | eee2644bd65cd23543ca806f102f5062 |

| group | cells | sustained | failed rate | partial | CloudWatch | GPU samples |
|---|---:|---:|---:|---:|---:|---:|
| hushar-tensor | 6 | 6 | 0 | 0 | 6 | 6 |
| triton | 6 | 6 | 0 | 0 | 0 | 6 |
| 100m | 12 | 12 | 0 | 0 | 6 | 12 |

Cells, server banners, generated service and model configurations, per-second host and GPU
samples, client output and CloudWatch points are under `raw/`. `raw/results.json` is this
report's only input. `verify.py` re-derives every number above from it.
