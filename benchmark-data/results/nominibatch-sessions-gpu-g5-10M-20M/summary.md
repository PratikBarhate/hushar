# hushar with no mini batch against Triton as it ships, on one A10G

`nominibatch-sessions-gpu-g5-10M-20M`

## Configuration

| setting | value |
|---|---|
| server | `g5.2xlarge`, 1x NVIDIA A10G, 8 vCPU |
| models | 10M (9,138,693 params, 2 layers) and 20M (18,589,701 params, 5 layers) |
| precision | FP32, TensorRT default builder settings |
| wire | one 690-float f32 vector per row, both arms; identity transformation on hushar, so no server-side feature vectorization |
| feature logging | `inference_log.sample_rate` 0.0 |
| candidate model | none; every request scores the one resident model |
| mini batch | none; `mini_batch_size` is unset on every hushar cell, so the plan is `whole request` and a request of any size is one engine call |
| hushar engine | ONNX Runtime 1.28.3, TensorRT execution provider |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `max_batch_size: 0` |
| TensorRT | 10.16.1, the same library for both arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and by a version check at preflight |
| parallelism | `threading.sessions_per_model` 2, 4, 8 on hushar; `instance_group` count 2, 4, 8 on Triton |
| window | 902s measured in 15 slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | 24 measured, 24 sustained the offered rate |

Every hushar cell leases one session per request (`threading.sessions_per_request` 1), so
`sessions_per_model` is also the requests in flight, which is what Triton's instance count
means. A row of the Grid is therefore a pair of cells differing in the server alone.

## Answer

| model | rows | TPS | hushar p50 | at sessions | Triton p50 | at instances | hushar / Triton | hushar graph | hushar held | Triton held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 500 | 200 | 8.75 | 2 | 5.09 | 2 | 1.72x | 3.179 | 3/3 | 3/3 |
| 10M | 100 | 1000 | 2.48 | 2 | 1.41 | 2 | 1.76x | 0.853 | 3/3 | 3/3 |
| 20M | 500 | 150 | 14.19 | 2 | 9.28 | 4 | 1.53x | 7.687 | 3/3 | 3/3 |
| 20M | 100 | 750 | 4.39 | 2 | 3.38 | 4 | 1.30x | 2.525 | 3/3 | 3/3 |

Each arm at its own lowest sustained median. `hushar graph` is `InferenceTime` at that
cell: wall time per request inside the backend.

## Grid

| model | rows x TPS | parallelism | hushar result | hushar p50 | hushar p95 | hushar p99 | Triton result | Triton p50 | Triton p95 | Triton p99 | hushar / Triton |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 10M | 500 x 200 | 2 | ok 200.0/200 | 8.75 | 9.17 | 10.61 | ok 200.0/200 | 5.09 | 5.31 | 5.32 | 1.72x |
| 10M | 500 x 200 | 4 | ok 200.0/200 | 8.77 | 9.14 | 10.12 | ok 200.0/200 | 5.30 | 5.32 | 5.33 | 1.65x |
| 10M | 500 x 200 | 8 | ok 200.0/200 | 8.77 | 9.20 | 10.35 | ok 200.0/200 | 5.29 | 5.31 | 5.33 | 1.66x |
| 10M | 100 x 1000 | 2 | ok 1000.0/1000 | 2.48 | 2.93 | 3.27 | ok 1000.0/1000 | 1.41 | 1.72 | 8.52 | 1.76x |
| 10M | 100 x 1000 | 4 | ok 1000.0/1000 | 2.53 | 3.02 | 3.40 | ok 1000.0/1000 | 1.45 | 1.71 | 6.12 | 1.74x |
| 10M | 100 x 1000 | 8 | ok 1000.0/1000 | 2.50 | 2.96 | 3.33 | ok 1000.0/1000 | 1.41 | 1.65 | 1.71 | 1.77x |
| 20M | 500 x 150 | 2 | ok 150.0/150 | 14.19 | 15.68 | 16.25 | ok 150.0/150 | 9.29 | 9.80 | 9.95 | 1.53x |
| 20M | 500 x 150 | 4 | ok 150.0/150 | 14.30 | 15.77 | 16.33 | ok 150.0/150 | 9.28 | 9.78 | 9.93 | 1.54x |
| 20M | 500 x 150 | 8 | ok 150.0/150 | 14.39 | 15.73 | 16.27 | ok 150.0/150 | 9.30 | 9.81 | 9.95 | 1.55x |
| 20M | 100 x 750 | 2 | ok 750.0/750 | 4.39 | 5.12 | 5.40 | ok 750.0/750 | 5.08 | 8.92 | 10.71 | 0.86x |
| 20M | 100 x 750 | 4 | ok 750.0/750 | 4.52 | 5.17 | 5.53 | ok 750.0/750 | 3.38 | 3.71 | 4.82 | 1.34x |
| 20M | 100 x 750 | 8 | ok 750.0/750 | 4.55 | 5.20 | 5.53 | ok 750.0/750 | 3.44 | 3.72 | 3.74 | 1.32x |

`parallelism` is hushar sessions and Triton instances. A ratio is withheld where either
cell of the pair did not hold its rate.

## hushar by session count

| model | rows x TPS | sessions | result | p50 | p95 | p99 | p50 vs fewest sessions |
|---|---:|---:|---|---:|---:|---:|---:|
| 10M | 500 x 200 | 2 | ok 200.0/200 | 8.75 | 9.17 | 10.61 | baseline |
| 10M | 500 x 200 | 4 | ok 200.0/200 | 8.77 | 9.14 | 10.12 | +0.02 |
| 10M | 500 x 200 | 8 | ok 200.0/200 | 8.77 | 9.20 | 10.35 | +0.02 |
| 10M | 100 x 1000 | 2 | ok 1000.0/1000 | 2.48 | 2.93 | 3.27 | baseline |
| 10M | 100 x 1000 | 4 | ok 1000.0/1000 | 2.53 | 3.02 | 3.40 | +0.05 |
| 10M | 100 x 1000 | 8 | ok 1000.0/1000 | 2.50 | 2.96 | 3.33 | +0.02 |
| 20M | 500 x 150 | 2 | ok 150.0/150 | 14.19 | 15.68 | 16.25 | baseline |
| 20M | 500 x 150 | 4 | ok 150.0/150 | 14.30 | 15.77 | 16.33 | +0.11 |
| 20M | 500 x 150 | 8 | ok 150.0/150 | 14.39 | 15.73 | 16.27 | +0.20 |
| 20M | 100 x 750 | 2 | ok 750.0/750 | 4.39 | 5.12 | 5.40 | baseline |
| 20M | 100 x 750 | 4 | ok 750.0/750 | 4.52 | 5.17 | 5.53 | +0.13 |
| 20M | 100 x 750 | 8 | ok 750.0/750 | 4.55 | 5.20 | 5.53 | +0.16 |


## Triton by instance count

| model | rows x TPS | instances | result | p50 | p95 | p99 | p50 vs fewest instances |
|---|---:|---:|---|---:|---:|---:|---:|
| 10M | 500 x 200 | 2 | ok 200.0/200 | 5.09 | 5.31 | 5.32 | baseline |
| 10M | 500 x 200 | 4 | ok 200.0/200 | 5.30 | 5.32 | 5.33 | +0.21 |
| 10M | 500 x 200 | 8 | ok 200.0/200 | 5.29 | 5.31 | 5.33 | +0.20 |
| 10M | 100 x 1000 | 2 | ok 1000.0/1000 | 1.41 | 1.72 | 8.52 | baseline |
| 10M | 100 x 1000 | 4 | ok 1000.0/1000 | 1.45 | 1.71 | 6.12 | +0.04 |
| 10M | 100 x 1000 | 8 | ok 1000.0/1000 | 1.41 | 1.65 | 1.71 | +0.00 |
| 20M | 500 x 150 | 2 | ok 150.0/150 | 9.29 | 9.80 | 9.95 | baseline |
| 20M | 500 x 150 | 4 | ok 150.0/150 | 9.28 | 9.78 | 9.93 | -0.01 |
| 20M | 500 x 150 | 8 | ok 150.0/150 | 9.30 | 9.81 | 9.95 | +0.01 |
| 20M | 100 x 750 | 2 | ok 750.0/750 | 5.08 | 8.92 | 10.71 | baseline |
| 20M | 100 x 750 | 4 | ok 750.0/750 | 3.38 | 3.71 | 4.82 | -1.70 |
| 20M | 100 x 750 | 8 | ok 750.0/750 | 3.44 | 3.72 | 3.74 | -1.64 |


## Where the median goes

| model | rows x TPS | hushar config | p50 | InferenceTime | VectorizationTime | ToTensorTime | remainder | buckets |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 10M | 500 x 200 | 2 sessions | 8.75 | 3.179 | 0.093 | 0.000 | 5.478 | 15 |
| 10M | 500 x 200 | 4 sessions | 8.77 | 3.178 | 0.108 | 0.000 | 5.484 | 15 |
| 10M | 500 x 200 | 8 sessions | 8.77 | 3.179 | 0.106 | 0.000 | 5.485 | 15 |
| 10M | 100 x 1000 | 2 sessions | 2.48 | 0.853 | 0.014 | 0.000 | 1.613 | 15 |
| 10M | 100 x 1000 | 4 sessions | 2.53 | 0.868 | 0.016 | 0.000 | 1.646 | 15 |
| 10M | 100 x 1000 | 8 sessions | 2.50 | 0.869 | 0.018 | 0.000 | 1.613 | 15 |
| 20M | 500 x 150 | 2 sessions | 14.19 | 7.687 | 0.097 | 0.000 | 6.406 | 15 |
| 20M | 500 x 150 | 4 sessions | 14.30 | 7.777 | 0.113 | 0.000 | 6.410 | 15 |
| 20M | 500 x 150 | 8 sessions | 14.39 | 7.777 | 0.115 | 0.000 | 6.498 | 15 |
| 20M | 100 x 750 | 2 sessions | 4.39 | 2.525 | 0.015 | 0.000 | 1.850 | 15 |
| 20M | 100 x 750 | 4 sessions | 4.52 | 2.802 | 0.017 | 0.000 | 1.701 | 15 |
| 20M | 100 x 750 | 8 sessions | 4.55 | 2.824 | 0.021 | 0.000 | 1.705 | 15 |

`InferenceTime`, `VectorizationTime` and `ToTensorTime` are hushar's own CloudWatch
measures, medians of whole minute buckets inside the measured window. `remainder` is the
client median less those three.

## Device and host

| model | rows x TPS | arm | config | GPU | GPU max | GPU MiB | power W | host CPU | CPU max |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 10M | 500 x 200 | hushar-tensor | 2 sessions | 59.5% | 61.0% | 597 | 182.7 | 17.1% | 36.9% |
| 10M | 500 x 200 | hushar-tensor | 4 sessions | 59.5% | 61.0% | 959 | 183.5 | 17.1% | 37.4% |
| 10M | 500 x 200 | hushar-tensor | 8 sessions | 59.5% | 61.0% | 771 | 182.8 | 16.9% | 29.2% |
| 10M | 500 x 200 | triton | 2 instances | 59.9% | 60.0% | 581 | 183.0 | 4.2% | 18.1% |
| 10M | 500 x 200 | triton | 4 instances | 59.9% | 60.0% | 777 | 183.1 | 4.3% | 18.6% |
| 10M | 500 x 200 | triton | 8 instances | 59.9% | 60.0% | 1165 | 182.5 | 4.5% | 24.8% |
| 10M | 100 x 1000 | hushar-tensor | 2 sessions | 67.1% | 69.0% | 443 | 179.4 | 21.9% | 41.3% |
| 10M | 100 x 1000 | hushar-tensor | 4 sessions | 66.9% | 69.0% | 681 | 179.2 | 22.2% | 42.3% |
| 10M | 100 x 1000 | hushar-tensor | 8 sessions | 66.9% | 69.0% | 1315 | 187.2 | 22.2% | 38.4% |
| 10M | 100 x 1000 | triton | 2 instances | 70.9% | 76.0% | 431 | 185.6 | 9.7% | 22.5% |
| 10M | 100 x 1000 | triton | 4 instances | 71.1% | 74.0% | 475 | 185.6 | 10.1% | 29.2% |
| 10M | 100 x 1000 | triton | 8 instances | 71.7% | 72.0% | 559 | 186.2 | 10.4% | 31.0% |
| 20M | 500 x 150 | hushar-tensor | 2 sessions | 99.3% | 100.0% | 661 | 298.4 | 22.1% | 40.2% |
| 20M | 500 x 150 | hushar-tensor | 4 sessions | 99.1% | 100.0% | 1083 | 298.4 | 22.5% | 41.6% |
| 20M | 500 x 150 | hushar-tensor | 8 sessions | 99.1% | 100.0% | 2177 | 298.3 | 22.2% | 36.3% |
| 20M | 500 x 150 | triton | 2 instances | 99.9% | 100.0% | 613 | 298.6 | 4.0% | 17.1% |
| 20M | 500 x 150 | triton | 4 instances | 99.9% | 100.0% | 809 | 298.7 | 4.1% | 19.2% |
| 20M | 500 x 150 | triton | 8 instances | 99.9% | 100.0% | 1197 | 298.6 | 4.2% | 21.2% |
| 20M | 100 x 750 | hushar-tensor | 2 sessions | 100.0% | 100.0% | 507 | 298.8 | 32.1% | 51.9% |
| 20M | 100 x 750 | hushar-tensor | 4 sessions | 99.4% | 100.0% | 809 | 298.8 | 35.3% | 58.0% |
| 20M | 100 x 750 | hushar-tensor | 8 sessions | 99.4% | 100.0% | 1701 | 299.1 | 35.5% | 53.4% |
| 20M | 100 x 750 | triton | 2 instances | 99.2% | 100.0% | 463 | 298.3 | 8.4% | 22.0% |
| 20M | 100 x 750 | triton | 4 instances | 99.7% | 100.0% | 507 | 298.6 | 8.6% | 28.7% |
| 20M | 100 x 750 | triton | 8 instances | 99.8% | 100.0% | 593 | 298.9 | 8.9% | 28.7% |


## Rate held

| model | arm | config | rows x TPS | result | slices held | shed | failed | window s |
|---|---|---|---:|---|---:|---:|---:|---:|
| 10M | hushar-tensor | 2 sessions | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 902 |
| 10M | hushar-tensor | 4 sessions | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 902 |
| 10M | hushar-tensor | 8 sessions | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 902 |
| 10M | triton | 2 instances | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 903 |
| 10M | triton | 4 instances | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 903 |
| 10M | triton | 8 instances | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 903 |
| 10M | hushar-tensor | 2 sessions | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 |
| 10M | hushar-tensor | 4 sessions | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 |
| 10M | hushar-tensor | 8 sessions | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 |
| 10M | triton | 2 instances | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 905 |
| 10M | triton | 4 instances | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 905 |
| 10M | triton | 8 instances | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 905 |
| 20M | hushar-tensor | 2 sessions | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 902 |
| 20M | hushar-tensor | 4 sessions | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 902 |
| 20M | hushar-tensor | 8 sessions | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 902 |
| 20M | triton | 2 instances | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 903 |
| 20M | triton | 4 instances | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 903 |
| 20M | triton | 8 instances | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 903 |
| 20M | hushar-tensor | 2 sessions | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 902 |
| 20M | hushar-tensor | 4 sessions | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 902 |
| 20M | hushar-tensor | 8 sessions | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 902 |
| 20M | triton | 2 instances | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 905 |
| 20M | triton | 4 instances | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 905 |
| 20M | triton | 8 instances | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 905 |


## TensorRT engines

| model | rows | engine shape | cold build s | cached s | ratio |
|---|---:|---:|---:|---:|---:|
| 10M | 100 | [100, 690] | 25.3 | 22.1 | 1.1x |
| 10M | 500 | [500, 690] | 26.0 | 22.1 | 1.2x |
| 20M | 100 | [100, 690] | 25.7 | 22.1 | 1.2x |
| 20M | 500 | [500, 690] | 26.5 | 22.1 | 1.2x |

One engine per row count, built once before the grid ran and then proved reloadable by a
fresh server, so no cell paid for a build inside its warm-up. A session count loads more
copies of the same engine, so it adds no shape.

## Engine and library versions

| host | arm | GPU | driver | TensorRT | CUDA | ONNX Runtime | ORT in use |
|---|---|---|---|---:|---:|---|---|
| g5-1 | hushar-tensor | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | loaded |
| g5-1 | triton | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | n/a (tensorrt backend) |
| g5-2 | hushar-tensor | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | loaded |
| g5-2 | triton | NVIDIA A10G | 595.112.07 | 10.16.1 | 13 | 1.28.3 | n/a (tensorrt backend) |

Read off each server at preflight and stamped on every cell of both arms. Triton's
`tensorrt` backend uses no ONNX Runtime, so the ONNX Runtime column applies to hushar
only; TensorRT is the library both arms execute on.

## Every cell

| cell | plan | engine shape | parallelism | per request | lanes | p50 | p95 | p99 | result | host |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `g5nb-hushar-tensor-10m-r100t1000-p2` | whole request | [100, 690] | 2 | 1 | 2 | 2.48 | 2.93 | 3.27 | ok | g5-1 |
| `g5nb-hushar-tensor-10m-r100t1000-p4` | whole request | [100, 690] | 4 | 1 | 4 | 2.53 | 3.02 | 3.40 | ok | g5-1 |
| `g5nb-hushar-tensor-10m-r100t1000-p8` | whole request | [100, 690] | 8 | 1 | 8 | 2.50 | 2.96 | 3.33 | ok | g5-1 |
| `g5nb-hushar-tensor-10m-r500t200-p2` | whole request | [500, 690] | 2 | 1 | 2 | 8.75 | 9.17 | 10.61 | ok | g5-1 |
| `g5nb-hushar-tensor-10m-r500t200-p4` | whole request | [500, 690] | 4 | 1 | 4 | 8.77 | 9.14 | 10.12 | ok | g5-1 |
| `g5nb-hushar-tensor-10m-r500t200-p8` | whole request | [500, 690] | 8 | 1 | 8 | 8.77 | 9.20 | 10.35 | ok | g5-1 |
| `g5nb-hushar-tensor-20m-r100t750-p2` | whole request | [100, 690] | 2 | 1 | 2 | 4.39 | 5.12 | 5.40 | ok | g5-2 |
| `g5nb-hushar-tensor-20m-r100t750-p4` | whole request | [100, 690] | 4 | 1 | 4 | 4.52 | 5.17 | 5.53 | ok | g5-2 |
| `g5nb-hushar-tensor-20m-r100t750-p8` | whole request | [100, 690] | 8 | 1 | 8 | 4.55 | 5.20 | 5.53 | ok | g5-2 |
| `g5nb-hushar-tensor-20m-r500t150-p2` | whole request | [500, 690] | 2 | 1 | 2 | 14.19 | 15.68 | 16.25 | ok | g5-2 |
| `g5nb-hushar-tensor-20m-r500t150-p4` | whole request | [500, 690] | 4 | 1 | 4 | 14.30 | 15.77 | 16.33 | ok | g5-2 |
| `g5nb-hushar-tensor-20m-r500t150-p8` | whole request | [500, 690] | 8 | 1 | 8 | 14.39 | 15.73 | 16.27 | ok | g5-2 |
| `g5nb-triton-10m-r100t1000-p2` | /root/triton/plans/10m-5539c97131c8-r100-fp32.plan | [100, 690] | 2 | -- | -- | 1.41 | 1.72 | 8.52 | ok | g5-1 |
| `g5nb-triton-10m-r100t1000-p4` | /root/triton/plans/10m-5539c97131c8-r100-fp32.plan | [100, 690] | 4 | -- | -- | 1.45 | 1.71 | 6.12 | ok | g5-1 |
| `g5nb-triton-10m-r100t1000-p8` | /root/triton/plans/10m-5539c97131c8-r100-fp32.plan | [100, 690] | 8 | -- | -- | 1.41 | 1.65 | 1.71 | ok | g5-1 |
| `g5nb-triton-10m-r500t200-p2` | /root/triton/plans/10m-5539c97131c8-r500-fp32.plan | [500, 690] | 2 | -- | -- | 5.09 | 5.31 | 5.32 | ok | g5-1 |
| `g5nb-triton-10m-r500t200-p4` | /root/triton/plans/10m-5539c97131c8-r500-fp32.plan | [500, 690] | 4 | -- | -- | 5.30 | 5.32 | 5.33 | ok | g5-1 |
| `g5nb-triton-10m-r500t200-p8` | /root/triton/plans/10m-5539c97131c8-r500-fp32.plan | [500, 690] | 8 | -- | -- | 5.29 | 5.31 | 5.33 | ok | g5-1 |
| `g5nb-triton-20m-r100t750-p2` | /root/triton/plans/20m-30f9c5330b80-r100-fp32.plan | [100, 690] | 2 | -- | -- | 5.08 | 8.92 | 10.71 | ok | g5-2 |
| `g5nb-triton-20m-r100t750-p4` | /root/triton/plans/20m-30f9c5330b80-r100-fp32.plan | [100, 690] | 4 | -- | -- | 3.38 | 3.71 | 4.82 | ok | g5-2 |
| `g5nb-triton-20m-r100t750-p8` | /root/triton/plans/20m-30f9c5330b80-r100-fp32.plan | [100, 690] | 8 | -- | -- | 3.44 | 3.72 | 3.74 | ok | g5-2 |
| `g5nb-triton-20m-r500t150-p2` | /root/triton/plans/20m-30f9c5330b80-r500-fp32.plan | [500, 690] | 2 | -- | -- | 9.29 | 9.80 | 9.95 | ok | g5-2 |
| `g5nb-triton-20m-r500t150-p4` | /root/triton/plans/20m-30f9c5330b80-r500-fp32.plan | [500, 690] | 4 | -- | -- | 9.28 | 9.78 | 9.93 | ok | g5-2 |
| `g5nb-triton-20m-r500t150-p8` | /root/triton/plans/20m-30f9c5330b80-r500-fp32.plan | [500, 690] | 8 | -- | -- | 9.30 | 9.81 | 9.95 | ok | g5-2 |


## Provenance

| host | instance | address | revision | ONNX Runtime | Triton image | model md5 |
|---|---|---|---|---|---|---|
| g5-1 | g5.2xlarge | 172.31.24.24 | 72e7c983 | 1.28.3 | nvcr.io/nvidia/tritonserver:26.05-py3 | 5539c97131c896db6c58088d9f9b9ee3 |
| g5-2 | g5.2xlarge | 172.31.18.195 | 72e7c983 | 1.28.3 | nvcr.io/nvidia/tritonserver:26.05-py3 | 30f9c5330b808a34b55fe88048a20da6 |

| group | cells | sustained | failed rate | partial | CloudWatch | GPU samples |
|---|---:|---:|---:|---:|---:|---:|
| hushar-tensor | 12 | 12 | 0 | 0 | 12 | 12 |
| triton | 12 | 12 | 0 | 0 | 0 | 12 |
| 10M | 12 | 12 | 0 | 0 | 6 | 12 |
| 20M | 12 | 12 | 0 | 0 | 6 | 12 |

Cells, server banners, generated service and model configurations, per-second host and GPU
samples, client output and CloudWatch points are under `raw/`. `raw/results.json` is this
report's only input. `verify.py` re-derives every number above from it.
