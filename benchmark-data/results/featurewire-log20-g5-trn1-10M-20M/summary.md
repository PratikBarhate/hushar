# hushar on its own wire with the log on: one A10G and one Trainium1

`featurewire-log20-g5-trn1-10M-20M` | collected 2026-11-13T18:26:34Z | rev `38cb6b0d`

## Configuration

| setting | value |
|---|---|
| servers | `g5.2xlarge` 1x NVIDIA A10G 8 vCPU; `trn1.2xlarge` 1x Trainium1 2 NeuronCore 8 vCPU |
| clients | `m9g.8xlarge`, 32 vCPU, one per lane, same availability zone |
| models | 10M and 20M, FP32 |
| wire | 210 named features per row, transformed and concatenated by hushar into the model's one 690-float input; server-side vectorization ON |
| feature logging | `inference_log.sample_rate` 0.2 -> 1 request in 5, to Amazon S3 |
| log object | one per 50,000 logged rows -- 500 requests at 100 rows, 100 requests at 500 rows, so ~50 MiB whichever packaging; 0 write failure(s) and 1 cell(s) that shed, across the round |
| candidate model | none; every request scores the one resident model |
| mini batch | none on the A10G (`mini_batch_size` unset); equal to the row count on Trainium, where a NEFF's leading dimension is fixed. Both are one engine call per request |
| parallelism | `threading.sessions_per_model` 2, `sessions_per_request` 1, so 2 requests in flight; `instance_group` 2 on Triton |
| Neuron placement | 1 core per replica, 1 execution context per replica, 1 replica per core, so both cores are occupied |
| hushar engine | ONNX Runtime 1.28.3 TensorRT provider on the A10G; libnrt 2.33.10.0 on Trainium |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `max_batch_size: 0`, the 690-float vector in, no feature logging |
| TensorRT | 10.16.1, the same library for both A10G arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and a version check at preflight |
| window | 902s measured in 15 slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | 18 measured, 17 sustained the offered rate |

## Sustained medians

| accel | model | rows x TPS | rows/s | stack | p50 | p95 | p99 | graph | vectorize | device | host CPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | 100,000 | hushar | 2.63 | 3.14 | 3.50 | 0.854 | 0.244 | 66.2% | 25.5% |
| A10G | 10M | 500 x 200 | 100,000 | hushar | 7.37 | 8.32 | 8.79 | 3.178 | 1.111 | 59.5% | 17.1% |
| A10G | 10M | 100 x 1000 | 100,000 | triton | 1.54 | 1.85 | 8.39 | -- | -- | 71.1% | 10.0% |
| A10G | 10M | 500 x 200 | 100,000 | triton | 4.96 | 5.30 | 5.31 | -- | -- | 59.9% | 5.6% |
| A10G | 20M | 100 x 700 | 70,000 | hushar | 4.14 | 5.20 | 5.73 | 2.362 | 0.244 | 96.0% | 30.9% |
| A10G | 20M | 100 x 720 | 72,000 | hushar | 4.17 | 5.41 | 7.45 | 2.361 | 0.246 | 97.6% | 31.9% |
| A10G | 20M | 500 x 150 | 75,000 | hushar | 11.86 | 13.44 | 15.66 | 7.658 | 1.116 | 98.9% | 22.2% |
| A10G | 20M | 100 x 700 | 70,000 | triton | 2.80 | 3.19 | 3.37 | -- | -- | 99.8% | 8.1% |
| A10G | 20M | 100 x 720 | 72,000 | triton | 3.11 | 3.51 | 3.64 | -- | -- | 99.8% | 8.3% |
| A10G | 20M | 100 x 750 | 75,000 | triton | 3.87 | 7.72 | 8.85 | -- | -- | 99.2% | 8.8% |
| A10G | 20M | 500 x 150 | 75,000 | triton | 9.37 | 9.85 | 9.99 | -- | -- | 99.9% | 4.9% |
| trn1 | 10M | 100 x 1000 | 100,000 | hushar | 3.48 | 4.30 | 5.07 | 1.644 | 0.224 | 75.2% | 33.3% |
| trn1 | 10M | 500 x 200 | 100,000 | hushar | 11.98 | 13.39 | 14.12 | 8.089 | 0.928 | 76.5% | 28.1% |
| trn1 | 20M | 100 x 400 | 40,000 | hushar | 4.93 | 5.23 | 5.46 | 3.859 | 0.235 | 72.7% | 23.5% |
| trn1 | 20M | 100 x 450 | 45,000 | hushar | 5.10 | 5.73 | 6.18 | 3.860 | 0.224 | 81.8% | 26.8% |
| trn1 | 20M | 500 x 75 | 37,500 | hushar | 22.20 | 23.19 | 23.47 | 18.985 | 0.914 | 69.8% | 20.8% |
| trn1 | 20M | 500 x 80 | 40,000 | hushar | 22.20 | 23.21 | 23.46 | 18.994 | 0.912 | 74.5% | 22.1% |

## Did not hold the offered rate

| accel | model | rows x TPS | stack | offered | achieved | shortfall | slices held | client shed | device | host CPU | graph |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 20M | 100 x 750 | hushar | 750 | 732.5 | 2.3% | 0/15 | 17617 | 99.0% | 33.9% | 2.370 |

No latency is quoted for these: a median taken while requests queue measures the queue. `graph` is still the per-request device time, which is what says whether the device or the host was the limit.

## Models

| accelerator | model | parameters | input width | classes | artifact |
|---|---|--:|--:|--:|---|
| A10G | 10M | 9,138,693 | 690 | 5 | ONNX graph |
| A10G | 20M | 18,589,701 | 690 | 5 | ONNX graph |
| trn1 | 10M | -- | 690 | 5 | NEFF |
| trn1 | 20M | -- | 690 | 5 | NEFF |

The Trainium parameter counts differ from the ONNX ones because `torch_neuronx.trace` rebuilds the same seeded architecture; both are recorded rather than treated as one number.

## Every cell

| accel | model | rows x TPS | stack | sessions | result | p50 | p95 | p99 | graph | vectorize | CPU | CPU max | device | slices |
|---|---|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | hushar | 2 | ok 1000.0/1000 | 2.63 | 3.14 | 3.50 | 0.854 | 0.244 | 25.5% | 45.0% | 66.2% | 15/15 |
| A10G | 10M | 500 x 200 | hushar | 2 | ok 200.0/200 | 7.37 | 8.32 | 8.79 | 3.178 | 1.111 | 17.1% | 29.7% | 59.5% | 15/15 |
| A10G | 10M | 100 x 1000 | triton | 2 | ok 1000.0/1000 | 1.54 | 1.85 | 8.39 | -- | -- | 10.0% | 25.8% | 71.1% | 15/15 |
| A10G | 10M | 500 x 200 | triton | 2 | ok 200.0/200 | 4.96 | 5.30 | 5.31 | -- | -- | 5.6% | 22.9% | 59.9% | 15/15 |
| A10G | 20M | 100 x 700 | hushar | 2 | ok 700.0/700 | 4.14 | 5.20 | 5.73 | 2.362 | 0.244 | 30.9% | 41.4% | 96.0% | 15/15 |
| A10G | 20M | 100 x 720 | hushar | 2 | ok 720.0/720 | 4.17 | 5.41 | 7.45 | 2.361 | 0.246 | 31.9% | 48.9% | 97.6% | 15/15 |
| A10G | 20M | 100 x 750 | hushar | 2 | FAIL 732.5/750 | -- | -- | -- | 2.370 | 0.328 | 33.9% | 57.4% | 99.0% | 0/15 |
| A10G | 20M | 500 x 150 | hushar | 2 | ok 150.0/150 | 11.86 | 13.44 | 15.66 | 7.658 | 1.116 | 22.2% | 37.0% | 98.9% | 15/15 |
| A10G | 20M | 100 x 700 | triton | 2 | ok 700.0/700 | 2.80 | 3.19 | 3.37 | -- | -- | 8.1% | 28.4% | 99.8% | 15/15 |
| A10G | 20M | 100 x 720 | triton | 2 | ok 720.0/720 | 3.11 | 3.51 | 3.64 | -- | -- | 8.3% | 25.7% | 99.8% | 15/15 |
| A10G | 20M | 100 x 750 | triton | 2 | ok 750.0/750 | 3.87 | 7.72 | 8.85 | -- | -- | 8.8% | 25.4% | 99.2% | 15/15 |
| A10G | 20M | 500 x 150 | triton | 2 | ok 150.0/150 | 9.37 | 9.85 | 9.99 | -- | -- | 4.9% | 23.9% | 99.9% | 15/15 |
| trn1 | 10M | 100 x 1000 | hushar | 2 | ok 1000.0/1000 | 3.48 | 4.30 | 5.07 | 1.644 | 0.224 | 33.3% | 51.4% | 75.2% | 15/15 |
| trn1 | 10M | 500 x 200 | hushar | 2 | ok 200.0/200 | 11.98 | 13.39 | 14.12 | 8.089 | 0.928 | 28.1% | 40.2% | 76.5% | 15/15 |
| trn1 | 20M | 100 x 400 | hushar | 2 | ok 400.0/400 | 4.93 | 5.23 | 5.46 | 3.859 | 0.235 | 23.5% | 34.6% | 72.7% | 15/15 |
| trn1 | 20M | 100 x 450 | hushar | 2 | ok 450.0/450 | 5.10 | 5.73 | 6.18 | 3.860 | 0.224 | 26.8% | 46.1% | 81.8% | 15/15 |
| trn1 | 20M | 500 x 75 | hushar | 2 | ok 75.0/75 | 22.20 | 23.19 | 23.47 | 18.985 | 0.914 | 20.8% | 39.6% | 69.8% | 15/15 |
| trn1 | 20M | 500 x 80 | hushar | 2 | ok 80.0/80 | 22.20 | 23.21 | 23.46 | 18.994 | 0.912 | 22.1% | 33.1% | 74.5% | 15/15 |

`graph` is hushar's `InferenceTime` and `vectorize` its `VectorizationTime`, medians of whole minute buckets inside the measured window. `device` is GPU utilization on the A10G and mean NeuronCore utilization on Trainium. A latency is withheld on a cell that did not hold its offered rate.

## A10G: hushar against Triton

| model | rows x TPS | hushar | p50 | p95 | Triton | p50 | p95 | hushar / Triton | hushar vectorize | hushar graph | Triton graph |
|---|--:|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| 10M | 100 x 1000 | ok 1000.0/1000 | 2.63 | 3.14 | ok 1000.0/1000 | 1.54 | 1.85 | 1.71x | 0.244 | 0.854 | -- |
| 10M | 500 x 200 | ok 200.0/200 | 7.37 | 8.32 | ok 200.0/200 | 4.96 | 5.30 | 1.49x | 1.111 | 3.178 | -- |
| 20M | 100 x 700 | ok 700.0/700 | 4.14 | 5.20 | ok 700.0/700 | 2.80 | 3.19 | 1.48x | 0.244 | 2.362 | -- |
| 20M | 100 x 720 | ok 720.0/720 | 4.17 | 5.41 | ok 720.0/720 | 3.11 | 3.51 | 1.34x | 0.246 | 2.361 | -- |
| 20M | 100 x 750 | FAIL 732.5/750 | -- | -- | ok 750.0/750 | 3.87 | 7.72 | -- | 0.328 | 2.370 | -- |
| 20M | 500 x 150 | ok 150.0/150 | 11.86 | 13.44 | ok 150.0/150 | 9.37 | 9.85 | 1.27x | 1.116 | 7.658 | -- |

Same GPU, same TensorRT library, same engine shape, same offered rate, one graph call per request on both sides. hushar additionally vectorizes 210 named features and writes one request in 5 of them to S3; Triton receives the finished tensor and writes nothing. So the ratio prices hushar as deployed against Triton as it ships, not two servers doing the same work.

## Against `featurewire-log50-g5-trn1-10M-20M`

| accel | model | rows x TPS | sessions | this round p50 | control p50 | delta | ratio | this round graph | control graph |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | 2 | 2.63 | 2.57 | +0.06 | 1.02x | 0.854 | 0.855 |
| A10G | 10M | 500 x 200 | 2 | 7.37 | 7.39 | -0.02 | 1.00x | 3.178 | 3.178 |
| A10G | 20M | 100 x 700 | 2 | 4.14 | -- | -- | -- | 2.362 | no control cell at this rate |
| A10G | 20M | 100 x 720 | 2 | 4.17 | -- | -- | -- | 2.361 | no control cell at this rate |
| A10G | 20M | 100 x 750 | 2 | -- | -- | -- | -- | 2.370 | 2.374 |
| A10G | 20M | 500 x 150 | 2 | 11.86 | 12.02 | -0.16 | 0.99x | 7.658 | 7.743 |
| trn1 | 10M | 100 x 1000 | 2 | 3.48 | 3.64 | -0.16 | 0.96x | 1.644 | 1.645 |
| trn1 | 10M | 500 x 200 | 2 | 11.98 | 12.26 | -0.28 | 0.98x | 8.089 | 8.222 |
| trn1 | 20M | 100 x 400 | 2 | 4.93 | -- | -- | -- | 3.859 | no control cell at this rate |
| trn1 | 20M | 100 x 450 | 2 | 5.10 | -- | -- | -- | 3.860 | no control cell at this rate |
| trn1 | 20M | 500 x 75 | 2 | 22.20 | -- | -- | -- | 18.985 | no control cell at this rate |
| trn1 | 20M | 500 x 80 | 2 | 22.20 | -- | -- | -- | 18.994 | no control cell at this rate |

The control is the same grid at `sample_rate` 0.5 on the same wire, so a delta here is the log rate alone. The Trainium 20M rates and the A10G's 100 x 700 and 100 x 720 differ between the two rounds, so those cells have no pair and report `--`. `graph` is carried for every row including the unpaired ones, because a device time that did not move is what says a latency change came from the host.
`featurewire-log50-g5-trn1-10M-20M` is **not retained** as a report. Its figures above are cited from `raw/control-featurewire-log50-g5-trn1-10M-20M.json`, which carries the p50, the achieved rate and the `InferenceTime` median per cell and names its source: s3://<bucket>/runs/featurewire-log50-g5-trn1-10M-20M/cells/ plus CloudWatch HusharBench InferenceTime. Each lane's own bucket reduction was reproduced -- the mean of per-minute p50s on the A10G, the median of them on Trainium -- and the eight figures that round published match to three decimals. The banners, samples and client output behind them are not here.

## Where the median goes

| accel | model | rows x TPS | p50 | graph | vectorize | to tensor | remainder | buckets | vectorize share |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | 2.63 | 0.854 | 0.244 | 0.000 | 1.532 | 15 | 9.3% |
| A10G | 10M | 500 x 200 | 7.37 | 3.178 | 1.111 | 0.000 | 3.081 | 15 | 15.1% |
| A10G | 20M | 100 x 700 | 4.14 | 2.362 | 0.244 | 0.000 | 1.534 | 15 | 5.9% |
| A10G | 20M | 100 x 720 | 4.17 | 2.361 | 0.246 | 0.000 | 1.563 | 15 | 5.9% |
| A10G | 20M | 100 x 750 | -- | 2.370 | 0.328 | 0.000 | -- | 15 | -- |
| A10G | 20M | 500 x 150 | 11.86 | 7.658 | 1.116 | 0.000 | 3.086 | 15 | 9.4% |
| trn1 | 10M | 100 x 1000 | 3.48 | 1.644 | 0.224 | -- | 1.612 | 15 | 6.4% |
| trn1 | 10M | 500 x 200 | 11.98 | 8.089 | 0.928 | -- | 2.963 | 15 | 7.7% |
| trn1 | 20M | 100 x 400 | 4.93 | 3.859 | 0.235 | -- | 0.836 | 15 | 4.8% |
| trn1 | 20M | 100 x 450 | 5.10 | 3.860 | 0.224 | -- | 1.016 | 15 | 4.4% |
| trn1 | 20M | 500 x 75 | 22.20 | 18.985 | 0.914 | -- | 2.301 | 15 | 4.1% |
| trn1 | 20M | 500 x 80 | 22.20 | 18.994 | 0.912 | -- | 2.294 | 15 | 4.1% |

`remainder` is the client median less the three server-side measures: gRPC, protobuf decode, scheduling, the log's own copy, and the network. `to tensor` is not published on the Neuron backend.

## Device and host

| accel | model | rows x TPS | stack | device | device max | device MiB | power W | host CPU | host CPU max | host mem | client CPU |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | hushar | 66.2% | 68.0% | 443 | 190.9 | 25.5% | 45.0% | 8.4% | 1.2% |
| A10G | 10M | 500 x 200 | hushar | 59.5% | 62.0% | 597 | 185.5 | 17.1% | 29.7% | 8.4% | 0.9% |
| A10G | 10M | 100 x 1000 | triton | 71.1% | 75.0% | 431 | 188.3 | 10.0% | 25.8% | 7.2% | 0.3% |
| A10G | 10M | 500 x 200 | triton | 59.9% | 60.0% | 581 | 185.2 | 5.6% | 22.9% | 7.2% | 0.3% |
| A10G | 20M | 100 x 700 | hushar | 96.0% | 98.0% | 507 | 276.6 | 30.9% | 41.4% | 8.8% | 0.9% |
| A10G | 20M | 100 x 720 | hushar | 97.6% | 100.0% | 809 | 284.3 | 31.9% | 48.9% | 12.4% | 0.9% |
| A10G | 20M | 100 x 750 | hushar | 99.0% | 100.0% | 507 | 286.7 | 33.9% | 57.4% | 9.3% | 1.1% |
| A10G | 20M | 500 x 150 | hushar | 98.9% | 100.0% | 661 | 289.0 | 22.2% | 37.0% | 8.9% | 0.8% |
| A10G | 20M | 100 x 700 | triton | 99.8% | 100.0% | 463 | 274.8 | 8.1% | 28.4% | 7.2% | 0.3% |
| A10G | 20M | 100 x 720 | triton | 99.8% | 100.0% | 463 | 276.5 | 8.3% | 25.7% | 7.2% | 0.3% |
| A10G | 20M | 100 x 750 | triton | 99.2% | 100.0% | 463 | 293.3 | 8.8% | 25.4% | 7.2% | 0.2% |
| A10G | 20M | 500 x 150 | triton | 99.9% | 100.0% | 613 | 289.8 | 4.9% | 23.9% | 7.2% | 0.2% |
| trn1 | 10M | 100 x 1000 | hushar | 75.2% | 75.4% | 407 | -- | 33.3% | 51.4% | 1.1% | 1.2% |
| trn1 | 10M | 500 x 200 | hushar | 76.5% | 77.1% | 426 | -- | 28.1% | 40.2% | 1.7% | 0.9% |
| trn1 | 20M | 100 x 400 | hushar | 72.7% | 72.9% | 485 | -- | 23.5% | 34.6% | 1.4% | 0.5% |
| trn1 | 20M | 100 x 450 | hushar | 81.8% | 82.1% | 485 | -- | 26.8% | 46.1% | 1.3% | 0.6% |
| trn1 | 20M | 500 x 75 | hushar | 69.8% | 70.8% | 531 | -- | 20.8% | 39.6% | 2.6% | 0.4% |
| trn1 | 20M | 500 x 80 | hushar | 74.5% | 74.6% | 531 | -- | 22.1% | 33.1% | 2.6% | 0.4% |

## Inference log

| accel | model | rows x TPS | rate | requests per object | rows per object | objects | written GiB | predicted MB/s | measured MB/s | shed batches | write failures |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | 100 x 1000 | 1 in 5 | 500 | 50,000 | 1442 | 74.4 | 22.2 | 66.5 | 0 | 0 |
| A10G | 10M | 500 x 200 | 1 in 5 | 100 | 50,000 | 480 | 23.6 | 21.3 | 21.1 | 0 | 0 |
| A10G | 20M | 100 x 700 | 1 in 5 | 500 | 50,000 | 336 | 17.4 | 15.5 | 15.5 | 0 | 0 |
| A10G | 20M | 100 x 720 | 1 in 5 | 500 | 50,000 | 345 | 17.8 | 16.0 | 15.8 | 0 | 0 |
| A10G | 20M | 100 x 750 | 1 in 5 | 500 | 50,000 | 1058 | 54.6 | 16.6 | 48.6 | 123900 | 0 |
| A10G | 20M | 500 x 150 | 1 in 5 | 100 | 50,000 | 360 | 17.7 | 15.9 | 15.8 | 0 | 0 |
| trn1 | 10M | 100 x 1000 | 1 in 5 | 500 | 50,000 | 1442 | 74.4 | 22.2 | 66.6 | 0 | 0 |
| trn1 | 10M | 500 x 200 | 1 in 5 | 100 | 50,000 | 638 | 31.4 | 21.3 | 28.1 | 0 | 0 |
| trn1 | 20M | 100 x 400 | 1 in 5 | 500 | 50,000 | 248 | 12.8 | 8.9 | 11.4 | 0 | 0 |
| trn1 | 20M | 100 x 450 | 1 in 5 | 500 | 50,000 | 650 | 33.5 | 10.0 | 30.0 | 0 | 0 |
| trn1 | 20M | 500 x 75 | 1 in 5 | 100 | 50,000 | 180 | 8.9 | 8.0 | 7.9 | 0 | 0 |
| trn1 | 20M | 500 x 80 | 1 in 5 | 100 | 50,000 | 192 | 9.5 | 8.5 | 8.5 | 0 | 0 |

`predicted MB/s` is the offered rate times the sampled fraction times the measured feature-wire payload. `measured MB/s` is the bytes that reached the bucket over warm-up plus measured window, which is the span they were written across. A non-zero `shed batches` means the accumulator dropped its oldest buffered batch because every send permit was out, so that cell paid less than the full cost of logging; `write failures` are PUTs that did not complete.

## Rate held

| accel | model | stack | rows x TPS | result | slices held | shed | failed | window s | rows/s offered | Gbps | % baseline |
|---|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| A10G | 10M | hushar | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 902 | 100,000 | -- | -- |
| A10G | 10M | hushar | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 | 100,000 | -- | -- |
| A10G | 10M | triton | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 903 | 100,000 | -- | -- |
| A10G | 10M | triton | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 905 | 100,000 | -- | -- |
| A10G | 20M | hushar | 100 x 700 | ok 700.0/700 | 15/15 | 0 | 0 | 902 | 70,000 | -- | -- |
| A10G | 20M | hushar | 100 x 720 | ok 720.0/720 | 15/15 | 0 | 0 | 909 | 72,000 | -- | -- |
| A10G | 20M | hushar | 100 x 750 | FAIL 732.5/750 | 0/15 | 17617 | 0 | 906 | 75,000 | -- | -- |
| A10G | 20M | hushar | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 902 | 75,000 | -- | -- |
| A10G | 20M | triton | 100 x 700 | ok 700.0/700 | 15/15 | 0 | 0 | 903 | 70,000 | -- | -- |
| A10G | 20M | triton | 100 x 720 | ok 720.0/720 | 15/15 | 0 | 0 | 903 | 72,000 | -- | -- |
| A10G | 20M | triton | 100 x 750 | ok 750.0/750 | 15/15 | 0 | 0 | 903 | 75,000 | -- | -- |
| A10G | 20M | triton | 500 x 150 | ok 150.0/150 | 15/15 | 0 | 0 | 905 | 75,000 | -- | -- |
| trn1 | 10M | hushar | 100 x 1000 | ok 1000.0/1000 | 15/15 | 0 | 0 | 900 | 100,000 | 1.08 | 34.7% |
| trn1 | 10M | hushar | 500 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 900 | 100,000 | 1.03 | 33.0% |
| trn1 | 20M | hushar | 100 x 400 | ok 400.0/400 | 15/15 | 0 | 0 | 900 | 40,000 | 0.43 | 13.9% |
| trn1 | 20M | hushar | 100 x 450 | ok 450.0/450 | 15/15 | 0 | 0 | 900 | 45,000 | 0.49 | 15.7% |
| trn1 | 20M | hushar | 500 x 75 | ok 75.0/75 | 15/15 | 0 | 0 | 900 | 37,500 | 0.39 | 12.5% |
| trn1 | 20M | hushar | 500 x 80 | ok 80.0/80 | 15/15 | 0 | 0 | 900 | 40,000 | 0.41 | 13.2% |

A cell sustains only when every slice held the offered rate within 2% with nothing shed and nothing failed.

## Provenance

| accel | host | instance | cells | engine | TensorRT | Triton image |
|---|---|---|--:|---|---|---|
| A10G | g5-1 | g5.2xlarge | 4 | ONNX Runtime 1.28.3 | 10.16.1 | nvcr.io/nvidia/tritonserver:26.05-py3 |
| A10G | g5-2 | g5.2xlarge | 4 | ONNX Runtime 1.28.3 | 10.16.1 | nvcr.io/nvidia/tritonserver:26.05-py3 |
| A10G | g5-3 | g5.2xlarge | 4 | ONNX Runtime 1.28.3 | 10.16.1 | nvcr.io/nvidia/tritonserver:26.05-py3 |
| trn1 | trn1-1 | trn1.2xlarge | 2 | libnrt 2.33.10.0 | -- | -- |
| trn1 | trn1-2 | trn1.2xlarge | 4 | libnrt 2.33.10.0 | -- | -- |


| measure | count |
|---|--:|
| cells | 18 |
| sustained | 17 |
| hushar | 12 |
| triton | 6 |
| with CloudWatch | 12 |
| with device samples | 18 |
| cells that shed the log | 1 |
| log write failures | 0 |

Cells, server banners, generated service and model configurations, per-second host and device samples, client output and CloudWatch points are under `raw/`. `raw/results-gpu.json` and `raw/results-neuron.json` are this report's only inputs. `verify.py` re-derives every number above from them.
