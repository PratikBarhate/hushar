# A 100M model on its own wire with the log on: FP32 against BF16 on one A10G and one Trainium1

`featurewire-log20-prec-g5-trn1-100M` | collected 2026-11-17T19:00:10Z | rev `4c25e535-dirty`

## Configuration

| setting | value |
|---|---|
| servers | `g5.2xlarge` 1x NVIDIA A10G 8 vCPU; `trn1.2xlarge` 1x Trainium1 2 NeuronCore 8 vCPU |
| clients | `m9g.8xlarge`, 32 vCPU, one per lane, same availability zone |
| model | one 100M graph -- 100,498,437 parameters, 31 encoder layers -- served at two precisions |
| precision | FP32 and BF16. On the A10G both arms build the SAME ONNX bytes into a TensorRT engine, FP32 at the builder's defaults and BF16 through `trt_bf16_enable=1` on hushar and `--bf16` on `trtexec`. On Trainium each arm is a separate `neuronx-cc` compile, `--auto-cast=none` against `--auto-cast=matmult --auto-cast-type=bf16` |
| precision is explicit | Both Trainium arms pass `--auto-cast` rather than relying on a default. `neuronx-cc`'s CLI reference gives the flag a default of `matmult`, but `torch_neuronx.trace` does NOT inherit it: compiling one model with no flag, with `--auto-cast=none` and with `--auto-cast=matmult` produced a BYTE-IDENTICAL NEFF for the first two. So the earlier Trainium rounds in this series, which passed no flag, were genuinely FP32 and ARE comparable with this round's FP32 arm. The flag is passed anyway so the label rests on the artifact rather than on a framework default -- see `raw/probe/autocast-default.txt` |
| wire | 210 named features per row, transformed and concatenated by hushar into the model's one 690-float input; server-side vectorization ON |
| feature logging | `inference_log.sample_rate` 0.2 -> 1 request in 5, to Amazon S3 |
| log object | one per 50,000 logged rows -- 500 requests at 100 rows, 100 requests at 500 rows, so ~50 MiB whichever packaging; 209 write failure(s) and 2 cell(s) that shed, across the round |
| candidate model | none; every request scores the one resident model |
| mini batch | none on the A10G (`mini_batch_size` unset); equal to the row count on Trainium, where a NEFF's leading dimension is fixed. Both are one engine call per request |
| parallelism | `threading.sessions_per_model` 2, `sessions_per_request` 1, so 2 requests in flight; `instance_group` 2 on Triton |
| Neuron placement | 1 core per replica, 1 execution context per replica, 1 replica per core, so both cores are occupied |
| hushar engine | ONNX Runtime 1.28.3 TensorRT provider on the A10G; libnrt 2.33.10.0 on Trainium |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-py3`, `tensorrt` backend, `max_batch_size: 0`, the 690-float vector in, no feature logging |
| TensorRT | 10.16.1, the same library for both A10G arms: CUDA, cuDNN and TensorRT are extracted from the Triton image and hushar links those copies, gated by `ldd` at deploy and a version check at preflight |
| window | 902s measured in 15 slices, after a 300s warm-up discarded; fresh server per cell, 60s idle between |
| cells | 18 measured, 14 sustained the offered rate |

## Sustained medians

| accel | precision | rows x TPS | rows/s | stack | p50 | p95 | p99 | graph | vectorize | device | host CPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | 10,000 | hushar | 10.97 | 11.34 | 11.57 | 9.817 | 0.249 | 93.3% | 14.4% |
| A10G | BF16 | 100 x 200 | 20,000 | hushar | 5.85 | 6.30 | 6.65 | 4.685 | 0.236 | 89.1% | 15.6% |
| A10G | FP32 | 500 x 20 | 10,000 | hushar | 44.00 | 44.52 | 44.82 | 40.757 | 1.112 | 79.8% | 11.9% |
| A10G | BF16 | 500 x 40 | 20,000 | hushar | 22.10 | 22.57 | 22.94 | 18.311 | 1.106 | 72.7% | 12.0% |
| A10G | FP32 | 100 x 100 | 10,000 | triton | 10.27 | 10.45 | 10.52 | -- | -- | 93.5% | 3.2% |
| A10G | BF16 | 100 x 200 | 20,000 | triton | 5.06 | 5.29 | 5.40 | -- | -- | 89.1% | 7.0% |
| A10G | FP32 | 500 x 20 | 10,000 | triton | 42.10 | 42.40 | 42.50 | -- | -- | 79.9% | 4.3% |
| A10G | BF16 | 500 x 40 | 20,000 | triton | 20.11 | 20.36 | 20.46 | -- | -- | 72.7% | 2.6% |
| trn1 | BF16 | 100 x 70 | 7,000 | hushar | 16.46 | 16.73 | 16.81 | 15.457 | 0.229 | 53.5% | 14.8% |
| trn1 | FP32 | 100 x 70 | 7,000 | hushar | 22.74 | 23.02 | 23.10 | 21.734 | 0.228 | 75.4% | 20.2% |
| trn1 | BF16 | 100 x 100 | 10,000 | hushar | 16.44 | 16.72 | 16.80 | 15.473 | 0.225 | 76.4% | 20.7% |
| trn1 | BF16 | 500 x 10 | 5,000 | hushar | 75.14 | 75.38 | 75.61 | 71.958 | 0.941 | 35.8% | 9.7% |
| trn1 | FP32 | 500 x 10 | 5,000 | hushar | 117.22 | 118.16 | 118.36 | 113.650 | 0.976 | 56.6% | 15.0% |
| trn1 | BF16 | 500 x 20 | 10,000 | hushar | 75.08 | 76.03 | 76.22 | 71.866 | 0.926 | 71.5% | 19.2% |

## Did not hold the offered rate

| accel | precision | rows x TPS | stack | offered | achieved | shortfall | slices held | client shed | device | host CPU | graph |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| trn1 | FP32 | 100 x 100 | hushar | 100 | 90.9 | 9.1% | 0/15 | 10510 | 97.9% | 26.3% | 21.927 |
| trn1 | BF16 | 100 x 200 | hushar | 200 | 127.1 | 36.5% | 0/15 | 86721 | 97.2% | 26.8% | 16.020 |
| trn1 | FP32 | 500 x 20 | hushar | 20 | 17.4 | 13.0% | 0/15 | 3015 | 97.2% | 26.0% | 114.895 |
| trn1 | BF16 | 500 x 40 | hushar | 40 | 27.4 | 31.5% | 0/15 | 14971 | 97.7% | 26.3% | 73.104 |

No latency is quoted for these: a median taken while requests queue measures the queue. `graph` is still the per-request device time, which is what says whether the device or the host was the limit.

## FP32 against BF16

| accel | stack | rows | TPS offered | FP32 | BF16 | FP32 graph | BF16 graph | graph speedup | FP32 rows/s | BF16 rows/s | throughput | artifact bytes |
|---|---|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|
| A10G | hushar | 100 rows | 100 -> 200 | ok 100.0/100 | ok 200.0/200 | 9.817 | 4.685 | 2.10x | 10,000 | 20,000 | 2.00x | 0.503x |
| A10G | hushar | 500 rows | 20 -> 40 | ok 20.0/20 | ok 40.0/40 | 40.757 | 18.311 | 2.23x | 10,000 | 20,000 | 2.00x | 0.511x |
| A10G | triton | 100 rows | 100 -> 200 | ok 100.0/100 | ok 200.0/200 | -- | -- | -- | 10,000 | 20,000 | 2.00x | 0.503x |
| A10G | triton | 500 rows | 20 -> 40 | ok 20.0/20 | ok 40.0/40 | -- | -- | -- | 10,000 | 20,000 | 2.00x | 0.511x |
| trn1 | hushar | 100 rows | 100 -> 200 | FAIL 90.9/100 | FAIL 127.1/200 | 21.927 | 16.020 | 1.37x | 9,090 | 12,710 | 1.40x | 0.429x |
| trn1 | hushar | 500 rows | 20 -> 40 | FAIL 17.4/20 | FAIL 27.4/40 | 114.895 | 73.104 | 1.57x | 8,700 | 13,700 | 1.57x | 0.448x |

The two arms are NOT offered the same rate -- BF16 is asked for double at each packaging -- so a median against a median would price the rate and the precision together. `graph speedup` is FP32's per-request device time over BF16's, which is what the precision bought; `throughput` is the rows per second each arm actually sustained. `artifact bytes` is the BF16 engine or NEFF over its FP32 pair: a builder that accepted the flag and kept every layer at full width would show the flag and not the shrink.

## Precision evidence

| accel | precision | builder flag | artifact bytes | trace error | BF16 flag in banner |
|---|---|---|--:|--:|--:|
| A10G | FP32 | builder defaults | 403,564,460, 403,567,372, 403,594,796, 403,617,164 | -- | no |
| A10G | BF16 | `trt_bf16_enable=1` | 202,952,636, 202,954,300, 206,186,236, 206,186,332 | -- | yes |
| trn1 | FP32 | `--auto-cast=none` | 377,519,104, 403,211,264 | 1.48e-06 | -- |
| trn1 | BF16 | `--auto-cast=matmult` | 162,008,064, 180,614,144 | 0.00232 | -- |

Three independent accounts of the same claim, because a precision label is the one thing in this round that could be wrong while every other gate passed. The A10G reads the flag back out of the server's own banner and measures the engine it built. Trainium cannot read a flag back, so it uses the compiled-versus-PyTorch maximum error: FP32 lands near 1e-6 and BF16 matmuls three orders above it, and `compile_neuron.py` REFUSES to write a NEFF whose error falls outside the window its label requires.

That check earned its place immediately. This round was planned on the belief that `neuronx-cc`'s documented `--auto-cast` default of `matmult` reached `torch_neuronx.trace`, which would have made every earlier Trainium round's FP32 label wrong. The recorded trace errors of those rounds -- ~1.5e-06, against the 2.3e-03 this round's BF16 arm produces -- said otherwise, and a measured three-way compile confirmed it: no flag and `--auto-cast=none` produce a byte-identical NEFF. The earlier rounds were FP32. `raw/probe/autocast-default.txt` has the comparison.

## FP32 against BF16 at the same offered rate

| accel | stack | rows x TPS | rows/s | FP32 | p50 | graph | device | BF16 | p50 | graph | device | p50 ratio | graph ratio |
|---|---|--:|--:|---|--:|--:|--:|---|--:|--:|--:|--:|--:|
| trn1 | hushar | 100 x 70 | 7,000 | ok 70.0/70 | 22.74 | 21.734 | 75.4% | ok 70.0/70 | 16.46 | 15.457 | 53.5% | 1.38x | 1.41x |
| trn1 | hushar | 100 x 100 | 10,000 | FAIL 90.9/100 | -- | 21.927 | 97.9% | ok 100.0/100 | 16.44 | 15.473 | 76.4% | -- | 1.42x |
| trn1 | hushar | 500 x 10 | 5,000 | ok 10.0/10 | 117.22 | 113.650 | 56.6% | ok 10.0/10 | 75.14 | 71.958 | 35.8% | 1.56x | 1.58x |
| trn1 | hushar | 500 x 20 | 10,000 | FAIL 17.4/20 | -- | 114.895 | 97.2% | ok 20.0/20 | 75.08 | 71.866 | 71.5% | -- | 1.60x |

Both arms offered the same rows and the same rate, so the ratio is the cast with no rate term in it. A `p50 ratio` is quoted only where BOTH arms held the rate; the `graph ratio` is quoted regardless, because the per-request device time is meaningful even on a cell that queued.

The rows where FP32 shows `FAIL` against a BF16 `ok` are the cast buying throughput rather than latency: the same offered rate is inside one arm's ceiling and outside the other's.

## What the Trainium cells measured

| precision | rows x TPS offered | offered | achieved | rows/s achieved | implied ms/call | server ms/call | agreement | NeuronCore | host CPU |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| FP32 | 100 x 100 | 100 | 90.9 | 9,090 | 22.00 | 21.93 | 0.3% | 97.9% | 26.3% |
| BF16 | 100 x 200 | 200 | 127.1 | 12,710 | 15.74 | 16.02 | 1.8% | 97.2% | 26.8% |
| FP32 | 500 x 20 | 20 | 17.4 | 8,700 | 114.94 | 114.89 | 0.0% | 97.2% | 26.0% |
| BF16 | 500 x 40 | 40 | 27.4 | 13,700 | 72.99 | 73.10 | 0.2% | 97.7% | 26.3% |

Not one Trainium cell held its offered rate, and that is what they were run to find out. The rates were set from the A10G, where all four were measured holding; one `trn1.2xlarge` is two NeuronCores, each executing one NEFF at a time. So these cells measure the CEILING, which no campaign had for this model size on this device.

`implied ms/call` is `2 / achieved`: with both cores busy and one execution each, the rate a saturated host completes requests at fixes the per-request device span. `server ms/call` is hushar's own `InferenceTime`. They are independent -- one is the client's completed requests, the other the server's timer -- and `agreement` is the gap between them.

**THE SERVER FIGURE IS A WHOLE-CELL MEDIAN ON THESE CELLS, NOT AN IN-WINDOW ONE.** Queueing for seconds saturated the host's async pool, which starved the server's own CloudWatch publishing: around 1,450 publishes per cell were lost to a 3.1 s connect timeout, leaving the measured window without one complete minute bucket. The whole-cell median needs only one bucket and survives, but it includes the warm-up. The same saturation is why these cells wrote almost no inference log -- see the log table.

## A10G against Trainium1

| precision | rows x TPS | A10G | p50 | graph | device | trn1 | p50 | graph | device | trn1 / A10G graph |
|---|--:|---|--:|--:|--:|---|--:|--:|--:|--:|
| FP32 | 100 x 100 | ok 100.0/100 | 10.97 | 9.817 | 93.3% | FAIL 90.9/100 | -- | 21.927 | 97.9% | 2.23x |
| FP32 | 500 x 20 | ok 20.0/20 | 44.00 | 40.757 | 79.8% | FAIL 17.4/20 | -- | 114.895 | 97.2% | 2.82x |
| BF16 | 100 x 200 | ok 200.0/200 | 5.85 | 4.685 | 89.1% | FAIL 127.1/200 | -- | 16.020 | 97.2% | 3.42x |
| BF16 | 500 x 40 | ok 40.0/40 | 22.10 | 18.311 | 72.7% | FAIL 27.4/40 | -- | 73.104 | 97.7% | 3.99x |

Both run hushar on the same wire at the same log rate and the same session count, so the row is the device and its toolchain. The graphs are NOT the same artifact: the A10G serves the ONNX file through TensorRT and Trainium serves a NEFF that `torch_neuronx.trace` rebuilt from the same seeded architecture, so the parameter counts differ slightly and are reported separately. A latency is withheld on a cell that did not hold its offered rate; `graph` is published regardless, because the per-request device time is what says where the ceiling is.

## Models

| accelerator | precision | parameters | layers | input width | classes | artifact | artifact bytes |
|---|---|--:|--:|--:|--:|---|--:|
| A10G | FP32 | 100,498,437 | 31 | 690 | 5 | TensorRT engine | 403,564,460, 403,567,372, 403,594,796, 403,617,164 |
| A10G | BF16 | 100,498,437 | 31 | 690 | 5 | TensorRT engine | 202,952,636, 202,954,300, 206,186,236, 206,186,332 |
| trn1 | FP32 | 100,561,925 | 31 | 690 | 5 | NEFF | 377,519,104, 403,211,264 |
| trn1 | BF16 | 100,561,925 | 31 | 690 | 5 | NEFF | 162,008,064, 180,614,144 |

One graph per accelerator, built twice. The parameter and layer counts do not move with the precision, because the cast is a builder or compiler choice rather than a change to the model. The Trainium parameter count differs from the ONNX one because `torch_neuronx.trace` rebuilds the same seeded architecture; both are recorded rather than treated as one number. Two artifact sizes per row are the two row packagings.

## Every cell

| accel | precision | rows x TPS | stack | sessions | result | p50 | p95 | p99 | graph | vectorize | CPU | CPU max | device | slices |
|---|---|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | hushar | 2 | ok 100.0/100 | 10.97 | 11.34 | 11.57 | 9.817 | 0.249 | 14.4% | 30.3% | 93.3% | 15/15 |
| A10G | BF16 | 100 x 200 | hushar | 2 | ok 200.0/200 | 5.85 | 6.30 | 6.65 | 4.685 | 0.236 | 15.6% | 33.0% | 89.1% | 15/15 |
| A10G | FP32 | 500 x 20 | hushar | 2 | ok 20.0/20 | 44.00 | 44.52 | 44.82 | 40.757 | 1.112 | 11.9% | 22.5% | 79.8% | 15/15 |
| A10G | BF16 | 500 x 40 | hushar | 2 | ok 40.0/40 | 22.10 | 22.57 | 22.94 | 18.311 | 1.106 | 12.0% | 27.5% | 72.7% | 15/15 |
| A10G | FP32 | 100 x 100 | triton | 2 | ok 100.0/100 | 10.27 | 10.45 | 10.52 | -- | -- | 3.2% | 17.1% | 93.5% | 15/15 |
| A10G | BF16 | 100 x 200 | triton | 2 | ok 200.0/200 | 5.06 | 5.29 | 5.40 | -- | -- | 7.0% | 25.0% | 89.1% | 15/15 |
| A10G | FP32 | 500 x 20 | triton | 2 | ok 20.0/20 | 42.10 | 42.40 | 42.50 | -- | -- | 4.3% | 25.7% | 79.9% | 15/15 |
| A10G | BF16 | 500 x 40 | triton | 2 | ok 40.0/40 | 20.11 | 20.36 | 20.46 | -- | -- | 2.6% | 22.7% | 72.7% | 15/15 |
| trn1 | BF16 | 100 x 70 | hushar | 2 | ok 70.0/70 | 16.46 | 16.73 | 16.81 | 15.457 | 0.229 | 14.8% | 31.1% | 53.5% | 15/15 |
| trn1 | FP32 | 100 x 70 | hushar | 2 | ok 70.0/70 | 22.74 | 23.02 | 23.10 | 21.734 | 0.228 | 20.2% | 36.5% | 75.4% | 15/15 |
| trn1 | BF16 | 100 x 100 | hushar | 2 | ok 100.0/100 | 16.44 | 16.72 | 16.80 | 15.473 | 0.225 | 20.7% | 31.6% | 76.4% | 15/15 |
| trn1 | FP32 | 100 x 100 | hushar | 2 | FAIL 90.9/100 | -- | -- | -- | 21.927 | -- | 26.3% | 37.5% | 97.9% | 0/15 |
| trn1 | BF16 | 100 x 200 | hushar | 2 | FAIL 127.1/200 | -- | -- | -- | 16.020 | -- | 26.8% | 46.6% | 97.2% | 0/15 |
| trn1 | BF16 | 500 x 10 | hushar | 2 | ok 10.0/10 | 75.14 | 75.38 | 75.61 | 71.958 | 0.941 | 9.7% | 22.0% | 35.8% | 15/15 |
| trn1 | FP32 | 500 x 10 | hushar | 2 | ok 10.0/10 | 117.22 | 118.16 | 118.36 | 113.650 | 0.976 | 15.0% | 27.0% | 56.6% | 15/15 |
| trn1 | BF16 | 500 x 20 | hushar | 2 | ok 20.0/20 | 75.08 | 76.03 | 76.22 | 71.866 | 0.926 | 19.2% | 36.6% | 71.5% | 15/15 |
| trn1 | FP32 | 500 x 20 | hushar | 2 | FAIL 17.4/20 | -- | -- | -- | 114.895 | -- | 26.0% | 45.5% | 97.2% | 0/15 |
| trn1 | BF16 | 500 x 40 | hushar | 2 | FAIL 27.4/40 | -- | -- | -- | 73.104 | -- | 26.3% | 44.2% | 97.7% | 0/15 |

`graph` is hushar's `InferenceTime` and `vectorize` its `VectorizationTime`, medians of whole minute buckets inside the measured window. `device` is GPU utilization on the A10G and mean NeuronCore utilization on Trainium. A latency is withheld on a cell that did not hold its offered rate.

## A10G: hushar against Triton

| precision | rows x TPS | hushar | p50 | p95 | Triton | p50 | p95 | hushar / Triton | hushar vectorize | hushar graph | Triton graph |
|---|--:|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| FP32 | 100 x 100 | ok 100.0/100 | 10.97 | 11.34 | ok 100.0/100 | 10.27 | 10.45 | 1.07x | 0.249 | 9.817 | -- |
| BF16 | 100 x 200 | ok 200.0/200 | 5.85 | 6.30 | ok 200.0/200 | 5.06 | 5.29 | 1.16x | 0.236 | 4.685 | -- |
| FP32 | 500 x 20 | ok 20.0/20 | 44.00 | 44.52 | ok 20.0/20 | 42.10 | 42.40 | 1.05x | 1.112 | 40.757 | -- |
| BF16 | 500 x 40 | ok 40.0/40 | 22.10 | 22.57 | ok 40.0/40 | 20.11 | 20.36 | 1.10x | 1.106 | 18.311 | -- |

Same GPU, same TensorRT library, same engine shape, same offered rate, one graph call per request on both sides. hushar additionally vectorizes 210 named features and writes one request in 5 of them to S3; Triton receives the finished tensor and writes nothing. So the ratio prices hushar as deployed against Triton as it ships, not two servers doing the same work.

## Against `nominibatch-precision-gpu-g5-100M`

| accel | precision | rows x TPS | sessions | this round p50 | control p50 | delta | ratio | this round graph | control graph |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | 2 | 10.97 | 11.05 | -0.08 | 0.99x | 9.817 | 9.652 |
| A10G | BF16 | 100 x 200 | 2 | 5.85 | 6.31 | -0.46 | 0.93x | 4.685 | 4.696 |
| A10G | FP32 | 500 x 20 | 2 | 44.00 | 45.90 | -1.90 | 0.96x | 40.757 | 40.176 |
| A10G | BF16 | 500 x 40 | 2 | 22.10 | 24.20 | -2.10 | 0.91x | 18.311 | 18.320 |
| trn1 | BF16 | 100 x 70 | 2 | 16.46 | -- | -- | -- | 15.457 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | FP32 | 100 x 70 | 2 | 22.74 | -- | -- | -- | 21.734 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | BF16 | 100 x 100 | 2 | 16.44 | -- | -- | -- | 15.473 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | FP32 | 100 x 100 | 2 | -- | -- | -- | -- | 21.927 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | BF16 | 100 x 200 | 2 | -- | -- | -- | -- | 16.020 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | BF16 | 500 x 10 | 2 | 75.14 | -- | -- | -- | 71.958 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | FP32 | 500 x 10 | 2 | 117.22 | -- | -- | -- | 113.650 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | BF16 | 500 x 20 | 2 | 75.08 | -- | -- | -- | 71.866 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | FP32 | 500 x 20 | 2 | -- | -- | -- | -- | 114.895 | no Trainium control: this model size has never been measured on Trainium |
| trn1 | BF16 | 500 x 40 | 2 | -- | -- | -- | -- | 73.104 | no Trainium control: this model size has never been measured on Trainium |

The control is the same model at the same two precisions and the same four loads on the same instance type, on the TENSOR wire with the inference log OFF, so an A10G delta here is the feature path plus the log. So an A10G delta is the 210-feature vectorization plus one request in five written to S3, measured against the same engine at the same rate. The Trainium rows have NO pair and report `--`: no campaign has measured this model size on Trainium. The round that measured this wire and log rate did so at 10M and 20M, and it was genuinely FP32, so it is comparable in kind but has no cell at this size. `graph` is carried for every row including the unpaired ones, because a device time that did not move is what says a latency change came from the host.

## Where the median goes

| accel | precision | rows x TPS | p50 | graph | vectorize | to tensor | remainder | buckets | vectorize share |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | 10.97 | 9.817 | 0.249 | 0.000 | 0.904 | 15 | 2.3% |
| A10G | BF16 | 100 x 200 | 5.85 | 4.685 | 0.236 | 0.000 | 0.929 | 15 | 4.0% |
| A10G | FP32 | 500 x 20 | 44.00 | 40.757 | 1.112 | 0.000 | 2.131 | 15 | 2.5% |
| A10G | BF16 | 500 x 40 | 22.10 | 18.311 | 1.106 | 0.000 | 2.683 | 15 | 5.0% |
| trn1 | BF16 | 100 x 70 | 16.46 | 15.457 | 0.229 | -- | 0.774 | 15 | 1.4% |
| trn1 | FP32 | 100 x 70 | 22.74 | 21.734 | 0.228 | -- | 0.778 | 15 | 1.0% |
| trn1 | BF16 | 100 x 100 | 16.44 | 15.473 | 0.225 | -- | 0.742 | 15 | 1.4% |
| trn1 | FP32 | 100 x 100 | -- | 21.927 | -- | -- | -- | 1 | -- |
| trn1 | BF16 | 100 x 200 | -- | 16.020 | -- | -- | -- | 1 | -- |
| trn1 | BF16 | 500 x 10 | 75.14 | 71.958 | 0.941 | -- | 2.241 | 15 | 1.3% |
| trn1 | FP32 | 500 x 10 | 117.22 | 113.650 | 0.976 | -- | 2.594 | 15 | 0.8% |
| trn1 | BF16 | 500 x 20 | 75.08 | 71.866 | 0.926 | -- | 2.288 | 15 | 1.2% |
| trn1 | FP32 | 500 x 20 | -- | 114.895 | -- | -- | -- | 1 | -- |
| trn1 | BF16 | 500 x 40 | -- | 73.104 | -- | -- | -- | 1 | -- |

`remainder` is the client median less the three server-side measures: gRPC, protobuf decode, scheduling, the log's own copy, and the network. `to tensor` is not published on the Neuron backend.

## Device and host

| accel | precision | rows x TPS | stack | device | device max | device MiB | power W | host CPU | host CPU max | host mem | client CPU |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | hushar | 93.3% | 95.0% | 1117 | 232.2 | 14.4% | 30.3% | 13.9% | 0.2% |
| A10G | BF16 | 100 x 200 | hushar | 89.1% | 90.0% | 713 | 208.5 | 15.6% | 33.0% | 14.0% | 0.3% |
| A10G | FP32 | 500 x 20 | hushar | 79.8% | 81.0% | 1271 | 247.0 | 11.9% | 22.5% | 14.0% | 0.2% |
| A10G | BF16 | 500 x 40 | hushar | 72.7% | 75.0% | 811 | 219.9 | 12.0% | 27.5% | 14.2% | 0.2% |
| A10G | FP32 | 100 x 100 | triton | 93.5% | 94.0% | 753 | 231.6 | 3.2% | 17.1% | 7.2% | 0.1% |
| A10G | BF16 | 100 x 200 | triton | 89.1% | 90.0% | 541 | 209.3 | 7.0% | 25.0% | 7.2% | 0.4% |
| A10G | FP32 | 500 x 20 | triton | 79.9% | 80.0% | 903 | 233.4 | 4.3% | 25.7% | 7.2% | 0.4% |
| A10G | BF16 | 500 x 40 | triton | 72.7% | 73.0% | 635 | 218.4 | 2.6% | 22.7% | 7.4% | 0.1% |
| trn1 | BF16 | 100 x 70 | hushar | 53.5% | 54.4% | 752 | -- | 14.8% | 31.1% | 2.6% | 0.2% |
| trn1 | FP32 | 100 x 70 | hushar | 75.4% | 75.6% | 1165 | -- | 20.2% | 36.5% | 3.8% | 0.2% |
| trn1 | BF16 | 100 x 100 | hushar | 76.4% | 76.6% | 752 | -- | 20.7% | 31.6% | 2.7% | 0.2% |
| trn1 | FP32 | 100 x 100 | hushar | 97.9% | 99.3% | 1165 | -- | 26.3% | 37.5% | 4.2% | 0.2% |
| trn1 | BF16 | 100 x 200 | hushar | 97.2% | 98.0% | 752 | -- | 26.8% | 46.6% | 3.3% | 0.3% |
| trn1 | BF16 | 500 x 10 | hushar | 35.8% | 35.8% | 912 | -- | 9.7% | 22.0% | 7.8% | 0.1% |
| trn1 | FP32 | 500 x 10 | hushar | 56.6% | 56.7% | 1445 | -- | 15.0% | 27.0% | 10.6% | 0.1% |
| trn1 | BF16 | 500 x 20 | hushar | 71.5% | 71.6% | 912 | -- | 19.2% | 36.6% | 7.9% | 0.2% |
| trn1 | FP32 | 500 x 20 | hushar | 97.2% | 100.0% | 1445 | -- | 26.0% | 45.5% | 11.1% | 0.2% |
| trn1 | BF16 | 500 x 40 | hushar | 97.7% | 100.0% | 912 | -- | 26.3% | 44.2% | 8.7% | 0.2% |

## Inference log

| accel | precision | rows x TPS | rate | requests per object | rows per object | objects | written GiB | predicted MB/s | measured MB/s | shed batches | write failures |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 x 100 | 1 in 5 | 500 | 50,000 | 48 | 2.5 | 2.2 | 2.2 | 0 | 0 |
| A10G | BF16 | 100 x 200 | 1 in 5 | 500 | 50,000 | 96 | 5.0 | 4.4 | 4.4 | 0 | 0 |
| A10G | FP32 | 500 x 20 | 1 in 5 | 100 | 50,000 | 48 | 2.4 | 2.1 | 2.1 | 0 | 0 |
| A10G | BF16 | 500 x 40 | 1 in 5 | 100 | 50,000 | 96 | 4.7 | 4.3 | 4.2 | 0 | 0 |
| trn1 | BF16 | 100 x 70 | 1 in 5 | 500 | 50,000 | 33 | 1.7 | 1.6 | 1.5 | 0 | 0 |
| trn1 | FP32 | 100 x 70 | 1 in 5 | 500 | 50,000 | 33 | 1.7 | 1.6 | 1.5 | 0 | 0 |
| trn1 | BF16 | 100 x 100 | 1 in 5 | 500 | 50,000 | 48 | 2.5 | 2.2 | 2.2 | 0 | 0 |
| trn1 | FP32 | 100 x 100 | 1 in 5 | 500 | 50,000 | 1 | 0.1 | 2.2 | 0.0 | 12300 | 42 |
| trn1 | BF16 | 100 x 200 | 1 in 5 | 500 | 50,000 | 0 | -- | 4.4 | -- | 19800 | 61 |
| trn1 | BF16 | 500 x 10 | 1 in 5 | 100 | 50,000 | 24 | 1.2 | 1.1 | 1.1 | 0 | 0 |
| trn1 | FP32 | 500 x 10 | 1 in 5 | 100 | 50,000 | 24 | 1.2 | 1.1 | 1.1 | 0 | 0 |
| trn1 | BF16 | 500 x 20 | 1 in 5 | 100 | 50,000 | 48 | 2.4 | 2.1 | 2.1 | 0 | 0 |
| trn1 | FP32 | 500 x 20 | 1 in 5 | 100 | 50,000 | 0 | -- | 2.1 | -- | 0 | 41 |
| trn1 | BF16 | 500 x 40 | 1 in 5 | 100 | 50,000 | 1 | 0.0 | 4.3 | 0.0 | 0 | 65 |

`predicted MB/s` is the offered rate times the sampled fraction times the measured feature-wire payload. `measured MB/s` is the bytes that reached the bucket over warm-up plus measured window, which is the span they were written across. A non-zero `shed batches` means the accumulator dropped its oldest buffered batch because every send permit was out, so that cell paid less than the full cost of logging; `write failures` are PUTs that did not complete.

## Rate held

| accel | precision | stack | rows x TPS | result | slices held | shed | failed | window s | rows/s offered | Gbps | % baseline |
|---|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| A10G | FP32 | hushar | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 902 | 10,000 | -- | -- |
| A10G | BF16 | hushar | 100 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 902 | 20,000 | -- | -- |
| A10G | FP32 | hushar | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 902 | 10,000 | -- | -- |
| A10G | BF16 | hushar | 500 x 40 | ok 40.0/40 | 15/15 | 0 | 0 | 902 | 20,000 | -- | -- |
| A10G | FP32 | triton | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 903 | 10,000 | -- | -- |
| A10G | BF16 | triton | 100 x 200 | ok 200.0/200 | 15/15 | 0 | 0 | 903 | 20,000 | -- | -- |
| A10G | FP32 | triton | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 905 | 10,000 | -- | -- |
| A10G | BF16 | triton | 500 x 40 | ok 40.0/40 | 15/15 | 0 | 0 | 905 | 20,000 | -- | -- |
| trn1 | BF16 | hushar | 100 x 70 | ok 70.0/70 | 15/15 | 0 | 0 | 900 | 7,000 | 0.08 | 2.4% |
| trn1 | FP32 | hushar | 100 x 70 | ok 70.0/70 | 15/15 | 0 | 0 | 900 | 7,000 | 0.08 | 2.4% |
| trn1 | BF16 | hushar | 100 x 100 | ok 100.0/100 | 15/15 | 0 | 0 | 900 | 10,000 | 0.11 | 3.5% |
| trn1 | FP32 | hushar | 100 x 100 | FAIL 90.9/100 | 0/15 | 10510 | 0 | 900 | 10,000 | 0.08 | 2.6% |
| trn1 | BF16 | hushar | 100 x 200 | FAIL 127.1/200 | 0/15 | 86721 | 0 | 900 | 20,000 | 0.11 | 3.7% |
| trn1 | BF16 | hushar | 500 x 10 | ok 10.0/10 | 15/15 | 0 | 0 | 900 | 5,000 | 0.05 | 1.7% |
| trn1 | FP32 | hushar | 500 x 10 | ok 10.0/10 | 15/15 | 0 | 0 | 900 | 5,000 | 0.05 | 1.7% |
| trn1 | BF16 | hushar | 500 x 20 | ok 20.0/20 | 15/15 | 0 | 0 | 900 | 10,000 | 0.10 | 3.3% |
| trn1 | FP32 | hushar | 500 x 20 | FAIL 17.4/20 | 0/15 | 3015 | 0 | 900 | 10,000 | 0.07 | 2.4% |
| trn1 | BF16 | hushar | 500 x 40 | FAIL 27.4/40 | 0/15 | 14971 | 0 | 900 | 20,000 | 0.12 | 3.8% |

A cell sustains only when every slice held the offered rate within 2% with nothing shed and nothing failed.

## Provenance

| accel | host | instance | cells | engine | TensorRT | Triton image |
|---|---|---|--:|---|---|---|
| A10G | g5-1 | g5.2xlarge | 4 | ONNX Runtime 1.28.3 | 10.16.1 | nvcr.io/nvidia/tritonserver:26.05-py3 |
| A10G | g5-2 | g5.2xlarge | 4 | ONNX Runtime 1.28.3 | 10.16.1 | nvcr.io/nvidia/tritonserver:26.05-py3 |
| trn1 | trn1-1 | trn1.2xlarge | 4 | libnrt 2.33.10.0 | -- | -- |
| trn1 | trn1-2 | trn1.2xlarge | 6 | libnrt 2.33.10.0 | -- | -- |

### Engines and NEFFs, primed before the grid

| accel | precision | shape | artifact | source | lane | when | build/restore s | reload proof s | MiB | parameters | trace error |
|---|---|--:|---|---|---|---|--:|--:|--:|--:|--:|
| A10G | FP32 | 100 | TensorRT engine | built | g5-1 | main | 31.1 | 22.1 | -- | -- | -- |
| A10G | FP32 | 500 | TensorRT engine | built | g5-1 | main | 32.4 | 22.2 | -- | -- | -- |
| A10G | BF16 | 100 | TensorRT engine | built | g5-2 | main | 55.3 | 22.1 | -- | -- | -- |
| A10G | BF16 | 500 | TensorRT engine | built | g5-2 | main | 62.5 | 22.2 | -- | -- | -- |
| trn1 | FP32 | 100 | NEFF | compiled | trn1-1 | main | 211.9 | 22.2 | 360.0 | 100,561,925 | 1.48e-06 |
| trn1 | FP32 | 500 | NEFF | compiled | trn1-1 | main | 625.4 | 22.3 | 384.5 | 100,561,925 | 2.13e-06 |
| trn1 | BF16 | 100 | NEFF | compiled | trn1-2 | main | 191.9 | 22.1 | 154.5 | 100,561,925 | 0.00232 |
| trn1 | BF16 | 500 | NEFF | compiled | trn1-2 | main | 442.3 | 22.3 | 172.2 | 100,561,925 | 0.00325 |

Each shape was built or restored and then served by a FRESH server, so the cells inherit a warm cache rather than paying for a build inside a discarded warm-up. `build/restore s` is a 20s trickle against a cold cache and `reload proof s` the same trickle against a fresh server that loaded it; the gap is what a cell would otherwise have paid. Every artifact here was BUILT -- no entry restored from the S3 cache, because no earlier campaign had compiled this model at either precision.

`trace error` is the compiled-versus-PyTorch maximum, published only on Trainium because the NEFF is compiled here and the ONNX graph is not. It is the arm's proof: `compile_neuron.py` declares a window per precision and refuses to write a NEFF whose error falls outside it. The A10G's `MiB` is blank because those two lanes were launched before the prime line learned to print its engine size; the sizes are in `Models` and come from the cells themselves.


| measure | count |
|---|--:|
| cells | 18 |
| sustained | 14 |
| hushar | 14 |
| triton | 4 |
| with CloudWatch | 14 |
| with device samples | 18 |
| cells that shed the log | 2 |
| log write failures | 209 |

Cells, server banners, generated service and model configurations, per-second host and device samples, client output and CloudWatch points are under `raw/`. `raw/results-gpu.json` and `raw/results-neuron.json` are this report's only inputs. `verify.py` re-derives every number above from them.
