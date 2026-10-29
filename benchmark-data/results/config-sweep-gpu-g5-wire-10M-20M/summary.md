# What the feature wire costs on an A10G

hushar serving the same graph, on the same host, at the same pinned parallelism, over two wires: its own 210 named features, and the single 690-float tensor that is Triton's contract. One variable. The sibling campaign [config-sweep-gpu-g5-10M-20M](../config-sweep-gpu-g5-10M-20M/summary.md) had shown the hushar-against-Triton gap was not the device -- identical TensorRT compute, transfer bounded at ~0.1 ms -- leaving ~1.2 ms of request-path overhead with two possible causes. This separates them.

## Answer

| model | rows x TPS | 210 features | 690-float tensor | switching to tensor | Triton, for reference |
|---|---:|---:|---:|---:|---:|
| 10M | 100 x 200 | 2.06 | 2.27 | +0.21 ms | 1.43 |
| 10M | 100 x 1000 | 2.55 | 2.53 | -0.02 ms | 1.64 |
| 20M | 100 x 200 | 3.16 | 3.17 | +0.01 ms | 2.39 |
| 20M | 100 x 750 | 4.44 | 4.73 | +0.29 ms | 3.36 |

- **The feature wire is a wash or slightly cheaper than the tensor wire.** Replacing 210 named features with the single 690-float tensor moves hushar's median by -0.02 to +0.29 ms. The hypothesis this run was built to test -- that decoding ~21,000 string-keyed values per request was the missing ~1.2 ms -- is **refused by the measurement**.
- **What actually happens is a trade, and it nets out at zero.** Dropping the features removes 0.22-0.24 ms of `vectorize`, exactly as it should. But `everything else` RISES by +0.18 to +0.42 ms, because the tensor is 2.54x the bytes -- 276,480 against 108,888 per request. hushar's 210-feature contract pays for its own decoding by putting less on the wire.
- **Server CPU agrees, independently.** It moves -1.4 to +0.2 percentage points between the two wires. Had parsing those values been a millisecond of CPU work per request, removing it would have shown up in the host sample. It does not.
- **So the overhead is hushar's request path, and it is present on both wires:** 1.04-1.86 ms that is neither the graph nor vectorization. On the tensor arm -- byte-identical to Triton, same graph, same device -- it is 1.47-1.86 ms, against roughly 0.6 ms of equivalent overhead measured for Triton in the sibling campaign. That is the comparison with nothing left to explain it away: admission, leasing a session, the blocking-pool hand-off, HTTP/2 framing and response encode.
- **Read 20M 100 x 750 differently: the GPU is saturated there (99.7%).** `graph` is measured around the whole backend call, so once the device is full it absorbs queueing for the device and stops being pure compute -- which is why it rises from ~1.68 ms at 200 TPS to ~2.7-2.9 ms here. The two wires are still comparable with each other; the number is just no longer a clean compute figure.
- **On identical bytes, hushar is 1.33x to 1.59x Triton's median.** The tensor arm and Triton receive the same 690-float payload and score the same engine on the same card, so this ratio is the two SERVERS and nothing else. (Cross-host: the Triton figures come from the sibling campaign, whose repeatability band was -10.8% to +2.7%.)
- 8 measured cells, 8 sustained, 0 failed the rate. Both wires held every rate offered, including 20M at 750 -- which the sibling campaign had bracketed only as "holds 600, fails 800".

## Where the time goes

| model | rows x TPS | wire | result | p50 | p95 | p99 | graph | vectorize | everything else | GPU | CPU | cell |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10M | 100 x 200 | 210 features | ok 200.0/200 | 2.06 | 2.45 | 2.56 | 0.785 | 0.232 | 1.043 | 14.0% | 5.5% | `g5-hushar-10m-r100t200-p2` |
| 10M | 100 x 200 | 690-float tensor | ok 200.0/200 | 2.27 | 2.67 | 2.82 | 0.790 | 0.013 | 1.467 | 14.0% | 5.2% | `g5-hushar-tensor-10m-r100t200-p2` |
| 10M | 100 x 1000 | 210 features | ok 1000.0/1000 | 2.55 | 3.01 | 3.25 | 0.840 | 0.248 | 1.462 | 67.3% | 24.3% | `g5-hushar-10m-r100t1000-p2` |
| 10M | 100 x 1000 | 690-float tensor | ok 1000.0/1000 | 2.53 | 3.01 | 3.26 | 0.872 | 0.014 | 1.644 | 66.5% | 22.9% | `g5-hushar-tensor-10m-r100t1000-p2` |
| 20M | 100 x 200 | 210 features | ok 200.0/200 | 3.16 | 3.53 | 3.67 | 1.680 | 0.235 | 1.245 | 32.0% | 7.7% | `g5-hushar-20m-r100t200-p4` |
| 20M | 100 x 200 | 690-float tensor | ok 200.0/200 | 3.17 | 3.58 | 3.73 | 1.680 | 0.014 | 1.476 | 31.9% | 7.4% | `g5-hushar-tensor-20m-r100t200-p4` |
| 20M | 100 x 750 | 210 features | ok 750.0/750 | 4.44 | 5.25 | 5.59 | 2.715 | 0.261 | 1.464 | 99.7% | 36.6% | `g5-hushar-20m-r100t750-p4` |
| 20M | 100 x 750 | 690-float tensor | ok 750.0/750 | 4.73 | 5.41 | 5.77 | 2.857 | 0.017 | 1.856 | 99.4% | 36.8% | `g5-hushar-tensor-20m-r100t750-p4` |

`graph` is hushar's `InferenceTime`, which brackets the whole backend call and so already contains the host-device transfer in both directions. `vectorize` is `VectorizationTime`. `everything else` is the remainder of the client median. On the tensor arm `vectorize` is the identity transformation over one feature, so it is near zero by construction -- the arm exists to empty that column and the decode cost behind it, not to make it cheaper.

## Method and limits

- **One variable.** Same graph bytes, same host, same TensorRT, same `mini_batch_size = rows = 100`, same sessions per size (10M 2, 20M 4). Only the wire differs. The parallelism is PINNED to what the sibling campaign's probe chose, and no probe ran here -- re-ranking configurations was not the question, and holding them fixed is what makes the subtraction legitimate.
- **Window.** 120s warm-up discarded, then 300s measured as one continuous load in 5 slices, a fresh server per cell, 60s idle between. Clients start on a minute boundary so the measured window is whole CloudWatch buckets.
- **The Triton column is cross-host**, quoted from [config-sweep-gpu-g5-10M-20M](../config-sweep-gpu-g5-10M-20M/summary.md) for orientation. That report measured host-to-host repeatability at -10.8% to +2.7%, so read the hushar-against-Triton ratios here as approximate and the feature-against-tensor ones -- which are within this host -- as exact.
- **FP32 only**, TensorRT default builder settings, one graph call per request at the engine's static `[100, 690]` profile.
- **Evidence.** `raw/results.json` is this report's input; cells, logs, per-second host and GPU samples and the CloudWatch points are under `raw/`. `verify.py` re-derives every number above from them.
