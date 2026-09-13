# c9g.24xlarge · xnnpack · 2026-09-12

**Model** — transformer encoder, 10,503,429 parameters, `raw` feature path (212 features per
row sent verbatim, the graph does its own scaling and lookups), **one** model resident as
`bench-raw`.

**Question** — does XNNPACK beat the default CPU provider on Graviton?

Two threadings, each run on both providers:

| Conf | mini batch | inf | intra (CPU) | intra (XNNPACK) | sess | requests per session |
|---|---|---|---|---|---|---|
| **A** | 5 | 16 | 1 | 1 · `xnnpack:1` | 1 | 16, ten engine calls each |
| **B** | 50 | 8 | 11 | 1 · `xnnpack:11` | 8 | **1** |

Conf-B is the balanced case: `inference_concurrency = sessions_per_model` puts exactly one
inference on a session, and `8 × 11 = 88` intra-op threads land on the 88 compute cores, one
each. It is the only arm the startup banner passes with no warning note.

> **Headline.** **XNNPACK is a regression on both threadings and should not be pursued.** Its
> engine call costs **77.8 ms** against the CPU provider's **9.9 ms** at the same shape — 7.9×
> — and it caps at 118.6 req/s where the CPU provider serves 400 comfortably. On conf-A it
> collapses outright: **16.8 req/s at 1.3 % CPU** with 88 cores idle, because mini-batching
> asks for ten concurrent engine calls per request and one session supplies one.

## Sustained results

| Instance | Model | Params | Provider | Parallelisation | rows | TPS | p50 | p95 | srv p50 | srv p95 | CPU % | Mem % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | 100 | 11.49 | 12.48 | 7.51 | 8.15 | 10.2 | 0.3 |
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | 200 | 11.54 | 12.48 | 7.48 | 7.75 | 14.6 | 0.3 |
| **c9g.24xlarge** | transformer / raw | 10M | **onnxruntime/CPU** | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | **400** | **12.21** | 13.85 | **8.07** | 8.51 | 36.5 | 0.3 |
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 100 | 13.44 | 18.84 | 9.61 | 14.16 | 10.2 | 0.5 |
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 200 | 13.67 | 15.26 | 9.26 | 10.10 | 19.7 | 0.5 |
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 400 | 14.23 | 16.85 | 9.85 | 10.31 | 33.8 | 0.6 |
| c9g.24xlarge | transformer / raw | 10M | onnxruntime/XNNPACK(threads=11) | conn 512 · async 16 · inf 8 · intra 1 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 100 | 82.15 | 83.62 | 77.78 | 78.36 | 8.2 | 1.0 |

Rows are **sustained only** — achieved ≈ offered with `shed = 0`, `failed = 0`. 90 s measured
per rate after a 20 s warm pass at 400 req/s. Inference logs to S3 at 20 % sampling with
`batch_size: 100`; timings to CloudWatch as one data point per scored batch.

## Where each arm stops

| Provider | conf | highest sustained | at 400 offered | CPU % there |
|---|---|---|---|---|
| CPU | A | **400 req/s** @ 12.21 ms | 400.0 achieved | 36.5 |
| CPU | B | **400 req/s** @ 14.23 ms | 400.0 achieved | 33.8 |
| XNNPACK(11) | B | 100 req/s @ 82.15 ms | **118.6**, 25,330 shed | 8.4 |
| XNNPACK(1) | **A** | **none** | **30.1**, 33,287 shed | 1.2 |

## Conf-A on XNNPACK: the collapse, and why

Conf-A is `mini batch 5 · sess 1`. Mini-batching cuts a 50-row request into **ten 5-row
batches scored at the same time**, so one request asks for **ten concurrent engine calls** —
and `sessions_per_model: 1` supplies one pool to serve them.

| offered | achieved | p50 | shed | CPU % | srv p50 |
|---|---|---|---|---|---|
| 100 | **16.8** | 32,310 ms | 7,486 | **1.3** | **1,293 ms** |
| 200 | 21.3 | 64,642 ms | 16,087 | 1.2 | 1,293 ms |
| 400 | 30.1 | 106,312 ms | 33,287 | 1.2 | — |

`failed = 0`: nothing errored, the requests simply were not served. Latencies are
milliseconds, so 106,312 is 106 seconds.

The proof that the queue is **inside the provider** rather than in front of it is the
server-side figure. `InferenceTime` p50 of **1,293 ms** times the engine call alone, with no
admission queue and no network in it. A 5-row transformer batch does not take 1.3 seconds of
arithmetic on a Graviton core; it takes 1.3 seconds waiting for a pool it shares with nine
siblings.

> **The rule.** With an execution provider that brings its own thread-pool,
> `sessions_per_model` must be **≥ the engine calls you want in flight**, and mini-batching
> multiplies that requirement by the mini batches per request. Conf-A needs at least ten
> sessions on an accelerator; it has one.

The CPU provider survives the same configuration because ONNX Runtime's own intra-op path
serves concurrent calls on one session, which XNNPACK's pthreadpool does not.

## XNNPACK is 7.9× slower per call

Engine time alone, both providers on conf-B — the identical shape, one 50-row batch on an
11-thread pool:

| Provider | srv p50 @ 200 | relative |
|---|---|---|
| onnxruntime/CPU (intra 11) | **9.26 ms** | 1.00× |
| onnxruntime/XNNPACK (threads=11) | **77.66 ms** | **8.4× slower** |

Against the round's best configuration (CPU on conf-A, srv p50 7.48 ms) XNNPACK is **10.4×
slower**. Its cost is also nearly **flat in the offered rate** — 77.78 / 77.66 / 77.71 ms at
100 / 200 / 400 req/s — so the engine is not being contended, it is simply that expensive.

XNNPACK targets quantised, mobile-shaped convolutional graphs; an FP32 transformer encoder is
not its workload, and nodes it cannot take fall back to the CPU provider with layout
conversions at each boundary.

## Mini batches beat a wide intra-op pool

The two CPU arms differ only in how the 50 rows are cut:

| Conf | shape | p50 @ 400 | p95 | p99 | srv p50 | CPU % |
|---|---|---|---|---|---|---|
| **A** | ten 5-row batches, no pool | **12.21** | 13.85 | 15.53 | **8.07** | 36.5 |
| B | one 50-row batch, 11-thread pool | 14.23 | 16.85 | 19.44 | 9.85 | 33.8 |

Threads inside one operator stop helping well before the core count, because the operators are
a chain and each is a barrier; separate mini batches share no barrier at all. Conf-A is 14 %
faster end to end and 22 % faster in the engine, for 3 points more CPU.

Conf-B is the better choice only if you need an accelerator, since conf-A cannot be run on one
without at least ten sessions.

## What this run says

**XNNPACK is a regression for this model on Graviton.** 7.9× the engine time at matched shape,
a 118.6 req/s ceiling against 400, and a total collapse on the lower-latency threading.

**An accelerator's concurrency is bounded by `sessions_per_model`.** One session against ten
concurrent engine calls is a 24× throughput loss and leaves 88 cores idle at 1.2 % CPU.

**Best on this host: 400 req/s × 50 rows = 20,000 rows/s at 12.21 ms p50 and 36.5 % CPU**,
p99 15.53 ms, nothing shed — the CPU provider on conf-A.

## Environment

Host: `c9g.24xlarge, 96 physical cores aarch64 (1 thread/core), 1 NUMA node (0-95),
asimd+sve+sve2, 185 GiB, Amazon Linux 2023.12.20260909, kernel 6.18.44,
ONNX Runtime 1.29.0 built from source with --use_xnnpack`.
The server was confined to 95 of the 96 logical CPUs (async 1-7, compute 8-95), so
**CPU % reads directly as utilisation of the cores it was given**.

Client: `m9g.16xlarge, 90 s measured per rate after a 20 s warm pass, WARMUP=0 on the measured
windows`, client host CPU under 3 % throughout.

XNNPACK is in no published ONNX Runtime archive; it was built on the instance with
`scripts/benchmark/build_ort.sh xnnpack`, which took **2 m 48 s** on 96 cores. Every arm ran
one binary built `--features cpu,xnnpack`, and each arm's provider was asserted against the
startup banner before its numbers were kept.

`srv` columns come from `raw/2026-09-12-accel/srv_latency.txt`, the frozen record of what
CloudWatch returned. Its extended statistics are **approximations**: re-querying the same past
window shifts a figure by a few percent, so these tables agree with that artefact rather than
with a fresh query. Treat `srv` as accurate to about ±0.5 ms.

Companion report, same model and load on Intel:
[c8i.24xlarge · openvino](../cpu-x86/c8i-24xlarge-openvino-2026-09-12.md). The two providers
fail differently, which is the most useful thing the pair shows.

Raw logs: `s3://hushar-bench-119928111785-us-west-2/runs/2026-09-12-accel/`
Inference logs: `s3://hushar-bench-119928111785-us-west-2/inference-logs/2026-09-12-accel/`
