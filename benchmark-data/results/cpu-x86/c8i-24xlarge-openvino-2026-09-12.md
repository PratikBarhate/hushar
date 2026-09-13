# c8i.24xlarge · openvino · 2026-09-12

**Model** — transformer encoder, 10,503,429 parameters, `raw` feature path (212 features per
row sent verbatim), **one** model resident as `bench-raw`. Byte-identical weights to the
[Graviton run](../cpu-arm64/c9g-24xlarge-xnnpack-2026-09-12.md) (md5 `aa119c1b…`), so the two
reports compare hardware rather than models.

**Question** — does OpenVINO beat the default CPU provider on a Xeon?

Two threadings, each run on both providers:

| Conf | mini batch | inf | intra (CPU) | intra (OpenVINO) | sess | requests per session |
|---|---|---|---|---|---|---|
| **A** | 5 | 16 | 1 | 1 · `threads=1:streams=1` | 1 | 16, ten engine calls each |
| **B** | 50 | 8 | 11 | 1 · `threads=11:streams=1` | 8 | **1** |

Conf-B is the balanced case: `inference_concurrency = sessions_per_model` puts exactly one
inference on a session, and `8 × 11 = 88` intra-op threads land on the 88 compute cores. It is
the only arm the startup banner passes with no warning note.

> **Headline.** **OpenVINO is genuinely faster per call and hard-capped at ~115 req/s.** Its
> engine call is **8.40 ms** against the CPU provider's 12.78 ms on conf-B — a real 1.5× win
> from Intel's kernels — but its server-side p95 is **101 ms** beside that 8.40 ms p50, and it
> sheds above 115 req/s where the CPU provider serves 400. On conf-A it collapses outright, at
> **19.2 req/s and 1.4 % CPU**. The CPU provider on conf-A sustains 400 req/s at **11.18 ms**,
> the best result measured on either machine.

## Sustained results

| Instance | Model | Params | Provider | Parallelisation | rows | TPS | p50 | p95 | srv p50 | srv p95 | CPU % | Mem % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | 100 | 9.09 | 10.09 | 4.63 | 7.12 | 9.9 | 0.4 |
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | 200 | 9.39 | 11.48 | 4.63 | 5.69 | 11.3 | 0.4 |
| **c8i.24xlarge** | transformer / raw | 10M | **onnxruntime/CPU** | conn 512 · async 16 · inf 16 · intra 1 · sess 1 · spin off · cores 1-7/8-95 · **mini batch 5** | 50 | **400** | **11.18** | 14.29 | **5.93** | 8.29 | 24.4 | 0.4 |
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 100 | 15.97 | 18.72 | 12.78 | 15.60 | 8.3 | 0.6 |
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 200 | 16.92 | 19.57 | 11.25 | 13.49 | 16.1 | 0.6 |
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/CPU | conn 512 · async 16 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 400 | 19.98 | 23.67 | 13.02 | 15.74 | 38.7 | 0.6 |
| c8i.24xlarge | transformer / raw | 10M | onnxruntime/OpenVINO(CPU, threads=11, streams=1) | conn 512 · async 16 · inf 8 · intra 1 · sess 8 · spin off · cores 1-7/8-95 · mini batch 50 | 50 | 100 | 13.72 | 14.82 | 8.40 | **101.34** | 10.3 | 1.4 |

Rows are **sustained only** — achieved ≈ offered with `shed = 0`, `failed = 0`. 90 s measured
per rate after a 20 s warm pass. Inference logs to S3 at 20 % sampling, `batch_size: 100`;
timings to CloudWatch, one data point per scored batch.

## Where each arm stops

| Provider | conf | highest sustained | at 400 offered | CPU % there |
|---|---|---|---|---|
| CPU | A | **400 req/s** @ 11.18 ms | 400.0 achieved | 24.4 |
| CPU | B | **400 req/s** @ 19.98 ms | 400.0 achieved | 38.7 |
| OpenVINO(11) | B | 100 req/s @ 13.72 ms | **115.3**, 25,619 shed | 9.9 |
| OpenVINO(1) | **A** | **none** | **32.5**, 33,073 shed | 1.3 |

## Conf-A on OpenVINO: the collapse

Conf-A is `mini batch 5 · sess 1`. Mini-batching cuts a 50-row request into ten 5-row batches
scored at the same time, so one request asks for **ten concurrent engine calls**, and
`sessions_per_model: 1` supplies one.

| offered | achieved | p50 | shed | CPU % | srv p50 |
|---|---|---|---|---|---|
| 100 | **19.2** | 26,695 ms | 7,276 | **1.4** | — |
| 200 | 23.6 | 53,520 ms | 15,873 | 1.4 | **1,219 ms** |
| 400 | 32.5 | 95,576 ms | 33,073 | 1.3 | 1,219 ms |

Identical in shape to XNNPACK's conf-A failure on the companion host. The server-side
`InferenceTime` of **1,219 ms** — which times the engine call alone, no admission queue, no
network — locates the wait inside the provider. The host sat at 1.3–1.4 % CPU while shedding a
third of the offer.

## The ceiling is global, not per-session

XNNPACK's ceiling on the companion host moves with `sessions_per_model`. OpenVINO's does not
move for anything:

| What changed | ceiling |
|---|---|
| conf-B baseline (`sess 8 · inf 8`) | **115.3 req/s** |
| `inference_concurrency` 8 → 16 | 110.0 req/s |
| sessions 8 → 16 | 107.5 req/s |
| physical cores 48 → 96 (`c8i.48xlarge`) | 109.2 req/s |

Read it as concurrency: `115.3 × 13.72 ms ≈ 1.6` inferences in flight. **OpenVINO admits
roughly 1.5 concurrent inferences however it is configured and whatever hardware it runs on** —
a process-wide serialisation rather than a per-session one. A per-session limit is
configurable; a global one is not.

The p95 is the early warning. Even at the sustained 100 req/s point, OpenVINO's server-side p95
is **101.34 ms** beside a p50 of 8.40 ms — a 12× spread, where the CPU provider's is 1.2×. The
arm was already queueing at the rate it passed.

**If OpenVINO is worth revisiting, the knob is `streams`.** Both arms ran `streams=1`, the
latency-oriented setting, and streams — not sessions, and demonstrably not
`inference_concurrency` — is OpenVINO's own concurrency mechanism.

*The three comparison rows above come from arms that are no longer tabulated in this report;
their artefacts remain under `raw/2026-09-12-accel/` and in S3.*

## OpenVINO's kernels really are faster

| Provider | srv p50 @ 100, conf-B | relative |
|---|---|---|
| onnxruntime/OpenVINO(CPU, threads=11) | **8.40 ms** | **1.52× faster** |
| onnxruntime/CPU (intra 11) | 12.78 ms | 1.00× |

Real, and unreachable in production above ~115 req/s.

## Mini batches beat a wide intra-op pool

| Conf | shape | p50 @ 400 | p95 | p99 | srv p50 | CPU % |
|---|---|---|---|---|---|---|
| **A** | ten 5-row batches, no pool | **11.18** | 14.29 | 16.48 | **5.93** | 24.4 |
| B | one 50-row batch, 11-thread pool | 19.98 | 23.67 | 26.33 | 13.02 | 38.7 |

**1.8× the latency for 14 points more CPU**, and the gap is wider here than on Graviton
because conf-B's 88 intra-op threads land on only **48 physical cores** on this host.

## Intel against Graviton: the winner flips with the threading

Both hosts are 96 vCPU "24xlarge", both given `cores 1-7/8-95`, both running the identical
model file — but not the same silicon:

| | c9g.24xlarge | c8i.24xlarge |
|---|---|---|
| physical cores | **96** | **48** (2 threads each) |
| `compute_cores 8-95` really means | 88 **physical** cores | 88 logical on **48 physical** |
| SIMD | asimd, sve, sve2 | avx2, avx512f, **amx_tile** |
| NUMA nodes | 1 | 2 |

p50 at 400 req/s, CPU provider:

| Conf | intra-op threads asked for | c9g (88 phys) | c8i (48 phys) | winner |
|---|---|---|---|---|
| A · mini batch 5 · sess 1 | few, mini-batch scoped | 12.21 | **11.18** | **Intel** by 8 % |
| B · mini batch 50 · sess 8 | 88 | **14.23** | 19.98 | **Graviton** by 29 % |

**The crossover is explained by physical cores.** Conf-A keeps few threads busy and rewards
per-core throughput, where `amx_tile` gives Intel a 27 % faster engine call (5.93 against
8.07 ms). Conf-B asks for 88 intra-op threads, which the Graviton places one-per-core and the
Xeon must fit onto 48 — a 1.8× oversubscription the Graviton does not pay.

**Choose the instance for the threading you intend to run:** wide intra-op pools want physical
cores and favour Graviton; narrow pools with mini-batching favour Intel's per-core speed.

## What this run says

**OpenVINO is not shippable for this model despite winning on kernel speed.** 1.5× faster per
engine call, a ~115 req/s ceiling that ignores sessions, `inference_concurrency` and a doubling
of the cores, and a p95 12× its p50 at the rate it passes.

**The two accelerators fail differently, and the difference is diagnostic.** XNNPACK's ceiling
is per-session and configurable; OpenVINO's is global and is not.

**The winner between Graviton and Intel depends on the threading.** Intel takes conf-A by 8 %,
Graviton takes conf-B by 29 %.

**Best overall: c8i.24xlarge, CPU provider, conf-A** — 400 req/s, 20,000 rows/s, 11.18 ms p50,
16.48 ms p99, 24.4 % CPU, nothing shed.

## Environment

Host: `c8i.24xlarge, Intel Xeon 6975P-C, 48 physical cores x86_64 (2 threads/core, 96 vCPU),
2 NUMA nodes (0-23,48-71 / 24-47,72-95), avx2+avx512f+amx_tile, 186 GiB,
Amazon Linux 2023.12.20260909, kernel 6.18.44, ONNX Runtime 1.29.0 built from source with
--use_openvino CPU against OpenVINO 2026.3.1, gcc 14.2.1`.

⚠️ `CPU %` here is a share of **96 logical CPUs on 48 physical cores**, while the Graviton
report's is a share of 96 logical CPUs on 96 physical cores. The two columns are not directly
comparable; the physical-core table above is the honest cross-machine view.

Client: `m9g.16xlarge, 90 s measured per rate after a 20 s warm pass`, client host CPU under
3 %.

OpenVINO is in no published ONNX Runtime archive; it was built on the instance with
`scripts/benchmark/build_ort.sh openvino`, and needs gcc 13+ and OpenVINO ≥ 2026.0. The build
finished inside 5 minutes on 96 vCPU. Every arm ran one binary built `--features cpu,openvino`,
and each arm's provider was asserted against the startup banner before its numbers were kept.

`srv` columns come from `raw/2026-09-12-accel/srv_latency.txt`, the frozen record of what
CloudWatch returned. Its extended statistics are **approximations**: re-querying the same past
window shifts a figure by a few percent, so these tables agree with that artefact rather than
with a fresh query. Treat `srv` as accurate to about ±0.5 ms.

Raw logs: `s3://hushar-bench-119928111785-us-west-2/runs/2026-09-12-accel/`
Inference logs: `s3://hushar-bench-119928111785-us-west-2/inference-logs/2026-09-12-accel/`
