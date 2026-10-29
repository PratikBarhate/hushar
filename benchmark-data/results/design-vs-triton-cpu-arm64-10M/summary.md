# design-vs-triton-cpu-arm64-10M

**Given the same traffic, which design serves it better — each at its own best?**

The traffic is fixed: *N* candidates, each described by 210 named features. What the two
designs ask **of the caller** differs, and that difference is the design. hushar takes the
named features and does the vectorization and a mini-batch split server-side. Triton's
contract has only one form, a finished 690-float tensor, so its caller must vectorize.

Three arms, because two cannot separate the wire from the server:

| arm | what it is |
|---|---|
| **hushar · 210 features** | our design: the caller sends 210 named features; the server transforms and concatenates them, then splits the request into mini batches scored at once |
| **hushar · tensor** | control: hushar on Triton's contract, so the gap to `hushar · 210 features` is what our feature API costs and the gap to Triton is the two servers on identical bytes |
| **Triton · tensor** | their design: the caller sends a finished 690-float tensor, Triton at its tuned configuration |

38 cells, 29 of which held their rate. Every cell is one continuous
900s load reported as 3 slices of 300s, and
holds only if **every** slice served its rate. Both stacks ran on **one host** per row
count, minutes apart, on the same graph and the same ONNX Runtime build.

## The answer

Every rung at which **both** designs held their rate. The ratio is only fair where neither
is shedding, and it moves with load, so all of them are shown rather than one.

| rows/req | req/s | rows/s | ours p50 | CPU | Triton p50 | CPU | ours is | on |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4000 | 4,000 | **1.04** | 96.0% | 1.10 | 95.3% | **1.1x faster** | 1.01x the CPU |
| 100 | 200 | 20,000 | **5.92** | 57.4% | 12.58 | 97.0% | **2.1x faster** | 0.59x the CPU |
| 100 | 240 | 24,000 | **6.98** | 70.0% | 13.07 | 98.1% | **1.9x faster** | 0.71x the CPU |
| 100 | 280 | 28,000 | **10.68** | 84.9% | 14.36 | 99.0% | **1.3x faster** | 0.86x the CPU |
| 200 | 100 | 20,000 | **8.67** | 59.3% | 24.10 | 84.3% | **2.8x faster** | 0.70x the CPU |
| 200 | 120 | 24,000 | **9.03** | 70.7% | 24.19 | 97.0% | **2.7x faster** | 0.73x the CPU |
| 200 | 140 | 28,000 | **13.33** | 82.9% | 24.68 | 98.8% | **1.9x faster** | 0.84x the CPU |

hushar is doing strictly more work in every row: the 210 transformations Triton requires
its caller to have done already.

## Seven insights

1. **At equal load our design is 1.1–2.8x faster on
   0.59–1.01x the CPU**, and the margin grows with
   candidates per request: at 20,000 rows/s it is
   2.1x at 100 rows, 2.8x at 200 rows.

2. **The margin shrinks as the host fills.** At 100 rows it is
   2.1x then 1.9x then 1.3x at 20k, 24k, 28k
   rows/s. The split buys latency by spending dispatch work, and near saturation there is
   less spare capacity to spend.

3. **Our contract is faster end to end *because* it is smaller — above about
   3 rows.** At 100 rows the 210 named features are
   2.5x fewer bytes than the tensor they concatenate into, and that
   saving exceeds the cost of the transformations: the feature wire beats the tensor wire
   on the same server. Vectorizing 100 rows costs 0.033 ms. At 1 row it
   inverts — see *What our contract costs*.

4. **The engine is identical, so every difference is outside the graph.** At 100 rows both
   hushar arms report 4.66 and 4.64 ms of `InferenceTime`. The
   control arm is doing its job.

5. **Sustained ceilings are mixed, not a sweep.** At 1 row we sustain
   8,000 rows/s against Triton's 4,000; at 100 rows both
   reach 28,000; at 200 rows Triton reaches 32,000 and
   we stop at 28,000. See *Ceilings* — the last of those is a difference
   in what each server does when it cannot keep up, not only in how much it can serve.

6. **Triton's tail degrades long before it sheds.** At 100 rows and
   28,000 rows/s its p95 is 13x its own p50 while ours is
   1.6x. It holds the rate by queueing; we hold it by serving.

7. **One row per request is where our design pays least.** Nothing to split, nothing to
   amortize 210 transformations over, and a payload that is
   2.4x *larger* than the tensor. At 4,000 rows/s the
   medians are within 6%
   (1.04 against 1.10 ms), though Triton's p95
   there is 6.1x ours.

## Per row count

Every rung of every ladder, including the ones that failed. A ladder stops at its first
unsustained rung, because every higher rung would fail too.

### 1 row per request

| arm | rows/req | req/s | rows/s | held | p50 ms | p95 ms | CPU | Mem | shed | engine ms |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|
| hushar · 210 features | 1 | 4000 | 4,000 | held | 1.04 | 2.19 | 96.0% | 1.8% | 0 | 0.36 |
| hushar · tensor | 1 | 4000 | 4,000 | held | 0.98 | 1.99 | 95.0% | 1.8% | 0 | 0.38 |
| Triton · tensor | 1 | 4000 | 4,000 | held | 1.10 | 13.43 | 95.3% | 1.8% | 0 | – |
| hushar · 210 features | 1 | 6000 | 6,000 | held | 1.35 | 2.78 | 98.5% | 1.8% | 0 | 0.37 |
| hushar · tensor | 1 | 6000 | 6,000 | held | 1.07 | 2.01 | 97.6% | 1.8% | 0 | 0.37 |
| Triton · tensor | 1 | 6000 | 6,000 | **✗ 4836/6000** | 4848.79 | 5125.64 | 87.0% | 2.5% | 186,756 | – |
| hushar · 210 features | 1 | 8000 | 8,000 | held | 4.29 | 31.31 | 99.6% | 1.9% | 0 | 0.37 |
| hushar · tensor | 1 | 8000 | 8,000 | held | 1.39 | 10.65 | 99.2% | 1.8% | 0 | 0.39 |
| hushar · 210 features | 1 | 10000 | 10,000 | **✗ 8440/10000** | 4735.01 | 4912.24 | 99.6% | 2.0% | 1,506,440 | 0.38 |
| hushar · tensor | 1 | 10000 | 10,000 | **✗ 9415/10000** | 4229.40 | 4385.59 | 99.7% | 1.9% | 528,536 | 0.39 |

### 100 rows per request

| arm | rows/req | req/s | rows/s | held | p50 ms | p95 ms | CPU | Mem | shed | engine ms |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|
| hushar · 210 features | 100 | 200 | 20,000 | held | 5.92 | 8.31 | 57.4% | 1.9% | 0 | 4.66 |
| hushar · tensor | 100 | 200 | 20,000 | held | 6.21 | 8.16 | 57.3% | 1.9% | 0 | 4.64 |
| Triton · tensor | 100 | 200 | 20,000 | held | 12.58 | 15.33 | 97.0% | 2.0% | 0 | – |
| hushar · 210 features | 100 | 240 | 24,000 | held | 6.98 | 10.11 | 70.0% | 1.9% | 0 | 5.11 |
| hushar · tensor | 100 | 240 | 24,000 | held | 7.23 | 10.20 | 69.9% | 1.9% | 0 | 5.01 |
| Triton · tensor | 100 | 240 | 24,000 | held | 13.07 | 18.24 | 98.1% | 2.0% | 0 | – |
| hushar · 210 features | 100 | 280 | 28,000 | held | 10.68 | 16.74 | 84.9% | 1.9% | 0 | 7.51 |
| hushar · tensor | 100 | 280 | 28,000 | held | 11.22 | 17.77 | 84.9% | 1.9% | 0 | 7.53 |
| Triton · tensor | 100 | 280 | 28,000 | held | 14.36 | 191.93 | 99.0% | 2.1% | 0 | – |
| hushar · 210 features | 100 | 320 | 32,000 | **✗ 292/320** | 4375.03 | 4405.14 | 92.3% | 2.1% | 26,935 | 11.62 |
| hushar · tensor | 100 | 320 | 32,000 | **✗ 293/320** | 4357.94 | 4378.35 | 92.2% | 2.3% | 25,893 | 11.55 |
| Triton · tensor | 100 | 320 | 32,000 | **✗ 311/320** | 4090.47 | 4297.79 | 99.9% | 2.4% | 8,341 | – |

### 200 rows per request

| arm | rows/req | req/s | rows/s | held | p50 ms | p95 ms | CPU | Mem | shed | engine ms |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|
| hushar · 210 features | 200 | 100 | 20,000 | held | 8.67 | 10.36 | 59.3% | 2.0% | 0 | 6.35 |
| hushar · tensor | 200 | 100 | 20,000 | held | 9.23 | 11.15 | 58.9% | 2.0% | 0 | 6.40 |
| Triton · tensor | 200 | 100 | 20,000 | held | 24.10 | 31.95 | 84.3% | 2.2% | 0 | – |
| hushar · 210 features | 200 | 120 | 24,000 | held | 9.03 | 15.06 | 70.7% | 1.9% | 0 | 6.54 |
| hushar · tensor | 200 | 120 | 24,000 | held | 9.43 | 15.49 | 70.4% | 1.9% | 0 | 6.55 |
| Triton · tensor | 200 | 120 | 24,000 | held | 24.19 | 31.00 | 97.0% | 2.2% | 0 | – |
| hushar · 210 features | 200 | 140 | 28,000 | held | 13.33 | 17.38 | 82.9% | 2.0% | 0 | 8.42 |
| hushar · tensor | 200 | 140 | 28,000 | held | 13.48 | 17.33 | 82.5% | 2.0% | 0 | 7.99 |
| Triton · tensor | 200 | 140 | 28,000 | held | 24.68 | 57.71 | 98.8% | 2.2% | 0 | – |
| hushar · 210 features | 200 | 160 | 32,000 | **✗ 155/160** | 4109.11 | 4141.40 | 97.8% | 2.4% | 4,153 | – |
| hushar · tensor | 200 | 160 | 32,000 | **✗ 156/160** | 4099.49 | 4131.79 | 97.7% | 2.7% | 3,810 | 19.18 |
| Triton · tensor | 200 | 160 | 32,000 | held | 98.14 | 645.98 | 99.7% | 2.3% | 0 | – |
| Triton · tensor | 200 | 180 | 36,000 | **✗ 160/180** | 4483.11 | 4742.70 | 99.9% | 2.6% | 18,715 | – |

## What our contract costs — and what it saves

The same server, the same graph, the same load, reached two ways: 210 named features that
hushar transforms and concatenates, against the finished 690-wide vector. `our contract` is
the feature wire against the tensor wire, so **negative is our contract winning**.

| rows/req | rows/s | our payload | Triton's | theirs ÷ ours | ours p50 | tensor p50 | our contract | vectorize |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4,000 | 6,788 B | 2,801 B | 0.41x | 1.04 | 0.98 | **+6.1%** | 0.012 ms |
| 1 | 6,000 | 6,788 B | 2,801 B | 0.41x | 1.35 | 1.07 | **+26.2%** | 0.012 ms |
| 1 | 8,000 | 6,788 B | 2,801 B | 0.41x | 4.29 | 1.39 | **+208.6%** | 0.010 ms |
| 100 | 20,000 | 108,888 B | 276,828 B | 2.54x | 5.92 | 6.21 | **-4.7%** | 0.033 ms |
| 100 | 24,000 | 108,888 B | 276,828 B | 2.54x | 6.98 | 7.23 | **-3.5%** | 0.033 ms |
| 100 | 28,000 | 108,888 B | 276,828 B | 2.54x | 10.68 | 11.22 | **-4.8%** | 0.034 ms |
| 200 | 20,000 | 211,288 B | 553,728 B | 2.62x | 8.67 | 9.23 | **-6.1%** | 0.037 ms |
| 200 | 24,000 | 211,288 B | 553,728 B | 2.62x | 9.03 | 9.43 | **-4.2%** | 0.038 ms |
| 200 | 28,000 | 211,288 B | 553,728 B | 2.62x | 13.33 | 13.48 | **-1.1%** | 0.043 ms |

From 100 rows up the feature wire is **faster** than the tensor wire on the same server,
despite performing 210 transformations the tensor wire skips. The payload columns say why.
Fitting each wire's fixed and per-row cost on the 1-row and 100-row measurements, which
bracket the crossing:

| wire | fixed per request | per row | crossing at |
|---|--:|--:|--:|
| 210 named features | 5,757 B | 1,031 B | **≈3 rows** |
| 690-float tensor | 33 B | 2,768 B | |

The named form carries 210 field headers whether the request has one row or two hundred, so
it starts **2.4x larger** at a single row and crosses below the tensor
at about 3 rows, after which it is 2.5x smaller — because
most of the 210 features are one scalar each and the 690 width only exists *after* the
transformations. The tensor form is almost pure payload: 2,768 B per row over a
fixed cost of 33 B.

Those two lines were fitted without using the 200-row measurement, so it is a held-out
check on them: they predict 212,019 B and 553,623 B against the measured
211,288 B and 553,728 B — 0.3% and 0.0% out.

That is the whole shape of the result. At 1 row our contract is both bigger on the wire and
doing more work, and it loses. From a few rows up it is smaller on the wire and still doing
more work, and it wins — the wire saving is larger than the transformation cost, which
`vectorize` shows is tens of microseconds per request.

This also prices what Triton's contract costs **its** caller, and the price is not zero.
The transformations do not disappear; they move to the caller, usually per candidate rather
than per batch. The `vectorize` column is that work done in batched Rust and is therefore a
floor on it, not an estimate of it.

## Ceilings

| rows/req | ours | Triton | at Triton's top rung |
|--:|--:|--:|---|
| 1 | 8,000 | 4,000 | p50 1.10 ms, p95 13.43 ms, CPU 95.3% |
| 100 | 28,000 | 28,000 | p50 14.36 ms, p95 191.93 ms, CPU 99.0% |
| 200 | 28,000 | 32,000 | p50 98.14 ms, p95 645.98 ms, CPU 99.7% |

At 1 row we sustain **2x** Triton's rate. At
100 rows the two are identical. At 200 rows Triton reaches one rung higher, and that rung
is worth reading closely rather than conceding or dismissing:

- Triton at 32,000 rows/s serves every request, at a p50 of
  98.14 ms and a p95 of 645.98 ms
  — against 24.68 and
  57.71 ms one rung below. It absorbed the load into a queue.
- Ours at the same rate shed 4,153 requests and its p50 went to
  4109.11 ms. It fell over.

Both are real. Triton degrades more gracefully than we do past saturation, and that is a
genuine advantage of its design at that point. Equally, under any tail-latency objective
below about 646 ms, Triton's 200-row ceiling is
28,000 rows/s — the same as ours. Which number is the ceiling depends on whether
a queue of 645.98 ms counts as service.

## Precision

One configuration repeated, changing nothing but the clock: our design at 100 rows and
10,000 rows/s.

| repeat | p50 ms | p95 ms | CPU |
|--:|--:|--:|--:|
| 1 | 5.47 | 6.45 | 28.5% |
| 2 | 5.43 | 6.24 | 28.5% |
| 3 | 5.43 | 6.41 | 28.4% |

Spread **0.7%** across three repeats, so every gap above about 1% in this report is
larger than the run's own repeatability. That is tight because the measured configuration
is a single operating point held for fifteen minutes, not because the server is quiet: it
is the same split configuration the tables above use.

## How it was measured

|  |  |
|---|---|
| cells | 38, 29 held their rate; ladders stop at the first unsustained rung |
| measured | 2026-10-24T00:55:19Z to 2026-10-24T08:29:01Z |
| window | 120s warmup discarded, 900s continuous, reported as 3 x 300s slices; a cell holds only if every slice did; server restarted between every cell |
| servers | c9g.16xlarge, 64 vCPU, arm64, us-west-2a. Each row count's arms are all on one host, so no comparison crosses machines |
| clients | m9g.8xlarge, one per server; never above 3.1% busy, so the generator is not the measurement |
| engine | ONNX Runtime 1.28.0 on both stacks, official CPU-only build; no accelerator provider library present on any host |
| Triton | `nvcr.io/nvidia/tritonserver:26.08-py3`, arm64, C++ onnxruntime backend, `--disable-auto-complete-config` |
| model | transformer encoder, 10 M parameters, `bench_model.onnx`, md5 `5539c97131c896db6c58088d9f9b9ee3`, checked on every host before it was allowed to measure |
| configurations | hushar `4x16 mb4 spin0` at 100 and 200 rows, `4x16 spin1` at 1 row where there is nothing to split; Triton `4x16` intra-op, `max_batch_size: 0`. All chosen by a 64-cell tuning pass over both row counts and two load points — see *How the configurations were chosen* |
| CPU | whole-host busy time from the scheduler's run-time counters (`/proc/schedstat`), not the tick-based figure, which this kernel can report as idle while busy |
| metrics | `engine ms` is the server's own `InferenceTime`, published per scored batch to CloudWatch; latency percentiles are the client's on both stacks, because Triton's server-side metrics are cumulative counters |
| revision | `ffba569-dirty` |

### How the configurations were chosen

64 cells, both row counts, two load points (10,000 and 24,000 rows/s), one stack per host,
five-minute windows. Evidence in `raw/tuning/`. Three results decided this run's pins:

- **hushar's mini-batch split needs its threads asleep.** At 100 rows and 24,000 rows/s
  `mb4 spin0` held 240/240 req/s on 69.8% of the host while `mb4 spin1` served 160.8 and
  shed 32,172 at 98.9%. The split's parallelism comes from the fan-out; a spinning pool
  competes with it for the same cores.
- **Triton's dynamic batcher does not help this workload.** At both row counts it lands
  inside a 5% tie band of no batching at all, and costs tail latency. A request already 100
  rows wide gains nothing from being merged with another. Parallel execution mode collapses
  outright — the graph is one chain.
- **The same configuration wins at both row counts**, so one pin per stack is defensible.
