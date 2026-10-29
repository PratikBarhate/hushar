# design-vs-triton-cpu-arm64-20M

**Does hushar's design advantage survive a model twice the weight, and where is each stack's ceiling of sustained load?**

47 cells (12 failed, 3 partial, 32 sustained). Every cell offers one continuous 900s load after a 120s discarded warmup, reported as 3 consecutive slices, with the server stopped and restarted and 60s of settling between cells. `sustained` means **every** slice held its offered rate; a cell that held some is `partial`.

## Conclusions

1. **The design advantage holds at double the weight.** hushar's p50 is lower at every rung both stacks sustained: **1.1×** at 1 row(s), **3.3×** at 100 row(s), **3.3–3.4×** at 200 row(s) against a 1.0% noise floor.

2. **The advantage does not come from the wire — it comes from the split.** The prediction going in was that a heavier trunk amortizes the feature path better, so the margin would *narrow*. It did not. The graph's share of hushar's own latency rose to **93%** at 100 rows, leaving the feature path nearly irrelevant, and the margin still held. What scales is the mini-batch split: it parallelizes graph work, and doubling the graph doubles what it has to parallelize. The 10M control arm pointed the same way — hushar on Triton's own tensor wire was within 3.5–6.1% of hushar on its feature wire.

3. **Neither stack can have its ceiling and its latency at once.** At 100 rows Triton's latency-best configuration lands on *exactly* hushar's ceiling while running 1.9× slower; the only configuration that beats hushar's ceiling costs 3.3× the latency. At 200 rows that second configuration costs 6×. hushar faces the same trade at one row, where the two configurations bracket the same ceiling but behave very differently inside it: at 4,000 req/s `8x8` sheds 354,109 requests and `16x4` sheds 275, and the price of that is 3.72 ms against 2.96 at the rung below.

4. **Triton wins the one-row case, and that is the design's known worst case.** It sustains 5,500 requests/s against hushar's 3,500, because its dynamic batcher merges many tiny requests into one wide graph call — the same amortization hushar gets by splitting a 100-row request, and cannot get when there is nothing to split. Part of this is a baseline artifact rather than a model effect: the 10M campaign judged the batcher "a wash" from 100-row data and never tested it at one row, where it is decisive.

## The 2x model, and what exactly is doubled

Five encoder layers instead of two. Nothing else: same 210 features, same 690-float input, same thirteen ONNX operator types, same matmul shapes — so ONNX Runtime selects the same kernels and a 10M↔20M difference is **more of the same work** rather than different work.

| | 10M | 20M | ratio |
|---|--:|--:|--:|
| parameters | 9,138,693 | 18,589,701 | **2.03×** |
| MACs per row | 53,288,960 | 128,983,040 | **2.42×** |
| onnx bytes | 36,571,281 | 74,398,005 | 2.03× |
| input width | 690 | 690 | 1.00× |
| ONNX op types | 13 | 13 | identical |

The two ratios differ, and it is a property of the design rather than an error in the sizing. `to_tokens` is a `Linear(690 → 8×512)` applied **once per row**, so it costs 2,830,336 parameters at any model size. Holding the input width at 690 is mandatory — change it and the wire bytes change and nothing is comparable with the 10M campaign — which pins that layer as a constant, so every added parameter lands in the trunk. Its share falls from 31.0% to 15.2%.

The rejected alternative is worth naming: doubling the token axis (8 → 16, four layers) lands parameters at exactly 2.00× and keeps **every component's share identical**, but attention is quadratic in tokens, so it carries **3.90×** the arithmetic. The option that distributes weights perfectly is the one that distorts compute worst.

## What each design served, at the same traffic

### 1 row(s) per request

The caller sends 210 named features — **6,788 bytes per request**, 6,788.0 per row. Triton's contract has no such form, so its caller sends the finished 690-float tensor and does the 210 transformations itself.

| rows/req | req/s | rows/s | stack · configuration | state | p50 ms | p95 ms | p99 ms | CPU % | shed |
|--:|--:|--:|---|---|--:|--:|--:|--:|--:|
| 1 | 3,000 | 3,000 | hushar `8x8 mb4 spin0` | sustained | 2.85 | 3.38 | 3.65 | 51.0 | 0 |
| 1 | 3,000 | 3,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 3.12 | 8.49 | 12.85 | 97.2 | 0 |
| 1 | 3,000 | 3,000 | hushar `16x4 mb4 spin0` | sustained | 3.13 | 3.46 | 3.68 | 38.4 | 0 |
| 1 | 3,500 | 3,500 | hushar `8x8 mb4 spin0` | sustained | 2.96 | 3.67 | 4.08 | 58.9 | 0 |
| 1 | 3,500 | 3,500 | hushar `16x4 mb4 spin0` | sustained | 3.72 | 4.91 | 5.38 | 54.9 | 0 |
| 1 | 4,000 | 4,000 | hushar `8x8 mb4 spin0` | failed | 4,375.55 | 4,431.05 | 4,462.37 | 62.5 | 354,109 |
| 1 | 4,000 | 4,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 3.93 | 11.48 | 15.44 | 97.5 | 0 |
| 1 | 4,000 | 4,000 | hushar `16x4 mb4 spin0` | partial 2/3 | 601.09 | 3,603.82 | 3,980.49 | 74.6 | 275 |
| 1 | 5,000 | 5,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 5.28 | 16.56 | 23.28 | 97.4 | 0 |
| 1 | 5,000 | 5,000 | hushar `16x4 mb4 spin0` | failed | 4,725.18 | 4,754.01 | 4,772.00 | 76.8 | 744,157 |
| 1 | 5,500 | 5,500 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 6.14 | 19.64 | 27.79 | 97.3 | 0 |
| 1 | 6,000 | 6,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | partial 2/3 | 8.27 | 26.12 | 37.88 | 96.6 | 0 |
| 1 | 7,000 | 7,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | partial 1/3 | 12.58 | 36.53 | 52.82 | 95.4 | 0 |

| rows/s | hushar p50 | Triton p50 | **hushar is faster by** |
|--:|--:|--:|--:|
| 3,000 | 2.85 | 3.12 | **1.1×** |

### 100 row(s) per request

The caller sends 210 named features — **108,888 bytes per request**, 1,088.9 per row. Triton's contract has no such form, so its caller sends the finished 690-float tensor and does the 210 transformations itself.

| rows/req | req/s | rows/s | stack · configuration | state | p50 ms | p95 ms | p99 ms | CPU % | shed |
|--:|--:|--:|---|---|--:|--:|--:|--:|--:|
| 100 | 80 | 8,000 | hushar `4x16 mb8 spin0` | sustained | 15.49 | 16.16 | 18.36 | 49.2 | 0 |
| 100 | 80 | 8,000 | Triton `8x8 on intra, one call per request` | sustained | 51.05 | 52.67 | 56.85 | 60.1 | 0 |
| 100 | 80 | 8,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 29.23 | 35.48 | 42.43 | 78.8 | 0 |
| 100 | 100 | 10,000 | hushar `4x16 mb8 spin0` | sustained | 16.07 | 17.02 | 18.50 | 63.0 | 0 |
| 100 | 100 | 10,000 | Triton `8x8 on intra, one call per request` | sustained | 53.02 | 59.72 | 65.77 | 77.9 | 0 |
| 100 | 100 | 10,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 30.07 | 38.61 | 55.50 | 96.2 | 0 |
| 100 | 120 | 12,000 | hushar `4x16 mb8 spin0` | sustained | 16.26 | 18.41 | 23.94 | 77.0 | 0 |
| 100 | 120 | 12,000 | Triton `8x8 on intra, one call per request` | sustained | 53.01 | 60.15 | 68.67 | 92.9 | 0 |
| 100 | 120 | 12,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 30.48 | 35.84 | 70.51 | 99.1 | 0 |
| 100 | 130 | 13,000 | hushar `4x16 mb8 spin0` | failed | 4,065.26 | 4,086.66 | 4,096.92 | 89.9 | 2,016 |
| 100 | 130 | 13,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | failed | 4,741.41 | 5,346.09 | 5,750.11 | 96.9 | 24,418 |
| 100 | 140 | 14,000 | hushar `4x16 mb8 spin0` | failed | 4,352.48 | 4,374.80 | 4,400.46 | 90.0 | 11,262 |
| 100 | 140 | 14,000 | Triton `8x8 on intra, one call per request` | sustained | 54.75 | 146.12 | 210.86 | 99.0 | 0 |
| 100 | 140 | 14,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | failed | 4,990.81 | 5,452.89 | 5,628.32 | 97.9 | 30,134 |
| 100 | 150 | 15,000 | Triton `8x8 on intra, one call per request` | failed | 4,058.58 | 4,187.40 | 4,426.28 | 99.9 | 2,783 |
| 100 | 160 | 16,000 | Triton `8x8 on intra, one call per request` | failed | 4,310.11 | 4,364.26 | 4,554.75 | 100.0 | 10,920 |

| rows/s | hushar p50 | Triton p50 | **hushar is faster by** |
|--:|--:|--:|--:|
| 8,000 | 15.49 | 51.05 | **3.3×** |
| 10,000 | 16.07 | 53.02 | **3.3×** |
| 12,000 | 16.26 | 53.01 | **3.3×** |

### 200 row(s) per request

The caller sends 210 named features — **211,288 bytes per request**, 1,056.4 per row. Triton's contract has no such form, so its caller sends the finished 690-float tensor and does the 210 transformations itself.

| rows/req | req/s | rows/s | stack · configuration | state | p50 ms | p95 ms | p99 ms | CPU % | shed |
|--:|--:|--:|---|---|--:|--:|--:|--:|--:|
| 200 | 40 | 8,000 | hushar `8x8 mb4 spin0` | sustained | 17.30 | 21.65 | 24.76 | 52.9 | 0 |
| 200 | 40 | 8,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 57.08 | 64.32 | 71.06 | 67.0 | 0 |
| 200 | 40 | 8,000 | Triton `8x8 on intra, one call per request` | sustained | 100.83 | 102.84 | 107.47 | 54.9 | 0 |
| 200 | 50 | 10,000 | hushar `8x8 mb4 spin0` | sustained | 17.13 | 24.67 | 31.49 | 66.0 | 0 |
| 200 | 50 | 10,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 58.68 | 77.07 | 105.88 | 86.2 | 0 |
| 200 | 50 | 10,000 | Triton `8x8 on intra, one call per request` | sustained | 105.26 | 113.83 | 122.78 | 71.2 | 0 |
| 200 | 55 | 11,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | sustained | 58.90 | 74.00 | 131.03 | 93.6 | 0 |
| 200 | 60 | 12,000 | hushar `8x8 mb4 spin0` | sustained | 17.63 | 31.66 | 33.73 | 79.5 | 0 |
| 200 | 60 | 12,000 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | failed | 4,136.06 | 4,609.85 | 4,956.27 | 98.2 | 5,707 |
| 200 | 60 | 12,000 | Triton `8x8 on intra, one call per request` | sustained | 106.27 | 120.40 | 134.61 | 86.9 | 0 |
| 200 | 65 | 13,000 | hushar `8x8 mb4 spin0` | sustained | 22.60 | 32.23 | 37.38 | 85.7 | 0 |
| 200 | 70 | 14,000 | hushar `8x8 mb4 spin0` | failed | 4,178.89 | 4,208.53 | 4,236.22 | 99.6 | 3,075 |
| 200 | 70 | 14,000 | Triton `8x8 on intra, one call per request` | sustained | 108.88 | 230.27 | 307.71 | 99.0 | 0 |
| 200 | 75 | 15,000 | Triton `8x8 on intra, one call per request` | failed | 4,047.34 | 4,124.26 | 4,329.40 | 100.0 | 1,219 |
| 200 | 80 | 16,000 | Triton `8x8 on intra, one call per request` | failed | 4,306.66 | 4,365.59 | 4,567.72 | 99.9 | 5,536 |

| rows/s | hushar p50 | Triton p50 | **hushar is faster by** |
|--:|--:|--:|--:|
| 8,000 | 17.30 | 57.08 | **3.3×** |
| 10,000 | 17.13 | 58.68 | **3.4×** |

## The ceilings, bracketed then bisected

The ladder steps 2,000 rows/s at 100 and 200 rows and 1,000 at one row, then `PHASE=refine` measures the midpoint of whatever bracket it left — so a bracket that starts 2,000 wide closes to 1,000. Only measured cells bound a ceiling; a rung the early stop skipped is not evidence and is kept in `raw/failures/`.

| rows/req | stack · configuration | ceiling (rows/s) | bracket | p50 at its highest sustained rung |
|--:|---|---|--:|--:|
| 1 | hushar `8x8 mb4 spin0` | 3,500 – 4,000 | 500 | 2.96 |
| 1 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | 5,500 – 6,000 | 500 | 6.14 |
| 1 | hushar `16x4 mb4 spin0` | 3,500 – 4,000 | 500 | 3.72 |
| 100 | hushar `4x16 mb8 spin0` | 12,000 – 13,000 | 1,000 | 16.26 |
| 100 | Triton `8x8 on intra, one call per request` | 14,000 – 15,000 | 1,000 | 54.75 |
| 100 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | 12,000 – 13,000 | 1,000 | 30.48 |
| 200 | hushar `8x8 mb4 spin0` | 13,000 – 14,000 | 1,000 | 22.60 |
| 200 | Triton `4x16 on intra, dynamic batching, max 4000, delay 0us` | 11,000 – 12,000 | 1,000 | 58.90 |
| 200 | Triton `8x8 on intra, one call per request` | 14,000 – 15,000 | 1,000 | 108.88 |

## Neither stack can have its ceiling and its latency at once

A ranking rule picks one configuration per stack per row count — sustained first, then p95, then p50 — and at several row counts the configuration with the best latency and the one with the highest ceiling are **different configurations**. Reporting only one would either understate a stack's ceiling or overstate its latency, so a runner-up was measured for **both** stacks.

| rows/req | stack | configuration | ceiling (rows/s) | verdict at the rung that separated them | p50 there |
|--:|---|---|---|---|--:|
| 1 | hushar | `8x8 mb4 spin0` | 3,500 – 4,000 | failed at 4,000 rows/s | 4,375.55 |
| 1 | hushar | `16x4 mb4 spin0` | 3,500 – 4,000 | partial 2/3 at 4,000 rows/s | 601.09 |
| 100 | Triton | `8x8 on intra, one call per request` | 14,000 – 15,000 | sustained at 14,000 rows/s | 54.75 |
| 100 | Triton | `4x16 on intra, dynamic batching, max 4000, delay 0us` | 12,000 – 13,000 | failed at 14,000 rows/s | 4,990.81 |
| 200 | Triton | `4x16 on intra, dynamic batching, max 4000, delay 0us` | 11,000 – 12,000 | failed at 12,000 rows/s | 4,136.06 |
| 200 | Triton | `8x8 on intra, one call per request` | 14,000 – 15,000 | sustained at 12,000 rows/s | 106.27 |

The one-row pair is worth reading closely, because it is the clearest thing this run has to say about **why fifteen minutes**: `16x4` *sustained* 4,000 req/s over the five-minute tuning window and is only `partial` over fifteen, shedding 275 requests in its third slice while achieving 3,999.7 of 4,000. A five-minute window cannot tell a rate that is sustained from a rate that is sustained for five minutes.

## Against the 10M campaign

The two ladders do not overlap — 20,000–32,000 rows/s there against 6,000–16,000 here — because the model got 2.42× heavier, which is the point. So no rung-for-rung comparison is attempted. What is comparable is each stack **at its own lowest sustained rung** (each at comfort, not at its edge) and each stack's **ceiling**.

| rows/req | stack | 10M p50 @ its lowest sustained rung | 20M p50 @ its lowest sustained rung | latency cost of 2× the model |
|--:|---|--:|--:|--:|
| 1 | hushar | 1.04 @ 4,000 | 2.85 @ 3,000 | 2.7× |
| 1 | Triton | 1.10 @ 4,000 | 3.12 @ 3,000 | 2.8× |
| 100 | hushar | 5.92 @ 20,000 | 15.49 @ 8,000 | 2.6× |
| 100 | Triton | 12.58 @ 20,000 | 51.05 @ 8,000 | 4.1× |
| 200 | hushar | 8.67 @ 20,000 | 17.30 @ 8,000 | 2.0× |
| 200 | Triton | 24.10 @ 20,000 | 57.08 @ 8,000 | 2.4× |

| rows/req | stack | 10M ceiling | 20M ceiling | ceiling fell by |
|--:|---|---|---|--:|
| 1 | hushar | 8,000 – 10,000 | 3,500 – 4,000 | 2.40× |
| 1 | Triton | 4,000 – 6,000 | 5,500 – 6,000 | 0.87× |
| 100 | hushar | 28,000 – 32,000 | 12,000 – 13,000 | 2.40× |
| 100 | Triton | 28,000 – 32,000 | 14,000 – 15,000 | 2.07× |
| 200 | hushar | 28,000 – 32,000 | 13,000 – 14,000 | 2.22× |
| 200 | Triton | 32,000 – 36,000 | 11,000 – 12,000 | 2.96× |

hushar's ceilings fell by about the arithmetic the model added (2.42× MACs per row), which is what should happen to a compute-bound server and is a useful check that nothing else changed. The one figure that moved the *wrong* way for the baseline is Triton's one-row ceiling, which went **up** — not because the model helped it, but because this campaign tuned its dynamic batcher at one row and the 10M campaign never did.

## Where the time goes, from the server's own metrics

hushar publishes one data point per scored batch, so the split between turning a request into a tensor and scoring it is measured rather than inferred. Triton publishes no equivalent for work its caller does, because its caller does that work.

| rows/req | rows/s | end-to-end p50 | `InferenceTime` p50 | graph share | `VectorizationTime` p50 | per row |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3,000 | 2.85 | 2.110 | 74% | 0.016 | 16.00 µs |
| 1 | 3,500 | 2.96 | 2.090 | 71% | 0.016 | 16.00 µs |
| 100 | 8,000 | 15.49 | 14.482 | 93% | 0.033 | 0.33 µs |
| 100 | 10,000 | 16.07 | 14.514 | 90% | 0.033 | 0.33 µs |
| 100 | 12,000 | 16.26 | 14.953 | 92% | 0.034 | 0.34 µs |
| 200 | 8,000 | 17.30 | 15.459 | 89% | 0.036 | 0.18 µs |
| 200 | 10,000 | 17.13 | 15.582 | 91% | 0.038 | 0.19 µs |
| 200 | 12,000 | 17.63 | 15.984 | 91% | 0.037 | 0.18 µs |
| 200 | 13,000 | 22.60 | 17.759 | 79% | 0.038 | 0.19 µs |

`VectorizationTime` is also the **lower bound on what Triton's contract moves to the caller**: the same 210 transformations, done in batched Rust, which is the cheapest they can possibly be. A caller doing them per candidate in a managed language pays more. It is reported here rather than added to Triton's latency column, because charging Triton for a strawman client would be the easier claim and the weaker one.

## Noise floor

One configuration — hushar `4x16 mb8 spin0` at 8,000 rows/s — repeated 3 times with the server stopped and restarted between runs, changing nothing but the clock:

| repeat | p50 ms | p95 ms | CPU % |
|--:|--:|--:|--:|
| 1 | 15.32 | 16.02 | 49.3 |
| 2 | 15.32 | 16.03 | 49.4 |
| 3 | 15.47 | 16.10 | 49.3 |

Spread **1.0%** (15.32 to 15.47 ms). Any gap smaller than this is not a result.

## Provenance

| | |
|---|---|
| model | `bench_model.onnx`, 20m, md5 `30f9c5330b808a34b55fe88048a20da6` |
| ONNX Runtime | **1.28.0** on both stacks |
| build | `ffba569-dirty` |
| instance | c9g.16xlarge, 64 vCPU, arm64 |
| AMI / kernel | `ami-025d1d5c1b9b090aa` / `6.18.54-121.166.amzn2023.aarch64` |
| baseline | `design-vs-triton-cpu-arm64-10M` |

Both stacks must score the graph with the same engine or the comparison is between two engines, so it is enforced rather than intended: `deploy.sh` installs exactly that ONNX Runtime for hushar, `deploy_triton.sh` reads the version out of the container's `libonnxruntime.so` and **refuses to deploy** on a mismatch, every cell aborts unless the server's own banner reports it, and the model md5 is checked in four places.

Memory stayed at 1.9–8.7% of the host across every cell on both stacks. The 10M campaign reported a large memory advantage for hushar; that figure did not replicate and should not be carried forward — it was page cache from a freshly pulled 16.6 GB container image, not resident model memory.
