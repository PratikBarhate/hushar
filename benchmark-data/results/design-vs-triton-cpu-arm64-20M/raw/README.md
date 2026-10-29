# design-vs-triton-cpu-arm64-20M — the evidence

`../summary.md` is generated from `results.json` by `driver/report.py` and checked against
these originals by `driver/verify.py` (1,246 checks). Nothing in the summary is
hand-written, so re-running both reproduces it exactly.

```bash
cd raw/driver
python report.py && python verify.py
```

## What is here

| path | what it is |
|---|---|
| `results.json` | the 47 head-to-head cells, enriched with CloudWatch percentiles and per-second utilisation means. The only input `report.py` reads. |
| `cells/` | the 47 cell records **as the hosts wrote them**, one object each, written to S3 the moment each finished. The record. |
| `failures/` | 22 cells that were never measured — all of them rungs the ladder's early stop skipped once a lower rung broke. A skipped rung is not evidence and never bounds a ceiling. |
| `server/` | per-second host samples, one log per cell. Columns: `epoch cpu_busy_% process_RSS_% process_RSS_kB host_memory_% tick_%`. CPU and memory in the summary are re-derived from these by column position. |
| `client/` | each client's own logs, including the line reporting the **measured** serialized payload size. |
| `hostinfo/` | each host's AMI, kernel and CPU count, uploaded by the deploy before the first cell rather than captured at the end. |
| `driver/` | the exact driver that produced this, plus each lane's orchestrator log. |
| `tuning/` | the 80-cell tuning campaign that chose the configurations. "Each at its own best" is only checkable if the ranking is. |

## Decoding a cell id

```
c9g-1-H-r100t120
└─┬──┘ │ └─┬──┘
  host arm rows/request and requests/second
```

The summary deliberately never prints an arm letter — a letter is an index into the
driver's plan, which a reader does not have. It prints configurations. The letters, for
reading `cells/` directly:

| arm | stack | what it was |
|---|---|---|
| `H` | hushar | the design, at its tuned primary configuration |
| `T` | Triton | their design, at its tuned primary configuration |
| `A` | hushar | hushar's **runner-up**, measured only at one row |
| `L` | Triton | Triton's **runner-up**, measured at 100 and 200 rows |
| `P` | hushar | the same configuration repeated three times, for the noise floor |
| `U` | either | a tuning cell (in `tuning/cells/` only) |

Two runner-up arms exist because at several row counts a stack's latency-best
configuration and its highest-ceiling configuration are **different configurations**.
Measuring only one would have understated a ceiling or overstated a latency — in Triton's
favour at 100 rows and in hushar's favour at one row, so both were added.

## Reading a verdict

Each cell offers one continuous 900 s load, reported by both clients as three consecutive
300 s slices.

| state | meaning |
|---|---|
| `sustained` | **every** slice held the offered rate, zero shed, zero failed |
| `partial` | some slices held; `broke_in_slice` says which one did not |
| `failed` | no slice held |

A five-minute window cannot tell a rate that is sustained from a rate that is sustained for
five minutes, and this run contains the proof: hushar `16x4 mb4 spin0` **sustained** 4,000
req/s at one row over the five-minute tuning window and is only `partial` over fifteen,
shedding 275 requests in its third slice. Compare `tuning/cells/c9g-3-U-hushar-r1-mb4-s16x4-t4000.json`
against `cells/c9g-3-A-r1t4000.json`.

## Phases

| phase | what it did |
|---|---|
| `tune` | 80 cells, 5-minute windows, ranked each stack's configurations at two load points per row count. Four lanes in parallel. |
| `head` | 15-minute windows over a ladder, stopping at the first rung that did not hold. |
| `refine` | re-read the collected verdicts, found each ladder's highest sustained rung and lowest **measured** failure, and measured the midpoint — closing every bracket to 1,000 rows/s, and to 500 at one row. |

A cell records its own `phase`, so a refined rung is identifiable in the ladder it refines.

## Provenance, and what is enforced rather than intended

- graph `bench_model.onnx`, md5 `30f9c5330b808a34b55fe88048a20da6`, 74,398,005 bytes,
  18,589,701 parameters, input `features FP32[rows,690]`. Checked in four places: both
  deploy scripts, the Triton model load, and the per-lane preflight.
- **ONNX Runtime 1.28.0 on both stacks.** `deploy.sh` installs exactly that for hushar;
  `deploy_triton.sh` reads the version out of the container's `libonnxruntime.so` and
  **refuses to deploy** on a mismatch; every cell aborts unless the server's own banner
  reports it. `verify.py` re-checks the banner of all 47 cells.
- `c9g.16xlarge`, 64 vCPU, arm64, `ami-025d1d5c1b9b090aa` (AL2023, 3 days old at launch),
  kernel `6.18.54-121.166.amzn2023.aarch64`. The rig was launched for this run and
  **terminated** after it, so no instance survived to go stale.
- build `ffba569-dirty` — the working tree, so an uncommitted change is testable.

## One caveat worth carrying forward

Memory sat at 1.9–8.7% of the host across every cell on both stacks. The 10M campaign
reported a large memory advantage for hushar; **that figure did not replicate** and should
not be quoted. It was page cache from a freshly pulled 16.6 GB container image, not
resident model memory.
