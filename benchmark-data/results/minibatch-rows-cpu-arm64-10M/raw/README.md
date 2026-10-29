# raw

Everything the hosts produced, as they produced it. `../summary.md` is derived from
`results.json`; nothing in here is edited.

| path | what |
|---|---|
| `results.json` | every cell, one object each, enriched with the CloudWatch percentiles and the sample means. The only file the report reads, and the one the verifier checks |
| `cells/` | 72 per-cell records, one JSON each, written to S3 the moment the cell finished |
| `server/` | each cell's server log, the service and model configuration it actually resolved, and its per-second host sample. Triton cells carry `config.pbtxt` and the container log instead |
| `client/` | each cell's load-generator output and its own host sample |
| `driver/` | the exact scripts that ran, plus the report generator and the verifier |
| `hostinfo.txt` | the four hosts' environment, captured before the rig was terminated |
| `failures/` | empty: no cell needed a second attempt |

## Reading a cell id

A cell id is `<host>-<arm>-<settings>`, and the settings are the configuration, so an id
can be read without a key:

```
c9g-1-R-l2000-r1t2000-spin1     host c9g-1, rows axis, 2000 rows/s offered as
                                1 row x 2000 req/s, spinning on
c9g-1-M-r100t100-mb4-spin0      host c9g-1, mini-batch arm, 100 rows x 100 req/s,
                                mini batch 4, spinning off
c9g-2-V-tensor-r50t200          host c9g-2, wire arm, tensor wire, 50 rows x 200 req/s
c9g-2-T-r50t200                 host c9g-2, Triton, 50 rows x 200 req/s
c9g-2-C-r100t400                host c9g-2, capacity ladder, 100 rows x 400 req/s
c9g-1-P-r100t100-rep2           host c9g-1, precision, repeat 2
```

`mbrows` means `mini_batch_size = rows`, which is one graph call per request and the
setting the Triton comparison uses. `spin1` is `allow_spinning: true`.

The arm letter is a driver detail and appears nowhere in the report, which describes every
cell by its settings instead. It decodes as:

| | question it answers | where it is reported |
|---|---|---|
| `R` | at the same rows per second, what does packaging change? | *The answer* |
| `C` | what can each packaging actually serve? | *What each packaging can serve* |
| `T` | Triton on the tensor wire | *Like for like* |
| `V` | hushar on both wires, at Triton's load | *Like for like*, *What the feature API costs* |
| `M` | the mini-batch split against the row count | *What hushar adds* |
| `P` | the same cell repeated, for the noise floor | *Precision* |

Arms `R` and `M` ran on `c9g-1`; `T`, `V` and `C` on `c9g-2`; `P` on both. An arm never
splits across hosts, because its table compares its cells with each other and the two
hosts carry a placement offset that repeating does not remove.

## What a cell records

`result` is the client's own windowed row: offered and achieved rate, shed and failed
counts, and p50/p95/p99. **A cell is only a result if `shed` and `failed` are both 0 and
achieved is within 2% of offered** — otherwise the server did not serve the rate it was
offered, and its latency describes whatever rate it managed. Those cells are kept and
marked `✗ <achieved>` in the report rather than dropped, because "this packaging cannot
hold this rate" is the finding the capacity ladder is looking for.

`wire` is `features` or `tensor` and is the one thing the two hushar arms differ by:
`features` sends 210 named features for the service to transform and concatenate,
`tensor` sends the finished 690-wide vector with an `identity` transformation. Both reach
the identical graph. The verifier checks the wire against the resolved model
configuration's `feature_order` rather than trusting the label.

`server_samples` and `client_samples` are means over the measured window's per-second
samples, so they exclude warm-up. The sampler writes six columns:

```
epoch  cpu_busy_%  process_RSS_%  process_RSS_kB  host_memory_%  tick_busy_%
```

`cpu_pct_sched` is the first, taken from the scheduler's own run-time counters, and
`cpu_pct_tick` is the last, from `/proc/stat`. Both are kept because they disagree on this
kernel: `hostinfo.txt` shows `CONFIG_NO_HZ_FULL=y` with
`CONFIG_VIRT_CPU_ACCOUNTING_GEN=y`, under which a busy CPU whose tick has stopped can be
accounted idle for a whole window. The report uses the scheduler's figure everywhere and
names the disagreement under *Confounds*. `memory_pct` is whole-host memory used including
page cache, which is the only memory figure comparable across the two stacks because the
sampler does not own Triton's container PID; process RSS is retained for audit and is not
compared.

`cloudwatch` holds the server's own timings for the cell's measured window:
`InferenceTime`, `VectorizationTime` and `ToTensorTime`, published one data point per
scored batch under a `ModelId` dimension naming the host and the wire.

**On a split cell these are a maximum, not a total.** The service reports one duration per
stage per request, and `scoring.rs` combines a split request's pieces with `max` on each
stage, because they run at once and the slowest one sets the request's span:

```rust
scored.timings.inference_time = scored.timings.inference_time.max(part.timings.inference_time);
```

So for a cell with `mini_batch < rows`, `InferenceTime` is the slowest piece's engine call
and `VectorizationTime` is that piece's share of the 210 features, neither of which is the
request's total. The report labels the column accordingly and prints no per-row engine
figure for a split cell; every per-row figure it does print comes from a cell with
`mini_batch = rows`, where the maximum is over a single piece and so is the request's own.

The percentiles are asked for in **minute** buckets clipped to the window, because
CloudWatch aligns a period to multiples of itself and a single large period does not mean
"this window". The mean is then exact; the percentiles are a sample-weighted mean of
per-minute percentiles, stored under a name that says so, with the buckets kept beside
them so the figure can be re-derived.

Every cell also carries `ort_version` read from the running server's own banner, the model
md5 checked on the host, the revision, the resolved settings, and the measured window as
timestamps. A cell whose banner disagreed with any of those was abandoned rather than
recorded; none did.

## Regenerating and checking

From this directory:

```bash
python driver/report.py      # rewrites ../summary.md from results.json
python driver/verify.py      # re-derives every figure and fails on disagreement
```

`verify.py` rebuilds the record set from `cells/` and the sample logs, asserts
`results.json` agrees with it field by field, checks each cell's printed settings against
the configuration the server actually resolved, re-runs `report.py` against its own
rebuild and requires the text to match `summary.md` byte for byte. 4,377 checks.

It cannot re-run a CloudWatch query from files, so the per-minute buckets are treated as
originals and their aggregates recomputed from them; `verify.py --cloudwatch` re-asks the
namespace as well, which needs credentials and works while the metrics are in retention.

`results.json` was collected with `driver/collect.py`, which needs `HUSHAR_BENCH_BUCKET`
set. Run it from the campaign's private driver directory or from here; it finds the
workspace root by looking for it rather than by counting directories, after an earlier
version wrote an entire collection into `benchmark-data/benchmark-data/`.

## Two things this run does not have

**The lane logs.** The rig was terminated for an outdated AMI before the orchestrators'
own logs were pulled off the client hosts, so there is no `orchestrator-*.log` here as
there is in the archived campaigns. What substitutes for them is stronger rather than
weaker: every cell carries its own timestamps and measured window, and `verify.py` checks
the measured set against the lane composition `driver/sweep.py` computes, so a lane that
had died early would show as missing cells rather than as a log nobody read. The driver
has since been left as it was; a future campaign should upload its lane log beside each
cell.

**Anything measured after 2026-10-23.** The hosts are gone, and a replacement rig is a
different kernel from the one in `hostinfo.txt`. Three questions were left open and are
listed at the end of `../summary.md`; cells answering them will be a separate environment
and a table mixing them has to say so.
