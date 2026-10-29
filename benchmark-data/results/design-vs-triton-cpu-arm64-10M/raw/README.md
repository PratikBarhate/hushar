# raw — design-vs-triton-cpu-arm64-10M

The design comparison: given the same traffic, which design serves it better, each at its
own best? `summary.md` beside this directory is the report.

This run closes the campaign. The two tuning passes that chose its configurations were
reported separately while the work was in flight; their reports are gone and their cells
live here under `tuning/`, so the claim "each at its own best" stays checkable from one
place.

## What is here

| path | what it is |
|---|---|
| `results.json` | every cell, enriched with CloudWatch percentiles and sample means. The only input `summary.md` reads |
| `cells/` | one original JSON record per cell, written by the host the moment it finished. **This is the record** |
| `failures/` | failed attempts and rungs skipped by the ladder's early stop, kept for diagnosis |
| `server/<host>/` | server banners, resolved configurations, Triton `config.pbtxt`, container logs, per-second host samples |
| `client/<host>/<cell>/` | load-generator output including the measured payload size, `meta.txt`, client host samples |
| `hostinfo/` | each host's kernel, AMI, CPU model and toolchain, captured by the deploy |
| `driver/` | the exact scripts that ran, both lane logs, the report generator and the verifier |
| `tuning/` | the two tuning passes' cells and results: `tune-*` at 100 rows, `tune200-*` at 200 rows |

## Cell IDs

`<host>-<arm>-r<rows>t<req/s>`, e.g. `c9g-1-H-r200t140` — arm `H`, 200 rows per request,
140 req/s, so 28,000 rows/s offered.

| arm | stack | wire | what it is |
|---|---|---|---|
| `H` | hushar | 210 named features | our design: server-side vectorization plus a mini-batch split |
| `X` | hushar | 690-float tensor | control: our server on Triton's contract |
| `T` | Triton | 690-float tensor | their design, at its tuned configuration |
| `P` | hushar | 210 named features | precision repeat, `-rep<n>` suffix |

`c9g-1` carried 100 and 200 rows; `c9g-2` carried 1 row and the precision repeats. Every
arm of a given row count is on **one host**, so no comparison in the report crosses
machines.

## Regenerating the report

```bash
cd driver
HUSHAR_BENCH_BUCKET=… python collect.py design-vs-triton-cpu-arm64-10M
python report.py                       # writes ../summary.md
python verify.py                       # 1013 checks
```

`summary.md` is generated and never hand-edited. `verify.py` reads the originals in
`cells/`, the per-second logs in `server/`, and the client logs in `client/`, then:

- re-derives every verdict from its three slices, rather than trusting the recorded one;
- re-derives CPU and memory from the per-second samples by column position;
- checks each cell's configuration against the banner the server actually printed —
  spinning policy, feature count, Triton's backend and model md5;
- checks every cell appears in a table with its own rows/req, req/s and p50;
- re-derives each arm's ceiling from the cells and checks the report names it;
- checks the answer table lists **only** rungs where both designs held;
- checks the payload figures come from the client logs rather than arithmetic;
- re-derives the noise floor and fails if it is too wide for the claims made.

Falsifying a single ceiling in `summary.md` makes it fail, which is how that last set was
tested.

## Read these before quoting a number

- **`held` is per slice.** Each cell is one continuous 900 s load reported as three 300 s
  slices, and holds only if **every** slice served its rate. This is the point of the long
  window: a five-minute measurement cannot distinguish a rate that is sustained from one
  that is sustained for five minutes. `result.shed` includes warmup, so it can exceed the
  slices' sum.
- **A ladder stops at its first unsustained rung**, because every higher rung would fail
  too. The skipped rungs are recorded in `failures/`, not silently absent.
- **`engine ms` is the server's own `InferenceTime`**, published per scored batch to
  CloudWatch, as a SampleCount-weighted mean of per-minute p50s clipped to the cell's
  window. Triton's server-side metrics are cumulative counters that cannot be differenced
  into a windowed quantile, so **every latency percentile in the report is the client's**,
  on both stacks.
- **CPU is the scheduler's run-time counter** (`/proc/schedstat`), not the tick-based
  `/proc/stat` figure, which on this tickless kernel can report a busy CPU as idle. Both
  are kept per cell.
- **Memory is whole-host, including page cache.** It is the only memory figure comparable
  across the two stacks, because the sampler does not own Triton's container PID. Do not
  read it as a working set: an earlier campaign recorded Triton at 17.8% on the same
  instance type purely because the 16.6 GB container image was still in page cache from a
  fresh pull. Here both stacks sit near 2%.

## Deliberately absent

- **A GPU, Trainium, x86, and any model but the 10M transformer.** One graph, one instance
  type, CPU only — the deploy refuses a runtime that ships an accelerator provider library.
- **A client-side measurement of what Triton's contract costs its caller.** The report
  bounds it from below with hushar's own `VectorizationTime` instead of building a client
  to lose the comparison. Measuring it properly is separate work.
- **Row counts between 1 and 100.** The payload crossing is estimated at ~3 rows from a
  two-point fit and validated against a held-out third point, not measured directly.
