# What is in here

`results.json` is the only input `driver/report.py` reads. Everything else is the evidence
it was built from, or the scripts that produced it.

| path | what it is |
|---|---|
| `results.json` | every cell, enriched with CloudWatch, host and NeuronCore series. The report's sole input. |
| `cells/` | one JSON record per cell, written to S3 the moment that cell finished |
| `prime/` | the compiled artifacts per model size: SHA-256, NEFF bytes, parameter and layer count, compiled-versus-PyTorch error, and whether each NEFF was built here or restored from the S3 cache |
| `server/trn1-1/`, `server/trn1-2/` | per cell: the server log, its RESOLVED model and service configuration, per-second host samples (`s-*.log`) and per-second NeuronCore samples (`n-*.log`) |
| `client/trn1-1/`, `client/trn1-2/` | per cell: the load generator's output, its per-slice summary, and client host samples |
| `driver/` | the exact scripts that ran, both lane logs, and the report generator |
| `failures/` | empty. No cell failed the harness; 23 cells failed to hold their offered rate, which is a result and lives in `cells/`. |

## Decoding a cell id

```
trn1-20m-r100t750-mb50-s4
     │   │        │     └── replicas (sessions_per_model); 2 cores, so s4 is 2 per core
     │   │        └──────── mini_batch_size, so pieces = rows / mini_batch
     │   └───────────────── 100-row requests at 750 req/s offered
     └───────────────────── model size, 20M parameters
```

A `-ctx1` suffix marks a **control** cell: the same configuration with
`execution_contexts_per_replica = 1` instead of tracking the piece count. Four of the 40
cells carry it. No other cell departs from `contexts = pieces`.

## Deliberately not here

- **The NEFFs themselves.** Ten binaries, 32–70 MiB each. They are a pure function of the
  seeded architecture, the target parameter count and the batch size, so `prime/` records
  each SHA-256 and `scripts/benchmark/compile_neuron.py` rebuilds them. They are cached in
  S3 under `models/neuron/` for the account that ran this.
- **`verify.py`.** Every other run in this series ships one, and this run does not. The
  report is generated from `results.json` rather than hand-edited, so its arithmetic is
  reproducible by re-running `driver/report.py` — but nothing here independently
  re-derives the displayed figures from the raw cell records and sample logs the way
  `minibatch-gpu-g5-10M-20M/raw/driver/verify.py` does for that run. Treat the tables as
  generated-but-unverified.
- **Server-side timings for 10 cells.** All are 20M 500-row. The service logged the
  reason: `metrics: 4 sends still in flight, so 300 buffered records were dropped to keep
  the newest; the destination is not keeping up`. CloudWatch could not absorb the metric
  rate under the heaviest overload. Every affected cell is unsustained and so enters no
  latency comparison; their achieved rates and NeuronCore occupancy come from the client
  and the sampler, neither of which depends on CloudWatch.
- **`rig-state.json`.** Operational state, not evidence. The hosts it named are
  terminated.

## One thing to know before reading the numbers

**`device span` is only an execution time at one replica per core.** hushar's
`InferenceTime` brackets the whole provider call, so when two or three replicas share a
core, a replica waiting its turn is counted inside it — which is why the 20M mini-batch-50
span reads 3.267 ms at `s2`, 6.129 at `s4` and 9.846 at `s6` for the same work. The lane
arithmetic in the report therefore only closes on the `s2` rows, and it closes there to
within 0.1%.
