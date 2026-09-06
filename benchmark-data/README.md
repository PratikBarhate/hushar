# benchmark-data

Everything the benchmark reads and writes. The split is by **what wrote it**: a file a
script can reproduce is not committed; a file a human wrote is.

```
benchmark-data/
├── generated/                          ← ignored, written by the generator
│   ├── bench_model.onnx                    ~190 MB  one [rows, 502] float input
│   ├── bench_named_model.onnx              ~190 MB  12 named inputs, 2 of them text
│   ├── bench_model_config.json                      how features reach each model
│   ├── bench_named_model_config.json
│   ├── bench_service_config.json                    port and connection concurrency
│   └── bench_string_vocabulary.json                 the tokens the client sends
└── results/
    ├── nomenclature.md                 ← tracked, what every column means
    ├── cpu-m1-pro-2026-09-03.md        ← tracked, one write-up per run
    └── raw/                            ← ignored, the output a run left behind
```

Two rules follow from that, and they are the whole reason for the layout:

- **`generated/` is never committed.** ~800 MB of weights, and every byte of it is a
  function of one script and a fixed seed.
- **`results/raw/` is never committed, `results/*.md` always is.** The logs are the
  evidence for a report; the report is the thing worth reading a year later.

## Generating what the benchmark needs

Nothing in `generated/` is committed, so this is the first step on a fresh clone, not an
optional one. Both commands run from the repository root:

```bash
pip install -r scripts/requirements.txt
python scripts/generate_benchmark_models.py
```

It writes both models, both configurations, the service configuration and the vocabulary
**together**, from one source — because the model's input width and the configuration's
feature widths have to agree, and deriving them separately is how a service comes to load
a model and then refuse every request.

`--with-batch1` additionally writes copies with the row axis pinned to 1, each paired with
a configuration declaring `"fixed_batch_size": 1`. The service **serves** those — it just
splits a request into batches of one row, padding the last if it is short. The graph's
pinned dimension and the
declared number come from one variable in one loop iteration, so they cannot disagree, and
three checks enforce it anyway: when the model loads, on every request, and in the runner
against `ROWS`. What pinning is worth is measured in
[results/cpu-m1-pro-2026-09-04.md](results/cpu-m1-pro-2026-09-04.md): nothing, on CPU.

## Running a benchmark

```bash
cargo build --release
export ORT_DYLIB_PATH=…   # the ONNX Runtime library for your platform

TPS="50 100 200 300 400 500" scripts/run_benchmark.sh
```

The runner starts the server per model, sweeps the rates against that one loaded session,
prints a table, and leaves everything in `$RUN_DIR` (`/tmp/hushar-benchmark` by default):

```
$RUN_DIR/
├── summary.txt                      the printed table
├── <tag>.model.json                 the config actually served, provider substituted in
├── <tag>.service.json
├── logs/<tag>.server.log            "metrics:" lines -- the server's own timings
├── logs/<tag>.<rate>.client.log     one per rate, what the table row was reduced from
└── inference-logs/                  the logging path, exercised under load (~50 MB)
```

`MODELS`, `PROVIDERS`, `DURATION`, `WARMUP`, `ROWS`, `CONCURRENCY`, `PORT` and `RUN_DIR`
are the other knobs; all of them are environment variables with defaults.

## Keeping a result

Worth keeping means a report, not a log directory. Copy in the small text files, then
write the reasoning next to them:

```bash
mkdir -p benchmark-data/results/raw
cp /tmp/hushar-benchmark/summary.txt benchmark-data/results/raw/summary-cpu.txt
grep '^metrics:' /tmp/hushar-benchmark/logs/*.server.log \
  > benchmark-data/results/raw/metrics-cpu.txt
```

Only the `.md` file is committed. Name it `<provider>-<host>-<date>.md`, follow
[`cpu-m1-pro-2026-09-03.md`](results/cpu-m1-pro-2026-09-03.md) for the shape, and state
the question, the answer, the numbers and the caveats — a table with no host load, row
count or thread count beside it cannot be compared against anything.
