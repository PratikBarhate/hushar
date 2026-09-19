# benchmark-data

What a benchmark reads and writes. A file a script can reproduce is not committed; a file a
human wrote is. One directory per benchmark, named for what it swept.

```
benchmark-data/
├── TEMPLATE.md                 ← this file: the layout, and the report to copy
├── generated/                  ← ignored, from generate_benchmark_models.py
├── dist/                       ← ignored, bundles from package.sh
└── results/
    └── <bench-name>/
        ├── summary.md          ← the report
        └── raw/                ← what the run left behind, tracked
```

`<bench-name>` says what the run swept — `config-sweep-cpu-12xlarge-50M` would be a
configuration sweep of a 50M-parameter model on CPU across 12xlarge hosts. A single host and
provider can name itself `c9g-24xlarge-cpu-2026-09-11`; anything spanning two hosts cannot, and
naming it for one of them would be a lie about its scope.

Generate the inputs on a fresh clone:

```bash
pip install -r scripts/requirements.txt
python scripts/generate_benchmark_models.py
```

The server host needs the `.onnx` files; the client only needs the configurations, the
vocabulary and the feature specification, and never opens the graph.

## raw/

Collect it **per cell, in the session that measured it** — a host's `/tmp` does not survive a
reboot, and collecting at the end is how output goes missing.

```
raw/
├── README.md                   ← what each file is, and anything that did not survive
├── driver/                     ← what drove the run, and the parsed result of every cell
├── server/<host>/              ← server logs with their banners, per-cell CPU samples
├── client/<host>/              ← per-cell summary.txt, meta.txt and client log
└── model-config.json           ← the model contract every cell was served with
```

Keep one copy in S3 as well, under a prefix named for the benchmark. Running a benchmark is
[documentation/benchmarking.md](../documentation/benchmarking.md).

---

# \<bench-name\>

**Model** — architecture, parameters, feature path, how many loaded.

| Instance | Model | Params | Provider | Parallelisation | rows | TPS | p50 | p95 | srv p50 | srv p95 | CPU % | Mem % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

One row per **sustained** rate: achieved ≈ offered with `shed = 0`. A rate that shed is
not a result — leave it out.

## The columns

```text
Instance         c9g.24xlarge          the instance type, dots and all
Model            transformer / raw     architecture / feature path
Params           10M                   parameters per model, not summed over arms
Provider         onnxruntime/CPU       what the BANNER registered, not what was asked for
Parallelisation  conn 512 · async 32 · inf 8 · intra 11 · sess 8 · spin off · cores 1-7/8-95
rows             100                   rows per request, the client's ROWS
TPS              400                   requests per second sustained
p50, p95         25.98  31.92          CLIENT ms, end to end
srv p50, srv p95 21.40  26.10          SERVER ms, the engine call alone
CPU %            40.9                  server host mean over the measured window
Mem %            3.1                   server process RSS as a share of host memory
```

`rows` is a column and not a line of prose because it decides the other numbers. The load on
the host is `rows × TPS`, and latency is close to linear in `rows` while parallelism inside
one request is not — so two rows differing only in `rows` are two results, not one.

`Parallelisation` is every field of the `threading` config, in banner order, and the banner is
where to read them:

| In the cell | `threading` field | Banner line |
|---|---|---|
| `conn 512` | `connection_concurrency` | `admit  : 512 in flight per connection` |
| `async 32` | `worker_threads` | `threads -> 32 async, …` |
| `inf 8` | `inference_concurrency` | `…, 8 inference x …` |
| `intra 11` | `intra_op_threads` | `… x 11 intra-op x …` |
| `sess 8` | `sessions_per_model` | `… x 8 session(s)` |
| `spin off` | `allow_spinning` | `spin   : idle … threads sleep` |
| `cores 1-7/8-95` | `worker_cores`/`compute_cores` | `cores  : async on …, compute on …` |

Omit `spin` and `cores` when the banner does not print them — unset means the runtime's own
default, and claiming a value it did not set makes the report disagree with the run.

**Client `p50`/`p95`** come from the client's `summary.txt`, in ms, and are what a caller
experienced: queueing and both network hops included.

**Server `srv p50`/`srv p95`** are CloudWatch `InferenceTime`, converted from µs to ms, and are
the engine call alone — no admission queue, no network.

> `client p50 − srv p50` is the whole cost of everything that is not the engine. When it grows
> with the offered rate, the queue in front of `inference_concurrency` is what is growing, and
> the fix is a threading one rather than a model one.

## Filling it in

Every figure comes from `raw/`, so nothing is transcribed from a terminal:

```bash
R=benchmark-data/results/<bench-name>/raw

grep '^rows'  $R/client/<host>/<cell>/meta.txt      # the load shape asked for
cat           $R/client/<host>/<cell>/summary.txt   # client p50/p95, achieved, shed
grep 'cores'  $R/server/<host>/<cell>.server.log    # the configuration that actually ran
python3       $R/driver/verify.py                   # check the report against all of it
```

`driver/*_results.json` carries every cell's parsed result — `p50`, `p95`, `shed`, host CPU,
and the banner it ran under — so a table row is checkable without re-reading a log.

## What this run says

Two or three findings, each a sentence with the numbers in it. A number without a claim
attached to it belongs in the table and nowhere else.

## Environment

Host: `<instance-type>` — `<vCPU> <arch>, <cores/threads>, <NUMA nodes>, <memory>, <OS>,
kernel <version>`, ONNX Runtime `<version>`. Execution provider, and whether any accelerator
library was built or loaded.
Client: `<instance-type>`, client CPU under `<n>` %.

Method: `<n>` cells. `<warmup>` warmup, `<measured>` measured, server restarted between cells
with a `<settle>` settle. `srv` columns are CloudWatch `InferenceTime`. CPU % and Mem % are
`sample_host.sh` over the same window, started per cell. Every configuration was read back
from the server's own banner rather than from what was asked for.

Raw logs: `s3://<bucket>/<prefix>/`
Metrics: CloudWatch namespace `<namespace>`
