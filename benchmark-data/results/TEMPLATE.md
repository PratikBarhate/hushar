# <instance-type> · <provider> · <YYYY-MM-DD>

**Model** — architecture, parameters, feature path, how many loaded.

| Instance | Model | Params | Provider | Parallelisation | rows | TPS | p50 | p95 | srv p50 | srv p95 | CPU % | Mem % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

One row per **sustained** rate: achieved ≈ offered with `shed = 0`. A rate that shed is
not a result — leave it out.

## The columns

```text
Instance         c9g.48xlarge          the instance type, dots and all
Model            transformer / raw     architecture / feature path
Params           10M                   parameters per model, not summed over arms
Provider         onnxruntime/CPU       what the BANNER registered, not what was asked for
Parallelisation  conn 500 · async 24 · inf 8 · intra 21 · sess 8 · spin off
rows             100                   rows per request, the client's ROWS
TPS              400                   requests per second sustained
p50, p95         25.98  31.92          CLIENT ms, end to end
srv p50, srv p95 21.40  26.10          SERVER ms, the engine call alone
CPU %            40.9                  server host mean over the measured window
Mem %            3.1                   server process RSS as a share of host memory
```

`rows` is a column and not a line of prose because it decides the other numbers. The load
on the host is `rows × TPS`, and latency is close to linear in `rows` while parallelism
inside one request is not — so the same host, weights and `Parallelisation` give 14 ms at
50 rows and 112 ms at 100. Two rows differing only in `rows` are two results, not one.

`Parallelisation` is every field of the `threading` config, in banner order, and the
banner is where to read them:

| In the cell | `threading` field | Banner line |
|---|---|---|
| `conn 500` | `connection_concurrency` | `admit  : 500 in flight per connection` |
| `async 24` | `worker_threads` | `threads -> 24 async, …` |
| `inf 8` | `inference_concurrency` | `…, 8 inference x …` |
| `intra 21` | `intra_op_threads` | `… x 21 intra-op x …` |
| `sess 8` | `sessions_per_model` | `pool   : 8 session(s) per model …` |
| `spin off` | `allow_spinning` | `spin   : idle … threads sleep` |
| `cores 0-23/24-191` | `worker_cores`/`compute_cores` | `cores  : async on …, compute on …` |

Omit `spin` and `cores` when the banner does not print them — unset means the runtime's
own default, and claiming a value it did not set makes the report disagree with the run.

**Client `p50`/`p95`** come from the client's `summary.txt`, in ms, and are what a caller
experienced: queueing and both network hops included.

**Server `srv p50`/`srv p95`** are CloudWatch `InferenceTime`, converted from µs to ms,
and are the engine call alone — no admission queue, no network. With two arms resident
they are **averaged across the two models**: the same weights under two `model_id`s, so
the pair is one compute measurement rather than two.

> `client p50 − srv p50` is the whole cost of everything that is not the engine. When it
> grows with the offered rate, the queue in front of `inference_concurrency` is what is
> growing, and the fix is a threading one rather than a model one.

## Filling it in

```bash
# rows — the load shape the client was told to send
grep '^rows' /tmp/hushar-client/<label>/meta.txt

# client p50/p95 — from the client host's summary
cat /tmp/hushar-client/<label>/summary.txt

# srv p50/p95 — µs from CloudWatch, one call per arm, averaged
scripts/benchmark/server_latency.sh HusharBench bench-raw bench-raw-candidate

# CPU % and Mem % — sampled on the server host throughout the sweep
scripts/benchmark/sample_host.sh report /tmp/hushar-server/samples.log
```

`documentation/benchmarking.md` §4 has the sampling and §6 the whole recording step.

## What this run says

Two or three findings, each a sentence with the numbers in it. A number without a claim
attached to it belongs in the table and nowhere else.

## Environment

Host: `<vCPU> <arch>, <NUMA nodes>, <memory>, <OS>, ONNX Runtime <version>`.
Client: `<instance-type>, <processes>, <duration>s measured after <warmup>s warmup`,
client CPU under `<n>` %.

Raw logs: `s3://<bucket>/<prefix>/`
