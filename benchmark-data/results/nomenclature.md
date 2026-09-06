# Reading a benchmark result

What every column and field means, and how they check against each other. Definitions are
taken from the code that prints them — `benchmark-client/src/main.rs` for the client,
`hushar/src/io/metrics.rs` and `hushar/src/inference/scoring.rs` for the server.

---

## The summary table

This is `$RUN_DIR/summary.txt`, one row per model per rate, assembled by
`scripts/summarise_benchmark.py` from the client's own printed output.

```
model                     offered   achieved   p50_ms   p90_ms   p99_ms    shed  failed
vectorized-cpu                400      400.0    15.29    16.50    19.12       0       0
                                 |          |        |                          |      |
                        what you        what you   how long it took        how many
                          asked for       got                             never ran / broke
```

| Column | Meaning | Source |
|---|---|---|
| **model** | `<model>-<provider>` — which feature path, on which execution provider | run script tag |
| **offered** | The rate you **asked** for. A constant from `--tps`, *not* a measurement | `args.tps` |
| **achieved** | The rate the server **delivered**: successful post-warmup requests ÷ measured seconds | `measured.len() / duration_secs` |
| **p50_ms** | Median round-trip latency, client-side, over the measured window | `Percentiles::of` |
| **p90_ms** | 90th percentile — 1 request in 10 was slower | " |
| **p99_ms** | 99th percentile — 1 request in 100 was slower | " |
| **shed** | Requests the generator wanted to send but **never sent**, because all in-flight slots were busy | `try_acquire_owned()` failed |
| **failed** | Requests that **were** sent and came back an error | `outcome.is_err()` |

### shed is not failed

The distinction that matters most. Both are losses, but only one is a problem:

```
        request wanted
              |
     is a slot free? ------ no ---->  SHED     never left the client.
              |                                the offered rate was too high.
             yes                               Normal above the knee.
              |
         sent to server
              |
       did it come back? ---- err -->  FAILED   the server rejected or errored.
              |                                 Should be 0. Investigate.
             ok
              |
       latency recorded  ---->  feeds achieved, p50, p90, p99
```

`shed` exists because this is an **open-loop** generator. When nothing is free it drops the
request and counts it, rather than waiting for a slot. If it waited, it would quietly slow its
own clock and you would be measuring the generator instead of the server. So `shed > 0` reads as
*"this rate is above what the configuration sustains"* — which is a finding, not a fault.
`failed > 0` reads as *"something is broken"*.

The client says so itself once shedding passes 1% of the offered total:

```
NOTE: 10% of the offered rate was shed, so 600 req/s is above what this
configuration sustains. The latencies above are for the requests that ran.
```

That last sentence is the trap to avoid: **latencies past the knee describe the survivors.**
A p50 that looks tolerable while 10% was shed is not a tolerable p50.

---

## The client's per-run output

The table above is a reduction of this, printed once per rate:

```
  achieved      :     538.0 req/s   (538.0 rows/s)     <- post-warmup successes / duration
  offered       :       600 req/s                      <- what you asked for
  measured      :     10760 requests  (of 12439 that succeeded overall)
                          |                    |
                  percentiles come         includes warmup
                  from these only
  latency  p50  :    115.02 ms
           p90  :    127.37 ms
           p99  :    135.24 ms
           max  :    145.57 ms
           mean :    114.25 ms
  failed        :         0
  shed          :      1361  (offered but never sent: 64 in flight was the cap)
```

`rows/s` is `achieved × --rows`. With one row per request the two are equal; batching separates
them.

### The counts close exactly

```
offered_count  =  succeeded  +  failed  +  shed
```

Worth checking when a result looks odd, because it will not close if something was miscounted.
`offered_count` is the rate times the **whole** run, warmup included:

```
vectorized @ 600, 3s warmup + 20s measured
    600 req/s x 23 s   =  13,800 offered
                          12,439 succeeded + 0 failed + 1,361 shed  =  13,800   ✓

    achieved  =  10,760 measured / 20 s   =  538.0 req/s              ✓
    shed      =   1,361 / 13,800          =  9.9%  ->  the "NOTE: 10%" line
```

### One asymmetry to remember

**`achieved` and the percentiles exclude the warmup. `shed` and `failed` do not.**

```
  |<---- warmup 3s ---->|<-------- measured 20s -------->|
  |                     |                               |
  |   latencies: cleared at this boundary ---------------|--> p50/p90/p99, achieved
  |                                                     |
  |<---- shed and failed counted across all 23s --------->
```

The latency buffer is cleared when warmup ends; the three counters never reset. So you cannot
divide `shed` by the measured duration and get a shed rate — divide by warmup + duration.

### How the percentiles are computed

Nearest-rank, no interpolation: sort ascending, index `floor(n × q)`, clamped to the last
element. So p99 of 8,000 samples is element 7,920 — the 80th slowest request.

This is why **p99 is thin at low rates.** At 50 req/s over 20 s there are only 1,000 samples, so
p99 is element 990 and just 9 requests sit above it. In this benchmark `named @ 400` measured a
p99 of 16.41 ms in one run and 128.12 ms in the other **on identical p50s** — the p50 is solid,
the p99 is a couple of requests. Read p99 as an order of magnitude, not a number.

---

## The server's own timings

Separate from the client, on stderr, one line per 500 batches (`$RUN_DIR/logs/<tag>.server.log`):

```
metrics: 500 batches | mean us: VectorizationTime 11, ToTensorTime 0, InferenceTime 10126 | max InferenceTime 34457
             |                              |                                                        |
      batches in this window        means over the window, microseconds            slowest single batch
```

Aggregated rather than printed per request: one line per request would be unreadable at rate and
would itself become the bottleneck.

| Field | What it times | Boundary in `scoring.rs` |
|---|---|---|
| **VectorizationTime** | Building the input batch from the request's features — cloning row ids and features, then every transformation | `build_start - stage_start` |
| **ToTensorTime** | Handing that batch to the backend | `inference_start - build_start` |
| **InferenceTime** | The engine call itself | `inference_end - inference_start` |
| **max InferenceTime** | Slowest single batch in the window, not a percentile | running max |

Two readings that follow from the boundaries:

- **`ToTensorTime` is 0 µs** and stays 0. The backend *wraps* the row-major buffers rather than
  copying them, so there is nothing between building the batch and calling the engine.
- **`VectorizationTime` ÷ `InferenceTime` is the service's share of a request.** ~10 µs against
  ~10 ms is ~0.1%, which is the number that says whether optimising the feature path is worth
  anything.

A caveat on attribution: the window is 500 batches, so lines do not line up with rate
boundaries. Compare the **first** windows (unsaturated) against the **last** (saturated) rather
than trying to pin a window to one rate.

### Client latency vs server InferenceTime

They answer different questions, and the gap between them is the point:

```
client p50 115 ms  =  queueing  +  transport  +  vectorization  +  inference
server InferenceTime 15 ms      ^                              ^
                                |                              |
                    where the time goes past the knee     ~0.1% of the total
```

Below the knee the two nearly agree (~10.6 ms client, ~10.1 ms engine). Past it they diverge, and
the difference is **waiting**, not the model getting slower. `InferenceTime` rising from ~10 ms
to ~15–19 ms under load is cores being shared between concurrent inferences.

---

## Quick checklist

| Symptom | Reading |
|---|---|
| `failed > 0` | Something is broken. Nothing in a healthy run should error |
| `shed > 0`, `achieved ≈ offered` | At the edge; the rate is just about sustained |
| `shed` large, `achieved < offered` | Past the ceiling. Latencies describe survivors only |
| `achieved ≈ offered`, flat p50 | Comfortably below the knee. This is the comparable regime |
| p99 ≫ p90 at a low rate | Probably a handful of samples. Check p50 and p90 first |
| `ToTensorTime > 0` | Unexpected — a copy crept into the batch path |
