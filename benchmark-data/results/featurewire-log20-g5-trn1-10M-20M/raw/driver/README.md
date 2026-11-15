# hushar on its own wire, logging at 20%, on an A10G and a Trainium1: the order of operations

`featurewire-log50-g5-trn1-10M-20M` turned server-side vectorization and feature logging
on together and found three cells over their device ceiling. It could not say how much of
that the LOG cost, because it moved the wire and the log at once. This round moves one knob:

> **At `sample_rate` 0.2 instead of 0.5, with everything else identical, what does one
> `g5.2xlarge` and one `trn1.2xlarge` serve, and at what latency?**

So every number here subtracts directly against that round. The wire, the session count,
the graphs, the engines, the window and the loads are unchanged on the A10G; only the
sampling rate moved, and only the Trainium 20M rates moved with it, for the reason below.

**THAT ROUND IS NOT RETAINED AS A REPORT**, so this one does not depend on it being there.
The figures the `Against ...` table cites were lifted into
`raw/control-featurewire-log50-g5-trn1-10M-20M.json` by `snapshot_control.py`, which names
its source and reproduces each lane's own bucket reduction; its cell records remain in the
bench bucket. A citation, not a copy: the banners, samples and client output behind those
numbers are not here.

## The grid

Eighteen cells. Two lanes per accelerator, one model size each, all four lanes in parallel.

| accelerator | model | rows x TPS | sessions | arm(s) | cells |
|---|---|---:|---:|---|---:|
| A10G | 10M | 100 x 1000, 500 x 200 | 2 | hushar, Triton | 4 |
| A10G | 20M | 100 x 750, 100 x 720, 100 x 700, 500 x 150 | 2 | hushar, Triton | 8 |
| Trainium1 | 10M | 100 x 1000, 500 x 200 | 2 | hushar | 2 |
| Trainium1 | 20M | 100 x 450, 100 x 400, 500 x 80, 500 x 75 | 2 | hushar | 4 |

Every cell: one engine call per request, `sessions_per_request` 1 so 2 requests in flight,
`sample_rate` 0.2 to S3, no candidate model, one core and one execution context per Neuron
replica, FP32. 5 min warm-up discarded, 15 min measured in 15 one-minute slices, a fresh
server per cell, 1 min idle after it stops.

## What is held, and what moved

This round is a SUBTRACTION against two runs that measured the same geometry, so almost
nothing is new:

| held fixed from the control runs | changed here |
|---|---|
| graph, engine shape, TensorRT library, NEFF | wire: 210 named features instead of the finished 690-float vector |
| session count (2), one session per request | inference log: `sample_rate` 0.2 to S3 instead of 0.0 |
| loads, window, warm-up, slices, async workers (4) | 20M on Trainium: 500 and 100 TPS instead of 750 and 150 |

The controls are
[`nominibatch-sessions-gpu-g5-10M-20M`](../../../nominibatch-sessions-gpu-g5-10M-20M/summary.md)
and
[`nominibatch-sessions-neuron-trn1-10M-20M`](../../../nominibatch-sessions-neuron-trn1-10M-20M/summary.md).

## Read this before spending the device hours

**THE SESSION COUNT IS PINNED, AND THAT IS A FINDING BEING SPENT, NOT A CORNER CUT.** Both
control runs swept 2, 4 and 8 at these loads. On the A10G the count was worth +0.02 to
+0.20 ms and 2 was the lowest sustained median in all four loads; on Trainium a count above
one replica per core buys queue lanes rather than device throughput, because a NeuronCore
executes one NEFF at a time and this host has two. Re-measuring it would spend eight cells
confirming a tie.

**THE 20M TRAINIUM RATES BRACKET A CEILING, AND THE ARITHMETIC IS WRITTEN DOWN.** The
control measured the device span per request on this exact configuration: 3.861 ms at 100
rows and 19.012 ms at 500. A NeuronCore executes one NEFF at a time and this host has two,
so the ceiling is 2/0.003861 = 518 req/s and 2/0.019012 = 105 req/s. The control's 500 and
100 — 97% and 95% of those — came back at 490.6 req/s holding 0 of 15 slices and 99.4
holding 5 of 15. Cutting the log from 50% to 20% removes host work, not device work, so the
ceiling barely moves. Each packaging therefore gets a PAIR straddling the knee, at 87%/77%
and 76%/71% occupancy: at least one of each reports a latency, and the two together say
where the knee is. Four cells on that lane is 1.5 h and still finishes inside the A10G
lane, so the second rate of each pair is free.

**THE TWO ARMS ON THE GPU NO LONGER RECEIVE THE SAME BYTES, AND THE REPORT MUST NOT IMPLY
THEY DO.** hushar takes 210 named features, vectorizes them and logs half of them; Triton
takes the finished tensor and writes nothing, because it has no counterpart for either.
That is Triton as it ships against hushar as it is deployed. `VectorizationTime` is
published on every hushar cell and the control run supplies the same cells without either
cost, so the report prices the difference rather than absorbing it.

**THE LOG SETTINGS ARE READ BACK OUT OF THE BANNER, NOT TRUSTED.** Each of these could be
wrong while the server served happily, and each would make the round a measurement of a
configuration nobody asked for:

| checked | because |
|---|---|
| `logging 1 request in 5 (20.0%)` | `Sampler::new` rounds a rate to the nearest one-in-n, so a cell handed 0.18 would log 20% and report 0.18 — and a literal left over from the 50% round would pass a banner logging at the wrong rate, so the verifier derives the percent from the fraction |
| `inference logs -> s3://…`, `Amazon S3` | a URI that failed to parse writes to the root volume, which is gp3 and cheaper than the real path |
| `every <N> batches` | `batch_size` counts REQUESTS, so the wrong value at 500 rows is a 2.6 GB object, and the flush is what disturbs latency |
| `<N> features transformed and concatenated` | a one-feature banner means the lane staged the control's pre-vectorized configuration under this run's name |

**ONE OBJECT PER 50,000 LOGGED ROWS, WHICH IS WHY THE BATCH SIZE IS NOT A CONSTANT.** A
request is 100 rows in one cell and 500 in another, so a fixed `batch_size` would hand S3
a 52 MiB object at one packaging and a 260 MiB object at the other. Scaling it by the row
count holds the object at ~50 MiB and the flush cadence at ~1/s on the heaviest cell of
each packaging, so the log path disturbs every cell the same way instead of being a hidden
variable. The payload it rests on is measured, not assumed:
`design-vs-triton-cpu-arm64-10M` recorded the feature wire at 5,757 B fixed + 1,031 B per
row from real encoded requests.

**SHEDDING THE LOG IS DESIGNED BEHAVIOUR AND IS THEREFORE RECORDED.** When every send
permit is out the accumulator drops the oldest batch and prints a `logs:` line, so that a
slow bucket cannot become slow inference. A cell that shed did not pay the full cost of
logging, so its latency would understate it. `run_cell` greps the shed line, counts failed
writes, and counts the objects that reached the bucket — on every cell, on both lanes.

**TWO LANES, ONE S3 PREFIX, SO THE CELL IDS AND THE INSTANCE TAGS CARRY THE LANE.** Cells
go to `runs/<run>/cells/` from both accelerators. Ids are prefixed `g5fw-` and `trn1fw-`,
which is also the CloudWatch `ModelId`: the control runs' `g5nb-`/`trn1nb-` ids share this
grid's loads AND its session count, so a shared prefix would average two configurations
into one time series. Instance Name tags carry the lane too — both rigs call their clients
`cl1` and `cl2`, and `rig.py adopt` finds hosts by that tag, so without it the second rig
would adopt the first lane's client and drive a server across an availability zone.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PROFILE=hushar-bench
export AWS_DEFAULT_REGION=us-west-2
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate local   # boto3
```

The account already carries the bucket, both instance profiles (`hushar-bench-server`,
`hushar-bench-client`) and both security groups. Nothing here creates them. The server
role's `InferenceLogs` statement already allows `s3:PutObject` to `inference-logs/*`,
which is where this round writes.

**Credentials outlive nothing.** A capacity wait can run for hours and these are temporary
credentials. Start the refresher before `launch` and kill it once the sweeps are detached —
the hosts run under instance profiles and do not need the laptop's session.

```bash
ACCOUNT=$(cat ../../../../.account-ids) nohup bash gpu/refresh_creds.sh \
  > /tmp/featurewire-creds.log 2>&1 &
```

## 0 · prove the grid before spending anything

```bash
./dryrun.py            # must print `0 failures`
```

Imports both sweeps with a fake rig and asserts every property this round was specified
with — the loads, the pinned session count, the feature wire, the 50% rate and the fraction
it resolves to, the object size, one core and one context per replica, no mini batch, no id
collision across the lanes. It launches nothing and reaches no network: the sweeps do all
of their grid arithmetic at import time.

## 1 · package

```bash
bash gpu/package.sh      # -> REV, MD5_10m, MD5_20m
bash neuron/package.sh   # -> REV
```

Each uploads the working tree as a bundle, its own sampler and sweep. The GPU one also
uploads both model directories and derives the Triton tensor feature spec from the
generated configuration. **No models travel for Neuron**: a cell runs a NEFF the host
restores from the S3 cache.

## 2 · launch both rigs

```bash
( cd gpu    && CAPACITY_TRIES=300 CAPACITY_WAIT=120 python rig.py launch )
( cd neuron && AMI_ALLOW_STALE=1 CAPACITY_TRIES=90 CAPACITY_WAIT=90 python rig.py launch )
python gpu/rig.py status; python neuron/rig.py status
```

Each rig is 2 servers + 2 clients and keeps its own `rig-state.json` in its own directory.

**`g5.2xlarge` is scarce** and offered in `us-west-2a`, `2b` and `2c`. **`trn1.2xlarge` is
offered only in `2b` and `2d`**; the other zones return `Unsupported` rather than a capacity
error. Each rig lands all four of its hosts in one zone, because a cross-zone client would
add a network hop to one lane and not the other — the two rigs need not share a zone with
each other, and usually cannot.

**One lane per rig is a normal outcome.** When a zone cannot hold a second server, `launch`
records one lane and both sizes run on it one after the other. That is *better* for the
10M↔20M comparison; it only costs wall clock. `AMI_ALLOW_STALE=1` is required for Neuron:
its server image is pinned to the PyTorch 2.9 DLAMI, which is older than `rig.py`'s 45-day
freshness limit and is the last image carrying `torch_neuronx.trace`.

## 3 · keys

```bash
( cd gpu && python keys.py ); ( cd neuron && python keys.py )
```

One ephemeral ed25519 keypair per lane, over SSM. The laptop reaches every host over SSM;
the orchestrator runs *inside* the VPC and needs SSH, which this installs.

## 4 · deploy

```bash
( cd gpu && python fanout.py servers,clients -f deploy.sh --timeout 5400 \
    --env REV=$REV_GPU \
    --env MODEL_SPECS="10m:$MD5_10m 20m:$MD5_20m" \
    --env ROWS_LIST="100 500" \
    --env RUN=featurewire-log20-g5-trn1-10M-20M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )

( cd neuron && python fanout.py all -f deploy.sh --timeout 4200 \
    --env REV=$REV_NEURON --env SIZES="10m 20m" \
    --env RUN=featurewire-log20-g5-trn1-10M-20M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
```

GPU server deploy, in order: toolchain; both graphs, md5-checked; docker and the Triton
image; **read the image's own TensorRT and CUDA versions**; extract CUDA, cuDNN and
TensorRT out of it; download the ONNX Runtime GPU build matching *that* CUDA major; `ldd`
the TensorRT provider against the extracted libraries and **refuse** on any unresolved
dependency; write `env.sh`; `cargo build --features cpu,tensorrt`; build one `trtexec` plan
per row count. Neuron server deploy: toolchain, the Neuron device and runtime verified, the
venv located by what it imports, `cargo build` with the Neuron backend, the sampler.

### on matching the TensorRT versions

This round was asked to put Triton and hushar on the same TensorRT. `deploy.sh` extracts
CUDA, cuDNN and TensorRT **out of the Triton image** and hushar links those copies, so both
arms execute on the same library bytes rather than on two builds that agree about a version
string. Three gates:

1. `ldd` on `libonnxruntime_providers_tensorrt.so` at deploy — any unresolved dependency
   refuses the deploy.
2. `preflight` compares the `libnvinfer` hushar will dlopen against the version the image
   ships, and refuses the lane if they differ.
3. The version is **stamped on every cell of both arms**, so the claim is checkable from
   the records and not only from a deploy log.

The ONNX Runtime *wrapper* versions cannot match and the report says so rather than
implying otherwise: hushar needs ORT ≥ 1.28 (`onnxrt-rs` targets `ORT_API_VERSION 28`) and
the image ships 1.24 in its `onnxruntime` backend. Triton therefore runs its **`tensorrt`**
backend, which uses no ONNX Runtime at all.

## 5 · run

```bash
RG=$(cd gpu && python -c "import hosts;print(hosts.rig_env())")
RN=$(cd neuron && python -c "import hosts;print(hosts.rig_env())")

( cd gpu && python fanout.py cl1 -f launch.sh --timeout 300 \
    --env REV=$REV_GPU --env MODEL_SPECS="10m:$MD5_10m" \
    --env RIG="$RG" --env SERVER_FOR=g5-1 \
    --env RUN=featurewire-log20-g5-trn1-10M-20M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
# and cl2 with SERVER_FOR=g5-2, MODEL_SPECS="20m:$MD5_20m"

( cd neuron && python fanout.py cl1 -f launch.sh --timeout 300 \
    --env REV=$REV_NEURON --env SIZES="10m" --env RIG="$RN" --env SERVER_FOR=trn1-1 \
    --env RUN=featurewire-log20-g5-trn1-10M-20M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
# and cl2 with SERVER_FOR=trn1-2, SIZES="20m"
```

Returns immediately. Each orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds — so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

Two phases per lane. There is no probe phase: the session count is pinned.

| phase | what it does | cost |
|---|---|---|
| `prime` | GPU: one TensorRT engine per row count, each in its own cache directory, built then **proved** reloadable by a fresh server. Neuron: one NEFF per row count, restored from the S3 cache or compiled, each then **proved** loadable and servable | 2 shapes per size per lane |
| `main` | the cells, 5 min warm-up + 15 min measured + 1 min idle, fresh server per cell | 4 per g5 lane, 2 and 4 on the trn1 lanes |

`b100` and `b500` are already cached for both sizes, so Neuron's prime restores rather than
compiles. A single phase can be run alone with `--env PHASE=prime`.

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/featurewire-log20-g5-trn1-10M-20M/cells/ | wc -l
( cd gpu && python fanout.py cl1,cl2 -c 'tail -40 /root/orchestrator-main-*.log' )
( cd neuron && python fanout.py cl1,cl2 -c 'tail -40 /root/orchestrator-main-*.log' )
```

Expect **18 cells**. A cell is 5 + 15 + 1 minutes plus up to a minute of minute-alignment
and about 90 s of server start, sampling and upload, so roughly **23 min each** — about
**1.6 h per g5 lane** and **0.8 h per trn1 lane**, all four in parallel, plus priming.
Check back in hours, not minutes.

## 6 · collect, report, verify

```bash
( cd gpu && ./collect.py )        # -> raw/results-gpu.json
( cd neuron && ./collect.py )     # -> raw/results-neuron.json
./snapshot_control.py --apply     # cites the control's figures into this round's raw/
./scrub.py --apply                # masks the bucket name in the synced evidence
./report.py                       # writes ../../summary.md from both
./verify.py                       # re-derives every number in it; must print 0 failures
```

`snapshot_control.py` runs once and is then idempotent. It reads the control round from
`results/` if it is still there, and otherwise rebuilds the same figures from the bench
bucket's cell records plus CloudWatch -- so a report generated after the control round was
removed shows the same numbers rather than silently losing the table.

**The scrub runs BETWEEN collect and report, and that order is load-bearing.** `collect.py`
syncs the archived server logs and resolved configurations straight out of S3, and on this
round those name the inference log's bucket -- whose name embeds an account id. Reporting
before scrubbing would copy it into `summary.md`. Re-run `scrub.py` after any later
`collect.py`, since a sync restores the unmasked copies.

Each collector takes only its own accelerator's cells, by the `accelerator` field on the
cell rather than by a prefix someone could rename.

## 7 · archive and terminate

```bash
( cd gpu && CONFIRM=terminate python rig.py terminate )
( cd neuron && CONFIRM=terminate python rig.py terminate )
pkill -f refresh_creds.sh
# The logged feature rows are not evidence the report reads -- only the latency they cost
# is -- and they are several hundred GB. The objects' COUNT and SIZE are already recorded
# on every cell, so deleting them loses nothing the report uses.
aws s3 rm "s3://$HUSHAR_BENCH_BUCKET/inference-logs/featurewire-log20-g5-trn1-10M-20M/" \
  --recursive
python ../../../../scripts/check_privacy.py
```

Run the privacy check before committing: the bucket name carries an account id and must not
reach a tracked file.

## Known limits of this round

- **THE PRIME RECORD IS KEYED PER LANE AND SIZE, NOT PER HOST.** The 100 x 720 and
  100 x 700 cells were added afterwards on a FRESH `g5.2xlarge` (`g5-3`, us-west-2b),
  and `prime` found `prime/gpu-20m.json` already in S3 from the original 20M lane and
  skipped -- so that host's TensorRT engine cache was cold and the engine compiled inside
  the FIRST of its four cells, within the 300 s discarded warm-up. The measured window is
  unaffected (a 20M engine builds in ~26 s against a 300 s warm-up, and the cache then
  serves the remaining three cells), and the engine build figures in the report come from
  the original lane. Keying the record per host would fix it and is not done here, because
  the fix is to key it per host AND keep the first lane's record, which is a change to make
  deliberately rather than mid-round.
- **Those four cells sit on a different machine and a different availability zone** from
  the 100 x 750 and 500 x 150 cells they are read against. The provenance table reports
  cells per host. This series measured host-to-host spread at -10.8% to +2.7%, so the
  700/720 pair is sound against each other and against its own Triton arm -- all on one
  host -- and the step from 720 to 750 carries that spread on top.

- **One change against the control, which is the point** — only `sample_rate` moved, so a
  cell-to-control delta is the log rate. What that still does NOT give is the log's
  absolute cost: both rounds log, so the pair prices 0.5 against 0.2 and extrapolating to
  0.0 assumes a linearity nothing here measures. A third round at 0.0 on this wire would
  close it.
- **The Trainium 20M rates differ from the control's**, so those four cells have no control
  pair and the subtraction table reports `--` for them. They answer the capacity question
  instead.
- **The GPU arms differ in what they were asked to do**, not only in which server did it.
  See above; the report states it rather than ranking the two as equals.
- **A session count of 2 is pinned on the strength of the control runs**, not re-measured.
  If the feature path shifts the optimum the shift is invisible to this grid.
- **No cell is repeated**, so the tie band is inherited. The first GPU campaign measured
  host-to-host repeatability at -10.8% to +2.7%.
- **FP32 only.** Both stacks take TensorRT's default builder settings and the NEFFs are
  FP32, so every number is a floor rather than a ceiling.
- **8 vCPUs now carry real host work.** On this wire the async pool performs 210
  transformations per row and serialises half of them to protobuf, where on the control's
  wire it did almost none. The CPU columns measure it rather than the report caveating it.
- **The Trainium parameter counts differ slightly from the ONNX ones** — 9,142,789 against
  9,138,693 at 10M — because `torch_neuronx.trace` rebuilds the same seeded architecture.
  Both are recorded; the report does not treat them as one number.
