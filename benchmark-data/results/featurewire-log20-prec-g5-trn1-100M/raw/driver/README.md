# A 100M model at two precisions, on its own wire with the log on: the order of operations

`featurewire-log20-g5-trn1-10M-20M` measured one `g5.2xlarge` and one `trn1.2xlarge` on
hushar's own feature wire at `sample_rate` 0.2, at 10M and 20M parameters.
`nominibatch-precision-gpu-g5-100M` measured a 100M model at FP32 against BF16, but on the
pre-vectorized tensor wire with the log off, and on the A10G only. This round is the
intersection neither had:

> **At 100M parameters, on the feature wire with one request in five logged to S3, what
> does one `g5.2xlarge` and one `trn1.2xlarge` serve at FP32 and at BF16?**

## The grid

Twelve cells. One precision per host, four lanes in parallel.

| accelerator | precision | rows x TPS | sessions | arm(s) | cells |
|---|---|---:|---:|---|---:|
| A10G | FP32 | 100 x 100, 500 x 20 | 2 | hushar, Triton | 4 |
| A10G | BF16 | 100 x 200, 500 x 40 | 2 | hushar, Triton | 4 |
| Trainium1 | FP32 | 100 x 100, 500 x 20 | 2 | hushar | 2 |
| Trainium1 | BF16 | 100 x 200, 500 x 40 | 2 | hushar | 2 |

Every cell: one engine call per request, `sessions_per_request` 1 so 2 requests in flight,
`sample_rate` 0.2 to S3, no candidate model, one core and one execution context per Neuron
replica, 210 named features vectorized on the server. 5 min warm-up discarded, 15 min
measured in 15 one-minute slices, a fresh server per cell, 1 min idle after it stops.

**BF16 IS OFFERED TWICE THE RATE OF FP32 AT EACH PACKAGING, BY INSTRUCTION.** So the pair
is NOT a like-for-like latency comparison and the report must not print one. The readings
that are comparable are the per-request device time, which is rate-independent at these
occupancies, and the throughput each arm actually sustained.

## Read this before spending the device hours

**THE PRECISION IS AN EXPLICIT FLAG ON BOTH ARMS, AND NOT BECAUSE THE DEFAULT IS WRONG.**

| arm | flags | trace error | NEFF at b100 |
|---|---|--:|--:|
| FP32 | `--auto-cast=none` | 1.48e-06 | 360.0 MiB |
| BF16 | `--auto-cast=matmult --auto-cast-type=bf16` | 2.32e-03 | 154.5 MiB |

This round was planned on the belief that `neuronx-cc`'s documented `--auto-cast` default
of `matmult` reached `torch_neuronx.trace`, which would have made every earlier Trainium
round's FP32 label wrong. **IT DOES NOT, AND THEY WERE RIGHT.** `neuron/probe_autocast.sh`
compiles one model three ways on one host: "no flag" and `--auto-cast=none` produce a
BYTE-IDENTICAL NEFF with an identical trace error, against a 0.40x NEFF and a
three-orders-larger error for `--auto-cast=matmult`. The CLI reference describes the
standalone `neuronx-cc compile` entry point, not the framework tracing path. Evidence:
`raw/probe/autocast-default.txt`.

So this round's FP32 arm **is** comparable with `featurewire-log20-g5-trn1-10M-20M`'s
Trainium rows -- same wire, same log rate, same session count, same cast, different model
size. A projection from that round's 2-layer and 5-layer device spans predicts this round's
31-layer FP32 cells to within -4.9% and +1.3%.

The flag is passed anyway, for a narrower reason that still holds: a precision label must
not rest on an undocumented framework default that a release can change without saying so.

`matmult` rather than `all` because it casts the matrix-multiply engine and leaves
normalization and accumulation in FP32 -- the same division TensorRT's `--bf16` makes on
the A10G. That correspondence is what lets a Trainium BF16 row be read against a GPU one.

**THE TRACE ERROR IS THE PROOF, AND IT IS A GATE.** BF16 keeps ~3 decimal digits, so
casting the matmuls moves the compiled-versus-PyTorch maximum three orders above FP32's
~1e-6. Each arm declares a window in `compile_neuron.py` and the compile REFUSES to write a
NEFF whose error falls outside it. A flag the compiler ignored, a stale cache entry or a
work directory holding another compile's NEFF would all leave the label correct and the
artifact wrong; none survives the error check -- and it was this gate's own numbers that
exposed the mistaken premise above before it cost a re-run. The A10G has two accounts of
its own: hushar echoes every provider option it registered, and the engine it builds is
half the size.

**A PRECISION ARM THAT RAN THE OTHER PRECISION IS THE ONE FAULT THIS ROUND COULD PUBLISH
AND HAVE BELIEVED.** Everything else makes a cell fail or a number look odd. This one
produces a clean, complete, internally consistent report whose headline is an artifact of a
flag that did not take. So three keys are disjoint per arm and `dryrun.py` asserts it before
anything launches:

| key | why |
|---|---|
| `/root/trt-cache/100m/<prec>` and `models/neuron/100m/<prec>/` | a shared cache means the second arm restores the first's artifact and passes every other gate |
| `runs/<run>/prime/{gpu,neuron}-100m-<prec>.json` | a shared prime record short-circuits the second arm, which then finds no artifact in its first cell |
| the cell id, `g5fwp-…-<prec>-…` / `trn1fwp-…-<prec>-…` | `run_cell` SKIPS a cell S3 already holds, so a shared id makes the whole second arm a silent no-op -- and the id is the CloudWatch `ModelId` |

**THE TRAINIUM RATES ARE SPECIFIED AND PROJECTED TO BE ABOVE WHAT THE HOST CAN SERVE.** They
were set from the A10G, where all four were MEASURED holding in
`nominibatch-precision-gpu-g5-100M` at 93.5%, 79.9%, 89.4% and 72.6% device occupancy. One
`trn1.2xlarge` is two NeuronCores, each executing one NEFF at a time. Projecting from the
10M/20M round's own measured device spans on this exact wire, host and session count --
0.738 ms per layer at 100 rows, 3.632 at 500 -- a 31-layer model lands at 23.1 ms and
113.4 ms per call, so the ceiling is ~87 and ~17.6 req/s against the 200 and 40 asked of
BF16 and lower again against FP32's 100 and 20.

They are run anyway, and that is the point: a cell over its ceiling reports its ACHIEVED
rate and its device time, which is what says where the ceiling is, and this series' rule is
that such a cell enters no latency comparison. A second wave of rates set from THESE cells'
measured device time -- not from the projection above -- is what earns a latency on that
lane.

**THE TWO GPU ARMS NO LONGER RECEIVE THE SAME BYTES AS EACH OTHER'S TRITON PAIR, AND THE
REPORT MUST NOT IMPLY THEY DO.** hushar takes 210 named features, vectorizes them and logs
one in five; Triton takes the finished tensor and writes nothing, because it has no
counterpart for either. That is Triton as it ships against hushar as it is deployed.
`VectorizationTime` is published on every hushar cell, and
`nominibatch-precision-gpu-g5-100M` supplies the same model at the same precisions without
either cost, so the report prices the difference rather than absorbing it.

**ONE PRECISION PER LANE, WHICH PUTS THE AXIS ACROSS TWO MACHINES.** The alternative --
lane by row packaging -- would put each precision pair on one host, but it would also load
both expensive b500 NEFF compiles onto one lane and make it the critical path, and no
campaign in this series had ever recorded how long a 100M NEFF takes. (It turned out to be
~3.5 min per shape.) This series measured host-to-host spread at -10.8% to +2.7%, against
an expected BF16 effect near 2x, and the report states it rather than absorbing it.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PROFILE=hushar-bench
export AWS_DEFAULT_REGION=us-west-2
export AWS_CLI=/opt/homebrew/bin/aws     # the conda env's awscli is broken
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate local   # boto3
```

The account already carries the bucket, both instance profiles (`hushar-bench-server`,
`hushar-bench-client`) and both security groups. Nothing here creates them.

**Credentials outlive nothing.** Start the refresher before `launch` and kill it once the
sweeps are detached -- the hosts run under instance profiles and do not need the laptop.

```bash
ACCOUNT=$(cat ../../../../.account-ids) nohup bash gpu/refresh_creds.sh \
  > /tmp/fwprec100m-creds.log 2>&1 &
```

## 0 · prove the grid before spending anything

```bash
./dryrun.py            # must print `0 failures`
```

Imports both sweeps at BOTH precisions with a fake rig and asserts every property this
round was specified with -- the loads per arm, the pinned session count, the feature wire,
the 0.2 rate and the fraction it resolves to, the object size, one core and one context per
replica, no mini batch, the builder flag on each arm, the disjoint caches, and no id
collision across four lanes. It launches nothing and reaches no network.

It also COMPILES every module, which `ast.parse` is not enough to do: a repeated keyword
argument is accepted by the parser and rejected only when the call is built, so a syntax
check that stops at the AST reports a clean file and the import then fails on the host.

## 1 · package

```bash
bash gpu/package.sh      # -> REV, MD5_100m
bash neuron/package.sh   # -> REV
```

Each uploads the working tree as a bundle, its own sampler and sweep. The GPU one also
uploads the model directory and derives the Triton tensor feature spec from the generated
configuration. **No models travel for Neuron**: a cell runs a NEFF the host builds or
restores from the S3 cache.

## 2 · launch both rigs

```bash
( cd gpu    && CAPACITY_TRIES=300 CAPACITY_WAIT=120 python rig.py launch )
( cd neuron && AMI_ALLOW_STALE=1 CAPACITY_TRIES=90 CAPACITY_WAIT=90 python rig.py launch )
python gpu/rig.py status; python neuron/rig.py status
```

Each rig is 2 servers + 2 clients and keeps its own `rig-state.json`.

**`g5.2xlarge` is scarce** and offered in `us-west-2a`, `2b` and `2c`. **`trn1.2xlarge` is
offered only in `2b` and `2d`**; the other zones return `Unsupported` rather than a capacity
error. Each rig lands all four of its hosts in one zone. `AMI_ALLOW_STALE=1` is required for
Neuron: its server image is pinned to the PyTorch 2.9 DLAMI, which is older than `rig.py`'s
45-day freshness limit and is the last image carrying `torch_neuronx.trace`.

**One lane per rig is a normal outcome.** When a zone cannot hold a second server, pass both
arms to one lane (`SPECS="100m:fp32 100m:bf16"`, or two `MODEL_SPECS` entries on the GPU)
and they run one after the other. That is *better* for the precision comparison, since both
arms then sit on one machine; it only costs wall clock.

**LAUNCH THE NEW RIG BEFORE TERMINATING AN OLD ONE.** g5 capacity is not guaranteed to come
back.

## 3 · keys

```bash
( cd gpu && python keys.py ); ( cd neuron && python keys.py )
```

## 4 · deploy

```bash
( cd gpu && python fanout.py servers,clients -f deploy.sh --timeout 10800 \
    --env REV=$REV --env MODEL_SPECS="100m:$MD5" --env ROWS_LIST="100 500" \
    --env PLAN_PRECISIONS="fp32 bf16" \
    --env RUN=featurewire-log20-prec-g5-trn1-100M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )

( cd neuron && python fanout.py all -f deploy.sh --timeout 7200 \
    --env REV=$REV --env SPECS="100m:fp32 100m:bf16" \
    --env RUN=featurewire-log20-prec-g5-trn1-100M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
```

GPU server deploy, in order: toolchain; the graph, md5-checked; docker and the Triton image;
**read the image's own TensorRT and CUDA versions**; extract CUDA, cuDNN and TensorRT out of
it; download the ONNX Runtime GPU build matching *that* CUDA major; `ldd` the TensorRT
provider against the extracted libraries and **refuse** on any unresolved dependency; write
`env.sh`; `cargo build --features cpu,tensorrt`; build **four** `trtexec` plans -- one per
(precision, row count).

**BOTH PRECISIONS' PLANS ARE BUILT ON EVERY SERVER** even though a lane measures one. They
cost under a minute each, and it means a lane that has to move hosts finds its plan already
there rather than compiling inside a cell's warm-up.

Neuron server deploy: toolchain, the Neuron device and runtime verified, the venv located by
what it imports *and by whether it has `trace`*, `cargo build` with the Neuron backend, the
sampler. **NEFFs are not built here** -- they are the `prime` phase's job, which caches each
in S3 *after a server has served it* and records its SHA-256 and its trace error.

### on matching the TensorRT versions

This round was asked to put Triton and hushar on the same TensorRT. `deploy.sh` extracts
CUDA, cuDNN and TensorRT **out of the Triton image** and hushar links those copies, so both
arms execute on the same library bytes rather than on two builds that agree about a version
string. Three gates: `ldd` at deploy, a `libnvinfer` comparison at preflight, and the version
stamped on every cell of both arms.

The ONNX Runtime *wrapper* versions cannot match and the report says so: hushar needs ORT
>= 1.28 (`onnxrt-rs` targets `ORT_API_VERSION 28`) and the image ships 1.24 in its
`onnxruntime` backend. Triton therefore runs its **`tensorrt`** backend, which uses no ONNX
Runtime at all.

## 5 · run

```bash
RG=$(cd gpu && python -c "import hosts;print(hosts.rig_env())")
RN=$(cd neuron && python -c "import hosts;print(hosts.rig_env())")

( cd gpu && python fanout.py cl1 -f launch.sh --timeout 300 \
    --env REV=$REV --env MODEL_SPECS="100m:$MD5:fp32" \
    --env RIG="$RG" --env SERVER_FOR=g5-1 --env LANE_TAG=fp32 \
    --env RUN=featurewire-log20-prec-g5-trn1-100M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
# and cl2 with SERVER_FOR=g5-2, MODEL_SPECS="100m:$MD5:bf16", LANE_TAG=bf16

( cd neuron && python fanout.py cl1 -f launch.sh --timeout 300 \
    --env REV=$REV --env SPECS="100m:fp32" --env RIG="$RN" --env SERVER_FOR=trn1-1 \
    --env LANE_TAG=fp32 --env RUN=featurewire-log20-prec-g5-trn1-100M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
# and cl2 with SERVER_FOR=trn1-2, SPECS="100m:bf16", LANE_TAG=bf16
```

**`LANE_TAG` IS THE ARM, AND IT IS LOAD-BEARING TWICE.** It keeps the four lane logs from
overwriting each other in S3, and `report.py` reads the precision out of the log's filename
when a log line does not name it -- which is how a round already in flight stays reportable
after a log format changes.

Returns immediately. Each orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds -- so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

| phase | what it does | cost |
|---|---|---|
| `prime` | GPU: one TensorRT engine per (precision, row count), built then **proved** reloadable by a fresh server. Neuron: one NEFF per (precision, row count), compiled or restored, each **proved** loadable and servable, and its trace error checked against the arm's window | 2 shapes per lane |
| `main` | the cells, 5 min warm-up + 15 min measured + 1 min idle, fresh server per cell | 4 per g5 lane, 2 per trn1 lane |

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/featurewire-log20-prec-g5-trn1-100M/cells/ | wc -l
( cd gpu && python fanout.py cl1,cl2 -c 'tail -40 /root/orchestrator-*.log' )
( cd neuron && python fanout.py cl1,cl2 -c 'tail -40 /root/orchestrator-*.log' )
```

Expect **12 cells**. A cell is 5 + 15 + 1 minutes plus up to a minute of minute-alignment
and about 90 s of server start, sampling and upload, so roughly **23 min each** -- about
**1.6 h per g5 lane** and **0.8 h per trn1 lane**, all four in parallel, plus priming
(~2 min per GPU engine, ~3.5 min per NEFF). Check back in hours, not minutes.

## 6 · collect, report, verify

```bash
( cd gpu && ./collect.py )        # -> raw/results-gpu.json
( cd neuron && ./collect.py )     # -> raw/results-neuron.json
./scrub.py --apply                # masks the bucket name in the synced evidence
./report.py                       # writes ../../summary.md from both
./verify.py                       # re-derives every number in it; must print 0 failures
```

**The scrub runs BETWEEN collect and report, and that order is load-bearing.** `collect.py`
syncs the archived server logs and resolved configurations straight out of S3, and those
name the inference log's bucket -- whose name embeds an account id. Reporting before
scrubbing would copy it into `summary.md`. Re-run `scrub.py` after any later `collect.py`.

The Neuron collector also **attaches each NEFF's trace error, SHA-256 and size onto the
cells that ran it**, keyed on (precision, mini batch). The error is measured once per NEFF
at compile and so lives in the prime record, but it is the only evidence of the precision
that does not come from a flag, and a verifier that has to join a cell to a prime document
by hand will eventually join it wrongly.

Each collector takes only its own accelerator's cells, by the `accelerator` field on the
cell rather than by a prefix someone could rename.

`verify.py` re-derives every number in the report AND runs a control test: it builds a
correct synthetic cell, confirms the checks pass it, then mutates it fifteen ways -- the
pre-vectorized wire, logging off, the previous round's rate, a mini batch, four sessions, a
precision outside the two arms, a precision missing from the cell id, a BF16 label with no
BF16 flag in the banner, an FP32 label with one -- and fails if any mutation is NOT caught.
A verifier that cannot fail is not evidence.

## 7 · archive and terminate

```bash
( cd gpu && CONFIRM=terminate python rig.py terminate )
( cd neuron && CONFIRM=terminate python rig.py terminate )
pkill -f refresh_creds.sh
# The logged feature rows are not evidence the report reads -- only the latency they cost
# is -- and they are several hundred GB. The objects' COUNT and SIZE are already recorded
# on every cell, so deleting them loses nothing the report uses.
aws s3 rm "s3://$HUSHAR_BENCH_BUCKET/inference-logs/featurewire-log20-prec-g5-trn1-100M/" \
  --recursive
python ../../../../scripts/check_privacy.py
```

**Do NOT delete `models/neuron/100m/*/`.** Those NEFFs are ~3.5 min each to rebuild and are
the artifacts every figure on that lane rests on; their SHA-256 is recorded per cell.

Run the privacy check before committing: the bucket name carries an account id and must not
reach a tracked file.

## Known limits of this round

- **THE PRECISION AXIS SPANS TWO MACHINES**, one arm per host, for the compile-balance
  reason above. Host-to-host spread in this series is -10.8% to +2.7% against an expected
  BF16 effect near 2x, so the ordering is safe and the magnitude carries that band.
- **THE TWO ARMS ARE OFFERED DIFFERENT RATES**, by instruction, so no median-against-median
  comparison is published. The device time and the sustained throughput are what the report
  compares, and both are stated as such.
- **THE TRAINIUM CELLS ARE EXPECTED NOT TO HOLD THEIR RATES.** They measure the ceiling,
  which nobody had for this model size on this device, and they report achieved rate rather
  than latency. The second wave that earns a latency is not in this round.
- **NO TRAINIUM CONTROL AT THIS MODEL SIZE.** No campaign has measured 100M on Trainium.
  The round that measured this wire, log rate and session count did so at 10M and 20M; it
  was genuinely FP32, so it is comparable in KIND and is what the device-time projection is
  anchored on, but it has no cell at this size to subtract against and the control table
  reports `--` for the Trainium rows. The gap actually worth closing is a BF16 arm at 10M
  and 20M -- see `FOLLOWUP-trn1-fp32-10M-20M.md`.
- **THE A10G CONTROL MOVED TWO THINGS AT ONCE.** `nominibatch-precision-gpu-g5-100M` is the
  same model at the same precisions and rates, but on the tensor wire AND with the log off,
  so a delta against it is the feature path PLUS the log and the two cannot be separated
  within this round.
- **THE TRAINIUM AND A10G GRAPHS ARE NOT THE SAME ARTIFACT.** The A10G serves the ONNX file
  (100,498,437 parameters) through TensorRT; Trainium serves a NEFF that
  `torch_neuronx.trace` rebuilt from the same seeded architecture (100,561,925). Both counts
  are recorded and the report never treats them as one number. What is shared is the
  210-feature wire, the 690-float input and the 5-class output.
- **A SESSION COUNT OF 2 IS PINNED** on the strength of two control rounds, not re-measured.
- **NO CELL IS REPEATED**, so the tie band is inherited rather than measured here.
- **`--auto-cast=all` IS NOT MEASURED.** The BF16 arm casts matmuls only, to correspond with
  TensorRT's `--bf16`. Casting the elementwise and normalization operations too is a further
  step with a known mechanism and no number here.
