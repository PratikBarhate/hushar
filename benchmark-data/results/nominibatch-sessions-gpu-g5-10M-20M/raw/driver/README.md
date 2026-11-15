# hushar with no mini batch against Triton as it ships: the order of operations

Every GPU campaign in this series measured hushar with its mini-batch split engaged -- a
scheduling decision Triton has no counterpart for. A hushar-against-Triton column therefore
priced two servers AND that decision at once. This run removes it:

> **With no mini batch, how does hushar compare with Triton as it ships?**

So every hushar cell scores the whole request in ONE engine call, and the only knob that
moves is how many model copies the server holds.

## The grid

| model | load | hushar `sessions_per_model` | Triton `instance_group` | rows/s |
|---|---:|---|---|---:|
| 10M | 100 x 1000 | 2, 4, 8 | 2, 4, 8 | 100,000 |
| 10M | 500 x 200 | 2, 4, 8 | 2, 4, 8 | 100,000 |
| 20M | 100 x 750 | 2, 4, 8 | 2, 4, 8 | 75,000 |
| 20M | 500 x 150 | 2, 4, 8 | 2, 4, 8 | 75,000 |

12 hushar cells and 12 Triton cells, 24 in total. A row is a PAIR that differs in the
server and nothing else: same GPU, same TensorRT library, same engine shape, same bytes on
the wire, same offered rate, one graph call per request on both sides.

`sessions_per_request` is 1, so `sessions_per_model` IS the requests in flight -- which is
also what a Triton instance count means. That is what makes the pair matched, and it is
the difference from `minibatch-sessions-gpu-g5-10M-20M`, which leased a GROUP of sessions
per request and so held requests in flight at 2 on every rung. **The two runs share cell
geometry and session numbers but not their meaning**, which is why these cell ids carry the
`g5nb` prefix: the id is the CloudWatch `ModelId`, and a shared prefix would make the two
campaigns one time series.

## Read this before spending the GPU hours

**THE SPLIT IS NOT A KNOB HERE, AND CANNOT BE SET.** There is no mini-batch axis in this
driver and no environment variable that reintroduces one -- `cell()` builds the
`whole request` plan unconditionally, and `sweep.py` refuses `RUNGS`, `MINI_BATCHES`,
`MINI_BATCH_SIZE` and `PIN_PARALLELISM` **by name** so a command line copied from the
previous campaign fails loudly instead of quietly measuring something else.

**The unsplit configuration is the ABSENCE of the key, not the key set to the row count.**
`ExecutionPlan::resolve` refuses `mini_batch_size` on a serializing unit when the group is
one session wide and the model demands no row count -- which is TensorRT with a dynamic
batch axis, so exactly this case. `whole request` scores every row in one call, which is
what `mini_batch_size == rows` would have meant anyway.

**The plan is read back out of the banner, not assumed.** `start_hushar` checks the plan
name, checks that no mini-batch line appears at all, and checks the session arithmetic.
`verify.py` re-checks all of it from the recorded banner, and its control test proves a
cell carrying a mini batch would fail.

**One engine shape per row count, not per session count.** A session count loads more
copies of the same engine, so `prime` builds two shapes per size (100 and 500) rather than
six. That is four priming builds for the whole run.

## On matching the TensorRT versions

This run was asked to put Triton and hushar on the same TensorRT. `deploy.sh` extracts
CUDA, cuDNN and TensorRT **out of the Triton image** and hushar links those copies, so both
arms execute on the same library bytes rather than on two builds that agree about a version
string. Three gates:

1. `ldd` on `libonnxruntime_providers_tensorrt.so` at deploy -- any unresolved dependency
   refuses the deploy.
2. `preflight` compares the `libnvinfer` hushar will dlopen against the version the image
   ships, and refuses the lane if they differ.
3. The version is **stamped on every cell of both arms** and `verify.py` fails a lane whose
   two arms recorded different numbers. The claim is therefore checkable from the records,
   not only from a deploy log.

The ONNX Runtime *wrapper* versions cannot match and the report says so rather than
implying otherwise: hushar needs ORT >= 1.28 (`onnxrt-rs` targets `ORT_API_VERSION 28`) and
the image ships 1.24 in its `onnxruntime` backend. Triton therefore runs its **`tensorrt`**
backend, which uses no ONNX Runtime at all.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PROFILE=hushar-bench
export AWS_DEFAULT_REGION=us-west-2
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate local   # boto3
```

The account already carries the bucket, both instance profiles (`hushar-bench-server`,
`hushar-bench-client`) and both security groups. Nothing here creates them.

**Credentials outlive nothing.** A capacity wait can run for hours and these are temporary
credentials. Start the refresher before `launch` and kill it once the sweep is detached --
the hosts run under instance profiles and do not need the laptop's session.

```bash
ACCOUNT=<account-id> nohup bash refresh_creds.sh > /tmp/nominibatch-creds.log 2>&1 &
```

## 0 · prove the reporting before spending the GPU

```bash
./selftest.py          # must print `0 failures`
```

`report.py` and `verify.py` are new code for this run, and the moment they are needed is
after a nine-hour grid has finished. `selftest.py` builds a results tree shaped exactly
like the real one -- from the previous campaign's real cells, rewritten into this run's
geometry -- and requires `verify.py` to report zero failures over it. It caught a real bug:
`nominibatch-...` contains `minibatch-...` as a substring, so the imported-figure check
fired on this run's own name.

## 1 · package

```bash
bash package.sh
# -> REV=..., MD5_10m=..., MD5_20m=...
```

Uploads the working tree as a bundle, the Triton harness, the GPU sampler, the sweep, and
both model directories. Both md5s must match `hosts.py`'s `MODELS` table -- the graphs are
the same ones every GPU campaign in this series used.

`package.sh` also regenerates `bench_model_tensor_config.json` from the real configuration,
which is the wire this run measures: one 690-float f32 vector per row, identity
transformation, so no feature vectorization on the server.

## 2 · launch the rig

```bash
CAPACITY_TRIES=300 CAPACITY_WAIT=120 python rig.py launch
python rig.py status
```

**`g5.2xlarge` is scarce.** Expect to wait; `RequestLimitExceeded` from the fleet API is
throttling rather than capacity and is retried the same way.

**One lane is a normal outcome.** When the zone cannot hold a second GPU, `launch` records
one lane and both sizes run on it one after the other. That is *better* for the 10M-20M
comparison; it only costs wall clock.

## 3 · keys

```bash
python keys.py
```

One ephemeral ed25519 keypair per lane, over SSM. The laptop reaches every host over SSM;
the orchestrator runs *inside* the VPC and needs SSH, which this installs.

## 4 · deploy

```bash
python fanout.py servers,clients -f deploy.sh --timeout 5400 \
  --env REV=$REV \
  --env MODEL_SPECS="10m:$MD5_10m 20m:$MD5_20m" \
  --env ROWS_LIST="100 500" \
  --env RUN=nominibatch-sessions-gpu-g5-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

Server deploy, in order: toolchain; both graphs, md5-checked; docker and the Triton image;
**read the image's own TensorRT and CUDA versions**; extract CUDA, cuDNN and TensorRT out
of it; download the ONNX Runtime GPU build matching *that* CUDA major; `ldd` the TensorRT
provider against the extracted libraries and **refuse** on any unresolved dependency; write
`env.sh`; `cargo build --features cpu,tensorrt`; build one `trtexec` plan per row count;
record provenance to S3.

Both row counts are staged on **both** servers, which is what makes a lane interchangeable
if one has to be re-driven.

## 5 · run

```bash
R=$(python -c "import hosts,json;print(json.dumps(hosts.rig_env()))" | tr -d '"')

python fanout.py cl1 -f launch.sh --timeout 300 \
  --env REV=$REV \
  --env MODEL_SPECS="10m:$MD5_10m 20m:$MD5_20m" \
  --env RIG="$R" --env SERVER_FOR=g5-1 \
  --env RUN=nominibatch-sessions-gpu-g5-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

On a two-lane rig, give `cl2` `SERVER_FOR=g5-2` and `MODEL_SPECS="20m:$MD5_20m"`, and give
`cl1` only `10m`.

Returns immediately. The orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds -- so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

Two phases. There is **no probe phase**: every rung states its own session count, so there
is nothing to rank.

| phase | what it does | cost |
|---|---|---|
| `prime` | one TensorRT engine per **row count** (100, 500), each in its own cache directory, each built and then **proved** reloadable by a fresh server, on the wire the cells use | 2 shapes x 2 sizes |
| `main` | the 24 cells, 5 min warm-up + 15 min measured + 1 min idle, fresh server per cell | 12 cells per size |

A single phase can be run alone with `--env PHASE=prime`.

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/nominibatch-sessions-gpu-g5-10M-20M/cells/ | wc -l
python fanout.py cl1 -c 'tail -40 /root/orchestrator-main-*.log'
```

A main cell is 5 + 15 + 1 minutes plus up to a minute of minute-alignment and about 90 s of
server start, sampling and upload, so roughly **23 min each** -- about **4.6 h of cells per
size**, so **9.2 h on a one-lane rig**, plus priming. Check back in hours, not minutes.

## 6 · collect, report, verify

```bash
./collect.py            # sync from S3 and enrich with host, GPU and CloudWatch data
./report.py             # writes ../../../results/<run>/summary.md
./verify.py             # re-derives every number in it; must print 0 failures
```

## 7 · archive and terminate

```bash
CONFIRM=terminate python rig.py terminate
pkill -f refresh_creds.sh
python ../../../../scripts/check_privacy.py
```

Destroys the instances, their root volumes and the capacity launch templates. Everything a
report needs is already in S3. Run the privacy check before committing: the bucket name
carries an account id and must not reach a tracked file.

## Known limits of this rig

- **A session count is not a free variable on 8 vCPUs.** At 8 sessions and one intra-op
  thread each, plus 4 async workers, the host is oversubscribed. The banner's `note` line
  records it and the host CPU column measures it -- but a hushar cell at 8 sessions is
  therefore competing for cores in a way the Triton cell beside it is not, because Triton's
  instances share one process and its own thread pool. The pairing is matched on *requests
  in flight*, not on host threads, and the report does not claim otherwise.
- **No cell is repeated**, so the tie band is inherited rather than measured here. The
  first GPU campaign measured host-to-host repeatability at -10.8% to +2.7%.
- **FP32 only.** Both stacks take TensorRT's default builder settings, so every number is a
  floor rather than a ceiling.
- **The tensor wire is 2.54x the bytes of hushar's own feature wire** -- 690 floats per row
  against 210 named features. At the 10M rungs that is ~276 MB/s inbound, near a
  `g5.2xlarge`'s sustained network baseline, for 15 minutes. If a network ceiling is
  reached it shows as shedding in the later slices, which is why the hold table reports
  slices rather than one aggregate.
- **This run prices the servers, not the APIs.** Both arms receive the finished 690-float
  tensor, so hushar's feature path is not exercised and `VectorizationTime` is near zero by
  construction. What a Triton caller would pay to build that tensor is outside both
  numbers.
