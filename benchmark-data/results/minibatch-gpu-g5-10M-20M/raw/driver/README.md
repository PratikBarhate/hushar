# GPU mini-batch runbook: does splitting a request help on one NVIDIA A10G?

What this measures, and the one thing that bounds the answer before any cell runs, are in
the docstrings of `hosts.py` and `sweep.py`. This file is the order of operations.

The short version: one A10G, one ONNX graph, one TensorRT, and the only thing that changes
down a group of cells is **how many rows go into one graph call**. `mini_batch_size` equal
to the request's rows is one call and is the baseline; anything smaller cuts the request
into pieces handed to a thread pool and scored at once.
[`minibatch-rows-cpu-arm64-10M`](../../results/minibatch-rows-cpu-arm64-10M/summary.md)
found that knob worth up to 2.64x on a 64-core CPU.

## Read this before spending the GPU hours

The split's two terms move in **opposite** directions here, and one of them is bounded by
hushar's own architecture rather than by the device. From `main.rs`:

```rust
let session = model.sessions.acquire();          // ONE session, per REQUEST
inference::scoring::score_features(session.as_ref(), …, mini_batch, …)
```

Every mini batch of one request calls `run_named()` on **that same session**, and the
TensorRT execution provider holds a per-session mutex around enqueue, so the pieces
**queue on the device**. `execution_contexts_per_replica` — the knob that would give each
piece its own device context — exists only on the Neuron backend.

| term | what splitting does to it | why |
|---|---|---|
| graph | can only get **worse** | N kernel launches and N host-device copies for the same arithmetic, at a smaller M |
| vectorize | can get **better** | the 210 transformations are host work on a rayon pool sized to `num_cpus` |

So the question is about **shares**, and the grid is built around that:

| model | load | mini batches | pieces | why this rung |
|---|---:|---|---:|---|
| 10M | 100 x 1000 | 100, 50, 25 | 1, 2, 4 | heaviest 100-row load 10M sustained; vectorize is ~9% of the median, so a win is arithmetically out of reach |
| 10M | 500 x 200 | 500, 250, 100 | 1, 2, 5 | same 100,000 rows/s in fifth as many requests; vectorize is ~a quarter of the median, so a win is possible |
| 20M | 100 x 750 | 100, 50, 25 | 1, 2, 4 | heaviest 100-row load 20M sustained |
| 20M | 500 x 150 | 500, 250, 100 | 1, 2, 5 | same 75,000 rows/s, fifth as many requests |

Holding rows/s constant across the two packagings also answers, for free, the question a
GPU actually rewards: whether a **bigger** request is cheaper per candidate. On a CPU the
lever is to split a request to win cores; on a GPU it is to merge requests to fill the
device. Opposite directions.

Every split divides exactly, so `is_fixed` pads nothing and each cell presents the engine
exactly one shape.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PROFILE=hushar-bench
conda activate local                     # boto3
```

The account already carries the bucket, both instance profiles
(`hushar-bench-server`, `hushar-bench-client`) and both security groups. Nothing here
creates them.

## 1 · package

```bash
bash package.sh
# -> REV=..., MD5_10m=..., MD5_20m=...
```

Uploads the working tree as a bundle (so an uncommitted fix is testable), the Triton
harness, the GPU sampler, the sweep, and both model directories. Both md5s must match
`hosts.py`'s `MODELS` table — the graphs are the same ones the prior GPU campaign used.

## 2 · launch the rig

```bash
CAPACITY_TRIES=300 CAPACITY_WAIT=120 python rig.py launch
python rig.py status
```

**`g5.2xlarge` is scarce, and it is only offered in `us-west-2a`, `2b` and `2c`** — `2d`
returns `Unsupported`, which the loop tolerates. Expect to wait; the first attempt at this
run waited through several rounds and then got only ONE of the two GPUs.

**One lane is the normal outcome, not a degraded one.** When the zone cannot hold a second
GPU, `launch` records one lane and both model sizes run on it one after the other. That is
*better* for the 10M↔20M comparison, since it puts both sizes on one machine; it only
costs wall clock. Nothing in the driver needs changing — `MODEL_SPECS` carries both sizes
into a single detached orchestrator.

## 3 · keys

```bash
python keys.py
```

One ephemeral ed25519 keypair per lane, over SSM. The laptop reaches every host over SSM;
the orchestrator runs *inside* the VPC and needs SSH, which this installs.

## 4 · deploy

```bash
python fanout.py g5-1,cl1 -f deploy.sh --timeout 5400 \
  --env REV=$REV \
  --env MODEL_SPECS="10m:$MD5_10m 20m:$MD5_20m" \
  --env ROWS_LIST="100 500" \
  --env RUN=minibatch-gpu-g5-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

`ROWS_LIST` is the one argument this run adds. Triton's plan pins the row dimension, so a
lane measuring two packagings needs **two plans per graph** — four here. `sweep.py` counts
them in `preflight` and refuses a cell whose plan was not already built, because building
one inside a cell would make the measurement include `trtexec`.

The server deploy, in order: toolchain; both graphs, md5-checked; docker and the Triton
image; **read the image's own TensorRT and CUDA versions**; extract CUDA, cuDNN and
TensorRT out of it; download the ONNX Runtime GPU build matching *that* CUDA major; `ldd`
the TensorRT provider against the extracted libraries and **refuse** on any unresolved
dependency; write `env.sh`; `cargo build --features cpu,tensorrt`; build the plans; record
provenance to S3.

The `ldd` gate is the important one. It turns "ONNX Runtime and Triton disagree about
TensorRT" from a confusing first-cell failure into a deploy that stops and says so.

### on matching ONNX Runtime versions

This run was asked to match Triton's ONNX backend version to hushar's. **It is not
satisfiable and the report says so rather than implying otherwise:**

- hushar needs ORT **≥ 1.28** — `onnxrt-rs` targets `ORT_API_VERSION 28`, and a 1.24
  library cannot satisfy it.
- `tritonserver:26.05-py3` ships ORT **1.24** in its `onnxruntime` backend.

Triton therefore runs its **`tensorrt`** backend, as the prior GPU campaign did, which
does not use ONNX Runtime at all. What **is** matched is **TensorRT** — the library that
actually executes the graph — because hushar links the copies extracted from that same
image and the `ldd` gate refuses anything else. That is strictly stronger than matching
the ORT wrapper around it.

## 5 · run

```bash
R=$(python -c "import hosts,json;print(json.dumps(hosts.rig_env()))" | tr -d '"')

python fanout.py cl1 -f launch.sh --timeout 300 \
  --env REV=$REV \
  --env MODEL_SPECS="10m:$MD5_10m 20m:$MD5_20m" \
  --env RIG="$R" --env SERVER_FOR=g5-1 \
  --env RUN=minibatch-gpu-g5-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

Returns immediately. The orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds — so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

Two phases, in order. There is **no probe phase**: `sessions_per_model` is pinned per size
(10M → 2, 20M → 4) from the prior campaign's own selection, because holding it fixed is
what makes the mini-batch column a one-variable subtraction.

| phase | what it does | cost |
|---|---|---|
| `prime` | one TensorRT engine per **shape** (25, 50, 100, 250, 500), each in its own cache directory, each built and then **proved** reloadable by a fresh server | ~5 shapes x 2 sizes |
| `main` | the grid at the pinned parallelism, 2 min warm-up + 15 min measured + 1 min idle, fresh server per cell | 8 cells per size |

A single phase can be run alone with `--env PHASE=prime`.

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/minibatch-gpu-g5-10M-20M/cells/ | wc -l
python fanout.py cl1 -c 'tail -30 /root/orchestrator-main-*.log'
```

Expect 16 cells on a one-lane rig. A main cell is 2 + 15 + 1 minutes plus about 90 s of
server start, sampling and upload, so roughly **19.5 min each** — about **5.5 h** of cells
per lane, plus priming. Check back in hours, not minutes.

## 6 · collect, report, verify

```bash
./collect.py            # sync from S3 and enrich with host, GPU and CloudWatch data
./report.py             # writes ../../../results/<run>/summary.md
./verify.py             # re-derives every number in it; must print 0 failures
```

`verify.py` is not a formality. Beyond re-deriving every table it checks the things a
mini-batch run specifically can get wrong:

- **the split was real** — each hushar cell's `mini_batch` is checked against its id, its
  engine cache path, its engine shape, its piece count *and the server banner it recorded
  at startup*. A cell that silently ran unsplit would look exactly like a split that
  bought nothing, which is this run's expected answer and therefore the one way the report
  could lie and be believed;
- **direction and magnitude separately** on every comparison, because a verifier that only
  checks the number cannot catch a sign flip;
- **the baseline is the unsplit cell of the same load** — not the largest split, not
  another row count, not another run;
- **nothing was imported** from `config-sweep-gpu-g5-10M-20M`, whose window was 300 s
  against this run's 900 s;
- **an unsustained cell appears in no comparison**, with the `sustained` verdict re-applied
  from the slices rather than trusted;
- and a **control**: a fabricated row must make the gain check fail. A verifier nobody has
  seen fail is a verifier nobody has tested.

## 7 · archive and terminate

```bash
CONFIRM=terminate python rig.py terminate
python ../../../../scripts/check_privacy.py
```

Destroys the instances, their root volumes and the capacity launch templates. Everything a
report needs is already in S3. Run the privacy check before committing: the bucket name
carries an account id and must not reach a tracked file.

## Known limits of this rig

- **The split's device term is decided by the server, not by the mini-batch size.** One
  session is leased per request and shared across its pieces, so this run measures the
  split *as shipped*. It does not measure what a split could do with per-piece context
  leasing, which is the change that would make the experiment different rather than the
  mini-batch size.
- **`graph` on a split cell is the MAXIMUM over the pieces**, not their sum — the service
  reports one duration per stage per request. So it is neither total device work nor a
  clean per-piece figure, and the report withholds the `everything else` subtraction on
  split rows rather than printing a number that does not hold.
- **No cell is repeated, so the tie band is inherited** (8.4%, the same-host spread the CPU
  mini-batch campaign measured on a split configuration). Two medians inside it are
  reported as `level`.
- **FP32 only.** Both stacks take TensorRT's default builder settings, so every number is
  a floor rather than a ceiling.
- **The wires differ**, as in every report in this series: Triton's plan takes the finished
  tensor, so the 210 transformations move to its caller. hushar's own `VectorizationTime`
  prices that as a lower bound.
