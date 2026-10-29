# GPU sweep runbook: hushar against Triton on one NVIDIA A10G

What this measures, and the one idea that makes it fair, are in the docstrings of
`hosts.py` and `sweep.py`. This file is the order of operations.

The short version: one A10G, one ONNX graph, one TensorRT. hushar scores it through ONNX
Runtime's TensorRT execution provider; Triton scores it through its own `tensorrt` backend
from a plan `trtexec` built. **hushar is given the TensorRT libraries copied out of the
Triton image**, so the engine on both sides is built by the same compiler — that is the
difference between comparing two servers and comparing two accelerator stacks.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
conda activate local                     # boto3
```

The account already carries the bucket, both instance profiles
(`hushar-bench-server`, `hushar-bench-client`) and both security groups. Nothing here
creates them.

## 1 · package

```bash
./package.sh
# -> REV=..., MD5_10m=..., MD5_20m=...
```

Uploads the working tree as a bundle (so an uncommitted fix is testable), the Triton
harness, the GPU sampler, the sweep, and both model directories. Derives each size's
tensor feature spec from its model configuration rather than trusting a checked-in copy.

## 2 · launch the rig

```bash
python rig.py launch          # 2 x g5.2xlarge + 2 x m9g.8xlarge, one AZ
python rig.py status
```

Both images are resolved at launch and age-checked: the GPU servers take the newest
*Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)*, the clients the
published Amazon Linux alias. A pinned AMI is how an earlier rig went stale.

**`g5.2xlarge` is scarce.** The first attempt at this run died on
`InsufficientInstanceCapacity` in all four us-west-2 zones. `launch` therefore asks for the
first GPU through an `instant` EC2 Fleet spanning every subnet — one request over the whole
set, rather than four requests that each lose the zone the previous one was told about —
and retries on a timer:

```bash
CAPACITY_TRIES=200 CAPACITY_WAIT=120 python rig.py launch   # ~6.7 h of patience
AZ=us-west-2c python rig.py launch                          # wait on one zone
```

Whichever zone the first GPU lands in is where the **whole rig** goes: a client in one
zone and its server in another would add a network hop to one lane and not the other. If
that zone cannot hold a *second* GPU, the run continues on **one lane** and measures both
model sizes on it, one after the other — see step 5.

To find out what the account can actually get right now, without guessing:

```bash
python probe_capacity.py            # asks for one of each single-GPU type, terminates it
```

## 3 · keys

```bash
python keys.py
```

One ephemeral ed25519 keypair per lane, over SSM. The laptop reaches every host over SSM;
the orchestrators run *inside* the VPC and need SSH, which this installs.

## 4 · deploy

Per lane, because each host only needs its own model and only builds that model's
TensorRT plan — building both would double the longest step for nothing.

```bash
R=$(python -c "import hosts,json;print(json.dumps(hosts.rig_env()))" | tr -d '"')

python fanout.py g5-1,cl1 -f deploy.sh --timeout 3600 \
  --env REV=$REV --env MODEL_SIZE=10m --env MODEL_MD5=$MD5_10m \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET &
python fanout.py g5-2,cl2 -f deploy.sh --timeout 3600 \
  --env REV=$REV --env MODEL_SIZE=20m --env MODEL_MD5=$MD5_20m \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET &
wait
```

The server deploy, in order: toolchain; the graph, md5-checked; docker and the Triton
image; **read the image's own TensorRT and CUDA versions**; extract CUDA, cuDNN and
TensorRT out of it; download the ONNX Runtime GPU build matching *that* CUDA major;
`ldd` the TensorRT provider against the extracted libraries and **refuse** on any
unresolved dependency; write `env.sh`; `cargo build --features cpu,tensorrt` natively;
build the plan with `trtexec`; record provenance to S3.

The `ldd` gate is the important one. It turns "ONNX Runtime and Triton disagree about
TensorRT" from a confusing first-cell failure into a deploy that stops and says so.

## 5 · run

```bash
python fanout.py cl1 -f launch.sh --timeout 300 \
  --env REV=$REV --env MODEL_SIZE=10m --env MODEL_MD5=$MD5_10m \
  --env RIG="$R" --env SERVER_FOR=g5-1 --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
python fanout.py cl2 -f launch.sh --timeout 300 \
  --env REV=$REV --env MODEL_SIZE=20m --env MODEL_MD5=$MD5_20m \
  --env RIG="$R" --env SERVER_FOR=g5-2 --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

Returns immediately. The orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds — so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

Phases run in order: `prime` (compile the TensorRT engine into a cache and prove a fresh
server reloads rather than rebuilds it), `probe` (parallelism 1/2/4 per stack at the
heaviest load), `main` (the grid at the chosen parallelism). One can be run alone with
`--env PHASE=probe`.

On a **one-lane** rig, run the same command twice on `cl1`, once per size, the second
after the first finishes:

```bash
... --env MODEL_SIZE=10m --env MODEL_MD5=$MD5_10m --env SERVER_FOR=g5-1
... --env MODEL_SIZE=20m --env MODEL_MD5=$MD5_20m --env SERVER_FOR=g5-1
```

Nothing in the driver needs changing: `MODEL_SIZE` is an input and S3 resume makes the
second invocation additive.

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/config-sweep-gpu-g5-wire-10M-20M/cells/ | wc -l
python fanout.py clients -c 'tail -20 /root/orchestrator-main-*.log'
```

Expect, per lane: 6 probes, then 8 cells (10M) or 4 cells (50M). A main cell is
2 min warm-up + 5 min measured + 1 min idle plus about 90 s of server start, sampling and
upload, so roughly 9.5 min each.

## 6 · collect, report, verify

```bash
./collect.py            # sync from S3 and enrich with host, GPU and CloudWatch data
./report.py             # writes ../../../results/<run>/summary.md
./verify.py             # re-derives every number in it; must print 0 failures
```

`verify.py` is not a formality. It parses each table back out of the generated file and
recomputes it from `raw/`, re-applies the sustained/partial/failed rule to the slices,
re-applies the parallelism selection rule to the probes, re-reads the Trainium comparison
from that run's own `results.json`, and asserts the structural claims — that every hushar
cell's banner named `TensorRT(device=0`, that every Triton cell was served by the
`tensorrt` backend with `max_batch_size: 0`, that one TensorRT version covers both hosts.

## 7 · terminate

```bash
CONFIRM=terminate python rig.py terminate
python ../../../../scripts/check_privacy.py
```

Destroys the instances, their root volumes and the capacity launch templates, and clears
the recorded ids. Everything a report needs is already in S3 — cells as they finished,
provenance at deploy — so nothing is lost. Run the privacy check before committing: the
bucket name carries an account id and must not reach a tracked file.

## Known limits of this rig

- **FP32 only.** Both stacks take TensorRT's default builder settings. An A10G's FP16 and
  INT8 tensor cores are most of what the device is for, so every number here is a floor.
  `make_trt_repo.py` already takes `FP16=1`, and hushar's provider string takes `fp16=1`;
  a follow-up pass needs both set together or the comparison breaks.
- **The wires differ.** Triton's plan takes the finished tensor, so the 210
  transformations move to its caller. The report prices that from hushar's own
  `VectorizationTime` as a lower bound. A tensor-wire hushar arm would close it exactly;
  the 10M CPU campaign already showed the graph cost is identical on both wires.
- **One pair per size, no repeatability floor.** Comparison groups never cross hosts, and
  no cross-host difference is claimed.
- **The plan is keyed to 100 rows.** `min = opt = max = [100, 690]`. A row count other
  than 100 needs a new plan, and `start_triton` deliberately fails a cell whose plan was
  not already built rather than building it inside the measurement.
