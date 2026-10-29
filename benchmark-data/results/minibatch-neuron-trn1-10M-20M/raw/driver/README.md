# Trainium mini-batch and replica runbook

What this measures is in the docstrings of `hosts.py` and `sweep.py`. This file is the
order of operations, plus the two things that cost this run most of its setup time and
would cost the next one the same.

The short version: one `trn1.2xlarge`, two NeuronCores, **one core per replica
throughout**, and two things change down the grid — how many rows go into one NEFF call
(`mini_batch_size`) and how many replicas serve (`sessions_per_model`).
[`minibatch-gpu-g5-10M-20M`](../../../results/minibatch-gpu-g5-10M-20M/summary.md) asked
the mini-batch question on one A10G, found it bought nothing, and named the knob it
lacked: `execution_contexts_per_replica`, which exists only on the Neuron backend. This
run asks the same question on the device that has it.

## Read this before spending the Trainium hours

**`cores_per_replica = 1` bounds the answer, by instruction.** A NeuronCore executes one
NEFF at a time, so all of a request's pieces run on the one core its replica owns and the
split's *device* term cannot improve. Two terms can still move:

| term | what splitting does to it | why |
|---|---|---|
| device | can only get **worse** | N executions for the same arithmetic, at a smaller M |
| vectorize | can get **better** | the 210 transformations fan out on a rayon pool sized to `num_cpus` (8) |
| DMA | can get better, **but only with contexts > 1** | a second tensor set lets one piece's host-device copy overlap another's execution |

`config-sweep-neuron-trn1-10M-50M` is the evidence for the bound: its 41% cut on 50M came
from `s1 · k2 · c2` — **one replica spanning two cores** — not from the split itself. The
pieces won there because they reached two *different* cores. A negative result here is
therefore a measurement and not a wasted lane, but the report must not present it as "mini
batching does not work on Trainium" when what it measured is "mini batching does not work
on **one core**".

**Contexts track the piece count**, which is hushar's own rule, printed at startup:

```
note   : a replica serves one request at a time, so
         "backend.execution_contexts_per_replica" is worth exactly the executions one
         request produces -- fewer serialize the request, more stay idle
```

So `contexts = ceil(rows / mini_batch)`, which is 1 for every unsplit cell — the baselines
are unchanged and the mini batch stays a one-variable subtraction. The `control` phase then
re-runs the largest split of each load at `contexts = 1` so that choice is priced rather
than assumed.

**The rates were specified, not probed, and the 20M ones are expected to be over the
ceiling.** Projecting from the two measured points in `config-sweep-neuron-trn1-10M-50M`
(1.644 ms per 100-row call at 10M, 10.99 ms at 50M, linear in parameters), 20M lands near
4.1 ms, so two serial cores cap near 485 req/s against the 750 asked and near 97 against
the 150 asked at 500 rows. Cells that do not hold their rate are reported by **achieved
rate** and enter no latency comparison. They still measure the ceiling, which is the figure
nobody had for 20M on this device.

## Why this run is FP32, and what that leaves open

The 20M rungs turn out to be **device-bound**, not host-bound: a NeuronCore executes one
NEFF at a time, a `trn1.2xlarge` has two, and 20M at 100x750 needs
`750 x 3.859 ms = 2.89` serial lanes. The measured ceiling tracks
`1.90 / device_span` to within a percent.

That makes precision the obvious lever — Trainium1's systolic array is bf16-native and
FP32 is the slower path on it, so `neuronx-cc --auto-cast=matmult --auto-cast-type=bf16`
would plausibly buy the 34% of device span needed to clear 750. **It was built, launched,
and then deliberately abandoned**, because every other run in this series is FP32: a cast
arm is not comparable with any of them and a number from it could not sit in the same
table. It stays an open follow-up with a known mechanism rather than a figure nobody can
read against anything.

The plumbing was removed rather than left dormant, so the diff to the tracked
`compile_neuron.py` is the two genuine bug fixes and nothing exploratory. If it is wanted
later it is a `compiler_args` entry in `compile_model`, a stem suffix, and a cell-id
suffix — and the NEFF must get its own S3 cache key, or a cast artifact will be restored
for an FP32 cell.

One thing the abandoned attempt did prove: **publishing a NEFF only after a server has
served it is the right ordering.** The bf16 lane was killed mid-compile and nothing
reached `models/neuron/`, so the cache held only artifacts that had been shown to work.

## Two findings that cost this run its setup time

### 1 · The server image is pinned to PyTorch 2.9, and that is not conservatism

Neuron does not compile ONNX, so this run builds its own NEFFs, and hushar's backend serves
a NEFF taking **one** input — the `[batch, 690]` feature tensor — with the parameters
compiled in as constants. `torch_neuronx.trace` produces exactly that.

The current `torch-neuronx 2.14` DLAMI (SDK 2.33) was tried first and **cannot be used**:

* `torch_neuronx.trace` is gone from it;
* `torch_neuronx.compiler.fx_to_stablehlo` and `fx_to_neff` both die in torch-mlir's
  importer on this model, because `torch.export(...).module()` lifts parameters to graph
  inputs and the importer then looks for `to_tokens.weight` as an attribute;
* `torch.compile(model, backend="neuron")` lowers and runs the model correctly — 2.7e-07
  against CPU PyTorch — but writes **no model NEFF to disk**, only a 338 KiB NKI kernel;
* and the graph it lowers is **stateless**: `aot_module_simplified` makes every parameter a
  graph input, so a NEFF built from its StableHLO has a weight tensor where hushar expects
  features. hushar refuses it with `NEFF input "input0" fixes its batch at 4096`.

`Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)` carries torch_neuronx
`2.9.0.2.15.32035+de43f57c` — byte-identical to the string
`config-sweep-neuron-trn1-10M-50M` recorded — and reproduces that campaign's artifacts
exactly: 10M b50 at 32.5 MiB with trace error 1.3e-06, b100 at 32.7 MiB with 1.43e-06, both
matching its table. So this is the toolchain that makes a NEFF here *comparable* rather than
merely newer.

**Disclosed cost:** that image is older than `rig.py`'s 45-day freshness limit, so launching
needs `AMI_ALLOW_STALE=1`. The limit exists because a long-lived rig was once flagged for an
outdated OS; these hosts are ephemeral and the alternative is a NEFF the service cannot
load. The report states the image and its age.

### 2 · SSM runs scripts under `dash` on Ubuntu

`AWS-RunShellScript` executes its commands with `/bin/sh`, which is **dash** on the Ubuntu
Neuron DLAMI and **bash** on the Amazon Linux clients. So `set -o pipefail` died on arrival
on every server (`Illegal option -o pipefail`) while the clients were fine — the two roles
disagreeing about one payload. `fanout.py` now writes the body to a file and runs it with
`/bin/bash` explicitly. It is written to a file rather than piped to `bash -s` because a
script read from stdin owns stdin, and a deploy runs package managers and installers.

Three smaller ones, each a one-line fix that reads as something else when it bites:

* the Ubuntu DLAMI roots at `/dev/sda1`, not `/dev/xvda`. A block device mapping naming the
  wrong one is not an error — it attaches a *second* disk and leaves the instance on the
  image's own 80 GB root. `rig.py` reads the device name out of `DescribeImages`.
* `import torch_neuronx` spawns `libneuronpjrt-path` from the venv's `bin`, so the venv must
  be on `PATH` *before* the import, not just before the compile. Without it the import
  fails with `FileNotFoundError` and a working venv is rejected.
* the venv name changed between images (`aws_neuronx_venv_pytorch_2_9` here,
  `aws_neuronx_venv_torch_neuronx_2_14_0` on SDK 2.33). `deploy.sh` discovers it by what it
  can **import** and whether it **has `trace`**, not by name.

`scripts/benchmark/compile_neuron.py` also had two live bugs before any of this: it called
`vectorized_config` with the pre-`model_id` signature, and it had never been run against a
torch-neuronx without `trace`. `compile_neuron-fix.patch` is the diff.

## Prerequisites

```bash
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PROFILE=hushar-bench
conda activate local                     # boto3
```

The account already carries the bucket, both instance profiles (`hushar-bench-server`,
`hushar-bench-client`) and both security groups. Nothing here creates them.

Credentials are **conduit**, not Isengard — this is a Gated Garden account, so
`ada ... --provider=isengard` fails with `AWSAccountNotFoundException`:

```bash
ada credentials update --account=$(cat ../../../../.account-ids) \
  --provider=conduit --role=IibsAdminAccess-DO-NOT-DELETE \
  --profile=hushar-bench --once
```

A long run outlives one credential session. The lanes do not care — they run on the hosts
under instance profiles, which refresh themselves — but the laptop's `fanout.py` will
start returning `SendFailed` with `exit=None`, which is what an expired token looks like
from here rather than anything about SSM.

`collect.py` pins `/opt/homebrew/bin/aws` for the same class of reason: run under
`conda run -n local`, a bare `aws` resolves to the conda environment's own awscli, which
fails on import with `KeyError: 'opsworkscm'`.

## 1 · package

```bash
bash package.sh
# -> REV=...
```

Uploads the working tree as a bundle (so an uncommitted fix is testable), the NeuronCore
sampler and the sweep. **No models travel**: there is no graph to pin by md5, because the
artifact a cell runs is a NEFF the host builds.

## 2 · launch the rig

```bash
AMI_ALLOW_STALE=1 CAPACITY_TRIES=60 CAPACITY_WAIT=90 python rig.py launch
python rig.py status
```

**`trn1.2xlarge` is offered only in `us-west-2b` and `us-west-2d`** — the other zones
return `Unsupported` rather than a capacity error, so only the real two are listed. All
four hosts land in whichever zone can hold the first server, because a cross-zone client
would add a network hop to one lane and not the other.

`launch` **adopts** hosts it already owns, found by tag. That is also how a lost
`rig-state.json` is rebuilt — which was needed once in this run, when
`benchmark-data/private/` was deleted out from under a live rig. Nothing was relaunched.

## 3 · keys

```bash
python keys.py
```

One ephemeral ed25519 keypair per lane, over SSM. The laptop reaches every host over SSM;
the orchestrator runs *inside* the VPC and needs SSH, which this installs.

## 4 · deploy

```bash
python fanout.py all -f deploy.sh --timeout 3600 \
  --env REV=$REV --env SIZES="10m 20m" \
  --env RUN=minibatch-neuron-trn1-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

Server: toolchain, the Neuron device and runtime verified, the venv located by what it
imports, `onnx` added to it, `cargo build` with the Neuron backend, the sampler. Client:
toolchain and `benchmark-client`. **The deploy builds no models** — see `prime`.

## 5 · run

```bash
R=$(python -c "import hosts;print(hosts.rig_env())")

python fanout.py cl1 -f launch.sh --timeout 300 \
  --env REV=$REV --env SIZES="10m" --env RIG="$R" --env SERVER_FOR=trn1-1 \
  --env RUN=minibatch-neuron-trn1-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET

python fanout.py cl2 -f launch.sh --timeout 300 \
  --env REV=$REV --env SIZES="20m" --env RIG="$R" --env SERVER_FOR=trn1-2 \
  --env RUN=minibatch-neuron-trn1-10M-20M \
  --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET
```

Returns immediately. The orchestrator is detached and its own session leader, every cell
goes to S3 the moment it finishes, and re-running a lane skips what S3 already holds — so
**the laptop can disconnect and stay disconnected**, and a resume is the same command.

Three phases, in order:

| phase | what it does | cost |
|---|---|---|
| `prime` | one NEFF per mini-batch size, **compiled or restored from S3**, each then PROVED loadable and servable by a fresh server | 5 shapes per size |
| `main` | the grid at `contexts = pieces`, 2 min warm-up + 15 min measured + 1 min idle, fresh server per cell | 18 cells per size |
| `control` | the largest split of each load at `contexts = 1` | 2 cells per size |

A NEFF is cached in S3 **only after a server has served it**. The first version of this
phase published straight after the compile, and a 338 KiB kernel NEFF that hushar could not
load was cached where every later lane and resume would restore it — a failure that
outlives the host that caused it. `prime` also refuses a NEFF smaller than half the FP32
weight size, which catches that case in one line instead of inside a cell.

A single phase can be run alone with `--env PHASE=prime`.

### watching

```bash
aws s3 ls s3://$HUSHAR_BENCH_BUCKET/runs/minibatch-neuron-trn1-10M-20M/cells/ | wc -l
python fanout.py cl1,cl2 -c 'tail -20 /root/orchestrator-main-*.log'
```

Expect **20 cells per lane**. A main cell is 2 + 15 + 1 minutes plus about 60 s of server
start, sampling and upload, so roughly **19 min each** — about **6.5 h per lane**, both
lanes in parallel. Check back in hours, not minutes.

## 6 · collect, report, verify

```bash
./collect.py            # sync from S3, enrich with host, NeuronCore and CloudWatch data
./report.py             # writes ../../../results/<run>/summary.md
./verify.py             # re-derives every number in it; must print 0 failures
```

## 7 · archive and terminate

```bash
CONFIRM=terminate python rig.py terminate
python ../../../../scripts/check_privacy.py
```

Destroys the instances, their root volumes and the capacity launch templates. Everything a
report needs is already in S3. Run the privacy check before committing: the bucket name
carries an account id and must not reach a tracked file.

## Known limits of this rig

- **One core per replica, by instruction.** So this run cannot say what a split does when
  the pieces reach *different* cores — which is the configuration the prior campaign found
  worth 41%. That is a different experiment, and `CORES_PER_REPLICA` is named in
  `launch.sh` so it is a decision rather than a default.
- **The rates are specified, not probed.** Two of the four 20M rungs are projected to be
  above the ceiling, so they are expected to report capacity rather than latency.
- **`neuroncore_utilization` is occupancy, not saturation** — the share of a sample period
  with work resident, exactly like `nvidia-smi`'s `utilization.gpu`. A split produces a
  stream of smaller executions and so reads high on a core with headroom. The column says
  the core was busy and never how full it was.
- **No cell is repeated**, so there is no same-host spread measured here; a tie band has to
  be inherited from a campaign that did measure one.
- **The 20M model has no Trainium precedent**, so its parameter count is checked against
  the generator rather than against a prior run. 10M is checked against
  `config-sweep-neuron-trn1-10M-50M`'s 9,142,789 and `prime` refuses on a mismatch.
