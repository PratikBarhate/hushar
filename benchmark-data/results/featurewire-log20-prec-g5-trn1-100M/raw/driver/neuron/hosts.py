#!/usr/bin/env python3
"""The rig: two `trn1.2xlarge` server hosts, one client each, LAUNCHED fresh for this run.

This is the Trainium lane of `featurewire-log20-prec-g5-trn1-100M`, whose A10G lane runs
the same grid on an NVIDIA A10G under the same run name.

    ONE MODEL, ONE HUNDRED MILLION PARAMETERS, TWO PRECISIONS.

THE AXIS IS PRECISION, AND ON TRAINIUM IT IS PASSED AS AN EXPLICIT COMPILER FLAG.
`neuronx-cc`'s CLI reference gives `--auto-cast` a default of `matmult`, which would mean a
compile passing no flag casts FP32 matrix multiplies to BF16 -- nearly all of a
transformer's arithmetic. **THAT DEFAULT DOES NOT REACH THIS CODE PATH.**
`torch_neuronx.trace` builds its own argument list, and a measured three-way compile on
this host type produced a BYTE-IDENTICAL NEFF and an identical trace error for "no flag"
and `--auto-cast=none`, against a 0.40x NEFF and a three-orders-larger error for
`--auto-cast=matmult`. The comparison is at
`results/featurewire-log20-prec-g5-trn1-100M/raw/probe/autocast-default.txt`.

    arm    compiler flags                                  trace error   NEFF at b100
    fp32   --auto-cast=none                                1.48e-06      360.0 MiB
    bf16   --auto-cast=matmult --auto-cast-type=bf16        2.32e-03      154.5 MiB

So the earlier Trainium campaigns in this series, which passed no flag, were GENUINELY
FP32 and their recorded `"precision": "fp32"` was right. This round's FP32 arm IS
comparable with `featurewire-log20-g5-trn1-10M-20M`'s Trainium rows: same wire, same log
rate, same session count, same cast, different model size.

The flag is passed anyway, for a narrower reason that still holds: a precision label must
not rest on an undocumented framework default that a release can change without saying so.
The error is the compiled-versus-PyTorch maximum that `compile_neuron.py` already measures,
and it is the PROOF -- BF16 keeps ~3 decimal digits, so casting the matmuls moves it three
orders. Each arm declares a window and the compile REFUSES to write a NEFF whose error
falls outside it, so an arm cannot be mislabelled even if the flag is dropped or a cache
entry is stale. That check is what caught the mistaken belief described above, before it
cost a re-run.

`matmult` rather than `all` because it casts the matrix-multiply engine and leaves
normalization and accumulation in FP32 -- the same division TensorRT's `--bf16` makes on
the A10G. That correspondence is what lets a Trainium BF16 row be read against a GPU one.

ONE PRECISION PER LANE. `trn1-1` is the FP32 arm, `trn1-2` the BF16 arm. Each compiles one
cheap NEFF (b100) and one expensive one (b500), so the two lanes carry equal compile
weight and neither becomes the critical path -- which matters more than usual here,
because no campaign in this series has ever recorded how long a NEFF compile takes. Every
prior entry was a cache restore. The cost of this split is that a precision comparison
spans two machines; this series measured host-to-host spread at -10.8% to +2.7%, against
an expected BF16 effect near 2x, and the report states it rather than absorbing it.

CORES PER REPLICA 1 AND CONTEXTS PER REPLICA 1, BOTH BY INSTRUCTION. These are tabular
models with no long sequence: nothing needs tensor parallelism, and a request puts exactly
one call on the one replica it leases, so a second tensor set per replica would be loaded
and never used. With `sessions` 2 and one replica per core, both NeuronCores are occupied
and a request's whole call runs on one of them.

NO MINI BATCH. A NEFF's leading dimension is FIXED, so the mini batch IS the row count and
a request is one call of exactly the rows it carries. It is spelled as a size rather than
as an absence only because the Neuron backend reads the row count from the configuration.

THE WIRE IS THE FEATURE WIRE, BY INSTRUCTION. The client sends 210 NAMED features per row
and the server transforms and concatenates them into the model's one [rows, 690] FP32
input, so host-side vectorization is ON and its cost is measured as `VectorizationTime`.
The inference log is on at `sample_rate` 0.2 -- one request in five, to Amazon S3.

THE RATES ARE SPECIFIED, AND PROJECTED TO BE ABOVE WHAT THIS HOST CAN SERVE. They were set
from the A10G, where all four were MEASURED holding in `nominibatch-precision-gpu-g5-100M`.
One trn1.2xlarge is a different machine: two NeuronCores, each executing one NEFF at a
time. Projecting from the 10M/20M round's own measured device spans on this exact wire and
host -- 0.738 ms per layer at 100 rows, 3.632 at 500 -- a 31-layer model lands at 23.1 ms
and 113.4 ms per call, so the ceiling is ~87 and ~17.6 req/s against the 200 and 40 asked
of BF16, and a lower ceiling again against the 100 and 20 asked of FP32. Cells over their
ceiling are reported by ACHIEVED rate and enter no latency comparison, which is this
series' convention; they are what measures the ceiling, and a second wave of rates set
from their MEASURED device time is what earns a latency on this lane.

TWO ZONES, NOT FOUR. `trn1.2xlarge` is offered ONLY in `us-west-2b` and `us-west-2d` --
`DescribeInstanceTypeOfferings` says so, and the other two return `Unsupported` rather
than a capacity error. Both are listed and the whole rig lands in whichever can hold the
first server, because a client in one zone and its server in another would add a network
hop to one lane and not the other.
"""

import json
import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "featurewire-log20-prec-g5-trn1-100M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

# ---------------------------------------------------------------- the models
#
# NEURON DOES NOT COMPILE ONNX, so unlike every other run in this series the graph is not
# the artifact. `scripts/benchmark/compile_neuron.py` rebuilds the SAME seeded PyTorch
# architecture `generate_benchmark_models.py` defines and keeps the NEFF that
# `torch_neuronx.trace` emits. The model is therefore pinned by its SEED and its target
# parameter count, and the NEFF's identity is recorded after the fact as a SHA-256 beside
# a compiled-versus-PyTorch error, which `compile_neuron.py` checks before writing it.
#
# That is why there is no md5 here as there is in the GPU rig: there is nothing to pin
# before the host builds it. What IS pinned is the target, and the parameter count the
# generator lands on for that target is deterministic -- `config-sweep-neuron-trn1-10M-50M`
# got 9,142,789 for `10M` from this same code path, which is the continuity check.
#
# The counts below are the NEURON models (`VectorizedModel`, depth chosen to hit the
# target). They differ slightly from the ONNX `bench_model.onnx` counts the CPU and GPU
# runs pin, because those are a different architecture in the same family. A report must
# not read a Trainium latency against a GPU one as though the graph were identical; what
# is shared is the 210-feature wire, the 690-float input and the 5-class output.
MODELS = {
    "10m": dict(target="10M", expect_params=9_142_789, expect_layers=2),
    # Never measured on Trainium before `featurewire-log20-g5-trn1-10M-20M`. The generator
    # lands 18,599,941 over 5 layers for this target, which is the same depth as the GPU
    # campaign's ONNX 20M (18,589,701) -- close enough that the two are the same design at
    # the same weight, and still not the same graph.
    "20m": dict(target="20M", expect_params=18_599_941, expect_layers=5),
    # THE MODEL OF THIS ROUND, and nothing on Trainium to check it against. `expect_params`
    # is None rather than invented: the parameter count is a deterministic function of the
    # target, so prime RECORDS what the generator lands on and the report quotes it. The
    # A10G lane's ONNX graph at this target is 100,561,925 over 31 layers; the two counts
    # are reported separately rather than treated as one number, because
    # `torch_neuronx.trace` rebuilds the same seeded architecture rather than loading that
    # file.
    "100m": dict(target="100M", expect_params=None, expect_layers=None),
}

WIDTH, CLASSES = 690, 5
# THE FEATURE WIRE: 210 NAMED features per row, transformed and concatenated by the server
# into the model's one [rows, 690] input. The earlier Trainium runs set this to 1 -- one
# feature that already WAS the finished row -- and that pre-vectorized configuration is
# refused here by a width check in `stage_config` and a banner gate in `start_hushar`.
N_FEATURES = 210

# THE TWO ARMS. One precision per lane; see the module docstring for why the flag is
# explicit on both and how the trace error proves it.
PRECISIONS = ("fp32", "bf16")

# THE LOADS, AS SPECIFIED, PER PRECISION. BF16 is asked for twice the rate of FP32 at each
# packaging, which is where the A10G measured the BF16 speedup landing.
#
# A NOTE THE REPORT MUST CARRY: all four are projected to sit ABOVE what one trn1.2xlarge
# can serve at this model size. See the module docstring for the arithmetic and for what a
# cell over its ceiling is still worth measuring.
LOADS = {
    ("10m", "fp32"): [(100, 1000), (500, 200)],
    ("20m", "fp32"): [(100, 450), (100, 400), (500, 80), (500, 75)],
    ("100m", "fp32"): [(100, 100), (500, 20)],
    ("100m", "bf16"): [(100, 200), (500, 40)],
}

# NO MINI BATCH. The mini batch IS the row count, so a request is one NEFF call of
# exactly the rows it carries. It is spelled as a size rather than as an absence because
# a NEFF's leading dimension is fixed and hushar's Neuron backend takes the row count
# from the configuration; the GPU counterpart omits the key entirely and both mean one
# engine call per request. Two shapes per size, b100 and b500, and every session count
# shares them -- a session count loads more copies of the same NEFF, not another shape.
MINI_BATCH_FOR_ROWS = {100: 100, 500: 500}

# THE ONE AXIS. `threading.sessions_per_model` -- Neuron model replicas, one complete
# NEFF copy each. `threading.sessions_per_request` is 1 for every cell, so a request
# leases ONE replica and this IS the requests in flight, which is the quantity Triton's
# `instance_group` count sets. That equivalence is what makes a trn1 row readable against
# `nominibatch-sessions-gpu-g5-10M-20M`.
SESSIONS = [2]

NEURON_CORES = 2
CORES_PER_REPLICA = 1          # pinned by instruction; see the module docstring
CONTEXTS_PER_REPLICA = 1       # pinned by instruction; one piece per replica needs one

# `trn1.2xlarge` from DescribeInstanceTypes. The feature wire is 2.54x FEWER bytes than
# the pre-vectorized tensor the earlier Trainium runs sent, so the network is far from the
# limit at these rates -- but every cell still measures its own rx/tx over the window,
# because a network ceiling must not be reported as a device one.
NET_BASELINE_GBPS = 3.125
NET_BURST_GBPS = 12.5

# The campaigns these numbers are read against, named so a report cannot drift from a
# baseline it never identifies.
#   BASELINE_RUN      None, and for a plain reason: no Trainium campaign has measured this
#                     MODEL SIZE. The round that measured this wire, this log rate and
#                     this session count did so at 10M and 20M. It was genuinely FP32 --
#                     `torch_neuronx.trace` does not inherit the compiler CLI's `matmult`
#                     default, which `raw/probe/autocast-default.txt` measures -- so it is
#                     comparable in KIND with this round's FP32 arm and is what the
#                     device-time projection in sweep.py is anchored on. It simply has no
#                     cell at 100M to subtract against, so no control table is published
#                     for this lane.
#   GPU_COUNTERPART   the A10G lane of THIS run, measured in the same round under the same
#                     run name, so the two accelerators are compared on identical settings
#                     rather than across campaigns.
BASELINE_RUN = None
NEURON_PRIOR_RUN = "featurewire-log20-g5-trn1-10M-20M"
GPU_COUNTERPART_RUN = "featurewire-log20-prec-g5-trn1-100M"
# The A10G campaign that measured this model at these precisions on the TENSOR wire with
# the log off, which is what the feature wire and the log are read against.
GPU_PRECISION_RUN = "nominibatch-precision-gpu-g5-100M"

# ---------------------------------------------------------------- what to launch

# `trn1.2xlarge` IS NOT OFFERED IN 2a OR 2c. DescribeInstanceTypeOfferings lists only
# these two, and the missing zones fail with `Unsupported` rather than a capacity error,
# which the launch loop tolerates either way. Listing only the real two keeps the fleet
# request asking a question that can be answered.
SUBNETS = [
    ("us-west-2b", "subnet-00c62a1237179cad1"),
    ("us-west-2d", "subnet-06fcad5c6d223b0ab"),
]
SERVER_SG = "sg-05b6ee0106612a481"           # hushar-bench-server
CLIENT_SG = "sg-0db48fc45a1d1f0f4"           # hushar-bench-client
SERVER_PROFILE = "hushar-bench-server"       # reads models/ and dist/, writes both + runs/
CLIENT_PROFILE = "hushar-bench-client"       # reads dist/ only: a client must not serve

SERVER_TYPE = "trn1.2xlarge"                 # 1 Trainium1, 2 NeuronCores, 8 vCPU, 32 GiB
CLIENT_TYPE = "m9g.8xlarge"                  # 32 vCPU; never above 4% busy in prior runs
SERVER_CORES = 8
CLIENT_CORES = 32
# The DLAMI's own root is 80 GB and it arrives most of the way full: the PyTorch venv, the
# Neuron compiler and the driver are all in the image. On top of that this run adds a
# cargo build and TWO 100M NEFFs with their retained compiler work directories -- a far
# larger compile than the 10M and 20M models this number was last set for, and one whose
# intermediate files nobody in this series has measured. 250 GB rather than 150: a gp3
# root costs cents for a run of this length, and a compile that dies on ENOSPC after an
# hour costs the lane.
SERVER_ROOT_GB = 250
CLIENT_ROOT_GB = 80

# THE SERVER IMAGE MUST CARRY `torch_neuronx.trace`, AND THAT PINS IT TO THE PyTorch 2.9
# LINE. Neuron does not compile ONNX, so this run builds its own NEFFs, and hushar's
# backend serves a NEFF that takes ONE input: the [batch, 690] feature tensor, with the
# model's parameters compiled in as constants. `torch_neuronx.trace` produces exactly
# that.
#
# The current `torch-neuronx 2.14` DLAMI (SDK 2.33) was tried first and cannot be used.
# `trace` is gone from it, and the dynamo backend that replaced it writes no model NEFF to
# disk and lowers a STATELESS graph -- every parameter becomes a graph input, so a NEFF
# built from its StableHLO has a weight tensor where hushar expects features, which hushar
# refuses with `NEFF input "input0" fixes its batch at 4096`. The full finding is in the
# docstring of scripts/benchmark/compile_neuron.py.
#
# PyTorch 2.9 is also the line `config-sweep-neuron-trn1-10M-50M` compiled with: that
# campaign recorded torch_neuronx 2.9.0.2.15.32035+de43f57c and this image carries the
# same string, so a NEFF built here is comparable with that campaign's rather than merely
# newer. Ubuntu 24.04 and kernel 6.17.0-1019-aws match its host too.
#
# THE COST, DISCLOSED: this image is older than the 45-day freshness limit `rig.py`
# enforces, so launching needs AMI_ALLOW_STALE=1. That limit exists because a long-lived
# rig was once flagged for an outdated OS; these hosts are ephemeral and live for one
# run, and the alternative is a NEFF the service cannot load. The report states the
# image and its age.
SERVER_AMI_NAME = "Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) *"
CLIENT_AMI_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"

TAG_PREFIX = "hushar-bench"
# WHICH LANE OF THIS RUN AN INSTANCE BELONGS TO, AND WHY IT IS IN THE TAG. This round
# measures two accelerators under ONE run name, and both rigs call their clients `cl1`
# and `cl2`. The instance Name tag is `{TAG_PREFIX}-{RUN}-{name}`, and `rig.py adopt`
# finds a host it already owns by exactly that tag -- so without a lane in it, whichever
# rig launched second would ADOPT the other lane's client instead of launching its own,
# and then drive a server across an availability zone it was never placed with. `trn1`
# is offered only in us-west-2b and 2d while g5 is offered in 2a, 2b and 2c, so the two
# lanes frequently land in different zones and that failure would be a quiet latency
# offset rather than an error.
LANE = "trn1"


def tag_name(name):
    """The instance Name tag, and the launch-template name, for one planned host."""
    return f"{TAG_PREFIX}-{RUN}-{LANE}-{name}"

# ---------------------------------------------------------------- the launched rig

RIG_SET = os.environ.get("RIG_SET", "main")

PLANS = {
    "main": {
        # ONE PRECISION PER LANE, and `size` is the same on both: the axis is the cast,
        # not the model. model_id is the host's own name; the CloudWatch `ModelId`
        # dimension is the CELL id, which sweep.py prefixes `trn1fwp-` and which carries
        # the precision -- load-bearing, because `HusharBench` already holds `trn1l20-`
        # ids from the 10M/20M round on this same wire, log rate and session count, and a
        # shared id would make two campaigns one time series.
        "trn1-1": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-1", size="100m", precision="fp32"),
        "trn1-2": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-2", size="100m", precision="bf16"),
        "cl1": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="trn1-1"),
        "cl2": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="trn1-2"),
    },
    # Kept so a follow-up -- the second wave of rates this round's over-ceiling cells
    # will call for, the two-core replica it pins away, or an inf2 arm -- can launch its
    # own hosts without editing this file.
    "probe": {
        "trn1-3": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-3", size="100m", precision="bf16"),
        "cl3": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="trn1-3"),
    },
}
if RIG_SET not in PLANS:
    raise SystemExit(f"RIG_SET={RIG_SET!r}; one of {sorted(PLANS)}")

_SUFFIX = "" if RIG_SET == "main" else f"-{RIG_SET}"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"rig-state{_SUFFIX}.json")

PLAN = PLANS[RIG_SET]

SIZE_OF = {name: spec["size"] for name, spec in PLAN.items()
           if spec["role"] == "server"}
# WHICH ARM A SERVER CARRIES. `launch.sh` reads this to set PRECISION, so the lane cannot
# be started on the wrong arm by a command line that omitted it.
PRECISION_OF = {name: spec["precision"] for name, spec in PLAN.items()
                if spec["role"] == "server"}
for _name, _prec in PRECISION_OF.items():
    if _prec not in PRECISIONS:
        raise SystemExit(f"{_name} is planned at precision {_prec!r}; "
                         f"one of {sorted(PRECISIONS)}")


def state():
    """What `rig.py launch` recorded, or an empty rig."""
    if not os.path.exists(STATE):
        return {}
    with open(STATE) as handle:
        return json.load(handle)


def save_state(doc):
    with open(STATE, "w") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write("\n")


def live():
    """The rig as PLAN plus the ids and addresses of the instances that exist.

    A host with no recorded id is present with `id=None`, so a caller that needs one fails
    saying which host is missing rather than with a KeyError.
    """
    recorded = state().get("hosts", {})
    out = {}
    for name, spec in PLAN.items():
        out[name] = dict(spec, id=recorded.get(name, {}).get("id"),
                         ip=recorded.get(name, {}).get("ip"))
    return out


ALL = live()

# Groups describe the rig that EXISTS, not the plan. A lane the zone could not hold is not
# part of this run, and a fan-out over "servers" should reach the servers there are.
# Naming a missing host directly still fails, which is the check worth keeping.
_PRESENT = [n for n in PLAN if ALL[n]["id"]] or list(PLAN)

GROUPS = {
    "servers": [n for n in _PRESENT if PLAN[n]["role"] == "server"],
    "clients": [n for n in _PRESENT if PLAN[n]["role"] == "client"],
    "sweep": list(_PRESENT),
    "all": list(_PRESENT),
}

# A lane needs both ends. A client whose server was never placed drives nothing.
LANES = {spec["drives"]: client
         for client, spec in PLAN.items()
         if spec["role"] == "client" and client in _PRESENT
         and spec["drives"] in _PRESENT}


def resolve(name):
    """A group name, a host name, or a comma-separated mix of either."""
    out = []
    for part in name.split(","):
        part = part.strip()
        if part in GROUPS:
            out += GROUPS[part]
        elif part in ALL:
            out.append(part)
        else:
            raise SystemExit(f"unknown host or group {part!r}; "
                             f"hosts {sorted(ALL)}, groups {sorted(GROUPS)}")
    names = list(dict.fromkeys(out))
    missing = [n for n in names if not ALL[n]["id"]]
    if missing:
        raise SystemExit(
            f"no instance id recorded for {', '.join(missing)}.\n"
            f"Run `rig.py launch` first -- this run launches its own hosts and records "
            f"them in {os.path.basename(STATE)}. `launch` ADOPTS hosts it already owns, "
            f"found by tag, so it is also how a lost state file is rebuilt."
        )
    return names


def rig_env():
    """`RIG` as the sweep wants it: `name=ip;name=ip` for every server in this set."""
    return ";".join(f"{n}={ALL[n]['ip']}" for n in GROUPS["servers"] if ALL[n]["ip"])
