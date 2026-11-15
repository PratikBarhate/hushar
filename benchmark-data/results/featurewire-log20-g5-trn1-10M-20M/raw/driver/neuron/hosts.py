#!/usr/bin/env python3
"""The rig: two `trn1.2xlarge` server hosts, one client each, LAUNCHED fresh for this run.

This is the Trainium counterpart of `minibatch-sessions-gpu-g5-10M-20M`. That run moved
the mini batch and the session count TOGETHER on one NVIDIA A10G so that each mini batch
of a request took its own ONNX Runtime session and its own device stream, rather than
queueing on the one session hushar used to lease per request.

    THIS RUN IS THAT GRID ON TRAINIUM, ON THE PRE-VECTORIZED WIRE.

A RUNG IS A PAIR, AND THE PAIRING IS THE POINT.

    mini batch   rows per NEFF call, `is_fixed` on. `mini_batch == rows` is the baseline.
    sessions     `threading.sessions_per_model` -- Neuron model replicas, one complete
                 NEFF copy each.

`threading.sessions_per_request` is set to the PIECE COUNT, so a request leases a group
of replicas and runs one mini batch on each. Requests in flight is
`sessions_per_model / sessions_per_request` = 2 in every cell: what moves down the grid
is how finely a request is cut and across how many replicas, never how many requests are
served at once.

    rung       mini batch   sessions   pieces   replicas/core   request lanes
    unsplit    rows                2        1               1               2
    halves     rows/2              4        2               2               2
    quarters   rows/4              8        4               4               2

CORES PER REPLICA 1 AND CONTEXTS PER REPLICA 1, BOTH BY INSTRUCTION. These are small
tabular models; nothing needs tensor parallelism and nothing needs a second tensor set
per replica, because a request puts exactly one piece on each replica of its group.

That pins the placement, and the placement is what the rungs actually do. `replica_cores`
puts replica `r` on core `(r / replicas_per_core) * cores_per_replica` and
`SessionGroup::split` cuts replicas into CONSECUTIVE groups, so with `sessions / 2`
replicas per core a request's group is exactly one NeuronCore's replicas. Every rung is
therefore two request lanes, one per core, with a request's pieces confined to its own
core. A NeuronCore runs one NEFF at a time, so the split's DEVICE term cannot improve;
what can improve is DMA, since each piece has its own tensor set. There is no host-side
vectorization left to fan out -- see the wire.

THE WIRE IS PRE-VECTORIZED, BY INSTRUCTION. The client sends one feature per row that
already IS the finished [rows, 690] FP32 vector and the configuration transforms it with
`identity`, so the server copies it in and derives nothing. `minibatch-neuron-trn1-10M-20M`
sent 210 named features and found the split's gain in the host-side transformations; this
run removes that term, so what is left is the call shape and the DMA overlap.

It also costs 2.54x the bytes. 100,000 rows/s is 276 MB/s inbound = 2.21 Gbps against
this instance type's 3.125 Gbps sustained figure, so every cell measures its own rx/tx
over the window and the report carries it. A network ceiling must not be reported as a
device one.

ONE MODEL PER LANE. `trn1-1` measures 10M, `trn1-2` measures 20M. Every comparison this
report draws is within one model size and one load, so a lane owns a whole size and no
table spans two machines -- the rule the 20M CPU campaign adopted after the 10M one found
a placement offset between two instances of the same type that repeating a cell does not
reduce. It also halves the wall clock.

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

RUN = "featurewire-log20-g5-trn1-10M-20M"
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
    # Never measured on Trainium before this run. The generator lands 18,599,941 over 5
    # layers for this target, which is the same depth as the GPU campaign's ONNX 20M
    # (18,589,701) -- close enough that the two are the same design at the same weight,
    # and still not the same graph.
    "20m": dict(target="20M", expect_params=18_599_941, expect_layers=5),
}

WIDTH, CLASSES = 690, 5
N_FEATURES = 1          # the pre-vectorized wire: one feature that IS the row

# THE LOADS, AS SPECIFIED. Two packagings of the same row throughput per size, so the
# grid also answers -- for free -- whether a bigger request is cheaper per candidate on a
# NeuronCore, which is the opposite lever from the CPU's and the same one the GPU run
# found to be the real win there.
#
# A NOTE THE REPORT MUST CARRY. These rates were specified, not probed, and two of the
# four are projected to sit ABOVE what one trn1.2xlarge can serve. Projecting from the
# two measured points in `config-sweep-neuron-trn1-10M-50M` -- 1.644 ms per 100-row call
# at 10M, 10.99 ms at 50M, linear in parameters -- 20M lands near 4.1 ms, so two serial
# cores cap out around 485 req/s against the 750 asked, and around 97 req/s at 500 rows
# against the 150 asked. The 10M rungs are inside their measured ceiling (1.10k TPS at
# 100 rows) but not far inside it. Cells that do not hold their rate are reported by
# achieved rate and enter no latency comparison, which is the convention every run in
# this series uses; they still measure the ceiling, which is the thing nobody had for 20M.
LOADS = {
    "10m": [(100, 1000), (500, 200)],
    "20m": [(100, 450), (100, 400), (500, 80), (500, 75)],
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

# `trn1.2xlarge` from DescribeInstanceTypes. The pre-vectorized wire is 2.54x the bytes
# of the named-feature wire, so this is a number the report has to carry: 100,000 rows/s
# is 2.21 Gbps, 71% of the sustained figure.
NET_BASELINE_GBPS = 3.125
NET_BURST_GBPS = 12.5

# The campaigns these numbers are read against, named so a report cannot drift from a
# baseline it never identifies.
#   BASELINE_RUN      the Trainium control: the SAME loads at 10M and the same session
#                     count, on the PRE-VECTORIZED wire with the inference log off.
#   GPU_COUNTERPART   the A10G lane of THIS run, measured in the same round under the
#                     same run name, so the two accelerators are compared on identical
#                     settings rather than across campaigns.
BASELINE_RUN = "nominibatch-sessions-neuron-trn1-10M-20M"
NEURON_PRIOR_RUN = "minibatch-neuron-trn1-10M-20M"
GPU_COUNTERPART_RUN = "featurewire-log20-g5-trn1-10M-20M"
# The Trainium campaign on the same wire WITH the split engaged, which is
# what removing the split is measured against.
NEURON_SPLIT_RUN = "minibatch-sessions-neuron-trn1-10M-20M"

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
# cargo build and ten NEFFs with their retained compiler work directories, so the root is
# grown rather than taken as it comes.
SERVER_ROOT_GB = 150
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
        # model_id is the host's own name; the CloudWatch `ModelId` dimension is the CELL
        # id, which sweep.py prefixes `trn1-tensor-`. That prefix is load-bearing:
        # `HusharBench` already carries `trn1-<size>-r100t1000-mb100-s2` and
        # `...-mb50-s4` from `minibatch-neuron-trn1-10M-20M`, whose grid shares two of
        # this one's rungs on a DIFFERENT wire. Reusing an id would silently make two
        # campaigns one time series.
        "trn1-1": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-1", size="10m"),
        "trn1-2": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-2", size="20m"),
        "cl1": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="trn1-1"),
        "cl2": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="trn1-2"),
    },
    # Kept so a follow-up -- the two-core replica this run pins away, or an inf2 arm --
    # can launch its own hosts without editing this file.
    "probe": {
        "trn1-3": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                       ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                       model_id="trn1-3", size="10m"),
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
