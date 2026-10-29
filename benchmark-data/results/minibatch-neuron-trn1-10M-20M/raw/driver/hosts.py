#!/usr/bin/env python3
"""The rig: two `trn1.2xlarge` server hosts, one client each, LAUNCHED fresh for this run.

This is the Trainium counterpart of `minibatch-gpu-g5-10M-20M`. That run asked whether
cutting a request into mini batches lowers its latency on one NVIDIA A10G and found it
did not, for a reason that was a property of the SERVER rather than the device: hushar
leases ONE ONNX Runtime session per request and the TensorRT provider serializes
concurrent `Run()` on it, so the pieces fanned out for the feature work and then queued
on the device. The report named the missing knob explicitly --
`execution_contexts_per_replica`, which exists only on the Neuron backend.

    THIS RUN ASKS THE SAME QUESTION ON THE DEVICE THAT HAS THAT KNOB,
    AND SWEEPS THE REPLICA COUNT BESIDE IT.

TWO AXES, BOTH SWEPT, ONE CORE PER REPLICA THROUGHOUT.

    mini batch   rows per NEFF call. `mini_batch_size == rows` is one call and is the
                 baseline; anything smaller cuts the request into pieces. Each piece
                 leases its own execution context, so unlike the GPU run the pieces are
                 not forced through one serialized handle.
    replicas     `threading.sessions_per_model`. On two NeuronCores, 2 is one replica per
                 core, 4 is two and 6 is three. Above one per core a core serves several
                 requests at once but still EXECUTES one at a time, so this axis buys
                 host-side overlap and queue lanes rather than device capacity -- which
                 is exactly what makes it worth measuring against latency.

CORES PER REPLICA IS PINNED AT 1, BY INSTRUCTION, AND THAT BOUNDS THE ANSWER. These are
small tabular models; nothing here needs tensor parallelism. But it is worth writing down
what it costs, because `config-sweep-neuron-trn1-10M-50M` measured exactly this: its 41%
latency cut on 50M came from `s1 · k2 · c2` -- ONE replica spanning TWO cores -- and not
from the split on its own. The pieces won there because they landed on two DIFFERENT
cores. With one core per replica a NeuronCore runs one NEFF at a time, so a split's device
term cannot improve; what a context can still buy it is DMA overlap, and what the host can
still buy it is the 210 transformations fanning out on the rayon pool. This run measures
whether those two terms outrun N times the per-call overhead. A negative answer here is a
measurement, not a wasted lane -- it closes the series the GPU run opened.

CONTEXTS TRACK THE PIECE COUNT, WHICH IS THE SERVER'S OWN RULE. hushar prints it at
startup:

    note   : a replica serves one request at a time, so
             "backend.execution_contexts_per_replica" is worth exactly the executions one
             request produces -- fewer serialize the request, more stay idle

So `contexts = ceil(rows / mini_batch)`. For every unsplit cell that is 1, so the
baselines are the setting they would have had anyway and the axis is still one variable.
A separate `control` phase re-runs the largest split of each load at `contexts = 1` to
PRICE that choice instead of assuming it.

ONE MODEL PER LANE. `trn1-1` measures 10M, `trn1-2` measures 20M. Every comparison this
report draws is within one model size and one load, so a lane owns a whole size and no
table spans two machines -- the rule the 20M CPU campaign adopted after the 10M one found
a placement offset between two instances of the same type that repeating a cell does not
reduce. It also halves the wall clock, which matters at 20 cells of 18 minutes per lane.

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

RUN = "minibatch-neuron-trn1-10M-20M"
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
N_FEATURES = 210

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
    "10m": [(100, 1000), (500, 200)],     # 100,000 rows/s both
    "20m": [(100, 750), (500, 150)],      #  75,000 rows/s both
}

# THE AXIS, per row count. Every entry must divide its row count exactly: a short last
# batch would make `is_fixed` pad it, which costs a second NEFF and makes the latency a
# blend of two compiled shapes rather than a measurement of one. The unsplit size must be
# present or the gain column has no baseline.
MINI_BATCHES = {
    100: [100, 50, 25],
    500: [500, 250, 100],
}

# The second axis. Two NeuronCores, one core per replica, so these are 1, 2 and 3
# replicas per core.
REPLICAS = [2, 4, 6]

NEURON_CORES = 2
CORES_PER_REPLICA = 1          # pinned by instruction; see the module docstring

# The campaigns these numbers are read against, named so a report cannot drift from a
# baseline it never identifies.
BASELINE_RUN = "config-sweep-neuron-trn1-10M-50M"
GPU_COUNTERPART_RUN = "minibatch-gpu-g5-10M-20M"

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

# ---------------------------------------------------------------- the launched rig

RIG_SET = os.environ.get("RIG_SET", "main")

PLANS = {
    "main": {
        # model_id is the CloudWatch `ModelId` prefix. `HusharBench` already carries
        # `trn1-*` from config-sweep-neuron-trn1-10M-50M, so every cell id in this run is
        # distinct from those by its `-mb<n>-s<n>` tail; reusing one would make two
        # campaigns a single time series.
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
