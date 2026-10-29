#!/usr/bin/env python3
"""The rig: two NVIDIA A10G server hosts, one client each, LAUNCHED fresh for this run.

This is the GPU counterpart of `config-sweep-neuron-trn1-10M-50M`. That run asked what one
Trainium host can serve at 100 rows; this one asks the same of one `g5.2xlarge` -- a single
NVIDIA A10G -- and adds the question the Trainium run had no second stack for:

    AT THE SAME OFFERED LOAD, ON THE SAME GPU, SCORING THE SAME TensorRT ENGINE,
    HOW DOES hushar COMPARE WITH NVIDIA'S OWN SERVER?

TWO ARMS.

    hushar   210 named features in, vectorized on the host CPU into the 690-float
             vector, scored by TensorRT through ONNX Runtime's TensorRT provider.
             This is the SAME request path the Trainium run measured, so these cells
             read directly against that report.
    triton   Triton Inference Server, `tensorrt` backend, a prebuilt `model.plan`,
             KIND_GPU. The 690-float vector in; nothing transformed.

THE WIRES DIFFER, AND THAT IS NOT HIDDEN. Triton's TensorRT plan takes the finished tensor,
so the 210 transformations do not vanish -- they move to the caller. hushar's latency
therefore carries work Triton's does not. The run does not quietly absorb that: hushar
publishes `VectorizationTime` on every cell, so the report can state what the feature path
costs INSIDE hushar's own number, as a lower bound on what a Triton caller pays outside it.
A tensor-wire hushar arm would close this exactly, and the 10M CPU campaign already ran it:
the graph cost was identical on both wires and the feature wire won on bytes moved. Spending
a third of this rig to re-confirm that on a GPU is not what the question needs.

ONE GPU, TWO HOSTS, SPLIT BY MODEL. Every comparison this report draws is hushar against
Triton at one size and one rate, so a lane owns a whole model: `g5-1` measures 10M, `g5-2`
measures 20M. No table ever spans two machines, which is the rule the 20M CPU campaign adopted
after the 10M one found a placement offset between two instances of the same type that
repeating a cell does not reduce. Splitting this way also halves the wall clock, which
matters because the laptop driving it is expected to disconnect.

ONE AVAILABILITY ZONE, CHOSEN AT LAUNCH RATHER THAN PINNED. Every latency this report
compares is measured over the link between a client and its server, so a cross-zone pair
would add a network hop to one lane and not the other. All four hosts therefore go in one
zone -- but WHICH zone cannot be written down, because `g5.2xlarge` is a scarce instance
type and the zone that has capacity is not knowable in advance. The abandoned `g7` attempt
at this run died exactly here, retrying one hardcoded subnet forever. `rig.py launch` tries
the zones in order and places the whole rig in the first that can hold the GPU servers.
"""

import json
import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "minibatch-gpu-g5-10M-20M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

# ---------------------------------------------------------------- the graphs

# Both graphs are pinned by md5 rather than discovered: hushar's TensorRT provider and
# Triton's `trtexec` must compile the SAME bytes, or the comparison is between two models
# and not two servers. Measured from the checked-in artifacts, and asserted on the host by
# `deploy.sh` and again by `preflight`.
#
# The two are the same design at two weights -- 2 encoder layers against 15 -- with the
# same 210 features, the same 690-float input, the same thirteen ONNX operator types and
# the same `ai.onnx` opset 17. There is no `ai.onnx.ml` node in either, which is what
# makes them expressible as a TensorRT engine at all: TensorRT has no string type, and the
# ONNX parser does not implement that domain. hushar's string handling happens in Rust,
# before the graph, so the graph itself is pure float arithmetic.
MODELS = {
    "10m": dict(md5="5539c97131c896db6c58088d9f9b9ee3", bytes=36571281,
                params=9138693, layers=2, macs_per_row=53288960),
    "20m": dict(md5="30f9c5330b808a34b55fe88048a20da6", bytes=74398005,
                params=18589701, layers=5, macs_per_row=128983040),
}

# THE 2x MODEL, AND WHAT EXACTLY IS DOUBLED. Same 210 features, same 690-float input, the
# same thirteen ONNX operator types, the same matmul shapes -- five encoder layers instead
# of two. That is the only change, and it is what makes a 10M-against-20M reading mean
# something: TensorRT selects the same kernels for the same shapes, so the difference is
# MORE OF THE SAME WORK rather than different work.
#
#   parameters   9,138,693 -> 18,589,701      2.03x
#   MACs per row    53.3M ->      129.0M      2.42x
#
# The two ratios differ, and that is a property of the design rather than a flaw in the
# sizing: `to_tokens` is a Linear(690 -> 8x512) applied ONCE per row, so it costs 2,830,336
# parameters at any model size. Holding the input width at 690 -- mandatory, or the wire
# bytes change and nothing is comparable -- pins that layer as a constant, and every added
# parameter lands in the trunk. So the report carries BOTH ratios rather than rounding them
# to "2x", and the arithmetic ratio is the one latency should be read against.
PARAM_RATIO = MODELS["20m"]["params"] / MODELS["10m"]["params"]
MACS_RATIO = MODELS["20m"]["macs_per_row"] / MODELS["10m"]["macs_per_row"]
MODEL_FILE = "bench_model.onnx"
ROWS = 100              # every cell in this run; the Trainium run's 100-row cells
WIDTH, CLASSES = 690, 5

# THE SAME LADDER FOR BOTH SIZES. The 10M grid is the one that was asked for; giving 20M
# the identical rungs is what makes the two columns subtractable at every rate, and it
# also finds 20M's knee for free -- 2.42x the arithmetic should move the ceiling down by
# roughly that factor, so a rung 20M cannot hold is a measurement and not a wasted cell.
LOADS = {
    "10m": [(100, 200), (100, 400), (100, 800), (100, 1000)],
    "20m": [(100, 200), (100, 400), (100, 800), (100, 1000)],
}

# The campaign these numbers are read against, named so the report cannot drift from a
# baseline it never identifies.
BASELINE_RUN = "config-sweep-neuron-trn1-10M-50M"

# ---------------------------------------------------------------- what to launch

# One availability zone for all four hosts: a cross-zone client would add a network hop to
# every latency this run compares, and at 100 rows and 3 ms that is not negligible. Which
# zone is decided at launch, not here: `g5.2xlarge` is scarce, and the first attempt at
# this run failed with InsufficientInstanceCapacity against a hardcoded us-west-2a. These
# are tried in order and the whole rig lands in the first zone that can hold the servers.
SUBNETS = [
    ("us-west-2a", "subnet-0cc468f8a7b87ca8d"),
    ("us-west-2b", "subnet-00c62a1237179cad1"),
    ("us-west-2c", "subnet-0f39db5099413d7c6"),
    ("us-west-2d", "subnet-06fcad5c6d223b0ab"),
]
SERVER_SG = "sg-05b6ee0106612a481"           # hushar-bench-server
CLIENT_SG = "sg-0db48fc45a1d1f0f4"           # hushar-bench-client
SERVER_PROFILE = "hushar-bench-server"       # reads models/ and dist/, writes runs/
CLIENT_PROFILE = "hushar-bench-client"       # reads dist/ only: a client must not serve

# DO NOT "UPGRADE" THIS. The A10G is Ampere, contemporary with Trainium1, which is the
# baseline this campaign is read against; the L4 in the g6 family is Ada, two years newer.
# Moving to g6 would flatter the GPU column and make the Trainium comparison mean less. It
# also happens to match the Trainium host at 8 vCPU, which is the axis the feature path
# cares about. g6 and gr6 were available at moments when g5 was not -- that is not a
# reason to switch; wait for g5 instead, which is what rig.py's capacity loop is for.
SERVER_TYPE = "g5.2xlarge"                   # 1x NVIDIA A10G 22888 MiB, 8 vCPU, 32 GiB
CLIENT_TYPE = "m9g.8xlarge"                  # 32 vCPU; never above 4% busy in prior runs
SERVER_CORES = 8
CLIENT_CORES = 32
# The server root carries the Triton image (~17 GB), two ONNX graphs, an ONNX Runtime GPU
# release, the CUDA/TensorRT libraries extracted from the image, two TensorRT plans and a
# cargo build. 80 GB is what the CPU rigs used and it is not enough here.
SERVER_ROOT_GB = 200
CLIENT_ROOT_GB = 80

# The AMI is an INPUT, not a constant, and is resolved at launch rather than pinned here,
# because pinning an AMI is how the previous long-lived rig went stale and had to be
# terminated mid-campaign. Each role resolves its own, since the two differ in both
# architecture and contents:
#
#   server  the NVIDIA kernel driver has to come from somewhere, and building it per run
#           would be a quarter of an hour of the measurement window spent on a kernel
#           module. AWS's Deep Learning BASE image ships the driver and the container
#           toolkit and no frameworks, which is exactly the surface this run wants: CUDA,
#           cuDNN and TensorRT USERSPACE all arrive later, extracted from the Triton
#           image, so that both stacks link one set of bytes. There is no SSM alias for
#           this image, so it is resolved by name filter, newest first.
#   client  plain Amazon Linux 2023 for arm64, from the published SSM alias.
SERVER_AMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023) *"
CLIENT_AMI_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"

# Tagged so an auditor can see what these are, and so a forgotten instance is
# identifiable rather than anonymous.
TAG_PREFIX = "hushar-bench"

# ---------------------------------------------------------------- the launched rig

RIG_SET = os.environ.get("RIG_SET", "main")

PLANS = {
    "main": {
        # model_id is the CloudWatch `ModelId` dimension, prefixed per CAMPAIGN rather
        # than only per host: `HusharBench` already carries `ceil-*` from the 10M CPU run
        # and `d20-*` from the 20M one, and reusing a name would make two runs on two
        # different graphs a single time series.
        "g5-1": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                     ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                     model_id="g5-1", size="10m"),
        "g5-2": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                     ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                     model_id="g5-2", size="20m"),
        "cl1": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="g5-1"),
        "cl2": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="g5-2"),
    },
    # Kept so a follow-up probe -- an FP16 pass, or a larger G instance -- can launch its
    # own hosts without editing this file.
    "probe": {
        "g5-3": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                     ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                     model_id="g5-3", size="10m"),
        "cl3": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="g5-3"),
    },
}
if RIG_SET not in PLANS:
    raise SystemExit(f"RIG_SET={RIG_SET!r}; one of {sorted(PLANS)}")

_SUFFIX = "" if RIG_SET == "main" else f"-{RIG_SET}"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"rig-state{_SUFFIX}.json")

PLAN = PLANS[RIG_SET]

# server -> the model size that lane measures. LANES is defined after `live()`, since a
# lane only exists if both of its hosts do.
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
# part of this run, and a fan-out over "servers" should reach the servers there are --
# otherwise one unplaceable GPU makes every subsequent command fail on a host nobody
# expected to have. Naming a missing host directly still fails, which is the check worth
# keeping: `fanout.py g5-2` should say so rather than silently do nothing.
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
            f"them in {os.path.basename(STATE)}."
        )
    return names


def rig_env():
    """`RIG` as the sweep wants it: `name=ip;name=ip` for every server in this set."""
    return ";".join(f"{n}={ALL[n]['ip']}" for n in GROUPS["servers"] if ALL[n]["ip"])
