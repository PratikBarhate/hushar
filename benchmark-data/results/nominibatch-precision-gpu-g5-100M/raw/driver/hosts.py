#!/usr/bin/env python3
"""The rig: two NVIDIA A10G server hosts, one client each, LAUNCHED fresh for this run.

WHAT THIS RUN ASKS. Every GPU campaign in this series so far measured hushar with its
mini-batch split engaged, which is a hushar feature Triton has no counterpart for. That
makes a hushar-against-Triton column a comparison of two servers AND one extra scheduling
decision at the same time. This run removes the extra decision:

    hushar WITH NO MINI BATCH, AGAINST TRITON AS IT SHIPS, ON THE SAME GPU, THE SAME
    TensorRT ENGINE AND THE SAME WIRE -- AT MATCHED PARALLELISM.

So every hushar cell scores the whole request in ONE engine call, and the only knob that
moves is how many model copies the server holds: `threading.sessions_per_model` on hushar,
`instance_group` count on Triton, at 2, 4 and 8 on both sides. A row of the grid is then a
pair of cells that differ in the server and nothing else.

TWO ARMS.

    hushar-tensor   hushar, one 690-float f32 vector per row, identity transformation, so
                    NO feature vectorization on the server. No mini batch, no candidate
                    model, feature logging off.
    triton          Triton Inference Server, `tensorrt` backend, a prebuilt `model.plan`,
                    KIND_GPU, `max_batch_size: 0`. The same 690-float vector in.

Both arms therefore receive the same bytes per row and call the graph once each.

ONE GPU, TWO HOSTS, SPLIT BY MODEL. Every comparison this report draws is hushar against
Triton at one size and one rate, so a lane owns a whole model: `g5-1` measures 10M, `g5-2`
measures 20M. No table ever spans two machines, which is the rule the 20M CPU campaign
adopted after the 10M one found a placement offset between two instances of the same type
that repeating a cell does not reduce. Splitting this way also halves the wall clock, which
matters because the laptop driving it is expected to disconnect.

ONE AVAILABILITY ZONE, CHOSEN AT LAUNCH RATHER THAN PINNED. Every latency this report
compares is measured over the link between a client and its server, so a cross-zone pair
would add a network hop to one lane and not the other. All four hosts therefore go in one
zone -- but WHICH zone cannot be written down, because `g5.2xlarge` is a scarce instance
type and the zone that has capacity is not knowable in advance. `rig.py launch` tries the
zones in order and places the whole rig in the first that can hold the GPU servers.
"""

import json
import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "nominibatch-precision-gpu-g5-100M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

# ---------------------------------------------------------------- the graph

# ONE GRAPH, PINNED BY MD5, SERVED AT TWO PRECISIONS. hushar's TensorRT provider and
# Triton's `trtexec` must compile the SAME bytes, or the comparison is between two models
# and not two servers. The precision is a BUILDER setting on both sides, not a second
# file, so the md5 below is what every cell of this campaign records whichever precision
# it measured -- which is what makes the fp32/bf16 pair a precision measurement.
#
# Measured from the checked-in artifact, and asserted on the host by `deploy.sh` and again
# by `preflight`.
#
# 31 encoder layers at d_model 512, the depth `generate_benchmark_models.py` solves for a
# 100M target. Same 210 features, same 690-float input, same thirteen ONNX operator types
# and the same `ai.onnx` opset 17 as the 10M and 20M graphs earlier campaigns measured.
# There is no `ai.onnx.ml` node, which is what makes it expressible as a TensorRT engine
# at all: TensorRT has no string type, and the ONNX parser does not implement that domain.
# hushar's string handling happens in Rust, before the graph.
MODELS = {
    "100m": dict(md5="eee2644bd65cd23543ca806f102f5062", bytes=402233324,
                 params=100561925, layers=31, macs_per_row=785000960),
}

# THE PRECISIONS, which in this run are the axis and the only one.
PRECISIONS = ("fp32", "bf16")

# WHAT A 100M GRAPH IS AGAINST THE SIZES THIS SERIES HAS MEASURED. The 10M graph is
# 9,138,693 parameters and 53.3M MAC/row; the 20M is 18,589,701 and 129.0M.
#
#   parameters   9,138,693 -> 100,561,925    11.00x
#   MACs per row     53.3M ->      785.0M    14.73x
#
# The two ratios differ, and that is a property of the design rather than a flaw in the
# sizing: `to_tokens` is a Linear(690 -> 8x512) applied ONCE per row, so it costs
# 2,830,336 parameters and 2,826,240 MACs at any depth, while each encoder layer applies
# its weights to all 8 tokens. Holding the input width at 690 -- mandatory, or the wire
# bytes change and nothing is comparable -- pins that layer as a constant, so arithmetic
# outruns parameters as the stack grows. The report carries BOTH ratios rather than
# rounding either to "10x", and the arithmetic one is what latency should be read against.
PARAMS_VS_10M = MODELS["100m"]["params"] / 9138693
MACS_VS_10M = MODELS["100m"]["macs_per_row"] / 53288960
MODEL_FILE = "bench_model.onnx"
WIDTH, CLASSES = 690, 5

# TWO PACKAGINGS, each carrying the same rows/s as the other, so the grid also prices a
# bigger request against a smaller one at constant throughput. sweep.py owns the
# authoritative copy; this one is documentation for a reader of the rig.
LOADS = {
    "100m": [(100, 100), (500, 20)],      # 10,000 rows/s both
}

# PINNED, NOT SWEPT. The previous campaign swept 2/4/8 over twelve rungs and the session
# count never moved the median by 0.2 ms, so the cells are spent on the precision axis
# instead. No rung configures a mini batch: every request is one engine call of every row
# it carries.
SESSION_COUNTS = (2,)

# This run does not read any earlier campaign's figures. The 10M and 20M parameter and
# arithmetic counts above are properties of those GRAPHS, quoted so the size of this one
# is legible; no latency, throughput or occupancy from a previous run enters this report,
# and `verify.py` fails a summary that imports one.
BASELINE_RUN = None

# ---------------------------------------------------------------- what to launch

# One availability zone for all four hosts: a cross-zone client would add a network hop to
# every latency this run compares, and at 100 rows and 3 ms that is not negligible. Which
# zone is decided at launch, not here: `g5.2xlarge` is scarce, and an earlier attempt at
# this campaign failed with InsufficientInstanceCapacity against a hardcoded us-west-2a.
# These are tried in order and the whole rig lands in the first zone that can hold the
# servers.
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
# release, the CUDA/TensorRT libraries extracted from the image, four TensorRT plans and a
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
        # than only per host: `HusharBench` already carries earlier campaigns' cell ids,
        # and reusing a name would make two runs one time series. sweep.py's CELL_PREFIX
        # is what keeps them apart.
        #
        # ONE LANE PER PRECISION. The graph is the same on both servers -- same file, same
        # md5 -- so a lane is distinguished by the builder setting it measures and nothing
        # else. That also makes the two lanes interchangeable: if the zone can only place
        # one GPU, it runs both precisions one after the other and the only cost is wall
        # clock. `launch.sh` takes `size:md5:precision` triples for exactly that reason.
        "g5-1": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                     ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                     model_id="g5-1", size="100m", precision="fp32"),
        "g5-2": dict(role="server", instance=SERVER_TYPE, arch="x86_64",
                     ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                     model_id="g5-2", size="100m", precision="bf16"),
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
                     model_id="g5-3", size="100m", precision="fp32"),
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
# server -> the PRECISION that lane measures, which in this run is the axis.
PRECISION_OF = {name: spec["precision"] for name, spec in PLAN.items()
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
