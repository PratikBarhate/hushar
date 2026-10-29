#!/usr/bin/env python3
"""The rig: four ARM64 server hosts, one client each, LAUNCHED fresh for this run.

The previous campaign kept long-lived instances and started and stopped them per run.
That rig was flagged for an outdated OS and terminated: an AL2023 AMI goes stale in weeks,
and an instance that is merely stopped keeps whatever kernel it was born with. So this run
owns its hosts for its own duration -- `rig.py launch` creates them from the CURRENT AMI,
`rig.py terminate` destroys them -- and nothing survives to go stale.

Instance ids and addresses are therefore NOT written here. `rig.py launch` records them in
rig-state.json beside this file, which is gitignored, and everything else reads them from
there. A file with no ids cannot describe a rig that no longer exists, which is the failure
mode the last campaign hit.

The two server hosts are not interchangeable: the 10M campaign measured a placement offset
between two instances of the same type that repeating a cell does not reduce. So the sweep
places whole COMPARISON GROUPS on one host -- every variant a table puts side by side is
measured on the same machine -- and one configuration is deliberately measured on both so
the report can say how large the offset was on this build.
"""

import json
import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "design-vs-triton-cpu-arm64-20M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

MODEL_SIZE = "20m"
# The graph every arm of every cell is scored with, on both stacks. Pinned rather than
# discovered: hushar and Triton must load the same bytes, or the comparison is between
# two models and not two servers.
#
# THE 2x MODEL, AND WHAT EXACTLY IS DOUBLED. Same 210 features, same 690-float input, same
# thirteen ONNX operator types -- five encoder layers instead of two. That is the only
# change, and it is the one that makes a 10M-against-20M reading mean something: ONNX
# Runtime selects the same kernels for the same matmul shapes, so a difference between the
# campaigns is MORE OF THE SAME WORK rather than different work.
#
#   parameters   9,138,693 -> 18,589,701      2.03x
#   MACs per row    53.3M ->      129.0M      2.42x
#
# The two ratios differ, and the reason is a property of the design rather than a flaw in
# the sizing: `to_tokens` is a Linear(690 -> 8x512) applied ONCE per row, so it costs
# 2,830,336 parameters at any model size. Holding the input width at 690 -- which is
# mandatory, or the wire bytes change and nothing is comparable with the 10M campaign --
# pins that layer as a constant, and every added parameter therefore lands in the trunk.
#
# So a heavier trunk AMORTIZES THE FEATURE PATH BETTER, and hushar's advantage over Triton
# should NARROW. That is the prediction this campaign tests, and it is why the report must
# carry both ratios rather than rounding them to "2x".
MODEL_FILE = "bench_model.onnx"
MODEL_MD5 = "30f9c5330b808a34b55fe88048a20da6"
MODEL_SIZE_BYTES = 74398005

# The campaign this one is read against. Named here so the report cannot drift from the
# numbers above: a ratio is only meaningful against a baseline that is identified.
BASELINE_RUN = "design-vs-triton-cpu-arm64-10M"
BASELINE_MODEL_MD5 = "5539c97131c896db6c58088d9f9b9ee3"
BASELINE_PARAMS = 9138693
PARAMS = 18589701
BASELINE_MACS_PER_ROW = 53288960
MACS_PER_ROW = 128983040

# ---------------------------------------------------------------- what to launch

# One availability zone for all eight hosts. The previous campaign had them all in
# us-west-2a, and a cross-zone client would add a network hop to every latency this run
# compares -- which at 100 rows and 25 ms is small, and at 1 row and 1 ms is not.
SUBNET = "subnet-0cc468f8a7b87ca8d"          # us-west-2a, 172.31.32.0/20
SERVER_SG = "sg-05b6ee0106612a481"           # hushar-bench-server
CLIENT_SG = "sg-0db48fc45a1d1f0f4"           # hushar-bench-client
SERVER_PROFILE = "hushar-bench-server"       # reads models/ and dist/, writes runs/
CLIENT_PROFILE = "hushar-bench-client"       # reads dist/ only: a client must not serve

SERVER_TYPE = "c9g.16xlarge"                 # 64 vCPU Graviton, as the 10M campaign
CLIENT_TYPE = "m9g.8xlarge"                  # 32 vCPU; never above 4% busy last time
ROOT_GB = 80

# The AMI is an INPUT, not a constant, and it is resolved at launch rather than pinned
# here, because pinning an AMI is exactly how the last rig went stale. `rig.py launch`
# takes it in this order:
#
#   1. AMI_ID in the environment            -- an explicit image, e.g. a Nebula AMI id
#   2. AMI_SSM_PARAM in the environment     -- an SSM parameter naming one
#   3. AMI_SSM_DEFAULT below                -- the current Amazon Linux 2023 for arm64
#
# Nebula AMI 2 (NAMI) is the hardened internal image and would be option 1. Two things
# have to be true before it can be: the AWS account must be onboarded to the Nebula AMI
# Registry Service, and the image id must be read from it. Both go through
# nebulaamiregistryservice-ca.corp.amazon.com, which resolves on the Amazon corporate
# network only, so neither can be done from a machine that is merely holding AWS
# credentials. The commands are in README.md next to this file. Until then the default
# below is the current AL2023 release, which is the same distribution the previous
# campaign measured and is freshly patched at launch.
AMI_SSM_DEFAULT = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"

# Tagged so an auditor can see what these are and when they may be deleted, and so a
# forgotten instance is identifiable rather than anonymous.
TAG_PREFIX = "hushar-bench"


# ---------------------------------------------------------------- the launched rig

# Which set of hosts this invocation talks about. A campaign sometimes needs a SECOND rig
# running at the same time as the first -- the 200-row tuning pass was added while the
# head-to-head was already measuring on `main`, and tuning on a busy host would have
# contaminated both. Each set owns its own state file, so neither can terminate the other's
# instances or be confused for them.
RIG_SET = os.environ.get("RIG_SET", "main")

# FOUR PAIRS IN ONE SET, which is the one structural change from the 10M campaign's rig.
#
# That campaign launched two pairs, tuned hushar on one and Triton on the other, and then
# had to launch a SECOND rig when the 200-row tuning pass was added -- because tuning on a
# host that is already measuring contaminates both. Four pairs tune all four combinations
# (hushar and Triton, at 100 and 200 rows) at the same time, which turns a ~3 h serial
# tuning phase into ~1.5 h and removes the second-rig dance entirely.
#
# It also shortens the head-to-head, which matters here for a reason beyond cost: this run
# is driven from a laptop that is expected to disconnect. Three lanes measuring one row
# count each finish in about the time one lane needs for two of them, so there is less
# wall clock in which the rig can be left running unattended after it has finished.
PLANS = {
    "main": {
        "c9g-1": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      model_id="d20-c9g-1"),
        "c9g-2": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      model_id="d20-c9g-2"),
        "c9g-3": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      model_id="d20-c9g-3"),
        "c9g-4": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      model_id="d20-c9g-4"),
        "cl1": dict(role="client", instance=CLIENT_TYPE, arch="arm64", drives="c9g-1"),
        "cl2": dict(role="client", instance=CLIENT_TYPE, arch="arm64", drives="c9g-2"),
        "cl3": dict(role="client", instance=CLIENT_TYPE, arch="arm64", drives="c9g-3"),
        "cl4": dict(role="client", instance=CLIENT_TYPE, arch="arm64", drives="c9g-4"),
    },
    # Kept so a follow-up probe can launch its own hosts without editing this file, the
    # way the 10M campaign's 200-row tuning pass had to.
    "probe": {
        "c9g-5": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      model_id="d20-c9g-5"),
        "cl5": dict(role="client", instance=CLIENT_TYPE, arch="arm64", drives="c9g-5"),
    },
}
if RIG_SET not in PLANS:
    raise SystemExit(f"RIG_SET={RIG_SET!r}; one of {sorted(PLANS)}")

_SUFFIX = "" if RIG_SET == "main" else f"-{RIG_SET}"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"rig-state{_SUFFIX}.json")

PLAN = PLANS[RIG_SET]

LANES = {name: client
         for client, spec in PLAN.items() if spec["role"] == "client"
         for name in [spec["drives"]]}


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

    A host with no recorded id is present in the mapping with `id=None`, so a caller that
    needs one fails saying which host is missing rather than with a KeyError.
    """
    recorded = state().get("hosts", {})
    out = {}
    for name, spec in PLAN.items():
        out[name] = dict(spec, id=recorded.get(name, {}).get("id"),
                         ip=recorded.get(name, {}).get("ip"))
    return out


ALL = live()

GROUPS = {
    "servers": [n for n, h in PLAN.items() if h["role"] == "server"],
    "clients": [n for n, h in PLAN.items() if h["role"] == "client"],
    "sweep": list(PLAN),
    "all": list(PLAN),
}


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
