#!/usr/bin/env python3
"""The rig: two c9g.16xlarge CPU server hosts, one client each, LAUNCHED fresh for this run.

WHAT THIS RUN ASKS. `minibatch-rows-cpu-arm64-10M` made rows per request the axis and
found the mini-batch split worth 1.05x at 10 rows rising to 2.64x at 200, with a SMALL
split winning everywhere it was tried -- but it only ever tried 4, 8 and 16, on a 10M
graph, with 4 sessions of 16 threads and the feature log off. Three of those four are
different here, so the optimum is an open question rather than an inherited number:

    AT A FIXED OFFERED LOAD, WHICH MINI-BATCH SIZE IS FASTEST -- ON ONE SESSION, WITH
    IDLE THREADS ASLEEP, WITH ONE REQUEST IN FIVE WRITTEN TO S3, AT TWO MODEL WEIGHTS
    AND TWO REQUEST SIZES?

So `mini_batch_size` is the ONLY knob that moves inside a comparison group. Every group
holds the model, the rows per request and the offered rate constant, and the largest
member of each group is `mini_batch_size == rows`, which is one graph call per request and
therefore the no-split baseline the others are read against.

ONE SESSION, AND WHY THAT IS NOT A CONSTRAINT ON CPU. `threading.sessions_per_model` 1
means one ONNX Runtime session. On an accelerator that would serialise the mini batches --
concurrent `Run()` calls on one device queue do -- which is why the TensorRT campaigns had
to lease a session per piece. On CPU they do not: ONNX Runtime's intra-op pool is shared
across concurrent `Run()` calls on the same session, so the pieces genuinely overlap and
one session is enough. `sessions_per_request` therefore stays 1 and the parallelism comes
entirely from the split, which is the arrangement this run is measuring.

FOUR HOSTS, ONE PER COMPARISON GROUP. `c9g-1` measures 10M at 100 rows, `c9g-2` 10M at
500, `c9g-3` 20M at 100, `c9g-4` 20M at 500. So a lane owns one whole mini-batch sweep --
its four sizes, its unsplit reference and its noise-floor repeats -- and every comparison
the primary question rests on is within one machine. That is the rule this series adopted
after the 10M CPU campaign measured a placement offset between two instances of the same
type that repeating does not reduce. Four lanes cost the same instance-hours as two and
halve the wall clock, which matters because the laptop driving this is expected to
disconnect.

Two readings do cross hosts, and the report names them: the packaging one -- 100 x 250
against 500 x 50, both 25,000 rows/s -- and 10M against 20M. Both are ratios several times
larger than the few percent a placement offset contributes.

ONE AVAILABILITY ZONE, CHOSEN AT LAUNCH RATHER THAN PINNED. Every latency here is measured
over the link between a client and its server, so a cross-zone pair would add a network hop
to one lane and not the other. All eight hosts go in one zone, and `rig.py launch` tries the
zones in order and places the whole rig in the first that can hold the servers.
"""

import json
import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "minibatch-size-cpu-c9g-10M-20M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

# ---------------------------------------------------------------- the graphs

# Both graphs are pinned by md5 rather than discovered, and asserted on the host by
# `deploy.sh` and again by `preflight`. The two are the same design at two weights -- 2
# encoder layers against 5 -- with the same 210 features, the same 690-float input and the
# same thirteen ONNX operator types. So a 10M-against-20M reading is MORE OF THE SAME WORK
# rather than different work.
#
#   parameters   9,138,693 -> 18,589,701      2.03x
#   MACs per row    53.3M ->      129.0M      2.42x
#
# The two ratios differ because `to_tokens` is a Linear(690 -> 8x512) applied ONCE per row,
# so it costs 2,830,336 parameters at any model size; holding the input width at 690 pins
# that layer as a constant and every added parameter lands in the trunk. The report carries
# both ratios rather than rounding them to "2x".
MODELS = {
    "10m": dict(md5="5539c97131c896db6c58088d9f9b9ee3", bytes=36571281,
                params=9138693, layers=2, macs_per_row=53288960),
    "20m": dict(md5="30f9c5330b808a34b55fe88048a20da6", bytes=74398005,
                params=18589701, layers=5, macs_per_row=128983040),
}
PARAM_RATIO = MODELS["20m"]["params"] / MODELS["10m"]["params"]
MACS_RATIO = MODELS["20m"]["macs_per_row"] / MODELS["10m"]["macs_per_row"]
MODEL_FILE = "bench_model.onnx"
WIDTH, CLASSES = 690, 5

# TWO PACKAGINGS PER SIZE, CARRYING THE SAME ROWS PER SECOND. 100 x 250 and 500 x 50 are
# both 25,000 rows/s; 100 x 100 and 500 x 20 are both 10,000. So the grid also prices a
# bigger request against a smaller one at constant throughput, which is the one comparison
# that reads across the two groups of a size. sweep.py owns the authoritative copy; this is
# documentation for a reader of the rig.
LOADS = {
    "10m": [(100, 250), (500, 50)],       # 25,000 rows/s both
    "20m": [(100, 100), (500, 20)],       # 10,000 rows/s both
}

# THE AXIS. Mini-batch size IN ROWS, per row count, largest first -- and the largest is
# the row count itself, which is one graph call and the no-split baseline. Ordering matters
# for a lane that dies part-way: what survives is then a group with its baseline rather
# than a set of ratios with nothing to divide by.
MINI_BATCHES = {
    100: [100, 50, 10, 5],
    500: [500, 100, 50, 10],
}

# PINNED, NOT SWEPT, and each for a reason the report states:
#
#   sessions 1    specified; and on CPU one session already overlaps the pieces, above
#   spin     0    specified; `minibatch-rows-cpu-arm64-10M` also measured it the better
#                 choice for exactly these configurations -- latency equal or slightly
#                 better AND whole-host CPU about a third lower, because the parallelism
#                 comes from the split and a spinning pool competes with it
#   sessions 8    NOT a thread-pool width: it is how many requests the server scores AT
#                 ONCE, because a request leases a session group exclusively while it
#                 scores. At 1 the ceiling is `1 / service_time`, which a probe measured
#                 at 103 req/s against 250 offered -- every cell would have measured
#                 queueing. 8 admits every lane's offered load with room to spare
#   spin     0    specified; `minibatch-rows-cpu-arm64-10M` also measured it the better
#                 choice for exactly these configurations -- latency equal or slightly
#                 better AND whole-host CPU about a third lower, because the parallelism
#                 comes from the split and a spinning pool competes with it
#   width    8    sessions x width = cores, this series' budget rule, so the whole host is
#                 spent and no more
#   overlap  64   `threading.concurrency_per_session`, held CONSTANT across the grid. Left
#                 derived it would be `cores / width` = 8, and the deeper splits would run
#                 in waves -- which a probe showed is not a mild effect: at width 16 every
#                 cell with more pieces than the ceiling COLLAPSED, serving 107 of 250
#                 req/s at 4.8 s on 31% of the host. Setting it means every cell's pieces
#                 overlap fully whatever the width, so the mini-batch SIZE is the only
#                 thing that differs inside a comparison group
SESSIONS = 8
ALLOW_SPINNING = 0
INTRA_WIDTH = 8
CONCURRENCY_PER_SESSION = 64
REFERENCE_INTRA_WIDTH = 8
WORKER_THREADS = 32        # the 10M campaign's phase W showed this is not a lever

# The campaign this one is read against: the same host type, the same engine build, the
# same wire, the same 10M graph -- with 4 sessions of 16 threads, spinning on, the log off,
# and the mini batch swept over 4/8/16 instead of the row count's own divisors.
BASELINE_RUN = "minibatch-rows-cpu-arm64-10M"

# ---------------------------------------------------------------- what to launch

# One availability zone for all four hosts. Which zone is decided at launch, not here:
# these are tried in order and the whole rig lands in the first that can hold the servers.
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

SERVER_TYPE = "c9g.16xlarge"                 # 64 vCPU arm64, 128 GiB, 34 Gbit
CLIENT_TYPE = "m9g.8xlarge"                  # 32 vCPU; never above 4% busy in prior runs
SERVER_CORES = 64
CLIENT_CORES = 32
# No Triton image and no CUDA on this rig -- just two ONNX graphs, an ONNX Runtime CPU
# release and a cargo build -- so the 80 GB the earlier CPU rigs used is right. The
# inference log goes to S3, not to the root volume, so load does not grow this.
SERVER_ROOT_GB = 80
CLIENT_ROOT_GB = 80

# BOTH ROLES ARE PLAIN AMAZON LINUX 2023 FOR arm64, from the published SSM alias. This is
# the change from every GPU rig in this series, where the server needed the Deep Learning
# Base image for its kernel driver and had to be resolved by name filter. There is no
# device here, so there is no driver, and the alias is both newer and simpler. Resolved at
# launch rather than pinned, because pinning an AMI is how a previous long-lived rig went
# stale and had to be terminated mid-campaign.
SERVER_AMI_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
CLIENT_AMI_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
# Kept resolvable so a follow-up can ask for a name-filtered image without a new driver.
SERVER_AMI_NAME = ""

TAG_PREFIX = "hushar-bench"
# The lane this rig belongs to, in the instance Name tag. `rig.py adopt` finds a host it
# already owns by exactly that tag, so without a lane in it a second rig under one run
# name would adopt the first lane's client instead of launching its own.
LANE = "c9g"


def tag_name(name):
    """The instance Name tag, and the launch-template name, for one planned host."""
    return f"{TAG_PREFIX}-{RUN}-{LANE}-{name}"


# ---------------------------------------------------------------- the launched rig

RIG_SET = os.environ.get("RIG_SET", "main")

PLANS = {
    "main": {
        # FOUR LANES, ONE PER COMPARISON GROUP. A lane owns one model at one request size,
        # so its four mini-batch cells, its unsplit reference and its noise-floor repeats
        # are all on ONE machine -- which is what the primary question needs, since the
        # 10M CPU campaign measured a placement offset between two instances of the same
        # type that repeating does not reduce. Four lanes cost the same instance-hours as
        # two and halve the wall clock, which matters because the laptop driving this is
        # expected to disconnect.
        #
        # What crosses hosts is the packaging reading -- 100 x 250 against 500 x 50, both
        # 25,000 rows/s -- and the 10M-against-20M one. Both are ratios several times
        # larger than the few percent a placement offset contributes, and the report says
        # which comparisons are within-host and which are not.
        #
        # model_id here is only the host's label; the CloudWatch `ModelId` dimension is the
        # CELL id, which carries sweep.py's own campaign prefix.
        "c9g-1": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                      model_id="c9g-1", size="10m", loads="100x250"),
        "c9g-2": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                      model_id="c9g-2", size="10m", loads="500x50"),
        "c9g-3": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                      model_id="c9g-3", size="20m", loads="100x100"),
        "c9g-4": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                      model_id="c9g-4", size="20m", loads="500x20"),
        "cl1": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="c9g-1"),
        "cl2": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="c9g-2"),
        "cl3": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="c9g-3"),
        "cl4": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="c9g-4"),
    },
    # Kept so a follow-up probe -- another mini-batch size, another intra-op width, a
    # Triton reference line -- can launch its own hosts without editing this file.
    "probe": {
        "c9g-5": dict(role="server", instance=SERVER_TYPE, arch="arm64",
                      ami="server", root_gb=SERVER_ROOT_GB, cores=SERVER_CORES,
                      model_id="c9g-5", size="10m", loads="100x250"),
        "cl5": dict(role="client", instance=CLIENT_TYPE, arch="arm64",
                    ami="client", root_gb=CLIENT_ROOT_GB, cores=CLIENT_CORES,
                    drives="c9g-5"),
    },
}
if RIG_SET not in PLANS:
    raise SystemExit(f"RIG_SET={RIG_SET!r}; one of {sorted(PLANS)}")

_SUFFIX = "" if RIG_SET == "main" else f"-{RIG_SET}"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"rig-state{_SUFFIX}.json")

PLAN = PLANS[RIG_SET]

# The architecture each ROLE needs, taken from the plan rather than written down twice:
# this rig is arm64 on both sides, where every GPU rig in the series was x86_64 servers
# and arm64 clients, and rig.py checks the resolved image against this.
ARCH_OF = {spec["role"]: spec["arch"] for spec in PLAN.values()}

# server -> the model size that lane measures, and the one load it measures it at. Both
# travel to the lane through launch.sh, so a lane cannot be started against the wrong half
# of the grid by a command line that forgot one of them.
SIZE_OF = {name: spec["size"] for name, spec in PLAN.items()
           if spec["role"] == "server"}
LOADS_OF = {name: spec["loads"] for name, spec in PLAN.items()
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
            f"them in {os.path.basename(STATE)}."
        )
    return names


def rig_env():
    """`RIG` as the sweep wants it: `name=ip;name=ip` for every server in this set."""
    return ";".join(f"{n}={ALL[n]['ip']}" for n in GROUPS["servers"] if ALL[n]["ip"])
