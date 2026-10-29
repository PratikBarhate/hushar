#!/usr/bin/env python3
"""The rig: two ARM64 server hosts, one client each.

This run is ARM64 only, so the two c8i hosts and the reference client stay stopped. Two
server/client pairs give two independent lanes, and a cell costs about ten minutes, so the
pair count is what sets the wall clock.

The two c9g hosts are not interchangeable. The 10M campaign measured a 1.7% placement
offset between them, which repeating a cell does not reduce, so the sweep places whole
COMPARISON GROUPS on one host: every variant a table puts side by side is measured on the
same machine, and only the choice of machine differs between groups. Arm P re-measures the
offset on this build so the report can say what the floor is.
"""

import os

REGION = "us-west-2"
PROFILE = "hushar-bench"
# A bucket name embeds an AWS account id, so it is supplied by the environment rather
# than written down here -- see documentation/benchmarking.md and scripts/check_privacy.py.
BUCKET = os.environ.get("HUSHAR_BENCH_BUCKET", "")

RUN = "minibatch-rows-cpu-arm64-10M"
PREFIX = f"runs/{RUN}"
CW_NAMESPACE = "HusharBench"

# The rig was TERMINATED on 2026-10-23, after this run's 72 cells were collected and
# raw/hostinfo.txt was captured, because the AMI was flagged as an outdated OS. Every
# instance id and address below is therefore historical: it is kept because the cell
# records and the report's provenance name these machines, and because a replacement rig
# needs the shape rather than the ids.
#
# A later run needs new instances, and then: the ids and ips here, the SERVERS table in
# sweep.py, and the HOST_NAME case in launch.sh. It will also be a DIFFERENT KERNEL --
# these cells ran 6.18.44-99.149.amzn2023 on Graviton5, recorded in
# raw/hostinfo.txt -- so cells measured on a replacement rig are a separate environment
# and a table mixing them has to say so.
RIG_TERMINATED = "2026-10-23"

MODEL_SIZE = "10m"
# The graph every arm of every cell is scored with, on both stacks. Pinned rather than
# discovered: hushar and Triton must load the same bytes, or the comparison is between
# two models and not two servers. Regenerated 2026-10-23 for the vectorized round; the
# retired raw and named arms cannot be served by this tree at all.
MODEL_FILE = "bench_model.onnx"
MODEL_MD5 = "5539c97131c896db6c58088d9f9b9ee3"
MODEL_SIZE_BYTES = 36571281

SERVERS = {
    "c9g-1": dict(id="i-09a4d0a122ffc09e7", ip="172.31.46.43",
                  instance="c9g.16xlarge", arch="arm64", model_id="vec-c9g-1"),
    "c9g-2": dict(id="i-08732e68254e73248", ip="172.31.33.183",
                  instance="c9g.16xlarge", arch="arm64", model_id="vec-c9g-2"),
}

CLIENTS = {
    "cl1": dict(id="i-0e3d98cc6d9e1bf01", ip="172.31.41.235",
                instance="m9g.8xlarge", drives="c9g-1"),
    "cl2": dict(id="i-085d4b92443db9366", ip="172.31.43.197",
                instance="m9g.8xlarge", drives="c9g-2"),
}

# Stopped for this run, listed so a reader knows they were deliberately left out rather
# than forgotten, and so `stop` can be pointed at them if one is started by accident.
IDLE = {
    "c8i-1": dict(id="i-097a5c2dffbe3c81c", instance="c8i.16xlarge", arch="x86_64"),
    "c8i-2": dict(id="i-07abf49669e77855e", instance="c8i.16xlarge", arch="x86_64"),
    "ref": dict(id="i-05f3a7a0aa1b89a86", instance="m9g.16xlarge", arch="arm64"),
}

LANES = {"c9g-1": "cl1", "c9g-2": "cl2"}

ALL = {**{k: dict(v, role="server") for k, v in SERVERS.items()},
       **{k: dict(v, role="client") for k, v in CLIENTS.items()}}

GROUPS = {
    "servers": list(SERVERS),
    "clients": list(CLIENTS),
    "sweep": list(SERVERS) + list(CLIENTS),
    "all": list(ALL),
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
    return list(dict.fromkeys(out))
