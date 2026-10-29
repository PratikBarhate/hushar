#!/usr/bin/env python3
"""Starts, stops and reports the hosts this run needs.

    rig.py status
    rig.py start            # the four sweep hosts, then wait for SSM
    rig.py stop             # the four sweep hosts

Stopping is the only state that costs nothing, so a run ends with `stop`. Only the hosts
this run names are ever touched: the two c8i servers and the reference client belong to
other campaigns and are listed in hosts.py as idle so a mistake is visible rather than
silent.

`start` waits for the SSM agent rather than for the instance, because a running instance
with no agent is a host the fan-out cannot reach, and that failure looks like a broken
script instead of a host that is not ready yet.
"""
import sys
import time

import boto3

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hosts  # noqa: E402


def clients():
    session = boto3.Session(profile_name=hosts.PROFILE, region_name=hosts.REGION)
    return session.client("ec2"), session.client("ssm")


def ids():
    return [hosts.ALL[n]["id"] for n in hosts.GROUPS["sweep"]]


def describe(ec2):
    out = {}
    for res in ec2.describe_instances(InstanceIds=ids())["Reservations"]:
        for inst in res["Instances"]:
            out[inst["InstanceId"]] = inst["State"]["Name"]
    return out


def ssm_online(ssm):
    online = set()
    paginator = ssm.get_paginator("describe_instance_information")
    for page in paginator.paginate():
        for info in page["InstanceInformationList"]:
            if info.get("PingStatus") == "Online":
                online.add(info["InstanceId"])
    return online


def status():
    ec2, ssm = clients()
    state = describe(ec2)
    online = ssm_online(ssm)
    print(f"{'host':<8} {'instance':<14} {'state':<10} ssm")
    for name in hosts.GROUPS["sweep"]:
        h = hosts.ALL[name]
        print(f"{name:<8} {h['instance']:<14} {state.get(h['id'], '?'):<10} "
              f"{'online' if h['id'] in online else '-'}")
    return state, online


def start(timeout=900):
    if getattr(hosts, "RIG_TERMINATED", None):
        sys.exit(
            f"the rig was terminated on {hosts.RIG_TERMINATED}; the instance ids in "
            f"hosts.py are historical.\nLaunch replacements, then update hosts.py, "
            f"sweep.py's SERVERS table and launch.sh's HOST_NAME case. Note that a "
            f"replacement rig is a different kernel from the one this run's cells were "
            f"measured on -- see raw/hostinfo.txt."
        )
    ec2, ssm = clients()
    state = describe(ec2)
    todo = [i for i, s in state.items() if s not in ("running", "pending")]
    if todo:
        print(f"starting {len(todo)} instance(s)")
        ec2.start_instances(InstanceIds=todo)
    else:
        print("all instances already running or pending")

    want = set(ids())
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = describe(ec2)
        online = ssm_online(ssm) & want
        running = {i for i, s in state.items() if s == "running"}
        print(f"  {len(running)}/{len(want)} running, {len(online)}/{len(want)} "
              f"SSM online", flush=True)
        if len(online) == len(want):
            print("all hosts reachable over SSM")
            return 0
        time.sleep(20)
    print(f"timed out after {timeout}s; `status` shows which host is missing")
    return 1


def stop():
    ec2, _ = clients()
    state = describe(ec2)
    todo = [i for i, s in state.items() if s in ("running", "pending")]
    if not todo:
        print("nothing running")
        return 0
    print(f"stopping {len(todo)} instance(s)")
    ec2.stop_instances(InstanceIds=todo)
    return 0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("status", "start", "stop"):
        sys.exit("usage: rig.py status|start|stop")
    if sys.argv[1] == "status":
        status()
        return 0
    return start() if sys.argv[1] == "start" else stop()


if __name__ == "__main__":
    sys.exit(main())
