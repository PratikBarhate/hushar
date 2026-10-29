#!/usr/bin/env python3
"""Creates, inspects and destroys the hosts this run measures on.

    rig.py launch           # four instances from the CURRENT AMI, then wait for SSM
    rig.py status
    rig.py stop             # stops them; the AMI they carry starts going stale
    rig.py start            # starts them again
    rig.py terminate        # destroys them, and forgets their ids

A run owns its hosts for its own duration. The previous campaign kept four long-lived
instances and started and stopped them per run; that rig was flagged for an outdated OS
and had to be terminated mid-campaign, because a stopped instance keeps the kernel it was
born with and an AL2023 AMI is superseded every few weeks. So `launch` resolves the image
at launch time rather than from a pinned id, and the end of a run is `terminate` rather
than `stop`.

`terminate` is destructive and irreversible: it takes the hosts and their root volumes.
Everything a report needs is in S3 by then, written cell by cell as each finished, and
`hostinfo` is captured by the deploy rather than at the end -- which is the one thing the
previous campaign nearly lost. It still asks before doing it.
"""
import json
import os
import sys
import time

import boto3

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hosts  # noqa: E402


def clients():
    session = boto3.Session(profile_name=hosts.PROFILE, region_name=hosts.REGION)
    return session.client("ec2"), session.client("ssm")


def recorded_ids():
    return [h["id"] for h in hosts.live().values() if h["id"]]


def describe(ec2, ids=None):
    ids = ids if ids is not None else recorded_ids()
    if not ids:
        return {}
    out = {}
    for res in ec2.describe_instances(InstanceIds=ids)["Reservations"]:
        for inst in res["Instances"]:
            out[inst["InstanceId"]] = inst["State"]["Name"]
    return out


def ssm_online(ssm):
    online = set()
    for page in ssm.get_paginator("describe_instance_information").paginate():
        for info in page["InstanceInformationList"]:
            if info.get("PingStatus") == "Online":
                online.add(info["InstanceId"])
    return online


# ---------------------------------------------------------------- launch

def resolve_ami(ec2, ssm):
    """The image to launch, and the provenance of that choice.

    Resolved now rather than pinned, because a pinned AMI is how the last rig went stale.
    An explicit AMI_ID wins -- that is where a Nebula AMI id goes -- then an SSM parameter
    naming one, then the current Amazon Linux 2023 release for arm64.
    """
    explicit = os.environ.get("AMI_ID")
    if explicit:
        source = "AMI_ID in the environment"
        ami = explicit
    else:
        param = os.environ.get("AMI_SSM_PARAM", hosts.AMI_SSM_DEFAULT)
        got = ssm.get_parameters(Names=[param])
        if not got["Parameters"]:
            sys.exit(f"SSM parameter {param} returned nothing; set AMI_ID instead")
        ami = got["Parameters"][0]["Value"]
        source = f"SSM parameter {param}"

    images = ec2.describe_images(ImageIds=[ami])["Images"]
    if not images:
        sys.exit(f"{ami} is not an image this account can see")
    image = images[0]
    if image["Architecture"] != "arm64":
        sys.exit(f"{ami} is {image['Architecture']}; this run is ARM64 only")

    # Age is the whole reason this is resolved rather than pinned, so it is reported and
    # a stale image is refused rather than quietly measured. An AL2023 AMI is superseded
    # every few weeks; anything older than 45 days will be flagged while the run is up.
    created = image["CreationDate"][:19]
    age_days = (time.time() - time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))) / 86400
    print(f"image   : {ami}  {image.get('Name', '?')}")
    print(f"          from {source}, created {created}Z, {age_days:.0f} days old")
    if image.get("DeprecationTime"):
        print(f"          deprecated {image['DeprecationTime'][:19]}Z")
    limit = float(os.environ.get("AMI_MAX_AGE_DAYS", "45"))
    if age_days > limit and not os.environ.get("AMI_ALLOW_STALE"):
        sys.exit(
            f"REFUSING: {ami} is {age_days:.0f} days old, over the {limit:.0f}-day limit. "
            f"This rig was terminated once for an outdated OS.\n"
            f"Resolve a current image, or set AMI_ALLOW_STALE=1 to override deliberately."
        )
    return ami, dict(ami=ami, name=image.get("Name"), source=source,
                     created=f"{created}Z", age_days_at_launch=round(age_days, 1))


def launch():
    if hosts.state().get("hosts"):
        print("rig-state.json already records a rig:")
        status()
        sys.exit("refusing to launch a second rig; `terminate` the recorded one first")

    ec2, ssm = clients()
    ami, provenance = resolve_ami(ec2, ssm)
    bucket = os.environ.get("HUSHAR_BENCH_BUCKET", "")
    if not bucket:
        sys.exit("set HUSHAR_BENCH_BUCKET: the hosts fetch everything they run from it")

    created = {}
    for name, spec in hosts.PLAN.items():
        is_server = spec["role"] == "server"
        print(f"launching {name} ({spec['instance']}, {spec['role']})")
        res = ec2.run_instances(
            ImageId=ami,
            InstanceType=spec["instance"],
            MinCount=1, MaxCount=1,
            SubnetId=hosts.SUBNET,
            SecurityGroupIds=[hosts.SERVER_SG if is_server else hosts.CLIENT_SG],
            IamInstanceProfile={
                "Name": hosts.SERVER_PROFILE if is_server else hosts.CLIENT_PROFILE},
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": hosts.ROOT_GB, "VolumeType": "gp3",
                        # The rig is destroyed at the end of the run, so a volume that
                        # outlived its instance would only ever be an orphan.
                        "DeleteOnTermination": True},
            }],
            # No public address and no key pair: the laptop reaches these over SSM and
            # the orchestrators reach the servers over SSH from inside the VPC, with an
            # ephemeral key keys.py installs. Nothing about this rig is reachable from
            # the internet.
            MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
            TagSpecifications=[{
                "ResourceType": rtype,
                "Tags": [
                    {"Key": "Name", "Value": f"{hosts.TAG_PREFIX}-{hosts.RUN}-{name}"},
                    {"Key": "hushar-bench-run", "Value": hosts.RUN},
                    {"Key": "hushar-bench-role", "Value": spec["role"]},
                    {"Key": "hushar-bench-ephemeral", "Value": "true"},
                    {"Key": "hushar-bench-launched", "Value": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                ],
            } for rtype in ("instance", "volume")],
        )
        inst = res["Instances"][0]
        created[name] = dict(id=inst["InstanceId"],
                             ip=inst.get("PrivateIpAddress"),
                             instance=spec["instance"], role=spec["role"])
        # Recorded after EVERY instance, not at the end: a failure half way through must
        # leave behind a file that can terminate what was already created.
        hosts.save_state(dict(run=hosts.RUN, image=provenance, subnet=hosts.SUBNET,
                              rig_set=hosts.RIG_SET,
                              launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              hosts=created))

    # The private address is sometimes absent from the RunInstances response, and the
    # sweep needs it, so it is read back rather than assumed.
    for _ in range(30):
        missing = [n for n, h in created.items() if not h["ip"]]
        if not missing:
            break
        time.sleep(5)
        for res in ec2.describe_instances(
                InstanceIds=[created[n]["id"] for n in missing])["Reservations"]:
            for inst in res["Instances"]:
                for n, h in created.items():
                    if h["id"] == inst["InstanceId"]:
                        h["ip"] = inst.get("PrivateIpAddress")
    hosts.save_state(dict(run=hosts.RUN, image=provenance, subnet=hosts.SUBNET,
                          rig_set=hosts.RIG_SET,
                          launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          hosts=created))
    print(f"\nrecorded {len(created)} host(s) in {os.path.basename(hosts.STATE)}")
    for name, h in sorted(created.items()):
        print(f"  {name:<6} {h['id']}  {h['ip']}")
    print()
    return wait_for_ssm([h["id"] for h in created.values()])


def wait_for_ssm(ids, timeout=1200):
    """Waits for the SSM agent, not the instance.

    A running instance with no agent is a host the fan-out cannot reach, and that failure
    reads as a broken script rather than a host that is not ready yet.
    """
    _, ssm = clients()
    deadline = time.time() + timeout
    want = set(ids)
    while time.time() < deadline:
        online = ssm_online(ssm) & want
        print(f"  {len(online)}/{len(want)} SSM online", flush=True)
        if len(online) == len(want):
            print("all hosts reachable over SSM")
            return 0
        time.sleep(20)
    print(f"timed out after {timeout}s; `status` shows which host is missing")
    return 1


# ---------------------------------------------------------------- the rest

def status():
    doc = hosts.state()
    if not doc.get("hosts"):
        print("no rig recorded; `rig.py launch` creates one")
        return {}, set()
    image = doc.get("image", {})
    print(f"run     : {doc.get('run')}")
    print(f"launched: {doc.get('launched')}")
    print(f"image   : {image.get('ami')}  {image.get('name')} "
          f"(created {image.get('created')})")
    ec2, ssm = clients()
    try:
        state = describe(ec2)
    except Exception as e:                      # a terminated id still describes, but a
        print(f"describe failed: {e}")          # purged one does not
        state = {}
    online = ssm_online(ssm)
    print()
    print(f"{'host':<8} {'instance':<14} {'id':<21} {'ip':<16} {'state':<12} ssm")
    for name, h in sorted(hosts.live().items()):
        print(f"{name:<8} {h['instance']:<14} {str(h['id']):<21} {str(h['ip']):<16} "
              f"{state.get(h['id'], '?'):<12} "
              f"{'online' if h['id'] in online else '-'}")
    return state, online


def start():
    ec2, _ = clients()
    ids = recorded_ids()
    if not ids:
        sys.exit("no rig recorded; `rig.py launch` creates one")
    state = describe(ec2, ids)
    todo = [i for i, s in state.items() if s not in ("running", "pending")]
    if todo:
        print(f"starting {len(todo)} instance(s)")
        ec2.start_instances(InstanceIds=todo)
    else:
        print("all instances already running or pending")
    return wait_for_ssm(ids)


def stop():
    ec2, _ = clients()
    ids = recorded_ids()
    state = describe(ec2, ids)
    todo = [i for i, s in state.items() if s in ("running", "pending")]
    if not todo:
        print("nothing running")
        return 0
    print(f"stopping {len(todo)} instance(s)")
    print("NOTE a stopped instance keeps the kernel it was launched with. If this rig is "
          "finished with, `terminate` it instead -- the last one was flagged for an "
          "outdated OS while stopped.")
    ec2.stop_instances(InstanceIds=todo)
    return 0


def terminate():
    ec2, _ = clients()
    ids = recorded_ids()
    if not ids:
        print("no rig recorded; nothing to terminate")
        return 0
    state = describe(ec2, ids)
    print("about to TERMINATE, which destroys these instances and their root volumes:")
    for name, h in sorted(hosts.live().items()):
        if h["id"]:
            print(f"  {name:<6} {h['instance']:<14} {h['id']}  {state.get(h['id'], '?')}")
    print("\nEverything a report needs is in S3: cells are written as they finish and "
          "hostinfo is captured by the deploy. Anything still only on a host is lost.")
    if os.environ.get("CONFIRM") != "terminate":
        sys.exit("refusing without CONFIRM=terminate in the environment")
    alive = [i for i, s in state.items() if s not in ("terminated", "shutting-down")]
    if alive:
        ec2.terminate_instances(InstanceIds=alive)
        print(f"terminating {len(alive)} instance(s)")
    doc = hosts.state()
    doc["terminated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["terminated_ids"] = {n: h["id"] for n, h in doc.get("hosts", {}).items()}
    doc["hosts"] = {}
    hosts.save_state(doc)
    print(f"ids cleared from {os.path.basename(hosts.STATE)}; a later run must `launch`")
    return 0


COMMANDS = {"launch": launch, "status": lambda: (status(), 0)[1], "start": start,
            "stop": stop, "terminate": terminate}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: rig.py {'|'.join(COMMANDS)}")
    return COMMANDS[sys.argv[1]]() or 0


if __name__ == "__main__":
    sys.exit(main())
