#!/usr/bin/env python3
"""Creates, inspects and destroys the hosts this run measures on.

    rig.py launch           # four instances from CURRENT images, then wait for SSM
    rig.py status
    rig.py terminate        # destroys them, and forgets their ids

A run owns its hosts for its own duration. An earlier campaign kept long-lived instances
and started and stopped them per run; that rig was flagged for an outdated OS and had to be
terminated mid-campaign, because a stopped instance keeps the kernel it was born with. So
`launch` resolves both images at launch time rather than from pinned ids, and the end of a
run is `terminate` rather than `stop`.

TWO IMAGES, because the two roles are not the same machine. The GPU servers need the NVIDIA
kernel driver, which comes from AWS's Deep Learning Base image for x86_64; the clients are
plain Amazon Linux 2023 for arm64. Each is resolved on its own and each is age-checked, so
a stale GPU image cannot ride in behind a fresh client one.

`terminate` is destructive and irreversible: it takes the hosts and their root volumes.
Everything a report needs is in S3 by then, written cell by cell as each finished, and
hostinfo is captured by the deploy rather than at the end. It still asks before doing it.
"""
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


# ---------------------------------------------------------------- images

def age_days(created):
    stamp = created[:19]
    return (time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))) / 86400


def resolve_one(ec2, ssm, kind):
    """The image for one role, and the provenance of that choice.

    `AMI_ID_SERVER` / `AMI_ID_CLIENT` in the environment win -- that is where a Nebula AMI
    id would go. Otherwise the server image is the newest match of a name filter, because
    the Deep Learning Base image has no SSM alias, and the client image comes from the
    published Amazon Linux alias.
    """
    explicit = os.environ.get(f"AMI_ID_{kind.upper()}")
    if explicit:
        ami, source = explicit, f"AMI_ID_{kind.upper()} in the environment"
    elif kind == "server":
        pattern = os.environ.get("SERVER_AMI_NAME", hosts.SERVER_AMI_NAME)
        found = ec2.describe_images(
            Owners=["amazon"],
            Filters=[{"Name": "name", "Values": [pattern]},
                     {"Name": "state", "Values": ["available"]},
                     {"Name": "architecture", "Values": ["x86_64"]}])["Images"]
        if not found:
            sys.exit(f"no image matches {pattern!r}; set AMI_ID_SERVER instead")
        newest = max(found, key=lambda i: i["CreationDate"])
        ami = newest["ImageId"]
        source = f"newest of {len(found)} matching {pattern!r}"
    else:
        param = os.environ.get("CLIENT_AMI_SSM", hosts.CLIENT_AMI_SSM)
        got = ssm.get_parameters(Names=[param])
        if not got["Parameters"]:
            sys.exit(f"SSM parameter {param} returned nothing; set AMI_ID_CLIENT instead")
        ami = got["Parameters"][0]["Value"]
        source = f"SSM parameter {param}"

    images = ec2.describe_images(ImageIds=[ami])["Images"]
    if not images:
        sys.exit(f"{ami} is not an image this account can see")
    image = images[0]
    want_arch = "x86_64" if kind == "server" else "arm64"
    if image["Architecture"] != want_arch:
        sys.exit(f"{ami} is {image['Architecture']}; the {kind} role needs {want_arch}")

    days = age_days(image["CreationDate"])
    print(f"{kind:<7} : {ami}  {image.get('Name', '?')}")
    print(f"          from {source}, created {image['CreationDate'][:19]}Z, "
          f"{days:.0f} days old")
    if image.get("DeprecationTime"):
        print(f"          deprecated {image['DeprecationTime'][:19]}Z")
    # Age is the whole reason this is resolved rather than pinned, so a stale image is
    # refused rather than quietly measured.
    limit = float(os.environ.get("AMI_MAX_AGE_DAYS", "45"))
    if days > limit and not os.environ.get("AMI_ALLOW_STALE"):
        sys.exit(
            f"REFUSING: {ami} is {days:.0f} days old, over the {limit:.0f}-day limit. "
            f"A rig was terminated once for an outdated OS.\n"
            f"Resolve a current image, or set AMI_ALLOW_STALE=1 to override deliberately."
        )
    return ami, dict(ami=ami, name=image.get("Name"), source=source,
                     created=f"{image['CreationDate'][:19]}Z",
                     architecture=image["Architecture"],
                     age_days_at_launch=round(days, 1))


# ---------------------------------------------------------------- launch

CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "Unsupported",
                   "InsufficientHostCapacity")
# A long capacity wait outlives a credential session. This profile has no
# `credential_process`, so the keys are refreshed out of band and there is a window in
# which every call fails -- which killed a 52-minute wait outright. These are retried
# rather than raised, and the client is rebuilt each attempt so refreshed keys are picked
# up when they arrive.
CREDENTIAL_ERRORS = ("RequestExpired", "ExpiredToken", "ExpiredTokenException",
                     "InvalidClientTokenId", "AuthFailure", "UnrecognizedClientException")


def launch_template_data(name, spec, images):
    """The shape of one instance, as an EC2 Fleet launch template wants it."""
    is_server = spec["role"] == "server"
    return {
        "ImageId": images[spec["ami"]],
        "InstanceType": spec["instance"],
        "SecurityGroupIds": [hosts.SERVER_SG if is_server else hosts.CLIENT_SG],
        "IamInstanceProfile": {
            "Name": hosts.SERVER_PROFILE if is_server else hosts.CLIENT_PROFILE},
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/xvda",
            "Ebs": {"VolumeSize": spec["root_gb"], "VolumeType": "gp3",
                    "Iops": 6000 if is_server else 3000,
                    "Throughput": 500 if is_server else 125,
                    "DeleteOnTermination": True},
        }],
        "TagSpecifications": [{
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
    }


def describe_eventually(ec2, iid, tries=20, wait=3):
    """DescribeInstances until the instance exists.

    A fleet returns an id before the control plane will describe it, and the first version
    of this function called describe once: the `InvalidInstanceID.NotFound` that came back
    propagated out of `launch` and left a RUNNING GPU with nobody recording it. The id is
    real, so the right answer is to wait for it.
    """
    last = None
    for _ in range(tries):
        try:
            res = ec2.describe_instances(InstanceIds=[iid])
            return res["Reservations"][0]["Instances"][0]
        except Exception as e:                        # NotFound, or an empty reservation
            last = e
            time.sleep(wait)
    raise RuntimeError(f"{iid} never became describable: {last}")


def fleet_instant(ec2, name, spec, images, subnets):
    """Asks for one instance in ANY of `subnets`, and lets EC2 choose.

    This is the one API that asks the capacity question properly. Sequential
    `RunInstances` against one subnet at a time asks four separate questions and loses a
    zone that was free at the moment the previous call was being refused -- which is
    exactly what the first attempt at this run saw, being told us-west-2b had capacity by
    the error message from us-west-2a and then finding it gone. An `instant` fleet is one
    request over the whole set.

    Returns (host-record, availability-zone) or (None, failure-text). An instance that
    cannot be described is TERMINATED rather than abandoned: scarce capacity held by a
    host nothing recorded is the worst of both outcomes.
    """
    ltname = f"{hosts.TAG_PREFIX}-{hosts.RUN}-{name}"[:125]
    data = launch_template_data(name, spec, images)
    try:
        ec2.create_launch_template(LaunchTemplateName=ltname,
                                   LaunchTemplateData=data)
    except ec2.exceptions.ClientError as e:
        if "AlreadyExists" not in str(e):
            raise
        version = ec2.create_launch_template_version(
            LaunchTemplateName=ltname, LaunchTemplateData=data,
            SourceVersion="$Latest")["LaunchTemplateVersion"]["VersionNumber"]
        ec2.modify_launch_template(LaunchTemplateName=ltname,
                                   DefaultVersion=str(version))

    res = ec2.create_fleet(
        Type="instant",
        LaunchTemplateConfigs=[{
            "LaunchTemplateSpecification": {"LaunchTemplateName": ltname,
                                            "Version": "$Default"},
            "Overrides": [{"SubnetId": sn} for _, sn in subnets],
        }],
        TargetCapacitySpecification={"TotalTargetCapacity": 1,
                                     "DefaultTargetCapacityType": "on-demand"},
        OnDemandOptions={"AllocationStrategy": "lowest-price"},
    )
    ids = [i for inst in res.get("Instances", []) for i in inst.get("InstanceIds", [])]
    if not ids:
        codes = sorted({e.get("ErrorCode", "?") for e in res.get("Errors", [])}) \
            or ["no capacity reported"]
        return None, ", ".join(codes)
    iid = ids[0]
    try:
        got = describe_eventually(ec2, iid)
    except Exception:
        ec2.terminate_instances(InstanceIds=[iid])
        raise
    return dict(id=iid, ip=got.get("PrivateIpAddress"), instance=spec["instance"],
                role=spec["role"], arch=spec["arch"], ami=images[spec["ami"]],
                subnet=got["SubnetId"]), got["Placement"]["AvailabilityZone"]


def run_in(ec2, name, spec, subnet, images):
    """Creates one instance in one specific subnet, for a host that must be co-located."""
    data = launch_template_data(name, spec, images)
    tags = data.pop("TagSpecifications")
    res = ec2.run_instances(MinCount=1, MaxCount=1, SubnetId=subnet,
                            TagSpecifications=tags, **data)
    inst = res["Instances"][0]
    return dict(id=inst["InstanceId"], ip=inst.get("PrivateIpAddress"),
                instance=spec["instance"], role=spec["role"], arch=spec["arch"],
                ami=images[spec["ami"]], subnet=subnet)


def adopt(ec2, images):
    """Hosts this run already owns, found by tag rather than by the state file.

    `launch` is interrupted by capacity, by an API that is briefly inconsistent, and by
    the laptop it runs on. Any of those can leave a correctly tagged instance running with
    nothing recording it -- which happened once here, with a scarce GPU. So the first thing
    `launch` does is ask EC2 what it already has, and only create what is missing. That
    makes the command idempotent and makes a half-finished launch resumable instead of
    something to clean up by hand.
    """
    found = {}
    res = ec2.describe_instances(Filters=[
        {"Name": "tag:hushar-bench-run", "Values": [hosts.RUN]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]}])
    by_name = {f"{hosts.TAG_PREFIX}-{hosts.RUN}-{n}": n for n in hosts.PLAN}
    for r in res["Reservations"]:
        for inst in r["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            name = by_name.get(tags.get("Name", ""))
            if not name or name in found:
                continue
            spec = hosts.PLAN[name]
            if inst["InstanceType"] != spec["instance"]:
                print(f"  ignoring {inst['InstanceId']}: tagged {name} but it is a "
                      f"{inst['InstanceType']}, not a {spec['instance']}")
                continue
            found[name] = dict(id=inst["InstanceId"],
                               ip=inst.get("PrivateIpAddress"),
                               instance=spec["instance"], role=spec["role"],
                               arch=spec["arch"], ami=inst.get("ImageId"),
                               subnet=inst["SubnetId"], adopted=True)
            print(f"  adopting {name} {inst['InstanceId']} "
                  f"({inst['Placement']['AvailabilityZone']}, "
                  f"{inst.get('PrivateIpAddress')})")
    return found


def launch():
    if hosts.state().get("hosts"):
        print("rig-state.json already records a rig:")
        status()
        sys.exit("refusing to launch a second rig; `terminate` the recorded one first")

    ec2, ssm = clients()
    if not os.environ.get("HUSHAR_BENCH_BUCKET"):
        sys.exit("set HUSHAR_BENCH_BUCKET: the hosts fetch everything they run from it")

    images, provenance = {}, {}
    for kind in ("server", "client"):
        images[kind], provenance[kind] = resolve_one(ec2, ssm, kind)
    print()

    servers = [n for n, s in hosts.PLAN.items() if s["role"] == "server"]
    subnets = hosts.SUBNETS
    only = os.environ.get("AZ")
    if only:
        subnets = [(az, sn) for az, sn in subnets if az == only]
        if not subnets:
            sys.exit(f"AZ={only} is not one of {[az for az, _ in hosts.SUBNETS]}")

    def snapshot(created, az, subnet, lanes):
        return dict(run=hosts.RUN, images=provenance, subnet=subnet,
                    availability_zone=az, rig_set=hosts.RIG_SET, lanes=lanes,
                    launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    hosts=created)

    print("looking for hosts this run already owns")
    created = adopt(ec2, images)
    az = subnet = None
    if created:
        anchor = next((created[n] for n in servers if n in created),
                      next(iter(created.values())))
        subnet = anchor["subnet"]
        az = next((z for z, sn in hosts.SUBNETS if sn == subnet), None)
        print(f"  the rig is already anchored in {az} ({subnet})")

    # 1 · the first GPU server, from any zone that has one. This is the scarce part, and
    #     it is also what decides where the rest of the rig goes.
    tries = int(os.environ.get("CAPACITY_TRIES", "40"))
    wait = int(os.environ.get("CAPACITY_WAIT", "120"))
    if servers[0] not in created:
        first = None
        for attempt in range(1, tries + 1):
            # Rebuilt every attempt, so a credential refresh that lands mid-wait is
            # picked up instead of being cached until the process dies.
            ec2, _ = clients()
            try:
                first, where = fleet_instant(ec2, servers[0], hosts.PLAN[servers[0]],
                                             images, subnets)
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if code not in CREDENTIAL_ERRORS:
                    raise
                print(f"attempt {attempt}/{tries}: credentials unusable ({code}); "
                      f"waiting {wait}s for a refresh", flush=True)
                time.sleep(wait)
                continue
            if first:
                az = where
                break
            print(f"attempt {attempt}/{tries}: no {hosts.SERVER_TYPE} capacity in "
                  f"{[z for z, _ in subnets]} ({where}); retrying in {wait}s", flush=True)
            time.sleep(wait)
        if not first:
            sys.exit(
                f"no availability zone could hold a {hosts.SERVER_TYPE} after "
                f"{tries} attempts over {tries * wait // 60} minutes.\n"
                f"Options: retry later, set AZ=<zone> to wait on one, or choose "
                f"another single-GPU instance type by editing SERVER_TYPE in "
                f"hosts.py -- but note that changing the vCPU count changes what "
                f"hushar's vectorization has to work with, and changing the GPU "
                f"changes the device under test, so either is a different "
                f"measurement and the report has to say so."
            )
        created[servers[0]] = first
        subnet = first["subnet"]
        print(f"{servers[0]} placed in {az} ({subnet})")
    hosts.save_state(snapshot(created, az, subnet, [servers[0]]))

    # 2 · everything else in THAT zone. A client in one zone and its server in another is
    #     a different measurement from the one this report claims, so co-location is not
    #     negotiable -- but a second LANE is. If the zone cannot hold another GPU, the run
    #     drops to one lane and measures both model sizes on it, one after the other.
    lanes = [n for n in servers if n in created]
    for name in servers[1:]:
        if name in created:
            continue
        try:
            created[name] = run_in(ec2, name, hosts.PLAN[name], subnet, images)
            lanes.append(name)
            print(f"{name} placed in {az}")
        except ec2.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in CAPACITY_ERRORS:
                raise
            print(f"{name}: {code} in {az} -- continuing with {len(lanes)} lane(s); "
                  f"both model sizes will run on {servers[0]} one after the other")
        hosts.save_state(snapshot(created, az, subnet, lanes))

    for name, spec in hosts.PLAN.items():
        if spec["role"] != "client" or name in created:
            continue
        if spec["drives"] not in created:
            print(f"{name} not launched: it drives {spec['drives']}, which has no host")
            continue
        created[name] = run_in(ec2, name, spec, subnet, images)
        print(f"{name} placed in {az}")
        hosts.save_state(snapshot(created, az, subnet, lanes))

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
                for h in created.values():
                    if h["id"] == inst["InstanceId"]:
                        h["ip"] = inst.get("PrivateIpAddress")
    hosts.save_state(snapshot(created, az, subnet, lanes))
    print(f"\nrecorded {len(created)} host(s) in {os.path.basename(hosts.STATE)}, "
          f"{len(lanes)} lane(s) in {az}")
    for name, h in sorted(created.items()):
        print(f"  {name:<6} {h['instance']:<14} {h['id']}  {h['ip']}"
              f"{'  (adopted)' if h.get('adopted') else ''}")
    print()
    return wait_for_ssm([h["id"] for h in created.values()])


def wait_for_ssm(ids, timeout=1500):
    """Waits for the SSM agent, not the instance.

    A running instance with no agent is a host the fan-out cannot reach, and that failure
    reads as a broken script rather than a host that is not ready yet. The Deep Learning
    image is large, so it takes noticeably longer to get there than Amazon Linux does.
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
    print(f"run     : {doc.get('run')}")
    print(f"launched: {doc.get('launched')}")
    for kind, image in sorted(doc.get("images", {}).items()):
        print(f"{kind:<8}: {image.get('ami')}  {image.get('name')} "
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
    # The launch templates the capacity fleet needed are this run's too, and a template
    # left behind is the kind of thing nobody finds again.
    for name in hosts.PLAN:
        ltname = f"{hosts.TAG_PREFIX}-{hosts.RUN}-{name}"[:125]
        try:
            ec2.delete_launch_template(LaunchTemplateName=ltname)
            print(f"  deleted launch template {ltname}")
        except ec2.exceptions.ClientError:
            pass
    doc = hosts.state()
    doc["terminated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["terminated_ids"] = {n: h["id"] for n, h in doc.get("hosts", {}).items()}
    doc["hosts"] = {}
    hosts.save_state(doc)
    print(f"ids cleared from {os.path.basename(hosts.STATE)}; a later run must `launch`")
    return 0


COMMANDS = {"launch": launch, "status": lambda: (status(), 0)[1],
            "terminate": terminate}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: rig.py {'|'.join(COMMANDS)}")
    return COMMANDS[sys.argv[1]]() or 0


if __name__ == "__main__":
    sys.exit(main())
