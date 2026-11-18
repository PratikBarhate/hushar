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
import botocore.config

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hosts  # noqa: E402


def clients():
    """Fresh EC2 and SSM clients, with retries sized for a throttled region.

    THE DEFAULT RETRY POLICY IS NOT ENOUGH HERE. botocore's legacy mode makes about four
    attempts, and this run's launch met a `RequestLimitExceeded` that let roughly half of
    identical calls through -- so the same request failed, then succeeded, then failed,
    with no option or zone explaining it. `adaptive` mode adds client-side rate limiting
    on top of more attempts, which is exactly the shape of that failure. Without it the
    outer loop spends two minutes per attempt re-asking a question the SDK should have
    re-asked itself in seconds.

    Rebuilt on every call rather than cached: this profile's keys are refreshed out of
    band, and a `boto3.Session` keeps whatever it resolved at construction.
    """
    session = boto3.Session(profile_name=hosts.PROFILE, region_name=hosts.REGION)
    config = botocore.config.Config(
        retries={"max_attempts": int(os.environ.get("API_ATTEMPTS", "10")),
                 "mode": "adaptive"})
    return session.client("ec2", config=config), session.client("ssm", config=config)


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

def is_credential_error(e):
    code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
    return code in CREDENTIAL_ERRORS


def with_fresh_credentials(fn, tries=60, wait=60, what="call"):
    """Runs `fn(ec2, ssm)` through a credential refresh.

    The clients are rebuilt on every attempt, because a `boto3.Session` caches the keys
    it was built with and this profile's keys are refreshed out of band. Covering only
    the capacity loop was not enough: a long wait expired the session and the NEXT run
    died in `DescribeImages`, before the loop it was protected by.
    """
    for attempt in range(1, tries + 1):
        ec2, ssm = clients()
        try:
            return fn(ec2, ssm)
        except Exception as e:
            if not is_credential_error(e):
                raise
            code = e.response["Error"]["Code"]
            print(f"{what}: credentials unusable ({code}), attempt {attempt}/{tries}; "
                  f"waiting {wait}s for a refresh", flush=True)
            time.sleep(wait)
    sys.exit(f"{what}: credentials never became usable after {tries} attempts")


CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "Unsupported",
                   "InsufficientHostCapacity")
# THROTTLES ARE NOT CAPACITY, AND LOOK NOTHING LIKE IT. `RequestLimitExceeded` from
# RunInstances or CreateFleet reads as "you are calling too fast", but EC2 also returns it
# when an availability zone has an operational issue of its own -- the message says so:
# "throttled due to an operational issue in availability zone usw2-azN". That is not
# something a caller can back off its way out of quickly, and it is PER ZONE: during this
# run's launch, us-west-2b and us-west-2c were throttled while us-west-2a was not. So it
# is retried like capacity rather than raised, and `AZ=<zone>` is how a launch is steered
# to a zone that is answering.
THROTTLE_ERRORS = ("RequestLimitExceeded", "RequestThrottled", "Throttling",
                   "ThrottlingException")
# A FRESH AMI IS NOT VISIBLE EVERYWHERE AT ONCE. The server image is resolved as the
# newest match, which during this run's launch was hours old, and `us-west-2d` answered
# `InvalidSnapshot.NotFound` for its backing snapshot while the other three zones served
# it. That is a zone being behind, not a bad image, so it is skipped like a zone with no
# capacity instead of taking the launch down.
SNAPSHOT_ERRORS = ("InvalidSnapshot.NotFound", "InvalidAMIID.NotFound",
                   "InvalidBlockDeviceMapping")
# AN ACCEPTED LAUNCH IS NOT A RUNNING HOST. `RunInstances` can return an id and then EC2
# fails the boot itself -- `Server.InternalError: Internal error on launch` -- which is
# what a degraded availability zone looks like from the caller's side. This run met it in
# `us-west-2a`, the same zone a throttle message had named, and the first version of this
# file recorded the dead id as a host: `rig-state.json` then described a rig that did not
# exist, and every later command failed somewhere confusing. `confirm_alive` is the check,
# and this code is retried like a zone with no capacity.
INTERNAL_ERRORS = ("Server.InternalError", "Client.InternalError",
                   "Server.InsufficientInstanceCapacity")


class LaunchDied(Exception):
    """An instance EC2 accepted and then failed to boot.

    Shaped like a botocore `ClientError` -- it carries a `response["Error"]["Code"]` --
    so every caller that already classifies failures by error code handles it without a
    second code path.
    """

    def __init__(self, iid, az, reason):
        super().__init__(f"{iid} in {az} died on launch: {reason}")
        self.response = {"Error": {"Code": "Server.InternalError"}}
        self.iid, self.az, self.reason = iid, az, reason


def confirm_alive(ec2, iid, az, timeout=150):
    """Waits until `iid` is running, or raises `LaunchDied` if EC2 kills it first.

    `pending` after the timeout is accepted rather than refused: a large root volume takes
    a while and `wait_for_ssm` is the next gate anyway. What must not pass is an instance
    already on its way out, because that is the one a caller would record and then build
    a whole run on top of.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            inst = describe_eventually(ec2, iid)
        except Exception:
            time.sleep(5)
            continue
        state = inst["State"]["Name"]
        if state == "running":
            return inst
        if state in ("shutting-down", "terminated", "stopping", "stopped"):
            reason = (inst.get("StateReason") or {}).get("Message") \
                or inst.get("StateTransitionReason") or state
            raise LaunchDied(iid, az, reason)
        time.sleep(5)
    return None
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
    """Creates one instance in one specific subnet, for a host that must be co-located.

    Returns only once EC2 has not killed it -- see `confirm_alive`. An accepted id is not
    a host, and recording one that is already shutting down is how a launch produces a
    `rig-state.json` describing a rig that does not exist.
    """
    data = launch_template_data(name, spec, images)
    tags = data.pop("TagSpecifications")
    res = ec2.run_instances(MinCount=1, MaxCount=1, SubnetId=subnet,
                            TagSpecifications=tags, **data)
    inst = res["Instances"][0]
    iid = inst["InstanceId"]
    az = next((z for z, sn in hosts.SUBNETS if sn == subnet), subnet)
    live = confirm_alive(ec2, iid, az)
    return dict(id=iid, ip=(live or inst).get("PrivateIpAddress"),
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


def place_first(ec2, name, spec, images, subnets):
    """Places the scarce first GPU, and reports where.

    WHY THIS IS NOT ONE `create-fleet` CALL, WHICH IS THE BETTER API. An `instant` fleet
    asks every zone in ONE request, so it cannot lose a zone that was free while another
    was being refused -- and that is the right way to ask a capacity question. But it
    reports failure as DATA, in `res["Errors"]`, not as an exception. Under the
    `RequestLimitExceeded` this run met, botocore therefore never saw a throttle to retry
    and the fleet returned throttled five times out of five, with the real capacity
    answer never reached. `run_instances` RAISES the same throttle, so the SDK's adaptive
    retry absorbs it and the genuine `InsufficientInstanceCapacity` underneath becomes
    visible.

    So each subnet is asked directly, in order, and the fleet is kept as the fallback for
    the case it is good at: if every zone answered with nothing but a throttle, one
    atomic request is still worth trying before sleeping.

    Returns (host-record, availability-zone) or (None, failure-text).
    """
    codes = {}
    for az, subnet in subnets:
        try:
            return run_in(ec2, name, spec, subnet, images), az
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code not in CAPACITY_ERRORS and code not in THROTTLE_ERRORS \
                    and code not in SNAPSHOT_ERRORS and code not in INTERNAL_ERRORS:
                raise
            codes[az] = code
            if isinstance(e, LaunchDied):
                print(f"  {az}: {e.reason} -- treating this zone as unavailable",
                      flush=True)
    # Every zone refused. If none of them actually answered the capacity question, the
    # atomic request is still worth one try.
    if subnets and all(c in THROTTLE_ERRORS for c in codes.values()):
        got, where = fleet_instant(ec2, name, spec, images, subnets)
        if got:
            return got, where
        codes["fleet"] = where
    return None, ", ".join(f"{az}:{code}" for az, code in sorted(codes.items()))


def launch():
    # A RECORDED RIG IS ONLY A REASON TO STOP IF IT IS COMPLETE. `adopt` exists precisely
    # so an interrupted launch can be resumed -- a capacity wait, a degraded zone or the
    # laptop can all end one part-way -- and refusing on any recorded host made that
    # unreachable, so the only way to finish a half-placed rig was to terminate the hosts
    # it had already got. A rig missing a host is resumed; a complete one still refuses.
    recorded = hosts.state().get("hosts") or {}
    if recorded and all(recorded.get(n, {}).get("id") for n in hosts.PLAN):
        print("rig-state.json already records a complete rig:")
        status()
        sys.exit("refusing to launch a second rig; `terminate` the recorded one first")
    if recorded:
        missing = [n for n in hosts.PLAN if not recorded.get(n, {}).get("id")]
        print(f"resuming a partly launched rig; missing {', '.join(missing)}\n")

    ec2, ssm = clients()
    if not os.environ.get("HUSHAR_BENCH_BUCKET"):
        sys.exit("set HUSHAR_BENCH_BUCKET: the hosts fetch everything they run from it")

    def _resolve(ec2, ssm):
        return {kind: resolve_one(ec2, ssm, kind) for kind in ("server", "client")}

    resolved = with_fresh_credentials(_resolve, what="resolving images")
    images = {k: v[0] for k, v in resolved.items()}
    provenance = {k: v[1] for k, v in resolved.items()}
    print()

    servers = [n for n, s in hosts.PLAN.items() if s["role"] == "server"]
    subnets = hosts.SUBNETS
    # `AZ` TAKES A LIST, not one zone. A degraded zone is a thing you need to steer AWAY
    # from, and with a single value the only way to do that was to pick one zone and hope
    # it was the right one. `AZ=us-west-2b,us-west-2c` says what this run actually
    # learned: us-west-2a was accepting launches and then failing them.
    only = [z.strip() for z in os.environ.get("AZ", "").split(",") if z.strip()]
    if only:
        known = dict(hosts.SUBNETS)
        unknown = [z for z in only if z not in known]
        if unknown:
            sys.exit(f"AZ names {unknown}, which are not among "
                     f"{[az for az, _ in hosts.SUBNETS]}")
        # IN THE ORDER GIVEN, not the order of the table. `place_first` tries subnets in
        # sequence, so the order is a preference -- which is how you put a zone that
        # accepted a launch and then failed it at the BACK of the queue rather than
        # excluding it outright. A zone that has recovered is still usable; it just
        # should not be asked first.
        subnets = [(z, known[z]) for z in only]
        print(f"zones, in preference order: {[az for az, _ in subnets]}\n")

    def snapshot(created, az, subnet, lanes):
        return dict(run=hosts.RUN, images=provenance, subnet=subnet,
                    availability_zone=az, rig_set=hosts.RIG_SET, lanes=lanes,
                    launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    hosts=created)

    print("looking for hosts this run already owns")
    created = with_fresh_credentials(lambda ec2, ssm: adopt(ec2, images),
                                     what="adopting hosts")
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
                first, where = place_first(ec2, servers[0], hosts.PLAN[servers[0]],
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
            print(f"attempt {attempt}/{tries}: no {hosts.SERVER_TYPE} in "
                  f"{[z for z, _ in subnets]} ({where}); retrying in {wait}s", flush=True)
            time.sleep(wait)
        if not first:
            sys.exit(
                f"no availability zone could hold a {hosts.SERVER_TYPE} after "
                f"{tries} attempts over {tries * wait // 60} minutes.\n"
                f"The last reason was reported above. `RequestLimitExceeded` there is "
                f"NOT your call rate -- EC2 returns it for an operational issue in one "
                f"availability zone, and it is per zone, so try `AZ=<zone>` against a "
                f"zone that answers a `run-instances --dry-run`.\n"
                f"Otherwise: retry later, or choose another single-GPU instance type by "
                f"editing SERVER_TYPE in hosts.py -- but note that changing the vCPU "
                f"count changes what hushar's request path has to work with, and "
                f"changing the GPU changes the device under test, so either is a "
                f"different measurement and the report has to say so."
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
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code not in CAPACITY_ERRORS and code not in THROTTLE_ERRORS \
                    and code not in INTERNAL_ERRORS:
                raise
            print(f"{name}: {code} in {az} -- continuing with {len(lanes)} lane(s); "
                  f"both model sizes will run on {servers[0]} one after the other")
        hosts.save_state(snapshot(created, az, subnet, lanes))

    # A CLIENT IS RETRIED RATHER THAN FATAL. It is an ordinary instance type, so a refusal
    # here is a throttle or a momentary shortage and not a property of the run -- but at
    # this point a scarce GPU is already running, and dying would leave it serving nothing
    # while the bill continues. (`adopt` would find it again on a re-run; that is a
    # recovery, not a reason to need one.)
    for name, spec in hosts.PLAN.items():
        if spec["role"] != "client" or name in created:
            continue
        if spec["drives"] not in created:
            print(f"{name} not launched: it drives {spec['drives']}, which has no host")
            continue
        for attempt in range(1, tries + 1):
            try:
                created[name] = run_in(ec2, name, spec, subnet, images)
                print(f"{name} placed in {az}")
                break
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if code not in CAPACITY_ERRORS and code not in THROTTLE_ERRORS \
                        and code not in CREDENTIAL_ERRORS \
                        and code not in INTERNAL_ERRORS:
                    raise
                print(f"{name}: {code} in {az}, attempt {attempt}/{tries}; "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
                ec2, _ = clients()
        else:
            print(f"{name} could not be placed in {az}; the lane it drives has a server "
                  f"but no client, so `launch` must be re-run to complete the rig")
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
