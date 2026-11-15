#!/usr/bin/env python3
"""Asks which single-GPU instance types this account can actually get right now.

    probe_capacity.py [type ...]

`g5.2xlarge` had no on-demand capacity in any us-west-2 zone when this run was first
launched, and there is no API that answers "is there capacity" without asking for the
instance. So this asks for one, in any zone, and TERMINATES IT IMMEDIATELY -- seconds of
on-demand time per type, which is the cheapest honest answer available.

Nothing here is left running. Every instance it gets is terminated before the next type is
tried, and the launch template is deleted at the end.
"""
import sys
import time

import boto3

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hosts  # noqa: E402

# Every one of these carries exactly ONE GPU, so the device under test stays a single
# accelerator. The g6 family all carry one L4; g5 carries one A10G, a generation older.
CANDIDATES = [
    ("g5.2xlarge", "1x A10G, 8 vCPU -- what this run measures"),
    ("g6.4xlarge", "1x L4, 16 vCPU -- same GPU, twice the host CPU"),
    ("g6.xlarge", "1x L4, 4 vCPU -- same GPU, half the host CPU"),
    ("g6.8xlarge", "1x L4, 32 vCPU -- same GPU, four times the host CPU"),
    ("gr6.4xlarge", "1x L4, 16 vCPU, more RAM"),
    ("g6.2xlarge", "1x L4, 8 vCPU -- same host shape, newer GPU"),
    ("g6e.2xlarge", "1x L40S, 8 vCPU -- a much larger GPU"),
]


def probe(ec2, itype, image):
    ltname = f"{hosts.TAG_PREFIX}-capacity-probe"
    data = {
        "ImageId": image,
        "InstanceType": itype,
        "SecurityGroupIds": [hosts.SERVER_SG],
        "IamInstanceProfile": {"Name": hosts.SERVER_PROFILE},
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "TagSpecifications": [{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"{hosts.TAG_PREFIX}-capacity-probe"},
            {"Key": "hushar-bench-ephemeral", "Value": "true"}]}],
    }
    try:
        ec2.create_launch_template(LaunchTemplateName=ltname, LaunchTemplateData=data)
    except ec2.exceptions.ClientError:
        v = ec2.create_launch_template_version(
            LaunchTemplateName=ltname, LaunchTemplateData=data,
            SourceVersion="$Latest")["LaunchTemplateVersion"]["VersionNumber"]
        ec2.modify_launch_template(LaunchTemplateName=ltname, DefaultVersion=str(v))

    res = ec2.create_fleet(
        Type="instant",
        LaunchTemplateConfigs=[{
            "LaunchTemplateSpecification": {"LaunchTemplateName": ltname,
                                            "Version": "$Default"},
            "Overrides": [{"SubnetId": sn} for _, sn in hosts.SUBNETS]}],
        TargetCapacitySpecification={"TotalTargetCapacity": 1,
                                     "DefaultTargetCapacityType": "on-demand"},
        OnDemandOptions={"AllocationStrategy": "lowest-price"},
    )
    ids = [i for inst in res.get("Instances", []) for i in inst.get("InstanceIds", [])]
    if not ids:
        codes = sorted({e.get("ErrorCode", "?") for e in res.get("Errors", [])})
        return False, ", ".join(codes) or "no capacity reported", None
    # Terminated in a `finally`, and only then is the placement read. The first version of
    # this function read the placement first and terminated after, so an exception between
    # the two LEAKED A RUNNING GPU INSTANCE -- which it duly did, and reported as a
    # ClientError rather than as the available capacity it actually was.
    try:
        got = ec2.describe_instances(InstanceIds=ids)["Reservations"][0]["Instances"][0]
        return True, "available", got["Placement"]["AvailabilityZone"]
    finally:
        ec2.terminate_instances(InstanceIds=ids)


def main():
    wanted = sys.argv[1:]
    cands = [(t, d) for t, d in CANDIDATES if not wanted or t in wanted]
    session = boto3.Session(profile_name=hosts.PROFILE, region_name=hosts.REGION)
    ec2 = session.client("ec2")
    image = ec2.describe_images(
        Owners=["amazon"],
        Filters=[{"Name": "name", "Values": [hosts.SERVER_AMI_NAME]},
                 {"Name": "state", "Values": ["available"]},
                 {"Name": "architecture", "Values": ["x86_64"]}])["Images"]
    image = max(image, key=lambda i: i["CreationDate"])["ImageId"]
    print(f"probing with {image} across "
          f"{[az for az, _ in hosts.SUBNETS]}\n")
    print(f"{'type':<14} {'result':<12} {'zone':<12} note")
    results = {}
    for itype, desc in cands:
        try:
            ok, why, az = probe(ec2, itype, image)
        except Exception as e:
            ok, why, az = False, type(e).__name__, None
        results[itype] = ok
        print(f"{itype:<14} {('AVAILABLE' if ok else 'none'):<12} {str(az or '-'):<12} "
              f"{desc if ok else why}")
        time.sleep(2)
    try:
        ec2.delete_launch_template(
            LaunchTemplateName=f"{hosts.TAG_PREFIX}-capacity-probe")
    except Exception:
        pass
    print("\nevery probe instance was terminated as soon as its zone was read")
    left = ec2.describe_instances(
        Filters=[{"Name": "tag:Name",
                  "Values": [f"{hosts.TAG_PREFIX}-capacity-probe"]},
                 {"Name": "instance-state-name",
                  "Values": ["pending", "running", "stopping", "stopped"]}])
    stranded = [i["InstanceId"] for r in left["Reservations"] for i in r["Instances"]]
    print(f"probe instances still alive: {stranded or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
