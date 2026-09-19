#!/usr/bin/env python3
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Runs a shell script on a benchmark host over SSM and prints its output.

    ssm.py <host> -c 'command'          one-liner
    ssm.py <host> -f script.sh          a local script
    ssm.py <host> -f script.sh --timeout 1800

`host` is c9g, c8i, client, or an instance id. SSM truncates captured output at
24 KB, so a long run should redirect to a file on the host and be read back in
pieces.
"""

import argparse
import sys
import time

import boto3

HOSTS = {
    "c9g": "i-09a4d0a122ffc09e7",  # c9g.24xlarge, aarch64
    "c8i": "i-097a5c2dffbe3c81c",  # c8i.24xlarge, x86_64
    "client": "i-05f3a7a0aa1b89a86",  # m9g.16xlarge, aarch64
}
PROFILE = "hushar-bench"
REGION = "us-west-2"


def run(host, script, timeout=900, quiet=False):
    """Sends `script` to `host`, waits, and returns (status, code, stdout, stderr)."""
    instance = HOSTS.get(host, host)
    ssm = boto3.Session(profile_name=PROFILE, region_name=REGION).client("ssm")

    sent = ssm.send_command(
        InstanceIds=[instance],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script], "executionTimeout": [str(max(timeout, 60))]},
        TimeoutSeconds=600,
    )
    cid = sent["Command"]["CommandId"]

    deadline = time.time() + timeout
    inv = None
    while time.time() < deadline:
        time.sleep(3)
        try:
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=instance)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            break
    else:
        return "Timeout", None, "", f"no result within {timeout}s (command {cid})"

    return (
        inv["Status"],
        inv.get("ResponseCode"),
        inv.get("StandardOutputContent", ""),
        inv.get("StandardErrorContent", ""),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("-c", "--command")
    ap.add_argument("-f", "--file")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if args.file:
        script = open(args.file).read()
    elif args.command:
        script = args.command
    else:
        ap.error("one of -c or -f is required")

    status, code, out, err = run(args.host, script, args.timeout)
    print(f"[{status}] exit={code}")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print("--- stderr ---\n" + err.rstrip())
    sys.exit(0 if status == "Success" else 1)


if __name__ == "__main__":
    main()
