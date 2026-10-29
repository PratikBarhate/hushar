#!/usr/bin/env python3
"""Runs one script on many benchmark hosts at once, over SSM, and reports each.

    fanout.py servers -f deploy.sh --env ROLE=server --timeout 2400
    fanout.py sweep   -c 'nproc; uname -m'
    fanout.py cl1     -c 'tail -5 /root/sweep.log'

SSM rather than SSH because these hosts have no key pair and no public address, and the
laptop running this is outside their VPC. The orchestrators themselves use SSH, which is
a different problem: they run *inside* the VPC and need a channel that survives being
called a few thousand times.

SSM truncates captured output at 24 KB, so anything long should be redirected to a file
on the host and read back in pieces. Output here is tailed for that reason.
"""
import argparse
import concurrent.futures as cf
import sys
import time

import boto3

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hosts  # noqa: E402


def run_one(name, script, timeout, env):
    h = hosts.ALL[name]
    ssm = boto3.Session(profile_name=hosts.PROFILE,
                        region_name=hosts.REGION).client("ssm")
    # Exported before the script body so a script can be used unchanged on every host and
    # differ only by what is passed in.
    preamble = "\n".join(f"export {k}={v!r}" for k, v in env.items())
    inner = f"set -o pipefail\nexport HOME=/root\n{preamble}\n{script}"

    # THE SCRIPT IS HANDED TO BASH EXPLICITLY, and this is not cosmetic. AWS-RunShellScript
    # executes its commands under `/bin/sh`, which on the Ubuntu Neuron DLAMI is dash --
    # so the `set -o pipefail` above died on arrival with "Illegal option -o pipefail",
    # and every host in this rig's server role failed before running a line of its own.
    # The client role is Amazon Linux, where /bin/sh IS bash, so the two roles disagreed
    # about the same payload.
    #
    # Written to a file and then run, rather than piped to `bash -s`: a script read from
    # stdin owns stdin, so any command inside it that happens to read would swallow the
    # rest of the script. Deploys run package managers and installers, which is exactly
    # the wrong place to find that out.
    if "HUSHAR_SSM_SCRIPT" in inner:
        return name, "SendFailed", None, "", \
            "the script contains the heredoc delimiter HUSHAR_SSM_SCRIPT"
    body = (
        'set -e\n'
        'f=/tmp/hushar-ssm.$$.sh\n'
        "cat > \"$f\" <<'HUSHAR_SSM_SCRIPT'\n"
        f"{inner}\n"
        'HUSHAR_SSM_SCRIPT\n'
        'set +e\n'
        '/bin/bash "$f"\n'
        'rc=$?\n'
        'rm -f "$f"\n'
        'exit $rc\n'
    )

    try:
        sent = ssm.send_command(
            InstanceIds=[h["id"]],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [body],
                        "executionTimeout": [str(max(timeout, 60))]},
            TimeoutSeconds=600,
        )
    except Exception as e:
        return name, "SendFailed", None, "", str(e)
    cid = sent["Command"]["CommandId"]

    deadline = time.time() + timeout
    inv = None
    while time.time() < deadline:
        time.sleep(5)
        try:
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=h["id"])
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            break
    else:
        return name, "Timeout", None, "", f"no result within {timeout}s (command {cid})"

    return (name, inv["Status"], inv.get("ResponseCode"),
            inv.get("StandardOutputContent", ""), inv.get("StandardErrorContent", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="host, group, or comma-separated mix")
    ap.add_argument("-c", "--command")
    ap.add_argument("-f", "--file")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--env", action="append", default=[],
                    help="K=V passed to every host; ROLE is filled in per host if unset")
    ap.add_argument("--tail", type=int, default=40)
    args = ap.parse_args()

    script = open(args.file).read() if args.file else args.command
    if not script:
        ap.error("one of -c or -f is required")
    base_env = dict(kv.split("=", 1) for kv in args.env)

    names = hosts.resolve(args.target)
    print(f"-> {len(names)} host(s): {', '.join(names)}\n", flush=True)

    failures = []
    with cf.ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {}
        for n in names:
            env = dict(base_env)
            # A script that behaves differently on a server and a client should not need
            # two invocations, so the role is always available to it.
            env.setdefault("ROLE", hosts.ALL[n]["role"])
            env.setdefault("HOST_NAME", n)
            futures[pool.submit(run_one, n, script, args.timeout, env)] = n
        for fut in cf.as_completed(futures):
            name, status, code, out, err = fut.result()
            ok = status == "Success" and code == 0
            print("=" * 70)
            print(f"{name:<8} {hosts.ALL[name]['instance']:<14} [{status}] exit={code}")
            print("=" * 70)
            for blob, label in ((out, ""), (err, "--- stderr ---")):
                if blob.strip():
                    if label:
                        print(label)
                    lines = blob.rstrip().splitlines()
                    if len(lines) > args.tail:
                        print(f"... {len(lines) - args.tail} earlier line(s) elided")
                    print("\n".join(lines[-args.tail:]))
            print(flush=True)
            if not ok:
                failures.append(name)

    if failures:
        print(f"FAILED on {len(failures)} host(s): {', '.join(sorted(failures))}")
        sys.exit(1)
    print(f"OK on all {len(names)} host(s)")


if __name__ == "__main__":
    main()
