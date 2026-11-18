#!/usr/bin/env python3
"""Gives each client an SSH key its own server accepts, over SSM.

    keys.py

One ephemeral keypair per lane, so a client can only reach the server it drives. The
private key AND its .pub are written to the client -- OpenSSH reads the .pub beside the
key to decide which key to offer, so a stale .pub from an earlier run makes ssh offer the
wrong key and fail without ever trying the right file -- and the public key replaces this
run's entry in the server's root authorized_keys.

The laptop reaches every host over SSM; this is for the orchestrators, which run inside the
VPC and need a channel that survives being called a few thousand times. Port 22 is already
open on the server security group from the client one.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hosts  # noqa: E402
from fanout import run_one  # noqa: E402

COMMENT = "hushar-ceiling-sweep"


def install(client, server):
    tmp = tempfile.mkdtemp()
    key = os.path.join(tmp, "sweep")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", COMMENT,
                    "-f", key], check=True)
    priv, pub = open(key).read(), open(key + ".pub").read().strip()
    subprocess.run(["rm", "-rf", tmp], check=False)

    client_script = f"""mkdir -p /root/.ssh && chmod 700 /root/.ssh
cat > /root/.ssh/sweep <<'KEY'
{priv}KEY
echo '{pub}' > /root/.ssh/sweep.pub
chmod 600 /root/.ssh/sweep /root/.ssh/sweep.pub
echo client-key-installed"""
    server_script = f"""mkdir -p /root/.ssh && chmod 700 /root/.ssh
grep -v {COMMENT} /root/.ssh/authorized_keys > /root/.ssh/ak.new 2>/dev/null || true
echo '{pub}' >> /root/.ssh/ak.new
mv /root/.ssh/ak.new /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
echo server-key-installed"""

    for name, script in ((client, client_script), (server, server_script)):
        _, status, code, out, err = run_one(name, script, 180, {})
        print(f"{name}: {status} exit={code} {out.strip()} {err.strip()[-200:]}")
        if status != "Success" or code != 0:
            return False

    ip = hosts.ALL[server]["ip"]
    _, status, code, out, err = run_one(
        client,
        f"ssh -i /root/.ssh/sweep -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "
        f"-o BatchMode=yes -o ConnectTimeout=15 root@{ip} "
        f"'echo ssh-ok $(hostname) $(nproc)'",
        180, {})
    print(f"{client} -> {server}: {status} exit={code} {out.strip()} "
          f"{err.strip()[-300:]}")
    return "ssh-ok" in out


def main():
    hosts.resolve("sweep")      # fails here if the rig has not been launched
    ok = True
    for server, client in hosts.LANES.items():
        print(f"== lane {client} -> {server}")
        ok &= install(client, server)
        print()
    if not ok:
        sys.exit("at least one lane has no working SSH channel")
    print("every lane can reach its server over SSH")


if __name__ == "__main__":
    main()
