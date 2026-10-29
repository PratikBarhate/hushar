#!/usr/bin/env bash
# Prepares one host of the Trainium sweep from S3 alone. Idempotent; re-run after failure.
#
#   REV=<rev> ROLE=server|client SIZES="10m 20m" \
#   HUSHAR_BENCH_BUCKET=<bucket> deploy.sh
#
# server (trn1.2xlarge, Neuron PyTorch 2.9 DLAMI, Ubuntu 24.04): toolchain, the Neuron
#   device and runtime verified, the PyTorch/Neuron venv located by what it can IMPORT,
#   `onnx` added to it, a native hushar build with the Neuron backend, and the NeuronCore
#   sampler.
# client (m9g.8xlarge, AL2023 arm64): toolchain and benchmark-client. The client never
#   opens a model.
#
# TWO PACKAGE MANAGERS, ON PURPOSE. The server has to be the Neuron DLAMI, which AWS
# publishes on Ubuntu, and the client is the same Amazon Linux arm64 image every other run
# in this series drives from. So this script cannot assume one package manager, and the
# branch is explicit rather than a chain of `||` fallbacks that hides which one ran.
#
# WHAT THIS DEPLOY DELIBERATELY DOES NOT DO: build the models. Neuron does not compile
# ONNX, so a NEFF is produced by `torch_neuronx.trace` for ONE fixed batch size, and this
# run needs five per model. Compiling them here would mean a deploy whose length depends
# on the grid, re-done on every re-deploy, with no record of which NEFF a cell ran. They
# are built by the sweep's `prime` phase instead, which caches each in S3 under
# `models/neuron/` AFTER a server has served it, records its SHA-256 and its
# compiled-versus-PyTorch error, and so makes a re-deploy or a second lane free.
set -euo pipefail

REV="${REV:?REV required}"
ROLE="${ROLE:?ROLE required, server or client}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
SIZES="${SIZES:-10m 20m}"
RUN="${RUN:-minibatch-neuron-trn1-10M-20M}"

export HOME=/root
export DEBIAN_FRONTEND=noninteractive
DIR="/root/hushar-bench-$REV"
GEN="$DIR/benchmark-data/generated/neuron"
say() { printf '\n== %s\n' "$1"; }

# ---------------------------------------------------------------- 1 · toolchain
say "toolchain"
# protoc is the one that catches people out: prost-build does not bundle it, so a build
# without protobuf-compiler fails inside build.rs rather than at link time.
if command -v apt-get >/dev/null 2>&1; then
  PM=apt
  # Retried, and its output shown on failure. Both matter: cloud-init's own apt run can
  # still hold /var/lib/dpkg/lock when a deploy starts promptly, and the first version of
  # this sent apt to /dev/null, so a failure under `set -e` printed nothing at all.
  ok=""
  for attempt in 1 2 3 4 5; do
    if apt-get update -qq >/tmp/pkg.log 2>&1 \
       && apt-get install -y -qq build-essential pkg-config protobuf-compiler curl \
            tar gzip python3-pip >>/tmp/pkg.log 2>&1; then
      ok=1; break
    fi
    echo "   apt attempt $attempt failed; $(tail -2 /tmp/pkg.log | tr '\n' ' ')"
    sleep 20
  done
elif command -v dnf >/dev/null 2>&1; then
  PM=dnf
  # --allowerasing: this AMI ships curl-minimal, which conflicts with curl below, and
  # without it the whole transaction aborts. AL2023 also fires dnf-makecache.timer on
  # first boot, so an early deploy can find the rpm lock held for a reason that is gone
  # seconds later.
  ok=""
  for attempt in 1 2 3 4 5; do
    if dnf install -q -y --allowerasing gcc gcc-c++ make protobuf-compiler tar gzip \
         curl pkgconf-pkg-config python3-pip >/tmp/pkg.log 2>&1; then
      ok=1; break
    fi
    echo "   dnf attempt $attempt failed; $(tail -2 /tmp/pkg.log | tr '\n' ' ')"
    sleep 20
  done
else
  echo "REFUSING: neither apt-get nor dnf on this host"; exit 1
fi
if [ -z "$ok" ]; then
  echo "REFUSING: could not install the toolchain after 5 attempts:"
  tail -25 /tmp/pkg.log | sed 's/^/   /'
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1 && [ ! -r "$HOME/.cargo/env" ]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
# shellcheck disable=SC1091
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
for tool in cc make protoc cargo aws; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool after install"; exit 1; }
done
echo "   $PM, protoc $(protoc --version), $(rustc --version)"

# ---------------------------------------------------------------- 2 · source
say "source $REV"
# Wiped rather than updated: a bundle unpacked over a previous rev leaves the previous
# rev's files behind wherever a path moved, and a stale scripts/ is the kind of thing
# that is only noticed by a banner that disagrees with the driver.
rm -rf "$DIR"
mkdir -p "$DIR"
aws s3 cp "s3://$BUCKET/dist/hushar-bench-$REV.tgz" /tmp/bundle.tgz --only-show-errors
tar xzf /tmp/bundle.tgz -C /root
[ -d "$DIR" ] || { echo "bundle did not unpack to $DIR"; ls -la /root | sed 's/^/   /'; exit 1; }
cd "$DIR"
grep -q "commit:    $REV" BUNDLE_INFO 2>/dev/null \
  || { echo "REFUSING: BUNDLE_INFO does not name $REV"; cat BUNDLE_INFO; exit 1; }
echo "   $(grep -c . BUNDLE_INFO) line(s) of provenance, commit $REV"
mkdir -p "$GEN"

# ---------------------------------------------------------------- 3 · role
if [ "$ROLE" = server ]; then
  say "neuron device"
  # The device, the driver and the runtime, each checked separately. A trn1 whose driver
  # did not load still boots, still accepts a deploy, and then fails inside the first
  # cell with an error about a NEFF -- which reads as a model problem rather than a host
  # one. `neuron-ls` output is kept because it names the device and core count the report
  # attributes every figure to.
  [ -c /dev/neuron0 ] || { echo "REFUSING: no /dev/neuron0; the Neuron driver is not loaded"; exit 1; }
  NEURON_LS="$(command -v neuron-ls 2>/dev/null || echo /opt/aws/neuron/bin/neuron-ls)"
  [ -x "$NEURON_LS" ] || { echo "REFUSING: no neuron-ls; this is not a Neuron DLAMI"; exit 1; }
  "$NEURON_LS" | sed 's/^/   /'
  NRT=/opt/aws/neuron/lib/libnrt.so.1
  [ -r "$NRT" ] || { echo "REFUSING: $NRT is missing"; exit 1; }
  # The core count the grid is built around, read from neuron-ls's JSON rather than its
  # table: the table has ONE row per DEVICE and reports the cores in a column, so
  # counting rows says 1 on a two-core trn1.2xlarge. 6 replicas over 2 cores is 3 per
  # core, and on a host with a different count that arithmetic would be wrong.
  CORES_SEEN=$("$NEURON_LS" -j 2>/dev/null | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); raise SystemExit
if isinstance(d, dict):
    d = d.get("neuron_devices") or d.get("devices") or []
print(sum(int(x.get("nc_count") or 0) for x in d) or "")' 2>/dev/null || true)
  echo "   libnrt $(strings "$NRT" 2>/dev/null | grep -oE '2\.[0-9]+\.[0-9]+\.[0-9]+' | head -1), NeuronCores reported: ${CORES_SEEN:-unparsed}"
  command -v neuron-monitor >/dev/null 2>&1 || [ -x /opt/aws/neuron/bin/neuron-monitor ] \
    || { echo "REFUSING: no neuron-monitor; the run has no device sampler without it"; exit 1; }

  say "neuron python environment"
  # The venv is DISCOVERED BY WHAT IT CAN IMPORT, not by its name. AWS has changed that
  # name across releases -- `aws_neuronx_venv_pytorch_2_9` here,
  # `aws_neuronx_venv_torch_neuronx_2_14_0` on the SDK 2.33 image -- so a name glob is a
  # deploy that breaks on the next DLAMI for no reason. The choice is recorded in env.sh
  # so a cell's provenance names the interpreter that built its NEFF.
  VENV=""
  for candidate in $(ls -d /opt/aws_neuronx_venv* /opt/aws_neuron_venv* 2>/dev/null | sort -V -r); do
    [ -x "$candidate/bin/python" ] || continue
    # PATH matters for the TEST as well as for the compile: importing torch_neuronx
    # spawns `libneuronpjrt-path` out of the venv's bin, so an import without it fails
    # with FileNotFoundError and a perfectly good venv is rejected.
    PATH="$candidate/bin:$PATH" "$candidate/bin/python" -c \
      'import torch_neuronx' >/dev/null 2>&1 || continue
    # `trace` is the whole reason this image was chosen. It compiles the parameters in as
    # CONSTANTS, which is what hushar's single-input NEFF contract needs; the dynamo
    # backend that replaced it in torch-neuronx 2.14 lowers a stateless graph whose NEFF
    # takes weights as inputs, and hushar refuses that. A venv that imports but has no
    # trace is therefore not a candidate.
    PATH="$candidate/bin:$PATH" "$candidate/bin/python" -c \
      'import sys, torch_neuronx; sys.exit(0 if hasattr(torch_neuronx, "trace") else 1)' \
      >/dev/null 2>&1 || { echo "   skipping $candidate: torch_neuronx has no trace"; continue; }
    VENV="$candidate"; break
  done
  if [ -z "$VENV" ]; then
    echo "REFUSING: no /opt/aws*_venv* whose python imports torch_neuronx AND has trace."
    echo "   Use a 'Deep Learning AMI Neuron PyTorch 2.9' image; torch-neuronx 2.14"
    echo "   removed trace and its replacement cannot build a NEFF hushar can load."
    echo "   candidates seen:"
    ls -d /opt/aws*venv* 2>/dev/null | sed 's/^/     /' || echo "     none"
    exit 1
  fi
  echo "   $VENV"
  export PATH="$VENV/bin:$PATH"
  "$VENV/bin/python" - <<'PY'
import sys
import torch
import torch_neuronx
print(f"   python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"torch_neuronx {getattr(torch_neuronx, '__version__', 'unknown')}, "
      f"trace {'present' if hasattr(torch_neuronx, 'trace') else 'MISSING'}")
PY
  # generate_benchmark_models.py imports onnx at module level -- compile_neuron.py loads
  # that file to get the ONE definition of the architecture, so the venv needs onnx even
  # though nothing here writes an ONNX graph. Installed only if absent, so a re-deploy
  # does not reach the network.
  "$VENV/bin/python" -c 'import onnx' >/dev/null 2>&1 \
    || { echo "   installing onnx into the venv"; \
         "$VENV/bin/pip" install -q onnx >/tmp/pip.log 2>&1 \
           || { echo "REFUSING: could not install onnx"; tail -15 /tmp/pip.log; exit 1; }; }
  "$VENV/bin/python" -c 'import onnx; print("   onnx", onnx.__version__)'

  say "hushar"
  # bootstrap.sh owns the Neuron path already: it checks neuron-ls and libnrt, writes
  # env.sh with NRT_DYLIB_PATH, and builds hushar with the Neuron backend (which is in
  # the default binary -- Neuron is not a cargo feature). Calling it rather than
  # reimplementing it keeps one rule in one place.
  scripts/benchmark/bootstrap.sh --role server --provider neuron --skip-packages \
    > /tmp/bootstrap.log 2>&1 || { echo "bootstrap failed:"; tail -40 /tmp/bootstrap.log; exit 1; }
  tail -4 /tmp/bootstrap.log | sed 's/^/   /'
  [ -x target/release/hushar ] || { echo "REFUSING: no hushar binary"; exit 1; }

  # The venv and the generated-model root travel in env.sh, so the sweep does not have to
  # rediscover either over SSH on every cell.
  grep -q '^export NEURON_VENV=' env.sh 2>/dev/null || cat >> env.sh <<EOF
export NEURON_VENV="\${NEURON_VENV:-$VENV}"
export NEURON_GEN_DIR="\${NEURON_GEN_DIR:-$GEN}"
EOF
  echo "   env.sh: $(grep -c '^export' env.sh) export(s)"

  say "neuron sampler"
  # Installed at a fixed path the sweep calls, and fetched on its own rather than out of
  # a tarball: a file inside an archive is one unpack away from moving.
  mkdir -p /root/neuron-driver
  aws s3 cp "s3://$BUCKET/dist/sample_neuron-$REV.sh" \
    /root/neuron-driver/sample_neuron.sh --only-show-errors
  chmod +x /root/neuron-driver/sample_neuron.sh
  bash -n /root/neuron-driver/sample_neuron.sh \
    || { echo "REFUSING: the sampler does not parse"; exit 1; }
  echo "   /root/neuron-driver/sample_neuron.sh"

  mkdir -p /tmp/hushar-server
  ROLE=server PROVIDER=neuron scripts/benchmark/hostinfo.sh > /tmp/hushar-server/hostinfo.txt 2>&1 || true
  aws s3 cp /tmp/hushar-server/hostinfo.txt \
    "s3://$BUCKET/runs/$RUN/raw/server/$(hostname)-hostinfo.txt" --only-show-errors || true

else
  say "benchmark-client"
  scripts/benchmark/bootstrap.sh --role client --skip-packages \
    > /tmp/bootstrap.log 2>&1 || { echo "bootstrap failed:"; tail -40 /tmp/bootstrap.log; exit 1; }
  tail -4 /tmp/bootstrap.log | sed 's/^/   /'
  [ -x target/release/benchmark-client ] \
    || { echo "REFUSING: no benchmark-client binary"; exit 1; }
  # The orchestrator runs HERE and reaches the server over SSH, so this host needs an ssh
  # client. It is present on AL2023, but a missing one would surface as every cell failing
  # to start a server, which is a long way from the cause.
  command -v ssh >/dev/null 2>&1 || { echo "REFUSING: no ssh client on the client host"; exit 1; }
  mkdir -p "$GEN"
  for size in $SIZES; do mkdir -p "$GEN/$size"; done
  echo "   staged $GEN for $SIZES; configurations arrive from the server during prime"
fi

echo
echo "== DEPLOY OK $REV $ROLE [$SIZES] $(hostname) $(uname -m) $(nproc) cpu"
