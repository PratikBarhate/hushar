#!/usr/bin/env bash
# Prepares one benchmark host from S3 alone, for the rows-and-mini-batch run.
#
#   REV=<rev> ROLE=server|client HUSHAR_BENCH_BUCKET=<bucket> MODEL_SIZE=20m \
#     MODEL_MD5=<md5> deploy.sh
#
# Idempotent: re-running a host that is already prepared re-checks everything and
# rebuilds, which is what makes a partially-failed fan-out safe to repeat.
#
# The ONNX Runtime is the OFFICIAL CPU-ONLY release, fetched here rather than reused from
# whatever the host already had. Earlier experiments on these hosts built libraries with
# XNNPACK, OpenVINO, TensorRT and CUDA compiled in; this run is CPU only, so a library
# that merely *could* register an accelerator disqualifies the host instead of being
# trusted not to.
#
# 1.28.0, which is the engine `nvcr.io/nvidia/tritonserver:26.08-py3` bundles. hushar's
# bindings target ORT_API_VERSION 28 as their default, so both stacks in this run share
# an engine build and the engine version is not a difference between them.
set -euo pipefail

REV="${REV:?REV required}"
ROLE="${ROLE:?ROLE required, server or client}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# The graph every arm of every cell is scored with, on both stacks. Pinned rather than
# discovered: hushar and Triton must load the same bytes, or the comparison is between
# two models and not two servers.
MODEL_MD5="${MODEL_MD5:?MODEL_MD5 required}"
# Which generated model directory to fetch, e.g. 20m, 10m or 50m. Required, because the model
# configuration names this path and a host staged with the wrong one would either fail to
# open the graph or -- worse, if both are present -- serve the other run's model.
MODEL_SIZE="${MODEL_SIZE:?MODEL_SIZE required, e.g. 20m, 10m or 50m}"
MODEL_FILE="${MODEL_FILE:-bench_model.onnx}"
# Where this host's environment block is filed in S3, so the provenance of every figure
# lands beside the cells rather than only on a host that will be destroyed.
RUN_NAME="${RUN_NAME:-design-vs-triton-cpu-arm64-20M}"

export HOME=/root
export DEBIAN_FRONTEND=noninteractive
ORT_VER=1.28.0
ORT_DIR="/root/ort-cpu-$ORT_VER"
DIR="/root/hushar-bench-$REV"

case "$(uname -m)" in
  aarch64) ORT_ARCH=aarch64 ;;
  x86_64)  ORT_ARCH=x64 ;;
  *) echo "unknown arch $(uname -m)"; exit 1 ;;
esac

say() { printf '\n== %s\n' "$1"; }

# 1 · toolchain. A freshly launched host has none of this; an existing one skips it.
say "toolchain"
for tool in cc make protoc; do
  command -v "$tool" >/dev/null 2>&1 || NEED_PKGS=1
done
if [ -n "${NEED_PKGS:-}" ]; then
  dnf install -q -y gcc gcc-c++ make protobuf-compiler tar gzip curl \
    pkgconf-pkg-config >/dev/null 2>&1 \
    || dnf install -q -y gcc gcc-c++ make protobuf-compiler >/dev/null 2>&1
fi
if ! command -v cargo >/dev/null 2>&1 && [ ! -r "$HOME/.cargo/env" ]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
for tool in cc make protoc cargo; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool after install"; exit 1; }
done
echo "   protoc $(protoc --version), $(rustc --version)"

# 2 · the official CPU-only runtime
say "onnx runtime $ORT_VER ($ORT_ARCH, official CPU-only)"
if [ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
  cd /tmp
  curl -sSL -o ort.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-$ORT_ARCH-$ORT_VER.tgz"
  rm -rf "$ORT_DIR" && mkdir -p "$ORT_DIR"
  tar xzf ort.tgz -C "$ORT_DIR" --strip-components=1
  rm -f ort.tgz
fi
# What matters is which provider libraries the package SHIPS. The CPU-only release still
# mentions every provider name in its strings, so grepping symbols proves nothing; an
# accelerator provider is a separate libonnxruntime_providers_<name>.so.
#
# providers_shared is NOT one -- it is the shim the dynamic EP interface loads through,
# and the CPU-only release always ships it -- so it is excluded by name.
ACCEL=$(ls -1 "$ORT_DIR/lib/" | grep 'providers_' | grep -v 'providers_shared' || true)
if [ -n "$ACCEL" ]; then
  echo "REFUSING: accelerator provider library present: $ACCEL"; exit 1
fi
echo "   CPU only, no accelerator provider library present"

# 3 · source
say "source $REV"
rm -rf "$DIR"
aws s3 cp "s3://$BUCKET/dist/hushar-bench-$REV.tgz" /tmp/b.tgz --only-show-errors
mkdir -p "$DIR" && tar xzf /tmp/b.tgz -C "$DIR" --strip-components=1 && rm -f /tmp/b.tgz
sed 's/^/   /' "$DIR/BUNDLE_INFO"

# 4 · model artifacts
say "artifacts"
GEN="$DIR/benchmark-data/generated"
mkdir -p "$GEN"
# The model configuration's model_path is benchmark-data/generated/$MODEL_SIZE/<file>, so
# mirror the S3 layout rather than flattening it: the configuration is the contract.
#
# Any other size directory is removed first. A host that previously served a different
# size would otherwise keep that model on disk, where a stale configuration could still
# find it.
for d in "$GEN"/*/; do
  [ -d "$d" ] || continue
  [ "$(basename "$d")" = "$MODEL_SIZE" ] || { echo "   removing stale $(basename "$d")"; rm -rf "$d"; }
done
if [ "$ROLE" = server ]; then
  aws s3 cp "s3://$BUCKET/models/$MODEL_SIZE/" "$GEN/$MODEL_SIZE/" \
    --recursive --only-show-errors
  GOT=$(md5sum "$GEN/$MODEL_SIZE/$MODEL_FILE" | awk '{print $1}')
  if [ "$GOT" != "$MODEL_MD5" ]; then
    echo "REFUSING: model md5 $GOT, expected $MODEL_MD5"; exit 1
  fi
  echo "   model md5 $GOT as expected, $(stat -c%s "$GEN/$MODEL_SIZE/$MODEL_FILE") bytes"
else
  # The client never opens the graph, so it takes the configurations alone -- and from
  # dist/, not models/: a client instance role can read the distribution prefix and not
  # the model one, which is the right way round for a host that must never be able to
  # serve.
  mkdir -p "$GEN/$MODEL_SIZE"
  aws s3 cp "s3://$BUCKET/dist/configs-$REV.tgz" /tmp/cfg.tgz --only-show-errors
  tar xzf /tmp/cfg.tgz -C "$GEN/$MODEL_SIZE/" && rm -f /tmp/cfg.tgz
  echo "   $(ls -1 "$GEN/$MODEL_SIZE"/*.json | wc -l) configuration(s), no graph"
fi

# Both wires, on both roles: the server to be told the model, the client to be told which
# features to send. The parity arm cannot be improvised on the host -- its configuration
# is derived on the laptop from the real one -- so a missing file is a stale deploy and
# must stop the host here rather than at the first cell.
say "wires"
for f in bench_model_config.json bench_model_tensor_config.json \
         bench_tensor_feature_spec.json; do
  [ -f "$GEN/$MODEL_SIZE/$f" ] || { echo "REFUSING: missing $GEN/$MODEL_SIZE/$f"; exit 1; }
done
# Checked on BOTH roles: the client's request shape comes from these files too, and a
# bundle packaged from a stale generated/ would otherwise be noticed on neither host.
for f in bench_model_config.json bench_model_tensor_config.json; do
  grep -q "generated/$MODEL_SIZE/$MODEL_FILE" "$GEN/$MODEL_SIZE/$f" \
    || { echo "REFUSING: $f does not name generated/$MODEL_SIZE/$MODEL_FILE"; exit 1; }
done
# The two wires must differ in exactly one way: how many features reach the same graph.
REAL=$(python3 -c "
import json; c=json.load(open('$GEN/$MODEL_SIZE/bench_model_config.json'))
print(len(c['vectorization_config']['feature_order']))")
PARITY=$(python3 -c "
import json; c=json.load(open('$GEN/$MODEL_SIZE/bench_model_tensor_config.json'))
v=c['vectorization_config']; o=v['feature_order']
print(len(o), len(v['feature_transformations'][o[0]]['default_val']))")
echo "   features wire: $REAL features; tensor wire: $PARITY (features, width)"
[ "$(echo "$PARITY" | cut -d' ' -f1)" = "1" ] \
  || { echo "REFUSING: the tensor wire does not declare exactly one feature"; exit 1; }

# 5 · env.sh -- run_server.sh and run_client.sh both source this
cat > "$DIR/env.sh" <<EOF
export PROVIDER="\${PROVIDER:-cpu}"
export ROLE="\${ROLE:-$ROLE}"
export ORT_DYLIB_PATH="\${ORT_DYLIB_PATH:-$ORT_DIR/lib/libonnxruntime.so}"
export GEN_DIR="\${GEN_DIR:-benchmark-data/generated/$MODEL_SIZE}"
. "\$HOME/.cargo/env"
EOF

# 6 · native release build, on the host that will be measured
say "build"
cd "$DIR"
if [ "$ROLE" = server ]; then
  cargo build --release -p hushar 2>&1 | tail -3
  test -x target/release/hushar || { echo "build produced no hushar"; exit 1; }
  echo "   built hushar"
else
  # --bins as well as --examples: run_client.sh drives target/release/benchmark-client,
  # and --examples alone silently builds only smoke and inspect_log.
  cargo build --release -p benchmark-client --bins --examples 2>&1 | tail -3
  test -x target/release/benchmark-client || { echo "build produced no client"; exit 1; }
  echo "   built benchmark-client"
fi
echo "== DEPLOY OK $REV $ROLE $(hostname) $(uname -m) $(nproc) cpu"

# 7 · this host's environment, to S3, NOW.
# Captured by the deploy rather than at the end of the run. The previous campaign took its
# hostinfo only when the rig was about to be terminated and nearly lost it: a kernel
# version and an AMI id are the provenance of every figure measured here and cannot be
# reconstructed from anywhere else once the instance is gone.
say "hostinfo -> s3"
HOSTINFO="/tmp/hostinfo-$(hostname).txt"
{
  echo "=== host $(hostname)"
  ROLE="$ROLE" scripts/benchmark/hostinfo.sh 2>&1 || true
  T=$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
  for f in instance-id instance-type ami-id placement/availability-zone; do
    printf '%-14s %s\n' "${f##*/}" \
      "$(curl -sf -H "X-aws-ec2-metadata-token: $T" \
           "http://169.254.169.254/latest/meta-data/$f" || echo '?')"
  done
  printf '%-14s %s\n' os "$(. /etc/os-release && echo "$PRETTY_NAME $VERSION")"
  printf '%-14s %s\n' kernel "$(uname -r)"
  printf '%-14s %s\n' deploy_rev "$REV"
  printf '%-14s %s\n' deployed_at "$(date -u +%FT%TZ)"
} > "$HOSTINFO" 2>&1
aws s3 cp "$HOSTINFO" \
  "s3://$BUCKET/runs/$RUN_NAME/raw/hostinfo/$(hostname).txt" --only-show-errors \
  && echo "   raw/hostinfo/$(hostname).txt" \
  || echo "   WARNING: could not upload hostinfo"
