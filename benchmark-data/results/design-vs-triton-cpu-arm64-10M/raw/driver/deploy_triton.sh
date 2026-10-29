#!/usr/bin/env bash
# Prepares one host for the Triton arm.
#
#   REV=<rev> ROLE=server|client HUSHAR_BENCH_BUCKET=<bucket> MODEL_SIZE=10m \
#     MODEL_MD5=<md5> deploy_triton.sh
#
# server -- docker, the Triton image, the harness, and the model repository skeleton
# client -- the harness and the Python load generator's dependencies
#
# The Python client, not a Rust one. Its front end builds and serializes every payload
# ONCE into a rotating pool and sends it through a gRPC multicallable with an identity
# serializer, so the per-request cost is the send and nothing else, and it samples its own
# CPU across the measured window so a reader can see it had headroom. Nothing about the
# inference path is Python: the model is served by Triton's C++ ONNX Runtime backend, and
# Triton's Python backend is never involved.
set -euo pipefail

REV="${REV:?REV required}"
ROLE="${ROLE:?ROLE required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3}"
# The graph both stacks score, and the directory it was staged under.
MODEL_SIZE="${MODEL_SIZE:?MODEL_SIZE required, e.g. 10m or 50m}"
MODEL_MD5="${MODEL_MD5:?MODEL_MD5 required}"
MODEL_FILE="${MODEL_FILE:-bench_model.onnx}"
ORT_VER="${ORT_VER:-1.28.0}"

export HOME=/root
. "$HOME/.cargo/env" 2>/dev/null || true
BUNDLE="/root/hushar-bench-$REV"
HARNESS="/root/triton-harness"

say() { printf '\n== %s\n' "$1"; }

say "harness"
rm -rf "$HARNESS" /tmp/th.tgz
aws s3 cp "s3://$BUCKET/dist/triton-harness-$REV.tgz" /tmp/th.tgz --only-show-errors
mkdir -p "$HARNESS"
tar xzf /tmp/th.tgz -C "$HARNESS" --strip-components=1 && rm -f /tmp/th.tgz
echo "   $(ls -1 "$HARNESS" | wc -l) entries"

if [ "$ROLE" = server ]; then
  say "python onnx (make_repo.py reads the graph's signature)"
  # make_repo.py derives the config.pbtxt input and output shapes FROM THE GRAPH rather
  # than from a hand-written table, so the server needs onnx. It will pip-install onnx
  # itself on first use, which fails on a freshly launched AL2023 host two ways: the system
  # python ships without the pip module, and the first cell is the wrong place to discover
  # that. Both are settled here instead.
  if ! python3 -c "import pip" >/dev/null 2>&1; then
    dnf install -q -y python3-pip >/tmp/pip-install.log 2>&1 \
      || { echo "PIP INSTALL FAILED"; tail -20 /tmp/pip-install.log; exit 1; }
  fi
  if ! python3 -c "import onnx" >/dev/null 2>&1; then
    python3 -m pip install --quiet --user onnx 2>&1 | tail -3
  fi
  python3 -c "import onnx; print('   onnx', onnx.__version__)" \
    || { echo "REFUSING: onnx is not importable, so make_repo.py cannot read the graph";
         exit 1; }

  say "docker"
  if ! command -v docker >/dev/null 2>&1; then
    dnf install -y docker >/tmp/docker-install.log 2>&1 \
      || { echo "DOCKER INSTALL FAILED"; tail -20 /tmp/docker-install.log; exit 1; }
  fi
  systemctl enable --now docker >/dev/null 2>&1 || true
  docker --version

  say "image $TRITON_IMAGE"
  if ! docker image inspect "$TRITON_IMAGE" >/dev/null 2>&1; then
    echo "   pulling, this takes a while"
    docker pull "$TRITON_IMAGE" 2>&1 | tail -3
  fi
  docker image inspect "$TRITON_IMAGE" \
    --format '   arch {{.Architecture}}, {{.Size}} bytes'

  # The comparison is only meaningful if both stacks run the same engine. hushar loads the
  # official CPU-only ONNX Runtime, and this reads the version out of the image so the
  # match is checked rather than assumed.
  say "onnx runtime inside the image"
  docker run --rm --entrypoint bash "$TRITON_IMAGE" -c \
    'ls /opt/tritonserver/backends/ | tr "\n" " "; echo; \
     strings /opt/tritonserver/backends/onnxruntime/libonnxruntime.so* 2>/dev/null \
       | grep -Eo "^1\.[0-9]+\.[0-9]+$" | sort -u | tail -3' 2>&1 | sed 's/^/   /'
  IMAGE_ORT=$(docker run --rm --entrypoint bash "$TRITON_IMAGE" -c \
    'strings /opt/tritonserver/backends/onnxruntime/libonnxruntime.so* 2>/dev/null \
       | grep -Eo "^1\.[0-9]+\.[0-9]+$" | sort -u | tail -1')
  if [ "$IMAGE_ORT" != "$ORT_VER" ]; then
    echo "REFUSING: image ships ONNX Runtime '$IMAGE_ORT', but hushar runs $ORT_VER;"
    echo "          the two stacks must share an engine or the comparison is confounded"
    exit 1
  fi
  echo "   engine matches hushar: $IMAGE_ORT"

  say "model"
  M="$BUNDLE/benchmark-data/generated/$MODEL_SIZE/$MODEL_FILE"
  ls -l "$M"
  GOT=$(md5sum "$M" | cut -c1-32)
  # Asserted, not just printed. Triton and hushar must score the same bytes, and this host
  # has both this run's model and whatever an earlier run left behind.
  if [ "$GOT" != "$MODEL_MD5" ]; then
    echo "REFUSING: model md5 $GOT, expected $MODEL_MD5"; exit 1
  fi
  echo "   md5 $GOT as expected"
  mkdir -p /root/triton/models
else
  say "python load generator"
  # Amazon Linux 2023 ships python3 without the pip module, so it is installed before it
  # is used rather than discovered missing by the first cell.
  if ! python3 -c "import pip" >/dev/null 2>&1; then
    dnf install -q -y python3-pip >/tmp/pip-install.log 2>&1 \
      || { echo "PIP INSTALL FAILED"; tail -20 /tmp/pip-install.log; exit 1; }
  fi
  # --user, into /root/.local, so `python3 triton_client.py` finds the packages without
  # any path plumbing in the sweep and without writing into the managed system python.
  # grpcio and numpy both have aarch64 manylinux wheels, so nothing is compiled here;
  # tritonclient is wanted only for its generated service_pb2.
  python3 -m pip install --quiet --user --upgrade pip >/dev/null 2>&1 || true
  python3 -m pip install --quiet --user "tritonclient[grpc]" numpy 2>&1 | tail -3
  python3 -c "
import grpc, numpy
from tritonclient.grpc import service_pb2
print('   grpcio', grpc.__version__, '| numpy', numpy.__version__, '| service_pb2 ok')"
  # The spec the client is told to send, staged by deploy.sh with the configurations.
  SPEC="$BUNDLE/benchmark-data/generated/$MODEL_SIZE/bench_tensor_feature_spec.json"
  [ -f "$SPEC" ] || { echo "REFUSING: missing $SPEC -- run deploy.sh first"; exit 1; }
  python3 -c "
import json; s=json.load(open('$SPEC'))
assert list(s) == ['features'], s
print('   tensor spec:', s)"
  python3 -m py_compile "$HARNESS/triton_client.py" \
    && echo "   triton_client.py compiles"
fi
echo "== TRITON DEPLOY OK $ROLE $(hostname) $(uname -m)"
