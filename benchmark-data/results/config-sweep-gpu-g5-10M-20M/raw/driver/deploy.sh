#!/usr/bin/env bash
# Prepares one host of the GPU sweep from S3 alone. Idempotent; re-run after any failure.
#
#   REV=<rev> ROLE=server|client MODEL_SPECS="10m:<md5> 20m:<md5>" \
#   HUSHAR_BENCH_BUCKET=<bucket> deploy.sh
#
# server (g5.2xlarge, Deep Learning Base GPU AMI, AL2023): toolchain, the graphs, docker
#   and the Triton image, the CUDA/cuDNN/TensorRT libraries extracted FROM that image, an
#   ONNX Runtime GPU release chosen to match the image's CUDA major, a native hushar build
#   with the tensorrt feature, and one prebuilt TensorRT plan per graph so no cell pays
#   for trtexec.
# client (m9g.8xlarge, AL2023): toolchain, benchmark-client built natively, the Triton
#   harness and its Python gRPC front end. The client never opens a graph.
#
# EVERY SIZE IS STAGED IN ONE PASS. A lane measures one model at a time, so a per-size
# deploy looks tempting -- but step 2 wipes $DIR, which is where the graphs live, so a
# second deploy would take the first size's artifacts with it. On a single-lane rig that
# runs both sizes one after the other, that would mean being present between them. Staging
# both up front costs one extra trtexec build and makes the rest of the run unattended.
#
# THE ONE IDEA THAT MAKES THE COMPARISON FAIR. Triton's `tensorrt` backend serves an engine
# that `trtexec` compiled; hushar's ONNX Runtime TensorRT provider compiles the same ONNX
# bytes at first use. Those are only the same engine if the same TensorRT built them -- and
# the way to guarantee that is not to pin a version triple and hope, but to give hushar the
# TensorRT libraries COPIED OUT OF THE TRITON IMAGE. ONNX Runtime's provider shared object
# is a thin shim over `libnvinfer`; point it at the image's copy and both stacks are driving
# literally the same builder. The ONNX Runtime download is then chosen by the image's CUDA
# major rather than written down, and `ldd` is the gate: an unresolved symbol here is a
# mismatch that would otherwise have surfaced as a confusing first-cell failure.
set -euo pipefail

REV="${REV:?REV required}"
ROLE="${ROLE:?ROLE required, server or client}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# `size:md5 size:md5`. The md5 travels WITH the size rather than as a parallel variable,
# so the two cannot be paired up wrongly.
MODEL_SPECS="${MODEL_SPECS:-${MODEL_SIZE:-}:${MODEL_MD5:-}}"
[ "$MODEL_SPECS" != ":" ] || { echo "set MODEL_SPECS, e.g. '10m:<md5> 20m:<md5>'"; exit 1; }
MODEL_FILE="${MODEL_FILE:-bench_model.onnx}"
# 1.28 is the ONNX Runtime line onnxrt-rs's bindings target (ORT_API_VERSION 28), so the
# FFI layer has no version skew. The patch release is the newest of that line.
ORT_VER="${ORT_VER:-1.28.3}"
# WHY 26.05 AND NOT SOMETHING NEWER. Triton moved to TensorRT 11 at 26.06, and no ONNX
# Runtime GPU release links it -- 1.28.3 through 1.31.0 all carry
# `NEEDED libnvinfer.so.10`. Read out of the image configs:
#
#   26.06 - 26.08   TensorRT 11.0 - 11.2   libnvinfer.so.11   no ORT pairs with it
#   26.05           TensorRT 10.16.1       libnvinfer.so.10   <- newest that pairs
#   26.01 - 26.04   TensorRT 10.14 - 10.16 libnvinfer.so.10
#
# 26.05 also ships CUDA 13.2 and cuDNN 9.22, whose SONAMEs (`libcudart.so.13`,
# `libcudnn.so.9`) are the ones the ORT provider wants. Crossing a SONAME major with a
# symlink would be an ABI lie, so the image is what moves, not the linkage. And since
# hushar loads the TensorRT copied out of THIS image, both stacks compile with TensorRT
# 10.16.1 whatever 10.x minor ONNX Runtime happened to be built against.
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.05-py3}"
ROWS="${ROWS:-100}"
RUN="${RUN:-config-sweep-gpu-g5-10M-20M}"

export HOME=/root
DIR="/root/hushar-bench-$REV"
HARNESS="/root/triton-harness"
ORT_DIR="/root/ort-gpu-$ORT_VER"
NV="/root/nvlibs"          # CUDA, cuDNN and TensorRT, extracted from the Triton image
CACHE="/root/trt-cache"    # hushar's ONNX Runtime TensorRT engine caches, one per size
say() { printf '\n== %s\n' "$1"; }

# 1 · toolchain
say "toolchain"
# dnf is run with --allowerasing and RETRIED, and its output is shown when it fails.
# All three matter, and each cost a deploy to learn:
#   * this AMI ships `curl-minimal`, which CONFLICTS with the `curl` in the list below,
#     so without --allowerasing the whole transaction aborts;
#   * AL2023 fires `dnf-makecache.timer` on first boot, so a deploy that starts promptly
#     can find the rpm lock held and fail for a reason that is gone seconds later;
#   * the first version of this sent dnf to /dev/null, so when it failed under `set -e`
#     the script died having printed only "== toolchain" and nothing else at all.
PKGS="gcc gcc-c++ make protobuf-compiler tar gzip curl pkgconf-pkg-config python3-pip"
dnf_ok=""
for attempt in 1 2 3 4 5; do
  if dnf install -q -y --allowerasing $PKGS >/tmp/dnf.log 2>&1; then
    dnf_ok=1
    break
  fi
  echo "   dnf attempt $attempt failed; $(tail -2 /tmp/dnf.log | tr '\n' ' ')"
  sleep 20
done
if [ -z "$dnf_ok" ]; then
  echo "REFUSING: could not install the toolchain after 5 attempts:"
  tail -25 /tmp/dnf.log | sed 's/^/   /'
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1 && [ ! -r "$HOME/.cargo/env" ]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
for tool in cc make protoc cargo; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool after install"; exit 1; }
done
echo "   protoc $(protoc --version), $(rustc --version)"
python3 -c "import pip" >/dev/null 2>&1 \
  || { echo "REFUSING: python3 has no pip module after installing python3-pip"; exit 1; }

# 2 · source
say "source $REV"
rm -rf "$DIR"
aws s3 cp "s3://$BUCKET/dist/hushar-bench-$REV.tgz" /tmp/b.tgz --only-show-errors
mkdir -p "$DIR" && tar xzf /tmp/b.tgz -C "$DIR" --strip-components=1 && rm -f /tmp/b.tgz
sed 's/^/   /' "$DIR/BUNDLE_INFO"

# 3 · model artifacts for every size, each checked rather than trusted
say "artifacts"
for spec in $MODEL_SPECS; do
  SIZE="${spec%%:*}"; WANT="${spec##*:}"
  [ -n "$SIZE" ] && [ -n "$WANT" ] || { echo "REFUSING: bad spec '$spec'"; exit 1; }
  GEN="$DIR/benchmark-data/generated/$SIZE"
  mkdir -p "$GEN"
  if [ "$ROLE" = server ]; then
    aws s3 cp "s3://$BUCKET/models/$SIZE/" "$GEN/" --recursive --only-show-errors \
      --exclude '*' --include "$MODEL_FILE" --include 'bench_model_config.json' \
      --include 'bench_model_tensor_config.json' --include 'bench_tensor_feature_spec.json'
    GOT=$(md5sum "$GEN/$MODEL_FILE" | cut -c1-32)
    [ "$GOT" = "$WANT" ] || {
      echo "REFUSING: $SIZE model md5 $GOT, expected $WANT"; exit 1; }
    echo "   $SIZE md5 $GOT as expected, $(stat -c%s "$GEN/$MODEL_FILE") bytes"
  else
    # The client reads a model configuration to know which features to send, but its role
    # can read dist/ and not models/, so the configurations travel as their own object.
    aws s3 cp "s3://$BUCKET/dist/configs-$SIZE-$REV.tgz" /tmp/cfg.tgz --only-show-errors
    tar xzf /tmp/cfg.tgz -C "$GEN" && rm -f /tmp/cfg.tgz
    echo "   $SIZE configurations staged"
  fi
  for f in bench_model_config.json bench_model_tensor_config.json \
           bench_tensor_feature_spec.json; do
    [ -f "$GEN/$f" ] || { echo "REFUSING: missing $GEN/$f"; exit 1; }
  done
  grep -q "generated/$SIZE/$MODEL_FILE" "$GEN/bench_model_config.json" \
    || { echo "REFUSING: $SIZE config does not name generated/$SIZE/$MODEL_FILE"; exit 1; }
done

if [ "$ROLE" = server ]; then
  # 4 · docker, the Triton image, and what accelerator stack it ships
  say "docker and the triton image"
  systemctl enable --now docker >/dev/null 2>&1 || true
  docker --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | sed 's/^/   /'
  # Persistence mode keeps the driver resident, so a cell's first request does not pay
  # for context creation. Best effort: not every instance family permits it.
  nvidia-smi -pm 1 >/dev/null 2>&1 && echo "   persistence mode on" \
    || echo "   persistence mode unavailable (not fatal)"
  docker pull "$TRITON_IMAGE" 2>&1 | tail -2 | sed 's/^/   /'
  docker run --rm --gpus all "$TRITON_IMAGE" nvidia-smi -L 2>&1 | sed 's/^/   /' | tail -2 \
    || { echo "REFUSING: the container cannot see the GPU"; exit 1; }

  # Discovered, not declared. The whole point is that hushar ends up on the image's own
  # TensorRT, so the image is the authority on which ONNX Runtime build to fetch.
  read -r IMG_TRT IMG_CUDA < <(docker run --rm --entrypoint bash "$TRITON_IMAGE" -c '
    trt=$(ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.*.*.* 2>/dev/null \
          | sed "s/.*libnvinfer\.so\.//" | sort -V | tail -1)
    [ -n "$trt" ] || trt=$(ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.* 2>/dev/null \
          | sed "s/.*libnvinfer\.so\.//" | sort -V | tail -1)
    cuda=$(ls /usr/local/cuda/lib64/libcudart.so.* /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.* 2>/dev/null \
          | sed "s/.*libcudart\.so\.//" | sort -V | tail -1)
    echo "$trt ${cuda%%.*}"')
  [ -n "$IMG_TRT" ] && [ -n "$IMG_CUDA" ] || {
    echo "REFUSING: could not read TensorRT/CUDA versions from $TRITON_IMAGE"; exit 1; }
  ORT_CUDA="cuda$IMG_CUDA"
  echo "   image ships TensorRT $IMG_TRT on CUDA $IMG_CUDA -> onnx runtime $ORT_CUDA build"

  # 5 · the GPU libraries hushar's provider dlopens, copied OUT of that image
  say "gpu libraries from the image"
  rm -rf "$NV" && mkdir -p "$NV"
  docker run --rm --entrypoint bash -v "$NV:/out" "$TRITON_IMAGE" -c '
    set -e
    cp -a /usr/lib/x86_64-linux-gnu/libnvinfer*.so* /out/
    cp -a /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so* /out/ 2>/dev/null || true
    cp -a /usr/lib/x86_64-linux-gnu/libcudnn*.so* /out/ 2>/dev/null || true
    for lib in libcudart libcublas libcublasLt libcufft libcurand libnvrtc \
               libnvrtc-builtins libnvJitLink libcupti libnvperf_host; do
      cp -a /usr/local/cuda/lib64/$lib.so* /out/ 2>/dev/null \
        || cp -a /usr/local/cuda/targets/x86_64-linux/lib/$lib.so* /out/ 2>/dev/null || true
    done
    ls /out | wc -l' | sed 's/^/   /'
  echo "   $(ls "$NV" | wc -l) files; TensorRT $(ls "$NV"/libnvinfer.so.* \
    | sed 's/.*libnvinfer\.so\.//' | sort -V | tail -1)"

  # 6 · ONNX Runtime GPU release, matched to the image's CUDA major
  say "onnx runtime $ORT_VER ($ORT_CUDA)"
  if [ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
    cd /tmp
    URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-x64-gpu_$ORT_CUDA-$ORT_VER.tgz"
    curl -fsSL -o ort.tgz "$URL" || { echo "REFUSING: cannot fetch $URL"; exit 1; }
    rm -rf "$ORT_DIR" && mkdir -p "$ORT_DIR"
    tar xzf ort.tgz -C "$ORT_DIR" --strip-components=1 && rm -f ort.tgz
  fi
  ls -1 "$ORT_DIR/lib/" | grep providers_ | sed 's/^/   /'
  export LD_LIBRARY_PATH="$NV:$ORT_DIR/lib"
  # The gate. A provider shared object with an unresolved dependency means the ONNX
  # Runtime build and the image's accelerator stack disagree, and the honest place to
  # find that out is here and not in the first cell.
  MISSING=$(ldd "$ORT_DIR/lib/libonnxruntime_providers_tensorrt.so" | grep 'not found' || true)
  [ -z "$MISSING" ] || {
    echo "REFUSING: the TensorRT provider has unresolved dependencies against the"
    echo "          libraries $TRITON_IMAGE ships (TensorRT $IMG_TRT, CUDA $IMG_CUDA):"
    echo "$MISSING" | sed 's/^/            /'
    echo "          Pick an ONNX Runtime whose TensorRT major matches the image, or an"
    echo "          image whose TensorRT major matches ONNX Runtime $ORT_VER."
    exit 1; }
  echo "   the TensorRT provider resolves every dependency:"
  ldd "$ORT_DIR/lib/libonnxruntime_providers_tensorrt.so" \
    | grep -E 'nvinfer|cudart|cudnn' | sed 's/^/   /'

  # 7 · env.sh -- run_server.sh sources this. $NV first, so the image's TensorRT wins.
  cat > "$DIR/env.sh" <<EOF
export PROVIDER="\${PROVIDER:-tensorrt}"
export ROLE="\${ROLE:-server}"
export ORT_DYLIB_PATH="\${ORT_DYLIB_PATH:-$ORT_DIR/lib/libonnxruntime.so}"
export LD_LIBRARY_PATH="$NV:$ORT_DIR/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export TRT_CACHE="$CACHE"
export IMAGE_TENSORRT="$IMG_TRT"
export IMAGE_CUDA_MAJOR="$IMG_CUDA"
export ORT_CUDA_BUILD="$ORT_CUDA"
. "\$HOME/.cargo/env"
EOF

  # 8 · native build with the tensorrt feature, on the host that will be measured
  say "build"
  cd "$DIR"
  cargo build --release -p hushar --features cpu,tensorrt 2>&1 | tail -3
  test -x target/release/hushar || { echo "build produced no hushar"; exit 1; }
  echo "   built hushar (cpu,tensorrt)"

  # 9 · the Triton harness, and one prebuilt plan per graph
  say "triton harness and the tensorrt plans"
  rm -rf "$HARNESS" /tmp/th.tgz
  aws s3 cp "s3://$BUCKET/dist/triton-harness-$REV.tgz" /tmp/th.tgz --only-show-errors
  mkdir -p "$HARNESS" && tar xzf /tmp/th.tgz -C "$HARNESS" --strip-components=1
  rm -f /tmp/th.tgz
  mkdir -p /root/triton/models /root/triton/plans /root/gpu-driver
  aws s3 cp "s3://$BUCKET/dist/sample_gpu-$REV.sh" /root/gpu-driver/sample_gpu.sh \
    --only-show-errors && chmod +x /root/gpu-driver/sample_gpu.sh
  # Building the plans at DEPLOY, not in a cell: trtexec on a 50M graph takes minutes, and
  # a cell that pays for it is measuring the compiler.
  for spec in $MODEL_SPECS; do
    SIZE="${spec%%:*}"; WANT="${spec##*:}"
    mkdir -p "$CACHE/$SIZE"
    BUNDLE="$DIR" REPO=/root/triton/models MODEL_SIZE="$SIZE" MODEL_MD5="$WANT" \
      INSTANCES=1 MAX_BATCH=0 ROWS="$ROWS" TRITON_IMAGE="$TRITON_IMAGE" \
      python3 "$HARNESS/make_trt_repo.py" 2>&1 | sed 's/^/   /'
  done
  ls -l /root/triton/plans/*.plan | sed 's/^/   /'

  # 10 · provenance, captured now rather than at the end: the rig is terminated when the
  #      run finishes, and this is the only record of what the numbers were measured on.
  . "$DIR/env.sh"
  ROLE=server PROVIDER=tensorrt scripts/benchmark/hostinfo.sh \
    > /tmp/hostinfo.txt 2>/dev/null || true
  {
    echo "triton_image=$TRITON_IMAGE"
    echo "image_tensorrt=$IMG_TRT"
    echo "image_cuda_major=$IMG_CUDA"
    echo "onnxruntime=$ORT_VER"
    echo "onnxruntime_build=onnxruntime-linux-x64-gpu_$ORT_CUDA-$ORT_VER"
    echo "onnxruntime_so=$(readlink -f "$ORT_DIR/lib/libonnxruntime.so" \
      | sed 's/.*libonnxruntime\.so\.//')"
    echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "gpu_memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    echo "gpu_compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    echo "kernel=$(uname -r)"
    echo "nproc=$(nproc)"
    echo "instance_type=$(TOK=$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); curl -sf \
      -H "X-aws-ec2-metadata-token: $TOK" \
      http://169.254.169.254/latest/meta-data/instance-type)"
    echo "model_specs=$MODEL_SPECS"
    echo "plans=$(cd /root/triton/plans && ls *.plan | tr '\n' ' ')"
  } > /tmp/stackinfo.txt
  sed 's/^/   /' /tmp/stackinfo.txt
  aws s3 cp /tmp/stackinfo.txt \
    "s3://$BUCKET/runs/$RUN/raw/server/$(hostname)-stackinfo.txt" --only-show-errors || true
  aws s3 cp /tmp/hostinfo.txt \
    "s3://$BUCKET/runs/$RUN/raw/server/$(hostname)-hostinfo.txt" --only-show-errors || true
  for f in /root/triton/plans/*.trtexec.log; do
    aws s3 cp "$f" "s3://$BUCKET/runs/$RUN/raw/server/" --only-show-errors || true
  done
else
  cat > "$DIR/env.sh" <<EOF
export ROLE="\${ROLE:-client}"
. "\$HOME/.cargo/env"
EOF
  say "build"
  cd "$DIR"
  cargo build --release -p benchmark-client --bins 2>&1 | tail -3
  test -x target/release/benchmark-client || { echo "build produced no client"; exit 1; }
  echo "   built benchmark-client"

  say "triton load generator"
  rm -rf "$HARNESS" /tmp/th.tgz
  aws s3 cp "s3://$BUCKET/dist/triton-harness-$REV.tgz" /tmp/th.tgz --only-show-errors
  mkdir -p "$HARNESS" && tar xzf /tmp/th.tgz -C "$HARNESS" --strip-components=1
  rm -f /tmp/th.tgz
  # --user, into /root/.local, so `python3 triton_client.py` finds the packages without a
  # virtualenv. grpcio and numpy both have aarch64 manylinux wheels, so nothing compiles
  # here; tritonclient is wanted only for its generated service_pb2.
  python3 -m pip install --quiet --user --upgrade pip >/dev/null 2>&1 || true
  python3 -m pip install --quiet --user "tritonclient[grpc]" numpy 2>&1 | tail -3
  python3 -c "
import grpc, numpy
from tritonclient.grpc import service_pb2
print('   grpcio', grpc.__version__, '| numpy', numpy.__version__, '| service_pb2 ok')"
  python3 -m py_compile "$HARNESS/triton_client.py" \
    && echo "   triton_client.py compiles"
fi
echo "== DEPLOY OK $REV $ROLE [$MODEL_SPECS] $(hostname) $(uname -m) $(nproc) cpu"
