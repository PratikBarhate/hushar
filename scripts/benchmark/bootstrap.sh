#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Prepares one benchmark instance: toolchain, ONNX Runtime, release build.
# Run it inside the unpacked bundle, once per host.
#
#   scripts/benchmark/bootstrap.sh --role server --provider cpu
#   scripts/benchmark/bootstrap.sh --role server --provider tensorrt
#   scripts/benchmark/bootstrap.sh --role client
#
# Roles matter because the two hosts need different things:
#
#   server   the hushar binary, and an ONNX Runtime to load
#   client   the benchmark-client binary only -- it never touches the engine,
#            so no ONNX Runtime is downloaded and the build is much shorter
#
# Writes env.sh in the bundle root. run_server.sh sources it, so ORT_DYLIB_PATH is set
# the same way every run rather than from whatever the shell happened to have.
#
# Options:
#   --role server|client|both   default both
#   --provider NAME             cpu (default), xnnpack, openvino, tensorrt
#   --ort-version V             default 1.29.0
#   --ort-url URL               a specific archive, overriding the derived one
#   --skip-packages             do not touch the package manager
#   --skip-build                set up only, build nothing

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# The header comment above is the help text, so the two cannot drift apart.
usage() { awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"; }

ROLE=both
PROVIDER=cpu
ORT_VERSION="${ORT_VERSION:-1.29.0}"
ORT_URL="${ORT_URL:-}"
ORT_HOME="${ORT_HOME:-$HOME/ort}"
SKIP_PACKAGES=0
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --role) ROLE="$2"; shift ;;
    --provider) PROVIDER="$2"; shift ;;
    --ort-version) ORT_VERSION="$2"; shift ;;
    --ort-url) ORT_URL="$2"; shift ;;
    --skip-packages) SKIP_PACKAGES=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

case "$ROLE" in server|client|both) ;; *) echo "bad --role $ROLE" >&2; exit 2 ;; esac
case "$PROVIDER" in
  cpu|xnnpack|openvino|tensorrt) ;;
  *) echo "bad --provider $PROVIDER (cpu, xnnpack, openvino, tensorrt)" >&2; exit 2 ;;
esac

say() { printf '\n== %s\n' "$1"; }

# ------------------------------------------------------------------ packages
#
# protoc is the one that catches people out: prost-build does not bundle it, so a build
# without protobuf-compiler installed fails in build.rs rather than at link time.
if [ "$SKIP_PACKAGES" = 0 ]; then
  say "packages"
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -q -y gcc gcc-c++ make protobuf-compiler tar gzip curl \
      pkgconf-pkg-config || sudo dnf install -q -y gcc gcc-c++ make protobuf-compiler
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -q -y gcc gcc-c++ make protobuf-compiler tar gzip curl
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential pkg-config protobuf-compiler curl
  else
    echo "no dnf/yum/apt-get found -- install a C toolchain and protoc yourself, then" >&2
    echo "re-run with --skip-packages" >&2
  fi
fi

for tool in cc make protoc curl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool" >&2; exit 1; }
done
echo "protoc  $(protoc --version)"

# ------------------------------------------------------------------ rust
say "rust"
if ! command -v cargo >/dev/null 2>&1; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
echo "rustc   $(rustc --version)"

# ------------------------------------------------------------------ onnx runtime
ORT_DYLIB=""
if [ "$ROLE" != client ]; then
  say "onnx runtime"
  ARCH="$(uname -m)"

  # Two providers are not in any published archive. Both are satisfied by setting
  # ORT_DYLIB_PATH at a library you built or extracted yourself, so the guidance below
  # only prints when that is missing -- a correctly set path is not a warning.
  needs_own_library() {
    [ -n "${ORT_DYLIB_PATH:-}" ] && return 0
    echo >&2
    echo "$PROVIDER is not in any published ONNX Runtime archive." >&2
    cat >&2
    echo >&2
    echo "Then re-run with the path set:" >&2
    echo "  ORT_DYLIB_PATH=/path/to/libonnxruntime.so \\" >&2
    echo "    scripts/benchmark/bootstrap.sh --role server --provider $PROVIDER" >&2
    return 1
  }

  if [ -z "$ORT_URL" ]; then
    BASE="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION"
    case "$PROVIDER:$ARCH" in
      tensorrt:x86_64) ASSET="onnxruntime-linux-x64-gpu_cuda12-$ORT_VERSION" ;;
      tensorrt:*)      echo "TensorRT builds are x86-64 only, not $ARCH" >&2; exit 1 ;;
      xnnpack:*)
        needs_own_library <<'EOF' || exit 1
Build one from source on this host (see documentation/hardware.md):

  git clone --recursive --branch v1.29.0 https://github.com/microsoft/onnxruntime
  cd onnxruntime
  ./build.sh --config Release --build_shared_lib --parallel \
    --use_xnnpack --skip_tests
  # -> build/Linux/Release/libonnxruntime.so
EOF
        ;;
      openvino:*)
        needs_own_library <<'EOF' || exit 1
OpenVINO is not in any prebuilt archive, and needs OpenVINO itself installed first.
Follow Intel's instructions for the version paired with this ONNX Runtime release,
then (see documentation/hardware.md):

  source /opt/intel/openvino/setupvars.sh
  git clone --recursive --branch v1.29.0 https://github.com/microsoft/onnxruntime
  cd onnxruntime
  ./build.sh --config Release --build_shared_lib --parallel \
    --use_openvino CPU --skip_tests
  # -> build/Linux/Release/libonnxruntime.so
EOF
        ;;
      *:x86_64)  ASSET="onnxruntime-linux-x64-$ORT_VERSION" ;;
      *:aarch64) ASSET="onnxruntime-linux-aarch64-$ORT_VERSION" ;;
      *) echo "no known archive for $PROVIDER on $ARCH" >&2; exit 1 ;;
    esac
    [ -n "${ASSET:-}" ] && ORT_URL="$BASE/$ASSET.tgz"
  fi

  if [ -n "${ORT_DYLIB_PATH:-}" ]; then
    ORT_DYLIB="$ORT_DYLIB_PATH"
    echo "using the ORT_DYLIB_PATH already set: $ORT_DYLIB"
  else
    mkdir -p "$ORT_HOME"
    echo "fetching $ORT_URL"
    curl -fsSL "$ORT_URL" | tar xz -C "$ORT_HOME"
    ORT_DYLIB="$(find "$ORT_HOME" -name 'libonnxruntime.so*' -not -name '*.dbg' \
      | sort | head -1)"
  fi

  [ -n "$ORT_DYLIB" ] && [ -r "$ORT_DYLIB" ] \
    || { echo "no libonnxruntime found under $ORT_HOME" >&2; exit 1; }
  echo "library $ORT_DYLIB"
fi

# ------------------------------------------------------------------ env.sh
#
# Every line is a DEFAULT rather than an override. run_server.sh sources this file before
# reading its own environment, so a plain `export PROVIDER=xnnpack` here would silently
# beat the caller's `PROVIDER=cpu run_server.sh start` -- which is the override its own
# help documents. Writing `${PROVIDER:-…}` keeps the value bootstrap chose while leaving
# one run free to name another.
cat > "$ROOT/env.sh" <<EOF
# Written by scripts/benchmark/bootstrap.sh -- source this, do not edit.
# Each value is a default, so any of them can be overridden for a single run:
#   PROVIDER=cpu scripts/benchmark/run_server.sh start
export PROVIDER="\${PROVIDER:-$PROVIDER}"
export ROLE="\${ROLE:-$ROLE}"
EOF
[ -n "$ORT_DYLIB" ] \
  && echo "export ORT_DYLIB_PATH=\"\${ORT_DYLIB_PATH:-$ORT_DYLIB}\"" >> "$ROOT/env.sh"
# A provider built as its own shared library -- XNNPACK, OpenVINO -- sits beside the
# engine, and OpenVINO's runtime with it. Without this the provider is simply missing from
# the banner at run time, even though bootstrap was given the path.
[ -n "${LD_LIBRARY_PATH:-}" ] \
  && echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-$LD_LIBRARY_PATH}\"" >> "$ROOT/env.sh"
[ -r "$HOME/.cargo/env" ] && echo '. "$HOME/.cargo/env"' >> "$ROOT/env.sh"
echo "wrote $ROOT/env.sh"

# ------------------------------------------------------------------ build
if [ "$SKIP_BUILD" = 0 ]; then
  # The provider feature is only meaningful for the server: it lets the config name that
  # hardware. The client has no features and no engine, so it is always the same build.
  #
  # `cpu` rides along with an accelerator so ONE binary serves both, which is what a
  # comparison needs: an accelerator arm and its CPU control arm then differ only in the
  # provider string, with no rebuild between them to cast doubt on the pair.
  if [ "$ROLE" != client ]; then
    FEATURES="$PROVIDER"
    [ "$PROVIDER" = cpu ] || FEATURES="cpu,$PROVIDER"
    say "building hushar --features $FEATURES"
    cargo build --release -p hushar --features "$FEATURES"
  fi
  if [ "$ROLE" != server ]; then
    say "building benchmark-client"
    cargo build --release -p benchmark-client
  fi
fi

say "ready"
ROLE="$ROLE" PROVIDER="$PROVIDER" ORT_DYLIB_PATH="${ORT_DYLIB:-}" \
  scripts/benchmark/hostinfo.sh
cat <<EOF

next, on this host:
  server   scripts/benchmark/run_server.sh start
  client   SERVER=http://<server-private-ip>:8279 scripts/benchmark/run_client.sh
EOF
