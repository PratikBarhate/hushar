#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Builds ONNX Runtime with an execution provider no published archive carries, on the
# instance that will run it.
#
#   scripts/benchmark/build_ort.sh xnnpack      # ARM64 -- Graviton
#   scripts/benchmark/build_ort.sh openvino     # x86-64 -- Intel
#
# Writes the engine and every provider library beside it into --out, so one
# LD_LIBRARY_PATH covers the set:
#
#   ORT_DYLIB_PATH=<out>/libonnxruntime.so LD_LIBRARY_PATH=<out> \
#     scripts/benchmark/bootstrap.sh --role server --provider xnnpack
#
# Verified on Amazon Linux 2023. Each of these fails the build rather than degrading it,
# which is why the script installs them rather than assuming them -- see
# documentation/hardware.md:
#
#   patch, zlib-devel   ONNX Runtime patches its FetchContent dependencies, and without
#                       patch the first one dies on Patch_EXECUTABLE-NOTFOUND
#   cmake 3.31.x        3.31 is the floor, and CMake 4 dropped shims the third-party
#                       dependencies still need -- so the window is narrow and distribution
#                       packages are usually below it
#   python 3.10+        tools/ci_build/build.py uses a match statement, and build.sh calls
#                       whatever `python3` is -- 3.9 on AL2023 -- so build.py is driven
#                       directly here
#   gcc 13+, openvino   the OpenVINO EP sources need C++20 <format>, and its EP needs
#     2026.0+           OpenVINO 2026.0 or newer, which is a FATAL_ERROR rather than a
#                       warning. Both are checked before the compile, not after it
#
# Options:
#   --version V     ONNX Runtime tag to build, default 1.29.0
#   --out DIR       where to assemble the libraries, default ~/ort-<provider>
#   --src DIR       where to clone, default ~/onnxruntime
#   --jobs N        compile parallelism, default nproc
#   --skip-packages do not touch the package manager or pip

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The header comment above is the help text, so the two cannot drift apart.
usage() { awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"; }

PROVIDER=""
VERSION=1.29.0
OUT=""
SRC="${HOME:-/root}/onnxruntime"
JOBS="$(nproc)"
SKIP_PACKAGES=0

while [ $# -gt 0 ]; do
  case "$1" in
    xnnpack|openvino) PROVIDER="$1" ;;
    --version) VERSION="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --src) SRC="$2"; shift ;;
    --jobs) JOBS="$2"; shift ;;
    --skip-packages) SKIP_PACKAGES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -n "$PROVIDER" ] || { echo "name a provider: xnnpack or openvino" >&2; exit 2; }
OUT="${OUT:-${HOME:-/root}/ort-$PROVIDER}"

case "$PROVIDER:$(uname -m)" in
  openvino:aarch64) echo "OpenVINO is an Intel provider; this host is aarch64" >&2; exit 2 ;;
esac

say() { printf '\n== %s\n' "$1"; }

PY=python3.11
# ------------------------------------------------------------------ packages
if [ "$SKIP_PACKAGES" = 0 ]; then
  say "packages"
  # `curl` is deliberately absent: asking for it conflicts with the curl-minimal that
  # Amazon Linux 2023 preinstalls, and the build only needs what curl-minimal provides.
  PACKAGES="git gcc gcc-c++ make patch zlib-devel protobuf-compiler pkgconf-pkg-config
            tar gzip python3.11 python3.11-pip python3.11-devel"
  [ "$PROVIDER" = openvino ] && PACKAGES="$PACKAGES gcc14 gcc14-c++"
  if command -v dnf >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    sudo dnf install -y -q $PACKAGES
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential git patch zlib1g-dev protobuf-compiler \
      pkg-config python3 python3-pip python3-dev
    PY=python3
  else
    echo "no dnf/apt-get -- install the packages listed above yourself, then" >&2
    echo "re-run with --skip-packages" >&2
    exit 1
  fi
  command -v "$PY" >/dev/null 2>&1 || PY=python3
  "$PY" -m pip install -q "cmake~=3.31.0" packaging setuptools wheel numpy
fi
command -v "$PY" >/dev/null 2>&1 || PY=python3
export PATH="/usr/local/bin:$PATH"

# build.py needs 3.10+ for a match statement, and the failure is a bare SyntaxError that
# does not say so, so it is worth checking by hand.
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "$PY is $("$PY" --version), but ONNX Runtime's build.py needs 3.10 or newer" >&2
  exit 1
}
echo "python  $("$PY" --version)"
echo "cmake   $(cmake --version | head -1)"
echo "patch   $(patch --version | head -1)"

BUILD_ARGS=(--config Release --build_shared_lib --parallel "$JOBS" --skip_tests
            --compile_no_warning_as_error)
# Only needed when building as root, which is what SSM and most bootstrap paths give you.
[ "$(id -u)" = 0 ] && BUILD_ARGS+=(--allow_running_as_root)

OV_LIBS=""
case "$PROVIDER" in
  xnnpack) BUILD_ARGS+=(--use_xnnpack) ;;
  openvino)
    say "openvino"
    # The PyPI wheel carries the runtime, the headers and the OpenVINOConfig.cmake that
    # find_package looks for, so a full /opt/intel install is not needed.
    "$PY" -m pip install -q "openvino>=2026.0"
    OV_ROOT="$("$PY" -c 'import openvino, os; print(os.path.dirname(openvino.__file__))')"
    OV_LIBS="$OV_ROOT/libs"
    export OpenVINO_DIR="$OV_ROOT/cmake"
    export LD_LIBRARY_PATH="$OV_LIBS:${LD_LIBRARY_PATH:-}"
    if [ -x /usr/bin/gcc14-g++ ]; then
      export CC=/usr/bin/gcc14-gcc CXX=/usr/bin/gcc14-g++
    fi
    echo "openvino $("$PY" -c 'import openvino; print(openvino.__version__)')"
    echo "compiler $("${CXX:-g++}" --version | head -1)"
    BUILD_ARGS+=(--use_openvino CPU)
    ;;
esac

# ------------------------------------------------------------------ source
say "source $VERSION"
[ -d "$SRC" ] || git clone --depth 1 --recursive --branch "v$VERSION" \
  https://github.com/microsoft/onnxruntime "$SRC"
cd "$SRC"

say "build ${BUILD_ARGS[*]}"
"$PY" tools/ci_build/build.py --build_dir "$SRC/build/Linux" "${BUILD_ARGS[@]}"

# ------------------------------------------------------------------ assemble
SO="$(find "$SRC/build" -name 'libonnxruntime.so*' -not -name '*.dbg' | sort | head -1)"
[ -n "$SO" ] || { echo "no libonnxruntime.so was produced" >&2; exit 1; }

mkdir -p "$OUT"
cp -aL "$SO" "$OUT/libonnxruntime.so"
# These two providers are built as separate shared libraries, so they have to sit beside
# the engine -- and OpenVINO's own runtime with them, or the provider is simply missing
# from the banner with no other explanation.
find "$SRC/build" -name 'libonnxruntime_providers_*.so' -exec cp -a {} "$OUT/" \; || true
[ -n "$OV_LIBS" ] && cp -a "$OV_LIBS"/*.so* "$OUT/" 2>/dev/null || true

say "ready"
ls -la "$OUT"
cat <<EOF

next, on this host:
  ORT_DYLIB_PATH=$OUT/libonnxruntime.so LD_LIBRARY_PATH=$OUT \\
    $ROOT/scripts/benchmark/bootstrap.sh --role server --provider $PROVIDER

--provider is also a Cargo feature: a server built without it refuses the configuration at
startup even though this library has the provider compiled in.
EOF
