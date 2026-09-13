#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Builds one tarball to copy to a benchmark instance: source, scripts, and optionally
# the generated configurations and models.
#
#   scripts/benchmark/package.sh                  # source + configs   (~2 MB)
#   scripts/benchmark/package.sh --with-models    # + the .onnx files  (~800 MB)
#   scripts/benchmark/package.sh --source-only    # source alone
#
# Source rather than binaries, deliberately. A release build for ARM64, x86-64 and CUDA
# from one laptop means three cross-compilation toolchains and a linker for each; the
# instance you are about to benchmark already has a native one. `bootstrap.sh` inside the
# bundle does the build, and it takes a few minutes.
#
# The bundle carries the WORKING TREE, not HEAD -- everything git would show as tracked
# or untracked-but-not-ignored -- so an uncommitted fix can be tested. BUNDLE_INFO
# records the commit and whether the tree was dirty.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# The header comment above is the help text, so the two cannot drift apart.
usage() { awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"; }

WITH_MODELS=0
WITH_CONFIGS=1
OUT_DIR="${OUT_DIR:-benchmark-data/dist}"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-models) WITH_MODELS=1 ;;
    --source-only) WITH_CONFIGS=0 ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
COMMIT="$(git describe --always --dirty 2>/dev/null || echo unknown)"
STEM="hushar-bench-$COMMIT"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/hushar-bundle.XXXXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
PREFIX="$STAGE/$STEM"

# Everything git considers part of the project: tracked files, plus untracked ones that
# are not ignored. `--cached` alone would silently drop a brand-new script, and no
# `--exclude-standard` would drag in build output and the 190 MB generated models.
git ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' file; do
  [ -f "$file" ] || continue
  mkdir -p "$PREFIX/$(dirname "$file")"
  cp "$file" "$PREFIX/$file"
done

GEN="benchmark-data/generated"
copy_generated() {
  local pattern="$1" found=0
  mkdir -p "$PREFIX/$GEN"
  for path in $GEN/$pattern; do
    [ -e "$path" ] || continue
    cp "$path" "$PREFIX/$GEN/"
    found=1
  done
  [ "$found" = 1 ]
}

if [ "$WITH_CONFIGS" = 1 ]; then
  # Small, and the CLIENT cannot run without them: it reads the model configuration to
  # know which features to send, and the vocabulary to send tokens the graph has seen.
  copy_generated '*.json' \
    || echo "note: no configurations in $GEN -- run scripts/generate_benchmark_models.py first" >&2
fi

if [ "$WITH_MODELS" = 1 ]; then
  copy_generated '*.onnx' \
    || echo "note: no models in $GEN -- run scripts/generate_benchmark_models.py first" >&2
fi

cat > "$PREFIX/BUNDLE_INFO" <<EOF
commit:    $COMMIT
packaged:  $(date -u '+%Y-%m-%dT%H:%M:%SZ')
by:        ${USER:-unknown}@$(hostname 2>/dev/null || echo unknown)
configs:   $([ "$WITH_CONFIGS" = 1 ] && echo yes || echo no)
models:    $([ "$WITH_MODELS" = 1 ] && echo yes || echo no)
EOF

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/$STEM.tgz"
# COPYFILE_DISABLE stops macOS bsdtar writing an AppleDouble "._name" beside every file
# carrying an extended attribute. Unpacked on Linux those become real files, and a
# "._structs.proto" landing next to the schema is picked up by the proto glob in build.rs
# and fails the build on binary junk. Ignored on Linux, so it is safe to set always.
COPYFILE_DISABLE=1 tar -czf "$TARBALL" -C "$STAGE" "$STEM"

echo "wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"
cat "$PREFIX/BUNDLE_INFO" | sed 's/^/  /'
echo
echo "next:"
echo "  scp $TARBALL <host>:~/"
echo "  ssh <host> 'tar xzf $STEM.tgz && cd $STEM && scripts/benchmark/bootstrap.sh --help'"
