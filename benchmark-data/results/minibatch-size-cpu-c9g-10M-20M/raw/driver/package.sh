#!/usr/bin/env bash
# Packages everything the hosts fetch from S3, and prints the values the deploy needs.
#
#   HUSHAR_BENCH_BUCKET=<bucket> package.sh
#
# Objects: the hushar bundle (the WORKING TREE, so an uncommitted fix is testable), the
# sweep driver, and both model directories with a small configs object per size. The
# models are gitignored, so they do not ride in the bundle and travel on their own.
#
# NO TRITON HARNESS AND NO DEVICE SAMPLER. This run has one arm -- hushar on the feature
# wire -- because Triton has no mini-batch split to compare against, and the host is a CPU
# so there is no device to sample. The tensor feature spec is still DERIVED here from the
# generated model configuration, because it costs nothing and leaves a follow-up Triton
# reference line one deploy away rather than one driver away.
set -euo pipefail

# The repository root, found by walking up to a marker rather than by counting `..`:
# this driver has a per-accelerator subdirectory, so it sits one level deeper than the
# campaigns it was derived from and a fixed count would resolve to benchmark-data/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/scripts/benchmark" ]; do
  ROOT="$(dirname "$ROOT")"
done
[ -d "$ROOT/scripts/benchmark" ] || { echo "cannot find the repository root above $0" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
MODEL_FILE="${MODEL_FILE:-bench_model.onnx}"
SIZES="${SIZES:-10m 20m}"
export AWS_PROFILE="${AWS_PROFILE:-hushar-bench}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_PAGER=""
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
cd "$ROOT"

for size in $SIZES; do
  [ -f "benchmark-data/generated/$size/$MODEL_FILE" ] || {
    echo "missing benchmark-data/generated/$size/$MODEL_FILE" >&2
    echo "  python scripts/generate_benchmark_models.py --params ${size^^} \\" >&2
    echo "    --out-dir benchmark-data/generated/$size" >&2
    exit 1; }
done

echo "== tensor feature specs"
for size in $SIZES; do
  python3 "$HERE/make_tensor_config.py" "benchmark-data/generated/$size" | sed 's/^/   /'
done

echo
echo "== bundle"
# --source-only: the configurations travel with the models under models/, so the bundle
# does not carry a second copy that could be the stale one.
scripts/benchmark/package.sh --source-only >/tmp/package.out 2>&1 || {
  cat /tmp/package.out; exit 1; }
TARBALL=$(ls -t benchmark-data/dist/hushar-bench-*.tgz | head -1)
REV=$(basename "$TARBALL" .tgz | sed 's/^hushar-bench-//')
$AWS s3 cp "$TARBALL" "s3://$BUCKET/dist/$(basename "$TARBALL")" --only-show-errors
echo "   $(basename "$TARBALL") -> dist/"

echo
echo "== sweep driver"
python3 -c "import py_compile; py_compile.compile('$HERE/sweep.py', doraise=True)"
$AWS s3 cp "$HERE/sweep.py" "s3://$BUCKET/dist/sweep-cpu-$REV.py" --only-show-errors
echo "   sweep-cpu-$REV.py -> dist/"

echo
echo "== models and configurations"
# The client needs the model configurations -- it reads one to know which features to
# send -- but its instance role can read dist/ and not models/. So the configurations
# travel as their own small object under dist/, which both roles can already fetch. The
# server still takes them from models/ beside the graph whose md5 it checks.
for size in $SIZES; do
  GEN="benchmark-data/generated/$size"
  COPYFILE_DISABLE=1 tar --no-xattrs -czf "benchmark-data/dist/configs-$size-$REV.tgz" \
    -C "$GEN" $(cd "$GEN" && ls *.json)
  $AWS s3 cp "benchmark-data/dist/configs-$size-$REV.tgz" \
    "s3://$BUCKET/dist/configs-$size-$REV.tgz" --only-show-errors
  $AWS s3 sync "$GEN/" "s3://$BUCKET/models/$size/" --only-show-errors --delete
  MD5=$(md5 -q "$GEN/$MODEL_FILE" 2>/dev/null || md5sum "$GEN/$MODEL_FILE" | cut -d' ' -f1)
  BYTES=$(stat -f%z "$GEN/$MODEL_FILE" 2>/dev/null || stat -c%s "$GEN/$MODEL_FILE")
  echo "   $size: $MODEL_FILE md5 $MD5, $BYTES bytes, $(cd "$GEN" && ls -1 *.json | wc -l | tr -d ' ') configs"
  eval "MD5_${size}=$MD5"
done

echo
echo "REV=$REV"
for size in $SIZES; do
  eval "echo \"MD5_$size=\$MD5_${size}\""
done
