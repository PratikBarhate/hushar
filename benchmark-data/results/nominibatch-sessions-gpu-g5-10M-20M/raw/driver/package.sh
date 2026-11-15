#!/usr/bin/env bash
# Packages everything the hosts fetch from S3, and prints the values the deploy needs.
#
#   HUSHAR_BENCH_BUCKET=<bucket> package.sh
#
# Objects: the hushar bundle (the WORKING TREE, so an uncommitted fix is testable), the
# Triton harness, the GPU sampler, the sweep driver, and both model directories with a
# small configs object per size. The harness and the models are gitignored, so neither
# rides in the bundle and both travel on their own.
#
# The tensor feature spec is DERIVED here from the generated model configuration rather
# than written by hand: Triton's client reads it to know how many floats a row carries,
# and the only way it cannot disagree with the graph is to compute it from the same
# source the graph was generated from.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
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
echo "== triton harness"
# arms/ is old results and __pycache__ is build output; neither belongs on a host.
# --no-xattrs because macOS stores com.apple.provenance as a PAX extended header, which
# GNU tar on the instance reports once per file and which nothing on Linux wants.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "benchmark-data/dist/triton-harness-$REV.tgz" \
  --exclude 'triton-bench/arms' --exclude '__pycache__' --exclude '.DS_Store' \
  triton-bench
$AWS s3 cp "benchmark-data/dist/triton-harness-$REV.tgz" \
  "s3://$BUCKET/dist/triton-harness-$REV.tgz" --only-show-errors
echo "   triton-harness-$REV.tgz -> dist/"

echo
echo "== gpu sampler"
# Travels on its own rather than inside the harness, because the deploy installs it at a
# fixed path the sweep calls and a file inside a tarball is one unpack away from moving.
$AWS s3 cp "$HERE/sample_gpu.sh" "s3://$BUCKET/dist/sample_gpu-$REV.sh" --only-show-errors
echo "   sample_gpu-$REV.sh -> dist/"

echo
echo "== sweep driver"
python3 -c "import py_compile; py_compile.compile('$HERE/sweep.py', doraise=True)"
$AWS s3 cp "$HERE/sweep.py" "s3://$BUCKET/dist/sweep-$REV.py" --only-show-errors
echo "   sweep-$REV.py -> dist/"

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
