#!/usr/bin/env bash
# Packages everything the hosts fetch from S3, and prints the REV the deploy needs.
#
#   HUSHAR_BENCH_BUCKET=<bucket> package.sh
#   HUSHAR_BENCH_BUCKET=<bucket> MODEL_SIZE=10m package.sh
#
# Four objects: the hushar bundle (the WORKING TREE, so an uncommitted fix is testable),
# the Triton harness, the sweep driver, and the model directory. The harness and the
# models are gitignored, so neither rides in the bundle and both travel on their own.
#
# The parity arm's two files are DERIVED here, from the generated model configuration,
# rather than written by hand on a host: the width of the identity transformation has to
# be the width the real configuration concatenates to, and deriving it is the only way the
# two cannot come to disagree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
MODEL_SIZE="${MODEL_SIZE:-10m}"
MODEL_FILE="${MODEL_FILE:-bench_model.onnx}"
export AWS_PROFILE="${AWS_PROFILE:-hushar-bench}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_PAGER=""
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
cd "$ROOT"

GEN="benchmark-data/generated/$MODEL_SIZE"
[ -f "$GEN/$MODEL_FILE" ] || {
  echo "missing $GEN/$MODEL_FILE" >&2
  echo "  python scripts/generate_benchmark_models.py --params ${MODEL_SIZE^^} --out-dir $GEN" >&2
  exit 1
}

echo "== parity arm"
python3 "$HERE/make_tensor_config.py" "$GEN" | sed 's/^/   /'

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
echo "   triton-harness-$REV.tgz -> dist/ ($(ls -1 triton-bench | wc -l | tr -d ' ') entries)"

echo
echo "== configurations"
# The client needs the model configurations -- it reads one to know which features to send
# -- but its instance role can read dist/ and not models/, and the bundle is gitignored
# clean of generated/. So the configurations travel as their own small object under dist/,
# which both roles can already fetch. The server still takes them from models/ beside the
# graph whose md5 it checks.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "benchmark-data/dist/configs-$REV.tgz" \
  -C "$GEN" $(cd "$GEN" && ls *.json)
$AWS s3 cp "benchmark-data/dist/configs-$REV.tgz" \
  "s3://$BUCKET/dist/configs-$REV.tgz" --only-show-errors
echo "   configs-$REV.tgz -> dist/ ($(cd "$GEN" && ls -1 *.json | wc -l | tr -d ' ') files)"

echo
echo "== sweep driver"
python3 -c "import py_compile; py_compile.compile('$HERE/sweep.py', doraise=True)"
$AWS s3 cp "$HERE/sweep.py" "s3://$BUCKET/dist/sweep-$REV.py" --only-show-errors
echo "   sweep-$REV.py -> dist/"

echo
echo "== models $MODEL_SIZE"
$AWS s3 sync "$GEN/" "s3://$BUCKET/models/$MODEL_SIZE/" --only-show-errors --delete
MD5=$(md5 -q "$GEN/$MODEL_FILE" 2>/dev/null || md5sum "$GEN/$MODEL_FILE" | cut -d' ' -f1)
echo "   $(ls -1 "$GEN" | wc -l | tr -d ' ') files, $MODEL_FILE md5 $MD5"

echo
echo "REV=$REV"
echo "MODEL_MD5=$MD5"
echo "MODEL_SIZE=$MODEL_SIZE"
