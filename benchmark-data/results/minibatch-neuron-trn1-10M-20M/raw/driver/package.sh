#!/usr/bin/env bash
# Packages everything the hosts fetch from S3, and prints the values the deploy needs.
#
#   HUSHAR_BENCH_BUCKET=<bucket> package.sh
#
# Objects: the hushar bundle (the WORKING TREE, so an uncommitted fix is testable), the
# NeuronCore sampler, and the sweep driver.
#
# NO MODELS TRAVEL. This is the one way the Neuron run differs from every other in this
# series, and it is worth being explicit about: Neuron does not compile ONNX, so the
# artifact a cell runs is a NEFF, and a NEFF is produced on the host by
# `torch_neuronx.trace` for ONE fixed batch size. There is nothing to upload from here
# and nothing to pin by md5. `compile_neuron.py` rebuilds the SAME seeded architecture
# `generate_benchmark_models.py` defines -- which travels inside the bundle -- so the
# model is pinned by seed and target parameter count, and `prime` records the SHA-256 and
# the compiled-versus-PyTorch error of what it built.
#
# The NEFFs are then cached in S3 under `models/neuron/` BY THE SERVER, not from here, and
# only after a server has actually served one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
export AWS_PROFILE="${AWS_PROFILE:-hushar-bench}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_PAGER=""
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
cd "$ROOT"

# The generator is what compile_neuron.py loads to get the architecture, so its absence
# would surface as a compile failure on the host rather than here.
[ -r scripts/generate_benchmark_models.py ] \
  || { echo "missing scripts/generate_benchmark_models.py" >&2; exit 1; }
[ -r scripts/benchmark/compile_neuron.py ] \
  || { echo "missing scripts/benchmark/compile_neuron.py" >&2; exit 1; }

echo "== bundle"
scripts/benchmark/package.sh --source-only >/tmp/package.out 2>&1 || {
  cat /tmp/package.out; exit 1; }
TARBALL=$(ls -t benchmark-data/dist/hushar-bench-*.tgz | head -1)
REV=$(basename "$TARBALL" .tgz | sed 's/^hushar-bench-//')
$AWS s3 cp "$TARBALL" "s3://$BUCKET/dist/$(basename "$TARBALL")" --only-show-errors
echo "   $(basename "$TARBALL") -> dist/"
# Proof the files the host needs are inside it, checked here rather than discovered by a
# compile that fails 40 minutes into a deploy.
for want in scripts/benchmark/compile_neuron.py scripts/generate_benchmark_models.py \
            scripts/benchmark/run_server.sh scripts/benchmark/bootstrap.sh; do
  tar tzf "$TARBALL" | grep -q "/$want$" \
    || { echo "   REFUSING: $want is not in the bundle" >&2; exit 1; }
  echo "   carries $want"
done

echo
echo "== neuron sampler"
# Travels on its own rather than inside the bundle, because the deploy installs it at a
# fixed path the sweep calls and a file inside a tarball is one unpack away from moving.
bash -n "$HERE/sample_neuron.sh" \
  || { echo "   REFUSING: the sampler does not parse" >&2; exit 1; }
$AWS s3 cp "$HERE/sample_neuron.sh" "s3://$BUCKET/dist/sample_neuron-$REV.sh" \
  --only-show-errors
echo "   sample_neuron-$REV.sh -> dist/"

echo
echo "== sweep driver"
python3 -c "import py_compile; py_compile.compile('$HERE/sweep.py', doraise=True)"
$AWS s3 cp "$HERE/sweep.py" "s3://$BUCKET/dist/sweep-$REV.py" --only-show-errors
echo "   sweep-$REV.py -> dist/"

echo
echo "== neuron NEFF cache"
CACHED=$($AWS s3 ls "s3://$BUCKET/models/neuron/" --recursive 2>/dev/null \
  | grep -c '\.neff$' || true)
echo "   $CACHED NEFF(s) already cached under models/neuron/ -- prime restores these"

echo
echo "REV=$REV"
