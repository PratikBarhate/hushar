#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Sweeps offered rates at a hushar server on ANOTHER host, and prints one table.
#
#   SERVER=http://10.0.1.23:8279 scripts/benchmark/run_client.sh
#   SERVER=… TPS="2 4 8 16 32" scripts/benchmark/run_client.sh
#   SERVER=… MODEL=raw-dcn ROWS=8 scripts/benchmark/run_client.sh
#
# Every rate runs against the same loaded server session, so the comparison between
# rates is exact. Start the server once with run_server.sh and leave it up.
#
# Environment:
#   SERVER       required, http://<server-private-ip>:<port>
#   MODEL        raw (default), raw-dcn, vectorized, named,
#                raw-batch1, raw-dcn-batch1, vectorized-batch1, named-batch1.
#                raw and raw-dcn are the two architectures, both sent every feature as
#                it is: raw is a transformer encoder, raw-dcn is DCNv2 + MaskNet.
#                vectorized and named have the server transform features instead
#                -- must match what run_server.sh was started with. For a two-model
#                server use the CONTROL arm's name: arch-ab and raw-ab are both driven
#                with MODEL=raw, since every arm takes the same features
#   TPS          "2 4 8 16"   request rates to sweep, in order. The load on the server
#                is TPS x ROWS rows per second, so these two knobs multiply: at the
#                defaults that is 200 to 1600 rows/s. Raise TPS if you lower ROWS
#   DURATION     30   measured seconds per rate
#   WARMUP       10   discarded seconds before the first rate; 3 for the rest
#   ROWS         100  rows per request. One engine call scores all of them, unless the
#                model pins a batch size -- see the batch1 configs, where 100 rows is
#                100 sequential calls
#   CONCURRENCY  unset  requests allowed in flight. Unset, the client sizes it per rate
#                as rate x 4 -- in flight has to cover rate x latency, so this covers a
#                4-second response and the client is never the bottleneck, which is the
#                point: we are measuring the server. Set a number to pin it
#   RUN_DIR      /tmp/hushar-client
#   LABEL        a name for this run, used for the output directory
#   TAG          the first column of the summary table, default $MODEL. Set it when one
#                table compares providers -- "vectorized-trt" -- since the client cannot
#                know which provider the server on the other host registered.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "${1:-}" in
  -h|--help)
    awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"
    exit 0 ;;
esac

# shellcheck disable=SC1091
[ -r "$ROOT/env.sh" ] && . "$ROOT/env.sh"

SERVER="${SERVER:-}"
MODEL="${MODEL:-raw}"
TPS="${TPS:-2 4 8 16}"
DURATION="${DURATION:-30}"
WARMUP="${WARMUP:-10}"
ROWS="${ROWS:-100}"
# Unset means the client sizes it from the rate; see its --concurrency help. Left to
# the client so the policy lives in one place rather than being duplicated here.
CONCURRENCY="${CONCURRENCY:-}"
INFLIGHT_ARGS=()
[ -n "$CONCURRENCY" ] && INFLIGHT_ARGS=(--concurrency "$CONCURRENCY")
RUN_DIR="${RUN_DIR:-/tmp/hushar-client}"
TAG="${TAG:-$MODEL}"
# Overridable so several model sizes can live side by side, each generated into its
# own directory -- see generate_benchmark_models.py --out-dir.
GEN_DIR="${GEN_DIR:-benchmark-data/generated}"

if [ -z "$SERVER" ]; then
  echo "SERVER is not set. The point of this runner is that the server is on another" >&2
  echo "host, so there is no sensible default:" >&2
  echo "  SERVER=http://10.0.1.23:8279 scripts/benchmark/run_client.sh" >&2
  exit 2
fi
[ -x ./target/release/benchmark-client ] \
  || { echo "no ./target/release/benchmark-client -- run bootstrap.sh first" >&2; exit 1; }

case "$MODEL" in
  raw)               config=$GEN_DIR/bench_raw_model_config.json ;;
  raw-dcn)           config=$GEN_DIR/bench_raw_dcn_model_config.json ;;
  vectorized)        config=$GEN_DIR/bench_model_config.json ;;
  named)             config=$GEN_DIR/bench_named_model_config.json ;;
  raw-batch1)        config=$GEN_DIR/bench_raw_model_batch1_config.json ;;
  raw-dcn-batch1)    config=$GEN_DIR/bench_raw_dcn_model_batch1_config.json ;;
  vectorized-batch1) config=$GEN_DIR/bench_model_batch1_config.json ;;
  named-batch1)      config=$GEN_DIR/bench_named_model_batch1_config.json ;;
  *) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
esac
# The client reads the configuration to know which features to send, and never opens the
# .onnx -- so a client host needs the small JSON files only, not the 190 MB model.
[ -f "$config" ] || {
  echo "missing $config -- package the bundle with configurations (the default)" >&2
  exit 1
}

# A raw model's configuration mentions no features, because its graph does the
# featurization -- so the generator is told what to send by the specification written
# beside it. Required rather than optional: without it the client has nothing to send.
SPEC_ARGS=()
case "$MODEL" in
  raw*)
    spec=$GEN_DIR/bench_raw_feature_spec.json
    [ -f "$spec" ] || {
      echo "missing $spec, which MODEL=$MODEL needs to know which features to send" >&2
      echo "  run scripts/generate_benchmark_models.py, or package with configurations" >&2
      exit 1
    }
    SPEC_ARGS=(--feature-spec "$spec") ;;
esac

LABEL="${LABEL:-$MODEL-$(date -u '+%Y%m%dT%H%M%SZ')}"
OUT="$RUN_DIR/$LABEL"
mkdir -p "$OUT/logs"
SUMMARY="$OUT/summary.txt"

# Captured before the run, not reconstructed afterwards from memory.
{
  echo "server        $SERVER"
  echo "model         $MODEL"
  echo "tag           $TAG"
  echo "rates         $TPS"
  echo "duration      ${DURATION}s measured, ${WARMUP}s warmup on the first rate"
  echo "rows          $ROWS"
  echo "concurrency   ${CONCURRENCY:-sized from each rate as rate x 4}"
  echo
  echo "NOTE the execution provider is a property of the SERVER host. Take it from that"
  echo "     host's startup banner, which names the provider actually registered."
  echo
  ROLE=client scripts/benchmark/hostinfo.sh
} > "$OUT/meta.txt" 2>&1
sed 's/^/  /' "$OUT/meta.txt"

printf '%-26s %7s %9s %7s %8s %8s %8s %6s %7s\n' \
  model offered achieved share p50_ms p95_ms p99_ms shed failed > "$SUMMARY"

# The vocabulary only matters for a configuration with string inputs. Without it those
# features are sent as words the graph has never seen, which costs the same to look up
# but exercises only the out-of-vocabulary row.
VOCAB_ARGS=()
if [ -f "$GEN_DIR/bench_string_vocabulary.json" ]; then
  VOCAB_ARGS=(--vocabulary "$GEN_DIR/bench_string_vocabulary.json")
fi

# One reachability check, so a security group mistake fails in a second rather than
# looking like a server that sheds everything.
echo
echo "checking $SERVER is reachable"
if ! ./target/release/benchmark-client \
      --server "$SERVER" --model-config "$config" ${VOCAB_ARGS[@]+"${VOCAB_ARGS[@]}"} \
      ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
      --tps 5 --duration-secs 2 --warmup-secs 0 --rows "$ROWS" --concurrency 4 \
      --label "reachability" > "$OUT/logs/reachability.log" 2>&1; then
  echo "the client could not talk to $SERVER:" >&2
  tail -10 "$OUT/logs/reachability.log" | sed 's/^/  /' >&2
  echo "check: server is up, the port is open to this host's security group," >&2
  echo "       and the server bound 0.0.0.0 rather than 127.0.0.1" >&2
  exit 1
fi
if grep -qE 'failed *: *[1-9]' "$OUT/logs/reachability.log"; then
  echo "reachable, but requests are failing -- fix that before measuring:" >&2
  grep -E 'failed|error' "$OUT/logs/reachability.log" | head -5 | sed 's/^/  /' >&2
  exit 1
fi
echo "reachable"

first=1
for rate in $TPS; do
  # Only the first rate pays full warmup; the session is hot for the rest.
  warm=$WARMUP; [ $first -eq 1 ] || warm=3
  first=0
  log="$OUT/logs/$rate.client.log"
  echo
  echo "=============================================================="
  echo "  $TAG @ $rate req/s -> $SERVER"
  echo "=============================================================="
  ./target/release/benchmark-client \
    --server "$SERVER" \
    --model-config "$config" \
    ${VOCAB_ARGS[@]+"${VOCAB_ARGS[@]}"} ${INFLIGHT_ARGS[@]+"${INFLIGHT_ARGS[@]}"} \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    --tps "$rate" --duration-secs "$DURATION" --warmup-secs "$warm" \
    --rows "$ROWS" \
    --label "$TAG @ $rate req/s" 2>&1 | tee "$log"
  scripts/benchmark/summarise.py "$TAG" "$rate" "$log" >> "$SUMMARY"
done

echo
echo "=============================================================="
echo "  summary"
echo "=============================================================="
cat "$SUMMARY"
cat <<EOF

client output: $OUT
  summary.txt        the table above -- paste this into the report
  meta.txt           this host, and the load shape
  logs/<rate>.client.log

still needed for a complete result, from the SERVER host:
  scripts/benchmark/hostinfo.sh                     the server's environment block
  grep '^metrics:' /tmp/hushar-server/*.server.log  the engine's own timings

then: cp benchmark-data/results/TEMPLATE.md \\
        benchmark-data/results/<class>/<instance>-<provider>-$(date -u '+%Y-%m-%d').md
EOF
