#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Sweeps offered rates at a hushar server on ANOTHER host, and prints one table.
#
#   SERVER=http://10.0.1.23:8279 scripts/benchmark/run_client.sh
#   SERVER=… TPS="2 4 8 16 32" scripts/benchmark/run_client.sh
#   SERVER=… MODEL=dcn ROWS=8 scripts/benchmark/run_client.sh
#
# Every rate runs against the same loaded server session, so the comparison between
# rates is exact. Start the server once with run_server.sh and leave it up.
#
# Environment:
#   SERVER       required, http://<server-private-ip>:<port>
#   MODEL        transformer (default), dcn, transformer-batch1, dcn-batch1.
#                The two architectures, both fed one concatenated feature vector the
#                server builds: transformer is an encoder stack, dcn is DCNv2 + MaskNet
#                -- must match what run_server.sh was started with. For a two-model
#                server use the CONTROL arm's name: arch-ab and transformer-ab are both
#                driven with MODEL=transformer, since every arm takes the same features
#   MODEL_CONFIG unset  use this configuration instead of MODEL's generated name;
#                required for a fixed-shape NEFF configuration
#   TPS          "2 4 8 16"   request rates to sweep, in order. The load on the server
#                is TPS x ROWS rows per second, so these two knobs multiply: at the
#                defaults that is 200 to 1600 rows/s. Raise TPS if you lower ROWS
#   DURATION     30   measured seconds per rate
#   SEGMENTS     1    report the measured window as this many equal slices as well as
#                whole. The load is continuous either way; only the reporting splits.
#                Use it for a long window, where one aggregate cannot distinguish a rate
#                that was held from one that was held for a while
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
#                table compares providers -- "transformer-trt" -- since the client cannot
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
MODEL="${MODEL:-transformer}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
TPS="${TPS:-2 4 8 16}"
DURATION="${DURATION:-30}"
SEGMENTS="${SEGMENTS:-1}"
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

if [ -n "$MODEL_CONFIG" ]; then
  config="$MODEL_CONFIG"
else
  case "$MODEL" in
    transformer)         config=$GEN_DIR/bench_model_config.json ;;
  dcn)                 config=$GEN_DIR/bench_dcn_model_config.json ;;
  transformer-batch1)  config=$GEN_DIR/bench_model_batch1_config.json ;;
  dcn-batch1)          config=$GEN_DIR/bench_dcn_model_batch1_config.json ;;
  *) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
  esac
fi
# The client reads the configuration to know which features to send, and never opens the
# .onnx -- so a client host needs the small JSON files only, not the 190 MB model.
[ -f "$config" ] || {
  echo "missing $config -- package the bundle with configurations (the default)" >&2
  exit 1
}

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
  echo "segments      $SEGMENTS slice(s) of the measured window, load continuous"
  echo "rows          $ROWS"
  echo "concurrency   ${CONCURRENCY:-sized from each rate as rate x 4}"
  echo
  echo "NOTE the engine is a property of the SERVER host. Take it from that host's"
  echo "     startup banner, which names the loaded ORT provider or Neuron runtime."
  echo
  ROLE=client scripts/benchmark/hostinfo.sh
} > "$OUT/meta.txt" 2>&1
sed 's/^/  /' "$OUT/meta.txt"

printf '%-26s %7s %9s %7s %8s %8s %8s %6s %7s\n' \
  model offered achieved share p50_ms p95_ms p99_ms shed failed > "$SUMMARY"

# One reachability check, so a security group mistake fails in a second rather than
# looking like a server that sheds everything.
echo
echo "checking $SERVER is reachable"
if ! ./target/release/benchmark-client \
      --server "$SERVER" --model-config "$config" \
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
    ${INFLIGHT_ARGS[@]+"${INFLIGHT_ARGS[@]}"} \
    --tps "$rate" --duration-secs "$DURATION" --warmup-secs "$warm" \
    --segments "$SEGMENTS" \
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

then record the run under benchmark-data/results/<bench-name>/ as documentation/benchmarking.md §6 describes
EOF
