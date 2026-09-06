#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Runs the benchmark for each model at each offered rate, and prints one table.
#
#   scripts/run_benchmark.sh                          # 500 req/s, both models, CPU
#   TPS="50 100 200 400 600" scripts/run_benchmark.sh # sweep, to find the capacity knee
#   MODELS=vectorized DURATION=60 scripts/run_benchmark.sh
#
# Needs a release build and an ONNX Runtime to load:
#
#   cargo build --release
#   export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
#
# MODELS takes four names. `vectorized` and `named` have a dynamic row axis and serve
# any batch. `vectorized-batch1` and `named-batch1` are pinned to one row and declare
# "fixed_batch_size": 1, so the service splits any request into batches of one row.
# They need the generator's --with-batch1:
#
#   TPS="50 500" ROWS=1 MODELS="vectorized-batch1 named-batch1" scripts/run_benchmark.sh
#
# PROVIDERS also takes an accelerator name, on a build with that feature enabled.

set -uo pipefail
cd "$(dirname "$0")/.."

# One or more rates. They run against the same loaded server, because a 190 MB model
# takes long enough to load that restarting per rate would dominate the wall clock, and
# reusing one session keeps the comparison between rates exact.
TPS="${TPS:-500}"
DURATION="${DURATION:-20}"
WARMUP="${WARMUP:-10}"
ROWS="${ROWS:-1}"
CONCURRENCY="${CONCURRENCY:-64}"
PORT="${PORT:-8279}"
PROVIDERS="${PROVIDERS:-cpu}"
MODELS="${MODELS:-vectorized named}"
RUN_DIR="${RUN_DIR:-/tmp/hushar-benchmark}"

# Written by scripts/generate_benchmark_models.py and not tracked by git, because a
# model is ~190 MB. See benchmark-data/README.md.
GEN_DIR="benchmark-data/generated"

if [ -z "${ORT_DYLIB_PATH:-}" ]; then
  echo "ORT_DYLIB_PATH is not set; the engine needs a libonnxruntime to load" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/inference-logs"
SUMMARY="$RUN_DIR/summary.txt"
printf '%-24s %8s %10s %8s %8s %8s %7s %7s\n' \
  model offered achieved p50_ms p90_ms p99_ms shed failed > "$SUMMARY"

cleanup() { [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for model in $MODELS; do
  case "$model" in
    vectorized)        base=$GEN_DIR/bench_model_config.json ;;
    named)             base=$GEN_DIR/bench_named_model_config.json ;;
    vectorized-batch1) base=$GEN_DIR/bench_model_batch1_config.json ;;
    named-batch1)      base=$GEN_DIR/bench_named_model_batch1_config.json ;;
    *) echo "unknown model $model" >&2; exit 1 ;;
  esac
  if [ ! -f "$base" ]; then
    echo "missing $base -- run scripts/generate_benchmark_models.py" >&2
    case "$model" in
      *-batch1) echo "  the pinned pair needs --with-batch1" >&2 ;;
    esac
    exit 1
  fi

  # A pinned model now serves any row count -- rows are split into batches of the
  # pinned size and the last is padded -- so a mismatch is no longer an error. It is
  # still worth saying out loud, because it changes what the number measures: at
  # ROWS=3 against a model pinned to 1, each request is three engine calls, so the
  # latency is three inferences and not one.
  pinned=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("fixed_batch_size",""))' "$base")
  if [ -n "$pinned" ] && [ "$pinned" != "$ROWS" ]; then
    batches=$(( (ROWS + pinned - 1) / pinned ))
    echo "note: $model is pinned to $pinned row(s) and ROWS=$ROWS, so each request runs \
$batches batch(es)$([ $((ROWS % pinned)) -ne 0 ] && echo ', the last padded')" >&2
  fi

  for provider in $PROVIDERS; do
    tag="$model-$(echo "$provider" | tr ':' '-')"
    model_config="$RUN_DIR/$tag.model.json"
    service_config="$RUN_DIR/$tag.service.json"

    # Only the provider differs between runs, so it is substituted rather than
    # duplicated into a committed file per combination.
    python3 -c '
import json, sys
base, model_out, service_out, provider, port = sys.argv[1:6]
config = json.load(open(base))
config["execution_provider"] = provider
json.dump(config, open(model_out, "w"), indent=2)
json.dump({"connection_concurrency": 100, "port_number": int(port),
           "model_config_path": model_out}, open(service_out, "w"), indent=2)
' "$base" "$model_config" "$service_config" "$provider" "$PORT"

    echo "=============================================================="
    echo "  $model on $provider"
    echo "=============================================================="
    ./target/release/hushar \
      --config-uri "$service_config" \
      --inference-log-uri "$RUN_DIR/inference-logs" \
      --port "$PORT" > "$RUN_DIR/logs/$tag.server.log" 2>&1 &
    SERVER_PID=$!

    # A 190 MB model takes a while to load, so wait for the listening line rather than
    # guessing a sleep.
    for _ in $(seq 1 240); do
      grep -q "listening on" "$RUN_DIR/logs/$tag.server.log" 2>/dev/null && break
      kill -0 $SERVER_PID 2>/dev/null || break
      sleep 1
    done
    if ! grep -q "listening on" "$RUN_DIR/logs/$tag.server.log" 2>/dev/null; then
      echo "  server did not start; last lines:"
      tail -6 "$RUN_DIR/logs/$tag.server.log" | sed 's/^/    /'
      kill $SERVER_PID 2>/dev/null; SERVER_PID=""
      continue
    fi
    grep -E "loaded on|inputs " "$RUN_DIR/logs/$tag.server.log" | head -2 | sed 's/^/  /'

    first=1
    for rate in $TPS; do
      # Only the first rate pays full warmup; the session is hot for the rest.
      warm=$WARMUP; [ $first -eq 1 ] || warm=3
      first=0
      log="$RUN_DIR/logs/$tag.$rate.client.log"
      ./target/release/benchmark-client \
        --server "http://127.0.0.1:$PORT" \
        --model-config "$model_config" \
        --vocabulary "$GEN_DIR/bench_string_vocabulary.json" \
        --tps "$rate" --duration-secs "$DURATION" --warmup-secs "$warm" \
        --rows "$ROWS" --concurrency "$CONCURRENCY" \
        --label "$tag @ $rate req/s" 2>&1 | tee "$log"
      scripts/summarise_benchmark.py "$tag" "$rate" "$log" >> "$SUMMARY"
    done

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    SERVER_PID=""
    echo
  done
done

echo "=============================================================="
echo "  summary"
echo "=============================================================="
cat "$SUMMARY"
echo
echo "logs and configs: $RUN_DIR"
