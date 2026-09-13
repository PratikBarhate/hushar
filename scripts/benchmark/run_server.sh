#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Runs hushar on the SERVER host, for the client on the other host to drive.
#
#   scripts/benchmark/run_server.sh start          # raw model, provider from env.sh
#   MODEL=raw-dcn scripts/benchmark/run_server.sh start
#   PROVIDER=trt:0 THREADS=4 scripts/benchmark/run_server.sh start
#   scripts/benchmark/run_server.sh status
#   scripts/benchmark/run_server.sh stop
#
# It starts, waits for the listening line, and returns -- so it does not hold the SSH
# session, and one server serves a whole rate sweep. Loading a ~190 MB model takes long
# enough that restarting per rate would dominate the wall clock, and one session keeps
# the comparison between rates exact.
#
# Environment:
#   MODEL        raw (default), raw-dcn, vectorized, named, arch-ab, raw-ab,
#                raw-dcn-ab, raw-batch1, raw-dcn-batch1, vectorized-batch1,
#                named-batch1.
#                raw and raw-dcn are the two architectures, both fed every feature
#                exactly as the caller sent it, with the graph doing its own scaling
#                and lookups: raw is a transformer encoder, raw-dcn is DCNv2 + MaskNet.
#                vectorized and named have the SERVICE transform features instead, and
#                are on the transformer, so that comparison holds the architecture still.
#                arch-ab loads both architectures and splits traffic between them, which
#                measures the pair in one session on one host. raw-ab and raw-dcn-ab are
#                the roll-out shape: one model loaded twice under two names, which is
#                what two arms of the same architecture cost
#   AB_PERCENT   50   share of traffic the candidate takes, with any of the -ab models.
#                Half each puts the same load on both, which is what makes their
#                latencies comparable; a real roll-out starts far lower
#   PROVIDER     cpu (default, or whatever env.sh set), trt:0, xnnpack:4
#   THREADS      threading.intra_op_threads, default 1 -- threads ONNX Runtime may use
#                inside one operator. Say which you used in the report
#   INFERENCE_CONCURRENCY  unset (one per core)  concurrent inferences. Lower it when
#                raising THREADS: the load is INFERENCE_CONCURRENCY x THREADS, so leaving
#                both high oversubscribes the cores
#   WORKER_THREADS    unset (one per core)  tokio async workers. These never run
#                inference, so 2 or 3 is plenty and the rest of the cores go to the engine
#   COMPUTE_CORES  unset  logical CPUs the inference threads may use, as "0-95" or
#                "2,4,6,8". On a multi-NUMA instance pin to ONE node -- see
#                documentation/threads-and-cores.md. Linux only
#   WORKER_CORES   unset  logical CPUs the async pool may use. Keep it disjoint from
#                COMPUTE_CORES, which is the whole point of setting either
#   PORT         8279
#   BIND         0.0.0.0   the client is on another host, so this must not be loopback
#   SESSIONS     1     independent ONNX Runtime sessions per model. The intra-op pool
#                belongs to the session, so 1 means every concurrent request to a model
#                queues for one pool and latency climbs with load on an idle machine.
#                n gives n pools, each pinned to its own slice of the cores. Costs one
#                copy of the weights per session, and wants ALLOW_SPINNING=0 -- most
#                pools are idle at any instant and a spinning idle pool costs a busy one
#   ALLOW_SPINNING  unset  whether idle ONNX Runtime intra-op threads spin waiting for
#                work. Unset leaves the runtime's own default, which is to spin. Set it
#                to 0 whenever the execution provider brings a thread-pool of its own --
#                XNNPACK and OpenVINO both do -- because two pools sized for the same
#                cores means the idle one burns the time the busy one needs. The pairing
#                that goes with it is THREADS=1, so ORT keeps no pool of its own:
#                  ALLOW_SPINNING=0 THREADS=1 PROVIDER=xnnpack:21
#   SAMPLE_RATE  0.2   fraction of requests whose features are logged. A fifth
#                exercises the log path without the feature copies dominating what is
#                being measured. 1.0 logs everything; 0.0 turns the log off
#   LOG_URI      $RUN_DIR/inference-logs   where inference logs go. An s3://bucket/prefix
#                measures the path a deployment actually uses, and keeps the rows after
#                the instance is gone
#   LOG_BATCH    5000  batches buffered before one write, as inference_log.batch_size.
#                This -- not SAMPLE_RATE -- is what sets the SIZE of a flush, and the
#                flush is what disturbs latency: 5000 batches of 100 rows x 212 features
#                is a ~2.8 GB object, and building it competes for memory bandwidth with
#                every inference in flight. Lowering the sample rate makes flushes rarer
#                but each one just as violent, so lower this instead when the tail matters
#   LOG_SENDS    4     writes allowed at once, as inference_log.max_sends_in_flight.
#                Raise it with a small LOG_BATCH, or frequent small writes queue and
#                the log sheds rows rather than the request path blocking
#   CW_NAMESPACE unset  publish metrics to this CloudWatch namespace. Unset prints them
#                to the server log instead, which is enough for one host but leaves
#                nothing to compare runs with afterwards
#   METRICS_BATCH  500  batches per metrics publish
#   CONN_LIMIT   unset (the server's own 500)  in-flight requests admitted per
#                connection, as threading.connection_concurrency. Admission, not work:
#                INFERENCE_CONCURRENCY is what bounds the cores. Set it only to measure
#                admission itself
#   MINI_BATCH_SIZE  unset  rows per mini batch, as the model config's
#                mini_batch_size. Set it and a request's rows are cut into batches of
#                this many and scored AT THE SAME TIME, one scoped thread each. That is
#                what lowers latency: threads inside one operator stop helping well
#                before the core count, because the operators are a chain of barriers,
#                while separate mini batches share no barrier at all. Pair it with
#                THREADS=1 and SESSIONS=1 -- the parallelism now comes from the split, a
#                pool inside each batch competes with it, and the scoped threads inherit
#                the caller's core slice so one session gives them the whole compute set
#   MINI_BATCH_FIXED  unset  pad the last mini batch up to MINI_BATCH_SIZE, as
#                is_fixed. Needed only by a graph whose leading dimension is pinned;
#                leave it unset and the last batch is however many rows remain
#   LABEL        derived from MODEL and PROVIDER   names this run's files. Set it when
#                the provider string carries options, since those make for an unwieldy
#                file name: LABEL=c9g-xnnpack
#   RUN_DIR      /tmp/hushar-server

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -r "$ROOT/env.sh" ] && . "$ROOT/env.sh"

MODEL="${MODEL:-raw}"
PROVIDER="${PROVIDER:-cpu}"
THREADS="${THREADS:-1}"
PORT="${PORT:-8279}"
BIND="${BIND:-0.0.0.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.2}"
CONN_LIMIT="${CONN_LIMIT:-}"
ALLOW_SPINNING="${ALLOW_SPINNING:-}"
SESSIONS="${SESSIONS:-1}"
CW_NAMESPACE="${CW_NAMESPACE:-}"
METRICS_BATCH="${METRICS_BATCH:-500}"
LOG_BATCH="${LOG_BATCH:-5000}"
LOG_SENDS="${LOG_SENDS:-4}"
AB_PERCENT="${AB_PERCENT:-50}"
WORKER_THREADS="${WORKER_THREADS:-}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-}"
COMPUTE_CORES="${COMPUTE_CORES:-}"
WORKER_CORES="${WORKER_CORES:-}"
RUN_DIR="${RUN_DIR:-/tmp/hushar-server}"
# Overridable so several model sizes can live side by side, each generated into its
# own directory -- see generate_benchmark_models.py --out-dir.
GEN_DIR="${GEN_DIR:-benchmark-data/generated}"

# A provider string carrying options -- "openvino:CPU:threads=21:streams=1" -- makes an
# unwieldy file name, so LABEL overrides it. The derived form keeps working for a bare
# provider, which is the common case.
TAG="${LABEL:-$MODEL-$(echo "$PROVIDER" | tr ':' '-')}"
LOG_URI="${LOG_URI:-$RUN_DIR/inference-logs}"
PID_FILE="$RUN_DIR/server.pid"
LOG="$RUN_DIR/$TAG.server.log"

status() {
  if [ -r "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running, pid $(cat "$PID_FILE")"
    [ -r "$RUN_DIR/current.log" ] && tail -3 "$(cat "$RUN_DIR/current.log")" | sed 's/^/  /'
    return 0
  fi
  echo "not running"
  return 1
}

stop() {
  if [ -r "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null || true
      echo "stopped pid $pid"
    fi
    rm -f "$PID_FILE"
  else
    echo "no pid file at $PID_FILE"
  fi
}

start() {
  [ -x ./target/release/hushar ] \
    || { echo "no ./target/release/hushar -- run bootstrap.sh first" >&2; exit 1; }
  [ -n "${ORT_DYLIB_PATH:-}" ] \
    || { echo "ORT_DYLIB_PATH is not set; the engine needs a libonnxruntime" >&2; exit 1; }

  case "$MODEL" in
    raw)               base=$GEN_DIR/bench_raw_model_config.json ;;
    raw-dcn)           base=$GEN_DIR/bench_raw_dcn_model_config.json ;;
    vectorized)        base=$GEN_DIR/bench_model_config.json ;;
    named)             base=$GEN_DIR/bench_named_model_config.json ;;
    # Both architectures resident, half the traffic each: the comparison in one session.
    arch-ab)           base=$GEN_DIR/bench_raw_model_config.json
                       candidate=$GEN_DIR/bench_raw_dcn_model_config.json ;;
    # One model loaded twice under two names -- the roll-out shape, without a second set
    # of weights that would cost the same to serve.
    raw-ab)            base=$GEN_DIR/bench_raw_model_config.json
                       candidate=$GEN_DIR/bench_raw_model_candidate_config.json ;;
    raw-dcn-ab)        base=$GEN_DIR/bench_raw_dcn_model_config.json
                       candidate=$GEN_DIR/bench_raw_dcn_model_candidate_config.json ;;
    raw-batch1)        base=$GEN_DIR/bench_raw_model_batch1_config.json ;;
    raw-dcn-batch1)    base=$GEN_DIR/bench_raw_dcn_model_batch1_config.json ;;
    vectorized-batch1) base=$GEN_DIR/bench_model_batch1_config.json ;;
    named-batch1)      base=$GEN_DIR/bench_named_model_batch1_config.json ;;
    *) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
  esac
  [ -f "$base" ] || {
    echo "missing $base" >&2
    echo "  package with --with-models, or run scripts/generate_benchmark_models.py here" >&2
    case "$MODEL" in *-batch1) echo "  the pinned copies need --with-batch1" >&2 ;; esac
    exit 1
  }

  if status >/dev/null 2>&1; then
    echo "already running, pid $(cat "$PID_FILE") -- stop it first" >&2
    exit 1
  fi

  # Only a local URI is a directory to create; an s3:// or kinesis:// one is not.
  mkdir -p "$RUN_DIR"
  case "$LOG_URI" in *://) ;; *://*) ;; *) mkdir -p "$LOG_URI" ;; esac
  model_config="$RUN_DIR/$TAG.model.json"
  service_config="$RUN_DIR/$TAG.service.json"

  # Only the provider, the thread count and the listener differ between runs, so they
  # are substituted rather than duplicated into a committed file per combination. Every
  # process setting is in this one file now -- the runner passes no flags but the URI.
  #
  # threading.connection_concurrency is omitted unless CONN_LIMIT is set, so every host
  # takes the same default of 500. Admission is not the thing a hardware comparison is
  # measuring -- INFERENCE_CONCURRENCY is -- and the banner records both, so the run is
  # reproducible from the report either way.
  python3 -c '
import json, sys
(base, model_out, service_out, provider, threads, port, bind, logs, rate, limit,
 candidate, candidate_out, percent, workers, inference,
 compute_cores, worker_cores, spinning, namespace, metrics_batch, sessions,
 log_batch, log_sends) = sys.argv[1:24]


# Substitutes the per-host provider into one model configuration. The thread counts
# are a service setting now, since they describe the host rather than the model.
def rewrite(path, out):
    config = json.load(open(path))
    config["execution_provider"] = provider
    # Mini batching is a model-config choice, so it is substituted here alongside the
    # provider rather than passed as a flag. Absent means one batch per request.
    import os
    size = os.environ.get("MINI_BATCH_SIZE", "")
    if size:
        config["mini_batch_size"] = int(size)
        if os.environ.get("MINI_BATCH_FIXED", "") not in ("", "0", "false", "no"):
            config["is_fixed"] = True
    json.dump(config, open(out, "w"), indent=2)


rewrite(base, model_out)
service = {
    "bind_address": bind,
    "port_number": int(port),
    "model_config_path": model_out,
    "inference_log": {"uri": logs, "batch_size": int(log_batch),
                      "sample_rate": float(rate),
                      "max_sends_in_flight": int(log_sends)},
    "metrics": {"batch_size": int(metrics_batch), "max_sends_in_flight": 4},
    "threading": {"intra_op_threads": int(threads),
                  "sessions_per_model": int(sessions)},
}
if namespace:
    service["metrics"]["cloudwatch_namespace"] = namespace
if workers:
    service["threading"]["worker_threads"] = int(workers)
if inference:
    service["threading"]["inference_concurrency"] = int(inference)
if compute_cores:
    service["threading"]["compute_cores"] = compute_cores
if worker_cores:
    service["threading"]["worker_cores"] = worker_cores
# Tri-state: unset leaves the runtime default rather than choosing one here, so a run
# that says nothing about spinning is not silently different from the runtime default.
if spinning:
    service["threading"]["allow_spinning"] = spinning not in ("0", "false", "no")
if limit:
    service["threading"]["connection_concurrency"] = int(limit)
if candidate:
    # Both arms get the same provider and thread count, or the comparison would be
    # measuring the host settings rather than the models.
    rewrite(candidate, candidate_out)
    service["candidate_model"] = {"config_path": candidate_out,
                                  "traffic_percent": int(percent)}
json.dump(service, open(service_out, "w"), indent=2)
' "$base" "$model_config" "$service_config" "$PROVIDER" "$THREADS" "$PORT" "$BIND" \
  "$LOG_URI" "$SAMPLE_RATE" "$CONN_LIMIT" \
  "${candidate:-}" "$RUN_DIR/$TAG.candidate.json" "$AB_PERCENT" \
  "$WORKER_THREADS" "$INFERENCE_CONCURRENCY" "$COMPUTE_CORES" "$WORKER_CORES" \
  "$ALLOW_SPINNING" "$CW_NAMESPACE" "$METRICS_BATCH" "$SESSIONS" \
  "$LOG_BATCH" "$LOG_SENDS"

  echo "starting $MODEL on $PROVIDER, ${THREADS} intra-op thread(s), $BIND:$PORT"
  nohup ./target/release/hushar \
    --config-uri "$service_config" > "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  echo "$LOG" > "$RUN_DIR/current.log"

  # A GPU provider compiles the graph on load and a 190 MB model takes a while to read,
  # so wait for the line rather than guessing a sleep.
  for _ in $(seq 1 600); do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
    sleep 1
  done
  if ! grep -q "listening on" "$LOG" 2>/dev/null; then
    echo "server did not start; last lines of $LOG:" >&2
    tail -20 "$LOG" | sed 's/^/  /' >&2
    rm -f "$PID_FILE"
    exit 1
  fi

  # The banner is the contract: it names the provider actually registered, which is not
  # necessarily the one asked for if the host's library lacks it.
  grep -E "loaded on|inputs |outputs |features |A/B|threads ->|batch  :|admit  :|pool   :|spin   :|cores  :|note   :|metrics ->|inference logs ->|listening" "$LOG" | sed 's/^/  /'
  echo
  # hostname -I is Linux-only and returns nothing elsewhere, so fall back to instance
  # metadata and then to a placeholder rather than printing "http://:8279". The `|| true`
  # is what makes that fallback reachable: pipefail would otherwise turn hostname's own
  # failure into a fatal one, and the script would exit 1 having already started a server.
  ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [ -z "$ip" ]; then
    token=$(curl -fsS -m 1 -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null) || token=""
    [ -n "$token" ] && ip=$(curl -fsS -m 1 -H "X-aws-ec2-metadata-token: $token" \
      "http://169.254.169.254/latest/meta-data/local-ipv4" 2>/dev/null)
  fi
  echo "pid $(cat "$PID_FILE"), log $LOG"
  echo "drive it from the client host with:"
  echo "  SERVER=http://${ip:-<this-host-ip>}:$PORT \\"
  echo "    MODEL=$MODEL scripts/benchmark/run_client.sh"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) stop; start ;;
  -h|--help)
    awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0" ;;
  *) echo "usage: $0 [start|stop|status|restart]" >&2; exit 2 ;;
esac
