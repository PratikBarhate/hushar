#!/usr/bin/env bash
# Starts one benchmark lane on a client host and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect and stay disconnected: every descriptor is redirected, the process is its own
# session leader, and each cell is written to S3 the moment it finishes. Re-running a lane
# skips whatever S3 already holds, so this is also how a lane is resumed.
#
#   fanout.py cl1 -f launch.sh --timeout 300 \
#     --env REV=... --env SPECS="100m:fp32" \
#     --env RIG='trn1-1=172.31.0.1;trn1-2=172.31.0.2' --env SERVER_FOR=trn1-1
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# WHAT THIS LANE MEASURES, as `size:precision` pairs, in order, inside a SINGLE detached
# orchestrator. One pair per lane is the plan -- `trn1-1` is the FP32 arm and `trn1-2` the
# BF16 arm, so the two lanes carry equal NEFF-compile weight -- but a rig that could only
# place ONE server passes both here and runs them one after the other with no attention
# in between. That is a normal outcome in `us-west-2b`/`2d` and is better for the
# precision comparison, since both arms then sit on one machine; it only costs wall clock.
#
# Unlike the GPU run there is no md5 beside the size: Neuron does not compile ONNX, so
# there is no graph to pin before the host builds its own NEFFs. The model is pinned by
# the seeded architecture, the target parameter count AND THE CAST, and `prime` records
# the SHA-256 of what it produced plus the compiled-versus-PyTorch error that proves the
# cast applied.
SPECS="${SPECS:-${MODEL_SIZE:+$MODEL_SIZE:${PRECISION:-fp32}}}"
[ -n "$SPECS" ] || {
  echo "set SPECS, e.g. '100m:fp32' or '100m:fp32 100m:bf16'" >&2; exit 2; }
for spec in $SPECS; do
  case "$spec" in
    *:fp32|*:bf16) ;;
    *) echo "SPECS entry '$spec' must be <size>:<fp32|bf16>" >&2; exit 2 ;;
  esac
done
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
# Which server this client drives, and every server's address. Passed in rather than
# written down: this run launches its own hosts, so an address in a script would be a
# guess about a machine that may not exist.
RIG="${RIG:?RIG required, e.g. 'trn1-1=172.31.0.1'}"
LANE_MAP="${LANE_MAP:-cl1=trn1-1;cl2=trn1-2;cl3=trn1-3}"
SERVER_FOR="${SERVER_FOR:-}"
RUN="${RUN:-featurewire-log20-prec-g5-trn1-100M}"
# prime | main | all
PHASE="${PHASE:-all}"
WARMUP="${WARMUP:-300}"
DURATION="${DURATION:-900}"
SEGMENTS="${SEGMENTS:-15}"
SETTLE="${SETTLE:-60}"
# THE ONE AXIS: session counts, comma separated. There is no mini-batch variable -- the
# mini batch IS the row count, so every request is one NEFF call, and sweep.py refuses
# the previous campaign's split variables BY NAME so a command line copied from it fails
# loudly instead of quietly measuring a split.
SESSIONS="${SESSIONS:-2}"
# BOTH PINNED BY INSTRUCTION, and named here rather than buried because a run that
# changes either is a different experiment. Raising CORES_PER_REPLICA is the knob
# config-sweep-neuron-trn1-10M-50M found worth 41% on 50M. Raising CONTEXTS_PER_REPLICA
# only matters when a replica serves more than one piece, which this grid never does.
CORES_PER_REPLICA="${CORES_PER_REPLICA:-1}"
CONTEXTS_PER_REPLICA="${CONTEXTS_PER_REPLICA:-1}"
MAX_CELLS="${MAX_CELLS:-0}"
LANE_TAG="${LANE_TAG:-main}"

if [ -n "$SERVER_FOR" ]; then
  SERVER="$SERVER_FOR"
else
  SERVER=$(echo "$LANE_MAP" | tr ';' '\n' | awk -F= -v c="$HOST_NAME" '$1 == c { print $2 }')
fi
[ -n "$SERVER" ] || {
  echo "no server mapped for $HOST_NAME in LANE_MAP=$LANE_MAP" >&2; exit 2; }
echo "$RIG" | tr ';' '\n' | grep -q "^$SERVER=" || {
  echo "$SERVER is not in RIG=$RIG" >&2; exit 2; }

export HOME=/root
cd /root

running() {
  ps -eo args= | awk -v want="$1" \
    '{ if ($1 == "python3" && $2 == want) n++ } END { exit (n ? 0 : 1) }'
}

aws s3 cp "s3://$BUCKET/dist/sweep-neuron-$REV.py" /root/sweep.py --only-show-errors
python3 -c "import py_compile; py_compile.compile('/root/sweep.py', doraise=True)"

if running /root/sweep.py; then
  echo "an orchestrator is already running on $HOST_NAME"
  ps -eo pid,args= | awk '$2 == "python3" && $3 ~ /sweep.*\.py$/'
  exit 0
fi

LOG="/root/orchestrator-$LANE_TAG-$SERVER.log"
# A SECOND RUN OF A LANE MUST NOT OVERWRITE THE FIRST'S LOG. LANE_TAG already separates
# the on-host file, but the S3 key did not use it -- so a `PHASE=prime` re-run replaced a
# completed lane's log, cells and all, with one containing only prime lines. The default
# tag keeps the familiar name so nothing else has to change.
if [ "$LANE_TAG" = main ]; then
  LANE_KEY="runs/$RUN/raw/driver/lane-$SERVER.log"
else
  LANE_KEY="runs/$RUN/raw/driver/lane-$LANE_TAG-$SERVER.log"
fi
: > "$LOG"
# LANE_LOG and LANE_KEY go through the environment rather than being interpolated into
# the detached shell's single-quoted body: a path built by nesting quotes inside quotes is
# how an earlier version of this file produced a log named `...log:` and an upload that
# could not find it.
setsid nohup env \
  HUSHAR_BENCH_BUCKET="$BUCKET" REV="$REV" RUN="$RUN" \
  SPECS="$SPECS" \
  RIG="$RIG" PHASE="$PHASE" \
  WARMUP="$WARMUP" DURATION="$DURATION" SEGMENTS="$SEGMENTS" SETTLE="$SETTLE" \
  SESSIONS="$SESSIONS" \
  CORES_PER_REPLICA="$CORES_PER_REPLICA" \
  CONTEXTS_PER_REPLICA="$CONTEXTS_PER_REPLICA" \
  COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-14400}" \
  LOADS="${LOADS:-}" MAX_CELLS="$MAX_CELLS" \
  SERVER="$SERVER" \
  LANE_TAG="$LANE_TAG" LANE_LOG="$LOG" LANE_KEY="$LANE_KEY" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER phase=$PHASE specs=[$SPECS]"
    RC=0
    for SPEC in $SPECS; do
      SIZE="${SPEC%%:*}"
      PREC="${SPEC##*:}"
      echo
      echo "===== $SIZE $PREC ($(date -u +%FT%TZ)) ====="
      MODEL_SIZE="$SIZE" PRECISION="$PREC" python3 /root/sweep.py "$SERVER"
      THIS=$?
      echo "===== $SIZE $PREC done rc=$THIS ($(date -u +%FT%TZ)) ====="
      # The log is pushed after every SPEC as well as at the end, so progress is readable
      # from the bucket alone even if a second arm is still running.
      aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
      # An arm that fails does not stop the next one: they are independent measurements
      # and half a run in S3 is worth more than none.
      [ "$THIS" -eq 0 ] || RC="$THIS"
    done
    echo "ALL DONE rc=$RC $(date -u +%FT%TZ)"
    aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
    test "$RC" -eq 0
  ' >> "$LOG" 2>&1 < /dev/null &

sleep 6
echo "$HOST_NAME -> $SERVER, lane $LANE_TAG, specs [$SPECS], log $LOG"
ps -eo pid,args= | awk '$2 == "python3" && $3 == "/root/sweep.py"'
head -20 "$LOG" 2>/dev/null || true
