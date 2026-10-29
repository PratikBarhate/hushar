#!/usr/bin/env bash
# Starts one benchmark lane on a client host and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect and stay disconnected: every descriptor is redirected, the process is its own
# session leader, and each cell is written to S3 the moment it finishes. Re-running a lane
# skips whatever S3 already holds, so this is also how a lane is resumed.
#
#   fanout.py cl1 -f launch.sh --timeout 300 \
#     --env REV=... --env SIZES="10m" \
#     --env RIG='trn1-1=172.31.0.1;trn1-2=172.31.0.2' --env SERVER_FOR=trn1-1
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# The model sizes this lane measures, in order, inside a SINGLE detached orchestrator. One
# size per lane is the plan -- every comparison this report draws is within one size, so a
# lane owns a whole model -- but a rig that could only place one server passes both here
# and needs no attention between them.
#
# Unlike the GPU run there is no md5 beside the size: Neuron does not compile ONNX, so
# there is no graph to pin before the host builds its own NEFFs. The model is pinned by
# the seeded architecture and the target parameter count, and `prime` records the SHA-256
# of what it produced.
SIZES="${SIZES:-${MODEL_SIZE:-}}"
[ -n "$SIZES" ] || { echo "set SIZES, e.g. '10m' or '10m 20m'" >&2; exit 2; }
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
# Which server this client drives, and every server's address. Passed in rather than
# written down: this run launches its own hosts, so an address in a script would be a
# guess about a machine that may not exist.
RIG="${RIG:?RIG required, e.g. 'trn1-1=172.31.0.1'}"
LANE_MAP="${LANE_MAP:-cl1=trn1-1;cl2=trn1-2;cl3=trn1-3}"
SERVER_FOR="${SERVER_FOR:-}"
RUN="${RUN:-minibatch-neuron-trn1-10M-20M}"
# prime | main | control | all
PHASE="${PHASE:-all}"
WARMUP="${WARMUP:-120}"
DURATION="${DURATION:-900}"
SEGMENTS="${SEGMENTS:-15}"
SETTLE="${SETTLE:-60}"
# The axes, empty by default so sweep.py's own tables apply. Set either to override, e.g.
# MINI_BATCHES='100=100,50;500=500,100' or REPLICAS='2,4'.
MINI_BATCHES="${MINI_BATCHES:-}"
REPLICAS="${REPLICAS:-2,4,6}"
# Pinned by instruction. Raising it is a DIFFERENT experiment -- it is the knob
# config-sweep-neuron-trn1-10M-50M found worth 41% on 50M -- so it is named here rather
# than buried, and a run that changes it has to say so in its report.
CORES_PER_REPLICA="${CORES_PER_REPLICA:-1}"
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

aws s3 cp "s3://$BUCKET/dist/sweep-$REV.py" /root/sweep.py --only-show-errors
python3 -c "import py_compile; py_compile.compile('/root/sweep.py', doraise=True)"

if running /root/sweep.py; then
  echo "an orchestrator is already running on $HOST_NAME"
  ps -eo pid,args= | awk '$2 == "python3" && $3 ~ /sweep.*\.py$/'
  exit 0
fi

LOG="/root/orchestrator-$LANE_TAG-$SERVER.log"
LANE_KEY="runs/$RUN/raw/driver/lane-$SERVER.log"
: > "$LOG"
# LANE_LOG and LANE_KEY go through the environment rather than being interpolated into
# the detached shell's single-quoted body: a path built by nesting quotes inside quotes is
# how an earlier version of this file produced a log named `...log:` and an upload that
# could not find it.
setsid nohup env \
  HUSHAR_BENCH_BUCKET="$BUCKET" REV="$REV" RUN="$RUN" \
  SIZES="$SIZES" \
  RIG="$RIG" PHASE="$PHASE" \
  WARMUP="$WARMUP" DURATION="$DURATION" SEGMENTS="$SEGMENTS" SETTLE="$SETTLE" \
  MINI_BATCHES="$MINI_BATCHES" REPLICAS="$REPLICAS" \
  CORES_PER_REPLICA="$CORES_PER_REPLICA" \
  LOADS="${LOADS:-}" MAX_CELLS="$MAX_CELLS" \
  SERVER="$SERVER" \
  LANE_TAG="$LANE_TAG" LANE_LOG="$LOG" LANE_KEY="$LANE_KEY" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER phase=$PHASE sizes=[$SIZES]"
    RC=0
    for SIZE in $SIZES; do
      echo
      echo "===== $SIZE ($(date -u +%FT%TZ)) ====="
      MODEL_SIZE="$SIZE" python3 /root/sweep.py "$SERVER"
      THIS=$?
      echo "===== $SIZE done rc=$THIS ($(date -u +%FT%TZ)) ====="
      # The log is pushed after every SIZE as well as at the end, so progress is readable
      # from the bucket alone even if a second size is still running.
      aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
      # A size that fails does not stop the next one: they are independent measurements
      # and half a run in S3 is worth more than none.
      [ "$THIS" -eq 0 ] || RC="$THIS"
    done
    echo "ALL DONE rc=$RC $(date -u +%FT%TZ)"
    aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
    test "$RC" -eq 0
  ' >> "$LOG" 2>&1 < /dev/null &

sleep 6
echo "$HOST_NAME -> $SERVER, lane $LANE_TAG, sizes [$SIZES], log $LOG"
ps -eo pid,args= | awk '$2 == "python3" && $3 == "/root/sweep.py"'
head -20 "$LOG" 2>/dev/null || true
