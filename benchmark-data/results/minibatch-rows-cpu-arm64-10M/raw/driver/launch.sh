#!/usr/bin/env bash
# Starts one benchmark lane and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect: every descriptor is redirected, the process is its own session leader, and
# each cell is written to S3 the moment it finishes. Re-running a lane skips whatever S3
# already holds, so this is also how a lane is resumed.
#
#   fanout.py clients -f launch.sh --env REV=... --env MODEL_MD5=... --timeout 300
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
MODEL_MD5="${MODEL_MD5:?MODEL_MD5 required}"
MODEL_SIZE="${MODEL_SIZE:-10m}"
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
RUN="${RUN:-minibatch-rows-cpu-arm64-10M}"
# Passed through to the sweep, so a probe and the main run share one launcher.
ARMS="${ARMS:-}"
ONLY_ROWS="${ONLY_ROWS:-}"
WARMUP="${WARMUP:-120}"
DURATION="${DURATION:-300}"
SETTLE="${SETTLE:-60}"
SHAPE="${SHAPE:-4x16}"
MAX_CELLS="${MAX_CELLS:-0}"
LANE_TAG="${LANE_TAG:-main}"

case "$HOST_NAME" in
  cl1) SERVER=c9g-1 ;;
  cl2) SERVER=c9g-2 ;;
  *) echo "no server mapped for $HOST_NAME" >&2; exit 2 ;;
esac

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
: > "$LOG"
setsid nohup env \
  HUSHAR_BENCH_BUCKET="$BUCKET" MODEL_MD5="$MODEL_MD5" MODEL_SIZE="$MODEL_SIZE" \
  REV="$REV" RUN="$RUN" SERVER="$SERVER" ARMS="$ARMS" ONLY_ROWS="$ONLY_ROWS" \
  WARMUP="$WARMUP" DURATION="$DURATION" \
  SETTLE="$SETTLE" SHAPE="$SHAPE" MAX_CELLS="$MAX_CELLS" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER"
    python3 /root/sweep.py "$SERVER"
    RC=$?
    echo "ALL DONE rc=$RC $(date -u +%FT%TZ)"
    test "$RC" -eq 0
  ' >> "$LOG" 2>&1 < /dev/null &

sleep 6
echo "$HOST_NAME -> $SERVER, lane $LANE_TAG, log $LOG"
ps -eo pid,args= | awk '$2 == "python3" && $3 == "/root/sweep.py"'
head -20 "$LOG" 2>/dev/null || true
