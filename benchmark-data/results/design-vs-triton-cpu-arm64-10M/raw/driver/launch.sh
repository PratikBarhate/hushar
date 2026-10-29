#!/usr/bin/env bash
# Starts one benchmark lane and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect and stay disconnected: every descriptor is redirected, the process is its own
# session leader, and each cell is written to S3 the moment it finishes. Re-running a lane
# skips whatever S3 already holds, so this is also how a lane is resumed.
#
#   fanout.py clients -f launch.sh --env REV=... --env MODEL_MD5=... \
#       --env RIG='c9g-1=172.31.0.1;c9g-2=172.31.0.2' --timeout 300
#
# The lane's own log is uploaded to S3 after every cell, so progress is readable from the
# bucket alone and nothing has to be pulled off a host that might be destroyed.
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
MODEL_MD5="${MODEL_MD5:?MODEL_MD5 required}"
MODEL_SIZE="${MODEL_SIZE:-10m}"
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
# Which server each client drives, and every server's address.
#   RIG       `name=ip;name=ip`  -- every server this invocation may talk to
#   LANE_MAP  `client=server;…`  -- which server THIS client drives
# Both are passed in rather than written down: this run launches its own hosts, so an
# address in a script would be a guess about a machine that may not exist. LANE_MAP is
# explicit rather than derived from the client's number, because a second rig's clients are
# `cl3` and `cl4` while its RIG has two entries, and index arithmetic silently produced an
# empty server name for both.
RIG="${RIG:?RIG required, e.g. 'c9g-1=172.31.0.1;c9g-2=172.31.0.2'}"
LANE_MAP="${LANE_MAP:-cl1=c9g-1;cl2=c9g-2;cl3=c9g-3;cl4=c9g-4}"
SERVER_FOR="${SERVER_FOR:-}"
RUN="${RUN:-design-vs-triton-cpu-arm64-10M}"
# Passed through to the sweep, so the tuning pass, a probe and the head-to-head all share
# one launcher.
PHASE="${PHASE:-head}"
# The winners the tuning pass measured, as JSON. Left empty, the sweep falls back to the
# previous campaign's winners AND records that it was not tuned.
BEST_HUSHAR="${BEST_HUSHAR:-}"
BEST_TRITON="${BEST_TRITON:-}"
ARMS="${ARMS:-}"
ONLY_ROWS="${ONLY_ROWS:-}"
WARMUP="${WARMUP:-120}"
DURATION="${DURATION:-900}"
SEGMENTS="${SEGMENTS:-3}"
SETTLE="${SETTLE:-60}"
SHAPE="${SHAPE:-4x16}"
MAX_CELLS="${MAX_CELLS:-0}"
EARLY_STOP="${EARLY_STOP:-1}"
LANE_TAG="${LANE_TAG:-main}"

# cl1 drives the first server named in RIG, cl2 the second, unless SERVER_FOR says
# otherwise. Derived from RIG rather than hardcoded, so adding a third pair needs no edit
# here.
# cl1 drives c9g-1, cl3 drives c9g-3, and so on, per LANE_MAP. SERVER_FOR overrides it.
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
# LANE_LOG and LANE_KEY go through the environment rather than being interpolated into the
# detached shell's single-quoted body: a path built by nesting quotes inside quotes is how
# this file once produced a log named `...log:` and an upload that could not find it.
setsid nohup env \
  HUSHAR_BENCH_BUCKET="$BUCKET" MODEL_MD5="$MODEL_MD5" MODEL_SIZE="$MODEL_SIZE" \
  REV="$REV" RUN="$RUN" SERVER="$SERVER" RIG="$RIG" \
  PHASE="$PHASE" BEST_HUSHAR="$BEST_HUSHAR" BEST_TRITON="$BEST_TRITON" \
  BEST_HUSHAR_BY_ROWS="${BEST_HUSHAR_BY_ROWS:-}" \
  BEST_TRITON_BY_ROWS="${BEST_TRITON_BY_ROWS:-}" \
  TUNE_ROWS="${TUNE_ROWS:-100}" TUNE_RATES="${TUNE_RATES:-10000,24000}" \
  ARMS="$ARMS" ONLY_ROWS="$ONLY_ROWS" \
  WARMUP="$WARMUP" DURATION="$DURATION" SEGMENTS="$SEGMENTS" \
  SETTLE="$SETTLE" SHAPE="$SHAPE" MAX_CELLS="$MAX_CELLS" EARLY_STOP="$EARLY_STOP" \
  LANE_TAG="$LANE_TAG" LANE_LOG="$LOG" LANE_KEY="$LANE_KEY" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER"
    python3 /root/sweep.py "$SERVER"
    RC=$?
    echo "ALL DONE rc=$RC $(date -u +%FT%TZ)"
    aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
    test "$RC" -eq 0
  ' >> "$LOG" 2>&1 < /dev/null &

sleep 6
echo "$HOST_NAME -> $SERVER, lane $LANE_TAG, log $LOG"
ps -eo pid,args= | awk '$2 == "python3" && $3 == "/root/sweep.py"'
head -20 "$LOG" 2>/dev/null || true
