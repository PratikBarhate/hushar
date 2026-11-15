#!/usr/bin/env bash
# Starts one benchmark lane on a client host and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect and stay disconnected: every descriptor is redirected, the process is its own
# session leader, and each cell is written to S3 the moment it finishes. Re-running a lane
# skips whatever S3 already holds, so this is also how a lane is resumed.
#
#   fanout.py cl1 -f launch.sh --timeout 300 \
#     --env REV=... --env MODEL_SPECS="10m:<md5> 20m:<md5>" \
#     --env RIG='c9g-1=172.31.0.1' --env SERVER_FOR=c9g-1
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# `size:md5 size:md5`, in the order they should be measured. One lane runs them one after
# the other inside a SINGLE detached orchestrator, so a rig that could only get one GPU
# still needs no attention between sizes. The md5 travels with the size so the two cannot
# be paired up wrongly.
MODEL_SPECS="${MODEL_SPECS:-${MODEL_SIZE:-}:${MODEL_MD5:-}}"
[ "$MODEL_SPECS" != ":" ] || { echo "set MODEL_SPECS, e.g. '10m:<md5> 20m:<md5>'" >&2; exit 2; }
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
# Which server this client drives, and every server's address. Passed in rather than
# written down: this run launches its own hosts, so an address in a script would be a
# guess about a machine that may not exist.
RIG="${RIG:?RIG required, e.g. 'g5-1=172.31.0.1'}"
LANE_MAP="${LANE_MAP:-cl1=c9g-1;cl2=c9g-2;cl3=c9g-3;cl4=c9g-4}"
SERVER_FOR="${SERVER_FOR:-}"
RUN="${RUN:-minibatch-size-cpu-c9g-10M-20M}"
# THE LANE'S HALF OF THE GRID. Both travel from hosts.py through fanout, because a lane
# owns one model at one request size and a command line that forgot either would measure
# the wrong half under this run's name.
LOADS="${LOADS:-}"
# THE AXIS, empty by default so sweep.py's own per-row-count table applies. Each entry is
# a mini-batch size IN ROWS. Override e.g. MINI_BATCHES='100=100,50,10,5'.
MINI_BATCHES="${MINI_BATCHES:-}"
# ONE THREAD PER GRAPH CALL, which is both the service's own default and the only width at
# which this grid's deepest splits can overlap: `session_profile` sizes a session's
# mini-batch overlap as cores/width, and `ExecutionPlan::resolve` refuses a split unless
# that is above 1. See the README.
INTRA="${INTRA:-}"
# The unsplit reference's width -- the full host, legal only because that cell configures
# no mini batch at all.
REF_INTRA="${REF_INTRA:-}"
REFERENCE="${REFERENCE:-1}"
REPEATS="${REPEATS:-2}"
REPEAT_MB="${REPEAT_MB:-}"
SPIN="${SPIN:-0}"
CONCURRENCY_PER_SESSION="${CONCURRENCY_PER_SESSION:-}"
SESSIONS="${SESSIONS:-}"
SAMPLE_RATE="${SAMPLE_RATE:-0.2}"
WARMUP="${WARMUP:-300}"
DURATION="${DURATION:-900}"
SEGMENTS="${SEGMENTS:-15}"
SETTLE="${SETTLE:-60}"
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

aws s3 cp "s3://$BUCKET/dist/sweep-cpu-$REV.py" /root/sweep.py --only-show-errors
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
  MODEL_SPECS="$MODEL_SPECS" \
  RIG="$RIG" \
  WARMUP="$WARMUP" DURATION="$DURATION" SEGMENTS="$SEGMENTS" SETTLE="$SETTLE" \
  LOADS="$LOADS" MINI_BATCHES="$MINI_BATCHES" \
  INTRA="$INTRA" REF_INTRA="$REF_INTRA" REFERENCE="$REFERENCE" \
  REPEATS="$REPEATS" REPEAT_MB="$REPEAT_MB" \
  SPIN="$SPIN" SESSIONS="$SESSIONS" SAMPLE_RATE="$SAMPLE_RATE" \
  CONCURRENCY_PER_SESSION="$CONCURRENCY_PER_SESSION" \
  CELL_PREFIX="${CELL_PREFIX:-}" \
  MAX_CELLS="$MAX_CELLS" \
  SERVER="$SERVER" \
  ORT_VER="${ORT_VER:-1.28.0}" \
  LANE_TAG="$LANE_TAG" LANE_LOG="$LOG" LANE_KEY="$LANE_KEY" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER specs=[$MODEL_SPECS] loads=[$LOADS]"
    RC=0
    for spec in $MODEL_SPECS; do
      SIZE="${spec%%:*}"; MD5="${spec##*:}"
      echo
      echo "===== $SIZE ($(date -u +%FT%TZ)) ====="
      MODEL_SIZE="$SIZE" MODEL_MD5="$MD5" python3 /root/sweep.py "$SERVER"
      THIS=$?
      echo "===== $SIZE done rc=$THIS ($(date -u +%FT%TZ)) ====="
      # The log is pushed after every SIZE as well as at the end, so progress is readable
      # from the bucket alone even if the second size is still running.
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
echo "$HOST_NAME -> $SERVER, lane $LANE_TAG, specs [$MODEL_SPECS], log $LOG"
ps -eo pid,args= | awk '$2 == "python3" && $3 == "/root/sweep.py"'
head -20 "$LOG" 2>/dev/null || true
