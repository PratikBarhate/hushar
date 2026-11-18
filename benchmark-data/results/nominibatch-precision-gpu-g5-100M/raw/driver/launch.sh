#!/usr/bin/env bash
# Starts one benchmark lane on a client host and returns immediately.
#
# The detached chain outlives the SSM command that started it, so the laptop can
# disconnect and stay disconnected: every descriptor is redirected, the process is its own
# session leader, and each cell is written to S3 the moment it finishes. Re-running a lane
# skips whatever S3 already holds, so this is also how a lane is resumed.
#
#   fanout.py cl1 -f launch.sh --timeout 300 \
#     --env REV=... --env MODEL_SPECS="100m:<md5>:fp32 100m:<md5>:bf16" \
#     --env RIG='g5-1=172.31.0.1' --env SERVER_FOR=g5-1
set -euo pipefail

REV="${REV:?REV required}"
BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
# `size:md5:precision`, in the order they should be measured. One lane runs them one after
# the other inside a SINGLE detached orchestrator, so a rig that could only get one GPU
# still needs no attention between precisions. The md5 and the precision travel WITH the
# size rather than as parallel variables, so the three cannot be paired up wrongly.
#
# The precision field is what this run's axis rides on. It is optional and defaults to
# fp32, so a two-field spec from an earlier campaign still means what it meant.
MODEL_SPECS="${MODEL_SPECS:-${MODEL_SIZE:-}:${MODEL_MD5:-}}"
[ "$MODEL_SPECS" != ":" ] || { echo "set MODEL_SPECS, e.g. '100m:<md5>:fp32 100m:<md5>:bf16'" >&2; exit 2; }
HOST_NAME="${HOST_NAME:?HOST_NAME required}"
# Which server this client drives, and every server's address. Passed in rather than
# written down: this run launches its own hosts, so an address in a script would be a
# guess about a machine that may not exist.
RIG="${RIG:?RIG required, e.g. 'g5-1=172.31.0.1'}"
LANE_MAP="${LANE_MAP:-cl1=g5-1;cl2=g5-2;cl3=g5-3}"
SERVER_FOR="${SERVER_FOR:-}"
RUN="${RUN:-nominibatch-precision-gpu-g5-100M}"
# prime | main | all. There is no probe phase: every rung states its own session count.
PHASE="${PHASE:-all}"
# The tensor wire: the client sends every feature as one vector of f32 and the server does
# no feature vectorization. Triton receives the same bytes.
ARMS="${ARMS:-hushar-tensor,triton}"
WARMUP="${WARMUP:-300}"
DURATION="${DURATION:-900}"
SEGMENTS="${SEGMENTS:-15}"
SETTLE="${SETTLE:-60}"
LOADS="${LOADS:-}"
# THE SESSION COUNT, empty by default so sweep.py's own per-row-count table applies -- a
# pinned 2, since the previous campaign showed the count does not move the median. Each
# entry is `threading.sessions_per_model` on hushar, `instance_group` count on Triton.
# Override e.g. SESSIONS='100=2,4;500=2,4'. There is no mini-batch knob: this run
# configures none, and sweep.py refuses RUNGS by name so a command line copied from the
# mini-batch campaign fails loudly instead of measuring something else.
SESSIONS="${SESSIONS:-}"
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
LANE_KEY="runs/$RUN/raw/driver/lane-$LANE_TAG-$SERVER.log"
: > "$LOG"
# LANE_LOG and LANE_KEY go through the environment rather than being interpolated into
# the detached shell's single-quoted body: a path built by nesting quotes inside quotes is
# how an earlier version of this file produced a log named `...log:` and an upload that
# could not find it.
setsid nohup env \
  HUSHAR_BENCH_BUCKET="$BUCKET" REV="$REV" RUN="$RUN" \
  MODEL_SPECS="$MODEL_SPECS" \
  RIG="$RIG" PHASE="$PHASE" ARMS="$ARMS" \
  WARMUP="$WARMUP" DURATION="$DURATION" SEGMENTS="$SEGMENTS" SETTLE="$SETTLE" \
  LOADS="$LOADS" \
  SESSIONS="$SESSIONS" MAX_CELLS="$MAX_CELLS" \
  SERVER="$SERVER" \
  ORT_VER="${ORT_VER:-1.28.3}" \
  TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.05-py3}" \
  LANE_TAG="$LANE_TAG" LANE_LOG="$LOG" LANE_KEY="$LANE_KEY" \
  bash -c '
    echo "launched $(date -u +%FT%TZ) for $SERVER phase=$PHASE specs=[$MODEL_SPECS]"
    RC=0
    for spec in $MODEL_SPECS; do
      SIZE="${spec%%:*}"
      REST="${spec#*:}"
      MD5="${REST%%:*}"
      # Three fields means the precision was named; two means fp32, which is what every
      # earlier campaign in this series measured.
      if [ "$REST" = "$MD5" ]; then PREC=fp32; else PREC="${REST#*:}"; fi
      echo
      echo "===== $SIZE $PREC ($(date -u +%FT%TZ)) ====="
      MODEL_SIZE="$SIZE" MODEL_MD5="$MD5" PRECISION="$PREC" \
        python3 /root/sweep.py "$SERVER"
      THIS=$?
      echo "===== $SIZE $PREC done rc=$THIS ($(date -u +%FT%TZ)) ====="
      # The log is pushed after every SPEC as well as at the end, so progress is readable
      # from the bucket alone even if the second precision is still running.
      aws s3 cp "$LANE_LOG" "s3://$HUSHAR_BENCH_BUCKET/$LANE_KEY" --only-show-errors || true
      # A precision that fails does not stop the next one: they are independent
      # measurements and half a run in S3 is worth more than none.
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
