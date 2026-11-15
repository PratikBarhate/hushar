#!/usr/bin/env bash
# Measures whether one (sessions x intra-op width) shape can SERVE each lane's offered
# load, on all four lanes at once.
#
#   SESSIONS=4 INTRA=16 TAG=s4i16 ./probe_shape.sh
#
# WHY THIS EXISTS. `threading.sessions_per_model` is not a thread-pool width -- it is the
# number of requests the server scores AT ONCE, because a request leases a session group
# exclusively while it scores ("1 concurrent request(s)" in the banner). So a shape's
# throughput ceiling is `sessions / service_time`, and a shape that cannot reach the
# offered rate makes every cell in its group a measurement of queueing instead of of the
# mini-batch size. The smoke cell found sessions=1 ceilinged at 103 req/s against 250
# offered, which would have cost the whole grid.
#
# Two cells per lane, which is the cheapest pair that can answer it: the mini-batch size
# equal to the row count (one call per request -- the SLOWEST per-request configuration,
# so the hardest to serve) and the deepest split in the grid (the most pieces, so the most
# exposed to the overlap ceiling). If both hold, everything between them will.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
: "${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
REV="${REV:?set REV}"
SESSIONS="${SESSIONS:?set SESSIONS}"
INTRA="${INTRA:?set INTRA}"
TAG="${TAG:-s${SESSIONS}i${INTRA}}"
RUN="${RUN:-minibatch-size-cpu-c9g-probe}"
WARMUP="${WARMUP:-30}"
DURATION="${DURATION:-90}"
SEGMENTS="${SEGMENTS:-3}"
SETTLE="${SETTLE:-5}"

MD5_10m=5539c97131c896db6c58088d9f9b9ee3
MD5_20m=30f9c5330b808a34b55fe88048a20da6

RIG=$(python rig_env.py)

# lane:client:size:md5:loads:mini-batches -- the baseline and the deepest split of each
lane() {
  local client="$1" server="$2" size="$3" md5="$4" loads="$5" mbs="$6"
  python fanout.py "$client" -f launch.sh --timeout 300 \
    --env REV="$REV" --env RUN="$RUN" \
    --env MODEL_SPECS="$size:$md5" \
    --env RIG="$RIG" --env SERVER_FOR="$server" \
    --env LOADS="$loads" --env MINI_BATCHES="$mbs" \
    --env SESSIONS="$SESSIONS" --env INTRA="$INTRA" \
    --env REFERENCE=0 --env REPEATS=0 \
    --env CELL_PREFIX="$TAG" \
    --env WARMUP="$WARMUP" --env DURATION="$DURATION" \
    --env SEGMENTS="$SEGMENTS" --env SETTLE="$SETTLE" \
    --env LANE_TAG="probe-$TAG" \
    --env HUSHAR_BENCH_BUCKET="$HUSHAR_BENCH_BUCKET" >"/tmp/probe-$TAG-$server.log" 2>&1
}

echo "== probing sessions=$SESSIONS intra=$INTRA (overlap ceiling $((64 / INTRA))) tag=$TAG"
lane cl1 c9g-1 10m "$MD5_10m" "100x250" "100=100,5"  &
lane cl2 c9g-2 10m "$MD5_10m" "500x50"  "500=500,10" &
lane cl3 c9g-3 20m "$MD5_20m" "100x100" "100=100,5"  &
lane cl4 c9g-4 20m "$MD5_20m" "500x20"  "500=500,10" &
wait
echo "== all four lanes launched; each runs 2 cells of about $(( (WARMUP + DURATION + SETTLE + 60) / 60 )) min"
