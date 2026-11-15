#!/usr/bin/env bash
# Polls both lanes from the laptop and appends one block per round to a local log.
#
#   HUSHAR_BENCH_BUCKET=... watch.sh [interval_seconds] > /tmp/watch-minibatch.log 2>&1 &
#   tail -40 /tmp/watch-minibatch.log
#
# Nothing here is required for the run to finish: the lanes are detached and write each
# cell to S3 as it completes, so this is a progress view and the laptop may be closed
# between rounds. Progress is the count of objects under the run's cells/ prefix, broken
# down by arm, against what the sweep says the lanes hold.
set -uo pipefail

BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
RUN="${RUN:-minibatch-rows-cpu-arm64-10M}"
EXPECTED="${EXPECTED:-72}"
INTERVAL="${1:-900}"
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
  echo "================ $(date -u +%FT%TZ) ================"
  "$AWS" --profile hushar-bench --region us-west-2 \
    s3 ls "s3://$BUCKET/runs/$RUN/cells/" 2>/dev/null \
    | awk '{print $4}' | sed -E 's/\.json$//' > /tmp/watch-cells.txt
  echo "cells: $(wc -l < /tmp/watch-cells.txt | tr -d ' ') / $EXPECTED"
  # The arm is the field after the host in a cell id, e.g. c9g-1-R-l2000-r1t2000-spin1.
  sed -E 's/^(c9g-[0-9])-([A-Z])-.*/\1 \2/' /tmp/watch-cells.txt \
    | sort | uniq -c | awk '{printf "    %s arm %s: %s\n", $2, $3, $1}'
  FAILED=$("$AWS" --profile hushar-bench --region us-west-2 \
    s3 ls "s3://$BUCKET/runs/$RUN/failures/" 2>/dev/null | wc -l | tr -d ' ')
  echo "failed attempts recorded: $FAILED"
  "${PYTHON:-python3}" "$HERE/fanout.py" clients \
    -c 'for f in /root/orchestrator-main-*.log; do
          echo "--- $(basename $f)"
          grep -c "done " "$f" 2>/dev/null | sed "s/^/    measured: /"
          grep -cE "FAILED|ABORT|EXCEPTION|WARNING" "$f" 2>/dev/null | sed "s/^/    problems: /"
          tail -2 "$f"
        done' --timeout 240 --tail 16 2>/dev/null \
    | grep -vE "^=|^$|LIBARCHIVE|^-> " || echo "(poll failed)"
  echo
  sleep "$INTERVAL"
done
