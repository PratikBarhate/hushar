#!/usr/bin/env bash
# Reports both lanes FROM S3 ALONE, once, and exits.
#
#   HUSHAR_BENCH_BUCKET=... watch.sh            # one report
#   HUSHAR_BENCH_BUCKET=... watch.sh --loop 1800 > /tmp/watch-ceiling.log 2>&1 &
#
# Nothing here is required for the run to finish: the lanes are detached and write each
# cell and their own log to S3 as they go, so this is a progress view and the laptop may
# be closed -- or off -- for hours between reports. It deliberately does NOT touch the
# hosts: a report that needs SSM is a report that cannot be run from a phone tether, and
# the bucket has everything.
#
# A ladder stops at its first unsustained rung, so the cell count is not a fixed
# denominator. What tells you a lane is finished is `ALL DONE` in its lane log.
set -uo pipefail

BUCKET="${HUSHAR_BENCH_BUCKET:?set HUSHAR_BENCH_BUCKET}"
RUN="${RUN:-design-vs-triton-cpu-arm64-10M}"
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
A=("$AWS" --profile "${AWS_PROFILE:-hushar-bench}" --region "${AWS_DEFAULT_REGION:-us-west-2}")

LOOP=0
[ "${1:-}" = "--loop" ] && LOOP="${2:-1800}"

report() {
  echo "================ $(date -u +%FT%TZ)  $RUN"
  "${A[@]}" s3 ls "s3://$BUCKET/runs/$RUN/cells/" 2>/dev/null \
    | awk '{print $4}' | sed -E 's/\.json$//' > /tmp/watch-ceiling-cells.txt
  echo "cells measured: $(wc -l < /tmp/watch-ceiling-cells.txt | tr -d ' ')"
  # c9g-1-N-r100t200 -> host, arm, the rest
  sed -E 's/^(c9g-[0-9])-([A-Z])-(.*)$/\1 \2 \3/' /tmp/watch-ceiling-cells.txt \
    | sort | awk '{ k=$1" arm "$2; n[k]++ } END { for (i in n) printf "    %s: %s\n", i, n[i] }' \
    | sort
  SKIPPED=$("${A[@]}" s3 ls "s3://$BUCKET/runs/$RUN/failures/" 2>/dev/null | wc -l | tr -d ' ')
  echo "failure/skip records: $SKIPPED"

  for SRV in c9g-1 c9g-2; do
    echo "--- lane $SRV"
    if "${A[@]}" s3 cp "s3://$BUCKET/runs/$RUN/raw/driver/lane-$SRV.log" \
         /tmp/watch-ceiling-$SRV.log --only-show-errors 2>/dev/null; then
      printf '    measured : %s\n' "$(grep -c ' done ' /tmp/watch-ceiling-$SRV.log)"
      printf '    skipped  : %s\n' "$(grep -c 'SKIPPED' /tmp/watch-ceiling-$SRV.log)"
      printf '    problems : %s\n' \
        "$(grep -cE 'FAILED|ABORT|EXCEPTION|WARNING' /tmp/watch-ceiling-$SRV.log)"
      grep -E 'ladder .* STOPS' /tmp/watch-ceiling-$SRV.log | sed 's/^/    /'
      grep -q 'ALL DONE' /tmp/watch-ceiling-$SRV.log \
        && echo "    LANE COMPLETE" \
        || tail -2 /tmp/watch-ceiling-$SRV.log | sed 's/^/    /'
    else
      echo "    no lane log in S3 yet (uploaded after the lane's first cell)"
    fi
  done
  echo
}

if [ "$LOOP" -gt 0 ]; then
  while true; do report; sleep "$LOOP"; done
else
  report
fi
