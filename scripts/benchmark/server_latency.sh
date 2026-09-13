#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Server-side latency percentiles, for the report's srv p50 and srv p95 columns.
#
#   scripts/benchmark/server_latency.sh HusharBench bench-raw
#   scripts/benchmark/server_latency.sh HusharBench bench-raw bench-raw-candidate
#   MINUTES=60 scripts/benchmark/server_latency.sh HusharBench bench-raw
#   START=2026-09-10T18:00:00Z END=2026-09-10T18:45:00Z \
#     scripts/benchmark/server_latency.sh HusharBench bench-raw bench-raw-candidate
#
# The server publishes one data point per scored batch, in microseconds, under a ModelId
# dimension -- so CloudWatch holds the distribution and this asks it for the percentiles.
# It needs the server to have run with CW_NAMESPACE set; without it timings only reach
# the server log, which carries means and not percentiles.
#
# With two arms resident the arms are averaged, which is the number the table takes: the
# same weights under two model_ids is one compute measurement, not two.
#
# Environment:
#   METRIC     InferenceTime   also VectorizationTime, ToTensorTime -- both are the
#              SERVICE's featurisation, so they are near zero on a raw model whose graph
#              does its own, and worth reading on a vectorized one
#   MINUTES    30              window ending now, when START/END are not given
#   START/END  ISO 8601, e.g. 2026-09-10T18:00:00Z. Prefer these: they cut the window to
#              the measured sweep rather than including the warmup and the idle tail
#   REGION     the CLI's default

set -uo pipefail

usage() { awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"; }

case "${1:-}" in
  -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

NAMESPACE="$1"; shift
[ $# -gt 0 ] || { echo "give at least one model_id" >&2; exit 2; }

METRIC="${METRIC:-InferenceTime}"
MINUTES="${MINUTES:-30}"
START="${START:-$(date -u -d "-${MINUTES} minutes" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
  || date -u -v-"${MINUTES}"M '+%Y-%m-%dT%H:%M:%SZ')}"
END="${END:-$(date -u '+%Y-%m-%dT%H:%M:%SZ')}"
REGION_ARGS=()
[ -n "${REGION:-}" ] && REGION_ARGS=(--region "$REGION")

command -v aws >/dev/null || { echo "the AWS CLI is required" >&2; exit 1; }

# One period covering the whole window, so each arm comes back as a single data point
# rather than a series to average by hand. Seconds, and a multiple of 60. python3 rather
# than date arithmetic, which differs between the laptop and the instance.
period=$(python3 -c '
import sys
from datetime import datetime
fmt = "%Y-%m-%dT%H:%M:%SZ"
start, end = (datetime.strptime(t, fmt) for t in sys.argv[1:3])
seconds = max(int((end - start).total_seconds()), 60)
print((seconds // 60 + 1) * 60)
' "$START" "$END") || {
  echo "START and END must look like 2026-09-10T18:00:00Z" >&2
  exit 2
}

printf '%s %s  %s .. %s\n' "$METRIC" "$NAMESPACE" "$START" "$END"

sum50=0 sum95=0 arms=0
for model in "$@"; do
  json=$(aws cloudwatch get-metric-statistics \
    ${REGION_ARGS[@]+"${REGION_ARGS[@]}"} \
    --namespace "$NAMESPACE" \
    --metric-name "$METRIC" \
    --dimensions "Name=ModelId,Value=$model" \
    --start-time "$START" --end-time "$END" \
    --period "$period" \
    --extended-statistics p50 p95 \
    --output json 2>&1) || {
    echo "  $model: the CloudWatch call failed" >&2
    echo "$json" | sed 's/^/    /' >&2
    exit 1
  }
  # Microseconds on the wire, milliseconds in the report -- the conversion is here rather
  # than in the reader's head, which is where it has been got wrong before.
  read -r p50 p95 <<EOF
$(printf '%s' "$json" | python3 -c '
import json, sys
points = json.load(sys.stdin)["Datapoints"]
if not points:
    print("- -")
else:
    # Several periods can come back if the window straddles one; the mean of their
    # percentiles is the best available answer and the note below says so.
    p50 = sum(p["ExtendedStatistics"]["p50"] for p in points) / len(points) / 1000
    p95 = sum(p["ExtendedStatistics"]["p95"] for p in points) / len(points) / 1000
    print(f"{p50:.2f} {p95:.2f}")
')
EOF
  if [ "$p50" = "-" ]; then
    printf '  %-24s no data points -- check the namespace, the model_id and the window\n' \
      "$model"
    continue
  fi
  printf '  %-24s p50 %8s ms   p95 %8s ms\n' "$model" "$p50" "$p95"
  sum50=$(awk -v a="$sum50" -v b="$p50" 'BEGIN { printf "%.4f", a + b }')
  sum95=$(awk -v a="$sum95" -v b="$p95" 'BEGIN { printf "%.4f", a + b }')
  arms=$((arms + 1))
done

if [ "$arms" -gt 1 ]; then
  awk -v s50="$sum50" -v s95="$sum95" -v n="$arms" 'BEGIN {
    printf "  %-24s p50 %8.2f ms   p95 %8.2f ms   <- the table takes these\n",
           "mean of " n " arm(s)", s50 / n, s95 / n }'
elif [ "$arms" = 1 ]; then
  echo "  one arm, so its row is what the table takes"
fi
