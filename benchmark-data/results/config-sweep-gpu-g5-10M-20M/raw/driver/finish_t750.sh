#!/usr/bin/env bash
# Waits for a g5.2xlarge, measures the one missing rung, reports, and destroys the rig.
#
#   nohup ./finish_t750.sh > /tmp/t750.log 2>&1 &
#
# WHY THIS IS A SCRIPT AND NOT A CHECKLIST. `g5.2xlarge` capacity has been intermittent
# for hours at a time, so the wait is unattended -- and an unattended wait that merely
# ACQUIRES a host leaves it idling at about $2.80/h until a human notices. This chains
# acquisition straight into the measurement and into `terminate`, so the rig exists for
# the ~35 minutes it is used and not a minute longer. The trap is the point: every exit
# path, including a failure or a Ctrl-C, destroys the rig.
#
# WHAT IT MEASURES. 20M, 100 rows x 750 req/s, BOTH arms:
#
#   hushar  210 named features, sessions 4   -- the ceiling row, with vectorization
#   triton  690-float tensor, instances 4    -- the half that has never been run
#
# Both on ONE host, which is the reason hushar is re-measured rather than reused: the
# cell currently in the report was measured by the wire run on a different instance, and
# a row whose two arms sit on two machines is the one thing every other row here avoids.
# The imported cell is deleted only AFTER a host is in hand, so a wait that never
# succeeds leaves the report exactly as it is.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export HUSHAR_BENCH_BUCKET="hushar-bench-$(cat ../../../../.account-ids)-us-west-2"
export AWS_PAGER=""
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"
PY="${PY:-python}"

REV="${REV:-ffba569-dirty}"
MD5_20M="${MD5_20M:-30f9c5330b808a34b55fe88048a20da6}"
RUN="config-sweep-gpu-g5-10M-20M"
CELL="g5-hushar-20m-r100t750-p4"
LOADS="20m=100x750"
PIN="20m:4"
# A cell is 120 s warm-up + 300 s measured + 60 s settle plus start-up and upload, so two
# cells need about twenty minutes. The cap is generous and bounded: a hung lane must not
# hold a GPU indefinitely.
RUN_TIMEOUT="${RUN_TIMEOUT:-4800}"

say() { printf '\n== %s  %s\n' "$1" "$(date -u +%FT%TZ)"; }

cleanup() {
  local rc=$?
  say "terminating the rig (exit $rc)"
  CONFIRM=terminate $PY rig.py terminate 2>&1 | tail -6 || true
  # Proven, not assumed: a GPU left running is the expensive failure here.
  local left
  left=$($AWS ec2 describe-instances --profile hushar-bench --region us-west-2 \
    --filters 'Name=instance-state-name,Values=pending,running,stopping,stopped' \
    --query 'length(Reservations)' --output text 2>/dev/null || echo "?")
  echo "   reservations still alive: $left"
  exit $rc
}
trap cleanup EXIT INT TERM

say "waiting for g5.2xlarge capacity"
rm -f rig-state.json
# rig.py retries both capacity and an expired credential session, and adopts anything it
# already created, so this is safe to re-run.
CAPACITY_TRIES=720 CAPACITY_WAIT=120 $PY rig.py launch || exit 1

say "ssh keys"
$PY keys.py || exit 1

say "deploy (20M only)"
$PY fanout.py sweep -f deploy.sh --timeout 5400 --tail 8 \
  --env REV="$REV" --env MODEL_SPECS="20m:$MD5_20M" \
  --env HUSHAR_BENCH_BUCKET="$HUSHAR_BENCH_BUCKET" || exit 1

# Only now that a host is secured is the borrowed cell given up, so that both arms of the
# row are measured on this machine.
say "dropping the imported hushar cell so it is re-measured here"
$AWS s3 rm "s3://$HUSHAR_BENCH_BUCKET/runs/$RUN/cells/$CELL.json" \
  --profile hushar-bench --region us-west-2 || true
rm -f "../../../results/$RUN/raw/cells/$CELL.json"

say "measuring 100 x 750, both arms"
R=$($PY -c "import hosts; print(hosts.rig_env())")
SERVER=$($PY -c "import hosts; print(hosts.GROUPS['servers'][0])")
CLIENT=$($PY -c "import hosts; print(hosts.GROUPS['clients'][0])")
echo "   rig $R, $CLIENT -> $SERVER"
$PY fanout.py "$CLIENT" -f launch.sh --timeout 300 --tail 8 \
  --env REV="$REV" --env MODEL_SPECS="20m:$MD5_20M" \
  --env RIG="$R" --env SERVER_FOR="$SERVER" --env PHASE=main \
  --env ARMS="hushar,triton" --env PIN_PARALLELISM="$PIN" --env LOADS="$LOADS" \
  --env LANE_TAG=t750 \
  --env HUSHAR_BENCH_BUCKET="$HUSHAR_BENCH_BUCKET" || exit 1

say "waiting for the lane"
deadline=$(( $(date +%s) + RUN_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 60
  if ! $PY fanout.py "$CLIENT" --timeout 180 -c 'pgrep -cf "[s]weep.py" || true' 2>/dev/null \
      | grep -qE '^[1-9]'; then
    echo "   lane finished"
    break
  fi
  echo "   still running, $(( (deadline - $(date +%s)) / 60 )) min of budget left"
done
$PY fanout.py "$CLIENT" --timeout 300 --tail 12 \
  -c 'grep -E "done |ALL DONE|FAILED|ABORT" /root/orchestrator-t750-*.log | tail -8' || true

say "collect, report, verify"
$PY collect.py 2>&1 | tail -4
$PY report.py 2>&1 | tail -2
$PY verify.py 2>&1 | tail -4

say "done -- the trap now destroys the rig"
