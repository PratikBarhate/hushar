#!/usr/bin/env bash
# Keeps the laptop's `hushar-bench` credentials usable while a long launch or deploy runs.
#
#   nohup bash refresh_creds.sh > /tmp/<run>-creds.log 2>&1 &
#
# WHY THIS EXISTS. `rig.py launch` can wait hours for `g5.2xlarge` capacity, and these are
# temporary credentials of a few hours. `rig.py` already RETRIES on an expired token
# instead of dying -- but it retries waiting for a refresh to arrive from somewhere, and
# nothing else here provides one. So this is that somewhere.
#
# Only the LAPTOP needs it. Once the sweep is detached on the client hosts they run under
# instance profiles, which refresh themselves, so this can be killed as soon as the deploy
# and the launch command have returned.
set -uo pipefail

ACCOUNT="${ACCOUNT:?set ACCOUNT}"
PROFILE="${PROFILE:-hushar-bench}"
ROLE="${ROLE:-IibsAdminAccess-DO-NOT-DELETE}"
EVERY="${EVERY:-1500}"          # 25 min, comfortably inside a session's life
HOURS="${HOURS:-14}"
AWS="${AWS_CLI:-/opt/homebrew/bin/aws}"

deadline=$(( $(date +%s) + HOURS * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ada credentials update --once --account="$ACCOUNT" --provider=conduit \
       --role="$ROLE" --profile="$PROFILE" >/dev/null 2>&1; then
    who=$("$AWS" sts get-caller-identity --profile "$PROFILE" --query Arn --output text \
            2>/dev/null | sed 's/[0-9]\{12\}/ACCT/g')
    echo "$(date -u +%FT%TZ) refreshed $PROFILE -- ${who:-unverified}"
  else
    # A failure here is usually an expired Midway cookie, which only a human can fix.
    # Said plainly rather than retried silently, because the symptom downstream is an
    # ExpiredToken that reads like an AWS fault.
    echo "$(date -u +%FT%TZ) REFRESH FAILED -- 'mwinit' may be needed on this laptop"
  fi
  sleep "$EVERY"
done
echo "$(date -u +%FT%TZ) refresher done after ${HOURS}h"
