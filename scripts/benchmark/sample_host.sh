#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Samples the SERVER host's CPU and the server process's memory, for the report's
# CPU % and Mem % columns.
#
#   scripts/benchmark/sample_host.sh start           # sample once a second, in background
#   scripts/benchmark/sample_host.sh report          # mean over every sample
#   scripts/benchmark/sample_host.sh report 45 90    # mean over one rate's window
#   scripts/benchmark/sample_host.sh stop
#
# Start it before the client's sweep and stop it after: one sampler covers every rate,
# and `report <from> <to>` slices the window afterwards. Both bounds are SECONDS FROM
# THE FIRST SAMPLE, so a sweep of 45-second rates is 0 45, 45 90, 90 135 -- no clock
# arithmetic, and no need to have noted when each rate began.
#
# Reads /proc directly rather than shelling out to mpstat, so there is nothing to
# install and the memory figure is the server process rather than the host's page cache.
#
# Environment:
#   RUN_DIR    /tmp/hushar-server   where run_server.sh put server.pid
#   OUT        $RUN_DIR/samples.log
#   INTERVAL   1                    seconds between samples

set -uo pipefail

RUN_DIR="${RUN_DIR:-/tmp/hushar-server}"
OUT="${OUT:-$RUN_DIR/samples.log}"
INTERVAL="${INTERVAL:-1}"
PID_FILE="$RUN_DIR/sampler.pid"

usage() { awk 'NR>3 && /^#/ { sub(/^# ?/, ""); print; next } NR>3 { exit }' "$0"; }

# Before the /proc check, so the help is readable from a laptop.
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

[ -r /proc/stat ] || {
  echo "sample_host.sh reads /proc, so it runs on the Linux server host only" >&2
  exit 1
}

# Busy is total minus idle, which is what "100 - %idle" means in mpstat -- iowait counts
# as busy in both, so a number from either is comparable with a number from the other.
cpu_snapshot() {
  awk '/^cpu / { idle = $5; total = 0; for (i = 2; i <= NF; i++) total += $i;
                 print total, idle; exit }' /proc/stat
}

server_pid() {
  if [ -r "$PID_FILE.server" ]; then cat "$PID_FILE.server"; return; fi
  [ -r "$RUN_DIR/server.pid" ] && cat "$RUN_DIR/server.pid"
}

# Four numbers per line, so `report` is one awk pass and the raw file is still readable:
#   epoch  cpu_busy_%  process_RSS_%_of_host  process_RSS_kB  host_memory_used_%
sample_loop() {
  local prev_total prev_idle total idle pid rss mem_total mem_avail now
  read -r prev_total prev_idle < <(cpu_snapshot)
  while :; do
    sleep "$INTERVAL"
    read -r total idle < <(cpu_snapshot)
    now=$(date -u '+%s')
    pid="$(server_pid)"
    rss=0
    [ -n "$pid" ] && [ -r "/proc/$pid/status" ] \
      && rss=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status")
    mem_total=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
    mem_avail=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
    # The timestamp comes from the shell rather than awk's systime(), which not every
    # awk has -- the whole file is then readable by the plainest one.
    awk -v t="$total" -v i="$idle" -v pt="$prev_total" -v pi="$prev_idle" \
        -v rss="${rss:-0}" -v mt="$mem_total" -v ma="$mem_avail" -v now="$now" \
        'BEGIN {
           dt = t - pt; di = i - pi;
           cpu = (dt > 0 ? 100 * (dt - di) / dt : 0);
           mem = (mt > 0 ? 100 * rss / mt : 0);
           host = (mt > 0 ? 100 * (mt - ma) / mt : 0);
           printf "%d %.2f %.3f %d %.2f\n", now, cpu, mem, rss, host;
         }'
    prev_total=$total
    prev_idle=$idle
  done
}

start() {
  if [ -r "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already sampling, pid $(cat "$PID_FILE") -- stop it first" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$OUT")"
  # The server's pid is copied now rather than read every second: run_server.sh restarted
  # mid-sweep would otherwise silently move the measurement onto a different process.
  pid="$(server_pid)"
  [ -n "$pid" ] || echo "note: no server pid in $RUN_DIR -- memory will read 0" >&2
  [ -n "$pid" ] && echo "$pid" > "$PID_FILE.server"
  : > "$OUT"
  sample_loop >> "$OUT" 2>/dev/null &
  echo $! > "$PID_FILE"
  echo "sampling every ${INTERVAL}s into $OUT, pid $(cat "$PID_FILE")"
  echo "  server pid ${pid:-none}"
  echo "  stop it after the sweep, then: $0 report"
}

stop() {
  if [ -r "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null && echo "stopped pid $(cat "$PID_FILE")"
    rm -f "$PID_FILE" "$PID_FILE.server"
  else
    echo "no sampler pid file at $PID_FILE"
  fi
  [ -r "$OUT" ] && echo "$(wc -l < "$OUT") sample(s) in $OUT"
}

# `from`/`to` are seconds from the first sample, so a rate's window is arithmetic on the
# sweep's own shape rather than on wall-clock timestamps nobody wrote down.
report() {
  [ -r "$OUT" ] || { echo "no samples at $OUT -- was the sampler started?" >&2; exit 1; }
  local from="${1:-0}" to="${2:-}"
  awk -v from="$from" -v to="${to:--1}" '
    NR == 1 { base = $1 }
    {
      offset = $1 - base;
      if (offset < from) next;
      if (to >= 0 && offset > to) next;
      cpu += $2; mem += $3; rss += $4; host += $5; n++;
      if (n == 1) first = offset;
      last = offset;
    }
    END {
      if (n == 0) { print "no samples in that window"; exit 1 }
      printf "cpu    %.1f %%   mean host utilisation\n", cpu / n;
      printf "mem    %.1f %%   server process, %.1f GiB resident\n",
             mem / n, rss / n / 1048576;
      printf "host   %.1f %%   memory used including page cache\n", host / n;
      printf "over   %d sample(s), %ds to %ds of the sweep\n", n, first, last;
    }' "$OUT"
}

case "${1:-report}" in
  start) start ;;
  stop) stop ;;
  report) shift || true; report "${1:-0}" "${2:-}" ;;
  *) echo "usage: $0 [start|stop|report [from-secs] [to-secs]]" >&2; exit 2 ;;
esac
