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
# CPU is the scheduler's own account of time spent running tasks -- the per-CPU
# `rq_cpu_time` counters in /proc/schedstat, summed over the host and divided by wall
# time and CPU count -- not the tick-based user/system/idle columns of /proc/stat. The
# tick columns depend on the timer tick firing on every busy CPU, and on the kernels these
# benchmarks ran on (6.18, NO_HZ) whole five-minute windows were seen with busy CPUs
# accounted as idle: the sampler read 0.05 % while the scheduler saw 30 % and the
# hypervisor agreed with the scheduler. `rq_cpu_time` is measured from the scheduler
# clock at every context switch, so it needs no tick. The tick figure is still written,
# as a sixth column, so the two can be compared for any window afterwards; on a kernel
# without /proc/schedstat (CONFIG_SCHEDSTATS off) the sampler falls back to the tick
# figure and says so.
#
# One property of the scheduler counter to know about: a task's run time is added when it
# leaves its CPU, so a thread that runs for several seconds without a context switch lands
# in one sample rather than being spread over them. Single samples can therefore exceed
# 100 %; the mean over a window is exact. Read the window mean, not one line.
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

# The scheduler's run-time counters exist only with CONFIG_SCHEDSTATS; nearly every
# distribution kernel has it. Without them the tick figure is all there is.
if [ -r /proc/schedstat ] && grep -q '^cpu0 ' /proc/schedstat; then
  SCHEDSTAT=1
else
  SCHEDSTAT=0
fi

# The tick-based figure: busy is total minus idle, which is what "100 - %idle" means in
# mpstat -- iowait counts as busy in both, so a number from either is comparable with a
# number from the other.
tick_snapshot() {
  awk '/^cpu / { idle = $5; total = 0; for (i = 2; i <= NF; i++) total += $i;
                 print total, idle; exit }' /proc/stat
}

# The scheduler's figure: on each `cpuN` line of /proc/schedstat the eighth field is
# `rq_cpu_time`, the nanoseconds tasks other than the idle task have spent running on
# that CPU (the same field in every version of the file since 15). Summed over the host,
# with the number of CPUs it was summed over.
run_snapshot() {
  awk '$1 ~ /^cpu[0-9]+$/ { run += $8; n++ } END { printf "%.0f %d\n", run, n }' \
    /proc/schedstat
}

server_pid() {
  if [ -r "$PID_FILE.server" ]; then cat "$PID_FILE.server"; return; fi
  [ -r "$RUN_DIR/server.pid" ] && cat "$RUN_DIR/server.pid"
}

# Six numbers per line, so `report` is one awk pass and the raw file is still readable:
#   epoch  cpu_busy_%  process_RSS_%_of_host  process_RSS_kB  host_memory_used_%  tick_busy_%
# cpu_busy_% is the scheduler figure; tick_busy_% is /proc/stat's for the same second, or
# `-` when the scheduler figure was unavailable and the second column is the tick figure.
sample_loop() {
  local prev_total prev_idle total idle prev_run run n prev_ns now_ns now dw
  local pid rss mem_total mem_avail
  read -r prev_total prev_idle < <(tick_snapshot)
  run=0; n=0
  [ "$SCHEDSTAT" = 1 ] && read -r prev_run _ < <(run_snapshot)
  prev_ns=$(date +%s%N)
  while :; do
    sleep "$INTERVAL"
    read -r total idle < <(tick_snapshot)
    [ "$SCHEDSTAT" = 1 ] && read -r run n < <(run_snapshot)
    now_ns=$(date +%s%N)
    now=$((now_ns / 1000000000))
    # The wall-clock delta is taken in the shell, whose integers are 64-bit; awk's doubles
    # would round a nanosecond epoch.
    dw=$((now_ns - prev_ns))
    pid="$(server_pid)"
    rss=0
    [ -n "$pid" ] && [ -r "/proc/$pid/status" ] \
      && rss=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status")
    mem_total=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
    mem_avail=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
    # Wall time comes from the shell's clock, measured between the two reads rather than
    # assumed to be INTERVAL, so a late wake-up does not inflate the figure.
    awk -v t="$total" -v i="$idle" -v pt="$prev_total" -v pi="$prev_idle" \
        -v run="$run" -v prun="${prev_run:-0}" -v n="$n" -v dw="$dw" -v sched="$SCHEDSTAT" \
        -v rss="${rss:-0}" -v mt="$mem_total" -v ma="$mem_avail" -v now="$now" \
        'BEGIN {
           dt = t - pt; di = i - pi;
           tick = (dt > 0 ? 100 * (dt - di) / dt : 0);
           if (sched) {
             cpu = (dw > 0 && n > 0 ? 100 * (run - prun) / (dw * n) : 0);
             tickcol = sprintf("%.2f", tick);
           } else {
             cpu = tick; tickcol = "-";
           }
           mem = (mt > 0 ? 100 * rss / mt : 0);
           host = (mt > 0 ? 100 * (mt - ma) / mt : 0);
           printf "%d %.2f %.3f %d %.2f %s\n", now, cpu, mem, rss, host, tickcol;
         }'
    prev_total=$total
    prev_idle=$idle
    prev_run=$run
    prev_ns=$now_ns
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
  if [ "$SCHEDSTAT" = 1 ]; then
    echo "  cpu from the scheduler's run time (/proc/schedstat); the tick figure is column 6"
  else
    echo "  note: no /proc/schedstat on this kernel -- cpu is the tick-based /proc/stat figure" >&2
  fi
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
#
# The first line is the CPU figure the drivers read. `tick` is printed only when the file
# carries the sixth column, so a log from the tick-only sampler reports as it always did.
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
      if (NF >= 6 && $6 != "-") { tick += $6; tn++ }
      if (n == 1) first = offset;
      last = offset;
    }
    END {
      if (n == 0) { print "no samples in that window"; exit 1 }
      printf "cpu    %.1f %%   mean host utilisation%s\n", cpu / n,
             (tn > 0 ? ", scheduler run time" : "");
      if (tn > 0)
        printf "tick   %.1f %%   the same window from /proc/stat, tick-based, for comparison\n",
               tick / tn;
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
