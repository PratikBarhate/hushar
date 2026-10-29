#!/usr/bin/env bash
# Samples the GPU once a second into OUT, alongside sample_host.sh's CPU/memory samples.
#
#   OUT=/tmp/hushar-server/g-<cell>.log sample_gpu.sh start|stop
#
# Line format:  epoch gpu_util% mem_util% mem_used_MiB power_W sm_MHz temp_C
#
# Two things to know before reading the numbers:
#
#   * gpu_util is nvidia-smi's utilization.gpu: the share of the last sample period in
#     which at least one kernel was resident. It says the device was BUSY, not how full
#     it was -- a stream of tiny kernels reads 100% on an idle-ish GPU. Treat it as
#     occupancy over time and nothing more.
#   * nvidia-smi reports `[N/A]` for a field a given driver or instance does not expose,
#     and `[N/A]` in the middle of a line used to make the collector discard the WHOLE
#     sample, so a missing power reading silently cost the utilization series too. Those
#     fields are normalised to `-` here and the collector averages each column over the
#     samples that have it.
set -uo pipefail
OUT="${OUT:?OUT required}"
PID_FILE="$OUT.pid"

case "${1:?start|stop}" in
start)
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already sampling, pid $(cat "$PID_FILE")"; exit 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi on this host" >&2; exit 1; }
  : > "$OUT"
  nohup bash -c '
    while :; do
      line=$(nvidia-smi \
        --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1 \
        | tr -d "," | sed -e "s/\[N\/A\]/-/g" -e "s/N\/A/-/g")
      [ -n "$line" ] && echo "$(date +%s) $line" >> "'"$OUT"'"
      sleep 1
    done' >/dev/null 2>&1 &
  echo $! > "$PID_FILE"
  echo "sampling gpu into $OUT, pid $!"
  ;;
stop)
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  echo "$(wc -l < "$OUT" 2>/dev/null || echo 0) gpu sample(s) in $OUT"
  ;;
*) echo "start|stop" >&2; exit 2 ;;
esac
