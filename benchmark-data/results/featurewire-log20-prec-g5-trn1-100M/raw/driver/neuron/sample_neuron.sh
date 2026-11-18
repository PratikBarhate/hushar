#!/usr/bin/env bash
# Samples the NeuronCores once a second into OUT, alongside sample_host.sh's CPU/memory.
#
#   OUT=/tmp/hushar-server/n-<cell>.log sample_neuron.sh start|stop|report
#
# Line format:  epoch nc0_util% nc1_util% nc_mean% device_mem_MiB host_mem_MiB runtimes
#
# This is the Trainium counterpart of the GPU run's sample_gpu.sh, and it inherits that
# file's two hard-won rules.
#
#   * `neuroncore_utilization` IS OCCUPANCY, NOT SATURATION. It is the share of the
#     sample period in which the core was executing something -- exactly like
#     nvidia-smi's `utilization.gpu`. A core running a stream of small NEFF calls reads
#     high while still having arithmetic headroom, and a split cell produces precisely
#     such a stream. So this column says the core was BUSY and never how full it was,
#     and a report comparing two mini-batch sizes must not read a higher number here as
#     more work done.
#   * A MISSING FIELD MUST NOT DISCARD THE WHOLE SAMPLE. The GPU sampler lost its
#     utilization series to an `[N/A]` power reading before this was fixed there. Every
#     field here is independently optional and absent ones are written `-`, so the
#     collector averages each column over the samples that carry it.
#
# One Neuron-specific trap, which is why `runtimes` is a column rather than an assumption:
# neuron-monitor reports per-core utilization only for cores held by a LIVE Neuron
# runtime. Before hushar starts, and after it stops, `neuron_runtime_data` is empty and
# every utilization field is legitimately absent. A window whose `runtimes` column is 0
# did not measure an idle device -- it measured no server, which is a different fact and
# must not be averaged in as 0% busy.
set -uo pipefail
OUT="${OUT:?OUT required}"
PID_FILE="$OUT.pid"
CONF="$OUT.monitor.json"
PERIOD="${PERIOD:-1s}"

NEURON_MONITOR="${NEURON_MONITOR:-}"
if [ -z "$NEURON_MONITOR" ]; then
  NEURON_MONITOR="$(command -v neuron-monitor 2>/dev/null || true)"
fi
[ -n "$NEURON_MONITOR" ] || NEURON_MONITOR=/opt/aws/neuron/bin/neuron-monitor

case "${1:?start|stop|report}" in
start)
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already sampling, pid $(cat "$PID_FILE")"; exit 0
  fi
  [ -x "$NEURON_MONITOR" ] \
    || { echo "no neuron-monitor at $NEURON_MONITOR" >&2; exit 1; }
  cat > "$CONF" <<EOF
{
  "period": "$PERIOD",
  "neuron_runtimes": [
    {
      "tag_filter": ".*",
      "metrics": [
        {"type": "neuroncore_counters"},
        {"type": "memory_used"}
      ]
    }
  ],
  "system_metrics": [
    {"type": "memory_info"}
  ]
}
EOF
  : > "$OUT"
  # neuron-monitor writes one JSON object per period on stdout. It is flattened here
  # rather than stored raw because a 15-minute cell is ~900 objects of several KB each,
  # and the per-second series a report actually reads is six numbers wide.
  nohup bash -c '
    "'"$NEURON_MONITOR"'" -c "'"$CONF"'" 2>/dev/null | python3 -u -c "
import json, sys, time

def fmt(value):
    return \"-\" if value is None else (\"%.1f\" % value)

for line in sys.stdin:
    line = line.strip()
    if not line.startswith(\"{\"):
        continue
    try:
        doc = json.loads(line)
    except ValueError:
        continue
    runtimes = doc.get(\"neuron_runtime_data\") or []
    # Per core, taking the MAXIMUM across runtimes rather than a sum: two runtime
    # processes reporting the same physical core each report that core, and adding them
    # would read 200% on a core that was merely busy.
    per_core, device_bytes = {}, 0.0
    live = 0
    for entry in runtimes:
        report = entry.get(\"report\") or {}
        counters = (report.get(\"neuroncore_counters\") or {}).get(\"neuroncores_in_use\")
        if counters:
            live += 1
            for core, values in counters.items():
                used = values.get(\"neuroncore_utilization\")
                if used is None:
                    continue
                try:
                    core_index = int(core)
                except (TypeError, ValueError):
                    continue
                per_core[core_index] = max(per_core.get(core_index, 0.0), float(used))
        used_bytes = (report.get(\"memory_used\") or {}).get(\"neuron_runtime_used_bytes\")
        if isinstance(used_bytes, dict):
            try:
                device_bytes += float(used_bytes.get(\"neuron_device\") or 0)
            except (TypeError, ValueError):
                pass
    system = doc.get(\"system_data\") or {}
    host_mb = None
    memory = system.get(\"memory_info\") or {}
    try:
        host_mb = float(memory[\"memory_used_bytes\"]) / 1048576.0
    except (KeyError, TypeError, ValueError):
        host_mb = None
    values = [per_core.get(0), per_core.get(1)]
    present = [v for v in values if v is not None]
    mean = sum(present) / len(present) if present else None
    device_mb = device_bytes / 1048576.0 if device_bytes else None
    print(\"%d %s %s %s %s %s %d\" % (
        int(time.time()), fmt(values[0]), fmt(values[1]), fmt(mean),
        fmt(device_mb), fmt(host_mb), live))
" >> "'"$OUT"'"' >/dev/null 2>&1 &
  echo $! > "$PID_FILE"
  echo "sampling neuron into $OUT, pid $!"
  ;;
stop)
  if [ -f "$PID_FILE" ]; then
    # The pid is the subshell; neuron-monitor and the python filter are its children and
    # have to be taken with it or a stopped cell leaves a sampler attached to the device.
    pkill -P "$(cat "$PID_FILE")" 2>/dev/null || true
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  pkill -f "[n]euron-monitor -c $CONF" 2>/dev/null || true
  rm -f "$CONF"
  "$0" report
  ;;
report)
  # Column means over whatever the window holds, each over the samples that carry it.
  python3 - "$OUT" <<'PY'
import sys

path = sys.argv[1]
names = ["nc0", "nc1", "nc_mean", "device_mb", "host_mb"]
totals = [0.0] * len(names)
counts = [0] * len(names)
rows = live = 0
try:
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 7:
                continue
            rows += 1
            live += 1 if parts[6] != "0" else 0
            for index, field in enumerate(parts[1:6]):
                if field == "-":
                    continue
                try:
                    totals[index] += float(field)
                except ValueError:
                    continue
                counts[index] += 1
except OSError:
    pass
if not rows:
    print("0 neuron sample(s)")
    sys.exit(0)
out = [f"{rows} neuron sample(s), {live} with a live runtime"]
for name, total, count in zip(names, totals, counts):
    out.append(f"{name} {total / count:.1f}" if count else f"{name} -")
print(" | ".join(out))
PY
  ;;
*) echo "start|stop|report" >&2; exit 2 ;;
esac
