#!/usr/bin/env bash
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Prints the environment block that makes a benchmark number comparable.
#
#   scripts/benchmark/hostinfo.sh                       # this host
#   ROLE=server PROVIDER=cpu scripts/benchmark/hostinfo.sh
#
# Run it on BOTH hosts and paste both blocks into the report's Environment section. A
# rate with no instance type, core count or thread setting beside it cannot be compared
# against anything, which is the only reason this script exists.

set -uo pipefail

ROLE="${ROLE:-}"
PROVIDER="${PROVIDER:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

field() { printf '%-20s %s\n' "$1" "${2:-unknown}"; }

# EC2 instance metadata, IMDSv2. Short timeouts so this is harmless off EC2.
imds() {
  local token
  token=$(curl -fsS -m 1 -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null) || return 1
  curl -fsS -m 1 -H "X-aws-ec2-metadata-token: $token" \
    "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null
}

lscpu_field() {
  command -v lscpu >/dev/null 2>&1 || return 1
  lscpu 2>/dev/null | awk -F: -v key="$1" '$1 == key { gsub(/^ +/, "", $2); print $2; exit }'
}

echo "=== host ==================================================="
[ -n "$ROLE" ] && field "role" "$ROLE"
field "hostname" "$(hostname 2>/dev/null)"
field "instance type" "$(imds instance-type)"
field "availability zone" "$(imds placement/availability-zone)"
field "private ipv4" "$(imds local-ipv4)"
field "kernel" "$(uname -sr)"

if [ -r /etc/os-release ]; then
  field "os" "$(. /etc/os-release && echo "$PRETTY_NAME")"
else
  field "os" "$(uname -s) $(uname -m)"
fi

echo
echo "=== cpu ===================================================="
cpu_model="$(lscpu_field 'Model name')"
if [ -z "$cpu_model" ] && [ -r /proc/cpuinfo ]; then
  cpu_model=$(awk -F: '/model name|Model/ { gsub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo)
fi
[ -z "$cpu_model" ] && cpu_model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
field "model" "$cpu_model"
field "architecture" "$(uname -m)"

sockets="$(lscpu_field 'Socket(s)')"
per_socket="$(lscpu_field 'Core(s) per socket')"
if [ -n "$sockets" ] && [ -n "$per_socket" ]; then
  field "physical cores" "$((sockets * per_socket))"
  field "threads per core" "$(lscpu_field 'Thread(s) per core')"
fi
if command -v nproc >/dev/null 2>&1; then
  field "vcpus (nproc)" "$(nproc)"
else
  field "vcpus" "$(sysctl -n hw.ncpu 2>/dev/null)"
fi
[ -n "$(lscpu_field 'Flags')" ] && field "simd" \
  "$(lscpu_field 'Flags' | tr ' ' '\n' | grep -E '^(avx512f|avx2|amx_tile|sve2?|sme2?|asimd|neon)$' | paste -sd, -)"

# NUMA decides what "compute_cores" should be on a large instance: a thread pool spread
# across nodes reaches its weights over the interconnect. Every node's CPU list is printed
# so a report can say which node a run was pinned to, and so the list can be copied
# straight into threading.compute_cores.
numa_nodes="$(lscpu_field 'NUMA node(s)')"
if [ -n "$numa_nodes" ]; then
  field "numa nodes" "$numa_nodes"
  lscpu 2>/dev/null | awk -F: '/NUMA node[0-9]+ CPU/ {
    gsub(/^ +/, "", $2); split($1, part, " "); printf "  %-20s %s\n", part[2] " cpus", $2 }'
fi

if [ -r /proc/meminfo ]; then
  field "memory" "$(awk '/MemTotal/ { printf "%.0f GiB", $2/1048576 }' /proc/meminfo)"
elif command -v sysctl >/dev/null 2>&1; then
  field "memory" "$(sysctl -n hw.memsize 2>/dev/null | awk '{ printf "%.0f GiB", $1/1073741824 }')"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "=== nvidia gpu ============================================="
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'
fi

echo
echo "=== software ==============================================="
field "provider" "$PROVIDER"
field "ORT_DYLIB_PATH" "${ORT_DYLIB_PATH:-}"
# The release archives carry their version in the directory name, which is the only
# version string available without loading the library.
if [ -n "${ORT_DYLIB_PATH:-}" ]; then
  field "ort version" "$(echo "$ORT_DYLIB_PATH" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1)"
fi
field "rustc" "$(rustc --version 2>/dev/null)"
if [ -r "$ROOT/BUNDLE_INFO" ]; then
  field "bundle" "$(awk -F': *' '/^commit/ { print $2 }' "$ROOT/BUNDLE_INFO")"
elif git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  field "commit" "$(git -C "$ROOT" describe --always --dirty 2>/dev/null)"
fi
