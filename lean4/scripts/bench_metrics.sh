#!/usr/bin/env bash
# Emit a JSON snapshot of host IO/memory pressure metrics.
# Usage: ./scripts/bench_metrics.sh > metrics.json

set -euo pipefail

now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
host="$(hostname)"
uname_str="$(uname -srm)"
if [ -r /proc/loadavg ]; then
  loadavg="$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "unknown")"
else
  loadavg="unknown"
fi

if [ -r /proc/meminfo ]; then
  mem_total_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_avail_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  swap_total_kb="$(awk '/SwapTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  swap_free_kb="$(awk '/SwapFree/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
else
  mem_total_kb=0
  mem_avail_kb=0
  swap_total_kb=0
  swap_free_kb=0
fi

psi_cpu=""
psi_io=""
psi_mem=""
if [ -r /proc/pressure/cpu ]; then
  psi_cpu="$(tr '\n' ' ' < /proc/pressure/cpu 2>/dev/null || true)"
fi
if [ -r /proc/pressure/io ]; then
  psi_io="$(tr '\n' ' ' < /proc/pressure/io 2>/dev/null || true)"
fi
if [ -r /proc/pressure/memory ]; then
  psi_mem="$(tr '\n' ' ' < /proc/pressure/memory 2>/dev/null || true)"
fi

vmstat_json=""
if command -v vmstat >/dev/null 2>&1; then
  vm_line="$(vmstat 1 2 | tail -n 1)"
  if [ -n "$vm_line" ]; then
    read -r r b swpd free buff cache si so bi bo in cs us sy id wa st <<<"$vm_line"
    vmstat_json="\"vmstat\":{\"r\":${r:-0},\"b\":${b:-0},\"swpd\":${swpd:-0},\"free\":${free:-0},\"buff\":${buff:-0},\"cache\":${cache:-0},\"si\":${si:-0},\"so\":${so:-0},\"bi\":${bi:-0},\"bo\":${bo:-0},\"in\":${in:-0},\"cs\":${cs:-0},\"us\":${us:-0},\"sy\":${sy:-0},\"id\":${id:-0},\"wa\":${wa:-0},\"st\":${st:-0}"
  fi
fi

df_paths=("/")
if [ -d /dev/shm ]; then
  df_paths+=("/dev/shm")
fi
if [ -d /run ]; then
  df_paths+=("/run")
fi
df_json="$(df -kP "${df_paths[@]}" 2>/dev/null | awk 'NR>1 {printf "{\"mount\":\"%s\",\"size_kb\":%s,\"used_kb\":%s,\"avail_kb\":%s,\"use_pct\":\"%s\"}", $6,$2,$3,$4,$5}' | paste -sd, -)"

printf '{'
printf '"timestamp":"%s",' "$now"
printf '"hostname":"%s",' "$host"
printf '"uname":"%s",' "$uname_str"
printf '"loadavg":"%s",' "$loadavg"
printf '"mem_total_kb":%s,' "$mem_total_kb"
printf '"mem_available_kb":%s,' "$mem_avail_kb"
printf '"swap_total_kb":%s,' "$swap_total_kb"
printf '"swap_free_kb":%s,' "$swap_free_kb"
printf '"psi_cpu":"%s",' "${psi_cpu:-}"
printf '"psi_io":"%s",' "${psi_io:-}"
printf '"psi_mem":"%s",' "${psi_mem:-}"
printf '"filesystems":[%s]' "${df_json:-}"
if [ -n "$vmstat_json" ]; then
  printf ',%s' "$vmstat_json"
fi
printf '}\n'
