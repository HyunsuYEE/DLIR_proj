#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

extract_metrics_summary() {
  local mode="$1"
  local log_path="out_${mode}.log"

  echo
  echo "===== ${mode} ====="
  if [[ ! -f "$log_path" ]]; then
    echo "Missing log file: $log_path"
    return 1
  fi

  awk '
    /^Metrics summary$/ {capture=1; block=$0 ORS; next}
    capture && /^  / {block=block $0 ORS; next}
    capture {capture=0}
    END {
      if (block != "") {
        printf "%s", block
      } else {
        exit 1
      }
    }
  ' "$log_path" || {
    echo "Metrics summary not found in $log_path"
    return 1
  }
}

echo "Running base benchmark..."
./run.sh --base "$@"

echo
echo "Running conservative benchmark..."
./run.sh --conservative "$@"

echo
echo "Running aggressive benchmark..."
./run.sh --aggressive "$@"

echo
echo "Running PRG benchmark..."
./run.sh --prg "$@"

echo
echo "Benchmark summaries"
extract_metrics_summary "base"
extract_metrics_summary "conservative"
extract_metrics_summary "aggressive"
extract_metrics_summary "prg"
