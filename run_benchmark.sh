#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

default_prg_risk_threshold="0.5"

log_path_for_mode() {
  local mode="$1"
  shift || true

  if [[ "$mode" != "prg" ]]; then
    echo "out_${mode}.log"
    return 0
  fi

  local prg_risk_threshold="$default_prg_risk_threshold"
  for override in "$@"; do
    case "$override" in
      world_model_env.diffusion_sampler.prg_risk_threshold=*)
        prg_risk_threshold="${override#*=}"
        ;;
    esac
  done
  local prg_threshold_suffix="${prg_risk_threshold//[^[:alnum:]._-]/_}"
  echo "out_${mode}_threshold_${prg_threshold_suffix}.log"
}

extract_metrics_summary() {
  local mode="$1"
  shift || true
  local log_path
  log_path="$(log_path_for_mode "$mode" "$@")"

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
extract_metrics_summary "base" "$@"
extract_metrics_summary "conservative" "$@"
extract_metrics_summary "aggressive" "$@"
extract_metrics_summary "prg" "$@"
