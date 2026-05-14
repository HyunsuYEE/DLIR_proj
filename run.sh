#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --base [extra hydra overrides...]
  ./run.sh --conservative [extra hydra overrides...]
  ./run.sh --aggressive [extra hydra overrides...]
  ./run.sh --prg [extra hydra overrides...]

Modes:
  --base          Current default DIAMOND path.
  --conservative DPM-Solver only.
  --aggressive   DPM-Solver + TeaCache.
  --prg          PRG-gated conservative/aggressive selection.

Examples:
  ./run.sh --base
  ./run.sh --base actor_critic.training.steps_first_epoch=200
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

mode="$1"
shift

default_prg_risk_threshold="0.5"

case "$mode" in
  --base)
    mode_name="base"
    mode_overrides=()
    ;;
  --conservative)
    mode_name="conservative"
    mode_overrides=(
      "world_model_env.diffusion_sampler.solver_type=dpm_solver"
      "world_model_env.diffusion_sampler.num_steps_denoising=2"
      "world_model_env.diffusion_sampler.dpm_solver_order=2"
      "world_model_env.diffusion_sampler.dpm_solver_method=multistep"
    )
    ;;
  --aggressive)
    mode_name="aggressive"
    mode_overrides=(
      "world_model_env.diffusion_sampler.solver_type=dpm_solver"
      "world_model_env.diffusion_sampler.num_steps_denoising=2"
      "world_model_env.diffusion_sampler.dpm_solver_order=2"
      "world_model_env.diffusion_sampler.dpm_solver_method=multistep"
      "world_model_env.diffusion_sampler.teacache_enabled=true"
      "world_model_env.diffusion_sampler.teacache_rel_l1_thresh=10.0"
      "world_model_env.diffusion_sampler.teacache_force_last=false"
    )
    ;;
  --prg)
    mode_name="prg"
    mode_overrides=(
      "world_model_env.diffusion_sampler.solver_type=prg"
      "world_model_env.diffusion_sampler.num_steps_denoising=2"
      "world_model_env.diffusion_sampler.dpm_solver_order=2"
      "world_model_env.diffusion_sampler.dpm_solver_method=multistep"
      "world_model_env.diffusion_sampler.teacache_rel_l1_thresh=10.0"
      "world_model_env.diffusion_sampler.teacache_force_last=false"
      "world_model_env.diffusion_sampler.prg_risk_threshold=${default_prg_risk_threshold}"
      "world_model_env.diffusion_sampler.prg_depth_weight=1.0"
      "world_model_env.diffusion_sampler.prg_policy_weight=1.0"
      "world_model_env.diffusion_sampler.prg_proxy_weight=1.0"
    )
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Error: unknown mode '$mode'." >&2
    usage >&2
    exit 2
    ;;
esac

log_path="out_${mode_name}.log"
if [[ "$mode_name" == "prg" ]]; then
  prg_risk_threshold="$default_prg_risk_threshold"
  for override in "$@"; do
    case "$override" in
      world_model_env.diffusion_sampler.prg_risk_threshold=*)
        prg_risk_threshold="${override#*=}"
        ;;
    esac
  done
  prg_threshold_suffix="${prg_risk_threshold//[^[:alnum:]._-]/_}"
  log_path="out_${mode_name}_threshold_${prg_threshold_suffix}.log"
fi

python src/main.py \
  env.train.id=BreakoutNoFrameskip-v4 \
  common.devices=0 \
  training.compile_wm=false \
  collection.train.first_epoch.min=100 \
  collection.train.first_epoch.max=100 \
  collection.train.first_epoch.threshold_rew=0 \
  collection.train.steps_per_epoch=100 \
  collection.train.num_steps_total=100 \
  training.num_final_epochs=1 \
  denoiser.training.steps_first_epoch=1 \
  rew_end_model.training.steps_first_epoch=1 \
  actor_critic.training.steps_first_epoch=100 \
  denoiser.training.batch_size=4 \
  rew_end_model.training.batch_size=4 \
  actor_critic.training.batch_size=4 \
  world_model_env.num_batches_to_preload=1 \
  "${mode_overrides[@]}" \
  "$@" \
  2>&1 | tee "$log_path"
