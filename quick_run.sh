#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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
  2>&1 | tee out.log
