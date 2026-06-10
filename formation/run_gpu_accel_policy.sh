#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICY_TYPE="${POLICY_TYPE:-shield_dqhl}"
EPISODES="${EPISODES:-200}"
STEPS="${STEPS:-140}"
EVAL_EVERY="${EVAL_EVERY:-5}"
SEED="${SEED:-126}"
N_ENVS="${N_ENVS:-8}"
OPT_LEVEL="${OPT_LEVEL:-3}"
CUPY_MEM_LIMIT_GB="${CUPY_MEM_LIMIT_GB:-1.0}"
COMM_BACKEND="${COMM_BACKEND:-auto}"
VEC_STEP_MODE="${VEC_STEP_MODE:-auto}"
RESOURCE_LOG_INTERVAL="${RESOURCE_LOG_INTERVAL:-5.0}"

case "$POLICY_TYPE" in
  shield_dqhl)
    DEFAULT_OUT_DIR="tmp/gpu_accel_shield_dqhl_$(date +%Y%m%d_%H%M%S)"
    CMD=(
      rtk proxy conda run -n tf212 --no-capture-output
      python -m gpu_optimized.launcher
      --optimization-level "$OPT_LEVEL"
      --n-envs "$N_ENVS"
      --cupy-mem-limit-gb "$CUPY_MEM_LIMIT_GB"
      --comm-backend "$COMM_BACKEND"
      --vec-step-mode "$VEC_STEP_MODE"
      --resource-log-interval "$RESOURCE_LOG_INTERVAL"
      --policy-type shield_dqhl
      --comm-policy heuristic
      --comm-gnn gat
      --episodes "$EPISODES"
      --steps "$STEPS"
      --eval-every "$EVAL_EVERY"
      --seed "$SEED"
      --n-up 10
      --n-down 10
      --train-mpr-values 0.35,0.40
      --eval-mpr-values 0.35,0.40
      --event-start 39
      --event-duration 82
      --event-center 620
      --event-length 180
      --batch-size 128
      --replay-size 20000
      --gamma 0.95
      --lr 1e-3
      --hidden-dims 128,96
      --target-update-interval 100
      --min-buffer-before-train 256
      --train-every-k-steps 1
      --train-updates-per-trigger 1
      --epsilon-start 1.0
      --epsilon-end 0.05
      --epsilon-decay-steps 6000
      --mc-dropout-samples 10
      --uncertainty-beta-base 0.30
      --uncertainty-beta-max 1.20
      --shield-num-quantiles 16
      --shield-cvar-alpha 0.25
      --shield-risk-cvar-alpha 0.15
      --shield-aux-weight 0.20
      --shield-risk-replay-fraction 0.82
      --shield-safe-replay-fraction 0.15
      --shield-penalty-weight 1.0
      --shield-split-gap-penalty 1.2
      --shield-emergency-gap-threshold 7.5
      --shield-emergency-penalty-weight 1.0
      --shield-speed-keep-penalty-weight 0.65
      --shield-speed-keep-threshold-ratio 0.82
      --shield-compact-overuse-penalty-weight 2.1
      --shield-split-context-penalty-weight 0.8
      --reward-energy-weight 0.043
      --reward-speed-excess-weight 0.0
      --reward-accel-weight 0.014
      --reward-energy-reference-kj 0.20
      --reward-speed-target-ratio 0.82
      --reward-eco-keep-bonus 0.068
      --reward-gap-recover-bonus-weight 0.44
      --reward-eco-context-bonus-weight 0.20
      --reward-eco-postevent-bonus-weight 0.24
      --governor-mid123-keep-eco-guard-enabled 1
      --skip-eval-plots
      --skip-training-plots
    )
    ;;
  vanilla_ddqn)
    DEFAULT_OUT_DIR="formation/results/training/gpu_vanilla_ddqn_$(date +%Y%m%d_%H%M%S)"
    CMD=(
      rtk proxy conda run -n tf212 --no-capture-output
      python -m gpu_optimized.launcher
      --optimization-level "$OPT_LEVEL"
      --n-envs "$N_ENVS"
      --cupy-mem-limit-gb "$CUPY_MEM_LIMIT_GB"
      --comm-backend "$COMM_BACKEND"
      --vec-step-mode "$VEC_STEP_MODE"
      --resource-log-interval "$RESOURCE_LOG_INTERVAL"
      --policy-type vanilla_ddqn
      --episodes "${EPISODES:-500}"
      --steps "${STEPS:-900}"
      --eval-every "${EVAL_EVERY:-10}"
      --seed "$SEED"
      --comm-policy agent
      --comm-gnn gat
      --auto-comm-weight-dir
      --train-mpr-values 0.30,0.35,0.375,0.40,0.50
      --eval-mpr-values 0.30,0.35,0.375,0.40,0.50
      --mid-mpr-focus-ratio 0.85
      --max-split-streak 24
      --max-reconfig-streak 36
      --split-cooldown-steps 18
    )
    ;;
  ppo)
    DEFAULT_OUT_DIR="formation/results/training/gpu_ppo_hl_$(date +%Y%m%d_%H%M%S)"
    CMD=(
      rtk proxy conda run -n tf212 --no-capture-output
      python -m gpu_optimized.launcher
      --optimization-level "$OPT_LEVEL"
      --n-envs "$N_ENVS"
      --cupy-mem-limit-gb "$CUPY_MEM_LIMIT_GB"
      --comm-backend "$COMM_BACKEND"
      --vec-step-mode "$VEC_STEP_MODE"
      --resource-log-interval "$RESOURCE_LOG_INTERVAL"
      --policy-type ppo
      --episodes "${EPISODES:-500}"
      --steps "${STEPS:-900}"
      --eval-every "${EVAL_EVERY:-10}"
      --seed "$SEED"
      --comm-policy agent
      --comm-gnn gat
      --auto-comm-weight-dir
      --train-mpr-values 0.30,0.35,0.375,0.40,0.50
      --eval-mpr-values 0.30,0.35,0.375,0.40,0.50
      --mid-mpr-focus-ratio 0.85
      --max-split-streak 24
      --max-reconfig-streak 36
      --split-cooldown-steps 18
    )
    ;;
  learned)
    DEFAULT_OUT_DIR="formation/results/training/gpu_learned_hl_$(date +%Y%m%d_%H%M%S)"
    CMD=(
      rtk proxy conda run -n tf212 --no-capture-output
      python -m gpu_optimized.launcher
      --optimization-level "$OPT_LEVEL"
      --n-envs "$N_ENVS"
      --cupy-mem-limit-gb "$CUPY_MEM_LIMIT_GB"
      --comm-backend "$COMM_BACKEND"
      --vec-step-mode "$VEC_STEP_MODE"
      --resource-log-interval "$RESOURCE_LOG_INTERVAL"
      --policy-type learned
      --episodes "${EPISODES:-500}"
      --steps "${STEPS:-900}"
      --eval-every "${EVAL_EVERY:-10}"
      --seed "$SEED"
      --comm-policy agent
      --comm-gnn gat
      --auto-comm-weight-dir
      --train-mpr-values 0.30,0.35,0.375,0.40,0.50
      --eval-mpr-values 0.30,0.35,0.375,0.40,0.50
      --mid-mpr-focus-ratio 0.85
      --max-split-streak 24
      --max-reconfig-streak 36
      --split-cooldown-steps 18
    )
    ;;
  *)
    echo "Unsupported POLICY_TYPE: $POLICY_TYPE" >&2
    exit 1
    ;;
esac

OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
CMD+=(--out-dir "$OUT_DIR")

printf 'Launching GPU-accelerated policy run\n'
printf 'policy_type: %s\n' "$POLICY_TYPE"
printf 'out_dir: %s\n' "$OUT_DIR"
printf 'n_envs: %s\n' "$N_ENVS"

"${CMD[@]}"
