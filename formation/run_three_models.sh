#!/bin/bash
# 串行训练三个高层策略模型，支持断点续传
# 模型: learned (HL), vanilla_ddqn, ppo

LOG_DIR="formation/results/training"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_training() {
    local policy_type=$1
    local out_dir=$2
    local log_file="$out_dir.log"

    echo "[$(date)] === Starting $policy_type ==="
    echo "[$(date)] Output: $out_dir"
    echo "[$(date)] Log: $log_file"

    # 检查是否有 checkpoint 可以续传
    local resume_arg=""
    if [ -f "$out_dir/checkpoint/checkpoint.json" ]; then
        echo "[$(date)] Found checkpoint, will resume from $out_dir"
        resume_arg="--resume-dir $out_dir"
    fi

    nohup conda run -n tf212 --no-capture-output python -m formation.run_joint_high_level_training \
        --policy-type "$policy_type" \
        --episodes 500 \
        --steps 900 \
        --eval-every 10 \
        --comm-policy agent \
        --comm-gnn gat \
        --auto-comm-weight-dir \
        --train-mpr-values "0.30,0.35,0.375,0.40,0.50" \
        --eval-mpr-values "0.35,0.375,0.40" \
        --mid-mpr-focus-ratio 0.85 \
        --max-split-streak 24 \
        --max-reconfig-streak 36 \
        --split-cooldown-steps 18 \
        $resume_arg \
        --out-dir "$out_dir" \
        > "$log_file" 2>&1

    local exit_code=$?
    echo "[$(date)] === $policy_type finished (exit=$exit_code) ==="
    return $exit_code
}

echo "[$(date)] Starting 3-model serial training chain"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# 1. Learned-HL
run_training "learned" "$LOG_DIR/paper_learned_hl"
if [ $? -ne 0 ]; then
    echo "[$(date)] WARNING: learned training failed, continuing..."
fi

# 2. Vanilla-DDQN
run_training "vanilla_ddqn" "$LOG_DIR/paper_vanilla_ddqn"
if [ $? -ne 0 ]; then
    echo "[$(date)] WARNING: vanilla_ddqn training failed, continuing..."
fi

# 3. PPO-HL
run_training "ppo" "$LOG_DIR/paper_ppo_hl"
if [ $? -ne 0 ]; then
    echo "[$(date)] WARNING: ppo training failed, continuing..."
fi

echo "[$(date)] === All 3 models completed ==="
