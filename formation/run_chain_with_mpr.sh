#!/bin/bash
# 串行训练三个模型 + 训练完自动跑 MPR 基准

LOG_DIR="formation/results/training"
MPR_LOG_DIR="formation/results/benchmarks/mpr_benchmarks"

run_training() {
    local policy_type=$1
    local out_dir=$2
    local log_file="$out_dir.log"

    echo "[$(date)] === Starting $policy_type ==="
    echo "[$(date)] Output: $out_dir"

    local resume_arg=""
    if [ -f "$out_dir/checkpoint/checkpoint.json" ]; then
        echo "[$(date)] Found checkpoint, resuming..."
        resume_arg="--resume-dir $out_dir"
    fi

    nohup conda run -n tf212 --no-capture-output python -m formation.run_joint_high_level_training \
        --policy-type "$policy_type" \
        --episodes 500 --steps 900 --eval-every 10 \
        --comm-policy agent --comm-gnn gat --auto-comm-weight-dir \
        --train-mpr-values "0.30,0.35,0.375,0.40,0.50" \
        --eval-mpr-values "0.35,0.375,0.40" \
        --mid-mpr-focus-ratio 0.85 \
        --max-split-streak 24 --max-reconfig-streak 36 --split-cooldown-steps 18 \
        --spawn-y-max 520 \
        $resume_arg \
        --out-dir "$out_dir" \
        > "$log_file" 2>&1

    echo "[$(date)] === $policy_type finished (exit=$?) ==="
}

run_mpr_benchmark() {
    local name=$1
    local weights_dir=$2
    local out_dir="$MPR_LOG_DIR/$name"
    local log_file="$MPR_LOG_DIR/${name}.log"

    echo "[$(date)] === Starting MPR benchmark: $name ==="

    mkdir -p "$out_dir"

    nohup conda run -n tf212 --no-capture-output python -m formation.run_mpr_method_benchmark \
        --steps 900 \
        --mpr-values "0.30,0.35,0.375,0.40,0.50" \
        --methods "heuristic,comm_aware,conservative,mobil_cidm_cacc,no_reconfiguration,no_communication,learned,vanilla_ddqn,ppo" \
        --policy-weights "$LOG_DIR/paper_learned_hl/best_checkpoints/reward/high_level_dqn.weights.h5" \
        --policy-meta "$LOG_DIR/paper_learned_hl/best_checkpoints/reward/high_level_policy_meta.json" \
        --vanilla-weights "$LOG_DIR/paper_vanilla_ddqn/best_checkpoints/reward/high_level_dqn.weights.h5" \
        --vanilla-meta "$LOG_DIR/paper_vanilla_ddqn/best_checkpoints/reward/high_level_policy_meta.json" \
        --ppo-weights "$LOG_DIR/paper_ppo_hl/best_checkpoints/reward/ppo_actor.weights.h5" \
        --ppo-meta "$LOG_DIR/paper_ppo_hl/best_checkpoints/reward/high_level_policy_meta.json" \
        --comm-policy agent --comm-gnn gat --comm-weights-dir communication/weight \
        --spawn-y-max 520 \
        --out-dir "$out_dir" \
        > "$log_file" 2>&1

    echo "[$(date)] === MPR benchmark $name finished ==="
}

echo "[$(date)] Starting full pipeline (3 models + MPR benchmarks)"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# 1. Train Learned-HL
run_training "learned" "$LOG_DIR/paper_learned_hl"

# 2. Train Vanilla-DDQN
run_training "vanilla_ddqn" "$LOG_DIR/paper_vanilla_ddqn"

# 3. Train PPO-HL
run_training "ppo" "$LOG_DIR/paper_ppo_hl"

# === MPR Benchmarks ===

# 4. MPR benchmark for Learned-HL
run_mpr_benchmark "paper_mpr_learned" \
    "$LOG_DIR/paper_learned_hl/best_checkpoints/reward"

# 5. MPR benchmark for Vanilla-DDQN
run_mpr_benchmark "paper_mpr_vanilla" \
    "$LOG_DIR/paper_vanilla_ddqn/best_checkpoints/reward"

# 6. MPR benchmark for PPO-HL
run_mpr_benchmark "paper_mpr_ppo" \
    "$LOG_DIR/paper_ppo_hl/best_checkpoints/reward"

echo "[$(date)] === All done! ==="
