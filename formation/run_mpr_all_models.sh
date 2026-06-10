#!/bin/bash
# 等当前 pipeline 完成后，补跑包含全部 9 种方法的 MPR 基准
# 包括: heuristic, comm_aware, conservative, mobil_cidm_cacc,
#       no_reconfiguration, no_communication, learned, vanilla_ddqn, ppo

LOG_DIR="formation/results/training"
MPR_LOG_DIR="formation/results/benchmarks/mpr_benchmarks"

echo "[$(date)] Waiting for current pipeline to finish..."
echo "[$(date)] Monitoring: $LOG_DIR/full_pipeline.log"

# 等待当前 pipeline 进程结束
while ps aux | grep "run_chain_with_mpr" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done

# 也等待可能残留的训练进程
while ps aux | grep "run_joint_high_level_training" | grep -v grep > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] Pipeline finished. Starting full MPR benchmark..."

# 检查三个模型的权重是否存在
LEARNED_W="$LOG_DIR/paper_learned_hl/best_checkpoints/reward/high_level_dqn.weights.h5"
VANILLA_W="$LOG_DIR/paper_vanilla_ddqn/best_checkpoints/reward/high_level_dqn.weights.h5"
PPO_W="$LOG_DIR/paper_ppo_hl/best_checkpoints/reward/ppo_actor.weights.h5"

for f in "$LEARNED_W" "$VANILLA_W" "$PPO_W"; do
    if [ ! -f "$f" ]; then
        echo "[$(date)] WARNING: Weight file not found: $f"
    fi
done

OUT_DIR="$MPR_LOG_DIR/paper_mpr_all_9_methods"
LOG_FILE="$MPR_LOG_DIR/paper_mpr_all_9_methods.log"
mkdir -p "$OUT_DIR"

nohup conda run -n tf212 --no-capture-output python -m formation.run_mpr_method_benchmark \
    --steps 900 \
    --mpr-values "0.30,0.35,0.375,0.40,0.50" \
    --methods "heuristic,comm_aware,conservative,mobil_cidm_cacc,no_reconfiguration,no_communication,learned,vanilla_ddqn,ppo" \
    --policy-weights "$LEARNED_W" \
    --policy-meta "$LOG_DIR/paper_learned_hl/best_checkpoints/reward/high_level_policy_meta.json" \
    --vanilla-weights "$VANILLA_W" \
    --vanilla-meta "$LOG_DIR/paper_vanilla_ddqn/best_checkpoints/reward/high_level_policy_meta.json" \
    --ppo-weights "$PPO_W" \
    --ppo-meta "$LOG_DIR/paper_ppo_hl/best_checkpoints/reward/high_level_policy_meta.json" \
    --comm-policy agent --comm-gnn gat --comm-weights-dir communication/weight \
    --spawn-y-max 520 \
    --out-dir "$OUT_DIR" \
    > "$LOG_FILE" 2>&1

echo "[$(date)] === Full 9-method MPR benchmark finished ==="
