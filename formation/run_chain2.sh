#!/bin/bash
# 补齐最后9个episode + 自动接MPR基准

echo "[$(date)] Starting 9-episode tail run..."

nohup conda run -n tf212 --no-capture-output python -m formation.run_joint_high_level_training \
  --episodes 9 \
  --steps 900 \
  --eval-every 3 \
  --comm-policy agent \
  --comm-gnn gat \
  --auto-comm-weight-dir \
  --train-mpr-values "0.30,0.35,0.375,0.40,0.50" \
  --eval-mpr-values "0.35,0.375,0.40" \
  --mid-mpr-focus-ratio 0.85 \
  --max-split-streak 24 \
  --max-reconfig-streak 36 \
  --split-cooldown-steps 18 \
  --out-dir formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig_tail \
  > formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig_tail.log 2>&1

echo "[$(date)] Tail run finished. Starting MPR benchmark..."

# 合并两次训练的权重（使用best_checkpoints中的reward权重）
mkdir -p formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig

nohup conda run -n tf212 --no-capture-output python -m formation.run_mpr_method_benchmark \
  --steps 900 \
  --mpr-values "0.30,0.35,0.375,0.40,0.50" \
  --methods "heuristic,comm_aware,conservative,no_reconfiguration,no_communication,learned" \
  --policy-weights formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig/best_checkpoints/reward/high_level_dqn.weights.h5 \
  --policy-meta formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig_tail/high_level_policy_meta.json \
  --comm-policy agent \
  --comm-gnn gat \
  --comm-weights-dir communication/weight \
  --out-dir formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig \
  > formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig.log 2>&1

echo "[$(date)] All done!"
