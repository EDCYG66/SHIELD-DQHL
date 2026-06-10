#!/bin/bash
# 等待当前训练完成后自动启动 MPR 基准实验

LOG="formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig.log"

echo "[$(date)] Waiting for training to finish..."
echo "[$(date)] Monitoring: $LOG"

# 等待训练进程结束
while ps aux | grep "run_joint_high_level_training" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Training finished! Starting MPR benchmark..."

# 创建输出目录
mkdir -p formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig

# 启动 MPR 基准实验
nohup conda run -n tf212 --no-capture-output python -m formation.run_mpr_method_benchmark \
  --steps 900 \
  --mpr-values "0.30,0.35,0.375,0.40,0.50" \
  --methods "heuristic,comm_aware,conservative,no_reconfiguration,no_communication,learned" \
  --policy-weights formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig/high_level_dqn.weights.h5 \
  --policy-meta formation/results/training/paper_joint_gatv2_comm_long_midmpr_reconfig/high_level_policy_meta.json \
  --comm-policy agent \
  --comm-gnn gat \
  --comm-weights-dir communication/weight \
  --out-dir formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig \
  > formation/results/benchmarks/mpr_benchmarks/paper_joint_gatv2_comm_long_midmpr_reconfig.log 2>&1 &

echo "[$(date)] MPR benchmark started (PID: $!)"

# 等待 MPR 基准完成
while ps aux | grep "run_mpr_method_benchmark" | grep -v grep > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] All experiments completed!"
