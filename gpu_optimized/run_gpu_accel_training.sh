#!/usr/bin/env bash
set -euo pipefail

# ===== GPU加速训练脚本 =====
# 使用 gpu_optimized 模块，optimization-level 3
# 包含资源监控 + 训练 + 最佳模型benchmark

RUN_TAG="gpu_accel_300ep_$(date +%Y%m%d_%H%M%S)"
TRAIN_DIR="formation/results/training/${RUN_TAG}"
BENCH_DIR="formation/results/benchmarks/mpr_benchmarks/${RUN_TAG}"
LOG_FILE="tmp/${RUN_TAG}.nohup.log"
MONITOR_DIR="tmp/${RUN_TAG}"
CONDA_BASE="/usr/local/miniconda3"

mkdir -p "$MONITOR_DIR" "tmp"

cat > "${MONITOR_DIR}/run_info.txt" <<INFO
run_tag: ${RUN_TAG}
train_dir: ${TRAIN_DIR}
bench_dir: ${BENCH_DIR}
log_file: ${LOG_FILE}
started: $(date -Iseconds)
mode: gpu_optimized (level 3, n_envs=16)
episodes: 300
steps: 900
description: GPU-accelerated training with CuPy + vectorized env + batch inference + optimized replay
INFO

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate tf212

# ===== 资源监控（后台） =====
(
  echo "timestamp,gpu_util_pct,gpu_mem_mib,cpu_load_1m" > "${MONITOR_DIR}/resource_monitor.csv"
  while true; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ' || echo "0 0")
    gpu_util=$(echo "$gpu" | awk '{print $1}')
    gpu_mem=$(echo "$gpu" | awk '{print $2}')
    cpu_load=$(awk '{print $1}' /proc/loadavg)
    echo "$(date -Iseconds),${gpu_util},${gpu_mem},${cpu_load}" >> "${MONITOR_DIR}/resource_monitor.csv"
    sleep 30
  done
) &
MONITOR_PID=$!

echo "=========================================="
echo "[GPU-Accel] Training started at $(date -Iseconds)"
echo "[GPU-Accel] Run tag: ${RUN_TAG}"
echo "[GPU-Accel] Log: ${LOG_FILE}"
echo "=========================================="

# ===== GPU加速训练 =====
python -m gpu_optimized.launcher \
  --optimization-level 3 \
  --n-envs 16 \
  --cupy-mem-limit-gb 4.0 \
  --policy-type learned \
  --episodes 300 \
  --steps 900 \
  --seed 42 \
  --eval-every 10 \
  --comm-policy agent \
  --comm-gnn gat \
  --auto-comm-weight-dir \
  --warmstart-heuristic-episodes 0 \
  --train-mpr-values 0.30,0.35,0.375,0.40,0.50 \
  --eval-mpr-values 0.30,0.35,0.375,0.40,0.50 \
  --epsilon-decay-steps 15000 \
  --out-dir "${TRAIN_DIR}"

echo "[GPU-Accel] Training finished at $(date -Iseconds)"

# ===== 最佳模型 Benchmark =====
for CKPT_NAME in reward safe_reward safety_first; do
  CKPT_DIR="${TRAIN_DIR}/best_checkpoints/${CKPT_NAME}"
  WEIGHTS="${CKPT_DIR}/high_level_dqn.weights.h5"
  META="${CKPT_DIR}/high_level_policy_meta.json"

  if [ ! -f "$WEIGHTS" ] || [ ! -f "$META" ]; then
    echo "[Benchmark] skip ${CKPT_NAME}: missing weights or meta"
    continue
  fi

  echo "[Benchmark] ${CKPT_NAME} started at $(date -Iseconds)"

  python -m formation.run_mpr_method_benchmark \
    --steps 900 \
    --mpr-values 0.30,0.35,0.375,0.40,0.50 \
    --methods learned \
    --policy-weights "${WEIGHTS}" \
    --policy-meta "${META}" \
    --comm-policy agent \
    --comm-gnn gat \
    --comm-weights-dir communication/weight \
    --parallel-workers 2 \
    --parallel-worker-device cpu \
    --out-dir "${BENCH_DIR}/${CKPT_NAME}"

  echo "[Benchmark] ${CKPT_NAME} finished at $(date -Iseconds)"
done

kill "$MONITOR_PID" 2>/dev/null || true
echo "=========================================="
echo "[GPU-Accel] All done at $(date -Iseconds)"
echo "[GPU-Accel] Results: ${TRAIN_DIR}"
echo "=========================================="
