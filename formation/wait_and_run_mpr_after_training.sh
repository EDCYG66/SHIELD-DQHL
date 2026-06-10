#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${TRAIN_PID:-3443}"
TRAIN_DIR="${TRAIN_DIR:-formation/results/training/paper_mpr_learned_hl_20260520_103555}"
BENCH_ROOT="${BENCH_ROOT:-formation/results/benchmarks/mpr_benchmarks/after_mpr_train_20260520_103555}"
CONDA_ENV="${CONDA_ENV:-tf212}"
COMM_WEIGHTS_DIR="${COMM_WEIGHTS_DIR:-communication/weight}"
MPR_VALUES="${MPR_VALUES:-0.30,0.35,0.375,0.40,0.50}"
STEPS="${STEPS:-900}"
CACHED_BENCH_CSV="${CACHED_BENCH_CSV:-formation/results/benchmarks/mpr_benchmarks/paper_mpr_all_9_methods/mpr_method_benchmark.csv}"
MPR_PARALLEL_WORKERS="${MPR_PARALLEL_WORKERS:-1}"
MPR_PARALLEL_WORKER_DEVICE="${MPR_PARALLEL_WORKER_DEVICE:-cpu}"

mkdir -p "$BENCH_ROOT"

log() {
    echo "[$(date)] $*"
}

pick_first_existing() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

reuse_cached_benchmark() {
    local out_dir="$1"
    local methods="$2"
    local expected="$3"

    if [[ ! -f "$CACHED_BENCH_CSV" ]]; then
        return 1
    fi

    SOURCE_CSV="$CACHED_BENCH_CSV" \
    OUT_DIR="$out_dir" \
    METHODS="$methods" \
    EXPECTED="$expected" \
    MPR_VALUES="$MPR_VALUES" \
    conda run -n "$CONDA_ENV" python - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd

source = Path(os.environ["SOURCE_CSV"])
out_dir = Path(os.environ["OUT_DIR"])
methods = [item.strip() for item in os.environ["METHODS"].split(",") if item.strip()]
expected = int(os.environ["EXPECTED"])
mpr_values = [round(float(item), 6) for item in os.environ["MPR_VALUES"].split(",") if item.strip()]

df = pd.read_csv(source)
if "method_key" not in df.columns or "mpr" not in df.columns:
    raise SystemExit(1)

df["mpr_round"] = df["mpr"].astype(float).round(6)
filtered = df[df["method_key"].isin(methods) & df["mpr_round"].isin(mpr_values)].copy()
filtered = filtered.drop(columns=["mpr_round"])

sort_cols = ["mpr"]
if "method_order" in filtered.columns:
    sort_cols.append("method_order")
filtered = filtered.sort_values(sort_cols).reset_index(drop=True)

if len(filtered) != expected:
    raise SystemExit(2)

out_dir.mkdir(parents=True, exist_ok=True)
filtered.to_csv(out_dir / "mpr_method_benchmark.csv", index=False)
(out_dir / "mpr_method_benchmark.json").write_text(
    json.dumps(filtered.to_dict(orient="records"), indent=2),
    encoding="utf-8",
)
(out_dir / "reuse_info.txt").write_text(
    f"source={source}\nmethods={','.join(methods)}\nrows={len(filtered)}\n",
    encoding="utf-8",
)
print(str(source))
PY
}

log "Waiting for training PID=$TRAIN_PID"
log "Training dir: $TRAIN_DIR"
log "Benchmark root: $BENCH_ROOT"

while ps -p "$TRAIN_PID" > /dev/null 2>&1; do
    sleep 60
done

log "Training PID ended. Resolving weights."

LEARNED_W="$(pick_first_existing \
    "$TRAIN_DIR/best_checkpoints/reward/high_level_dqn.weights.h5" \
    "$TRAIN_DIR/high_level_dqn.weights.h5" || true)"
LEARNED_META="$(pick_first_existing \
    "$TRAIN_DIR/best_checkpoints/reward/high_level_policy_meta.json" \
    "$TRAIN_DIR/high_level_policy_meta.json" || true)"

VANILLA_W="$(pick_first_existing \
    "formation/results/training/paper_vanilla_ddqn/best_checkpoints/reward/high_level_dqn.weights.h5" \
    "formation/results/training/paper_vanilla_ddqn/high_level_dqn.weights.h5" || true)"
VANILLA_META="$(pick_first_existing \
    "formation/results/training/paper_vanilla_ddqn/best_checkpoints/reward/high_level_policy_meta.json" \
    "formation/results/training/paper_vanilla_ddqn/high_level_policy_meta.json" || true)"

PPO_W="$(pick_first_existing \
    "formation/results/training/paper_ppo_hl/best_checkpoints/reward/ppo_actor.weights.h5" \
    "formation/results/training/paper_ppo_hl/ppo_actor.weights.h5" || true)"
PPO_META="$(pick_first_existing \
    "formation/results/training/paper_ppo_hl/best_checkpoints/reward/ppo_policy_meta.json" \
    "formation/results/training/paper_ppo_hl/ppo_policy_meta.json" || true)"

if [[ -z "$LEARNED_W" || -z "$LEARNED_META" ]]; then
    log "ERROR missing learned policy artifacts under $TRAIN_DIR"
    exit 1
fi

expected_rows() {
    local methods="$1"
    local method_count
    method_count=$(awk -F',' '{print NF}' <<< "$methods")
    local mpr_count
    mpr_count=$(awk -F',' '{print NF}' <<< "$MPR_VALUES")
    echo $((method_count * mpr_count))
}

benchmark_complete() {
    local out_dir="$1"
    local expected="$2"
    local csv="$out_dir/mpr_method_benchmark.csv"
    [[ -f "$csv" ]] || return 1

    local rows
    rows=$(conda run -n "$CONDA_ENV" python - "$csv" <<'PY'
import sys
import pandas as pd
try:
    print(len(pd.read_csv(sys.argv[1])))
except Exception:
    print(-1)
PY
)
    [[ "$rows" == "$expected" ]]
}

run_benchmark() {
    local name="$1"
    local methods="$2"
    local weights="$3"
    local meta="$4"
    local out_dir="$BENCH_ROOT/$name"
    local log_file="$BENCH_ROOT/$name.log"
    local done_file="$out_dir/.complete"
    local expected
    expected=$(expected_rows "$methods")

    if [[ -f "$done_file" ]] && benchmark_complete "$out_dir" "$expected"; then
        echo "[$(date)] Skipping completed benchmark: $name"
        return 0
    fi

    if benchmark_complete "$out_dir" "$expected"; then
        touch "$done_file"
        log "Skipping completed benchmark: $name"
        return 0
    fi

    mkdir -p "$out_dir"
    rm -f "$done_file"

    if [[ ! -f "$weights" || ! -f "$meta" ]]; then
        local reused_source=""
        if reused_source=$(reuse_cached_benchmark "$out_dir" "$methods" "$expected" 2>/dev/null); then
            touch "$done_file"
            log "Reused cached benchmark for $name from $reused_source"
            return 0
        fi
        printf 'skipped\nmissing_weights_or_meta\n' > "$out_dir/benchmark_status.txt"
        log "WARNING skipping benchmark: $name (missing artifacts and no reusable cached benchmark)"
        return 0
    fi

    log "Starting benchmark: $name"
    conda run -n "$CONDA_ENV" --no-capture-output python -m formation.run_mpr_method_benchmark \
        --steps "$STEPS" \
        --mpr-values "$MPR_VALUES" \
        --methods "$methods" \
        --policy-weights "$weights" \
        --policy-meta "$meta" \
        --parallel-workers "$MPR_PARALLEL_WORKERS" \
        --parallel-worker-device "$MPR_PARALLEL_WORKER_DEVICE" \
        --comm-policy agent \
        --comm-gnn gat \
        --comm-weights-dir "$COMM_WEIGHTS_DIR" \
        --spawn-y-max 520 \
        --out-dir "$out_dir" \
        > "$log_file" 2>&1

    if ! benchmark_complete "$out_dir" "$expected"; then
        log "ERROR incomplete benchmark: $name" >&2
        exit 1
    fi

    touch "$done_file"
    log "Finished benchmark: $name"
}

run_benchmark "learned_ablation" "heuristic,comm_aware,conservative,mobil_cidm_cacc,no_reconfiguration,no_communication,learned" "$LEARNED_W" "$LEARNED_META"
run_benchmark "vanilla_ddqn" "vanilla_ddqn" "$VANILLA_W" "$VANILLA_META"
run_benchmark "ppo_hl" "ppo" "$PPO_W" "$PPO_META"

log "Merging benchmark tables."

conda run -n "$CONDA_ENV" python - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd

root = Path(os.environ.get("BENCH_ROOT", "formation/results/benchmarks/mpr_benchmarks/after_mpr_train_20260520_103555"))
targets = [
    ("learned_ablation", root / "learned_ablation" / "mpr_method_benchmark.csv"),
    ("vanilla_ddqn", root / "vanilla_ddqn" / "mpr_method_benchmark.csv"),
    ("ppo_hl", root / "ppo_hl" / "mpr_method_benchmark.csv"),
]

dfs = []
available = []
missing = []
for name, path in targets:
    if path.exists():
        dfs.append(pd.read_csv(path))
        available.append(name)
    else:
        missing.append(name)

if not dfs:
    raise FileNotFoundError("No benchmark csv files were produced.")

df = pd.concat(dfs, ignore_index=True)
method_order = {
    "heuristic": 0,
    "comm_aware": 1,
    "conservative": 2,
    "mobil_cidm_cacc": 3,
    "no_reconfiguration": 4,
    "no_communication": 5,
    "learned": 6,
    "vanilla_ddqn": 7,
    "ppo": 8,
}
df["method_rank"] = df["method_key"].map(method_order).fillna(99)
df = df.sort_values(["mpr", "method_rank"]).reset_index(drop=True)

df.to_csv(root / "mpr_all_9_methods_merged.csv", index=False)
(root / "mpr_all_9_methods_merged.json").write_text(
    json.dumps(df.to_dict(orient="records"), indent=2),
    encoding="utf-8",
)
(root / "benchmark_chain_status.json").write_text(
    json.dumps(
        {
            "available_benchmarks": available,
            "missing_benchmarks": missing,
            "rows": len(df),
        },
        indent=2,
    ),
    encoding="utf-8",
)

df[["MPR", "Method", "avg_platoon_rate", "peak_max_platoon_length", "time_metric_s", "total_lane_changes"]].to_csv(
    root / "mpr_platoon_table_merged.csv",
    index=False,
)
df[["MPR", "Method", "avg_speed_all", "total_energy_kj"]].to_csv(
    root / "mpr_efficiency_table_merged.csv",
    index=False,
)

print(f"merged={root / 'mpr_all_9_methods_merged.csv'}")
print(f"rows={len(df)}")
print(f"available={available}")
print(f"missing={missing}")
PY

log "All done: $BENCH_ROOT"
