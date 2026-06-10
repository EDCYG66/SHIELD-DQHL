import subprocess
import json
import os
import pandas as pd
import sys
from pathlib import Path

# ================= 配置区域 =================
# 定义输出总目录
BASE_OUT_DIR = "runs/ablation_study_final"

# 定义通用的实验参数 (保持环境一致)
COMMON_ARGS = [
    "--topo", "tree",
    "--n-up", "8", 
    "--n-down", "12",          # 总车数 20 (中等密度，适合观察收敛)
    "--lanes", "4",
    "--train-steps", "3000",   # 训练步数 (建议 3000 或 4000)
    "--test-every", "100",     # 每 100 步测试一次
    "--test-sample", "100",    # 测试采样次数
    "--seed", "2025",          # 固定随机种子
    "--epsilon-min", "0.01",
    "--demand-amount", "150",  # 稍微增加难度以拉开差距
    "--v2v-limit", "0.04"
]

# 定义要对比的模型列表
MODELS = ["gat", "sage", "fc"]

# ===========================================

def run_command(cmd):
    """执行 Shell 命令并实时打印输出"""
    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        # 使用 subprocess 调用 run_compare_gnn_plus.py
        subprocess.check_call([sys.executable] + cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    base_path = Path(BASE_OUT_DIR)
    base_path.mkdir(parents=True, exist_ok=True)
    
    summary_data = {}

    print("========================================")
    print("   开始运行消融实验 (Ablation Study)")
    print("   Models: GATv2, GraphSAGE, FC-DQN")
    print("========================================")

    for model in MODELS:
        print(f"\n[Task] Training model: {model.upper()} ...")
        
        # 为每个模型创建子目录
        model_out_dir = base_path / model
        
        # 构造命令
        # 调用 run_compare_gnn_plus.py
        cmd = [
            "run_compare_gnn_plus.py",
            "--out-dir", str(model_out_dir),
            "--gnn-type", model
        ] + COMMON_ARGS
        
        run_command(cmd)
        
        # 读取运行生成的 summary json
        json_path = model_out_dir / "comparison_summary.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                # 提取该模型的曲线数据
                # JSON 结构通常是 { "gat": { ... }, "settings": ... }
                # 或者 FC 模式下是 { "fc": { ... }, "settings": ... }
                if model in data:
                    model_data = data[model]
                    # 我们只需要 v2v_curve (steps, success_rate)
                    summary_data[model] = model_data.get("v2v_curve", [])
                    print(f"   -> {model.upper()} data loaded. Final V2V: {model_data.get('v2v_final'):.4f}")
                elif "graphsage" in data and model == "sage": # 兼容可能的键名差异
                    summary_data[model] = data["graphsage"].get("v2v_curve", [])
                    print(f"   -> SAGE data loaded.")
        else:
            print(f"   [Warning] Output JSON not found for {model}")

    # 将汇总数据保存到总目录
    final_json_path = base_path / "ablation_all_curves.json"
    with open(final_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print("\n========================================")
    print("   所有实验运行完毕！")
    print(f"   汇总数据已保存至: {final_json_path}")
    print("========================================")

if __name__ == "__main__":
    main()