import json
from pathlib import Path

import matplotlib.pyplot as plt

# 1. 读取 comparison_summary.json
summary_path = Path("runs/gat_10000/comparison_summary.json")
with summary_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

gat = data["gat"]
ts = gat["last_timeseries"]

t = ts["t"]
v2i = ts["v2i"]
v2v = ts["v2v_succ"]
nveh = ts["nveh"]

# 2. 画三条曲线（类似你图 10）
fig, ax1 = plt.subplots(figsize=(8, 5))

# 左 Y 轴：V2V 成功率、V2I 速率
color_v2v = "tab:blue"
color_v2i = "tab:orange"

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("V2V / V2I", color="black")
l1, = ax1.plot(t, v2v, color=color_v2v, label="Instantaneous V2V Success Rate")
l2, = ax1.plot(t, v2i, color=color_v2i, label="Instantaneous V2I Rate")
ax1.tick_params(axis="y", labelcolor="black")

# 右 Y 轴：车辆数量
ax2 = ax1.twinx()
color_nveh = "tab:green"
ax2.set_ylabel("Number of Vehicles", color=color_nveh)
l3, = ax2.plot(t, nveh, color=color_nveh, label="Number of Vehicles")
ax2.tick_params(axis="y", labelcolor=color_nveh)

# 合并图例
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Dynamics of V2V/V2I and Vehicle Count (GAT)")
plt.tight_layout()

out_dir = summary_path.parent
plt.savefig(out_dir / "gat_timeseries.png", dpi=300)
plt.close()