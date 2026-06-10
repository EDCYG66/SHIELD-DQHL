# 面向突发事件场景的车队队列重构与通信协同控制研究

本仓库是论文课题的主工作区，围绕“突发事件场景下的车队队列重构与通信协同控制”展开，包含通信侧资源调度、编队侧高层重构控制、SHIELD-DQHL 训练与对比实验、论文写作材料以及若干隔离实验沙箱。

当前项目可以理解为两条主线协同推进：

- `communication/`：通信模块，侧重 C-V2X 场景下的 V2V/V2I 资源调度，已形成较完整的实验与投稿材料。
- `formation/`：编队模块，侧重突发事件场景下的高层重构策略、低层协同控制、安全盾约束与联合评价。

## 仓库结构

```text
MyProject/
├── communication/              # 通信侧代码与投稿材料
├── formation/                  # 编队控制主代码
├── gpu_optimized/              # GPU / CuPy / TensorFlow 相关加速验证
├── formation_hl_backup_20260607/  # 历史高层训练备份

```

## 核心模块说明

### `communication/`

通信模块主要实现基于 GATv2-DDQN 的资源调度方法，包含：

- 通信环境与图结构建模
- GAT / GATv2 / GraphSAGE 等图神经网络实现
- DQN 智能体与对比、消融、鲁棒性实验脚本
- 部分投稿材料与图表生成脚本

如果你的目标是查看通信论文对应实现，优先从这里开始。

### `formation/`

编队模块是当前主攻方向，主要包含：

- 高层队列重构策略
- 低层 CACC / C-IDM / 混合控制逻辑
- 安全盾与风险修正机制
- 可训练高层策略、PPO、vanilla DDQN、SHIELD-DQHL 等实现
- 场景生成、评测、可视化与结果分析脚本

如果你的目标是继续优化 SHIELD-DQHL 或整理编队实验，优先从这里开始。



## 当前研究状态

- 通信模块已基本完成，已有较完整实验链和投稿材料。
- 编队模块已完成基础框架、联合状态设计、典型策略与若干中等训练实验。
- SHIELD-DQHL 仍处在持续优化阶段，重点问题通常集中在安全、reward、能耗与动作分布之间的平衡。
- `hl_risk_lab/` 用于隔离验证新的高层策略路线，稳定后再考虑迁回主线。

## 运行环境

本仓库长期在 WSL 环境下开发，默认约定如下：

- Shell 命令优先使用 `rtk <cmd>`
- TensorFlow / SHIELD 相关运行优先使用 `tf212`
- 常用方式：

```bash
conda run -n tf212 python <script>.py
```

或：

```bash
/home/edcyg/miniconda3/envs/tf212/bin/python <script>.py
```

说明：

- 当前活跃 shell 不一定带有可用 TensorFlow
- 与 GPU 加速相关的修复和验证主要集中在 `gpu_optimized/`

## 常见入口

通信侧示例：

```bash
rtk conda run -n tf212 python communication/main_highway.py
```

编队训练示例：

```bash
rtk conda run -n tf212 python formation/run_trainable_high_level_policy.py
```

```


## 使用建议

如果你是第一次接触这个仓库，推荐阅读顺序：

1. 先看本 README，理解仓库分区
2. 看 `plan/project-overview.md` 了解课题目标
3. 根据任务进入 `communication/` 或 `formation/`
4. 若是 SHIELD-DQHL 延续实验，重点查看 `SHIELD-DQHL test/`
5. 若要试新想法，优先在 `hl_risk_lab/` 隔离验证

## 说明

这个仓库首先是研究工作区，其次才是可复用代码仓库，因此会同时存在：

- 主代码
- 历史备份
- 论文材料
- 临时实验目录
- 中间结果与辅助脚本

如果后续需要对外发布，建议再从当前仓库中抽取一个更干净的公开版，仅保留 `communication/`、`formation/`、必要文档和可复现实验入口。
