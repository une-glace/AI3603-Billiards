# 评估指南 (Evaluation Guide)

本目录包含用于测试和评估 Agent 性能的文件。

## 1. 文件说明

- `eval_script.py`: 评估脚本，运行 100 局比赛并输出 NewAgent 对战 BasicAgent 的胜率。
- `checkpoints/`: 存放训练好的模型文件。请将表现最好的模型重命名为 `best_model.zip` 并放入此文件夹。

## 2. 如何运行测试

在项目根目录下运行：

```bash
python eval/eval_script.py
```

或者，如果您想指定特定的模型路径或局数，可以在代码中修改 `evaluate` 函数的调用参数。

## 3. 复现说明

为了复现 ≥70% 的胜率：
1. 完成 `train/` 目录下的训练流程。
2. 将训练得到的最佳模型（例如 `train/ppo_billiards_final.zip` 或 `train/checkpoints/` 下的中间存档）复制到 `eval/checkpoints/best_model.zip`。
3. 运行 `eval_script.py`。

NewAgent 采用混合策略：
- 如果加载了 RL 模型，首先使用 PPO 网络预测一个候选动作。
- 同时使用几何启发式（Ghost Ball）生成候选动作。
- 通过 Pooltool 物理引擎模拟这些候选动作（Simulation Verification），选择得分最高的动作执行。
这种方法结合了 RL 的长期规划能力和物理模拟的精确性。
