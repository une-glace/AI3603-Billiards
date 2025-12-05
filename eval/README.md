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

1.  **训练模型**：完成 `train/` 目录下的训练流程。
    *   推荐使用 PPO 算法训练至少 500,000 步。
    *   确保 `train/logs/training_plots/reward_curve.png` 显示 Reward 呈上升趋势并收敛。

2.  **部署模型**：
    *   训练结束后，`train/` 目录下会生成 `ppo_billiards_final.zip`。
    *   将该文件复制到 `eval/checkpoints/` 目录，并重命名为 `best_model.zip`。
    *   或者，你也可以选择 `train/checkpoints/` 中表现最好的中间存档。

3.  **运行评估**：
    *   运行 `python eval/eval_script.py`。
    *   脚本将自动加载 `eval/checkpoints/best_model.zip` 并进行 100 局对战测试。

## 4. 策略说明 (NewAgent)

NewAgent 采用混合策略 (Hybrid Strategy)：

*   **强化学习 (RL)**：加载 PPO 模型，根据当前盘面生成动作建议（主要负责复杂局面的决策）。
*   **启发式规则 (Heuristic)**：使用几何法（Ghost Ball）生成候选动作（主要负责简单直球的精确打击）。
*   **模拟验证 (Simulation Verification)**：
    *   Agent 会在内部使用 Pooltool 物理引擎对 RL 动作和启发式动作进行快速模拟。
    *   选择预期得分最高（如必定进球）的动作执行。
    *   这种方法结合了 RL 的长期规划能力和物理模拟的精确性，能显著提高胜率。
