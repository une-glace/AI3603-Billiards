# AI3603-Billiards
AI3603课程台球大作业

## 关键文件说明

| 文件 | 作用 | 在最终测试中是否可修改 |
|------|------|-----------|
| `poolenv.py` | 台球环境（游戏规则） | ❌ 不可修改 |
| `agent.py` | Agent 定义（在 `NewAgent` 中实现你的算法） | ✅ 可修改 `NewAgent` |
| `evaluate.py` | 评估脚本（运行对战） | ✅ 可修改 `agent_b` |
| `PROJECT_GUIDE.md` | 项目详细指南 | 📖 参考文档 |
| `GAME_RULES.md` | 游戏规则说明 | 📖 参考文档 |

---

## 强化学习训练说明

本项目使用 PPO 算法进行训练。

### 1. 训练命令

在项目根目录下运行：

```bash
python train/train_ppo.py
```

**常用参数：**
*   `--n_envs`: 并行环境数量（默认为 4）。建议设置为 CPU 核心数。
    ```bash
    python train/train_ppo.py --n_envs 8
    ```
*   `--total_timesteps`: 训练总步数（默认为 1,000,000）。
*   `--save_freq`: 模型保存频率（步数）。

### 2. 训练特性

*   **断点续训**：
    *   随时可以使用 `Ctrl+C` 中断训练。
    *   中断时会自动保存当前模型为 `train/ppo_billiards_final.zip`。
    *   再次运行命令时，程序会自动检测并加载最新的 Checkpoint（包括 `final.zip` 或 `checkpoints/` 目录下的定期存档），继续训练。
    *   旧的 `final.zip` 会被自动备份为 `ppo_billiards_final_backup_{时间戳}.zip`，防止覆盖。

*   **日志与可视化**：
    *   **TensorBoard**：
        *   运行 `tensorboard --logdir=train/logs` 查看训练曲线。
        *   每次运行都会生成一个新的 Run（带时间戳），方便对比不同次实验。
        *   可以同时勾选多个 Run 来查看连续的训练历史。
    *   **实时图片**：
        *   在 `train/logs/training_plots/` 下会生成 `reward_curve.png`。
        *   该曲线会自动加载历史数据，保持视觉上的连续性。

### 3. 文件结构

*   `train/train_ppo.py`: 主训练脚本。
*   `train/pool_gym.py`: Gym 环境包装器（定义了 Reward 和 Observation）。
*   `train/visualize_callback.py`: 自定义回调，用于实时绘图。
*   `train/checkpoints/`: 定期保存的模型文件。
*   `train/logs/`: TensorBoard 日志和 Monitor 数据（`.monitor.csv`）。
