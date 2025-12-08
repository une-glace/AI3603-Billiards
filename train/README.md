# 训练指南 (Training Guide)

本目录包含用于训练强化学习 Agent (NewAgent) 的相关代码和说明。

## 1. 环境配置 (Environment Configuration)

确保项目根目录下的 `requirements.txt` 中的依赖已安装。此外，需要安装 `stable-baselines3` 和 `shimmy` (用于兼容 gymnasium)。

```bash
pip install stable-baselines3 shimmy gymnasium
```

## 2. 文件说明 (File Description)

- `pool_gym.py`: 包含 `BilliardsGymEnv` 类，这是一个将 `PoolEnv` 包装为 Gymnasium 兼容环境的 Wrapper。它定义了动作空间（连续控制）和观测空间（球位置 + 目标类型）。
- `train_ppo.py`: 使用 PPO (Proximal Policy Optimization) 算法进行训练的主脚本。包含模型定义、训练循环和 Checkpoint 保存逻辑。
- `visualize_callback.py`: 自定义回调函数，用于在训练过程中实时绘制和保存 Reward 曲线。
- `logs/`: 存放训练日志（Monitor CSV）和 TensorBoard 事件文件。
- `checkpoints/`: 存放定期保存的模型权重（.zip 文件）。

## 3. 训练命令 (Training Command)

在项目根目录下运行以下命令启动训练：

```bash
# 默认训练 1,000,000 步，每 10,000 步保存一次
python train/train_ppo.py

# 使用 8 个并行环境加速训练（建议设置为 CPU 核心数）
python train/train_ppo.py --n_envs 8

# 自定义训练总步数（例如 200,000 步）
python train/train_ppo.py --total_timesteps 200000

# 自定义模型保存频率（例如每 5000 步保存一次）
python train/train_ppo.py --save_freq 5000
```

### 断点续训与自动保存
- **中断保护**: 随时可以使用 `Ctrl+C` 中断训练。程序会自动保存当前模型为 `train/ppo_billiards_final.zip`，并尝试绘制最终的训练曲线。
- **自动续训**: 每次启动时，程序会自动检测 `final.zip` 或 `checkpoints/` 目录下的最新模型，并从上次中断的地方继续训练。
- **安全备份**: 旧的 `final.zip` 文件会被自动重命名备份（例如 `ppo_billiards_final_backup_1701234567.zip`），确保历史最佳模型不会被意外覆盖。

### 训练日志与输出
- **日志记录 (Logs)**: 所有的训练日志（Monitor CSV, TensorBoard event files）都保存在 `train/logs/` 目录下。
- **实时图像 (Plots)**: 
    - `train/logs/training_plots/reward_curve.png`: 实时更新的 Reward 曲线图。脚本会自动加载历史 Monitor 数据，确保曲线在视觉上是连续的。
    - `train/logs/training_plots/final_reward_curve.png`: 训练结束（或中断）时生成的完整历史曲线。
- **模型存档 (Checkpoints)**: 中间模型默认每隔 10000 步保存一次（可通过 `--save_freq` 修改），存放在 `train/checkpoints/`。
- **最终模型 (Final Model)**: 训练完成后的最终模型将保存为 `train/ppo_billiards_final.zip`。

### 如何查看训练过程 (Monitoring)

**方式一：查看自动生成的图像 (推荐)**
直接打开 `train/logs/training_plots/reward_curve.png`。
- 该图片会随着训练进行自动更新（每 10 个 episode 更新一次）。
- 红色曲线代表滑动平均值，能更清晰地反映趋势。

**方式二：使用 TensorBoard (高级)**
如果您想查看 Loss 曲线、学习率变化等更详细的指标，可以使用 TensorBoard：

1. 打开终端，进入项目根目录。
2. 运行命令：
   ```bash
   tensorboard --logdir ./train/logs/
   ```
3. 在浏览器中访问终端输出的地址（通常是 `http://localhost:6006`）。
   - **注意**: TensorBoard 可能会显示多个 Run（如 `PPO_run_170...`）。您可以同时勾选多个 Run 来查看连续的训练历史。

## 4. 图像绘制与数据说明 (Visualization Guide)

### 图片文件的含义
- **`reward_curve.png` (实时训练曲线)**
  - **生成时机**: 训练过程中实时生成，每完成 10 个 Episode 更新一次。
  - **数据来源**: 内存中累积的 Reward 数据 + 启动时加载的历史 Monitor 日志。
  - **用途**: 用于实时监控训练是否有进展（Reward 是否在上升）。
  - **注意**: 如果中断后重新开始，脚本会读取之前的日志，接续绘制，保持曲线连续。

- **`final_reward_curve.png` (最终复盘曲线)**
  - **生成时机**: 训练正常结束或被 `Ctrl+C` 中断时生成。
  - **数据来源**: 直接读取磁盘上所有的 `*.monitor.csv` 日志文件。
  - **用途**: 最完整、最准确的历史记录。即使中间内存数据丢失，只要日志文件还在，这张图就是完整的。

### 关键概念解释
- **Episode (回合)**: 
  - 指一局完整的台球游戏。
  - 结束条件: 赢了 (打进黑8)、输了 (母球进袋/打错黑8)、或达到最大步数 (60步)。
- **Step (步数)**: 
  - Agent 做出一次击球动作算 1 步。
  - 一个 Episode 通常包含 10~60 个 Steps。
- **多核训练 (Multi-Core)**:
  - 使用 `--n_envs 8` 时，相当于 8 个台球桌同时开打。
  - **Total Timesteps** 是 8 个核步数的总和。
  - **Episode 计数** 是所有核完成局数的总和 (任何一个核打完一局，总 Episode 数 +1)。

### TensorBoard 记录机制
- TensorBoard 并不是每一步都记录。
- **记录频率**: PPO 算法每收集满 `n_steps` (默认 2048) 个数据，进行一次网络更新，然后记录一次日志。
- **现象**: 刚开始训练时 (小于 2048 步)，TensorBoard 可能是空的，这是正常现象。

## 5. 超参数设置 (Hyperparameters)

在 `train_ppo.py` 中使用了以下 PPO 默认超参数（可根据需要在代码中调整）：

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MlpPolicy (多层感知机)
- **Learning Rate**: 3e-4
- **n_envs**: 4 (默认并行环境数，可通过命令行调整)
- **n_steps**: 2048 (每次更新收集的步数)
- **batch_size**: 64
- **n_epochs**: 10
- **gamma**: 0.99 (折扣因子)
- **gae_lambda**: 0.95
- **clip_range**: 0.2
- **ent_coef**: 0.01 (熵系数，鼓励探索)

## 6. 奖励函数设计 (Reward Function)

奖励函数在 `pool_gym.py` 的 `_compute_reward` 方法中定义，主要包含：
- **进球奖励**: 每打进一个己方目标球 +10
- **胜利奖励**: 赢得比赛（打进黑8）+100
- **失败/犯规惩罚**: 
  - 母球进袋: -10 (若导致输掉比赛则额外 -50)
  - 犯规（无接触、首球错误等）: -2 ~ -5
  - 击球未进球: -0.1 (鼓励尽快进球)
