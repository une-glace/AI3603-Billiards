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

## 3. 训练命令 (Training Command)

在项目根目录下运行以下命令启动训练：

```bash
python train/train_ppo.py
```

训练过程中：
- TensorBoard 日志将保存在 `train/logs/`
- 模型 Checkpoint 将保存在 `train/checkpoints/`
- 最终模型将保存为 `train/ppo_billiards_final.zip`

## 4. 超参数设置 (Hyperparameters)

在 `train_ppo.py` 中使用了以下 PPO 默认超参数（可根据需要在代码中调整）：

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MlpPolicy (多层感知机)
- **Learning Rate**: 3e-4
- **n_steps**: 2048 (每次更新收集的步数)
- **batch_size**: 64
- **n_epochs**: 10
- **gamma**: 0.99 (折扣因子)
- **gae_lambda**: 0.95
- **clip_range**: 0.2
- **ent_coef**: 0.01 (熵系数，鼓励探索)
- **Total Timesteps**: 1,000,000 (一百万步)

## 5. 奖励函数设计 (Reward Function)

奖励函数在 `pool_gym.py` 的 `_compute_reward` 方法中定义，主要包含：
- **进球奖励**: 每打进一个己方目标球 +10
- **胜利奖励**: 赢得比赛（打进黑8）+100
- **失败/犯规惩罚**: 
  - 母球进袋: -10 (若导致输掉比赛则额外 -50)
  - 犯规（无接触、首球错误等）: -2 ~ -5
  - 击球未进球: -0.1 (鼓励尽快进球)
