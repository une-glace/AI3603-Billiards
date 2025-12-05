import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.pool_gym import BilliardsGymEnv

def train():
    # 1. 创建环境
    env = BilliardsGymEnv()
    
    # 2. 定义模型 (PPO)
    # 使用 MlpPolicy 因为输入是向量
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # 增加探索
        tensorboard_log="./train/logs/"
    )
    
    # 3. 设置检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./train/checkpoints/',
        name_prefix='ppo_billiards'
    )
    
    # 4. 开始训练
    print("开始训练 PPO Agent...")
    try:
        model.learn(
            total_timesteps=1000000, # 训练步数，可根据需要调整
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("训练被用户中断")
    
    # 5. 保存最终模型
    model.save("train/ppo_billiards_final")
    print("模型已保存至 train/ppo_billiards_final.zip")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("./train/logs/", exist_ok=True)
    os.makedirs("./train/checkpoints/", exist_ok=True)
    
    train()
