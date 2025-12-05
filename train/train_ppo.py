import os
import sys
import glob
import time
import shutil
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.pool_gym import BilliardsGymEnv
from train.visualize_callback import TrainingVisualizerCallback

def get_latest_checkpoint(path):
    # 查找目录下所有符合模式的 checkpoint 文件
    # path 通常是 ./train/checkpoints/
    checkpoints = glob.glob(os.path.join(path, "ppo_billiards_*_steps.zip"))
    
    # 同时也检查 final checkpoint
    # 注意：path 可能以 / 结尾，导致 dirname 结果不同
    # 使用 normpath 去掉末尾的 /
    norm_path = os.path.normpath(path)
    parent_dir = os.path.dirname(norm_path) # ./train
    
    final_checkpoint = os.path.join(parent_dir, "ppo_billiards_final.zip") # train/ppo_billiards_final.zip
    if os.path.exists(final_checkpoint):
        checkpoints.append(final_checkpoint)
    
    # 还有可能是直接在 checkpoint_dir 下的 final (虽然代码里是保存在 train/ 下，但以防万一)
    final_checkpoint_2 = os.path.join(path, "ppo_billiards_final.zip")
    if os.path.exists(final_checkpoint_2) and final_checkpoint_2 not in checkpoints:
        checkpoints.append(final_checkpoint_2)

    if not checkpoints:
        return None

    # 按修改时间排序，取最新的
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def cleanup_logs(log_dir):
    """清除旧的 monitor.csv 文件，只保留最新的训练记录"""
    print(f"正在清理 {log_dir} 下的旧 monitor 日志...")
    files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"删除失败 {f}: {e}")
    print("旧日志清理完成。")

def save_final_model(model, path):
    """保存模型并自动备份旧文件"""
    # path example: "train/ppo_billiards_final"
    zip_path = path + ".zip" if not path.endswith(".zip") else path
    save_base = path[:-4] if path.endswith(".zip") else path
    
    if os.path.exists(zip_path):
        try:
            timestamp = int(time.time())
            backup_path = f"{save_base}_backup_{timestamp}.zip"
            shutil.copy2(zip_path, backup_path)
            print(f"旧模型已备份至: {backup_path}")
        except Exception as e:
            print(f"备份失败: {e}")
            
    model.save(save_base)
    print(f"模型已保存至 {zip_path}")

def make_env(rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = BilliardsGymEnv()
        # 使用时间戳+rank创建唯一的 monitor 文件名
        monitor_path = os.path.join(log_dir, f"monitor_{int(time.time())}_{rank}")
        env = Monitor(env, monitor_path)
        env.reset(seed=seed + rank)
        return env
    return _init

def train(total_timesteps=1000000, save_freq=10000, n_envs=1):
    # 0. 准备日志目录
    log_dir = "./train/logs/"
    checkpoint_dir = "./train/checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 清理旧日志 (只保留本次训练记录)
    # cleanup_logs(log_dir)  # 注释掉此行以保留历史日志，确保绘图连续

    # 生成本次运行的唯一标识 (用于 TensorBoard)
    timestamp = int(time.time())
    run_name = f"PPO_run_{timestamp}"
    print(f"本次训练 TensorBoard Run Name: {run_name}")

    # 1. 创建环境
    print(f"正在创建 {n_envs} 个并行环境...")
    if n_envs > 1:
        # 使用多进程并行环境 (SubprocVecEnv)
        # start_method='spawn' 是 Windows 下必须的，但 SubprocVecEnv 默认会处理
        env = SubprocVecEnv([make_env(i, log_dir) for i in range(n_envs)])
    else:
        # 单进程
        env = BilliardsGymEnv()
        monitor_path = os.path.join(log_dir, f"monitor_{timestamp}")
        env = Monitor(env, monitor_path)
    
    # 2. 尝试恢复训练或定义新模型
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"发现 Checkpoint: {latest_checkpoint}，正在恢复训练...")
        # 加载模型，需要传入 env 和 tensorboard_log 以便继续记录
        # 注意: PPO.load 会覆盖掉这里传入的 tensorboard_log 和 tb_log_name，
        # 除非我们在 load 后手动修改或者使用 custom_objects
        model = PPO.load(latest_checkpoint, env=env)
        
        # 关键修正：手动更新 TensorBoard 记录位置和名称
        model.tensorboard_log = log_dir
        model.tb_log_name = run_name
        
        # 重新初始化 Logger，使其使用新的路径
        # Stable-Baselines3 不会自动因为修改了属性就重置 logger，需要手动配置
        from stable_baselines3.common.logger import configure
        new_logger = configure(os.path.join(log_dir, run_name), ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        print(f"已加载模型，当前训练步数: {model.num_timesteps}")
    else:
        print("未发现 Checkpoint，开始新的训练...")
        # 定义模型 (PPO)
        # 使用 MlpPolicy 因为输入是向量
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048 // n_envs if n_envs > 1 else 2048, # 调整每个环境的步数，保持总 buffer 大小大致不变
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, # 增加探索
            tensorboard_log=log_dir
        )
    
    # 3. 设置回调函数
    # Checkpoint 回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1), # 调整保存频率以适应多环境 (save_freq 是每个环境的步数)
        save_path='./train/checkpoints/',
        name_prefix='ppo_billiards'
    )
    
    # 可视化回调：实时绘制 Reward 曲线
    visualizer_callback = TrainingVisualizerCallback(log_dir=log_dir)
    
    # 组合回调
    callbacks = CallbackList([checkpoint_callback, visualizer_callback])
    
    # 4. 开始训练
    print("开始训练 PPO Agent...")
    print(f"训练日志和图像将保存在: {log_dir}")
    
    target_timesteps = total_timesteps
    current_timesteps = model.num_timesteps
    remaining_timesteps = max(0, target_timesteps - current_timesteps)
    
    print(f"目标总步数: {target_timesteps}")
    print(f"当前已训练步数: {current_timesteps}")
    print(f"本次将训练步数: {remaining_timesteps}")
    
    if remaining_timesteps > 0:
        try:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=run_name
            )
        except KeyboardInterrupt:
            print("\n收到中断信号 (Ctrl+C)，正在保存模型进度...")
            try:
                save_final_model(model, "train/ppo_billiards_final")
                print("模型进度保存成功！下次训练将自动加载此进度。")
                # 中断时也尝试绘制最终曲线
                print("正在绘制最终训练曲线...")
                visualizer_callback.plot_final_losses(log_dir)
            except Exception as e:
                print(f"保存模型或绘图时出错: {e}")
            
            env.close()
            print("环境已关闭。训练结束。")
            return
    else:
        print("训练已达到或超过目标步数，如需继续训练请增加 target_timesteps。")
    
    # 5. 训练结束，绘制最终的详细曲线
    visualizer_callback.plot_final_losses(log_dir)
    
    # 6. 保存最终模型
    save_final_model(model, "train/ppo_billiards_final")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="训练总步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="保存模型的频率(步数)")
    parser.add_argument("--n_envs", type=int, default=4, help="并行环境数量 (建议设置为 CPU 核心数)")
    args = parser.parse_args()

    # 确保目录存在
    os.makedirs("./train/logs/", exist_ok=True)
    os.makedirs("./train/checkpoints/", exist_ok=True)
    
    train(total_timesteps=args.total_timesteps, save_freq=args.save_freq, n_envs=args.n_envs)