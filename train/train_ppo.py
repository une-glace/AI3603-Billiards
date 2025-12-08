import os
import sys
import glob
import time
import shutil
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.pool_gym import BilliardsGymEnv
from train.visualize_callback import TrainingVisualizerCallback
from eval.eval_script import evaluate

class PerformanceEvalCallback(BaseCallback):
    def __init__(self, eval_freq, n_games, checkpoint_dir, target_win_rate=0.7, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.checkpoint_dir = checkpoint_dir
        self.target_win_rate = target_win_rate
        self._last_eval_step = 0
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            latest = get_latest_checkpoint(self.checkpoint_dir)
            if latest:
                try:
                    results = evaluate(n_games=self.n_games, model_path=latest)
                    total = max(1, sum(results.values()))
                    win_rate_b = results.get('AGENT_B_WIN', 0) / total
                    if self.verbose:
                        print(f"周期评估: AGENT_B_WIN={results.get('AGENT_B_WIN',0)}, 总局数={total}, 胜率={win_rate_b*100:.1f}%")
                    if win_rate_b >= self.target_win_rate:
                        try:
                            save_final_model(self.model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_billiards_final"), self.training_env)
                        except Exception:
                            pass
                        return False
                except Exception as e:
                    print(f"评估失败: {e}")
        return True

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

def save_final_model(model, path, env=None):
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
            
            # 同时也备份 vecnormalize
            vec_path = f"{save_base}_vecnormalize.pkl"
            if os.path.exists(vec_path):
                shutil.copy2(vec_path, f"{save_base}_vecnormalize_backup_{timestamp}.pkl")
        except Exception as e:
            print(f"备份失败: {e}")
            
    model.save(save_base)
    print(f"模型已保存至 {zip_path}")
    
    # 保存 VecNormalize 统计数据
    if env is not None:
        # 如果 env 被其他 wrapper 包裹，需要解包找到 VecNormalize
        # 但通常 env 就是 VecNormalize (或者 VecNormalize 也是 Wrapper)
        # 尝试直接保存
        try:
            if isinstance(env, VecNormalize):
                env.save(f"{save_base}_vecnormalize.pkl")
                print(f"Normalization stats saved to {save_base}_vecnormalize.pkl")
            # 处理 env 是 Monitor 或其他 Wrapper 的情况 (如果 VecNormalize 在里面)
            elif hasattr(env, 'venv') and isinstance(env.venv, VecNormalize):
                 env.venv.save(f"{save_base}_vecnormalize.pkl")
                 print(f"Normalization stats saved (from venv) to {save_base}_vecnormalize.pkl")
            # 还有可能是 get_wrapper_attr
            elif hasattr(env, "get_wrapper_attr"):
                try:
                    norm_env = env.get_wrapper_attr("save")
                    # 这比较难判断是不是 VecNormalize 的 save，简单起见只处理最外层
                except:
                    pass
        except Exception as e:
            print(f"保存 VecNormalize 失败: {e}")

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

def train(total_timesteps=1000000, save_freq=10000, n_envs=1, eval_freq=100000, eval_games=50, target_win_rate=0.7):
    # 0. 准备日志目录 (使用绝对路径以适应不同运行位置)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    
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
        env = SubprocVecEnv([make_env(i, log_dir) for i in range(n_envs)])
    else:
        # 为了使用 VecNormalize，单环境也需要包装成 VecEnv
        env = DummyVecEnv([make_env(0, log_dir)])
    
    # 2. 尝试恢复训练或定义新模型
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    # 处理 VecNormalize
    # 如果存在 checkpoint，尝试寻找对应的 vecnormalize.pkl
    vec_norm_path = None
    if latest_checkpoint:
        # 尝试: ppo_billiards_10000_steps_vecnormalize.pkl
        base_name = os.path.splitext(latest_checkpoint)[0]
        possible_path = base_name + "_vecnormalize.pkl"
        if os.path.exists(possible_path):
            vec_norm_path = possible_path
        else:
             # 尝试 final
             final_path = os.path.join(checkpoint_dir, "ppo_billiards_final_vecnormalize.pkl")
             if os.path.exists(final_path):
                 vec_norm_path = final_path

    if vec_norm_path:
        print(f"正在加载 VecNormalize 统计: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        print("初始化新的 VecNormalize Wrapper...")
        # norm_obs=True: 标准化观测
        # norm_reward=True: 标准化奖励 (这对 PPO 很重要)
        # clip_obs=10.: 截断观测值
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

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
        def lr_schedule(progress_remaining: float):
            return 4e-4 * max(0.2, progress_remaining)
        def clip_schedule(progress_remaining: float):
            return 0.2 * max(0.5, progress_remaining)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=2048 // n_envs if n_envs > 1 else 2048, # 调整每个环境的步数，保持总 buffer 大小大致不变
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_schedule,
            ent_coef=0.02,
            tensorboard_log=log_dir,
            policy_kwargs={"net_arch": [512, 512]},
            use_sde=True,
            sde_sample_freq=4
        )
    
    # 3. 设置回调函数
    # Checkpoint 回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1), # 调整保存频率以适应多环境 (save_freq 是每个环境的步数)
        save_path=checkpoint_dir,
        name_prefix='ppo_billiards',
        save_vecnormalize=True
    )
    
    # 可视化回调：实时绘制 Reward 曲线
    visualizer_callback = TrainingVisualizerCallback(log_dir=log_dir)
    
    # 组合回调
    perf_eval_cb = PerformanceEvalCallback(eval_freq=eval_freq, n_games=eval_games, checkpoint_dir=checkpoint_dir, target_win_rate=target_win_rate)
    callbacks = CallbackList([checkpoint_callback, visualizer_callback, perf_eval_cb])
    
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
                save_final_model(model, os.path.join(current_dir, "ppo_billiards_final"), env)
                print("模型进度保存成功！下次训练将自动加载此进度。")
                # 中断时也尝试绘制最终曲线
                print("正在绘制最终训练曲线...")
                visualizer_callback.plot_final_losses(log_dir)
            except Exception as e:
                print(f"保存模型或绘图时出错: {e}")
            
            env.close()
            print("环境已关闭。训练结束。")
            return
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            print("尝试保存紧急备份...")
            try:
                save_final_model(model, os.path.join(current_dir, "ppo_billiards_crash_backup"), env)
            except Exception as save_err:
                print(f"紧急保存失败: {save_err}")
            env.close()
            raise e
    else:
        print("训练已达到或超过目标步数，如需继续训练请增加 target_timesteps。")
    
    # 5. 训练结束，绘制最终的详细曲线
    visualizer_callback.plot_final_losses(log_dir)
    
    # 6. 保存最终模型
    save_final_model(model, os.path.join(current_dir, "ppo_billiards_final"), env)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="训练总步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="保存模型的频率(步数)")
    parser.add_argument("--n_envs", type=int, default=4, help="并行环境数量 (建议设置为 CPU 核心数)")
    parser.add_argument("--eval_freq", type=int, default=100000, help="评估频率(步)")
    parser.add_argument("--eval_games", type=int, default=50, help="每次评估局数")
    parser.add_argument("--target_win_rate", type=float, default=0.7, help="达到该胜率则早停")
    args = parser.parse_args()

    # 确保目录存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "checkpoints"), exist_ok=True)
    
    train(total_timesteps=args.total_timesteps, save_freq=args.save_freq, n_envs=args.n_envs, eval_freq=args.eval_freq, eval_games=args.eval_games, target_win_rate=args.target_win_rate)
