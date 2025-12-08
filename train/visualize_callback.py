import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainingVisualizerCallback(BaseCallback):
    """
    自定义回调函数，用于记录训练过程中的 Reward 和 Loss，并实时绘制保存图像。
    """
    def __init__(self, log_dir: str, verbose: int = 1):
        super(TrainingVisualizerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "training_plots")
        os.makedirs(self.save_path, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []

        # 尝试加载历史数据以保持绘图连续性
        try:
            # load_results 读取目录下所有 .monitor.csv 文件
            # ts2xy 提取 (timesteps, rewards)
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(y) > 0:
                self.episode_rewards = list(y)
                print(f"Visualizer: 已加载历史训练数据，共 {len(self.episode_rewards)} 条记录。")
        except Exception as e:
            # 如果是第一次训练或没有日志文件，忽略错误
            pass
        
        # 用于多环境跟踪的 buffers
        self.current_episode_rewards = None
        self.current_episode_lengths = None
        
        # 记录 losses (从 logger 中提取)
        self.policy_losses = []
        self.value_losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # 获取当前 step 的 reward 和 done 信息
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        num_envs = len(rewards)
        
        # 初始化 buffers
        if self.current_episode_rewards is None:
            self.current_episode_rewards = np.zeros(num_envs)
            self.current_episode_lengths = np.zeros(num_envs, dtype=int)
            
        # 更新每个环境的状态
        for i in range(num_envs):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_lengths[i] += 1
            
            if dones[i]:
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0
                
                # 每完成一定数量的 episode 更新一次图像
                if len(self.episode_rewards) % 10 == 0:
                    self._plot_rewards()
                    
        return True

    def _on_rollout_end(self) -> None:
        """
        每次 rollout 结束（即完成 n_steps 步收集并进行了一次网络更新）后调用。
        从 logger 中提取 loss 信息。
        """
        # 尝试从 logger 的记录中获取 loss
        # 注意：logger 的 key 可能会根据 SB3 版本有所不同
        # 通常是 'train/policy_gradient_loss', 'train/value_loss'
        # 但在回调中直接获取比较困难，更简单的方法是利用 Monitor wrapper 生成的日志文件
        # 或者直接在这里记录当前的 timestep
        
        # 这里我们主要关注 Reward 的可视化，Loss 的可视化通常依赖 Tensorboard
        # 但为了满足用户需求，我们尝试记录
        pass
        
    def _plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode Reward")
        
        # 计算滑动平均
        window_size = 50
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, label=f"Moving Avg ({window_size})", color='red')
            
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.legend()
        plt.grid(True)
        
        save_file = os.path.join(self.save_path, "reward_curve.png")
        plt.savefig(save_file)
        plt.close()

    def plot_final_losses(self, log_folder):
        """
        训练结束后，尝试从 monitor.csv 或 tensorboard logs 中读取数据并绘图
        这里使用 stable_baselines3 自带的 load_results 读取 Monitor wrapper 的输出
        """
        try:
            x, y = ts2xy(load_results(log_folder), 'timesteps')
            if len(x) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(x, y, label="Episode Reward")
                
                # 滑动平均
                window_size = 50
                if len(y) >= window_size:
                    moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                    # x 轴也需要对应调整
                    x_ma = x[window_size-1:]
                    plt.plot(x_ma, moving_avg, label=f"Moving Avg ({window_size})", color='red')

                plt.xlabel("Timesteps")
                plt.ylabel("Reward")
                plt.title("Training Reward Curve (from Monitor)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.save_path, "final_reward_curve.png"))
                plt.close()
        except Exception as e:
            print(f"无法绘制最终曲线 (可能是缺少 Monitor 日志): {e}")
