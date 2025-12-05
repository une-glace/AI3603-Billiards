import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pooltool as pt
import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
import utils

class BilliardsGymEnv(gym.Env):
    """
    强化学习包装环境，适配 gymnasium 接口
    """
    def __init__(self, env_config=None):
        super(BilliardsGymEnv, self).__init__()
        
        self.env = PoolEnv()
        self.env.enable_noise = True # 训练时开启噪声以增强鲁棒性
        
        # 动作空间: [V0, phi, theta, a, b]
        # V0: [0.5, 8.0] -> 归一化到 [-1, 1]
        # phi: [0, 360] -> 归一化到 [-1, 1]
        # theta: [0, 90] -> 归一化到 [-1, 1]
        # a: [-0.5, 0.5] -> 归一化到 [-1, 1]
        # b: [-0.5, 0.5] -> 归一化到 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # 观测空间: 
        # 包含所有球的位置 (x, y) (z坐标通常为0或定值，忽略)
        # 16个球 * 2维坐标 = 32维
        # 加上当前目标球类型 (实心/花色/黑8) -> one-hot编码 (3维)
        # 总维度 = 35
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
        
        self.current_target_type = 0 # 0: solid, 1: stripe, 2: eight
        self.max_steps = 60
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机选择开球目标类型
        target_type = np.random.choice(['solid', 'stripe'])
        self.env.reset(target_ball=target_type)
        
        self.current_step = 0
        
        # 确定当前agent的目标类型
        player = self.env.get_curr_player()
        targets = self.env.player_targets[player]
        if targets == ['8']:
            self.current_target_type = 2
        elif '1' in targets:
            self.current_target_type = 0
        else:
            self.current_target_type = 1
            
        return self._get_obs(), {}

    def step(self, action):
        # 将归一化的动作映射回物理空间
        phys_action = {
            'V0': float((action[0] + 1) / 2 * (8.0 - 0.5) + 0.5),
            'phi': float((action[1] + 1) / 2 * 360.0),
            'theta': float((action[2] + 1) / 2 * 90.0),
            'a': float(action[3] * 0.5),
            'b': float(action[4] * 0.5)
        }
        
        # 记录击球前的状态用于计算奖励
        prev_balls = self.env.balls.copy()
        prev_targets = self.env.player_targets[self.env.get_curr_player()]
        
        # 执行动作
        step_info = self.env.take_shot(phys_action)
        self.current_step += 1
        
        # 获取新的状态
        done, info = self.env.get_done()
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._compute_reward(step_info, prev_balls, prev_targets, done, info)
        
        # 如果换人了（没打进或犯规），对当前RL agent来说这一轮step结束
        # 注意：在多智能体环境中，如果换人了，通常意味着当前agent的回合结束
        # 这里简化处理：如果换人，视为本次交互结束，给予相应奖励
        current_player = self.env.get_curr_player()
        # 简单的单智能体训练逻辑：如果换人了，或者游戏结束，则当前episode结束
        # 实际上应该训练两个agent或者自我博弈，这里简化为训练一个能连续进球的agent
        
        # 强制截断：如果步数超过限制
        truncated = self.current_step >= self.max_steps
        
        # 如果游戏结束或换人，则 terminated = True
        # 注意：这里我们训练的目标是“尽可能连续进球并赢下比赛”
        # 如果换人了，说明这一杆没打好，虽然游戏没结束，但对RL agent来说本次控制权丧失
        terminated = done
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        balls, targets, table = self.env.get_observation()
        
        # 更新当前目标类型（用于构建 obs，同时也更新类属性）
        player = self.env.get_curr_player()
        current_targets = self.env.player_targets[player]
        
        # 使用 utils 中的统一方法构建观测向量
        obs = utils.get_obs_vector(balls, current_targets)
        return obs

    def _compute_reward(self, step_info, prev_balls, prev_targets, done, info):
        reward = 0.0
        
        # 1. 进球奖励
        my_pocketed = step_info.get('ME_INTO_POCKET', [])
        reward += len(my_pocketed) * 10.0 # 进一个目标球 +10
        
        # 2. 赢/输奖励
        if done:
            if info['winner'] == 'SAME':
                reward += 0.0
            elif info['winner'] == self.env.get_curr_player(): # 如果赢的是当前玩家（注意：get_curr_player在done时可能已经切换，需核实）
                 # 这里逻辑比较绕，因为take_shot内部会切换player。
                 # 如果take_shot导致游戏结束，且赢家是“执行take_shot的那个player”，则加分
                 # 简单判断：如果是take_shot之前的那个player赢了
                 pass 
            
            # 简化逻辑：
            # 如果是我方打进了黑8且合法 -> 巨额奖励
            if '8' in my_pocketed:
                 # 需要判断是否合法黑8，这里假设env逻辑正确，如果是合法黑8，winner应该是当前玩家
                 # 实际上 poolenv 在 take_shot 内部处理了胜负逻辑
                 if info['winner'] != 'SAME' and info['winner'] is not None:
                     # 假设我们在训练 'A'，而 winner 是 'A'
                     # 由于 gym 环境并不区分 A/B，只关心当前策略的表现
                     # 如果赢了，给予大奖励
                     reward += 100.0
            elif step_info.get('WHITE_BALL_INTO_POCKET', False) and '8' in step_info.get('BALLS', {}):
                 # 白球进袋输了
                 reward -= 50.0
        
        # 3. 犯规惩罚
        if step_info.get('FOUL_FIRST_HIT', False):
            reward -= 5.0
        if step_info.get('NO_POCKET_NO_RAIL', False):
            reward -= 2.0
        if step_info.get('NO_HIT', False):
            reward -= 5.0
        if step_info.get('WHITE_BALL_INTO_POCKET', False):
            reward -= 10.0
            
        # 4. 稠密奖励：击球质量（母球位置控制）
        # 简化的稠密奖励：如果没进球，但母球距离任一目标球变近了？
        # 或者：如果没进球，给予微小惩罚，鼓励尽快进球
        if len(my_pocketed) == 0:
            reward -= 0.1
            
        return reward
