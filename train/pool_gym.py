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
        self.total_steps = 0
        self._base_noise = dict(self.env.noise_std)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.total_steps < 200000:
            target_type = 'solid'
        elif self.total_steps < 400000:
            target_type = 'stripe'
        else:
            target_type = np.random.choice(['solid', 'stripe'])
        self.env.reset(target_ball=target_type)
        
        self.current_step = 0
        self._apply_noise_schedule()
        
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
        # 记录当前击球的玩家，用于后续判断胜负归属
        shooting_player = self.env.players[self.env.curr_player]
        prev_targets = self.env.player_targets[shooting_player]
        
        # 执行动作
        try:
            step_info = self.env.take_shot(phys_action)
        except Exception as e:
            print(f"Error in env.take_shot: {e}")
            # 发生严重错误时，返回惩罚并结束当前 episode
            # 为了防止 observation 出错，重新获取一次（如果可能）或返回全0/最后一次的obs
            try:
                obs = self._get_obs()
            except:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # 给予大惩罚
            return obs, -100.0, True, False, {"error": str(e)}

        self.current_step += 1
        self.total_steps += 1
        self._apply_noise_schedule()
        
        # 获取新的状态
        done, info = self.env.get_done()
        obs = self._get_obs()
        
        # 计算奖励
        # 传递 phys_action 用于计算基于角度的引导奖励
        reward = self._compute_reward(step_info, prev_balls, prev_targets, done, info, shooting_player, phys_action)
        
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
        terminated = done or (self.env.get_curr_player() != shooting_player)
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        balls, targets, table = self.env.get_observation()
        
        # 更新当前目标类型（用于构建 obs，同时也更新类属性）
        player = self.env.get_curr_player()
        current_targets = self.env.player_targets[player]
        
        # 使用 utils 中的统一方法构建观测向量
        obs = utils.get_obs_vector(balls, current_targets)
        return obs

    def _compute_reward(self, step_info, prev_balls, prev_targets, done, info, shooting_player, phys_action=None):
        reward = 0.0
        
        # 1. 进球奖励
        my_pocketed = step_info.get('ME_INTO_POCKET', [])
        reward += len(my_pocketed) * 10.0 # 进一个目标球 +10
        enemy_pocketed = step_info.get('ENEMY_INTO_POCKET', [])
        if len(enemy_pocketed) > 0:
            reward -= 3.0 * len(enemy_pocketed)
        
        # 2. 赢/输奖励
        if done:
            if info['winner'] == shooting_player:
                reward += 100.0
            elif info['winner'] != 'SAME' and info['winner'] is not None:
                reward -= 50.0 # 输了
        
        # 3. 犯规惩罚
        if step_info.get('FOUL_FIRST_HIT', False):
            reward -= 5.0
        if step_info.get('NO_POCKET_NO_RAIL', False):
            penalty = 1.0 + min(1.0, self.total_steps / 300000.0)
            reward -= penalty
        if step_info.get('NO_HIT', False):
            reward -= 5.0
        if step_info.get('WHITE_BALL_INTO_POCKET', False):
            reward -= 10.0
            
        # 4. 稠密奖励 (Reward Shaping)
        
        # 4.1 击中奖励：鼓励 Agent 至少打中球
        if not step_info.get('NO_HIT', False) and not step_info.get('FOUL_FIRST_HIT', False):
            reward += 4.0
            
        # 4.2 基于角度的引导奖励 (Angle Guidance)
        # 如果没打中 (NO_HIT)，根据瞄准角度与最近目标球的偏差给予反馈
        # 这样即使没打中，只要方向对了，也能得到比完全乱打更好的分数
        if phys_action is not None:
            cue_pos = prev_balls['cue'].state.rvw[0]
            min_angle_diff = 180.0
            
            # 计算白球到每个目标球的角度
            for tid in prev_targets:
                if tid in prev_balls and prev_balls[tid].state.s != 4: # 未进袋
                    target_pos = prev_balls[tid].state.rvw[0]
                    dx = target_pos[0] - cue_pos[0]
                    dy = target_pos[1] - cue_pos[1]
                    
                    # 理想角度 (度数)
                    ideal_phi = math.degrees(math.atan2(dy, dx))
                    if ideal_phi < 0:
                        ideal_phi += 360.0
                        
                    # 计算与动作角度的差值 (考虑 0/360 周期性)
                    action_phi = phys_action['phi']
                    diff = abs(action_phi - ideal_phi)
                    if diff > 180.0:
                        diff = 360.0 - diff
                        
                    if diff < min_angle_diff:
                        min_angle_diff = diff
            factor = 2.0 if step_info.get('NO_HIT', False) else 1.0
            guidance_bonus = (180.0 - min_angle_diff) / 180.0 * factor
            if guidance_bonus > 0:
                reward += guidance_bonus
        
        pockets = list(self.env.table.pockets.values())
        def _min_dist_to_pocket(pos):
            return min(np.linalg.norm(p.center - pos) for p in pockets)
        progress = 0.0
        for tid in prev_targets:
            if tid in prev_balls:
                before = _min_dist_to_pocket(prev_balls[tid].state.rvw[0])
                after_ball = step_info.get('BALLS', {}).get(tid)
                if after_ball is not None and after_ball.state.s != 4:
                    after = _min_dist_to_pocket(after_ball.state.rvw[0])
                    delta = before - after
                    if delta > 0:
                        progress += delta
        if progress > 0:
            reward += min(5.0, 3.0 * progress)
            
        # 如果没进球，给予微小惩罚，鼓励尽快完成
        if len(my_pocketed) == 0:
            reward -= 0.02
            if not step_info.get('NO_POCKET_NO_RAIL', False):
                reward += 0.5

        still_turn = (not done) and (self.env.get_curr_player() == shooting_player)
        if still_turn:
            reward += 1.5
            
        return reward

    def _apply_noise_schedule(self):
        t = self.total_steps
        f = 0.3 + 0.7 * min(1.0, t / 500000.0)
        for k in self._base_noise:
            self.env.noise_std[k] = self._base_noise[k] * f
