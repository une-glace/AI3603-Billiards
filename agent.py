"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 基于强化学习 (PPO) + 启发式搜索的改进实现
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
import pickle
import numpy as np
from datetime import datetime
import random
import utils

# 尝试导入 stable-baselines3，用于加载 RL 模型
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable-baselines3 not found. NewAgent will fallback to heuristic mode.")

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 5
        self.OPT_SEARCH = 2
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            sim_count = 0
            def reward_fn_wrapper(V0, phi, theta, a, b):
                nonlocal sim_count
                sim_count += 1
                if sim_count % 1 == 0:
                    print(f"  > [BasicAgent] 正在进行第 {sim_count} 次模拟...", end='\r')
                
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """
    基于强化学习 (PPO) + 启发式搜索的改进 Agent
    
    设计思路：
    1. 尝试加载预训练的 PPO 模型进行快速推理。
    2. 如果模型不可用，回退到基于几何的启发式搜索 + 少量物理模拟验证。
    3. 混合策略：RL 给出大致方向，通过模拟微调（Domain Randomization Robustness）。
    """
    
    def __init__(self, model_path="train/ppo_billiards_final.zip"):
        super().__init__()
        self.model = None
        self._norm = None
        if RL_AVAILABLE:
            try:
                # 尝试加载模型
                # 注意：实际路径可能需要根据运行位置调整
                abs_path = os.path.abspath(model_path)
                if os.path.exists(abs_path):
                    self.model = PPO.load(abs_path)
                    print(f"NewAgent (RL-PPO) 模型加载成功: {abs_path}")
                    base = abs_path[:-4] if abs_path.endswith('.zip') else abs_path
                    vec_path = base + "_vecnormalize.pkl"
                    if os.path.exists(vec_path):
                        try:
                            with open(vec_path, 'rb') as f:
                                data = pickle.load(f)
                            self._norm = {
                                'mean': np.array(data['obs_rms']['mean'], dtype=np.float32),
                                'var': np.array(data['obs_rms']['var'], dtype=np.float32),
                                'clip': float(data.get('clip_obs', 10.0))
                            }
                            print(f"VecNormalize 统计已加载: {vec_path}")
                        except Exception as e:
                            print(f"VecNormalize 加载失败: {e}")
                else:
                    print(f"NewAgent (RL-PPO) 警告: 模型文件不存在 {abs_path}，将使用启发式模式。")
            except Exception as e:
                print(f"NewAgent (RL-PPO) 模型加载失败: {e}")
        
        # 启发式搜索参数
        self.num_simulation = 20 # 每次决策进行的模拟次数
        print("NewAgent (RL + Simulation Search) 已初始化。")
        
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None:
            return self._random_action()
            
        # 策略 1: 如果有 RL 模型，优先询问模型
        rl_action = None
        if self.model:
            try:
                obs = utils.get_obs_vector(balls, my_targets)
                if self._norm is not None:
                    eps = 1e-8
                    obs = (obs - self._norm['mean']) / np.sqrt(self._norm['var'] + eps)
                    obs = np.clip(obs, -self._norm['clip'], self._norm['clip'])
                action_vec, _ = self.model.predict(obs, deterministic=True)
                
                # 映射动作回物理空间
                rl_action = {
                    'V0': float((action_vec[0] + 1) / 2 * (8.0 - 0.5) + 0.5),
                    'phi': float((action_vec[1] + 1) / 2 * 360.0),
                    'theta': float((action_vec[2] + 1) / 2 * 90.0),
                    'a': float(action_vec[3] * 0.5),
                    'b': float(action_vec[4] * 0.5)
                }
                
                # 如果完全信任 RL，可以直接返回
                # 但为了稳健性（Robustness），我们可以把 RL 的动作作为一个高质量的候选
                # 并与几何启发式的动作一起进行模拟验证
            except Exception as e:
                print(f"RL 推理出错: {e}")
        
        # 策略 2: 几何启发式生成候选动作 (Ghost Ball)
        candidates = []
        if rl_action:
            candidates.append(rl_action)
            
        # 确定有效目标球
        valid_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not valid_targets: valid_targets = ['8']
        
        cue_pos = balls['cue'].state.rvw[0]
        BALL_RADIUS = 0.028575
        
        for target_id in valid_targets:
            target_pos = balls[target_id].state.rvw[0]
            
            # 对每个袋口
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                
                # 计算幽灵球位置
                to_pocket = pocket_pos - target_pos
                to_pocket[2] = 0
                dist = np.linalg.norm(to_pocket)
                if dist < 1e-4: continue
                
                dir_to_pocket = to_pocket / dist
                ghost_pos = target_pos - dir_to_pocket * (2 * BALL_RADIUS)
                
                # 计算母球到幽灵球的击球角度
                aim_vec = ghost_pos - cue_pos
                aim_vec[2] = 0
                aim_dist = np.linalg.norm(aim_vec)
                if aim_dist < 1e-4: continue
                
                aim_angle_rad = math.atan2(aim_vec[1], aim_vec[0])
                aim_angle_deg = math.degrees(aim_angle_rad) % 360
                
                # 检查是否被阻挡 (简单的点积判断或距离判断，这里省略复杂检测)
                # 生成几个不同力度的候选
                for v in [2.0, 4.0, 6.5]:
                    candidates.append({
                        'V0': v,
                        'phi': aim_angle_deg,
                        'theta': 0.0, # 平射
                        'a': 0.0,
                        'b': 0.0
                    })
                    
        # 如果没有候选（比如都被挡住了），随机生成几个
        if not candidates:
            for _ in range(5):
                candidates.append(self._random_action())
                
        # 策略 3: 并行/串行 模拟验证 (Simulate & Select)
        # 从候选动作中选出得分最高的
        best_score = -float('inf')
        best_action = candidates[0]
        
        # 如果候选太多，随机采样一部分，保证实时性
        if len(candidates) > 20:
            candidates = random.sample(candidates, 20)
            if rl_action: candidates.append(rl_action) # 确保 RL 动作在列表里
            
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        for action in candidates:
            # 模拟一次 (不加噪声或加少量噪声)
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            shot.cue.set_state(V0=action['V0'], phi=action['phi'], theta=action['theta'], a=action['a'], b=action['b'])
            pt.simulate(shot, inplace=True)
            
            score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # 简单的位置启发式奖励：如果母球停在中心区域，加分
            final_cue = shot.balls['cue']
            if final_cue.state.s != 4: # 没进袋
                # 简单的中心倾向
                pass 
                
            if score > best_score:
                best_score = score
                best_action = action
                
        print(f"[NewAgent] 最佳模拟得分: {best_score:.1f}, 动作: V0={best_action['V0']:.1f}, phi={best_action['phi']:.1f}")
        return best_action
