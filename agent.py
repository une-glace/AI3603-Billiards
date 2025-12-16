"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import joblib
import utils_ai
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

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
            
        if len(remaining_own_before) > 0:
            if first_contact_ball_id in opponent_plus_eight:
                foul_first_hit = True
        else:
            if first_contact_ball_id != '8':
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
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
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
            def reward_fn_wrapper(V0, phi, theta, a, b):
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
    """Contextual Q Regression Agent"""
    
    def __init__(self, model_path='eval/checkpoints/cqr.joblib'):
        self.model = None
        self.scaler = None
        try:
            if os.path.exists(model_path):
                print(f"[NewAgent] Loading model from {model_path}")
                checkpoint = joblib.load(model_path)
                self.model = checkpoint['model']
                self.scaler = checkpoint['scaler']
            else:
                print(f"[NewAgent] Warning: Model file {model_path} not found. Running in heuristic mode.")
        except Exception as e:
            print(f"[NewAgent] Error loading model: {e}")

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()

        try:
            # 1. Identify targets
            legal_targets = utils_ai.get_legal_target_balls(balls, my_targets)
            
            cue_ball = balls['cue']
            cue_pos = cue_ball.state.rvw[0]
            radius = utils_ai.get_ball_radius(balls)

            # 2. Generate candidates
            candidates = []
            
            for tid in legal_targets:
                target_ball = balls[tid]
                target_pos = target_ball.state.rvw[0]
                
                for pid, pocket in table.pockets.items():
                    pocket_pos = pocket.center
                    
                    # Ghost Ball
                    ghost_pos = utils_ai.calculate_ghost_ball_pos(target_pos, pocket_pos, radius)
                    
                    # Aiming
                    v_aim = ghost_pos - cue_pos
                    dist = np.linalg.norm(v_aim)
                    if dist < 1e-6: continue
                    
                    aim_dir = v_aim / dist
                    phi = np.degrees(np.arctan2(aim_dir[1], aim_dir[0])) % 360
                    
                    # Cut angle check
                    v_shot_line = pocket_pos - target_pos
                    if np.linalg.norm(v_shot_line) < 1e-6: continue
                    shot_dir = v_shot_line / np.linalg.norm(v_shot_line)
                    cos_cut = np.dot(aim_dir, shot_dir)
                    if cos_cut < 0: continue
                    
                    # Clearance
                    is_clear_cg = utils_ai.is_path_clear(cue_pos, ghost_pos, balls, ['cue', tid], radius)
                    is_clear_tp = utils_ai.is_path_clear(target_pos, pocket_pos, balls, ['cue', tid], radius)
                    
                    if not is_clear_tp: continue

                    # Variations
                    V0s = [1.5, 2.5, 3.5, 4.5, 6.0]
                    Spins = [(0.0, 0.0), (0.0, 0.2), (0.0, -0.2), (0.0, 0.4), (0.0, -0.4)]
                    
                    for v0 in V0s:
                        for a, b in Spins:
                            action = {'V0': v0, 'phi': phi, 'theta': 0.0, 'a': a, 'b': b}
                            
                            feats = utils_ai.get_features(
                                cue_pos, target_pos, pocket_pos, ghost_pos, balls, table, radius,
                                V0_norm=v0/8.0, clear_cg=is_clear_cg, clear_tp=is_clear_tp,
                                a=a, b=b
                            )
                            candidates.append((action, feats))

            if not candidates:
                # If no clear shots, maybe try a random shot or safety?
                return self._random_action()
            
            # 3. Score candidates with Model
            if self.model:
                X_cand = np.array([c[1] for c in candidates])
                X_cand_scaled = self.scaler.transform(X_cand)
                scores = self.model.predict(X_cand_scaled)
            else:
                # Fallback heuristic if no model
                scores = []
                for c in candidates:
                    feats = c[1]
                    # feats: [cos_cut, d_cg, d_tp, clear_cg, clear_tp, v0]
                    # Simple heuristic: maximize cut_angle (closer to 1), minimize distance
                    score = feats[0] * 10 - feats[1] - feats[2] + feats[3]*5 + feats[4]*5
                    scores.append(score)
                scores = np.array(scores)

            # 4. Top-K Selection
            K = 32 # Check more candidates
            top_indices = np.argsort(scores)[-K:]
            # Reverse to have best first
            top_indices = top_indices[::-1]
            top_candidates = [candidates[i] for i in top_indices]
            
            # 5. Simulation Validation (Refinement)
            best_action = None
            best_sim_score = -9999
            
            # Snapshot for simulation
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            print(f"[NewAgent] Evaluating {len(top_candidates)} candidates...")
            
            for action, _ in top_candidates:
                # Simulate
                sim_table = copy.deepcopy(table)
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_cue = pt.Cue(cue_ball_id="cue")
                sim_cue.set_state(**action)
                
                shot = pt.System(table=sim_table, balls=sim_balls, cue=sim_cue)
                try:
                    pt.simulate(shot, inplace=True)
                    rew = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                except:
                    rew = -500
                
                # If we find a very good shot (e.g. pot ball + no foul), maybe stop early?
                # A good score is > 50 (pot own ball)
                # But we want the BEST shot.
                
                if rew > best_sim_score:
                    best_sim_score = rew
                    best_action = action
            
            if best_action is None:
                return self._random_action()
                
            print(f"[NewAgent] Selected action: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f} (Sim Score: {best_sim_score})")
            return best_action

        except Exception as e:
            print(f"[NewAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
