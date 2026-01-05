import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .agent import Agent

class NewAgent(Agent):
    """基于几何计算与搜索的改进 Agent"""
    
    def __init__(self):
        super().__init__()
        self.V0_list = [2.0, 4.5] # 减少速度候选项，去掉过小或过大的
        print("NewAgent (Geometric Planning + Simulation Search) 已初始化。")
        
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            balls: 球状态字典
            my_targets: 目标球列表
            table: 球桌对象
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None or table is None:
            return self._random_action()
        
        # 1. 确定实际目标球
        # 过滤掉已经进袋的球
        valid_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果目标球全进了，就打黑8
        if not valid_targets:
            valid_targets = ['8']
            
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0] # [x, y, z]
        
        candidates = [] # 存储候选击球参数 (V0, phi)
        
        # 2. 几何分析生成候选动作
        for target_id in valid_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 遍历所有球袋
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # --- 幽灵球计算 ---
                # 目标球到袋口的向量
                to_pocket_vec = pocket_pos - target_pos
                # 忽略Z轴（只看平面）
                to_pocket_vec[2] = 0
                dist_to_pocket = np.linalg.norm(to_pocket_vec)
                
                if dist_to_pocket < 1e-4:
                    continue
                    
                # 单位方向向量 (目标球 -> 袋口)
                dir_to_pocket = to_pocket_vec / dist_to_pocket
                
                # 幽灵球位置：目标球位置沿反方向延伸 2*半径 (约0.057m)
                BALL_RADIUS = 0.028575
                ghost_pos = target_pos - dir_to_pocket * (2 * BALL_RADIUS)
                
                # --- 计算母球击打角度 ---
                cue_to_ghost_vec = ghost_pos - cue_pos
                cue_to_ghost_vec[2] = 0
                dist_cue_to_ghost = np.linalg.norm(cue_to_ghost_vec)
                
                if dist_cue_to_ghost < 1e-4:
                    continue
                
                # 计算切球角度 (Cut Angle)
                # 幽灵球向量 与 进袋向量 的夹角
                # 实际上就是 cue_to_ghost_vec 与 to_pocket_vec 的夹角
                dot_prod = np.dot(cue_to_ghost_vec, to_pocket_vec)
                cos_theta = dot_prod / (dist_cue_to_ghost * dist_to_pocket)
                # 限制范围防止数值误差
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                cut_angle_rad = np.arccos(cos_theta)
                cut_angle_deg = np.degrees(cut_angle_rad)
                
                # 如果切球角度过大 (>70度)，很难打进且模拟耗时，跳过（收紧阈值）
                if abs(cut_angle_deg) > 70:
                    continue
                    
                # 计算 phi (水平角度)
                phi_rad = np.arctan2(cue_to_ghost_vec[1], cue_to_ghost_vec[0])
                phi_deg = np.degrees(phi_rad)
                # 规范化到 [0, 360)
                phi_deg = phi_deg % 360
                
                # --- 添加候选项 ---
                # 减少每个几何解生成的候选数量
                # 只有当角度真的很正的时候才多尝试几种力度
                for v in self.V0_list:
                    candidates.append({'V0': v, 'phi': phi_deg})
                    # 只有在大力出奇迹或者角度比较刁钻时才微调角度，这里为了速度，去掉微调
                    # 或者只保留一个微调方向
                    # candidates.append({'V0': v, 'phi': (phi_deg + 0.5) % 360}) 
                    # candidates.append({'V0': v, 'phi': (phi_deg - 0.5) % 360})


        # 如果没有几何候选项（例如被完全遮挡或角度都不对），回退到随机
        if not candidates:
            return self._random_action()

        # 3. 模拟并评分
        best_score = -float('inf')
        best_action = None
        
        # 保存击球前状态快照
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        for cand in candidates:
            # 构建模拟环境
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            # 设置击球参数 (theta, a, b 设为默认值)
            # 可以在后续改进中优化 a,b (加塞)
            params = {
                'V0': cand['V0'],
                'phi': cand['phi'],
                'theta': 0.0,
                'a': 0.0,
                'b': 0.0
            }
            shot.cue.set_state(**params)
            
            try:
                pt.simulate(shot, inplace=True)
                score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            except:
                score = -1000
            
            if score > best_score:
                best_score = score
                best_action = params
        
        if best_action is None:
            return self._random_action()
            
        return best_action