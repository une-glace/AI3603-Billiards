import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal

from .agent import Agent
from .basic_agent_pro import analyze_shot_for_reward


class NewAgent(Agent):
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        self.candidate_limit = 40
        self.top_k = 6

    def _compute_easy_shots(self, shot, player_targets):
        balls = shot.balls
        cue = balls.get('cue')
        if cue is None:
            return 0
        cue_pos = cue.state.rvw[0]
        pockets = shot.table.pockets
        easy = 0
        for tid in player_targets:
            ball = balls.get(tid)
            if ball is None:
                continue
            if ball.state.s == 4:
                continue
            tpos = ball.state.rvw[0]
            for pid, pocket in pockets.items():
                ppos = pocket.center
                v_tp = ppos - tpos
                v_tp[2] = 0
                d_tp = np.linalg.norm(v_tp)
                if d_tp < 1e-4:
                    continue
                v_ct = tpos - cue_pos
                v_ct[2] = 0
                d_ct = np.linalg.norm(v_ct)
                if d_ct < 1e-4:
                    continue
                dot = np.dot(v_ct, v_tp)
                cosang = dot / (d_ct * d_tp)
                cosang = np.clip(cosang, -1.0, 1.0)
                ang = np.degrees(np.arccos(cosang))
                if ang < 35.0 and d_ct < 1.8 and d_tp < 1.2:
                    easy += 1
                    break
        return easy

    def _positional_bonus(self, shot, my_targets):
        balls = shot.balls
        remaining_mine = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        all_ids = [bid for bid in balls.keys() if bid not in ['cue']]
        opp_targets = [bid for bid in all_ids if bid not in remaining_mine and bid != '8']
        my_easy = self._compute_easy_shots(shot, remaining_mine)
        opp_easy = self._compute_easy_shots(shot, opp_targets)
        return 20.0 * (my_easy - opp_easy)

    def _is_path_clear(self, start_pos, end_pos, balls, ignore_ids):
        p0 = np.array(start_pos, dtype=float).copy()
        p1 = np.array(end_pos, dtype=float).copy()
        p0[2] = 0.0
        p1[2] = 0.0
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-4:
            return True
        for bid, ball in balls.items():
            if bid in ignore_ids:
                continue
            if ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0], dtype=float).copy()
            pos[2] = 0.0
            v = pos - p0
            t = np.dot(v, seg) / (seg_len * seg_len)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            proj = p0 + t * seg
            dist = np.linalg.norm(pos - proj)
            if dist < 2.0 * self.ball_radius * 0.98:
                return False
        return True

    def _simulate_deterministic(self, balls, table, base_params, last_state_snapshot, my_targets):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        cue.set_state(**base_params)
        try:
            pt.simulate(shot, inplace=True)
            score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            score += 0.7 * self._positional_bonus(shot, my_targets)
        except Exception:
            score = -1000.0
        return score

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None:
            return self._random_action()
        valid_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not valid_targets:
            valid_targets = ['8']
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        candidates = []
        candidate_limit = self.candidate_limit
        for target_id in valid_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                to_pocket_vec = pocket_pos - target_pos
                to_pocket_vec[2] = 0
                dist_to_pocket = np.linalg.norm(to_pocket_vec)
                if dist_to_pocket < 1e-4:
                    continue
                dir_to_pocket = to_pocket_vec / dist_to_pocket
                ghost_pos = target_pos - dir_to_pocket * (2 * self.ball_radius)
                if not self._is_path_clear(cue_pos, ghost_pos, balls, ignore_ids={'cue', target_id}):
                    continue
                if not self._is_path_clear(target_pos, pocket_pos, balls, ignore_ids={'cue', target_id}):
                    continue
                cue_to_ghost_vec = ghost_pos - cue_pos
                cue_to_ghost_vec[2] = 0
                dist_cue_to_ghost = np.linalg.norm(cue_to_ghost_vec)
                if dist_cue_to_ghost < 1e-4:
                    continue
                v_base = 1.2 + dist_cue_to_ghost * 1.4
                v_base = float(np.clip(v_base, 1.4, 7.5))
                dot_prod = np.dot(cue_to_ghost_vec, to_pocket_vec)
                cos_theta = dot_prod / (dist_cue_to_ghost * dist_to_pocket)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                cut_angle_rad = np.arccos(cos_theta)
                cut_angle_deg = np.degrees(cut_angle_rad)
                if abs(cut_angle_deg) > 70:
                    continue
                geom_score = cos_theta * 8.0 - 0.6 * dist_cue_to_ghost - 0.9 * dist_to_pocket
                phi_rad = np.arctan2(cue_to_ghost_vec[1], cue_to_ghost_vec[0])
                phi_deg = np.degrees(phi_rad) % 360
                speed_list = [v_base - 0.5, v_base, v_base + 0.7]
                angle_list = [phi_deg - 0.4, phi_deg, phi_deg + 0.4]
                for v in speed_list:
                    v_clipped = float(np.clip(v, 1.0, 8.0))
                    for phi_candidate in angle_list:
                        phi_norm = float(phi_candidate % 360)
                        params = {
                            'V0': v_clipped,
                            'phi': phi_norm,
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0
                        }
                        candidates.append((geom_score, params))
        if len(candidates) > candidate_limit:
            random.shuffle(candidates)
            candidates = candidates[:candidate_limit]
        if not candidates:
            return self._random_action()
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = min(self.top_k, len(candidates))
        best_score = -float('inf')
        best_action = None
        for i in range(top_k):
            _, params = candidates[i]
            s = self._simulate_deterministic(balls, table, params, last_state_snapshot, my_targets)
            if s > best_score:
                best_score = s
                best_action = params
        if best_action is None:
            return self._random_action()
        return best_action
