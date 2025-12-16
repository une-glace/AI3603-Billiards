
import numpy as np
import pooltool as pt

def get_ball_radius(balls):
    # Try to get radius from one of the balls
    for b in balls.values():
        if hasattr(b, 'params') and hasattr(b.params, 'R'):
            return b.params.R
    return 0.028575 # Default fallback

def get_legal_target_balls(balls, my_targets):
    """Get list of target ball IDs that are still on table"""
    candidates = [bid for bid in my_targets if balls[bid].state.s != 4] # 4 is pocketed
    if not candidates:
        return ['8']
    return candidates

def calculate_ghost_ball_pos(target_pos, pocket_pos, radius):
    """Calculate ghost ball position for aiming"""
    vec = pocket_pos - target_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return target_pos
    direction = vec / dist
    ghost_pos = target_pos - (2 * radius) * direction
    return ghost_pos

def is_path_clear(start_pos, end_pos, balls, exclude_ids, radius, margin=0.0):
    """Check if path is clear of obstacles"""
    p1 = start_pos
    p2 = end_pos
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return True
    dir_v = vec / length
    
    for bid, ball in balls.items():
        if bid in exclude_ids:
            continue
        if ball.state.s == 4: # Pocketed
            continue
            
        b_pos = ball.state.rvw[0]
        
        v_b = b_pos - p1
        proj = np.dot(v_b, dir_v)
        
        if proj < 0 or proj > length:
            dist = min(np.linalg.norm(b_pos - p1), np.linalg.norm(b_pos - p2))
        else:
            closest = p1 + proj * dir_v
            dist = np.linalg.norm(b_pos - closest)
            
        if dist < (2 * radius + margin):
            return False
    return True

def get_features(cue_pos, target_pos, pocket_pos, ghost_pos, balls, table, radius, V0_norm, clear_cg, clear_tp, a=0.0, b=0.0):
    """
    Extract features for the shot.
    """
    v_cue_ghost = ghost_pos - cue_pos
    d_cue_ghost = np.linalg.norm(v_cue_ghost)
    
    v_target_pocket = pocket_pos - target_pos
    d_target_pocket = np.linalg.norm(v_target_pocket)
    
    if d_cue_ghost < 1e-6:
        cut_angle_cos = 1.0
    else:
        dir_cue_ghost = v_cue_ghost / d_cue_ghost
        if d_target_pocket < 1e-6:
            dir_target_pocket = np.zeros(3)
        else:
            dir_target_pocket = v_target_pocket / d_target_pocket
            
        cut_angle_cos = np.dot(dir_cue_ghost, dir_target_pocket)
    
    return [
        cut_angle_cos,          # 0
        d_cue_ghost,            # 1
        d_target_pocket,        # 2
        float(clear_cg),        # 3
        float(clear_tp),        # 4
        V0_norm,                # 5
        a,                      # 6
        b                       # 7
    ]
