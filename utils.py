import numpy as np

def get_obs_vector(balls, my_targets):
    """
    构建统一的观测向量，供 RL 模型和 Agent 使用。
    
    参数：
        balls: dict, 球的状态字典 {ball_id: Ball对象}
        my_targets: list, 当前玩家的目标球列表 ['1', '2', ...]
        
    返回：
        np.array: shape=(35,), dtype=np.float32
    """
    ball_pos = []
    # 固定的球ID顺序: cue, 8, 1-7, 9-15
    ball_ids = ['cue', '8'] + [str(i) for i in range(1, 8)] + [str(i) for i in range(9, 16)]
    
    for bid in ball_ids:
        if bid in balls and balls[bid].state.s != 4: # 4 代表 pocketed (进袋)
            pos = balls[bid].state.rvw[0] # rvw[0] 是位置 (x, y, z)
            ball_pos.extend([pos[0], pos[1]])
        else:
            # 进袋或不存在，用特殊值表示，例如 (-10, -10)
            ball_pos.extend([-10.0, -10.0])
            
    # 目标类型 one-hot
    target_vec = [0, 0, 0]
    if my_targets == ['8']:
        target_vec[2] = 1
    elif '1' in my_targets: # solid (1-7)
        target_vec[0] = 1
    else: # stripe (9-15)
        target_vec[1] = 1
        
    return np.array(ball_pos + target_vec, dtype=np.float32)
