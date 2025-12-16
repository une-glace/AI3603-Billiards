
import os
import sys
import argparse
import random
import numpy as np
import pooltool as pt
import copy
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poolenv import PoolEnv
from agent import analyze_shot_for_reward
import utils_ai

def collect_data(num_games=100, samples_per_obs=30, output_file='train/dataset.npz'):
    env = PoolEnv()
    env.enable_noise = True # Enable noise for robust data
    
    X_data = []
    y_data = []
    
    pbar = tqdm(total=num_games)
    
    for game_idx in range(num_games):
        # Randomize start
        target_type = random.choice(['solid', 'stripe'])
        env.reset(target_ball=target_type)
        
        while not env.done:
            # Get observation
            balls, my_targets, table = env.get_observation()
            player = env.get_curr_player()
            
            # Identify target balls
            legal_targets = utils_ai.get_legal_target_balls(balls, my_targets)
            
            cue_ball = balls['cue']
            cue_pos = cue_ball.state.rvw[0]
            radius = utils_ai.get_ball_radius(balls)
            
            # Collect candidates for this state
            candidates = [] # List of (action, features, target_id)
            
            for tid in legal_targets:
                target_ball = balls[tid]
                target_pos = target_ball.state.rvw[0]
                
                # Iterate pockets
                for pid, pocket in table.pockets.items():
                    pocket_pos = pocket.center
                    
                    # 1. Calculate Ghost Ball
                    ghost_pos = utils_ai.calculate_ghost_ball_pos(target_pos, pocket_pos, radius)
                    
                    # 2. Calculate aiming angle (phi)
                    v_aim = ghost_pos - cue_pos
                    dist = np.linalg.norm(v_aim)
                    if dist < 1e-6:
                        continue
                        
                    aim_dir = v_aim / dist
                    phi = np.degrees(np.arctan2(aim_dir[1], aim_dir[0])) % 360
                    
                    # 3. Check cut angle validity (e.g. > 90 degrees is impossible)
                    # Cut angle is angle between v_aim and (pocket - target)
                    v_shot_line = pocket_pos - target_pos
                    if np.linalg.norm(v_shot_line) < 1e-6:
                        continue
                    shot_dir = v_shot_line / np.linalg.norm(v_shot_line)
                    cos_cut = np.dot(aim_dir, shot_dir)
                    
                    if cos_cut < 0: # Angle > 90 degrees, impossible cut
                        continue
                    
                    # 4. Check basic clearance
                    # Path: Cue -> Ghost
                    is_clear_cg = utils_ai.is_path_clear(cue_pos, ghost_pos, balls, ['cue', tid], radius)
                    # Path: Target -> Pocket
                    is_clear_tp = utils_ai.is_path_clear(target_pos, pocket_pos, balls, ['cue', tid], radius)
                    
                    if not is_clear_tp:
                         # If target path blocked, skip
                         continue

                    # 5. Generate Velocity and Spin Variations
                    V0s = [1.5, 2.5, 3.5, 4.5, 6.0]
                    Spins = [(0.0, 0.0), (0.0, 0.2), (0.0, -0.2), (0.0, 0.4), (0.0, -0.4)] # (a, b) - simplified, mainly vertical spin (b) for follow/draw
                    
                    for v0 in V0s:
                        for a, b in Spins:
                            # Create action
                            action = {
                                'V0': v0,
                                'phi': phi,
                                'theta': 0.0,
                                'a': a,
                                'b': b
                            }
                            
                            # Get Features
                            feats = utils_ai.get_features(
                                cue_pos, target_pos, pocket_pos, ghost_pos, balls, table, radius,
                                V0_norm=v0/8.0, clear_cg=is_clear_cg, clear_tp=is_clear_tp,
                                a=a, b=b
                            )
                            
                            candidates.append((action, feats))
            
            # If no candidates (e.g. ball in hand or blocked), random shot?
            if not candidates:
                # Random action just to proceed game
                action = {
                    'V0': 3.0,
                    'phi': random.uniform(0, 360),
                    'theta': 0, 'a': 0, 'b': 0
                }
                env.take_shot(action)
                continue
                
            # Downsample candidates if too many
            if len(candidates) > samples_per_obs:
                candidates = random.sample(candidates, samples_per_obs)
                
            # Evaluate candidates
            # Snapshot current state for reward calculation
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            for act, feats in candidates:
                # Create sim system
                sim_table = copy.deepcopy(table)
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_cue = pt.Cue(cue_ball_id="cue")
                sim_cue.set_state(
                    V0=act['V0'], 
                    phi=act['phi'], 
                    theta=act['theta'], 
                    a=act['a'], 
                    b=act['b']
                )
                
                shot = pt.System(table=sim_table, balls=sim_balls, cue=sim_cue)
                
                # Simulate
                try:
                    pt.simulate(shot, inplace=True)
                    
                    # Calculate Reward
                    reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    
                    # Store
                    X_data.append(feats)
                    y_data.append(reward)
                    
                except Exception as e:
                    pass # Simulation failed
            
            # Find best action from collected candidates to advance the game
            # We assume the last batch of data corresponds to current state
            current_batch_rewards = y_data[-len(candidates):]
            if current_batch_rewards:
                best_local_idx = np.argmax(current_batch_rewards)
                best_action = candidates[best_local_idx][0]
                env.take_shot(best_action)
            else:
                 env.take_shot(candidates[0][0])
                 
        pbar.update(1)
        
    pbar.close()
    
    # Save dataset
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"Collected {len(X)} samples.")
    np.savez(output_file, X=X, y=y)
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--out', type=str, default='train/dataset_final.npz')
    args = parser.parse_args()
    
    collect_data(args.games, args.samples, args.out)
