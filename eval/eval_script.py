"""
eval_script.py - 评估脚本

用于加载训练好的模型并测试 NewAgent 的性能。
"""

import sys
import os
import numpy as np

# 将项目根目录加入 python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

def evaluate(n_games=100, model_path=None):
    env = PoolEnv()
    
    # 初始化 Agents
    # Agent A: 基准 (BasicAgent)
    agent_a = BasicAgent()
    
    # Agent B: 待测试 (NewAgent)
    # 如果指定了 model_path，则加载该模型；否则尝试加载默认路径
    if model_path:
        agent_b = NewAgent(model_path=model_path)
    else:
        # 默认尝试加载 eval/checkpoints 下的模型
        default_model = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.zip')
        if os.path.exists(default_model):
            agent_b = NewAgent(model_path=default_model)
        else:
            # 如果没有找到 best_model，NewAgent 会尝试加载 train 下的默认模型或回退到启发式
            print("Warning: No model found in eval/checkpoints/best_model.zip. Using default/heuristic.")
            agent_b = NewAgent()

    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

    print(f"开始评估: {n_games} 局")
    print(f"Agent A: BasicAgent")
    print(f"Agent B: NewAgent (RL/Hybrid)")

    for i in range(n_games):
        env.reset(target_ball=target_ball_choice[i % 4])
        
        # 简单的进度条
        print(f"\rGame {i+1}/{n_games} ...", end="")
        
        while True:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            if player == 'A':
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)
                
            step_info = env.take_shot(action)
            done, info = env.get_done()
            
            if done:
                if info['winner'] == 'SAME':
                    results['SAME'] += 1
                elif info['winner'] == 'A':
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                else:
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                break
    
    print("\n评估结束!")
    
    # 计算胜率
    total = n_games
    win_rate_a = results['AGENT_A_WIN'] / total
    win_rate_b = results['AGENT_B_WIN'] / total
    draw_rate = results['SAME'] / total
    
    print(f"Results over {n_games} games:")
    print(f"BasicAgent (A) Win Rate: {win_rate_a*100:.1f}%")
    print(f"NewAgent   (B) Win Rate: {win_rate_b*100:.1f}%")
    print(f"Draws:                   {draw_rate*100:.1f}%")
    
    return results

if __name__ == "__main__":
    # 默认运行 100 局
    evaluate(n_games=100)
