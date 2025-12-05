import warnings
# 忽略 pooltool 的数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pooltool")

from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

# 初始化环境和 Agent
env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 4  # 快速测试 4 局

agent_a, agent_b = BasicAgent(), NewAgent()
players = [agent_a, agent_b]
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

for i in range(n_games): 
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    print(f"本局 Player A: {players[i % 2].__class__.__name__}, 目标球型: {target_ball_choice[i % 4]}")
    
    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        
        # 轮流决策
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
            
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if done:
            winner = info['winner']
            print(f"本局结束，获胜者: {winner}")
            
            if winner == 'SAME':
                results['SAME'] += 1
            elif winner == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            break

print("\n测试结果：", results)