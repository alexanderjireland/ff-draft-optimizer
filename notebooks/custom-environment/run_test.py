import sys
sys.path.append('env/custom_environment')
from custom_environment import *
from policies import BasicHeuristicPolicy
import pandas as pd

players = pd.read_csv('data\player_projections\model_06_12_predictions.csv')
players_2023 = players[players['season'] == 2023]

env = CustomEnvironment(players_2023, num_teams=2, draft_type='snake', rounds=14)
num_players = len(env.player_pool)
num_episodes = 1

agents = env.possible_agents
num_agents = len(agents)
policies = {agent: BasicHeuristicPolicy(env, num_agents) for agent in agents}

for episode in range(num_episodes):
    env.reset()
    while env.agent_selction is not None:
        agent = env.agent_selection
        action = policies[agent].select_action(env)
        obs = env.observe(agent)
        prev_pick = env.current_pick
        env.step(action)
        reward = env.rewards[agent]
        if env.current_pick > prev_pick:
            policies[agent].update(action, reward)
        
    if episode % 10 == 0:
        print(f"[Episode {episode}] Scores: {[env.rewards[a] for a in agents]}")


