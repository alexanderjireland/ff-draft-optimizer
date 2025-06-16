from custom_environment import CustomEnvironment
from heuristic_policy import BasicHeuristicPolicy
import pandas as pd

players = pd.read_csv('data/player_projections/model_06_12_predictions_with_position_ranks.csv')
players_2023 = players[players['season'] == 2023]
print(players_2023.columns)
env = CustomEnvironment(players_2023, num_teams=2, draft_type='snake', rounds=14)
num_players = len(env.player_pool)
num_episodes = 1

agents = env.possible_agents
num_agents = len(agents)
policies = {agent: BasicHeuristicPolicy(num_agents=num_agents, player_pool_size=num_players, epsilon=-1) for agent in agents}

for episode in range(num_episodes):
    env.reset()
    while env.agent_selection is not None:
        agent = env.agent_selection
        obs = env.observe(agent)
        print(f'obs: {obs}')
        team_positions_available = env.team_positions_available[agent]
        print(f"team pos available: {team_positions_available}")
        action = policies[agent].select_action(obs, team_positions_available)
        print(f"Action : {action}")
        prev_pick = env.current_pick
        env.step(action)
        reward = env.rewards[agent]
        if env.current_pick > prev_pick:
            policies[agent].update_policy(action, reward)
        
    if episode % 10 == 0:
        print(f"[Episode {episode}] Scores: {[env.rewards[a] for a in agents]}")


