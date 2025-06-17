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
    print(f"Initial agent selection after reset: {env.agent_selection}")
    step_count = 0
    max_steps = 1000
    while env.agent_selection is not None and step_count < max_steps:
        print(f'step count {step_count}')

        agent = env.agent_selection
        print(f"Selected agent: {agent}")
        print(f"Draft history so far: {env.draft_history}")
        print(f"Available players left: {len(env.available_players)}")


        obs = env.observe(agent)
        print(f"obs: {obs}") 

        team_positions_available = env.team_positions_available[agent]
        print(f"team pos available: {team_positions_available}")
        try:
            action = policies[agent].select_action(obs, env.team_positions_available[agent])
        except Exception as e:
            print(f"Error during select_action: {e}")
            break

        print(f"Action : {action}")
        prev_pick = env.current_pick
        env.step(action)
        print(f"Agent {agent} took action {action}")
        print(f"New agent_selection: {env.agent_selection}")
        print(f"Current pick: {env.current_pick}")


        reward = env.rewards[agent]
        if env.current_pick > prev_pick:
            policies[agent].update_policy(action, reward)
        step_count += 1
    

    print("\n=== Final Team Rosters ===")
    print(env.full_roster_df)
    for agent in env.possible_agents:
        print(f"{agent}: {env._get_named_team_positions_roster()[agent]}")
        print(f"{agent} score: {env.rewards[agent]}")
        print(f"{agent} optimized lineup: {env.optimized_lineups[env.optimized_lineups['agent']==agent]}")
        
    print("Exited loop after", step_count, "steps.")


    if episode % 10 == 0:
        print(f"[Episode {episode}] Scores: {[env.rewards[a] for a in agents]}")


