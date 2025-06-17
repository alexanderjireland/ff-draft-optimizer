from custom_environment import CustomEnvironment
from heuristic_policy import BasicHeuristicPolicy
import pandas as pd
from tqdm import tqdm

players = pd.read_csv('data/player_projections/model_06_12_predictions_with_position_ranks.csv')
players_2023 = players[players['season'] == 2023]

env = CustomEnvironment(players_2023, num_teams=12, draft_type='snake', rounds=14)
num_players = len(env.player_pool)
num_episodes = 100

agents = env.possible_agents
num_agents = len(agents)
policies = {agent: BasicHeuristicPolicy(num_agents, num_players, epsilon=0.1) for agent in agents}

for episode in tqdm(range(num_episodes)):
    env.reset()
    step_count = 0
    max_steps = 1000
    while env.agent_selection is not None and step_count < max_steps:

        agent = env.agent_selection
        obs = env.observe(agent)
        #print(f"obs: {obs}") 

        team_positions_available = env.team_positions_available[agent]
        #print(f"team pos available: {team_positions_available}")
        try:
            action = policies[agent].select_action(obs, env.team_positions_available[agent])
        except Exception as e:
            print(f"Error during select_action: {e}")
            break

        prev_pick = env.current_pick
        env.step(action)

        reward = env.rewards[agent]
        if env.current_pick > prev_pick:
            policies[agent].update_policy(action, reward)
        step_count += 1

    """
    print("\n=== Final Team Rosters ===")
    print(env.full_roster_df)
    for agent in env.possible_agents:
        print(f"{agent}: {env._get_named_team_positions_roster()[agent]}")
        print(f"{agent} score: {env.rewards[agent]}")
        print(f"{agent} optimized lineup: {env.optimized_lineups[env.optimized_lineups['agent']==agent]}")

    print("Exited loop after", step_count, "steps.")
    """

    if episode % 10 == 0:
        tqdm.write(f"[Episode {episode}] Scores: {[env.rewards[a] for a in agents]}")


