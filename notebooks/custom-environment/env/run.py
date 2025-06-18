from custom_environment import CustomEnvironment
from heuristic_policy import BasicHeuristicPolicy
import pandas as pd
from tqdm import tqdm
from pettingzoo import AECEnv
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

players = pd.read_csv('data/player_projections/model_06_12_predictions_with_position_ranks.csv')
players_2023 = players[players['season'] == 2023]

def env_creator(config):
    return PettingZooEnv(CustomEnvironment(players_2023, **config))

register_env("draft_env", env_creator)

temp_env = env_creator({"num_teams": 12, "draft_type": "snake", "rounds": 14})
obs_space = temp_env.observation_space
act_space = temp_env.action_space

def policy_mapping_fn(agent_id):
    return "shared_policy" 

config = {
    "env": "draft_env",
    "env_config": {
        "num_teams": 12,
        "draft_type": "snake",
        "rounds": 14},
    "multiagent": {
        "policies": {
            "shared_policy": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
    "framework": "torch",
    "num_workers": 1,
}

trainer = PPO(config=config)

for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: reward mean = {result['episode_reward_mean']}")

