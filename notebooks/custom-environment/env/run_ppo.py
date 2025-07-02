import pandas as pd
import numpy as np
import time
from datetime import datetime
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import tune, init
from tqdm import tqdm
#import matplotlib.pyplot as plt
import os
import sys

from draft_environment import DraftEnvironment

data_df = pd.read_csv('data/player_projections/model_06_12_predictions_with_position_ranks.csv')
train_df = data_df[data_df['season']==2023]
test_df = data_df[data_df['season']==2024]

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

def train_env_creator(config):
    return PettingZooEnv(DraftEnvironment(player_df=train_df, **config))

def test_env_creator(config):
    return PettingZooEnv(DraftEnvironment(player_df=test_df, **config))

def quick_run(iterations=50):
    save_dir = f"/models/rllib_models/model_{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
    os.makedirs(save_dir, exist_ok=True)

    register_env("draft_env", train_env_creator)

    config = (
        PPOConfig()
        .environment("draft_env", env_config={
            "num_teams": 2,
            "draft_type": "regular",
            "rounds": 14,
        })
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(num_env_runners=1)
        .resources(num_gpus=0)
    )

    algo = config.build()

    results = []
    best_reward = float('-inf')
    best_checkpoint = None

    for i in tqdm(range(iterations)):
        result = algo.train()
        reward = result.get('episode_reward_mean')

        if reward is not None:
            results.append({'iter': i+1, 'reward': reward})

            if reward > best_reward:
                best_reward = reward
                try:
                    checkpoint_path = algo.save(os.path.join(save_dir, "best_model"))
                    best_checkpoint = checkpoint_path
                    print(f"Saved best model at iteration {i+1} to {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")


            if (i+1) % 10 == 0:
                print(f"Iteration: {i+1}, Reward: {reward}")
        else:
            print(f"Iteration: {i+1}, Episode reward mean not available yet.")

    try:
        final_checkpoint_path = algo.save(os.path.join(save_dir, "final_model"))
        print(f"Saved final model to {final_checkpoint_path}")
    except Exception as e:
        print(f"Error saving final checkpoint: {e}")


    algo.stop()

    if results:
        iterations_with_reward = [r['iter'] for r in results]
        rewards = [r['reward'] for r in results]
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_with_reward, rewards)
        plt.title('Training Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.show()
    else:
        print("No episodes completed with recorded rewards.")


    print(f'Training complete. Best reward: {best_reward}')

    return best_checkpoint, results

def evaluate_model(checkpoint, num_games=20):

    register_env("test_draft_env", test_env_creator)

    init(ignore_reinit_error=True, include_dashboard=False)

    config = (
        PPOConfig()
        .environment("test_draft_env", env_config={
            "num_teams": 2,
            "draft_type": "snake",
            "rounds": 14,
        })
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(num_env_runners=1)
        .resources(num_gpus=0)
    )

    algo = config.build()
    algo.restore(checkpoint)

    wins=0
    scores=[]

    for game in range(num_games):
        env = PettingZooEnv(DraftEnvironment(player_df=test_df, **config))
        obs = env.reset()

        while not all(env.terminations.values()):
            if env.agent_selection:
                action = algo.compute_single_action(obs, policy_id='shared_policy')
                env.step(action)
                obs = env.observe(env.agent_selection)

        team_0_df = env.full_roster_df[env.full_roster_df['agent']=='team_0']
        team_1_df = env.full_roster_df[env.full_roster_df['agent']=='team_1']

        score_0 = env._get_optimized_score(team_0_df)
        score_1 = env._get_optimized_score(team_1_df)

        scores.append({'team_0': score_0, 'team_1': score_1})

        if score_0 > score_1:
            wins+=1

        if (game + 1) % 10 == 0:
            print(f"Game: {game+1}/{num_games}, Current win rate: {wins/(game+1)}")

    win_rate = wins/num_games
    avg_score_team_0 = np.mean([score['team_0'] for score in scores])
    avg_score_team_1 = np.mean([score['team_1'] for score in scores])

    print(f"Win rate: {win_rate}")
    print(f"Average score for team 0: {avg_score_team_0}")
    print(f"Average score for team 1: {avg_score_team_1}")

    algo.stop()
    return {'win_rate': win_rate, 'scores':scores}

print('Starting training')
best_checkpoint, training_results = quick_run(iterations=200)
print(f"Best model saved at {best_checkpoint}")

