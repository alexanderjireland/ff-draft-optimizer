import streamlit as st
st.set_page_config(layout="wide")
#from streamlit_extras.let_it_rain import rain
import pandas as pd
import time
import numpy as np
#import pymc as pm
import arviz as az
import json
import seaborn as sns
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import gymnasium as spaces
from ray.rllib.env.env_context import EnvContext
from mock_draft_env import MockDraftEnvironment
import sys
#sys.path.append("../../../src")
#import ff_projections
from pathlib import Path
import os

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, 
                num_cpus=1, 
                num_gpus=0 
                log_to_driver=True,
                object_store_memory=100_000_000,
                )

# Relative path to the data file
csv_path = Path("streamlit_data") / "model_06_12_predictions_with_position_ranks.csv"

class DraftEnvironmentWrapper(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()

        self.player_df = ray.get(config["player_df_ref"])
        self.num_teams = config.get('num_teams', 4)
        self.draft_type = config.get('draft_type', 'linear')
        self.rounds = config.get('rounds', 14)
        self.random_pool_size = config.get('random_pool_size', 200)
        self.flatten_obs = config.get('flatten_obs', True)
        self.your_team_id = config.get('your_team_id', 0)

        self.episode_count = 0

        self.env = MockDraftEnvironment(
            player_df=self.player_df,
            num_teams=self.num_teams,
            draft_type=self.draft_type,
            rounds=self.rounds,
            random_pool_size=self.random_pool_size,
            flatten_obs=self.flatten_obs
        )

        self._agent_ids = set(self.env.possible_agents)
        self._observation_space = self.env.observation_space(self.env.possible_agents[0])
        self._action_space = self.env.action_space(self.env.possible_agents[0])

        self.episode_rewards = {agent: 0.0 for agent in self._agent_ids}
        self.previous_rewards = {agent: 0.0 for agent in self._agent_ids}

    def reset(self, *, seed=None, options=None):
        try:
            self.episode_rewards = {agent: 0.0 for agent in self._agent_ids}
            self.previous_rewards = {agent: 0.0 for agent in self._agent_ids}
            self.episode_count += 1
            print(f"EPISODE {self.episode_count} RESET")

            raw_obs = self.env.reset(seed=seed, options=options)

            current_agent = self.env.agent_selection
            print(f"[WRAPPER RESET] Agent selection: {current_agent}")

            obs_dict = {}
            for agent in self._agent_ids:
                if agent == current_agent:
                    obs_dict[agent] = raw_obs
                else:
                    obs_dict[agent] = self._make_dummy_obs()

            if not hasattr(self.env, '_cumulative_rewards'):
                self.env._cumulative_rewards = {agent: 0.0 for agent in self._agent_ids}
            else:
                self.env._cumulative_rewards = {agent: 0.0 for agent in self._agent_ids}

            infos = {agent: {} for agent in self._agent_ids}
            print(f"[ENV RESET] current_pick: {self.env.current_pick}, total_picks: {self.env.total_picks}")

            return obs_dict, infos

        except Exception as e:
            print(f"Exception in reset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _make_dummy_obs(self):
        if self.flatten_obs:
            return np.zeros(self._observation_space.shape, dtype=np.float32)
        else:
            return {k: np.zeros_like(space.sample()) for k, space in self._observation_space.spaces.items()}

    def step(self, action_dict):
        current_agent = self.env.agent_selection

        if current_agent in action_dict:
            action = action_dict[current_agent]

            current_obs = self.env.observe(current_agent)
            if not self.flatten_obs and 'action_mask' in current_obs:
                action_mask = current_obs['action_mask']
                if not action_mask[action]:
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
                        print(f"Corrected invalid action for {current_agent}: {action_dict[current_agent]} -> {action}")

        try:
            if current_agent in action_dict:
                self.env.step(action)
            else:
                self.env.step(None)
        except Exception as e:
            print(f"CRASH during env.step: {e}")
            raise

        current_step_rewards = self.env.rewards.copy()

        for agent in self._agent_ids:
            step_reward = current_step_rewards.get(agent, 0.0)
            self.episode_rewards[agent] += step_reward

        rewards_for_rllib = {}
        for agent in self._agent_ids:
            rewards_for_rllib[agent] = current_step_rewards.get(agent, 0.0)

        observations, terminateds, truncateds, infos = {}, {}, {}, {}

        for agent in self._agent_ids:
            if agent == self.env.agent_selection and not self.env.terminations.get(agent, False):
                observations[agent] = self.env.observe(agent)
            else:
                observations[agent] = self._make_dummy_obs()

            terminateds[agent] = self.env.terminations.get(agent, False)
            truncateds[agent] = self.env.truncations.get(agent, False)

            infos[agent] = self.env.infos.get(agent, {})
            infos[agent]['episode_reward'] = self.episode_rewards[agent]

            if terminateds[agent]:
                infos[agent]['final_episode_reward'] = self.episode_rewards[agent]
                print(f"Episode finished for {agent}. Total reward: {self.episode_rewards[agent]:.2f}")

        episode_done = all(terminateds.values())
        terminateds["__all__"] = episode_done
        truncateds["__all__"] = episode_done
        #print(f"[STEP] Terminateds: {terminateds}, Truncateds: {truncateds}")
        #print(f"[RLlib Step] Rewards this step: {rewards_for_rllib}")
        #print(f"[RLlib Step] Episode rewards so far: {self.episode_rewards}")

        return observations, rewards_for_rllib, terminateds, truncateds, infos

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_agent_ids(self):
        return self._agent_ids
    
def env_creator(config: EnvContext):
    #print(f"[ENV_CREATOR] Creating environment with config: {config}")
    return DraftEnvironmentWrapper(config)

def get_unflattened_obs(obs_dict):
    num_draftable_positions = 4
    blank_obs_dict = {
                    "action_mask": np.zeros(num_draftable_positions, dtype=np.int8),
                    "pos_available": np.zeros(num_draftable_positions, dtype=np.int8),
                    "team_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "next_opponent_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "projected_pts": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_replacement": np.zeros(num_draftable_positions, dtype=np.float32),
                    "hurt_score": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_current_worst_starter": np.zeros(num_draftable_positions, dtype=np.float32)
                }
    obs_dict_list = list(obs_dict)
    try:
        current_index = 0
        for key in blank_obs_dict.keys():
            end_slice_index = current_index + num_draftable_positions
            blank_obs_dict[key] = obs_dict_list[current_index:end_slice_index]
            current_index = end_slice_index
        return blank_obs_dict
    except Exception as e:
        print(f"Exception in get_unflattened_obs: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def get_obs_df(obs_dict):
    unflattened_dict = get_unflattened_obs(obs_dict)
    index = st.session_state.env.env.draftable_positions
    obs_df = pd.DataFrame(unflattened_dict, index=index)
    
    player_names = []
    player_ids = []
    for i in range(4):
        player_names.append(st.session_state.env.env.gsis_to_name.get(st.session_state.env.env._get_player_from_action(i), "N/A"))
        player_ids.append(st.session_state.env.env._get_player_from_action(i))
    obs_df['player'] = player_names
    new_obs_index = ["player", "projected_pts", "difference_with_replacement", "hurt_score", "difference_with_current_worst_starter", "action_mask", "pos_available", "team_needs", "next_opponent_needs"]
    obs_df_transpose = obs_df.T
    new_obs_df_transpose = obs_df_transpose.loc[new_obs_index]
    new_obs_df_transpose = new_obs_df_transpose.astype(str)
    return new_obs_df_transpose, player_ids


@st.cache_resource
def obtain_train_and_test_data(path=csv_path):
    data_df = pd.read_csv(path)
    train_df = data_df[data_df['season']==2023]
    test_df = data_df[data_df['season']==2024]
    return train_df, test_df

def flatten_obs_dict(obs_dict):
    if isinstance(obs_dict, np.ndarray):
        return obs_dict

    keys = [
        "action_mask", "pos_available", "team_needs", "next_opponent_needs",
        "projected_pts", "difference_with_replacement", "hurt_score",
        "difference_with_current_worst_starter"
    ]

    flat_obs = np.concatenate([obs_dict[key] for key in keys])
    return flat_obs.astype(np.float32)

@st.cache_resource
def setup_ray_and_load_model():
    train_df, test_df = obtain_train_and_test_data()
    df_ref = ray.put(train_df)
    test_df_ref = ray.put(test_df)

    register_env("draft_env", env_creator)

    # Temporary env_config
    test_env_config = {'player_df_ref': df_ref, 'num_teams': 2, 'rounds': 14}
    test_env = env_creator(test_env_config)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment("draft_env", env_config={
            "player_df_ref": df_ref,
            "num_teams": 2,
            "draft_type": "regular",
            "rounds": 14,
        })
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model={"fcnet_hiddens": [512, 256], "fcnet_activation": "relu"},
            gamma=0.99,
            lr=1e-4,
            train_batch_size=2000,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )
    
    algo = config.build()
    checkpoint_path = Path("streamlit_data/final_checkpoint_20250627_015625").resolve()
    algo.restore(str(checkpoint_path))
    policy = algo.get_policy("shared_policy")

    #trace = az.from_netcdf("bayesian_regression_model/full_send_model_06_03.nc")
    trace_path = "bayesian_regression_model/full_send_model_06_03.nc"
    X_test = pd.read_csv("bayesian_regression_model/X_test_06_26.csv", index_col=0)
    y_test = pd.read_csv("bayesian_regression_model/y_test_06_26.csv")
    with open("bayesian_regression_model/index_to_playerid_dict.json", 'r') as f:
        index_dict = json.load(f)
    reverse_index_dict = {value: key for key, value in index_dict.items()}
    pm_test = pd.read_csv("bayesian_regression_model/pm_test_06_26.csv")
    pm_test_2024 = pm_test[pm_test['season']==2024]
    pm_test_cols = ['season', 'gsis_id', 'full_name_all_players', 'fantasy_pts',
        'ff_pts_prev_year', 'years_exp', 'Rank', 'ESPN', 'AVG', 'position_rank',
       'injury_prone', 'team_change', 'reception_prev_season',
       'passing_yards_prev_season', 'pass_touchdown_prev_season',
       'rush_touchdown_prev_season', 'interception_prev_season',
       'fumble_lost_prev_season', 'rushing_yards_prev_season',
       'two_pt_prev_season', 'receiving_yards_prev_season',
       'receive_touchdown_prev_season', 'team_rank_prev_season',
       'ff_pts_diff_prev_season', 'Rank_prev_season', 'ESPN_prev_season',
       'AVG_prev_season', 'position_rank_prev_season',
       'position_season_end_rank_prev_season', 'season_end_rank_prev_season',
       'position_season_end_rank_diff_prev_season',
       'season_end_rank_diff_prev_season', 'ESPN_reranked_prev_season',
       'ADP_diff_prev_season', 'injury_count_num_weeks_prev_season',
       'significant_injury_prev_season', 'cum_player_mean_prev_season',
       'cum_player_std_prev_season', 'cum_player_min_prev_season',
       'cum_player_noninjured_min_prev_season', 'cum_player_max_prev_season',
       'position_QB', 'position_RB', 'position_TE', 'position_WR']
    pm_test_cols_to_drop = ['fantasy_pts', 'Rank', 'ESPN', 'ff_pts_diff_prev_season', 'Rank_prev_season', 'ESPN_prev_season',
       'AVG_prev_season', 'position_rank_prev_season',
       'position_season_end_rank_prev_season', 'season_end_rank_prev_season',
       'position_season_end_rank_diff_prev_season',
       'season_end_rank_diff_prev_season', 'ESPN_reranked_prev_season',
       'ADP_diff_prev_season', 'injury_count_num_weeks_prev_season',
       'significant_injury_prev_season', 'cum_player_mean_prev_season',
       'cum_player_std_prev_season', 'cum_player_min_prev_season',
       'cum_player_noninjured_min_prev_season', 'cum_player_max_prev_season',
       'position_QB', 'position_RB', 'position_TE', 'position_WR']
    pm_test_cols_after_drop = [col_name for col_name in pm_test_cols if col_name not in pm_test_cols_to_drop]
    pm_test_display_cols = ["Season", "ID", "Full Name", "Total Fantasy Points (2023)", "Years of Experience", "AVG ADP", "Position Rank", 
    "Injury Prone", "Team Change (2023-2024)", "Receptions (2023)", "Passing Yards (2023)", "Passing Touchdowns (2023)", "Rushing Touchdowns (2023)",
    "Interceptions (2023)", "Fumbles (2023)", "Rushing Yards (2023)", "Two Point Conversions (2023)", "Receiving Yards (2023)", "Receiving Touchdowns (2023)",
    "Team Fantasy Rank (2023)"]
    pm_col_dict = dict(zip(pm_test_cols_after_drop, pm_test_display_cols))
    pm_test_2024 = pm_test_2024[pm_test_cols_after_drop].rename(columns=pm_col_dict)

    reordered_cols = ["Season", "ID", "Full Name", "Years of Experience", "Injury Prone", "Total Fantasy Points (2023)", "AVG ADP", "Position Rank", 
         "Passing Yards (2023)", "Passing Touchdowns (2023)", "Interceptions (2023)", "Fumbles (2023)", "Rushing Yards (2023)", "Rushing Touchdowns (2023)",
         "Receptions (2023)", "Receiving Yards (2023)", "Receiving Touchdowns (2023)","Two Point Conversions (2023)", "Team Fantasy Rank (2023)", "Team Change (2023-2024)"]
    
    return algo, policy, test_df_ref, X_test, y_test, index_dict, reverse_index_dict, trace_path, pm_test_2024


#------------------------------------- Bayesian Regresion --------------------------------
def create_credible_interval(posterior_pred_samples, interval_size):
    begin = (100 - interval_size)/2
    return np.percentile(posterior_pred_samples, [begin, (100-begin)])

@st.cache_data(show_spinner=False)
def get_posterior_predictive_samples(i, trace_path, X_test):
    # Identify the player features and true fantasy points for the i-th player in the test set
    trace = az.from_netcdf(trace_path)
    player_features = X_test.iloc[int(i)]

    # Extract the posterior samples from the trace
    intercept_samples = trace.posterior["intercept"].values.flatten()
    betas_samples = trace.posterior["betas"].values
    sigma_samples = trace.posterior["sigma"].values.flatten()

    # Reshape the betas_samples to match the player features
    n_chains, n_draws, n_features = betas_samples.shape
    betas_samples = betas_samples.reshape(n_chains * n_draws, n_features)

    # Calculate the posterior predictive distribution
    mu_samples = intercept_samples + np.dot(betas_samples, player_features)

    # Take mean and std of the posterior predictive distribution to create samples
    posterior_pred_samples = np.random.normal(mu_samples, sigma_samples)
    return posterior_pred_samples

if 'current_player_fig' not in st.session_state:
    st.session_state.current_player_fig = None
if 'current_player_ax' not in st.session_state:
    st.session_state.current_player_ax = None
if 'current_player_id_for_plot' not in st.session_state:
    st.session_state.current_player_id_for_plot = None
if 'current_posterior_pred_samples' not in st.session_state:
    st.session_state.current_posterior_pred_samples = None
if 'current_player_name_for_plot' not in st.session_state:
    st.session_state.current_player_name_for_plot = None

def create_base_predict_player_plot(i, trace_path, X_test, index_dict):
    posterior_pred_samples = get_posterior_predictive_samples(i, trace_path, X_test)
    player_id = index_dict[str(i)]
    player_name = st.session_state.env.env.gsis_to_name.get(player_id)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(posterior_pred_samples, bins=50, kde=True, color="skyblue", ax=ax)
    
    projected_median = np.median(posterior_pred_samples)
    ax.axvline(projected_median, color="red", linestyle="--", label=f"Median: {projected_median:.1f}")
    
    ax.set_title(f"Posterior Predictive Distribution for {player_name}")
    ax.set_xlabel("Predicted Season Points")
    ax.axis(xmin=0, xmax=500, ymin=0, ymax=650)
    ax.set_ylabel("Density")
    ax.grid(True)
    
    return fig, ax, posterior_pred_samples, player_name

@st.cache_data(show_spinner=False)
def create_base_predict_player_plot_cached(i, trace_path, X_test, index_dict):
    return create_base_predict_player_plot(i, trace_path, X_test, index_dict)

def update_plot_with_slider_value(ax, posterior_pred_samples, threshold, player_name):
    for line in ax.lines:
        if line.get_color() == 'purple':
            line.remove()
    for text in ax.texts:
        if text.get_color() == 'purple':
            text.remove()

    prob_gtt = np.mean(posterior_pred_samples > threshold)
    
    ax.axvline(threshold, color="purple", label=f"Probability Threshold: {threshold}")
    ax.text(threshold + 10, 550, f"Prob > {threshold}: {prob_gtt:.2%}", 
            color="purple", verticalalignment="top", 
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5))
    
    ax.legend(loc="upper right")

    return ax

def color_position(val):
    if 'QB' in str(val):
        return 'background-color: lightblue'
    elif 'RB' in str(val):
        return 'background-color: lightgreen'
    elif 'WR' in str(val):
        return 'background-color: lightcoral'
    elif 'TE' in str(val):
        return 'background-color: lightsalmon'
    else:
        return ''
#------------------------------------- Draft Loop -------------------------------------

st.title("2024 Fantasy Football Mock Draft")

if 'draft_started' not in st.session_state:
    st.session_state.draft_started = False
if 'env_initialized' not in st.session_state:
    st.session_state.env_initialized = False
if 'algo' not in st.session_state:
    st.session_state.algo = None
if 'policy' not in st.session_state:
    st.session_state.policy = None
if 'test_df_ref' not in st.session_state:
    st.session_state.test_df_ref = None

with st.sidebar:
    if "done" in st.session_state and not st.session_state.done:
            st.subheader(f"Pick {min((st.session_state.env.env.current_pick + 1), st.session_state.env.env.total_picks)} / {st.session_state.env.env.total_picks}")
    st.markdown("---")
    st.subheader("Draft Configuration")
    NUM_TEAMS = st.slider("Number of teams drafting", 2, 12, value=2)
    POOL_SIZE = st.slider("Select player pool size", 40, 430, value=100, help="Select the pool size of random players taken from all eligible fantasy players in 2024.")
    ROUNDS = st.slider("Number of draft rounds", 1, 14, value=14)
    teams = [f"team_{i}" for i in range(NUM_TEAMS)]
    st.session_state.teams_dict = {team: team.capitalize().replace("_", " ") for team in teams}
    st.session_state.get_actual_team = {name: team for team, name in st.session_state.teams_dict.items()}
    HUMAN_TEAM_DISPLAY = st.pills("Pick your team", list(st.session_state.teams_dict.values()), default="Team 0")
    HUMAN_TEAM = st.session_state.get_actual_team.get(HUMAN_TEAM_DISPLAY)
    DRAFT_TYPE = st.pills("Draft Type", ["Snake", "Linear"], default="Linear", help="The draft order reverses each round in a snake draft, while the order remains the same in a linear draft.")
    fill_draftboard = st.checkbox("Color Fill Draft Board", value=False)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Draft", disabled=st.session_state.draft_started, key="start_draft_button"):
            st.session_state.ROUNDS = ROUNDS
            st.session_state.NUM_TEAMS = NUM_TEAMS
            st.session_state.draft_started = True
            st.session_state.env_initialized = False
            st.rerun()
    with col2:
        # Reset button
        if st.button("Reset Draft"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.markdown("*Created by Alexander Ireland*      [Github](https://github.com/alexanderjireland/ff-draft-optimizer)          [LinkedIn](https://www.linkedin.com/in/alexanderjireland)")

if st.session_state.draft_started:
    if not st.session_state.env_initialized:
        if st.session_state.algo is None:
            st.session_state.algo, \
            st.session_state.policy, \
            st.session_state.test_df_ref, \
            st.session_state.X_test, \
            st.session_state.y_test, \
            st.session_state.index_dict, \
            st.session_state.reverse_index_dict, \
            st.session_state.trace_path, \
            st.session_state.pm_test = setup_ray_and_load_model()
        
        env_config = {
            'player_df_ref': st.session_state.test_df_ref,
            'num_teams': NUM_TEAMS,
            'draft_type': DRAFT_TYPE.lower(),
            'rounds': st.session_state.ROUNDS,
            'random_pool_size': POOL_SIZE,
            'flatten_obs': True,
            'your_team_id': int(HUMAN_TEAM.strip("team_"))
        }
        st.session_state.env = env_creator(env_config)
        st.session_state.obs_dict, st.session_state.infos = st.session_state.env.reset()
        st.session_state.human_agent_name = HUMAN_TEAM

        st.session_state.step = 0
        st.session_state.done = False
        st.session_state.env_initialized = True
        st.session_state.map_col_names = {'player_name': 'Player Name', 'position': 'Position', 'projected_pts':"Projected Points", "fantasy_pts":"Total Fantasy Points (2024)"}
        
    # Display the available player pool
    with st.expander("Draft Player Pool", expanded=False):
        st.dataframe(st.session_state.env.env.player_pool_df[['position', 'player_name']].sort_values('position').rename(columns=st.session_state.map_col_names), hide_index=True)

    col1, col2 = st.columns(2, border=False)
    
    st.session_state.done = all(st.session_state.env.env.terminations.values())

    with col2:
        if not st.session_state.done:

            st.header("Draft Board")
            max_roster = st.session_state.ROUNDS
            df = {}
            for agent, info in st.session_state.env.env.team_info.items():
                current_roster = info['roster']
                current_roster_names = [f"{st.session_state.env.env.gsis_to_position[player]}: {st.session_state.env.env.gsis_to_name[player]}" for player in list(current_roster)]
                players = current_roster_names + ([None]*(max_roster - len(current_roster_names)))
                df[agent] = players
            df = pd.DataFrame(df).rename(columns=st.session_state.teams_dict)
            if fill_draftboard:
                st.dataframe(df.style.map(lambda x: color_position(x)))
            else:
                st.dataframe(df)

            current_agent = st.session_state.env.env.agent_selection
            st.subheader("My Roster")
            roster = {}
            for key, value in st.session_state.env.env.team_info[st.session_state.human_agent_name]['pos_roster'].items():
                roster[key] = [st.session_state.env.env.gsis_to_name.get(player, 'None') for player in value]
            roster_dict = {key: ", ".join(value) for key, value in roster.items()}
            st.dataframe({"Roster":roster_dict})
            if current_agent == st.session_state.human_agent_name:
                obs = st.session_state.env.env.observe(current_agent)
                flat_obs = flatten_obs_dict(obs)
                model_action_index = st.session_state.policy.compute_single_action(flat_obs, explore=False)[0]
                suggested_position = st.session_state.env.env.draftable_positions[model_action_index]
                suggested_player_id = st.session_state.env.env._get_player_from_action(model_action_index)
                suggested_player_name = st.session_state.env.env.gsis_to_name.get(suggested_player_id, "N/A")

                st.info(f"**Model Suggestion:** Draft {suggested_position}: {suggested_player_name}")

                options = [
                    f"{i}: {pos} ({st.session_state.env.env.gsis_to_name.get(st.session_state.env.env._get_player_from_action(i), 'N/A')})"
                    for i, pos in enumerate(st.session_state.env.env.draftable_positions)
                ]

                user_choice_str = st.selectbox(
                    "Make your selection:",
                    ["Model Suggestion"] + options,
                    key=f"user_input_{st.session_state.step}"
                )

                if st.button("Confirm Pick", key=f"confirm_button_{st.session_state.step}"):
                    action_to_take = None
                    if user_choice_str == "Model Suggestion":
                        action_to_take = model_action_index
                    else:
                        action_to_take = int(user_choice_str.split(":")[0].strip())

                    player_id = st.session_state.env.env._get_player_from_action(action_to_take)
                    player_name = st.session_state.env.env.gsis_to_name.get(player_id, "N/A")
                    st.success(f"You drafted {player_name}.")

                                    
                    st.session_state.env.env.step(action_to_take)
                    st.session_state.step += 1

                    st.session_state.current_player_fig = None
                    st.session_state.current_player_ax = None
                    st.session_state.current_player_id_for_plot = None

                    st.rerun()

    with col1:
        if st.session_state.env.env.current_pick >= st.session_state.env.env.total_picks:
            st.session_state.done = True

        if not st.session_state.done:

            if current_agent == st.session_state.human_agent_name:
                st.subheader(f"Your Turn ({st.session_state.teams_dict.get(current_agent)})")

                obs = st.session_state.env.env.observe(current_agent)
                flat_obs = flatten_obs_dict(obs)

                st.write("Current Observation:")
                obs_df, ids = get_obs_df(flat_obs)
                st.dataframe(obs_df)

                ppdist_display = True
                default_player_option = None
                options = [f"{st.session_state.env.env.gsis_to_position.get(id)}: {st.session_state.env.env.gsis_to_name.get(id)}" for id in ids if id is not None]
                try:
                    default_player_option=options[model_action_index]
                except:
                    try:
                        default_player_option=options[0]
                    except:
                        ppdist_display = False
                
                if ppdist_display and default_player_option:
                    selected_player_display = st.segmented_control(
                        "Posterior Predictive Distributions", 
                        options, 
                        default=default_player_option
                    )
                    if selected_player_display in options:
                        player_id_to_plot = ids[options.index(selected_player_display)]
                        player_idx_to_plot = st.session_state.reverse_index_dict.get(player_id_to_plot)

                        if player_idx_to_plot is not None:
                            if st.session_state.current_player_id_for_plot != player_id_to_plot:
                                st.session_state.current_player_fig, \
                                st.session_state.current_player_ax, \
                                st.session_state.current_posterior_pred_samples, \
                                st.session_state.current_player_name_for_plot = create_base_predict_player_plot(
                                    player_idx_to_plot, 
                                    st.session_state.trace_path, 
                                    st.session_state.X_test, 
                                    st.session_state.index_dict
                                )
                                st.session_state.current_player_id_for_plot = player_id_to_plot
                            
                            current_fig = st.session_state.current_player_fig
                            current_ax = st.session_state.current_player_ax
                            
                            threshold = st.slider(
                                f"Set Probability threshold for {st.session_state.current_player_name_for_plot}", 
                                0, 500, 200, step=5,
                                key=f"threshold_slider_{player_id_to_plot}_{st.session_state.step}" # Unique key for player/step
                            )
                            
                            update_plot_with_slider_value(
                                current_ax, 
                                st.session_state.current_posterior_pred_samples, 
                                threshold, 
                                st.session_state.current_player_name_for_plot
                            )
                            
                            st.pyplot(current_fig)
                        else:
                            st.warning(f"Player index: Model suggestion '{player_idx_to_plot}' not available.")

            else:
                st.subheader(f"{st.session_state.teams_dict.get(current_agent)}'s Turn (AI)")

                with st.spinner(f"AI ({st.session_state.teams_dict.get(current_agent)}) is thinking..."):
                    #time.sleep((np.random.rand())) # Give the illusion of the AI thinking hard hehe
                    obs = st.session_state.env.env.observe(current_agent)
                    flat_obs = flatten_obs_dict(obs)

                    ai_action_idx = st.session_state.policy.compute_single_action(flat_obs, explore=False)[0]

                    position = st.session_state.env.env.draftable_positions[ai_action_idx]
                    player_id = st.session_state.env.env._get_player_from_action(ai_action_idx)
                    player_name = st.session_state.env.env.gsis_to_name.get(player_id, "N/A")

                    st.write(f"AI agent **{st.session_state.teams_dict.get(current_agent)}** selects **{position}**: {player_name}.")
                    time.sleep(0.1)

                    st.session_state.env.env.step(ai_action_idx)

                    st.session_state.step += 1
                    st.rerun()

    if not st.session_state.done:
        display_player_data = st.toggle(f"Display {st.session_state.current_player_name_for_plot} Data", value=True)
        if display_player_data:
            st.dataframe(st.session_state.pm_test[st.session_state.pm_test['ID']==st.session_state.current_player_id_for_plot], hide_index=True)

    if st.session_state.done:
        st.header("Draft Board")
        max_roster = st.session_state.ROUNDS
        df = {}
        for agent, info in st.session_state.env.env.team_info.items():
            current_roster = info['roster']
            current_roster_names = [f"{st.session_state.env.env.gsis_to_position[player]}: {st.session_state.env.env.gsis_to_name[player]}" for player in list(current_roster)]
            players = current_roster_names + ([None]*(max_roster - len(current_roster_names)))
            df[agent] = players
        df = pd.DataFrame(df).rename(columns=st.session_state.teams_dict)
        if fill_draftboard:
            st.dataframe(df.style.map(lambda x: color_position(x)))
        else:
            st.dataframe(df)


        st.header("Draft Complete", divider="gray", help=None)
        #time.sleep(.6)
        #rain(
        #    emoji="🏈",
        #    font_size=54,
        #    falling_speed=5,
        #    animation_length=3,
        #)
        with st.expander("Final Rewards"):
            for agent, reward in st.session_state.env.env.rewards.items():
                if agent == st.session_state.human_agent_name:
                    st.metric(label=f"**Your Team** ({st.session_state.teams_dict.get(agent)}) Final Reward", value = f"{reward:.2f}")
                else:
                    st.metric(label=f"{st.session_state.teams_dict.get(agent)} Final Reward", value = f"{reward:.2f}")

        def display_final_results():
            if agent == st.session_state.human_agent_name:
                st.subheader(f"**Your Final Roster ({st.session_state.teams_dict.get(agent)})**")
            else:
                st.subheader(f"{st.session_state.teams_dict.get(agent)}'s Final Roster")

            roster_df = st.session_state.env.env.full_roster_df[st.session_state.env.env.full_roster_df['agent'] == agent].copy()
            roster_df['projected_pts'] = roster_df['gsis_id'].map(st.session_state.env.env.gsis_to_projections)

            st.write("Full Roster:")
            st.dataframe(roster_df[['position', 'player_name', 'projected_pts', 'fantasy_pts']].sort_values('projected_pts', ascending=False).rename(columns=st.session_state.map_col_names), hide_index=True)

            optimized_roster_df = st.session_state.env.env._get_optimized_lineup(roster_df)
            st.write("Optimized Starting Lineup:")
            if optimized_roster_df is not None: # Figure out how to handle small player pool
                st.dataframe(optimized_roster_df[['position', 'player_name', 'projected_pts', 'fantasy_pts']].sort_values('projected_pts', ascending=False).rename(columns=st.session_state.map_col_names), hide_index=True)
                st.metric(label="Total Projected Points (Starters)", value=f"{optimized_roster_df['projected_pts'].sum():.2f}")
                st.metric(label="Total Fantasy Points (Starters)", value=f"{optimized_roster_df['fantasy_pts'].sum():.2f}")
                st.divider()

        if st.session_state.NUM_TEAMS < 3:
            fin_col1, fin_col2 = st.columns(2)
            for i, agent in enumerate(st.session_state.env.env.team_info):
                if i % 2 == 0:
                    with fin_col1:
                        display_final_results()
                else:
                    with fin_col2:
                        display_final_results()
        else:
            fin_col1, fin_col2, fin_col3 = st.columns(3)
            for i, agent in enumerate(st.session_state.env.env.team_info):
                if i % 3 == 0:
                    with fin_col1:
                        display_final_results()
                elif i % 3 ==1:
                    with fin_col2:
                        display_final_results()
                else:
                    with fin_col3:
                        display_final_results()
