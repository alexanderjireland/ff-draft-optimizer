import streamlit as st
st.set_page_config(layout="wide")
from streamlit_extras.let_it_rain import rain
import pandas as pd
import time
import numpy as np
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
    for i in range(4):
        player_names.append(st.session_state.env.env.gsis_to_name.get(st.session_state.env.env._get_player_from_action(i), "N/A"))
    obs_df['player'] = player_names
    new_obs_index = ["player", "projected_pts", "difference_with_replacement", "hurt_score", "difference_with_current_worst_starter", "action_mask", "pos_available", "team_needs", "next_opponent_needs"]
    obs_df_transpose = obs_df.T
    new_obs_df_transpose = obs_df_transpose.loc[new_obs_index]
    return new_obs_df_transpose


@st.cache_resource
def obtain_train_and_test_data(path='../../../data/player_projections/model_06_12_predictions_with_position_ranks.csv'):
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
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0, log_to_driver=True)

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
    checkpoint_path = r"C:\Users\irela\Documents\NSS_Projects\ff-draft-optimizer\models\fantasy_rl_checkpoints\final_checkpoint_20250626_195555"
    algo.restore(checkpoint_path)
    policy = algo.get_policy("shared_policy")
    
    return algo, policy, test_df_ref


#------------------------------------- Draft Loop -------------------------------------

st.title("Fantasy Football Mock Draft")

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
    st.markdown("---")
    st.subheader("Draft Configuration")
    NUM_TEAMS = st.slider("Number of teams drafting", 2, 12, value=2)
    POOL_SIZE = st.slider("Select player pool size", 1, 430, value=100, help="Select the pool size of random players taken from all eligible fantasy players in 2024.")
    ROUNDS = st.slider("Number of draft rounds", 1, 14, value=14)
    teams = [f"team_{i}" for i in range(NUM_TEAMS)]
    st.session_state.teams_dict = {team: team.capitalize().replace("_", " ") for team in teams}
    st.session_state.get_actual_team = {name: team for team, name in st.session_state.teams_dict.items()}
    HUMAN_TEAM_DISPLAY = st.pills("Pick your team", list(st.session_state.teams_dict.values()), default="Team 0")
    HUMAN_TEAM = st.session_state.get_actual_team.get(HUMAN_TEAM_DISPLAY)
    DRAFT_TYPE = st.pills("Draft Type", ["Snake", "Linear"], default="Linear", help="The draft order reverse each round in a snake draft, while the order remains the same in a linear draft.")

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

if st.session_state.draft_started:
    if not st.session_state.env_initialized:
        if st.session_state.algo is None:
            st.session_state.algo, st.session_state.policy, st.session_state.test_df_ref = setup_ray_and_load_model()
        
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

    # Add draft board? Displaying rosters...
    st.header("Draft Board")
    max_roster = st.session_state.ROUNDS
    df = {}
    for agent, info in st.session_state.env.env.team_info.items():
        current_roster = info['roster']
        current_roster_names = [f"{st.session_state.env.env.gsis_to_position[player]}: {st.session_state.env.env.gsis_to_name[player]}" for player in list(current_roster)]
        players = current_roster_names + ([None]*(max_roster - len(current_roster_names)))
        df[agent] = players
    st.dataframe(pd.DataFrame(df).rename(columns=st.session_state.teams_dict))

    col1, col2 = st.columns(2, border=False)
    
    st.session_state.done = all(st.session_state.env.env.terminations.values())

    with col2:
        if not st.session_state.done:
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
                    time.sleep(0.5)
                    
                    st.session_state.env.env.step(action_to_take)
                    st.session_state.step += 1
                    st.rerun()

    with col1:
        if st.session_state.env.env.current_pick >= st.session_state.env.env.total_picks:
            st.session_state.done = True

        if not st.session_state.done:

            current_agent = st.session_state.env.env.agent_selection
            st.header(f"Pick {st.session_state.env.env.current_pick + 1} / {st.session_state.env.env.total_picks}")

            if current_agent == st.session_state.human_agent_name:
                st.subheader(f"Your Turn ({st.session_state.teams_dict.get(current_agent)})")

                obs = st.session_state.env.env.observe(current_agent)
                flat_obs = flatten_obs_dict(obs)

                st.write("Current Observation:")
                st.dataframe(get_obs_df(flat_obs))

            else:
                st.subheader(f"{st.session_state.teams_dict.get(current_agent)}'s Turn (AI)")

                with st.spinner(f"AI ({st.session_state.teams_dict.get(current_agent)}) is thinking..."):
                    time.sleep((np.random.rand())) # Give the illusion of the AI thinking hard hehe
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

    if st.session_state.done:
        st.header("Draft Complete", divider="gray", help=None)
        time.sleep(.2)
        rain(
            emoji="🏈",
            font_size=54,
            falling_speed=5,
            animation_length=3,
        )
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