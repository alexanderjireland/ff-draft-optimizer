import tianshou as ts
import torch
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from collections import deque
from typing import Deque, Tuple
from dataclasses import dataclass

# --- Your DraftEnvironment Code (START) ---
# Paste your entire DraftEnvironment class and related dataclasses/imports here.
# For brevity, I'm omitting it here, but it should be present in the final code block.

@dataclass
class PositionDQ:
    players: Deque[Tuple[str, float]]
    diffs: Deque[float]

class DraftEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
        "is_parallelizable": True,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14, random_pool_size=100, real_scores_at_draft_end=False, render_mode=None, flatten_obs=False):
        print("Initializing DraftEnvironment")
        super().__init__()

        self.render_mode = render_mode
        self.flatten_obs = flatten_obs
        
        self.player_df = player_df
        self.random_pool_size = random_pool_size
        self.real_scores = real_scores_at_draft_end

        # Initialize environment parameters
        self.num_teams = num_teams
        self.snake_draft = draft_type == 'snake'
        self.max_rounds = rounds
        self.total_picks = self.max_rounds * self.num_teams
        self.position_limits = {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1,
            'BENCH': 7,
        }
        self.full_team_number = sum([limit for limit in self.position_limits.values()])
        if self.full_team_number < rounds:
            raise ValueError(f"{rounds} rounds exceed the total number of available slots on team: {self.full_team_number}")

        self.draftable_positions = ['QB', 'RB', 'WR', 'TE']
        self.flex_positions = ['RB', 'WR', 'TE']

        self._initialize_agents()
        self._initialize_spaces()

        self.position_str_to_index = {pos: i for i, pos in enumerate(self.draftable_positions)}

        # Draft tracking
        self.draft_order = self._get_draft_order()

        self.invalid_action_penalty = -50
        print("Finished Initializing DraftEnvironment")

    def _initialize_player_metadata(self):
        # print("_initialize_player_metadata") # Too verbose
        self.player_pool_df = self.player_df.sample(
            n=min(self.random_pool_size, len(self.player_df)), 
            random_state=np.random.randint(0, 10000)
        )
    
        self.player_pool = self.player_pool_df['gsis_id'].to_list()
        self.gsis_to_name = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['player_name']))
        self.gsis_to_position = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['position']))
        self.gsis_to_projections = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['median_prediction']))

        self.pos_player_pool = self.player_pool_df.groupby('position')['gsis_id'].agg(list).to_dict()

        for pos in self.draftable_positions:
            if pos not in self.pos_player_pool:
                self.pos_player_pool[pos] = []

        self.pos_dqs = {}

        for pos in self.draftable_positions:
            player_id_and_projections = self._sort_and_create_dq(pos)
            diffs = self._create_diffs_dq(player_id_and_projections)
            self.pos_dqs[pos] = PositionDQ(players=player_id_and_projections, diffs=diffs)

    def _sort_and_create_dq(self, pos):
        # print("_sort_and_create_dq") # Too verbose
        if pos not in self.draftable_positions:
                raise ValueError(f"{pos} not in possible starting positions: {self.draftable_positions}")
        
        if not self.pos_player_pool[pos]:
            return deque()
        
        return deque(sorted([(id, self.gsis_to_projections[id]) for id in self.pos_player_pool[pos]],
                                key=lambda x: x[1],
                                reverse=True))
    
    def _create_diffs_dq(self, pos_dq):
        # print("_create_diffs_dq") # Too verbose
        if not pos_dq:
            return deque()
        dq = [proj_pts for _, proj_pts in pos_dq]
        dq.append(0)
        return deque([a - b for a, b in zip(dq, dq[1:])])
    
    def _update_pos_dqs(self, pos):
        # print("_update_pos_dqs") # Too verbose
        if len(self.pos_dqs[pos].players) > 0:
            self.pos_dqs[pos].players.popleft()
        if len(self.pos_dqs[pos].diffs) > 0:
            self.pos_dqs[pos].diffs.popleft()

    def _initialize_spaces(self):
        # print("_initialize_spaces") # Too verbose
        num_draftable_positions = 4
        
        if self.flatten_obs:
            total_size = num_draftable_positions * 8
            self._observation_spaces = {
                agent: spaces.Box(low=-500, high=500, shape=(total_size,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            self._observation_spaces = {
                agent: spaces.Dict({
                    "action_mask": spaces.MultiBinary(num_draftable_positions),
                    "pos_available": spaces.MultiBinary(num_draftable_positions),
                    "team_needs": spaces.MultiBinary(num_draftable_positions),
                    "next_opponent_needs": spaces.MultiBinary(num_draftable_positions),
                    "projected_pts": spaces.Box(low=-20, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                    "difference_with_replacement": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                    "hurt_score": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                    "difference_with_current_worst_starter": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32)
                }) for agent in self.agents
            }
        
        self._action_spaces = {
            agent: spaces.Discrete(num_draftable_positions) for agent in self.agents
        }

    def _initialize_agents(self):
        # print("_initialize_agents") # Too verbose
        self.possible_agents = [f"team_{i}" for i in range(self.num_teams)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}


    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed) # Also set random module seed for sample
        
        self.current_pick = 0
        self.draft_history = []
        self.agents = self.possible_agents[:]
        
        self._initialize_player_metadata()

        print("-" * 50)
        # print(f"PLAYER POOL DF HEAD: {self.player_pool_df.head()}") # Too verbose
        print("-" * 50)
        
        # Collect all available players
        self.available_players = self.player_pool.copy()
        
        self.team_info = {
            agent: {
                'roster': [],
                'pos_counts': {pos: 0 for pos in self.draftable_positions},
                'pos_roster': {pos: [None] * self.position_limits[pos] for pos in self.position_limits},
                'pos_projections': {pos: [0] * self.position_limits[pos] for pos in self.position_limits}
            } for agent in self.agents
        }
        
        self.full_roster_df = None
        self.optimized_lineups = None
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.draft_pick_reward_values = {}
        
        self.agent_selection = self.current_agent()
        
        initial_observations = {a: self.observe(a) for a in self.agents}
        return initial_observations, self.infos


    def step(self, action):
        # print(f"Step for agent {self.agent_selection} with action {action}") # Too verbose
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            obs = {a: self.observe(a) for a in self.agents}
            return obs, self.rewards, self.terminations, self.truncations, self.infos
        
        # Check if action is valid given the action_mask (pre-computation)
        current_observation = self.observe(agent)
        if self.flatten_obs:
            action_mask_len = len(self.draftable_positions)
            action_mask = current_observation[:action_mask_len]
        else:
            action_mask = current_observation["action_mask"]

        if not (isinstance(action, (int, np.integer)) and 0 <= action < len(self.draftable_positions) and action_mask[action] == 1):
            print(f'[Invalid Action] {agent} attempted invalid action (action={action}). Penalizing and skipping turn.')
            self.rewards[agent] += self.invalid_action_penalty
            self._advance_draft(None)
        else:
            player = self._get_player_from_action(action)
            if player:
                if self._draft_player(agent, action, player):
                    self._advance_draft(action)
                else:
                    print(f"[Invalid Pick (Game Logic)] {agent} attempted invalid selection (action={action}). Penalizing and skipping turn.")
                    self.rewards[agent] += self.invalid_action_penalty
                    self._advance_draft(None)
            else:
                print(f"[Invalid Pick (No Player)] {agent} attempted invalid selection (action={action}). Needs to retry.")
                self.rewards[agent] += self.invalid_action_penalty
                self._advance_draft(None)
        
        # After step, collect observations, rewards, etc. for all agents
        observations = {a: self.observe(a) for a in self.agents}
        rewards_out = self.rewards
        terminations_out = self.terminations
        truncations_out = self.truncations
        infos_out = self.infos

        self.rewards = {agent_id: 0.0 for agent_id in self.agents} # Reset rewards for the next step

        return observations, rewards_out, terminations_out, truncations_out, infos_out

    def _was_dead_step(self, action):
        # print("_was_dead_step") # Too verbose
        if self.current_pick < self.total_picks:
            self._advance_draft(None)

    def _get_player_from_action(self, action):
        # print("_get_player_from_action") # Too verbose
        position = self.draftable_positions[action]
        player_queue = self.pos_dqs[position].players
        diffs_queue = self.pos_dqs[position].diffs

        while player_queue:
            player_id, _ = player_queue[0]
            if player_id in self.available_players:
                return player_id
            else:
                # print(f"DEBUG: Player {player_id} at top of {position} queue is not available. Removing.") # Too verbose
                player_queue.popleft()
                if diffs_queue:
                    diffs_queue.popleft()
        return None

    def _draft_player(self, agent, action, player):
        # print("_draft_player") # Too verbose
        # Ensure the player is valid and available
        if player not in self.available_players:
            print(f'player {self.gsis_to_name.get(player, player)} not available.')
            return False
        
        succussful_update = self._update_team_info(agent, action, player)
        if not succussful_update:
            print('Unsuccessful update of team info')
            return False
        
        # Remove the player from available players and update draft history
        self.available_players.remove(player)
        self.draft_history.append((agent, player))
        self._get_draft_pick_reward(agent, action)
        return True
    
    def _update_team_info(self, agent, action, player):
        # print("_update_team_info") # Too verbose
        player_pos = self.draftable_positions[action]
        player_proj = self.gsis_to_projections[player]

        # Update the team roster and positions such that position players are chosen first, then FLEX, then BENCH
        self.team_info[agent]['roster'].append(player)

        success = False

        current_flex_players_count = len([player for player in self.team_info[agent]['pos_roster']['FLEX'] if player is not None])
        current_bench_players_count = len([player for player in self.team_info[agent]['pos_roster']['BENCH'] if player is not None])

        if self.team_info[agent]['pos_counts'][player_pos] < self.position_limits[player_pos]:
            idx = self.team_info[agent]['pos_counts'][player_pos]
            self.team_info[agent]['pos_roster'][player_pos][idx] = player
            self.team_info[agent]['pos_projections'][player_pos][idx] = player_proj
            self.team_info[agent]['pos_counts'][player_pos] += 1
            success = True

        elif player_pos in self.flex_positions and (current_flex_players_count < self.position_limits['FLEX']):
            idx = current_flex_players_count
            self.team_info[agent]['pos_roster']['FLEX'][idx] = player
            self.team_info[agent]['pos_projections']['FLEX'][idx][0] = player_proj # Update the first element of FLEX
            self.team_info[agent]['pos_counts'][player_pos] += 1 # This counts total players of this position type, not just in their primary slot
            success = True

        elif current_bench_players_count < self.position_limits['BENCH']:
            idx = current_bench_players_count
            self.team_info[agent]['pos_roster']['BENCH'][idx] = player
            self.team_info[agent]['pos_projections']['BENCH'][idx] = player_proj
            self.team_info[agent]['pos_counts'][player_pos] += 1 # This counts total players of this position type
            success = True

        if not success:
            print(f'No room left on team {agent} left to draft {self.gsis_to_name.get(player, player_pos)} ({player_pos}). Roster full.')

        return success
        
    def _advance_draft(self, action):
        # print("_advance_draft") # Too verbose
        self.current_pick += 1

        if action is not None:
            position = self.draftable_positions[action]
            self._update_pos_dqs(position)

        if self.current_pick >= self.total_picks:
            self._finalize_draft()
        else:
            self.agent_selection = self.current_agent()

    def _finalize_draft(self):
        print("Finalizing Draft...")
        try:
            self.full_roster_df = self._get_full_roster_df()
            if not self.full_roster_df.empty:
                self.optimized_lineups = self.full_roster_df.groupby('agent').apply(self._get_optimized_lineup)
                print("+"*50)
                print(f"Optimized Lineups: {self.optimized_lineups}")
                print("+"*50)
            for agent in self.possible_agents:
                self.terminations[agent] = True
                if not isinstance(self.infos[agent], dict):
                    self.infos[agent] = {}
                        
            self._calculate_final_rewards()
                        
        except Exception as e:
            print(f"Error in _finalize_draft: {e}")
            for agent in self.possible_agents:
                self.terminations[agent] = True
                self.rewards[agent] = -100
                if not isinstance(self.infos[agent], dict):
                    self.infos[agent] = {}

    def observe(self, agent):
        # print(f"observe for {agent}") # Too verbose
        if agent is None:
            # Return an observation with correct shape/structure for a "done" agent
            num_draftable_positions = len(self.draftable_positions)
            if self.flatten_obs:
                return np.zeros(num_draftable_positions * 8, dtype=np.float32)
            else:
                return {
                    "action_mask": np.zeros(num_draftable_positions, dtype=np.int8),
                    "pos_available": np.zeros(num_draftable_positions, dtype=np.int8),
                    "team_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "next_opponent_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "projected_pts": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_replacement": np.zeros(num_draftable_positions, dtype=np.float32),
                    "hurt_score": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_current_worst_starter": np.zeros(num_draftable_positions, dtype=np.float32)
                }

        if self.terminations[agent] or self.truncations[agent] or self.current_pick >= self.total_picks:
            num_draftable_positions = len(self.draftable_positions)
            if self.flatten_obs:
                return np.zeros(num_draftable_positions * 8, dtype=np.float32)
            else:
                return {
                    "action_mask": np.zeros(num_draftable_positions, dtype=np.int8),
                    "pos_available": np.zeros(num_draftable_positions, dtype=np.int8),
                    "team_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "next_opponent_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                    "projected_pts": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_replacement": np.zeros(num_draftable_positions, dtype=np.float32),
                    "hurt_score": np.zeros(num_draftable_positions, dtype=np.float32),
                    "difference_with_current_worst_starter": np.zeros(num_draftable_positions, dtype=np.float32)
                }
        
        action_mask = [int(self._can_draft_position(agent, pos) and len(self.pos_dqs[pos].players) > 0) for pos in self.draftable_positions]
        
        # Ensure all features are available before creating the observation
        pos_available_feat = self._get_available_pos()
        team_needs_feat = self._get_team_needs(agent)
        next_opponent_needs_feat = self._get_next_opponent_needs(agent)
        projected_pts_feat = self._get_top_pos_proj_pts()
        diff_with_replacement_feat = self._get_difference_with_replacement()
        hurt_score_feat = self._get_hurt_score(agent)
        diff_with_worst_starter_feat = self._get_diff_with_current_worst_starter(agent)

        obs_dict = {
            "action_mask": np.array(action_mask, dtype=np.int8),
            "pos_available": np.array(pos_available_feat, dtype=np.int8),
            "team_needs": np.array(team_needs_feat, dtype=np.int8),
            "next_opponent_needs": np.array(next_opponent_needs_feat, dtype=np.int8),
            "projected_pts": np.array(projected_pts_feat, dtype=np.float32),
            "difference_with_replacement": np.array(diff_with_replacement_feat, dtype=np.float32),
            "hurt_score": np.array(hurt_score_feat, dtype=np.float32),
            "difference_with_current_worst_starter": np.array(diff_with_worst_starter_feat, dtype=np.float32)
        }
        
        if self.flatten_obs:
            return np.concatenate([
                obs_dict["action_mask"],
                obs_dict["pos_available"],
                obs_dict["team_needs"], 
                obs_dict["next_opponent_needs"],
                obs_dict["projected_pts"],
                obs_dict["difference_with_replacement"],
                obs_dict["hurt_score"],
                obs_dict["difference_with_current_worst_starter"]
            ]).astype(np.float32)
        else:
            return obs_dict
    
    def render(self):
        round_num = self.current_pick // self.num_teams + 1
        print(f"\n--- Round {round_num} ---")       
        print(f"Current pick: {self.current_pick}, Agent: {self.agent_selection}")
        for agent in self.possible_agents:
            roster_size = len(self.team_info[agent]['roster'])
            print(f"{agent}: {roster_size} players_drafted")
        print(f"Remaining players: {len(self.available_players)}")

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def current_agent(self):
        if self.current_pick >= self.total_picks:
            return None
        agent_index = self.draft_order[self.current_pick]
        return self.possible_agents[agent_index]
    
    def _get_hurt_score(self, agent):
        diffs = self._get_difference_with_replacement()
        next_opponent_needs = self._get_next_opponent_needs(agent) # returns binary array
        return np.multiply(diffs, next_opponent_needs)

    def _get_difference_with_replacement(self):
        result = []
        for pos in self.draftable_positions:
            if len(self.pos_dqs[pos].diffs) > 0:
                result.append(max(0, self.pos_dqs[pos].diffs[0]))
            else:
                result.append(0)
        return result
    
    def _get_top_pos_proj_pts(self):
        result = []
        for pos in self.draftable_positions:
            if len(self.pos_dqs[pos].players) > 0:
                result.append(max(0, self.pos_dqs[pos].players[0][1]))
            else:
                result.append(0)
        return result     


    def _get_team_needs(self, agent):
        needs = np.zeros(len(self.draftable_positions), dtype=np.int8)
        info = self.team_info[agent]
        pos_counts = info['pos_counts']
        
        if pos_counts['QB'] < self.position_limits['QB']:
            needs[self.position_str_to_index['QB']] = 1
            
        current_flex_players_count = len([p for p in info['pos_roster']['FLEX'] if p is not None])
        has_flex_room = current_flex_players_count < self.position_limits['FLEX']
        
        for pos in self.flex_positions:
            has_dedicated_room = pos_counts[pos] < self.position_limits[pos]
            
            if has_dedicated_room or has_flex_room:
                needs[self.position_str_to_index[pos]] = 1
                
        return needs

    def _can_draft_position(self, agent, pos):
        info = self.team_info[agent]
        
        if len(info['roster']) >= self.full_team_number:
            return False

        if info['pos_counts'][pos] < self.position_limits[pos]:
            return True

        if pos in self.flex_positions:
            current_flex_players = [p for p in info['pos_roster']['FLEX'] if p is not None]
            if len(current_flex_players) < self.position_limits['FLEX']:
                return True

        current_bench_players = [p for p in info['pos_roster']['BENCH'] if p is not None]
        if len(current_bench_players) < self.position_limits['BENCH']:
            return True

        return False

    def _flex_room(self, agent):
        excess_pos = sum([max(0, self.team_info[agent]['pos_counts'][pos] - self.position_limits[pos]) for pos in self.flex_positions])
        flex_room = excess_pos < self.position_limits['FLEX']
        return flex_room, excess_pos
    
    def _get_next_opponent_needs(self, agent):
        next_agent = self._get_next_opponent(agent)
        return self._get_team_needs(next_agent)
    
    def _get_next_opponent(self, agent):
        current_agent_idx_in_possible = self.possible_agents.index(agent)
        
        for i in range(self.current_pick + 1, self.total_picks):
            next_drafter_index_in_draft_order = self.draft_order[i]
            next_drafter_agent = self.possible_agents[next_drafter_index_in_draft_order]
            if next_drafter_agent != agent:
                return next_drafter_agent
        
        other_agents = [a for a in self.agents if a != agent]
        if other_agents:
            return other_agents[0]
        else:
            return agent # Fallback: if no other agent exists

    def _get_available_pos(self):
        return [int(len(self.pos_dqs[pos].players)>0) for pos in self.draftable_positions]
    
    def _get_diff_with_current_worst_starter(self, agent):
        top_pts = self._get_top_pos_proj_pts()
        agent_roster_worst_pos = []
        
        for pos in self.draftable_positions:
            pos_projections = [proj for proj in self.team_info[agent]['pos_projections'][pos] if proj > 0]
            
            if self.team_info[agent]['pos_counts'][pos] < self.position_limits[pos]:
                min_proj_for_pos = 0 
            elif pos_projections:
                min_proj_for_pos = min(pos_projections)
            else:
                min_proj_for_pos = 0 
            
            if pos in self.flex_positions and self.position_limits['FLEX'] > 0:
                current_flex_projections = [proj for proj in self.team_info[agent]['pos_projections']['FLEX'] if proj > 0]
                if current_flex_projections:
                    min_flex_proj = min(current_flex_projections)
                    if min_proj_for_pos == 0: 
                        min_proj_for_pos = min_flex_proj
                    else:
                         min_proj_for_pos = min(min_proj_for_pos, min_flex_proj) 

            agent_roster_worst_pos.append(min_proj_for_pos)

        return np.array(top_pts) - np.array(agent_roster_worst_pos)


    def _get_draft_order(self):
        draft_order = []
        if self.snake_draft:
            for round in range(1, self.max_rounds+1):
                if round % 2 == 0:
                    round_order = list(reversed(range(self.num_teams)))
                else:
                    round_order = list(range(self.num_teams))
                draft_order.extend(round_order)
        else:
            draft_order = list(range(self.num_teams)) * self.max_rounds
        return draft_order

    def _get_full_roster_df(self):
        try:
            rows = []
            for agent in self.agents:
                for gsis_id in self.team_info[agent]['roster']:
                    rows.append({"agent": agent, "gsis_id": gsis_id})
            
            if not rows:
                print("Warning: No roster data to create DataFrame")
                return pd.DataFrame()
                
            roster_df = pd.DataFrame(rows)
            
            required_cols = ["gsis_id", "player_name", "position", "fantasy_pts"]
            available_cols = [col for col in required_cols if col in self.player_df.columns]
            
            if "fantasy_pts" not in available_cols:
                if "median_prediction" in self.player_df.columns:
                    temp_df = self.player_df.copy()
                    temp_df["fantasy_pts"] = temp_df["median_prediction"]
                    available_cols.append("fantasy_pts")
                else:
                    print("Warning: No fantasy_pts or median_prediction column found")
                    return roster_df 
            
            return roster_df.merge(
                self.player_df[available_cols], 
                on="gsis_id", 
                how="left"
            )
            
        except Exception as e:
            print(f"Error in _get_full_roster_df: {e}")
            return pd.DataFrame()
    
    def _get_optimized_lineup(self, df):
        try:
            if df.empty:
                return pd.DataFrame()
                
            lineup = []
            used_ids = set() 

            for pos, limit in self.position_limits.items():
                if pos not in ['FLEX', 'BENCH']:
                    pos_players = df[(df['position'] == pos) & (~df['gsis_id'].isin(used_ids))]
                    if not pos_players.empty:
                        top_players = pos_players.nlargest(min(limit, len(pos_players)), 'fantasy_pts')
                        lineup.append(top_players)
                        used_ids.update(top_players['gsis_id'])
            
            flex_pool = df[(df['position'].isin(self.flex_positions)) & (~df['gsis_id'].isin(used_ids))]
            if not flex_pool.empty and self.position_limits.get('FLEX', 0) > 0:
                flex_players = flex_pool.nlargest(min(self.position_limits['FLEX'], len(flex_pool)), 'fantasy_pts')
                lineup.append(flex_players)
                used_ids.update(flex_players['gsis_id'])
            
            return pd.concat(lineup) if lineup else pd.DataFrame()
            
        except Exception as e:
            print(f"Error in _get_optimized_lineup: {e}")
            return pd.DataFrame()
    
    def _get_optimized_score(self, df):
        lineup = self._get_optimized_lineup(df)
        return lineup['fantasy_pts'].sum()
    
    def _calculate_final_rewards(self):
        if self.full_roster_df is None or len(self.full_roster_df) == 0:
            print("Warning: No roster data available for final rewards")
            for agent in self.agents:
                self.rewards[agent] += -50
            return
        
        try:
            final_scores = {}
            for agent in self.agents:
                agent_df = self.full_roster_df[self.full_roster_df['agent'] == agent]
                if len(agent_df) > 0:
                    final_scores[agent] = self._get_optimized_score(agent_df)
                else:
                    final_scores[agent] = 0
                    print(f"Warning: No players found for {agent}")
            
            if not final_scores:
                return
                
            max_score = max(final_scores.values())
            
            for agent in self.agents:
                agent_score = final_scores[agent]
                relative_reward = (agent_score - max_score) * 2
                
                if agent_score == max_score:
                    relative_reward += 100
                
                self.rewards[agent] += relative_reward
                
                self.infos[agent].update({
                    'final_score': agent_score,
                    'max_score': max_score,
                    'relative_reward': relative_reward,
                    'won_draft': agent_score == max_score
                })
                
            print(f"Final scores: {final_scores}")
            
        except Exception as e:
            print(f"Error calculating final rewards: {e}")
            for agent in self.agents:
                self.rewards[agent] += -100

    
    def _get_draft_pick_reward(self, agent, action):
        value_over_replacement = self._get_difference_with_replacement()[action]
        hurt_score = self._get_hurt_score(agent)[action]

        prev_round = self.current_pick // self.num_teams
        round_multiplier = max(0.5, 1-(prev_round)*0.05)

        in_draft_reward = round_multiplier * (value_over_replacement + hurt_score)

        self.draft_pick_reward_values[f'{agent}_{self.current_pick}'] = {
            "value_over_replacement": value_over_replacement,
            "hurt_score": hurt_score,
            "round_multiplier": round_multiplier,
            "total_in_draft_reqard": in_draft_reward
        }

        self.rewards[agent] += in_draft_reward
# --- Your DraftEnvironment Code (END) ---


# --- Tianshou Integration Test Script ---

# 1. Create a dummy player_df
# This DataFrame needs enough players for the environment to run without errors,
# especially for sampling and position-based lookups.
# Ensure 'gsis_id', 'player_name', 'position', and 'median_prediction' columns exist.
data = {
    'gsis_id': [f'player_{i:03d}' for i in range(200)],
    'player_name': [f'Player Name {i}' for i in range(200)],
    'position': ['QB', 'RB', 'WR', 'TE'] * 50, # Distribute positions
    'median_prediction': np.random.rand(200) * 100 + 50 # Random fantasy points
}
player_df_example = pd.DataFrame(data)

print("--- Starting Tianshou Integration Test ---")
print(f"Dummy Player DataFrame Head:\n{player_df_example.head()}")

# 2. Initialize your custom environment, making sure to use flatten_obs=True for simpler observation space handling with CommonNet
# If flatten_obs=False, you would need a more complex network architecture (e.g., a DictNet from Tianshou or custom preprocessor)
env = DraftEnvironment(player_df=player_df_example, num_teams=2, rounds=2, flatten_obs=True, random_pool_size=50) # Reduced rounds for quicker test

# Wrap the PettingZoo environment for Tianshou
tianshou_env = ts.env.PettingZooEnv(env)

# Determine the observation and action spaces for Tianshou's policy
# Tianshou's PettingZooEnv will automatically handle the multi-agent aspect for policy inputs
# Get observation and action spaces for a single agent from the wrapped environment
# (Assuming all agents have identical observation and action spaces)
obs_shape = tianshou_env.observation_space.shape
action_shape = tianshou_env.action_space.n # Discrete action space

print(f"\nEnvironment observation space shape: {obs_shape}")
print(f"Environment action space size: {action_shape}")
print(f"Possible agents: {tianshou_env.agents}")

# 3. Define a simple neural network for both actor and critic
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu" # For testing, CPU is fine and avoids CUDA issues

# CommonNet combines the actor and critic networks
net = ts.utils.net.CommonNet(
    observation_shape=obs_shape,
    action_shape=action_shape,
    hidden_sizes=[64, 64],
    device=device
).to(device)

actor = ts.policy.PPOPolicy.actor(net).to(device)
critic = ts.policy.PPOPolicy.critic(net).to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# 4. Create PPO policy for each agent and manage them with MultiAgentPolicyManager
policy_map = {}
for agent_id in tianshou_env.agents:
    policy_map[agent_id] = ts.policy.PPOPolicy(
        actor,
        critic,
        optim,
        dist_fn=torch.distributions.Categorical, # For discrete action spaces
        action_space=tianshou_env.action_space, # Need to pass the gym.spaces object
        # PPO specific hyperparameters (minimal for test)
        discount_factor=0.99,
        max_grad_norm=0.5,
        vf_coef=0.25,
        ent_coef=0.0, # No entropy regularization for simple test
        reward_normalization=False,
        advantage_normalization=True,
        eps_clip=0.2,
        value_clip=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        critic_learn_phase=1,
        actor_learn_phase=1,
        # Other PPO params can be added here
    )

# The MultiAgentPolicyManager handles agent selection and policy dispatch
policy = ts.policy.MultiAgentPolicyManager(policy_map, tianshou_env.agents)

print("\nPolicies and Policy Manager created.")

# 5. Set up collectors
# The collector interacts with the environment to gather transitions
# num_train_envs defines how many environments to run in parallel for training
num_train_envs = 1
train_envs = ts.env.DummyVectorEnv([lambda: ts.env.PettingZooEnv(DraftEnvironment(player_df=player_df_example, flatten_obs=True, num_teams=2, rounds=2, random_pool_size=50)) for _ in range(num_train_envs)])
buffer = ts.data.VectorReplayBuffer(total_size=1000, buffer_num=len(train_envs))
collector = ts.data.Collector(policy, train_envs, buffer, exploration_noise=True)

print(f"\nCollector created with buffer size {len(buffer)} and {num_train_envs} parallel environments.")

# 6. Run a dummy training loop (just a few steps to test pipeline)
print("\nCollecting initial data (warm-up)...")
collector.collect(n_step=100) # Collect 100 steps from the environment

print("\nStarting a very short dummy training loop (1 epoch)...")
result = ts.trainer.onpolicy_trainer(
    policy,
    collector,
    test_collector=None, # No test collector for this basic test
    max_epoch=1,         # Just 1 epoch to see if it runs
    step_per_epoch=10,   # Collect 10 steps per epoch
    repeat_per_collect=1,
    episode_per_collect=1,
    batch_size=4,
    train_fn=lambda epoch, env_step: print(f"Epoch {epoch}, Env Step {env_step}"),
    stop_fn=lambda mean_rewards: False, # Never stop prematurely
    verbose=True,
    show_progress=True,
)

print("\n--- Tianshou Test Complete ---")
print(f"Training Result (dummy): {result}")
print("If the script ran without major errors, Tianshou is likely integrated with your environment.")
print("Next steps: Increase training steps, tune hyperparameters, and visualize learning.")

# Clean up (optional)
train_envs.close()

