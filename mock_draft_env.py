import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

@dataclass
class PositionDQ:
     players: Deque[Tuple[str, float]]
     diffs: Deque[float]

class MockDraftEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
        "is_parallelizable": True,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14, random_pool_size=100, real_scores_at_draft_end=False, render_mode=None, flatten_obs=False, your_team_id=0):
        super().__init__()

        self.render_mode = render_mode
        self.flatten_obs = flatten_obs

        self.your_team_id = your_team_id # Human team
        
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

        self.draft_order = self._get_draft_order()

        self.invalid_action_penalty = -50

    def _initialize_player_metadata(self):
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
        if pos not in self.draftable_positions:
                raise ValueError(f"{pos} not in possible starting positions: {self.draftable_positions}")
        
        if not self.pos_player_pool[pos]:
            return deque()
        
        return deque(sorted([(id, self.gsis_to_projections.get(id, 0)) for id in self.pos_player_pool[pos]],
                                key=lambda x: x[1],
                                reverse=True))
    
    def _create_diffs_dq(self, pos_dq):
        if not pos_dq:
            return deque()
        dq = [proj_pts for _, proj_pts in pos_dq]
        dq.append(0)
        return deque([a - b for a, b in zip(dq, dq[1:])])
    
    def _update_pos_dqs(self, pos):
        if len(self.pos_dqs[pos].players) > 0:
            self.pos_dqs[pos].players.popleft()
        if len(self.pos_dqs[pos].diffs) > 0:
            self.pos_dqs[pos].diffs.popleft()
        # What happens when these cannot execute?

    def _initialize_spaces(self):
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
        self.possible_agents = [f"team_{i}" for i in range(self.num_teams)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
    
        self.current_pick = 0
        self.draft_history = []
        self.agents = self.possible_agents[:]
        
        self._initialize_player_metadata()
        
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
        
        if self.agent_selection:
            return self.observe(self.agent_selection)
        return self.observe(self.possible_agents[0]), {}


    def step(self, action):
        if self.agent_selection is None:
            return

        agent = self.agent_selection

        step_reward = 0

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= len(self.draftable_positions):
            print(f'[Invalid Action] {agent} attempted invalid action (action={action}). Penalizing and skipping turn.')
            step_reward = self.invalid_action_penalty
            self.rewards[agent] = step_reward
            print(f"Rewards after invalid action: {self.rewards}")
            self._advance_draft(None)
            return
        
        player = self._get_player_from_action(action)
        if player:
            draft_successfull, step_reward = self._draft_player(agent, action, player)
            if draft_successfull:
                self.rewards[agent] = step_reward
                #print(f"[DEBUG] Reward assigned to {agent}: {self.rewards}")
                self._advance_draft(action)
            else:
                print(f"[Invalid Pick (Game Logic)] {agent} attempted invalid selection (action={action}). Penalizing and skipping turn.")
                step_reward = self.invalid_action_penalty
                self.rewards[agent] = step_reward
                print(f"Rewards after invalid: {self.rewards}")
                self._advance_draft(None)
        else:
            print(f"[Invalid Pick (No Player)] {agent} attempted invalid selection (action={action}). Needs to retry.")
            step_reward = self.invalid_action_penalty
            self.rewards[agent] = step_reward
            print(f"Rewards: {self.rewards}")
            self._advance_draft(None)

    def _was_dead_step(self, action):
        if self.current_pick < self.total_picks:
            self._advance_draft(None)

    def _get_player_from_action(self, action):
        position = self.draftable_positions[action]
        player_queue = self.pos_dqs[position].players
        diffs_queue = self.pos_dqs[position].diffs
        if position not in self.pos_dqs:
            print(f"[ERROR] pos_dqs missing for position {position}")
            return None

        while player_queue:
            player_id, _ = player_queue[0]
            if player_id in self.available_players:
                return player_id
            else:
                print(f"DEBUG: Player {player_id} at top of {position} queue is not available. Removing.")
                player_queue.popleft()
                if diffs_queue:
                    diffs_queue.popleft()
        return None

    def _draft_player(self, agent, action, player):
        # Ensure the player is valid and available
        if player not in self.available_players:
            print(f'player {self.gsis_to_name[player]} not available.')
            return False, 0
        
        succussful_update = self._update_team_info(agent, action, player)
        if not succussful_update:
            print('Unsuccessful update of team info')
            return False, 0
        
        # Remove the player from available players and update draft history
        self.available_players.remove(player)
        self.draft_history.append((agent, player))
        step_reward = self._get_draft_pick_reward(agent, action)

        player_name = self.gsis_to_name.get(player, 'Unknown')
        position = self.draftable_positions[action]
        #print(f"{agent} drafted {player_name} ({position}) - Reward: {step_reward:.2f}")
        
        return True, step_reward
    
    def _update_team_info(self, agent, action, player):
        player_pos = self.draftable_positions[action]
        player_proj = self.gsis_to_projections[player]

        # Update the team roster and positions such that position players are chosen first, then FLEX, then BENCH

        success = False

        current_flex_players_count = len([player for player in self.team_info[agent]['pos_roster']['FLEX'] if player is not None])
        current_bench_players_count = len([player for player in self.team_info[agent]['pos_roster']['BENCH'] if player is not None])

        if self.team_info[agent]['pos_counts'][player_pos] < self.position_limits[player_pos]:
            idx = self.team_info[agent]['pos_counts'][player_pos]
            self.team_info[agent]['pos_roster'][player_pos][idx] = player
            self.team_info[agent]['pos_projections'][player_pos][idx] = player_proj
            self.team_info[agent]['pos_counts'][player_pos] += 1
            self.team_info[agent]['roster'].append(player)
            success = True

        elif player_pos in self.flex_positions and (current_flex_players_count < self.position_limits['FLEX']):
            idx = current_flex_players_count
            self.team_info[agent]['pos_roster']['FLEX'][idx] = player
            self.team_info[agent]['pos_projections']['FLEX'][idx] = player_proj
            self.team_info[agent]['pos_counts'][player_pos] += 1
            self.team_info[agent]['roster'].append(player)
            success = True

        elif current_bench_players_count < self.position_limits['BENCH']:
            idx = current_bench_players_count
            self.team_info[agent]['pos_roster']['BENCH'][idx] = player
            self.team_info[agent]['pos_projections']['BENCH'][idx] = player_proj
            self.team_info[agent]['pos_counts'][player_pos] += 1
            self.team_info[agent]['roster'].append(player)
            success = True

        if not success:
            print(f'No room left on team {agent} left to draft {self.gsis_to_name.get(player)} ({player_pos}). Roster full.')

        return success
        
    def _advance_draft(self, action):
        #print(f"_advance_draft: current_pick={self.current_pick}, total_picks={self.total_picks}")

        if action is not None:
            position = self.draftable_positions[action]
            self._update_pos_dqs(position)

        if self.current_pick >= self.total_picks - 1:
            #print(f"Draft complete: processed pick {self.current_pick} (final pick)")
            self.current_pick += 1 
            self._finalize_draft()
            return
        
        self.current_pick += 1
        #print(f"Advanced to pick {self.current_pick}/{self.total_picks}")

        next_agent = self.current_agent()
        if next_agent is None:
            print(f"No valid next agent found for pick {self.current_pick}, finalizing draft")
            self._finalize_draft()
            return
        
        self.agent_selection = next_agent
        #print(f'Current agent now {self.agent_selection}')
        
        if self.current_pick >= self.total_picks:
            #print(f"Draft completion detected after agent assignment")
            self._finalize_draft()

    def _finalize_draft(self):
        try:
            should_terminate = True # Can remove?
            
            try:
                self.full_roster_df = self._get_full_roster_df()
                if not self.full_roster_df.empty:
                    self.optimized_lineups = self.full_roster_df.groupby('agent').apply(self._get_optimized_lineup)
                    #print("Optimized lineups calculated successfully")
            except Exception as e:
                print(f"Error building roster DF: {e}")

            try:
                self._calculate_final_rewards()
                #print("Final rewards calculated successfully")
            except Exception as e:
                print(f"Error calculating final rewards: {e}")
                for agent in self.possible_agents:
                    self.rewards[agent] = -100

            for agent in self.possible_agents:
                self.terminations[agent] = True
                if not isinstance(self.infos[agent], dict):
                    self.infos[agent] = {}
                self.infos[agent]['draft_completed'] = True
                
            print(f"[FINAL] All agents terminated. Final rewards: {self.rewards}")
            
            self.agent_selection = None

        except Exception as e:
            print(f"Critical error in _finalize_draft: {e}")
            for agent in self.possible_agents:
                self.terminations[agent] = True
                self.rewards[agent] = -100
                if not isinstance(self.infos[agent], dict):
                    self.infos[agent] = {}
            self.agent_selection = None

    def observe(self, agent):
        #print(f"[OBSERVE] agent: {agent} | Pick: {self.current_pick}/{self.total_picks}, term={self.terminations.get(agent)} trunc={self.truncations.get(agent)}")
        if agent is None:
            print("Observe agent is none.")
            return {}
        
        if (self.terminations.get(agent, False) or 
                self.truncations.get(agent, False) or 
                self.current_pick >= self.total_picks):            
            num_draftable_positions = len(self.draftable_positions)
            if self.flatten_obs:
                return np.zeros(num_draftable_positions * 8, dtype=np.float32)
            else:
                return
        
        action_mask = [self._can_draft_position(agent, pos) for pos in self.draftable_positions]

        obs_dict = {
            "action_mask": np.array(action_mask, dtype=np.int8),
            "pos_available": np.array(self._get_available_pos(), dtype=np.int8),
            "team_needs": np.array(self._get_team_needs(agent), dtype=np.int8),
            "next_opponent_needs": np.array(self._get_next_opponent_needs(agent), dtype=np.int8),
            "projected_pts": np.array(self._get_top_pos_proj_pts(), dtype=np.float32),
            "difference_with_replacement": np.array(self._get_difference_with_replacement(), dtype=np.float32),
            "hurt_score": np.array(self._get_hurt_score(agent), dtype=np.float32),
            "difference_with_current_worst_starter": np.array(self._get_diff_with_current_worst_starter(agent), dtype=np.float32)
        }
        
        if self.flatten_obs:
            flat_obs = np.concatenate([
                obs_dict["action_mask"],
                obs_dict["pos_available"],
                obs_dict["team_needs"], 
                obs_dict["next_opponent_needs"],
                obs_dict["projected_pts"],
                obs_dict["difference_with_replacement"],
                obs_dict["hurt_score"],
                obs_dict["difference_with_current_worst_starter"]
            ]).astype(np.float32)
            #print(f"[OBSERVE] Flattened obs shape: {flat_obs.shape}, dtype: {flat_obs.dtype}")
            return flat_obs
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
            #print(f"Draft complete: pick {self.current_pick} >= total {self.total_picks}")
            return None
        if self.current_pick >= len(self.draft_order):
            print(f"Error: pick {self.current_pick} exceeds draft_order length {len(self.draft_order)}")
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
        for i in range(self.current_pick + 1, self.total_picks):
            next_drafter_index = self.draft_order[i]
            next_drafter_agent = self.possible_agents[next_drafter_index]
            if next_drafter_agent != agent:
                return next_drafter_agent
        other_agents = [a for a in self.agents if a!=agent]
        return other_agents[0] # just return a different agent
    
    def _get_available_pos(self):
        return [int(len(self.pos_dqs[pos].players)>0) for pos in self.draftable_positions]
    
    def _get_diff_with_current_worst_starter(self, agent):
        top_pts = self._get_top_pos_proj_pts()
        agent_roster_worst_pos = []
        flex_room, _ = self._flex_room(agent)
        
        if not flex_room and len(self.team_info[agent]['pos_projections']['FLEX']) > 0:
            flex_projections = [proj for proj in self.team_info[agent]['pos_projections']['FLEX'] if proj > 0]
            if flex_projections:
                min_flex_proj_pts = min(flex_projections)
            else:
                min_flex_proj_pts = 0
            
            qb_projections = [proj for proj in self.team_info[agent]['pos_projections']['QB'] if proj > 0]
            min_qb_proj = min(qb_projections) if qb_projections else 0
            
            agent_roster_worst_pos = [min_qb_proj, min_flex_proj_pts, min_flex_proj_pts, min_flex_proj_pts]
        else:
            for pos in self.draftable_positions:
                pos_projections = [proj for proj in self.team_info[agent]['pos_projections'][pos] if proj > 0]
                min_proj = min(pos_projections) if pos_projections else 0
                agent_roster_worst_pos.append(min_proj)
        
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

            temp_df = self.player_df.copy()
            temp_df["fantasy_pts"] = temp_df["median_prediction"]

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
            for pos, limit in self.position_limits.items():
                if pos not in ['FLEX', 'BENCH']:
                    pos_players = df[df['position'] == pos]
                    if not pos_players.empty:
                        top_players = pos_players.nlargest(min(limit, len(pos_players)), 'fantasy_pts')
                        lineup.append(top_players)
            
            if lineup:
                used_ids = pd.concat(lineup)['gsis_id'].tolist()
            else:
                used_ids = []
  
            flex_pool = df[(df['position'].isin(self.flex_positions)) & (~df['gsis_id'].isin(used_ids))]
            if not flex_pool.empty and self.position_limits.get('FLEX', 0) > 0:
                flex_players = flex_pool.nlargest(min(self.position_limits['FLEX'], len(flex_pool)), 'fantasy_pts')
                lineup.append(flex_players)
            
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
                self._cumulative_rewards[agent] += -50
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
                agent_score = final_scores.get(agent, 0)
                relative_reward = (agent_score - max_score) * 2
                
                if agent_score == max_score:
                    relative_reward += 100
                
                self.rewards[agent] += relative_reward
                self._cumulative_rewards[agent] += relative_reward
                
                self.infos[agent].update({
                    'final_score': agent_score,
                    'max_score': max_score,
                    'relative_reward': relative_reward,
                    'won_draft': agent_score == max_score,
                    'cumulative_reward': self._cumulative_rewards[agent]
                })
                
            #print(f"Final scores: {final_scores}")
            #print(f"Final cumulative rewards: {self._cumulative_rewards}")
            
        except Exception as e:
            print(f"Error calculating final rewards: {e}")
            for agent in self.agents:
                self.rewards[agent] += -100
                self._cumulative_rewards[agent] += -100

    
    def _get_draft_pick_reward(self, agent, action):
        # Value Over Replacement + Hurt Score
        value_over_replacement = self._get_difference_with_replacement()[action]
        hurt_score = self._get_hurt_score(agent)[action]


        prev_round = self.current_pick // self.num_teams
        round_multiplier = max(0.5, 1-(prev_round)*0.05)

        in_draft_reward = round_multiplier * (value_over_replacement + hurt_score)
        #print(f"[REWARD DEBUG] Agent: {agent}, Pick: {self.current_pick}")
        #print(f"  Value over replacement: {value_over_replacement:.2f}")
        #print(f"  Hurt score: {hurt_score:.2f}")
        #print(f"  Round multiplier: {round_multiplier:.2f}")
        #print(f"  Total reward: {in_draft_reward:.2f}")

        self.draft_pick_reward_values[f'{agent}_{self.current_pick}'] = {
            "value_over_replacement": value_over_replacement,
            "hurt_score": hurt_score,
            "round_multiplier": round_multiplier,
            "total_in_draft_reqard": in_draft_reward
        }
        #print("value_over_replacement:", value_over_replacement)
        #print("hurt_score:", hurt_score)
        #print("round_multiplier:", round_multiplier)

        return in_draft_reward



    """
    def _get_named_team_positions_roster(self):
        return {
            agent: {
                pos: [self.gsis_to_name.get(pid) if pid is not None else None for pid in players]
                for pos, players in positions.items()
            }
            for agent, positions in self.team_positions_roster.items() # Need to update
        }
    """