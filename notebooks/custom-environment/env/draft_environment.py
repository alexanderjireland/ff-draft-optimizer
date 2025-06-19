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

class DraftEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
        "is_parallelizable": True,
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14, random_pool_size=100, real_scores_at_draft_end=False, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        
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

    def _initialize_player_metadata(self):
        self.player_pool_df = self.player_df.sample(min(self.random_pool_size, len(self.player_df)))
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
        
        return deque(sorted([(id, self.gsis_to_projections[id]) for id in self.pos_player_pool[pos]],
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
        self._action_spaces = {
            agent: spaces.Discrete(num_draftable_positions) for agent in self.agents
        }
        self._observation_spaces = {
            agent: spaces.Dict({
                "pos_available": spaces.MultiBinary(num_draftable_positions),
                "team_needs": spaces.MultiBinary(num_draftable_positions),
                "next_opponent_needs": spaces.MultiBinary(num_draftable_positions),
                "projected_pts": spaces.Box(low=-20, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                "difference_with_replacement": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                "hurt_score": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32),
                "difference_with_current_worst_starter": spaces.Box(low=-500, high=500, shape=(num_draftable_positions,), dtype=np.float32)
            }) for agent in self.agents
        }

    def _initialize_agents(self):
        self.possible_agents = [f"team_{i}" for i in range(self.num_teams)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}


    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        # Will need to be called at the start of each new draft
        self.current_pick = 0
        self.draft_history = []
        self.agents = self.possible_agents[:]
        self.agent_selection = self.current_agent()

        self._initialize_player_metadata() # reset to new random sample

        # Collect all available players
        self.available_players = self.player_pool.copy()

        #self.team_rosters = {agent: [] for agent in self.possible_agents}
        #self.team_positions = {agent: {pos: 0 for pos in self.position_limits} for agent in self.possible_agents}
        #self.team_positions_roster = { # Can this one be removed?
        #    agent: {
        #        pos: [None] * self.position_limits[pos]
        #        for pos in self.position_limits
        #    }
        #    for agent in self.agents
        #}

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

        return self.observe(self.agent_selection) if self.agent_selection else {}

    def step(self, action):
        # Ensure the action is valid
        if self.agent_selection is None:
            return

        agent = self.agent_selection

        # If the agent has already terminated, skip the step
        if self.terminations[agent] or self.truncations[agent]:
            super()._was_dead_step(None)
            return
        
        if action < 0 or action >= len(self.draftable_positions) or action is None:
            print(f'[Invalid Action] {agent} attempted invalid action (action={action}). Penalizing and skipping turn.')
            self.rewards[agent] += self.invalid_action_penalty
            self._advance_draft(None)
            return
        
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

    def _get_player_from_action(self, action):
        # returns player id for action/position selection
        # action is an int between 0 and 3 for the four draftable positions
        position = self.draftable_positions[action]
        player_queue = self.pos_dqs[position].players
        if not player_queue:
            return None
        
        player_id, _ = player_queue[0]
        return player_id

    def _draft_player(self, agent, action, player):
        # Ensure the player is valid and available
        if player not in self.available_players:
            return False
        
        succusesfull_update = self._update_team_info(agent, action, player)
        if not succusesfull_update:
            return False
        
        # Remove the player from available players and update draft history
        self.available_players.remove(player)
        self.draft_history.append((agent, player))
        self._get_draft_pick_reward(agent, action)
        return True
    
    def _update_team_info(self, agent, action, player):
        player_pos = self.draftable_positions[action]
        player_proj = self.gsis_to_projections[player]

        position_room = self.team_info[agent]['pos_counts'][player_pos] < self.position_limits[player_pos]
        flex_room, num_excess_of_starting_players = self._flex_room(agent)
        bench_room = (not position_room) & (not flex_room) & (num_excess_of_starting_players < (self.position_limits['BENCH'] + self.position_limits['FLEX']))

        # Update the team roster and positions such that position players are chosen first, then FLEX, then BENCH
        self.team_info[agent]['roster'].append(player)
        if position_room:
            i = self.team_info[agent]['pos_counts'][player_pos]
            self.team_info[agent]['pos_roster'][player_pos][i] = player
            self.team_info[agent]['pos_projections'][player_pos][i] = player_proj
        elif flex_room:
            i = num_excess_of_starting_players
            self.team_info[agent]['pos_roster']['FLEX'][i] = player
            self.team_info[agent]['pos_projections']['FLEX'][i] = player_proj
        elif bench_room:
            i = num_excess_of_starting_players - self.position_limits['FLEX']
            self.team_info[agent]['pos_roster']['BENCH'][i] = player
            self.team_info[agent]['pos_projections']['BENCH'][i] = player_proj
        else:
            print(f'No room left on team to draft.') # logger out?
            return False

        self.team_info[agent]['pos_counts'][player_pos] += 1
        return True
        
    def _advance_draft(self, action):
        # Advance the draft to next pick
        self.current_pick += 1

        if action:
            position = self.draftable_positions[action]
            self._update_pos_dqs(position)

        if self.current_pick >= self.total_picks:
            self._finalize_draft()
        else:
            #self.agent_name_mapping = self.current_agent() 
            self.agent_selection = self.current_agent() # Now that current pick has incremented
            print(f'Current agent now {self.agent_selection}')

    def _finalize_draft(self):
        self.full_roster_df = self._get_full_roster_df()
        optimized_scores = self.full_roster_df.groupby('agent').apply(self._get_optimized_score)
        self.optimized_lineups = self.full_roster_df.groupby('agent').apply(self._get_optimized_lineup)
        for agent in self.possible_agents:
            self.terminations[agent] = True
            self.rewards[agent] += optimized_scores[agent]
        print(f"_finalize_draft: agents = {self.agents}")
        print(f"_finalize_draft: possible_agents = {self.possible_agents}")

    def observe(self, agent):
        if agent is None:
            return {}
        
        if self.terminations[agent] or self.truncations[agent] or self.current_pick >= self.total_picks:
            num_draftable_positions = len(self.draftable_positions)
            return {
                "pos_available": np.zeros(num_draftable_positions, dtype=np.int8),
                "team_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                "next_opponent_needs": np.zeros(num_draftable_positions, dtype=np.int8),
                "projected_pts": np.zeros(num_draftable_positions, dtype=np.float32),
                "difference_with_replacement": np.zeros(num_draftable_positions, dtype=np.float32),
                "hurt_score": np.zeros(num_draftable_positions, dtype=np.float32),
                "difference_with_current_worst_starter": np.zeros(num_draftable_positions, dtype=np.float32)
            }
        
        return {
                "pos_available": np.array(self._get_available_pos(), dtype=np.int8), # if no more QBs left in pool (for example) return [0, 1, 1, 1]
                "team_needs": np.array(self._get_team_needs(agent), dtype=np.int8),
                "next_opponent_needs": np.array(self._get_next_opponent_needs(agent), dtype=np.int8),
                "projected_pts": np.array(self._get_top_pos_proj_pts(), dtype=np.float32),
                "difference_with_replacement": np.array(self._get_difference_with_replacement(), dtype=np.float32),
                "hurt_score": np.array(self._get_hurt_score(agent), dtype=np.float32),
                "difference_with_current_worst_starter": np.array(self._get_diff_with_current_worst_starter(agent), dtype=np.float32)
        }
    
    def render(self):
        round_num = self.current_pick // self.num_teams + (1 if self.current_pick % self.num_teams != 0 else 0)
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
        counts = self.team_info[agent]['pos_counts']

        if counts['QB'] < self.position_limits['QB']:
            needs[self.position_str_to_index['QB']] = 1
        
        # Determine if FLEX is already filled by how many RBs, WRs, and TEs we have
        flex_room, _ = self._flex_room(agent)

        for pos in self.flex_positions:
            has_pos_need = counts[pos] < self.position_limits[pos]
            if has_pos_need or flex_room:
                needs[self.position_str_to_index[pos]] = 1
        return needs

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
            min_flex_proj_pts = min([proj for proj in self.team_info[agent]['pos_projections']['FLEX']])
            agent_roster_worst_pos = [min(self.team_info[agent]['pos_projections']['QB']), min_flex_proj_pts, min_flex_proj_pts, min_flex_proj_pts]
        else:
            for pos in self.draftable_positions:
                agent_roster_worst_pos.append(min(self.team_info[agent]['pos_projections'][pos]))
        return np.array(self._get_top_pos_proj_pts()) - np.array(agent_roster_worst_pos)

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
        rows = []
        for agent in self.agents:
            for gsis_id in self.team_info[agent]['roster']:
                rows.append({"agent": agent, "gsis_id": gsis_id})
        roster_df = pd.DataFrame(rows)
        print(f'ROSTER DF: {roster_df}')
        return roster_df.merge(self.player_df[["gsis_id", "player_name", "position", "fantasy_pts"]], on="gsis_id", how="left")
    
    def _get_optimized_lineup(self, df):
        #real_scores = self.real_scores
        # Add here logic for taking samples from posterior if real_scores = False
        lineup = []
        for pos, limit in self.position_limits.items():
            if pos not in ['FLEX', 'BENCH']:
                top_players = df[df['position']==pos].nlargest(limit, 'fantasy_pts')
                lineup.append(top_players)
        
        used_ids = pd.concat(lineup)['gsis_id']
        
        flex_pool = df[(df['position'].isin(["RB", "WR", "TE"])) & (~df['gsis_id'].isin(used_ids))]
        lineup.append(flex_pool.nlargest(self.position_limits['FLEX'], 'fantasy_pts'))
        return pd.concat(lineup)
    
    def _get_optimized_score(self, df):
        lineup = self._get_optimized_lineup(df)
        return lineup['fantasy_pts'].sum()
    
    
    def _get_draft_pick_reward(self, agent, action):
        # Value Over Replacement + Hurt Score
        value_over_replacement = self._get_difference_with_replacement()[action]
        hurt_score = self._get_hurt_score(agent)[action]
        in_draft_reward = value_over_replacement + hurt_score
        self.rewards[agent] += in_draft_reward

           

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