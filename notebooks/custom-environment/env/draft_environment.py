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
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14, random_pool_size=100):
        super().__init__()
        
        self.player_df = player_df
        self.random_pool_size = random_pool_size

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

        self.possible_starting_positions = ['QB', 'RB', 'WR', 'TE']
        self.flex_positions = ['RB', 'WR', 'TE']

        self._initialize_agents()
        self._initialize_player_metadata()
        self._initialize_spaces()

        self.position_str_to_index = {pos: i for i, pos in enumerate(self.possible_starting_positions)}

        # Collect all available players
        self.available_players = self.player_pool.copy()

        # Draft tracking
        self.draft_order = self._get_draft_order()

    def _initialize_player_metadata(self):
        self.player_pool_df = self.player_df.sample(self.random_pool_size)
        self.gsis_to_name = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['player_name']))
        self.gsis_to_position = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['position']))
        self.gsis_to_projections = dict(zip(self.player_pool_df['gsis_id'], self.player_pool_df['median_prediction']))

        self.pos_player_pool = self.player_pool_df.groupby('position')['gsis_id'].agg(list).to_dict()

        self.pos_dqs = {}

        for pos in self.possible_starting_positions:
             player_id_and_projections = self._sort_and_create_dq(pos)
             diffs = self._create_diffs_dq(player_id_and_projections)
             self.pos_dqs[pos] = PositionDQ(players=player_id_and_projections, diffs=diffs)

    def _sort_and_create_dq(self, pos):
            if pos not in self.possible_starting_positions:
                 raise ValueError(f"{pos} not in possible starting positions: {self.possible_starting_positions}")
            return deque(sorted([(id, self.gsis_to_projections[id]) for id in self.pos_player_pool[pos]],
                                 key=lambda x: x[1],
                                 reverse=True))
    
    def _create_diffs_dq(self, pos_dq):
         dq = [proj_pts for _, proj_pts in pos_dq]
         dq.append(0)
         return deque([a - b for a, b in zip(dq, dq[1:])])
    
    def _update_pos_dqs(self, pos):
         self.pos_dqs[pos].players.popleft()
         self.pos_dqs[pos].diffs.popleft()

    def _initialize_spaces(self):
        self._action_spaces = {
            agent: spaces.Discrete(4) for agent in self.agents
        }
        self._observation_spaces = {
            agent: spaces.Dict({
                "pos_available": spaces.MultiBinary(4),
                "team_needs": spaces.MultiBinary(4),
                "next_opponent_needs": spaces.MultiBinary(4),
                "projected_pts": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "difference_with_replacement": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "hurt_score": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "difference_with_current_worst_starter": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            }) for agent in self.agents
        }

    def _initialize_agents(self):
        self.possible_agents = [f"team_{i}" for i in range(self.num_teams)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}


    def reset(self):
        # Reset the environment to its initial state
        # Will need to be called at the start of each new draft
        self.current_pick = 0
        self.draft_history = []
        self.agents = self.possible_agents[:]
        self.agent_selection = self.current_agent()

        self.team_rosters = {agent: [] for agent in self.possible_agents}
        self.team_positions = {agent: {pos: 0 for pos in self.position_limits} for agent in self.possible_agents}
        self.team_positions_roster = {
            agent: {
                pos: [None] * self.position_limits[pos]
                for pos in self.position_limits
            }
            for agent in self.agents
        }
        self.team_positions_projections = {
            agent: {
                pos: [0] * self.position_limits[pos]
                for pos in self.position_limits
            }
            for agent in self.agents
        }
        self.team_positions_available = {
            agent: {
                pos: 1 for pos in self.position_limits
            }
            for agent in self.agents
        }

        self.full_roster_df = None
        self.optimized_lineups = None

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #self._cumulative_rewards = {agent: 0 for agent in self.agents}


    def step(self, action):
        # Ensure the action is valid
        assert self.agent_selection is not None

        agent = self.agent_selection

        # If the agent has already terminated, skip the step
        if self.terminations[agent]:
            super()._was_dead_step(action)
            return
        
        player = self._get_player_fom_action(action)
        if player and self._draft_player(agent, player):
            self._advance_draft(agent, player)
        else:
            self.rewards[agent] = 0 # or -1 if we want to penalize invalid picks
            tqdm.write(f"[Invalid Pick] {agent} attempted invalid selection (action={action}). Needs to retry.")

    def observe(self, agent):
        return {
                "pos_available": self._get_available_pos, # if no more QBs left in pool (for example) return [0, 1, 1, 1]
                "team_needs": self._get_team_needs(agent),
                "next_opponent_needs": self._get_next_opponent_needs(agent),
                "pos_projected_pts": self._get_top_pos_proj_pts(),
                "difference_with_replacement": self._get_difference_with_replacement(),
                "hurt_score": self._get_hurt_score(agent),
                "difference_with_current_worst_starter": self._get_diff_with_current_worst_starter(agent)
        }
    
    def _get_hurt_score(self, agent):
        diffs = self._get_difference_with_replacement()
        next_opponent_needs = self._get_next_opponent_needs(agent) # returns binary array
        return np.multiply(diffs, next_opponent_needs)

    def _get_difference_with_replacement(self):
        return [self.pos_dqs[pos].diffs[0] for pos in self.possible_starting_positions]
    
    def _get_top_pos_proj_pts(self):
        return [self.pos_dqs[pos].players[0][1] for pos in self.possible_starting_positions]
    
    def _get_team_needs(self, agent):
        if self.team_positions_available[agent]['FLEX'] & all(self.team_positions_available[agent][flex_pos]==0 for flex_pos in self.flex_positions):
            return [self.team_positions_available[agent]['QB'], 1, 1, 1]
        return [self.team_positions_available[agent][pos] for pos in self.possible_starting_positions]
    
    def _get_next_opponent_needs(self, agent):
        next_agent = self._get_next_opponent(agent)
        return self._get_team_needs(next_agent)
    
    def _get_next_opponent(self, agent):
        agent = int(agent.str.replace('team_', ''))
        next_agent = agent
        pick_num = self.current_pick
        while next_agent == agent and pick_num < self.total_picks:
            next_agent = self.draft_order[pick_num+1]
        return f'team_{next_agent}'
    
    def _get_available_pos(self):
        return [int(len(self.pos_dqs[pos].players)>0) for pos in self.possible_starting_positions]
    
    def _get_diff_with_current_worst_starter(self, agent):
        # Need to handle FLEX as well
        agent_roster_worst_pos = []
        if not self.team_positions_available[agent]['FLEX']:
            flex_proj_pts = self.team_positions_projections[agent]['FLEX']
            agent_roster_worst_pos = [min(self.team_positions_projections[agent]['QB']), flex_proj_pts, flex_proj_pts, flex_proj_pts]
        else:
            for pos in self.possible_starting_positions:
                agent_roster_worst_pos.append(min(self.team_positions_projections[agent][pos]))
        return np.array(self._get_top_pos_proj_pts()) - np.array(agent_roster_worst_pos)
    



    def _get_draft_order(self):
        draft_order = []
        if self.snake_draft:
            for round in range(1, self.max_rounds+1):
                if self.snake_draft and round % 2 == 0:
                    round_order = list(reversed(range(self.num_teams)))
                else:
                    round_order = list(range(self.num_teams))
                draft_order.extend(round_order)
        else:
            draft_order = list(range(self.num_teams)) * self.max_rounds
        return draft_order

    #####################

    #[0, 1, 1, 1] displayed for FLEX