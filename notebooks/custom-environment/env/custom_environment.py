from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import pandas as pd
from gymnasium import spaces
import numpy as np

class CustomEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14):
        # Need to add rules about restrictions on positions, FLEX, etc.
        self.player_df = player_df
        self.player_pool = list(player_df['player_name'])
        self.player_positions = dict(zip(player_df['player_name'], player_df['position']))

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
            'BENCH': 8,
        }

        # Initialize state variables
        self.possible_agents = [f"team_{i}" for i in range(num_teams)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}


        # Collect all players
        self.player_pool = self._generate_all_players()
        self.available_players = self.player_pool.copy()

        # Initialize action spaces for each agent (equal to number of players, .i.e., possible draft picks/actions)
        self._action_spaces = {
            agent: spaces.Discrete(len(self.player_pool)) for agent in self.agents
        }

        # Initialize observation spaces for each agent
        self._observation_spaces = {
            agent: spaces.Dict({
                "available_players": spaces.MultiBinary(len(self.player_pool)),
                "team_roster": spaces.MultiBinary(len(self.player_pool)),
                "team_positions": spaces.Dict({
                    pos: spaces.Discrete(limit) for pos, limit in self.position_limits.items()
                })
            }) for agent in self.agents
        }

        # Draft tracking
        self.draft_order = self._get_draft_order()

        self.agents = []
        
    def reset(self):
        # Reset the environment to its initial state
        # Will need to be called at the start of each new draft
        self.current_pick = 0
        self.available_players = self.player_pool.copy()
        self.draft_history = []
        self.team_rosters = {agent: [] for agent in self.possible_agents}
        self.team_positions = {agent: {pos: 0 for pos in self.position_limits} for agent in self.possible_agents}
        self.agents = self.possible_agents[:]
        self.current_pick = 0
        self.agent_selection = self.current_agent()
        self.team_positions_roster = {
            agent: {
                pos: [None] * self.position_limits[pos]
                for pos in self.position_limits
            }
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents} # Figure out how to handle rewards
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

    def observe(self, agent):
        return {
            "available_players": self._player_vector(self.available_players),
            "team_roster": self._player_vector(self.team_rosters[agent]),
            "team_positions": {
                pos: self.team_positions[agent][pos] for pos in self.position_limits
            }
        }
    
    def step(self, action):
        # Ensure the action is valid
        assert self.agent_selection is not None

        # Select the current agent
        agent = self.agent_selection

        # If the agent has already terminated, skip the step
        if self.terminations[agent]:
            self._was_dead_step(action)
            return
        
        player = None
        valid_pick = False

        # Ensure the action is a valid integer within the range of available players
        if 0 <= action < len(self.player_pool):
            player = self.player_pool[action]
            position = self.player_positions[player]
            
            # Check if the player is available and if the team has room for the position
            player_available = player in self.available_players
            position_room = self.team_positions[agent][position] < self.position_limits[position]
            flex_room = (self.team_positions[agent]['FLEX'] < self.position_limits['FLEX']) & (position != 'QB')
            bench_room = self.team_positions[agent]['BENCH'] < self.position_limits['BENCH']

            valid_pick = player_available and (position_room or flex_room or bench_room)

        if valid_pick:
            # Update the team roster and positions such that position players are chosen first, then FLEX, then BENCH
            self.team_rosters[agent].append(player)
            if position_room:
                i = self.team_positions[agent][position]
                self.team_positions_roster[agent][position][i] = player
                self.team_positions[agent][position] += 1
            elif flex_room:
                i = self.team_positions[agent]['FLEX']
                self.team_positions_roster[agent]['FLEX'][i] = player
                self.team_positions[agent]['FLEX'] += 1
            else:
                i = self.team_positions[agent]['BENCH']
                self.team_positions_roster[agent]['BENCH'][i] = player
                self.team_positions[agent]['BENCH'] += 1

            # Remove the player from available players and update draft history
            self.available_players.remove(player)
            self.draft_history.append((agent, player))
            self.rewards[agent] = 1

            # Update rewards
            self._cumulative_rewards[agent] += self.rewards[agent]

            # Advance the draft to next pick
            self.current_pick += 1

            if self.current_pick >= self.total_picks:
                for agent in self.agents:
                    self.terminations[agent] = True
            # Move to the next agent in the draft order
            self.agent_selection = self.current_agent()

        else:
            self.rewards[agent] = 0 # or -1 if we want to penalize invalid picks
            print(f"[Invalid Pick] {agent} attempted invalid selection (action={action}). Needs to retry.")

    def render(self):
        round_num = self.current_pick // self.num_teams + (1 if self.current_pick % self.num_teams != 0 else 0)
        print(f"\n--- Round {round_num} ---")       
        for agent in self.possible_agents:
            print(f"{agent} roster: {self.team_positions_roster[agent]}")
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

    def _generate_all_players(self):
        return list(self.player_df['player_name'])
    
    def _draft_player(self, agent, player):
        if player in self.available_players:
            self.available_players.remove(player)
            self.team_rosters[agent].append(player)
            self.draft_history.append((agent, player))
            return True
        return False
    
    def _get_available_players(self):
        return self.available_players
    
    def _get_draft_order(self):
        draft_order = []
        if self.snake_draft:
            for round in range(1, self.max_rounds+1):
                if self.snake_draft and round % 2 == 0:
                    round_order = list(reversed(range(self.num_teams)))
                else:
                    round_order = list(range(self.num_teams))
                draft_order.extend(round_order)
        return draft_order

    def _player_vector(self, players):
        vec = [1 if p in players else 0 for p in self.player_pool]
        return vec

players = pd.read_csv("data/player_projections/model_06_12_predictions.csv")
player_positions = pd.read_csv("data/processed/projection_models_test_06_02.csv")
position_columns = ["position_QB", "position_RB", "position_WR", "position_TE"]
player_positions['position'] = player_positions[position_columns].idxmax(axis=1)
player_positions['position']= player_positions['position'].str.replace('position_', '', regex=False)
players = players.merge(player_positions[['gsis_id', 'season', 'position']], on=['gsis_id', 'season'], how='left')
players_2023 = players[players['season'] == 2023]

env = CustomEnvironment(players_2023, num_teams=12, draft_type='snake', rounds=14)
env.reset()

while env.agent_selection is not None:
    print(f"\nCurrent Agent: {env.agent_selection}")
    agent = env.agent_selection

    for i, player in enumerate(env.player_pool): # Super basic policy, just select next player
        if player in env.available_players:
            print(f"\nAgent {agent} picking at pick #{env.current_pick + 1}")
            print(f"{agent} positions: {env.team_positions[agent]}")
            action = i
            break

    env.step(action)

    env.render()

    if env.current_pick >= env.total_picks:
        break

print("\n=== Final Team Rosters ===")
for agent in env.possible_agents:
    print(f"{agent}: {env.team_rosters[agent]}")
    print(f"{agent} positions: {env.team_positions[agent]}")

            


