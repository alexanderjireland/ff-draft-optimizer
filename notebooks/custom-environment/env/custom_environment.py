from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import pandas as pd
from gymnasium import spaces

class CustomEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14):
        # Need to add rules about restrictions on positions, FLEX, etc.
        self.player_df = player_df
        self.player_pool = list(player_df['player_name'])

        # Initialize environment parameters
        self.num_teams = num_teams
        self.snake_draft = draft_type == 'snake'
        self.max_rounds = rounds
        self.total_picks = self.max_rounds * self.num_teams

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
        self.agents = self.possible_agents[:]
        self.current_pick = 0
        self.agent_selection = self.current_agent()

        self.rewards = {agent: 0 for agent in self.agents} # Figure out how to handle rewards
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

    def observe(self, agent):
        return {
            "available_players": self._player_vector(self.available_players),
            "team_roster": self._player_vector(self.team_rosters[agent])
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
        
        # Ensure the action is a valid integer within the range of available players
        if 0 <= action < len(self.player_pool):
            # Selected player to draft
            player = self.player_pool[action]
            if player in self.available_players:
                self.team_rosters[agent].append(player)
                self.available_players.remove(player)
                self.draft_history.append((agent, player))
                self.rewards[agent] = 1
            else:
                self.rewards[agent] = -1
        else:
            self.rewards[agent] = -1

        # Update rewards
        self._cumulative_rewards[agent] += self.rewards[agent]

        # Advance the draft to next pick
        self.current_pick += 1

        if self.current_pick >= self.total_picks:
            for agent in self.agents:
                self.terminations[agent] = True

        self.agent_selection = self.current_agent()

    def render(self):
        round_num = self.current_pick // self.num_teams + (1 if self.current_pick % self.num_teams != 0 else 0)
        print(f"\n--- Round {round_num} ---")       
        for agent in self.possible_agents:
            print(f"{agent} roster: {self.team_rosters[agent]}")
        print(f"Remaining players: {len(self.available_players)}")

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def current_agent(self):
        if self.current_pick >= self.total_picks:
            return None
        round_num = self.current_pick // self.num_teams + 1
        pick_index = self.current_pick % self.num_teams
        agent_index = self.draft_order[round_num][pick_index]
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
        draft_order = {}
        if self.snake_draft:
            for round in range(1, self.max_rounds+1):
                if self.snake_draft and round % 2 == 0:
                    draft_order[round] = list(reversed(range(self.num_teams)))
                else:
                    draft_order[round] = list(range(self.num_teams))
        else:
            for round in range(1, self.max_rounds+1):
                draft_order[round] = list(range(self.num_teams))
        return draft_order

    def _player_vector(self, players):
        vec = [1 if p in players else 0 for p in self.player_pool]
        return vec


players = pd.DataFrame({"player_name": [f"Player {i}" for i in range(20)]})

env = CustomEnvironment(players, num_teams=2, draft_type='not snake', rounds=5)
env.reset()

while env.agents:
    agent = env.agent_selection
    if agent is None:
        break

    for i, player in enumerate(env.player_pool): # Super basic policy, just select next player
        if player in env.available_players:
            action = i
            break

    env.step(action)

    env.render()

print("\n=== Final Team Rosters ===")
for agent in env.possible_agents:
    print(f"{agent}: {env.team_rosters[agent]}")

            


