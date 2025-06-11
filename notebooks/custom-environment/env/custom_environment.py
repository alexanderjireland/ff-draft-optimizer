from pettingzoo import ParallelEnv
import pandas as pd
from gymnasium import spaces

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, player_df:pd.DataFrame, num_teams=2, draft_type=None, rounds=14):
        # Need to add rules about restrictions on positions, FLEX, etc.
        self.player_df = player_df
        self.num_teams = num_teams
        self.snake_draft = draft_type == 'snake'
        self.max_rounds = rounds
        self.total_picks = self.max_rounds * self.num_teams

        self.possible_agents = [f"team_{i}" for i in range(num_teams)]
        self.agents = self.possible_agents[:]

        self.player_pool = self._generate_all_players()
        self.available_players = self.player_pool.copy()

        self._action_spaces = {
            agent: spaces.Discrete(len(self.player_pool)) for agent in self.agents
        }
        self._observation_spaces = {
            agent: spaces.Dict({
                "available_players": spaces.MultiBinary(len(self.player_pool)),
                "team_roster": spaces.MultiBinary(len(self.player_pool)),
            }) for agent in self.agents
        }

        # Draft tracking
        self.draft_order = self._get_draft_order()
        self.draft_history = []
        self.team_rosters = {agent: [] for agent in self.agents}
        self.current_pick = 0
        
    def reset(self, seed=None, options=None):
        self.current_pick = 0
        self.available_players = self.player_pool.copy()
        self.draft_history = []
        self.team_rosters = {agent: [] for agent in self.agents}
        self.agents = self.possible_agents[:]
        return {agent: self._get_observation(agent) for agent in self.agents}

    def _get_observation(self, agent):
        obs = {
            "available_players": self._player_vector(self.available_players),
            "team_roster": self._player_vector(self.team_rosters[agent])
        }
        return obs

    def _player_vector(self, players):
        vec = [0] * len(self.player_pool)
        for i, player in enumerate(self.player_pool):
            if player in players:
                vec[i] = 1
        return vec
    
    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.current_pick >= self.total_picks:
            for agent in self.agents:
                terminated[agent] = True
            self.agents = []
            return rewards, terminated, truncated, infos

        round_num = self.current_pick // self.num_teams + 1
        pick_index = self.current_pick % self.num_teams
        agent_index = self.draft_order[round_num][pick_index]
        pick_agent = self.possible_agents[agent_index]

        print(f"Next pick #: {self.current_pick}, Round: {round_num}, Pick index: {pick_index}, Agent: {pick_agent}")

        if pick_agent in actions:
            action = actions[pick_agent]
            if 0 <= action < len(self.player_pool):
                player = self.player_pool[action]
                if self._draft_player(pick_agent, player):
                    rewards[pick_agent] = 1
                    self.current_pick += 1
                    
        if pick_agent not in actions:
            print(f"[DEBUG] pick_agent {pick_agent} not in actions: {actions}")

        if self.current_pick >= self.total_picks:
            for agent in self.agents:
                terminated[agent] = True
            self.agents = []
            observations = {agent: self._get_observation(agent) for agent in self.possible_agents}
            return observations, rewards, terminated, truncated, infos

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, rewards, terminated, truncated, infos

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
    





players = pd.DataFrame({"player_name": [f"Player {i}" for i in range(20)]})
env = CustomEnvironment(players, num_teams=2, draft_type='regular', rounds=5)

obs = env.reset()

while env.agents:
    agent = env.current_agent()
    if agent is None:
        break

    # Dummy policy: always pick the first available player
    for i, player in enumerate(env.player_pool):
        if player in env.available_players:
            action = i
            break

    actions = {agent: action}
    obs, rew, term, trunc, info = env.step(actions)
    env.render()

    if env.current_pick >= env.total_picks:
        break
            


