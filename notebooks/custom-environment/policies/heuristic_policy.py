# Basic Heuristic Policy for Reinforcement Learning

import pandas as pd
import numpy as np

class BasicHeuristicPolicy:
    def __init__(self, num_agents, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, obs):
        available_players = obs['available_players']
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_players)
        else:
            return self._heuristic_action(obs)
        
    def _heuristic_action(self, obs):
        projections = obs['player_projections']
        available_players = obs['available_players']
        best_player = max(available_players, key=lambda player: projections[player])
        return best_player
    
    def update_policy(self, action, reward):
        self.q_table[action] += self.alpha * (reward - self.q_table[action])
        