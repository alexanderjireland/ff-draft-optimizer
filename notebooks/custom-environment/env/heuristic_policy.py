# Basic Heuristic Policy for Reinforcement Learning

import pandas as pd
import numpy as np

class BasicHeuristicPolicy:
    def __init__(self, num_agents, player_pool_size, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros(player_pool_size)

    def select_action(self, obs, team_positions_available):
        available_players = obs['available_players']
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.where(available_players)[0])
        else:
            return self._heuristic_action(obs, team_positions_available)
        
    def _heuristic_action(self, obs, team_positions_available):
        projections = obs['player_projections']
        position_available = [team_positions_available[pos] for pos in obs['player_positions']]
        available_players = np.multiply(obs['available_players'], position_available)
        if np.sum(available_players) > 0:
            best_player = np.argmax(np.multiply(projections, available_players))
        else:
            print("No valid players found...")
        return best_player
    
    def update_policy(self, action, reward):
        self.q_table[action] += self.alpha * (reward - self.q_table[action])
        #print(self.q_table)
        