"""
Reinforcement Learning Agent for advanced game strategy.

This module implements a more sophisticated RL agent that can learn
complex patterns and strategies beyond the basic GameAI.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json


class RLAgent:
    """
    Advanced Reinforcement Learning agent for Rock-Paper-Scissors.
    
    This agent uses Q-learning with function approximation to learn
    optimal strategies against human players.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        memory_size: int = 1000
    ):
        """
        Initialize the RL agent.
        
        Args:
            learning_rate: Learning rate for Q-learning updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
            memory_size: Size of experience replay memory
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        
        # Q-table for state-action values
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}
        )
        
        # Experience replay memory
        self.memory: deque = deque(maxlen=memory_size)
        
        # Game history and statistics
        self.game_history: List[Tuple[str, str, str, float]] = []
        self.opponent_history: List[str] = []
        self.action_history: List[str] = []
        self.reward_history: List[float] = []
        
        # Pattern recognition
        self.pattern_memory: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'rock': 0, 'paper': 0, 'scissors': 0}
        )
        
        # State representation
        self.state_features = {
            'last_moves': deque(maxlen=5),
            'opponent_frequency': {'rock': 0, 'paper': 0, 'scissors': 0},
            'win_streak': 0,
            'total_games': 0
        }
        
        # Action mapping
        self.actions = ['rock', 'paper', 'scissors']
        self.winning_moves = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock'
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_rewards': 0.0,
            'win_rate': 0.0,
            'exploration_rate': self.epsilon,
            'q_table_size': 0
        }
    
    def get_state_representation(self) -> str:
        """
        Create a state representation from current game features.
        
        Returns:
            String representation of the current state
        """
        # Create state from recent moves and frequencies
        recent_moves = list(self.state_features['last_moves'])
        if len(recent_moves) < 3:
            recent_moves.extend(['none'] * (3 - len(recent_moves)))
        
        # Frequency ratios
        total_games = max(1, self.state_features['total_games'])
        freq_ratios = [
            self.state_features['opponent_frequency']['rock'] / total_games,
            self.state_features['opponent_frequency']['paper'] / total_games,
            self.state_features['opponent_frequency']['scissors'] / total_games
        ]
        
        # Discretize frequency ratios
        freq_bins = [int(ratio * 10) for ratio in freq_ratios]
        
        # Win streak (capped at 5)
        win_streak = min(5, max(-5, self.state_features['win_streak']))
        
        state = f"{'-'.join(recent_moves[-3:])}_freq_{'-'.join(map(str, freq_bins))}_streak_{win_streak}"
        return state
    
    def select_action(self, state: str) -> str:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            
        Returns:
            Selected action ('rock', 'paper', or 'scissors')
        """
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(self.actions)
        else:
            # Exploitation: best action according to Q-table
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        
        return action
    
    def update_q_table(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool = False
    ):
        """
        Update Q-table using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = max(self.q_table[next_state].values())
            target_q = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def calculate_reward(self, agent_action: str, opponent_action: str) -> float:
        """
        Calculate reward based on game outcome.
        
        Args:
            agent_action: Action taken by the agent
            opponent_action: Action taken by the opponent
            
        Returns:
            Reward value (1.0 for win, 0.0 for tie, -1.0 for loss)
        """
        if agent_action == opponent_action:
            return 0.0  # Tie
        elif self.winning_moves[opponent_action] == agent_action:
            return 1.0  # Win
        else:
            return -1.0  # Loss
    
    def update_state_features(self, opponent_action: str, reward: float):
        """
        Update internal state features based on game outcome.
        
        Args:
            opponent_action: Opponent's action
            reward: Reward received
        """
        # Update move history
        self.state_features['last_moves'].append(opponent_action)
        
        # Update frequency counts
        self.state_features['opponent_frequency'][opponent_action] += 1
        self.state_features['total_games'] += 1
        
        # Update win streak
        if reward > 0:
            self.state_features['win_streak'] = max(0, self.state_features['win_streak']) + 1
        elif reward < 0:
            self.state_features['win_streak'] = min(0, self.state_features['win_streak']) - 1
        else:
            self.state_features['win_streak'] = 0
        
        # Update pattern memory
        if len(self.opponent_history) >= 2:
            pattern = f"{self.opponent_history[-2]}_{self.opponent_history[-1]}"
            self.pattern_memory[pattern][opponent_action] += 1
    
    def play_round(self, opponent_action: str) -> str:
        """
        Play a single round against the opponent.
        
        Args:
            opponent_action: Opponent's action
            
        Returns:
            Agent's action for this round
        """
        # Get current state
        current_state = self.get_state_representation()
        
        # Select action
        agent_action = self.select_action(current_state)
        
        # Calculate reward
        reward = self.calculate_reward(agent_action, opponent_action)
        
        # Update state features
        self.update_state_features(opponent_action, reward)
        
        # Get next state
        next_state = self.get_state_representation()
        
        # Store experience in memory
        experience = (current_state, agent_action, reward, next_state, False)
        self.memory.append(experience)
        
        # Update Q-table
        self.update_q_table(current_state, agent_action, reward, next_state)
        
        # Store in history
        self.game_history.append((current_state, agent_action, opponent_action, reward))
        self.opponent_history.append(opponent_action)
        self.action_history.append(agent_action)
        self.reward_history.append(reward)
        
        # Update performance metrics
        self.performance_metrics['total_rewards'] += reward
        if len(self.reward_history) >= 10:
            recent_wins = sum(1 for r in self.reward_history[-10:] if r > 0)
            self.performance_metrics['win_rate'] = recent_wins / 10.0
        
        self.performance_metrics['exploration_rate'] = self.epsilon
        self.performance_metrics['q_table_size'] = len(self.q_table)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return agent_action
    
    def experience_replay(self, batch_size: int = 32):
        """
        Perform experience replay to improve learning.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the agent's performance.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'performance_metrics': self.performance_metrics.copy(),
            'state_features': {
                'total_games': self.state_features['total_games'],
                'opponent_frequency': self.state_features['opponent_frequency'].copy(),
                'win_streak': self.state_features['win_streak'],
                'recent_moves': list(self.state_features['last_moves'])
            },
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'current_epsilon': self.epsilon,
                'q_table_entries': len(self.q_table)
            },
            'pattern_analysis': dict(self.pattern_memory)
        }
        
        # Add recent performance
        if len(self.reward_history) > 0:
            stats['recent_performance'] = {
                'last_10_rewards': self.reward_history[-10:],
                'average_reward': np.mean(self.reward_history),
                'reward_std': np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0
            }
        
        return stats
    
    def save_model(self, filepath: str):
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'q_table': dict(self.q_table),
            'state_features': {
                'opponent_frequency': self.state_features['opponent_frequency'],
                'win_streak': self.state_features['win_streak'],
                'total_games': self.state_features['total_games'],
                'last_moves': list(self.state_features['last_moves'])
            },
            'pattern_memory': dict(self.pattern_memory),
            'performance_metrics': self.performance_metrics,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore Q-table
        self.q_table = defaultdict(
            lambda: {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}
        )
        for state, actions in model_data['q_table'].items():
            self.q_table[state] = actions
        
        # Restore state features
        state_data = model_data['state_features']
        self.state_features['opponent_frequency'] = state_data['opponent_frequency']
        self.state_features['win_streak'] = state_data['win_streak']
        self.state_features['total_games'] = state_data['total_games']
        self.state_features['last_moves'] = deque(state_data['last_moves'], maxlen=5)
        
        # Restore pattern memory
        self.pattern_memory = defaultdict(
            lambda: {'rock': 0, 'paper': 0, 'scissors': 0}
        )
        for pattern, counts in model_data['pattern_memory'].items():
            self.pattern_memory[pattern] = counts
        
        # Restore performance metrics
        self.performance_metrics = model_data['performance_metrics']
        
        # Restore hyperparameters
        hyperparams = model_data['hyperparameters']
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.epsilon = hyperparams['epsilon']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.min_epsilon = hyperparams['min_epsilon']
    
    def reset(self):
        """Reset the agent for a new game session."""
        self.game_history.clear()
        self.opponent_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.memory.clear()
        
        # Reset state features
        self.state_features = {
            'last_moves': deque(maxlen=5),
            'opponent_frequency': {'rock': 0, 'paper': 0, 'scissors': 0},
            'win_streak': 0,
            'total_games': 0
        }
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_rewards': 0.0,
            'win_rate': 0.0,
            'exploration_rate': self.epsilon,
            'q_table_size': len(self.q_table)
        }
        
        # Keep Q-table and pattern memory for transfer learning
