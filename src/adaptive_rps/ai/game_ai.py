"""Game AI module with reinforcement learning capabilities."""

import random
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


class GameAI:
    """AI that learns from user's previous moves to make predictions."""
    
    def __init__(self):
        """Initialize the Game AI."""
        self.user_history: List[str] = []
        self.move_frequency: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
        
        # Track transitions (what move follows each move)
        self.transitions: Dict[str, Dict[str, int]] = {
            'rock': {'rock': 0, 'paper': 0, 'scissors': 0},
            'paper': {'rock': 0, 'paper': 0, 'scissors': 0},
            'scissors': {'rock': 0, 'paper': 0, 'scissors': 0},
            'start': {'rock': 0, 'paper': 0, 'scissors': 0},
        }
        
        # Track patterns of length 2-3
        self.patterns: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'rock': 0, 'paper': 0, 'scissors': 0}
        )
        
        # Track user's last few moves for pattern detection
        self.recent_moves: deque = deque(maxlen=5)
        
        # Strategy weights - these can be adapted based on performance
        self.strategies: Dict[str, float] = {
            'counter_frequency': 0.3,    # Counter what the user plays most often
            'counter_last': 0.3,         # Counter the user's last move
            'counter_pattern': 0.3,      # Counter predicted move based on pattern
            'random': 0.1                # Play randomly
        }
        
        # Winning moves mapping
        self.winning_moves: Dict[str, str] = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock'
        }
        
        # Performance tracking for strategy adaptation
        self.strategy_performance: Dict[str, List[float]] = {
            strategy: [] for strategy in self.strategies.keys()
        }
        
        self.next_move: Optional[str] = None
        self.last_strategy_used: Optional[str] = None
    
    def update(self, user_move: str) -> None:
        """
        Update AI's knowledge based on user's move.
        
        Args:
            user_move: The move the user made ('rock', 'paper', 'scissors')
        """
        if user_move not in ['rock', 'paper', 'scissors']:
            return
        
        self.user_history.append(user_move)
        self.move_frequency[user_move] += 1
        
        # Update transitions
        if len(self.user_history) == 1:
            self.transitions['start'][user_move] += 1
        else:
            prev_move = self.user_history[-2]
            self.transitions[prev_move][user_move] += 1
        
        # Update recent moves for pattern detection
        self.recent_moves.append(user_move)
        
        # Update patterns of length 2 and 3
        if len(self.recent_moves) >= 3:
            pattern2 = f"{self.recent_moves[-3]},{self.recent_moves[-2]}"
            self.patterns[pattern2][user_move] += 1
        
        # Update strategy performance if we have a previous prediction
        if self.next_move and self.last_strategy_used:
            self._update_strategy_performance(user_move)
    
    def _update_strategy_performance(self, actual_user_move: str) -> None:
        """Update performance tracking for the last strategy used."""
        if self.next_move and self.last_strategy_used:
            # Check if our prediction was correct (if we would have won)
            user_would_beat = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
            correct_prediction = user_would_beat.get(actual_user_move) == self.next_move
            
            performance_score = 1.0 if correct_prediction else 0.0
            self.strategy_performance[self.last_strategy_used].append(performance_score)
            
            # Keep only recent performance data
            if len(self.strategy_performance[self.last_strategy_used]) > 20:
                self.strategy_performance[self.last_strategy_used].pop(0)
    
    def _adapt_strategy_weights(self) -> None:
        """Adapt strategy weights based on recent performance."""
        total_weight = 0
        new_weights = {}
        
        for strategy, performances in self.strategy_performance.items():
            if len(performances) >= 5:  # Need enough data
                avg_performance = np.mean(performances[-10:])  # Recent performance
                # Boost successful strategies
                weight = max(0.05, self.strategies[strategy] * (1 + avg_performance))
            else:
                weight = self.strategies[strategy]
            
            new_weights[strategy] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for strategy in new_weights:
                self.strategies[strategy] = new_weights[strategy] / total_weight
    
    def prepare_next_move(self) -> str:
        """
        Prepare AI's next move based on history.
        
        Returns:
            str: The move the AI will make ('rock', 'paper', 'scissors')
        """
        if not self.user_history:
            # First move - no data to predict from
            self.next_move = random.choice(['rock', 'paper', 'scissors'])
            self.last_strategy_used = 'random'
            return self.next_move
        
        # Adapt strategy weights based on recent performance
        self._adapt_strategy_weights()
        
        # Strategy 1: Predict based on overall frequency
        most_frequent = max(self.move_frequency, key=self.move_frequency.get)
        
        # Strategy 2: Predict based on transition from last move
        transition_prediction = None
        if self.user_history:
            last_move = self.user_history[-1]
            transitions = self.transitions[last_move]
            if sum(transitions.values()) > 0:
                transition_prediction = max(transitions, key=transitions.get)
        
        # Strategy 3: Predict based on pattern
        pattern_prediction = None
        if len(self.recent_moves) >= 2:
            pattern = f"{self.recent_moves[-2]},{self.recent_moves[-1]}"
            pattern_data = self.patterns[pattern]
            if sum(pattern_data.values()) > 0:
                pattern_prediction = max(pattern_data, key=pattern_data.get)
        
        # Collect available strategies with their predictions
        strategies = []
        
        if sum(self.move_frequency.values()) > 0:
            strategies.append(('counter_frequency', most_frequent))
        
        if transition_prediction:
            strategies.append(('counter_last', transition_prediction))
        
        if pattern_prediction:
            strategies.append(('counter_pattern', pattern_prediction))
        
        # Always include random as fallback
        strategies.append(('random', random.choice(['rock', 'paper', 'scissors'])))
        
        # Choose strategy based on weights
        strategy_weights = [self.strategies[s[0]] for s in strategies]
        
        # Normalize weights
        total_weight = sum(strategy_weights)
        if total_weight > 0:
            strategy_weights = [w/total_weight for w in strategy_weights]
        else:
            strategy_weights = [1/len(strategies)] * len(strategies)
        
        # Select strategy
        chosen_strategy_index = np.random.choice(len(strategies), p=strategy_weights)
        strategy_name, predicted_move = strategies[chosen_strategy_index]
        
        # Store which strategy we used
        self.last_strategy_used = strategy_name
        
        # Return the move that beats the predicted move
        self.next_move = self.winning_moves[predicted_move]
        return self.next_move
    
    def get_next_move(self) -> str:
        """
        Get the already-prepared next move.
        
        Returns:
            str: The AI's next move
        """
        if self.next_move is None:
            return self.prepare_next_move()
        return self.next_move
    
    def get_statistics(self) -> Dict:
        """
        Get AI performance and learning statistics.
        
        Returns:
            Dict: Statistics about the AI's learning and performance
        """
        total_games = len(self.user_history)
        
        stats = {
            'total_games': total_games,
            'user_move_frequency': self.move_frequency.copy(),
            'strategy_weights': self.strategies.copy(),
            'recent_patterns': dict(self.patterns),
            'transition_matrix': self.transitions,
        }
        
        # Add strategy performance
        strategy_performance = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_performance[strategy] = {
                    'avg_performance': np.mean(performances),
                    'recent_performance': np.mean(performances[-5:]) if len(performances) >= 5 else 0,
                    'games_played': len(performances)
                }
            else:
                strategy_performance[strategy] = {
                    'avg_performance': 0,
                    'recent_performance': 0,
                    'games_played': 0
                }
        
        stats['strategy_performance'] = strategy_performance
        
        return stats
    
    def reset(self) -> None:
        """Reset the AI for a new game session."""
        self.user_history = []
        self.move_frequency = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.transitions = {
            'rock': {'rock': 0, 'paper': 0, 'scissors': 0},
            'paper': {'rock': 0, 'paper': 0, 'scissors': 0},
            'scissors': {'rock': 0, 'paper': 0, 'scissors': 0},
            'start': {'rock': 0, 'paper': 0, 'scissors': 0},
        }
        self.patterns = defaultdict(lambda: {'rock': 0, 'paper': 0, 'scissors': 0})
        self.recent_moves.clear()
        self.next_move = None
        self.last_strategy_used = None
        
        # Reset strategy weights to default
        self.strategies = {
            'counter_frequency': 0.3,
            'counter_last': 0.3,
            'counter_pattern': 0.3,
            'random': 0.1
        }
        
        # Keep some performance history for learning across sessions
        for strategy in self.strategy_performance:
            if len(self.strategy_performance[strategy]) > 10:
                # Keep only the last 10 performances
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-10:]
