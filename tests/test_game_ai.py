"""
Unit tests for GameAI class.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.ai.game_ai import GameAI


class TestGameAI:
    """Test cases for GameAI class."""
    
    @pytest.fixture
    def game_ai(self):
        """Create a GameAI instance for testing."""
        return GameAI()
    
    def test_initialization(self, game_ai):
        """Test GameAI initialization."""
        assert game_ai.user_history == []
        assert game_ai.move_frequency == {'rock': 0, 'paper': 0, 'scissors': 0}
        assert len(game_ai.strategies) == 4
        assert game_ai.next_move is None
    
    def test_update_with_valid_move(self, game_ai):
        """Test updating AI with valid moves."""
        game_ai.update('rock')
        
        assert len(game_ai.user_history) == 1
        assert game_ai.user_history[0] == 'rock'
        assert game_ai.move_frequency['rock'] == 1
        assert game_ai.transitions['start']['rock'] == 1
    
    def test_update_with_invalid_move(self, game_ai):
        """Test updating AI with invalid moves."""
        initial_history_length = len(game_ai.user_history)
        game_ai.update('invalid_move')
        
        # Should not update with invalid move
        assert len(game_ai.user_history) == initial_history_length
    
    def test_prepare_next_move_first_game(self, game_ai):
        """Test preparing next move for first game."""
        move = game_ai.prepare_next_move()
        
        assert move in ['rock', 'paper', 'scissors']
        assert game_ai.next_move == move
        assert game_ai.last_strategy_used == 'random'
    
    def test_prepare_next_move_with_history(self, game_ai):
        """Test preparing next move with game history."""
        # Add some history
        moves = ['rock', 'paper', 'scissors', 'rock', 'paper']
        for move in moves:
            game_ai.update(move)
        
        next_move = game_ai.prepare_next_move()
        
        assert next_move in ['rock', 'paper', 'scissors']
        assert game_ai.next_move == next_move
        assert game_ai.last_strategy_used is not None
    
    def test_get_next_move(self, game_ai):
        """Test getting the prepared next move."""
        # First call should prepare and return move
        move1 = game_ai.get_next_move()
        assert move1 in ['rock', 'paper', 'scissors']
        
        # Second call should return same move
        move2 = game_ai.get_next_move()
        assert move1 == move2
    
    def test_winning_moves_mapping(self, game_ai):
        """Test winning moves mapping."""
        assert game_ai.winning_moves['rock'] == 'paper'
        assert game_ai.winning_moves['paper'] == 'scissors'
        assert game_ai.winning_moves['scissors'] == 'rock'
    
    def test_pattern_detection(self, game_ai):
        """Test pattern detection functionality."""
        # Create a pattern
        pattern_moves = ['rock', 'paper', 'scissors', 'rock', 'paper']
        for move in pattern_moves:
            game_ai.update(move)
        
        # Check if pattern is recorded
        pattern_key = "paper,scissors"
        assert pattern_key in game_ai.patterns
        assert game_ai.patterns[pattern_key]['rock'] == 1
    
    def test_transition_tracking(self, game_ai):
        """Test move transition tracking."""
        moves = ['rock', 'paper', 'rock', 'scissors']
        for move in moves:
            game_ai.update(move)
        
        # Check transitions
        assert game_ai.transitions['rock']['paper'] == 1
        assert game_ai.transitions['paper']['rock'] == 1
        assert game_ai.transitions['rock']['scissors'] == 1
    
    def test_strategy_adaptation(self, game_ai):
        """Test strategy weight adaptation."""
        # Add some moves and simulate strategy performance
        moves = ['rock'] * 10
        for move in moves:
            game_ai.update(move)
        
        # Simulate some strategy performance
        game_ai.strategy_performance['counter_frequency'] = [1.0, 1.0, 0.0, 1.0, 1.0]
        game_ai.strategy_performance['random'] = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        initial_weights = game_ai.strategies.copy()
        game_ai._adapt_strategy_weights()
        
        # Successful strategy should have higher weight
        assert game_ai.strategies['counter_frequency'] >= initial_weights['counter_frequency']
    
    def test_get_statistics(self, game_ai):
        """Test getting AI statistics."""
        # Add some game data
        moves = ['rock', 'paper', 'scissors', 'rock']
        for move in moves:
            game_ai.update(move)
        
        stats = game_ai.get_statistics()
        
        assert 'total_games' in stats
        assert 'user_move_frequency' in stats
        assert 'strategy_weights' in stats
        assert 'recent_patterns' in stats
        assert 'transition_matrix' in stats
        assert 'strategy_performance' in stats
        
        assert stats['total_games'] == 4
        assert stats['user_move_frequency']['rock'] == 2
    
    def test_reset(self, game_ai):
        """Test resetting the AI."""
        # Add some data
        moves = ['rock', 'paper', 'scissors']
        for move in moves:
            game_ai.update(move)
        
        game_ai.next_move = 'rock'
        game_ai.last_strategy_used = 'counter_frequency'
        
        # Reset
        game_ai.reset()
        
        assert game_ai.user_history == []
        assert game_ai.move_frequency == {'rock': 0, 'paper': 0, 'scissors': 0}
        assert game_ai.next_move is None
        assert game_ai.last_strategy_used is None
        assert len(game_ai.recent_moves) == 0
    
    def test_strategy_performance_update(self, game_ai):
        """Test strategy performance updating."""
        # Setup initial state
        game_ai.next_move = 'paper'  # AI chose paper
        game_ai.last_strategy_used = 'counter_frequency'
        
        # User plays rock (AI should win)
        game_ai._update_strategy_performance('rock')
        
        # Check performance was recorded
        assert len(game_ai.strategy_performance['counter_frequency']) == 1
        assert game_ai.strategy_performance['counter_frequency'][0] == 1.0
        
        # Test losing scenario
        game_ai.next_move = 'rock'  # AI chose rock
        game_ai.last_strategy_used = 'counter_frequency'
        game_ai._update_strategy_performance('paper')  # User plays paper (AI loses)
        
        assert game_ai.strategy_performance['counter_frequency'][-1] == 0.0
