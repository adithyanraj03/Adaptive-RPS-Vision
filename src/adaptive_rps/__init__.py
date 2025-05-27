"""
AdaptiveRPS-Vision: Intelligent Rock-Paper-Scissors Detection with Adaptive AI

A comprehensive computer vision system for real-time hand gesture detection
and adaptive gameplay using YOLOv8 and reinforcement learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.detector import RPSDetector
from .core.trainer import RPSTrainer
from .ai.game_ai import GameAI
from .gui.main_window import MainWindow

__all__ = [
    'RPSDetector',
    'RPSTrainer', 
    'GameAI',
    'MainWindow'
]
