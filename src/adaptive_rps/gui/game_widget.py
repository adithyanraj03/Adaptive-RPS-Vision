"""
Game widget for interactive Rock-Paper-Scissors gameplay.

This widget handles the game mode interface, managing rounds, scoring,
AI opponent interaction, and game flow.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QGroupBox, QLCDNumber, QProgressBar,
                            QTextEdit, QGridLayout, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from ..core.detector import RPSDetector
from ..utils.camera import CameraManager
from ..utils.image_utils import ImageProcessor
from ..ai.game_ai import GameAI


class GameWidget(QWidget):
    """
    Widget for interactive Rock-Paper-Scissors game interface.
    
    Manages game rounds, scoring, AI opponent, countdown timers,
    and provides comprehensive game statistics and history.
    """
    
    # Signals
    game_finished = pyqtSignal(dict)  # Emits final game results
    round_completed = pyqtSignal(dict)  # Emits round results
    
    def __init__(self, detector: RPSDetector, camera: CameraManager):
        super().__init__()
        
        self.detector = detector
        self.camera = camera
        self.game_ai = GameAI()
        
        # Game state
        self.game_active = False
        self.round_in_progress = False
        self.current_round = 0
        self.max_rounds = 10
        self.countdown_value = 5
        
        # Scores
        self.user_score = 0
        self.ai_score = 0
        self.tie_count = 0
        
        # Round data
        self.user_move = None
        self.ai_move = None
        self.round_result = None
        self.round_history: List[Dict] = []
        
        # Timing
        self.round_start_time = None
        self.capture_frame = None
        
        # UI setup
        self.init_ui()
        self.setup_timers()
        self.reset_game()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Header with scores
        header_panel = self.create_header_panel()
        layout.addWidget(header_panel)
        
        # Main game area
        game_area = self.create_game_area()
        layout.addWidget(game_area, 1)
        
        # Control buttons
        controls_panel = self.create_controls_panel()
        layout.addWidget(controls_panel)
    
    def create_header_panel(self) -> QWidget:
        """Create the score header panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # User score
        user_group = QGroupBox("Your Score")
        user_layout = QVBoxLayout(user_group)
        self.user_score_lcd = QLCDNumber(2)
        self.user_score_lcd.setMinimumHeight(60)
        self.user_score_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #2D3142;
                color: #55FF55;
                border: 2px solid #55FF55;
                border-radius: 5px;
            }
        """)
        user_layout.addWidget(self.user_score_lcd)
        layout.addWidget(user_group)
        
        # Round counter
        round_group = QGroupBox("Round")
        round_layout = QVBoxLayout(round_group)
        self.round_lcd = QLCDNumber(2)
        self.round_lcd.setMinimumHeight(60)
        self.round_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #2D3142;
                color: #FFFF55;
                border: 2px solid #FFFF55;
                border-radius: 5px;
            }
        """)
        round_layout.addWidget(self.round_lcd)
        layout.addWidget(round_group)
        
        # AI score
        ai_group = QGroupBox("AI Score")
        ai_layout = QVBoxLayout(ai_group)
        self.ai_score_lcd = QLCDNumber(2)
        self.ai_score_lcd.setMinimumHeight(60)
        self.ai_score_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #2D3142;
                color: #FF5555;
                border: 2px solid #FF5555;
                border-radius: 5px;
            }
        """)
        ai_layout.addWidget(self.ai_score_lcd)
        layout.addWidget(ai_group)
        
        return widget
    
    def create_game_area(self) -> QWidget:
        """Create the main game area."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # User panel
        user_panel = self.create_user_panel()
        layout.addWidget(user_panel, 2)
        
        # Center panel with countdown and status
        center_panel = self.create_center_panel()
        layout.addWidget(center_panel, 1)
        
        # AI panel
        ai_panel = self.create_ai_panel()
        layout.addWidget(ai_panel, 2)
        
        return widget
    
    def create_user_panel(self) -> QGroupBox:
        """Create user move display panel."""
        group = QGroupBox("Your Move")
        layout = QVBoxLayout(group)
        
        # Camera feed
        self.user_camera_label = QLabel()
        self.user_camera_label.setAlignment(Qt.AlignCenter)
        self.user_camera_label.setMinimumSize(320, 240)
        self.user_camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #BFC0C0;
                border-radius: 5px;
                background-color: #1a1a1a;
            }
        """)
        layout.addWidget(self.user_camera_label)
        
        # User move label
        self.user_move_label = QLabel("Get ready...")
        self.user_move_label.setAlignment(Qt.AlignCenter)
        self.user_move_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.user_move_label.setMinimumHeight(40)
        layout.addWidget(self.user_move_label)
        
        # User move visualization
        self.user_move_display = QLabel()
        self.user_move_display.setAlignment(Qt.AlignCenter)
        self.user_move_display.setMinimumSize(120, 120)
        self.user_move_display.setStyleSheet("""
            QLabel {
                border: 2px solid #4F5D75;
                border-radius: 10px;
                background-color: #2D3142;
            }
        """)
        layout.addWidget(self.user_move_display)
        
        return group
    
    def create_center_panel(self) -> QGroupBox:
        """Create center status and countdown panel."""
        group = QGroupBox("Game Status")
        layout = QVBoxLayout(group)
        
        # Game status
        self.game_status_label = QLabel("Ready to Play!")
        self.game_status_label.setAlignment(Qt.AlignCenter)
        self.game_status_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.game_status_label.setMinimumHeight(50)
        layout.addWidget(self.game_status_label)
        
        # Countdown display
        self.countdown_label = QLabel("5")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setFont(QFont("Arial", 72, QFont.Bold))
        self.countdown_label.setMinimumHeight(120)
        self.countdown_label.setStyleSheet("""
            QLabel {
                color: #FF5555;
                border: 3px solid #FF5555;
                border-radius: 15px;
                background-color: #2D3142;
            }
        """)
        layout.addWidget(self.countdown_label)
        
        # Round result
        self.round_result_label = QLabel("")
        self.round_result_label.setAlignment(Qt.AlignCenter)
        self.round_result_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.round_result_label.setMinimumHeight(60)
        layout.addWidget(self.round_result_label)
        
        # Progress bar for round
        self.round_progress = QProgressBar()
        self.round_progress.setMaximum(self.max_rounds)
        self.round_progress.setValue(0)
        self.round_progress.setTextVisible(True)
        self.round_progress.setFormat("Round %v / %m")
        layout.addWidget(self.round_progress)
        
        return group
    
    def create_ai_panel(self) -> QGroupBox:
        """Create AI move display panel."""
        group = QGroupBox("AI's Move")
        layout = QVBoxLayout(group)
        
        # AI move visualization
        self.ai_move_display = QLabel()
        self.ai_move_display.setAlignment(Qt.AlignCenter)
        self.ai_move_display.setMinimumSize(320, 240)
        self.ai_move_display.setStyleSheet("""
            QLabel {
                border: 2px solid #BFC0C0;
                border-radius: 5px;
                background-color: #1a1a1a;
            }
        """)
        layout.addWidget(self.ai_move_display)
        
        # AI move label
        self.ai_move_label = QLabel("AI is thinking...")
        self.ai_move_label.setAlignment(Qt.AlignCenter)
        self.ai_move_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.ai_move_label.setMinimumHeight(40)
        layout.addWidget(self.ai_move_label)
        
        # AI strategy info
        self.ai_strategy_label = QLabel("Learning your patterns...")
        self.ai_strategy_label.setAlignment(Qt.AlignCenter)
        self.ai_strategy_label.setFont(QFont("Arial", 12))
        self.ai_strategy_label.setWordWrap(True)
        self.ai_strategy_label.setMinimumHeight(60)
        layout.addWidget(self.ai_strategy_label)
        
        return group
    
    def create_controls_panel(self) -> QWidget:
        """Create game control buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Start/Next round button
        self.next_round_button = QPushButton("Start Game")
        self.next_round_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.next_round_button.setMinimumHeight(50)
        self.next_round_button.clicked.connect(self.start_next_round)
        layout.addWidget(self.next_round_button)
        
        # Restart game button
        self.restart_button = QPushButton("Restart Game")
        self.restart_button.setFont(QFont("Arial", 14))
        self.restart_button.setMinimumHeight(50)
        self.restart_button.clicked.connect(self.restart_game)
        self.restart_button.setEnabled(False)
        layout.addWidget(self.restart_button)
        
        # Game statistics button
        self.stats_button = QPushButton("View Statistics")
        self.stats_button.setFont(QFont("Arial", 14))
        self.stats_button.setMinimumHeight(50)
        self.stats_button.clicked.connect(self.show_statistics)
        layout.addWidget(self.stats_button)
        
        return widget
    
    def setup_timers(self):
        """Setup game timers."""
        # Countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        # Camera update timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(50)  # 20 FPS
        
        # AI thinking timer
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_thinking_complete)
    
    def reset_game(self):
        """Reset game to initial state."""
        self.game_active = False
        self.round_in_progress = False
        self.current_round = 0
        self.user_score = 0
        self.ai_score = 0
        self.tie_count = 0
        self.round_history.clear()
        
        # Reset UI
        self.user_score_lcd.display(0)
        self.ai_score_lcd.display(0)
        self.round_lcd.display(0)
        self.round_progress.setValue(0)
        
        self.game_status_label.setText("Ready to Play!")
        self.countdown_label.setText("5")
        self.round_result_label.setText("")
        self.user_move_label.setText("Get ready...")
        self.ai_move_label.setText("AI is ready...")
        
        self.next_round_button.setText("Start Game")
        self.next_round_button.setEnabled(True)
        self.restart_button.setEnabled(False)
        
        # Reset AI
        self.game_ai.reset()
    
    def start_next_round(self):
        """Start the next round."""
        if not self.game_active:
            self.game_active = True
            self.next_round_button.setText("Next Round")
        
        if self.current_round >= self.max_rounds:
            self.end_game()
            return
        
        self.current_round += 1
        self.round_in_progress = True
        self.user_move = None
        self.ai_move = None
        self.round_result = None
        self.capture_frame = None
        
        # Update UI
        self.round_lcd.display(self.current_round)
        self.round_progress.setValue(self.current_round)
        self.game_status_label.setText(f"Round {self.current_round}")
        self.round_result_label.setText("")
        self.user_move_label.setText("Get ready to show your move!")
        self.ai_move_label.setText("AI is preparing...")
        
        # Clear move displays
        self.user_move_display.clear()
        self.ai_move_display.clear()
        
        # Disable next round button
        self.next_round_button.setEnabled(False)
        
        # Start countdown
        self.countdown_value = 5
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_timer.start(1000)  # 1 second intervals
        
        # Record round start time
        self.round_start_time = time.time()
    
    def update_countdown(self):
        """Update countdown timer."""
        self.countdown_value -= 1
        
        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
            # Change color as countdown progresses
            if self.countdown_value <= 2:
                self.countdown_label.setStyleSheet("""
                    QLabel {
                        color: #FF0000;
                        border: 3px solid #FF0000;
                        border-radius: 15px;
                        background-color: #2D3142;
                    }
                """)
        else:
            self.countdown_timer.stop()
            self.countdown_label.setText("SHOW!")
            self.countdown_label.setStyleSheet("""
                QLabel {
                    color: #00FF00;
                    border: 3px solid #00FF00;
                    border-radius: 15px;
                    background-color: #2D3142;
                }
            """)
            
            # Capture user move
            self.capture_user_move()
    
    def capture_user_move(self):
        """Capture and analyze user's move."""
        # Get current frame
        ret, frame = self.camera.read_frame()
        if ret:
            self.capture_frame = frame.copy()
            
            # Detect gesture
            detections = self.detector.detect(frame)
            
            if detections:
                # Get best detection
                best_detection = max(detections, key=lambda x: x[1])
                self.user_move, confidence, bbox = best_detection
                self.user_move_label.setText(f"You: {self.user_move.upper()}")
                
                # Show captured frame
                self.display_user_move(frame, bbox)
            else:
                self.user_move = None
                self.user_move_label.setText("No move detected!")
        
        # Get AI move
        self.ai_move = self.game_ai.prepare_next_move()
        self.ai_move_label.setText("AI is analyzing...")
        
        # Show AI thinking animation
        self.ai_timer.start(2000)  # 2 seconds thinking time
    
    def ai_thinking_complete(self):
        """Complete AI thinking phase."""
        self.ai_timer.stop()
        
        # Show AI move
        self.ai_move_label.setText(f"AI: {self.ai_move.upper()}")
        self.display_ai_move(self.ai_move)
        
        # Determine round winner
        self.determine_round_winner()
        
        # Update AI with user's move
        if self.user_move:
            self.game_ai.update(self.user_move)
        
        # Update strategy display
        self.update_ai_strategy_display()
        
        # Check if game is complete
        if self.current_round >= self.max_rounds:
            self.restart_button.setEnabled(True)
            self.next_round_button.setText("Game Complete")
        else:
            self.next_round_button.setEnabled(True)
        
        self.round_in_progress = False
    
    def display_user_move(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Display user's captured move."""
        # Draw detection on frame
        result_frame = self.detector.visualize_detections(frame, [(self.user_move, 1.0, bbox)])
        
        # Convert to QPixmap and display
        h, w, c = result_frame.shape
        bytes_per_line = 3 * w
        q_image = QImage(result_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(
            self.user_camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.user_camera_label.setPixmap(scaled_pixmap)
        
        # Create move visualization
        move_image = ImageProcessor.create_move_visualization(self.user_move, (120, 120))
        h, w, c = move_image.shape
        bytes_per_line = 3 * w
        q_image = QImage(move_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        move_pixmap = QPixmap.fromImage(q_image)
        self.user_move_display.setPixmap(move_pixmap)
    
    def display_ai_move(self, ai_move: str):
        """Display AI's move."""
        # Create AI move visualization
        move_image = ImageProcessor.create_move_visualization(ai_move, (320, 240))
        h, w, c = move_image.shape
        bytes_per_line = 3 * w
        q_image = QImage(move_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.ai_move_display.setPixmap(pixmap)
    
    def determine_round_winner(self):
        """Determine the winner of the current round."""
        if not self.user_move:
            self.round_result = "No Move Detected"
            self.round_result_label.setText(self.round_result)
            self.round_result_label.setStyleSheet("color: #FFAA00;")
        elif self.user_move == self.ai_move:
            self.round_result = "Tie!"
            self.tie_count += 1
            self.round_result_label.setText(self.round_result)
            self.round_result_label.setStyleSheet("color: #FFFF55;")
        elif self.is_user_winner(self.user_move, self.ai_move):
            self.round_result = "You Win!"
            self.user_score += 1
            self.user_score_lcd.display(self.user_score)
            self.round_result_label.setText(self.round_result)
            self.round_result_label.setStyleSheet("color: #55FF55;")
        else:
            self.round_result = "AI Wins!"
            self.ai_score += 1
            self.ai_score_lcd.display(self.ai_score)
            self.round_result_label.setText(self.round_result)
            self.round_result_label.setStyleSheet("color: #FF5555;")
        
        # Record round data
        round_data = {
            'round': self.current_round,
            'user_move': self.user_move,
            'ai_move': self.ai_move,
            'result': self.round_result,
            'timestamp': datetime.now(),
            'round_time': time.time() - self.round_start_time if self.round_start_time else 0
        }
        self.round_history.append(round_data)
        self.round_completed.emit(round_data)
    
    def is_user_winner(self, user_move: str, ai_move: str) -> bool:
        """Check if user wins the round."""
        winning_combinations = {
            ('rock', 'scissors'),
            ('paper', 'rock'),
            ('scissors', 'paper')
        }
        return (user_move, ai_move) in winning_combinations
    
    def update_ai_strategy_display(self):
        """Update AI strategy information display."""
        stats = self.game_ai.get_statistics()
        
        if stats['total_games'] > 0:
            # Show most frequent user move
            freq = stats['user_move_frequency']
            most_frequent = max(freq, key=freq.get) if any(freq.values()) else "none"
            
            # Show current strategy weights
            weights = stats['strategy_weights']
            dominant_strategy = max(weights, key=weights.get)
            
            strategy_text = f"Tracking: {most_frequent.upper()} most frequent\n"
            strategy_text += f"Strategy: {dominant_strategy.replace('_', ' ').title()}"
            
            self.ai_strategy_label.setText(strategy_text)
        else:
            self.ai_strategy_label.setText("Learning your patterns...")
    
    def update_camera_feed(self):
        """Update camera feed display."""
        if not self.round_in_progress or self.countdown_value > 0:
            ret, frame = self.camera.read_frame()
            if ret:
                # Convert to QPixmap and display
                h, w, c = frame.shape
                bytes_per_line = 3 * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                
                scaled_pixmap = pixmap.scaled(
                    self.user_camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.user_camera_label.setPixmap(scaled_pixmap)
    
    def restart_game(self):
        """Restart the game."""
        self.reset_game()
    
    def end_game(self):
        """End the current game."""
        self.game_active = False
        
        # Determine overall winner
        if self.user_score > self.ai_score:
            overall_result = "You Win the Game!"
            result_color = "#55FF55"
        elif self.ai_score > self.user_score:
            overall_result = "AI Wins the Game!"
            result_color = "#FF5555"
        else:
            overall_result = "Game Tied!"
            result_color = "#FFFF55"
        
        self.game_status_label.setText(overall_result)
        self.game_status_label.setStyleSheet(f"color: {result_color};")
        
        # Prepare final game data
        game_data = {
            'user_score': self.user_score,
            'ai_score': self.ai_score,
            'tie_count': self.tie_count,
            'total_rounds': self.current_round,
            'round_history': self.round_history,
            'ai_statistics': self.game_ai.get_statistics(),
            'overall_result': overall_result
        }
        
        self.game_finished.emit(game_data)
        
        # Enable restart
        self.restart_button.setEnabled(True)
        self.next_round_button.setEnabled(False)
    
    def show_statistics(self):
        """Show detailed game statistics."""
        # This could open a detailed statistics dialog
        # For now, we'll just update the AI strategy display
        self.update_ai_strategy_display()
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return {
            'game_active': self.game_active,
            'current_round': self.current_round,
            'user_score': self.user_score,
            'ai_score': self.ai_score,
            'tie_count': self.tie_count,
            'round_in_progress': self.round_in_progress,
            'round_history': self.round_history
        }
