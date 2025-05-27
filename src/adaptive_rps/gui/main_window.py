"""
Main application window for AdaptiveRPS-Vision.

This module provides the main GUI window that manages different modes
(detection and game) and coordinates between various components.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QStackedWidget, QPushButton, QMenuBar, QMenu, 
                            QAction, QStatusBar, QMessageBox, QFileDialog,
                            QLabel, QFrame, QSplitter)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from datetime import datetime
from typing import Optional, Dict, Any

from ..core.detector import RPSDetector
from ..utils.camera import CameraManager
from ..utils.colors import Colors
from .detector_widget import DetectorWidget
from .game_widget import GameWidget


class MainWindow(QMainWindow):
    """
    Main application window for AdaptiveRPS-Vision.
    
    Manages the overall application state, coordinates between detection
    and game modes, and provides the main user interface.
    """
    
    def __init__(
        self, 
        model_path: str = "./models/best.pt",
        camera_id: int = 0,
        confidence_threshold: float = 0.25,
        device: str = "auto"
    ):
        super().__init__()
        
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Initialize components
        self.detector: Optional[RPSDetector] = None
        self.camera: Optional[CameraManager] = None
        
        # UI components
        self.stacked_widget: Optional[QStackedWidget] = None
        self.detector_widget: Optional[DetectorWidget] = None
        self.game_widget: Optional[GameWidget] = None
        
        # Application state
        self.current_mode = "detection"  # "detection" or "game"
        self.session_data = {
            'start_time': datetime.now(),
            'total_detections': 0,
            'games_played': 0
        }
        
        # Setup UI and initialize components
        self.init_ui()
        self.init_components()
        self.setup_connections()
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AdaptiveRPS-Vision - Intelligent Gesture Detection")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Create central widget and stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create header
        header = self.create_header()
        layout.addWidget(header)
        
        # Create main content area
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget, 1)
        
        # Create footer
        footer = self.create_footer()
        layout.addWidget(footer)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
    
    def create_header(self) -> QWidget:
        """Create the application header."""
        header = QFrame()
        header.setFrameStyle(QFrame.StyledPanel)
        header.setMaximumHeight(80)
        
        layout = QHBoxLayout(header)
        
        # Application title
        title_label = QLabel("AdaptiveRPS-Vision")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #EAE8FF; padding: 10px;")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Mode toggle buttons
        self.detection_mode_btn = QPushButton("Detection Mode")
        self.detection_mode_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.detection_mode_btn.setMinimumSize(150, 40)
        self.detection_mode_btn.clicked.connect(self.switch_to_detection_mode)
        layout.addWidget(self.detection_mode_btn)
        
        self.game_mode_btn = QPushButton("Game Mode")
        self.game_mode_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.game_mode_btn.setMinimumSize(150, 40)
        self.game_mode_btn.clicked.connect(self.switch_to_game_mode)
        layout.addWidget(self.game_mode_btn)
        
        return header
    
    def create_footer(self) -> QWidget:
        """Create the application footer."""
        footer = QFrame()
        footer.setFrameStyle(QFrame.StyledPanel)
        footer.setMaximumHeight(50)
        
        layout = QHBoxLayout(footer)
        
        # Session info
        self.session_label = QLabel("Session: 00:00:00")
        self.session_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.session_label)
        
        layout.addStretch()
        
        # Version info
        version_label = QLabel("v1.0.0")
        version_label.setFont(QFont("Arial", 10))
        layout.addWidget(version_label)
        
        # Update session timer
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_time)
        self.session_timer.start(1000)
        
        return footer
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Load model action
        load_model_action = QAction('Load Model...', self)
        load_model_action.setShortcut('Ctrl+O')
        load_model_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_model_action)
        
        file_menu.addSeparator()
        
        # Export session action
        export_action = QAction('Export Session Data...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_session_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        # Camera settings
        camera_action = QAction('Camera Settings...', self)
        camera_action.triggered.connect(self.show_camera_settings)
        settings_menu.addAction(camera_action)
        
        # Model settings
        model_action = QAction('Model Settings...', self)
        model_action.triggered.connect(self.show_model_settings)
        settings_menu.addAction(model_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Documentation action
        docs_action = QAction('Documentation', self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
    
    def create_status_bar(self):
        """Create the application status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Add permanent widgets to status bar
        self.mode_status = QLabel("Mode: Detection")
        self.status_bar.addPermanentWidget(self.mode_status)
        
        self.camera_status = QLabel("Camera: Disconnected")
        self.status_bar.addPermanentWidget(self.camera_status)
        
        self.model_status = QLabel("Model: Not Loaded")
        self.status_bar.addPermanentWidget(self.model_status)
    
    def init_components(self):
        """Initialize core components."""
        try:
            # Initialize detector
            self.detector = RPSDetector(
                self.model_path,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            self.model_status.setText(f"Model: Loaded ({self.device})")
            
            # Initialize camera
            self.camera = CameraManager(self.camera_id)
            if self.camera.open_camera():
                self.camera_status.setText("Camera: Connected")
            else:
                self.camera_status.setText("Camera: Failed")
                QMessageBox.warning(
                    self, "Camera Error", 
                    "Failed to open camera. Please check your camera connection."
                )
            
            # Create widgets
            self.detector_widget = DetectorWidget(
                self.model_path, self.camera_id, self.confidence_threshold
            )
            self.game_widget = GameWidget(self.detector, self.camera)
            
            # Add widgets to stacked widget
            self.stacked_widget.addWidget(self.detector_widget)
            self.stacked_widget.addWidget(self.game_widget)
            
            # Set initial mode
            self.switch_to_detection_mode()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Initialization Error",
                f"Failed to initialize components: {str(e)}"
            )
    
    def setup_connections(self):
        """Setup signal connections between components."""
        if self.detector_widget:
            self.detector_widget.screenshot_saved.connect(self.on_screenshot_saved)
            self.detector_widget.detection_event.connect(self.on_detection_event)
        
        if self.game_widget:
            self.game_widget.game_finished.connect(self.on_game_finished)
            self.game_widget.round_completed.connect(self.on_round_completed)
    
    def switch_to_detection_mode(self):
        """Switch to detection mode."""
        self.current_mode = "detection"
        self.stacked_widget.setCurrentIndex(0)
        self.mode_status.setText("Mode: Detection")
        
        # Update button states
        self.detection_mode_btn.setEnabled(False)
        self.game_mode_btn.setEnabled(True)
        
        self.status_bar.showMessage("Switched to Detection Mode")
    
    def switch_to_game_mode(self):
        """Switch to game mode."""
        self.current_mode = "game"
        self.stacked_widget.setCurrentIndex(1)
        self.mode_status.setText("Mode: Game")
        
        # Update button states
        self.detection_mode_btn.setEnabled(True)
        self.game_mode_btn.setEnabled(False)
        
        self.status_bar.showMessage("Switched to Game Mode")
    
    def update_session_time(self):
        """Update session time display."""
        elapsed = datetime.now() - self.session_data['start_time']
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"Session: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        self.session_label.setText(time_str)
    
    def on_screenshot_saved(self, filename: str):
        """Handle screenshot saved event."""
        self.status_bar.showMessage(f"Screenshot saved: {filename}", 3000)
    
    def on_detection_event(self, detection_info: Dict[str, Any]):
        """Handle detection event."""
        self.session_data['total_detections'] += 1
    
    def on_game_finished(self, game_data: Dict[str, Any]):
        """Handle game finished event."""
        self.session_data['games_played'] += 1
        
        # Show game results
        result_msg = f"Game Finished!\n\n"
        result_msg += f"Final Score: You {game_data['user_score']} - {game_data['ai_score']} AI\n"
        result_msg += f"Total Rounds: {game_data['total_rounds']}\n"
        result_msg += f"Ties: {game_data['tie_count']}\n"
        result_msg += f"Result: {game_data['overall_result']}"
        
        QMessageBox.information(self, "Game Results", result_msg)
    
    def on_round_completed(self, round_data: Dict[str, Any]):
        """Handle round completed event."""
        pass  # Could be used for round-by-round logging
    
    def load_model_dialog(self):
        """Show load model dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.pt *.onnx);;All Files (*)"
        )
        
        if filename:
            try:
                # Reinitialize detector with new model
                self.detector = RPSDetector(filename, device=self.device)
                self.model_path = filename
                self.model_status.setText(f"Model: {filename.split('/')[-1]}")
                self.status_bar.showMessage(f"Model loaded: {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Model Load Error",
                    f"Failed to load model: {str(e)}"
                )
    
    def export_session_data(self):
        """Export session data to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Session Data", 
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                import json
                session_data = {
                    'session_info': self.session_data,
                    'current_mode': self.current_mode,
                    'model_path': self.model_path,
                    'camera_id': self.camera_id,
                    'export_time': datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)
                
                self.status_bar.showMessage(f"Session data exported: {filename}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Failed to export session data: {str(e)}"
                )
    
    def show_camera_settings(self):
        """Show camera settings dialog."""
        QMessageBox.information(
            self, "Camera Settings",
            "Camera settings dialog would be implemented here."
        )
    
    def show_model_settings(self):
        """Show model settings dialog."""
        QMessageBox.information(
            self, "Model Settings",
            "Model settings dialog would be implemented here."
        )
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>AdaptiveRPS-Vision</h2>
        <p>Version 1.0.0</p>
        <p>An intelligent Rock-Paper-Scissors detection system using YOLOv8 
        object detection with reinforcement learning-based adaptive AI opponent.</p>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time hand gesture detection</li>
        <li>Adaptive AI opponent</li>
        <li>Interactive gaming experience</li>
        <li>Performance analytics</li>
        </ul>
        
        <p><b>Technology Stack:</b></p>
        <ul>
        <li>YOLOv8 for object detection</li>
        <li>PyTorch for deep learning</li>
        <li>PyQt5 for GUI</li>
        <li>OpenCV for computer vision</li>
        </ul>
        
        <p>© 2025 AdaptiveRPS-Vision Team</p>
        """
        
        QMessageBox.about(self, "About AdaptiveRPS-Vision", about_text)
    
    def show_documentation(self):
        """Show documentation."""
        QMessageBox.information(
            self, "Documentation",
            "Documentation can be found in the docs/ directory or online at:\n"
            "https://github.com/yourusername/adaptive-rps-vision"
        )
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D3142;
                color: #EAE8FF;
            }
            QWidget {
                background-color: #2D3142;
                color: #EAE8FF;
            }
            QPushButton {
                background-color: #4F5D75;
                color: #EAE8FF;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BFC0C0;
                color: #2D3142;
            }
            QPushButton:pressed {
                background-color: #9A9A9A;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
            QMenuBar {
                background-color: #4F5D75;
                color: #EAE8FF;
                border: none;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #BFC0C0;
                color: #2D3142;
            }
            QMenu {
                background-color: #4F5D75;
                color: #EAE8FF;
                border: 1px solid #BFC0C0;
            }
            QMenu::item:selected {
                background-color: #BFC0C0;
                color: #2D3142;
            }
            QStatusBar {
                background-color: #4F5D75;
                color: #EAE8FF;
                border-top: 1px solid #BFC0C0;
            }
            QFrame {
                background-color: #4F5D75;
                border: 1px solid #BFC0C0;
                border-radius: 4px;
            }
            QLabel {
                color: #EAE8FF;
            }
        """)
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Clean up resources
        if self.detector_widget:
            self.detector_widget.close()
        
        if self.camera:
            self.camera.release()
        
        event.accept()
