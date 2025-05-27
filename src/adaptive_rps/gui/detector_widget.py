"""
Detector widget for real-time gesture detection display.

This widget handles the detection mode interface, showing live camera feed
with gesture detection overlays and performance metrics.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QGroupBox, QProgressBar, QTextEdit,
                            QGridLayout, QFrame, QSlider, QSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from ..core.detector import RPSDetector
from ..utils.camera import CameraManager
from ..utils.colors import Colors


class DetectionThread(QThread):
    """Thread for running detection inference."""
    
    detection_ready = pyqtSignal(list)  # Emits detection results
    frame_ready = pyqtSignal(np.ndarray)  # Emits processed frame
    fps_update = pyqtSignal(float)  # Emits FPS value
    
    def __init__(self, detector: RPSDetector, camera: CameraManager):
        super().__init__()
        self.detector = detector
        self.camera = camera
        self.running = False
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def run(self):
        """Main detection loop."""
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read_frame()
            if not ret:
                continue
            
            start_time = time.time()
            
            # Run detection
            detections = self.detector.detect(frame)
            
            # Draw detections on frame
            result_frame = self.detector.visualize_detections(frame, detections)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                fps = self.fps_counter / (time.time() - self.fps_start_time)
                self.fps_update.emit(fps)
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Emit results
            self.detection_ready.emit(detections)
            self.frame_ready.emit(result_frame)
            
            # Small delay to prevent overwhelming the GUI
            self.msleep(30)
    
    def stop(self):
        """Stop the detection thread."""
        self.running = False
        self.wait()


class DetectorWidget(QWidget):
    """
    Widget for real-time gesture detection interface.
    
    Provides live camera feed with detection overlays, performance metrics,
    detection history, and control options.
    """
    
    # Signals
    screenshot_saved = pyqtSignal(str)
    detection_event = pyqtSignal(dict)
    
    def __init__(self, model_path: str, camera_id: int = 0, confidence: float = 0.25):
        super().__init__()
        
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence
        
        # Initialize components
        self.detector: Optional[RPSDetector] = None
        self.camera: Optional[CameraManager] = None
        self.detection_thread: Optional[DetectionThread] = None
        
        # Detection tracking
        self.detection_history: List[Dict[str, Any]] = []
        self.detection_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.session_start_time = datetime.now()
        
        # Performance tracking
        self.current_fps = 0.0
        self.frame_count = 0
        
        # UI setup
        self.init_ui()
        self.setup_connections()
        
        # Initialize detector and camera
        self.init_detector()
        self.init_camera()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        
        # Left panel - Camera feed
        camera_panel = self.create_camera_panel()
        layout.addWidget(camera_panel, 3)
        
        # Right panel - Controls and info
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 2)
    
    def create_camera_panel(self) -> QGroupBox:
        """Create the camera display panel."""
        panel = QGroupBox("Live Camera Feed")
        layout = QVBoxLayout(panel)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #BFC0C0;
                border-radius: 5px;
                background-color: #1a1a1a;
                color: #EAE8FF;
            }
        """)
        self.camera_label.setText("Camera feed will appear here")
        layout.addWidget(self.camera_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.fps_label)
        
        self.frame_count_label = QLabel("Frames: 0")
        self.frame_count_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.frame_count_label)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.detection_count_label)
        
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        return panel
    
    def create_control_panel(self) -> QWidget:
        """Create the control and information panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Current detection display
        detection_group = self.create_detection_display()
        layout.addWidget(detection_group)
        
        # Settings panel
        settings_group = self.create_settings_panel()
        layout.addWidget(settings_group)
        
        # Statistics panel
        stats_group = self.create_statistics_panel()
        layout.addWidget(stats_group)
        
        # Detection history
        history_group = self.create_history_panel()
        layout.addWidget(history_group)
        
        # Control buttons
        buttons_group = self.create_buttons_panel()
        layout.addWidget(buttons_group)
        
        layout.addStretch()
        return widget
    
    def create_detection_display(self) -> QGroupBox:
        """Create current detection display."""
        group = QGroupBox("Current Detection")
        layout = QVBoxLayout(group)
        
        self.current_detection_label = QLabel("No detection")
        self.current_detection_label.setAlignment(Qt.AlignCenter)
        self.current_detection_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.current_detection_label.setMinimumHeight(80)
        self.current_detection_label.setStyleSheet("""
            QLabel {
                border: 2px solid #4F5D75;
                border-radius: 10px;
                background-color: #2D3142;
                color: #EAE8FF;
                padding: 10px;
            }
        """)
        layout.addWidget(self.current_detection_label)
        
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.confidence_label)
        
        return group
    
    def create_settings_panel(self) -> QGroupBox:
        """Create settings control panel."""
        group = QGroupBox("Detection Settings")
        layout = QGridLayout(group)
        
        # Confidence threshold slider
        layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(int(self.confidence_threshold * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        layout.addWidget(self.confidence_slider, 0, 1)
        
        self.confidence_value_label = QLabel(f"{self.confidence_threshold:.2f}")
        layout.addWidget(self.confidence_value_label, 0, 2)
        
        # Device selection
        layout.addWidget(QLabel("Device:"), 1, 0)
        self.device_label = QLabel("Loading...")
        layout.addWidget(self.device_label, 1, 1, 1, 2)
        
        return group
    
    def create_statistics_panel(self) -> QGroupBox:
        """Create statistics display panel."""
        group = QGroupBox("Session Statistics")
        layout = QGridLayout(group)
        
        # Detection counts
        layout.addWidget(QLabel("Rock:"), 0, 0)
        self.rock_count_label = QLabel("0")
        layout.addWidget(self.rock_count_label, 0, 1)
        
        layout.addWidget(QLabel("Paper:"), 1, 0)
        self.paper_count_label = QLabel("0")
        layout.addWidget(self.paper_count_label, 1, 1)
        
        layout.addWidget(QLabel("Scissors:"), 2, 0)
        self.scissors_count_label = QLabel("0")
        layout.addWidget(self.scissors_count_label, 2, 1)
        
        # Session info
        layout.addWidget(QLabel("Session Time:"), 3, 0)
        self.session_time_label = QLabel("00:00:00")
        layout.addWidget(self.session_time_label, 3, 1)
        
        # Update timer for session time
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_time)
        self.session_timer.start(1000)  # Update every second
        
        return group
    
    def create_history_panel(self) -> QGroupBox:
        """Create detection history panel."""
        group = QGroupBox("Recent Detections")
        layout = QVBoxLayout(group)
        
        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(150)
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.history_text)
        
        return group
    
    def create_buttons_panel(self) -> QGroupBox:
        """Create control buttons panel."""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
        # Screenshot button
        self.screenshot_button = QPushButton("Save Screenshot")
        self.screenshot_button.clicked.connect(self.save_screenshot)
        layout.addWidget(self.screenshot_button)
        
        # Clear history button
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self.clear_detection_history)
        layout.addWidget(self.clear_history_button)
        
        # Reset statistics button
        self.reset_stats_button = QPushButton("Reset Statistics")
        self.reset_stats_button.clicked.connect(self.reset_statistics)
        layout.addWidget(self.reset_stats_button)
        
        # Toggle detection button
        self.toggle_detection_button = QPushButton("Stop Detection")
        self.toggle_detection_button.clicked.connect(self.toggle_detection)
        layout.addWidget(self.toggle_detection_button)
        
        return group
    
    def setup_connections(self):
        """Setup signal connections."""
        pass
    
    def init_detector(self):
        """Initialize the gesture detector."""
        try:
            self.detector = RPSDetector(
                self.model_path,
                confidence_threshold=self.confidence_threshold
            )
            device = self.detector.device
            self.device_label.setText(f"{device}")
        except Exception as e:
            self.device_label.setText(f"Error: {str(e)}")
    
    def init_camera(self):
        """Initialize the camera."""
        try:
            self.camera = CameraManager(self.camera_id)
            if self.camera.open_camera():
                self.start_detection()
            else:
                self.camera_label.setText("Failed to open camera")
        except Exception as e:
            self.camera_label.setText(f"Camera error: {str(e)}")
    
    def start_detection(self):
        """Start the detection thread."""
        if self.detector and self.camera:
            self.detection_thread = DetectionThread(self.detector, self.camera)
            self.detection_thread.detection_ready.connect(self.on_detection_ready)
            self.detection_thread.frame_ready.connect(self.on_frame_ready)
            self.detection_thread.fps_update.connect(self.on_fps_update)
            self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the detection thread."""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
    
    def on_detection_ready(self, detections: List[Tuple]):
        """Handle new detection results."""
        self.frame_count += 1
        self.frame_count_label.setText(f"Frames: {self.frame_count}")
        
        if detections:
            # Get best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x[1])
            class_name, confidence, bbox = best_detection
            
            # Update current detection display
            self.current_detection_label.setText(class_name.upper())
            self.confidence_label.setText(f"Confidence: {confidence:.3f}")
            
            # Set color based on class
            color_map = {
                'rock': '#FF5555',
                'paper': '#55FF55',
                'scissors': '#FFFF55'
            }
            color = color_map.get(class_name, '#EAE8FF')
            self.current_detection_label.setStyleSheet(f"""
                QLabel {{
                    border: 2px solid {color};
                    border-radius: 10px;
                    background-color: #2D3142;
                    color: {color};
                    padding: 10px;
                }}
            """)
            
            # Update detection counts
            if class_name in self.detection_counts:
                self.detection_counts[class_name] += 1
                self.update_detection_counts()
            
            # Add to history
            detection_info = {
                'timestamp': datetime.now(),
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            }
            self.detection_history.append(detection_info)
            self.update_detection_history()
            
            # Emit detection event
            self.detection_event.emit(detection_info)
        
        else:
            # No detections
            self.current_detection_label.setText("No detection")
            self.confidence_label.setText("Confidence: --")
            self.current_detection_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4F5D75;
                    border-radius: 10px;
                    background-color: #2D3142;
                    color: #EAE8FF;
                    padding: 10px;
                }
            """)
        
        # Update total detection count
        total_detections = sum(self.detection_counts.values())
        self.detection_count_label.setText(f"Detections: {total_detections}")
    
    def on_frame_ready(self, frame: np.ndarray):
        """Handle new processed frame."""
        # Convert frame to QPixmap and display
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
    
    def on_fps_update(self, fps: float):
        """Handle FPS update."""
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_confidence_threshold(self, value: int):
        """Update confidence threshold from slider."""
        threshold = value / 100.0
        self.confidence_threshold = threshold
        self.confidence_value_label.setText(f"{threshold:.2f}")
        
        if self.detector:
            self.detector.set_confidence_threshold(threshold)
    
    def update_detection_counts(self):
        """Update detection count displays."""
        self.rock_count_label.setText(str(self.detection_counts['rock']))
        self.paper_count_label.setText(str(self.detection_counts['paper']))
        self.scissors_count_label.setText(str(self.detection_counts['scissors']))
    
    def update_detection_history(self):
        """Update detection history display."""
        # Keep only last 10 detections
        if len(self.detection_history) > 10:
            self.detection_history = self.detection_history[-10:]
        
        # Format history text
        history_text = ""
        for detection in reversed(self.detection_history[-5:]):  # Show last 5
            timestamp = detection['timestamp'].strftime("%H:%M:%S")
            class_name = detection['class']
            confidence = detection['confidence']
            history_text += f"{timestamp} - {class_name.upper()} ({confidence:.2f})\n"
        
        self.history_text.setPlainText(history_text)
    
    def update_session_time(self):
        """Update session time display."""
        elapsed = datetime.now() - self.session_start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        self.session_time_label.setText(time_str)
    
    def save_screenshot(self):
        """Save current frame as screenshot."""
        if self.camera_label.pixmap():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
            pixmap = self.camera_label.pixmap()
            if pixmap.save(filename):
                self.screenshot_saved.emit(filename)
                self.add_to_history(f"Screenshot saved: {filename}")
    
    def clear_detection_history(self):
        """Clear detection history."""
        self.detection_history.clear()
        self.history_text.clear()
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.detection_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.frame_count = 0
        self.session_start_time = datetime.now()
        
        self.update_detection_counts()
        self.frame_count_label.setText("Frames: 0")
        self.detection_count_label.setText("Detections: 0")
        self.clear_detection_history()
    
    def toggle_detection(self):
        """Toggle detection on/off."""
        if self.detection_thread and self.detection_thread.running:
            self.stop_detection()
            self.toggle_detection_button.setText("Start Detection")
        else:
            self.start_detection()
            self.toggle_detection_button.setText("Stop Detection")
    
    def add_to_history(self, message: str):
        """Add message to detection history."""
        current_text = self.history_text.toPlainText()
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_text = f"{timestamp} - {message}\n{current_text}"
        
        # Limit history length
        lines = new_text.split('\n')
        if len(lines) > 20:
            lines = lines[:20]
        
        self.history_text.setPlainText('\n'.join(lines))
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_detection()
        if self.camera:
            self.camera.release()
        event.accept()
