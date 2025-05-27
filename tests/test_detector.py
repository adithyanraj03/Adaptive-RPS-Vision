"""
Unit tests for RPSDetector class.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.core.detector import RPSDetector


class TestRPSDetector:
    """Test cases for RPSDetector class."""
    
    @pytest.fixture
    def mock_model_path(self):
        """Mock model path for testing."""
        return "test_model.pt"
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_detector_initialization(self, mock_yolo, mock_model_path):
        """Test detector initialization."""
        detector = RPSDetector(mock_model_path)
        
        assert detector.model_path == mock_model_path
        assert detector.confidence_threshold == 0.25
        assert detector.class_names == {0: 'rock', 1: 'paper', 2: 'scissors'}
        mock_yolo.assert_called_once_with(mock_model_path)
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_detector_with_custom_params(self, mock_yolo, mock_model_path):
        """Test detector with custom parameters."""
        detector = RPSDetector(
            mock_model_path,
            confidence_threshold=0.5,
            iou_threshold=0.7,
            device="cpu"
        )
        
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.7
        assert detector.device == "cpu"
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_detect_empty_image(self, mock_yolo, mock_model_path):
        """Test detection with empty image."""
        detector = RPSDetector(mock_model_path)
        
        # Test with None image
        result = detector.detect(None)
        assert result == []
        
        # Test with empty array
        empty_image = np.array([])
        result = detector.detect(empty_image)
        assert result == []
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_detect_with_results(self, mock_yolo, mock_model_path, sample_image):
        """Test detection with mock results."""
        # Setup mock
        mock_model_instance = Mock()
        mock_yolo.return_value = mock_model_instance
        
        # Create mock detection results
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 200]
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.95
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 1  # paper
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        
        mock_model_instance.return_value = [mock_result]
        
        detector = RPSDetector(mock_model_path)
        results = detector.detect(sample_image)
        
        assert len(results) == 1
        class_name, confidence, bbox = results[0]
        assert class_name == 'paper'
        assert confidence == 0.95
        assert bbox == (100, 100, 200, 200)
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_set_confidence_threshold(self, mock_yolo, mock_model_path):
        """Test setting confidence threshold."""
        detector = RPSDetector(mock_model_path)
        
        detector.set_confidence_threshold(0.8)
        assert detector.confidence_threshold == 0.8
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            detector.set_confidence_threshold(1.5)
        
        with pytest.raises(ValueError):
            detector.set_confidence_threshold(-0.1)
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_get_class_color(self, mock_yolo, mock_model_path):
        """Test getting class colors."""
        detector = RPSDetector(mock_model_path)
        
        assert detector.get_class_color('rock') == (0, 0, 255)
        assert detector.get_class_color('paper') == (0, 255, 0)
        assert detector.get_class_color('scissors') == (255, 255, 0)
        assert detector.get_class_color('unknown') == (128, 128, 128)
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_performance_stats(self, mock_yolo, mock_model_path):
        """Test performance statistics."""
        detector = RPSDetector(mock_model_path)
        
        stats = detector.get_performance_stats()
        
        assert 'detection_counts' in stats
        assert 'total_detections' in stats
        assert 'model_info' in stats
        assert stats['detection_counts'] == {'rock': 0, 'paper': 0, 'scissors': 0}
        assert stats['total_detections'] == 0
    
    @patch('adaptive_rps.core.detector.YOLO')
    def test_reset_stats(self, mock_yolo, mock_model_path):
        """Test resetting statistics."""
        detector = RPSDetector(mock_model_path)
        
        # Manually set some counts
        detector.detection_counts['rock'] = 5
        detector.inference_times = [0.1, 0.2, 0.3]
        
        detector.reset_stats()
        
        assert detector.detection_counts == {'rock': 0, 'paper': 0, 'scissors': 0}
        assert len(detector.inference_times) == 0
