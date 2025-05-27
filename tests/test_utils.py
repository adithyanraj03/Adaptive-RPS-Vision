"""
Unit tests for utility modules.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.utils.colors import Colors
from adaptive_rps.utils.image_utils import ImageProcessor
from adaptive_rps.utils.camera import CameraManager


class TestColors:
    """Test cases for Colors utility class."""
    
    def test_colored_text(self):
        """Test colored text generation."""
        text = "test"
        colored = Colors.colored_text(text, Colors.GREEN)
        assert Colors.GREEN in colored
        assert Colors.ENDC in colored
        assert text in colored
    
    def test_success_text(self):
        """Test success text formatting."""
        text = "Success!"
        success_text = Colors.success(text)
        assert Colors.GREEN in success_text
        assert text in success_text
    
    def test_error_text(self):
        """Test error text formatting."""
        text = "Error!"
        error_text = Colors.error(text)
        assert Colors.FAIL in error_text
        assert text in error_text
    
    def test_warning_text(self):
        """Test warning text formatting."""
        text = "Warning!"
        warning_text = Colors.warning(text)
        assert Colors.WARNING in warning_text
        assert text in warning_text
    
    def test_info_text(self):
        """Test info text formatting."""
        text = "Info!"
        info_text = Colors.info(text)
        assert Colors.CYAN in info_text
        assert text in info_text


class TestImageProcessor:
    """Test cases for ImageProcessor utility class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_resize_maintain_aspect_ratio(self, sample_image):
        """Test image resizing with aspect ratio maintenance."""
        target_size = (200, 150)
        resized = ImageProcessor.resize_maintain_aspect_ratio(sample_image, target_size)
        
        assert resized.shape[:2] == (150, 200)  # height, width
        assert resized.dtype == np.uint8
    
    def test_get_class_color(self):
        """Test getting class colors."""
        assert ImageProcessor.get_class_color('rock') == (0, 0, 255)
        assert ImageProcessor.get_class_color('paper') == (0, 255, 0)
        assert ImageProcessor.get_class_color('scissors') == (255, 255, 0)
        assert ImageProcessor.get_class_color('unknown') == (128, 128, 128)
    
    def test_create_move_visualization(self):
        """Test creating move visualizations."""
        # Test rock visualization
        rock_img = ImageProcessor.create_move_visualization('rock', (100, 100))
        assert rock_img.shape == (100, 100, 3)
        assert rock_img.dtype == np.uint8
        
        # Test paper visualization
        paper_img = ImageProcessor.create_move_visualization('paper', (100, 100))
        assert paper_img.shape == (100, 100, 3)
        
        # Test scissors visualization
        scissors_img = ImageProcessor.create_move_visualization('scissors', (100, 100))
        assert scissors_img.shape == (100, 100, 3)
    
    def test_draw_detection_box(self, sample_image):
        """Test drawing detection boxes."""
        bbox = (10, 10, 50, 50)
        class_name = "rock"
        confidence = 0.95
        color = (0, 255, 0)
        
        result = ImageProcessor.draw_detection_box(
            sample_image, bbox, class_name, confidence, color
        )
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
        # The image should be modified (not equal to original)
        assert not np.array_equal(result, sample_image)
    
    @patch('cv2.imwrite')
    @patch('pathlib.Path.mkdir')
    def test_save_image(self, mock_mkdir, mock_imwrite, sample_image):
        """Test saving images."""
        mock_imwrite.return_value = True
        
        result = ImageProcessor.save_image(sample_image, "test/image.jpg")
        
        assert result is True
        mock_mkdir.assert_called_once()
        mock_imwrite.assert_called_once()


class TestCameraManager:
    """Test cases for CameraManager utility class."""
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_video_capture):
        """Test camera manager initialization."""
        camera = CameraManager(camera_id=0, width=640, height=480)
        
        assert camera.camera_id == 0
        assert camera.width == 640
        assert camera.height == 480
        assert camera.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_open_camera_success(self, mock_video_capture):
        """Test successful camera opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraManager()
        result = camera.open_camera()
        
        assert result is True
        assert camera.is_opened is True
        mock_cap.set.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_open_camera_failure(self, mock_video_capture):
        """Test camera opening failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        camera = CameraManager()
        result = camera.open_camera()
        
        assert result is False
        assert camera.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_read_frame_success(self, mock_video_capture):
        """Test successful frame reading."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        camera = CameraManager()
        camera.open_camera()
        
        ret, frame = camera.read_frame()
        
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    def test_read_frame_not_opened(self, mock_video_capture):
        """Test frame reading when camera not opened."""
        camera = CameraManager()
        
        ret, frame = camera.read_frame()
        
        assert ret is False
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_release_camera(self, mock_video_capture):
        """Test camera release."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraManager()
        camera.open_camera()
        camera.release()
        
        mock_cap.release.assert_called_once()
        assert camera.is_opened is False
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_video_capture):
        """Test camera as context manager."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        with CameraManager() as camera:
            assert camera.is_opened is True
        
        mock_cap.release.assert_called_once()
