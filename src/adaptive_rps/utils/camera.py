"""Camera management utilities."""

import cv2
import logging
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """Manages camera operations for the application."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera manager.
        
        Args:
            camera_id: Camera device ID
            width: Camera frame width
            height: Camera frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
        self.logger = logging.getLogger(__name__)
    
    def open_camera(self) -> bool:
        """
        Open the camera device.
        
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"Could not open camera with ID {self.camera_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_opened = True
            self.logger.info(f"Camera {self.camera_id} opened successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_opened or self.camera is None:
            return False, None
        
        try:
            ret, frame = self.camera.read()
            if ret:
                # Flip frame horizontally for natural view
                frame = cv2.flip(frame, 1)
            return ret, frame
            
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return False, None
    
    def release(self) -> None:
        """Release the camera resource."""
        if self.camera is not None:
            self.camera.release()
            self.is_opened = False
            self.logger.info("Camera released")
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            dict: Camera properties
        """
        if not self.is_opened or self.camera is None:
            return {}
        
        return {
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
            'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.open_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
