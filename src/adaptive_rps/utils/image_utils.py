"""Image processing utilities."""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


class ImageProcessor:
    """Utilities for image processing and manipulation."""
    
    @staticmethod
    def resize_maintain_aspect_ratio(
        image: np.ndarray, 
        target_size: Tuple[int, int],
        fill_color: Tuple[int, int, int] = (114, 114, 114)
    ) -> np.ndarray:
        """
        Resize image maintaining aspect ratio and pad if necessary.
        
        Args:
            image: Input image
            target_size: (width, height) target size
            fill_color: RGB color for padding
            
        Returns:
            np.ndarray: Resized and padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
        
        # Calculate padding offsets
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    @staticmethod
    def draw_detection_box(
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection box with label on image.
        
        Args:
            image: Input image
            box: (x1, y1, x2, y2) bounding box coordinates
            class_name: Name of detected class
            confidence: Detection confidence
            color: BGR color for box and text
            thickness: Line thickness
            
        Returns:
            np.ndarray: Image with drawn detection
        """
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return image
    
    @staticmethod
    def get_class_color(class_name: str) -> Tuple[int, int, int]:
        """
        Get BGR color for a class name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        color_map = {
            'rock': (0, 0, 255),      # Red
            'paper': (0, 255, 0),     # Green
            'scissors': (255, 255, 0) # Yellow
        }
        return color_map.get(class_name.lower(), (128, 128, 128))
    
    @staticmethod
    def create_move_visualization(
        move: str, 
        size: Tuple[int, int] = (360, 480),
        background_color: Tuple[int, int, int] = (45, 45, 45)
    ) -> np.ndarray:
        """
        Create a visual representation of a move.
        
        Args:
            move: Move name ('rock', 'paper', 'scissors')
            size: (width, height) of the output image
            background_color: RGB background color
            
        Returns:
            np.ndarray: Image representing the move
        """
        width, height = size
        img = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        center_x, center_y = width // 2, height // 2
        color = ImageProcessor.get_class_color(move)
        
        if move.lower() == 'rock':
            # Draw a circle for rock
            cv2.circle(img, (center_x, center_y), 100, color, -1)
            text = "ROCK"
        elif move.lower() == 'paper':
            # Draw a rectangle for paper
            cv2.rectangle(
                img, 
                (center_x - 100, center_y - 100), 
                (center_x + 100, center_y + 100), 
                color, -1
            )
            text = "PAPER"
        elif move.lower() == 'scissors':
            # Draw scissors (two elongated ellipses)
            cv2.ellipse(
                img, (center_x - 30, center_y), (50, 120), 45, 0, 360, color, -1
            )
            cv2.ellipse(
                img, (center_x + 30, center_y), (50, 120), -45, 0, 360, color, -1
            )
            text = "SCISSORS"
        else:
            text = "UNKNOWN"
        
        # Add text label
        font_scale = 1.5
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        text_x = (width - text_width) // 2
        text_y = height - 50
        
        cv2.putText(
            img, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )
        
        return img
    
    @staticmethod
    def save_image(image: np.ndarray, filepath: str) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            filepath: Output file path
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            return cv2.imwrite(filepath, image)
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
