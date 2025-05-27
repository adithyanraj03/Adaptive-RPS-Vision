"""
Rock-Paper-Scissors gesture detection module.

This module provides the main detection functionality using YOLOv8
for real-time hand gesture recognition.
"""

import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from ultralytics import YOLO

from ..utils.image_utils import ImageProcessor


class RPSDetector:
    """
    Main detector class for Rock-Paper-Scissors gesture recognition.
    
    This class handles model loading, inference, and post-processing
    for real-time gesture detection.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize the RPS detector.
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        
        # Class names mapping
        self.class_names = {0: 'rock', 1: 'paper', 2: 'scissors'}
        self.num_classes = len(self.class_names)
        
        # Performance tracking
        self.inference_times = []
        self.detection_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        
        # Initialize model
        self.model = None
        self.is_loaded = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load the model
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """
        Setup and validate the compute device.
        
        Args:
            device: Requested device
            
        Returns:
            Validated device string
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        else:
            return device
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
            
            self.is_loaded = True
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(
        self,
        image: np.ndarray,
        return_confidence: bool = True,
        return_bbox: bool = True
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect gestures in an image.
        
        Args:
            image: Input image as numpy array
            return_confidence: Whether to return confidence scores
            return_bbox: Whether to return bounding boxes
            
        Returns:
            List of detections as (class_name, confidence, bbox) tuples
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if image is None or image.size == 0:
            return []
        
        try:
            import time
            start_time = time.time()
            
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            # Process results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Extract box data
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names.get(class_id, 'unknown')
                        
                        # Update detection counts
                        if class_name in self.detection_counts:
                            self.detection_counts[class_name] += 1
                        
                        # Create detection tuple
                        detection = [class_name]
                        if return_confidence:
                            detection.append(confidence)
                        if return_bbox:
                            detection.append((x1, y1, x2, y2))
                        
                        detections.append(tuple(detection))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        return_confidence: bool = True,
        return_bbox: bool = True
    ) -> List[List[Tuple[str, float, Tuple[int, int, int, int]]]]:
        """
        Detect gestures in a batch of images.
        
        Args:
            images: List of input images
            return_confidence: Whether to return confidence scores
            return_bbox: Whether to return bounding boxes
            
        Returns:
            List of detection lists for each image
        """
        batch_results = []
        
        for image in images:
            detections = self.detect(image, return_confidence, return_bbox)
            batch_results.append(detections)
        
        return batch_results
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Tuple[str, float, Tuple[int, int, int, int]]],
        show_confidence: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image: Input image
            detections: List of detections
            show_confidence: Whether to show confidence scores
            thickness: Line thickness for bounding boxes
            
        Returns:
            Image with visualized detections
        """
        result_image = image.copy()
        
        for detection in detections:
            class_name = detection[0]
            confidence = detection[1] if len(detection) > 1 else 1.0
            bbox = detection[2] if len(detection) > 2 else None
            
            if bbox is None:
                continue
            
            # Get class color
            color = self.get_class_color(class_name)
            
            # Draw detection
            result_image = ImageProcessor.draw_detection_box(
                result_image, bbox, class_name, confidence, color, thickness
            )
        
        return result_image
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get BGR color for a class name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            BGR color tuple
        """
        color_map = {
            'rock': (0, 0, 255),      # Red
            'paper': (0, 255, 0),     # Green
            'scissors': (255, 255, 0) # Yellow
        }
        return color_map.get(class_name.lower(), (128, 128, 128))
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Confidence threshold updated to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def set_iou_threshold(self, threshold: float):
        """
        Update the IoU threshold.
        
        Args:
            threshold: New IoU threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.iou_threshold = threshold
            self.logger.info(f"IoU threshold updated to {threshold}")
        else:
            raise ValueError("IoU threshold must be between 0.0 and 1.0")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = {
            'detection_counts': self.detection_counts.copy(),
            'total_detections': sum(self.detection_counts.values()),
            'model_info': {
                'model_path': self.model_path,
                'device': self.device,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            }
        }
        
        if self.inference_times:
            stats['performance'] = {
                'avg_inference_time': np.mean(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'fps': 1.0 / np.mean(self.inference_times),
                'inference_count': len(self.inference_times)
            }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.detection_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.inference_times.clear()
        self.logger.info("Performance statistics reset")
    
    def export_onnx(self, output_path: str, dynamic: bool = True, simplify: bool = True):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            dynamic: Whether to use dynamic input shapes
            simplify: Whether to simplify the ONNX model
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            self.model.export(
                format='onnx',
                dynamic=dynamic,
                simplify=simplify
            )
            self.logger.info(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise
    
    def __del__(self):
        """Cleanup when detector is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
