"""Model configuration settings."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for YOLO model training and inference."""
    
    # Model settings
    model_name: str = "yolov8n.pt"
    num_classes: int = 3
    class_names: Dict[int, str] = None
    
    # Training settings
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    device: str = "auto"  # "auto", "cpu", "cuda", or device ID
    
    # Optimization settings
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    final_lr_factor: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Training schedule
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    patience: int = 15
    
    # Loss weights
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5
    
    # Inference settings
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    
    # Export settings
    export_formats: list = None
    dynamic_export: bool = True
    simplify_onnx: bool = True
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
        if self.export_formats is None:
            self.export_formats = ['onnx', 'torchscript']
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'device': self.device,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'final_lr_factor': self.final_lr_factor,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'warmup_momentum': self.warmup_momentum,
            'warmup_bias_lr': self.warmup_bias_lr,
            'patience': self.patience,
            'box_loss_weight': self.box_loss_weight,
            'cls_loss_weight': self.cls_loss_weight,
            'dfl_loss_weight': self.dfl_loss_weight,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'export_formats': self.export_formats,
            'dynamic_export': self.dynamic_export,
            'simplify_onnx': self.simplify_onnx
        }
