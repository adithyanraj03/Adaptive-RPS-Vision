"""Dataset configuration settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for dataset management."""
    
    # Dataset paths
    dataset_root: str = "./datasets/rps_dataset"
    train_images: str = "train/images"
    val_images: str = "valid/images"
    test_images: str = "test/images"
    
    # Dataset info
    dataset_name: str = "rock_paper_scissors"
    dataset_version: str = "v1.0"
    
    # Class mapping
    class_names: Dict[int, str] = None
    
    # Augmentation settings
    enable_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = {0: 'rock', 1: 'paper', 2: 'scissors'}
    
    @property
    def train_path(self) -> Path:
        """Get full path to training images."""
        return Path(self.dataset_root) / self.train_images
    
    @property
    def val_path(self) -> Path:
        """Get full path to validation images."""
        return Path(self.dataset_root) / self.val_images
    
    @property
    def test_path(self) -> Path:
        """Get full path to test images."""
        return Path(self.dataset_root) / self.test_images
    
    def create_yaml_config(self, output_path: Optional[str] = None) -> str:
        """Create YAML configuration file for YOLO training."""
        if output_path is None:
            output_path = "dataset_config.yaml"
        
        yaml_content = {
            'path': str(Path(self.dataset_root).absolute()),
            'train': self.train_images,
            'val': self.val_images,
            'test': self.test_images,
            'names': self.class_names
        }
        
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        return output_path
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create DatasetConfig from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DatasetConfig to dictionary."""
        return {
            'dataset_root': self.dataset_root,
            'train_images': self.train_images,
            'val_images': self.val_images,
            'test_images': self.test_images,
            'dataset_name': self.dataset_name,
            'dataset_version': self.dataset_version,
            'class_names': self.class_names,
            'enable_augmentation': self.enable_augmentation,
            'augmentation_probability': self.augmentation_probability,
            'validation_split': self.validation_split,
            'test_split': self.test_split
        }
