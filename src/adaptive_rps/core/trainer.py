"""Training module for RPS detection model."""

import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from ultralytics import YOLO

from ..utils.colors import Colors, print_header
from config.model_config import ModelConfig


class RPSTrainer:
    """Trainer class for Rock-Paper-Scissors detection model."""
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration object
        """
        self.config = model_config
        self.model: Optional[YOLO] = None
        self.model_save_path: Optional[str] = None
        
    def train(
        self,
        dataset_yaml: str,
        output_dir: str = "./runs",
        run_name: Optional[str] = None,
        resume: Optional[str] = None,
        save_period: int = 20,
        validate: bool = True
    ) -> Any:
        """
        Train the model.
        
        Args:
            dataset_yaml: Path to dataset YAML configuration
            output_dir: Output directory for results
            run_name: Name for this training run
            resume: Path to checkpoint to resume from
            save_period: Save model every N epochs
            validate: Whether to run validation
            
        Returns:
            Training results
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
        
        # Create output directory
        output_path = Path(output_dir) / run_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"{Colors.info('Loading model...')}")
        if resume:
            self.model = YOLO(resume)
            print(f"Resumed from: {Colors.CYAN}{resume}{Colors.ENDC}")
        else:
            self.model = YOLO(self.config.model_name)
            print(f"Loaded model: {Colors.CYAN}{self.config.model_name}{Colors.ENDC}")
        
        # Start training
        try:
            results = self.model.train(
                data=dataset_yaml,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.config.image_size,
                device=self.config.device,
                patience=self.config.patience,
                project=output_dir,
                name=run_name,
                verbose=True,
                exist_ok=True,
                pretrained=True,
                optimizer=self.config.optimizer,
                lr0=self.config.learning_rate,
                lrf=self.config.final_lr_factor,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                warmup_epochs=self.config.warmup_epochs,
                warmup_momentum=self.config.warmup_momentum,
                warmup_bias_lr=self.config.warmup_bias_lr,
                box=self.config.box_loss_weight,
                cls=self.config.cls_loss_weight,
                dfl=self.config.dfl_loss_weight,
                plots=True,
                save=True,
                save_period=save_period,
                val=validate
            )
            
            # Store model save path
            self.model_save_path = str(output_path / "weights" / "best.pt")
            
            return results
            
        except Exception as e:
            print(f"{Colors.error('Training failed:')} {str(e)}")
            raise
    
    def export_model(self, formats: Optional[list] = None) -> dict:
        """
        Export model to different formats.
        
        Args:
            formats: List of export formats
            
        Returns:
            Dictionary of export paths
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        if formats is None:
            formats = self.config.export_formats
        
        export_paths = {}
        
        for format_name in formats:
            try:
                print(f"Exporting to {format_name}...")
                export_path = self.model.export(
                    format=format_name,
                    dynamic=self.config.dynamic_export,
                    simplify=self.config.simplify_onnx if format_name == 'onnx' else False
                )
                export_paths[format_name] = export_path
                print(f"├─ {format_name.upper()}: {Colors.CYAN}{export_path}{Colors.ENDC}")
                
            except Exception as e:
                print(f"├─ {format_name.upper()}: {Colors.error('Failed')} - {str(e)}")
        
        return export_paths
    
    def validate(self, dataset_yaml: str, split: str = 'val') -> Any:
        """
        Run validation on the model.
        
        Args:
            dataset_yaml: Path to dataset YAML
            split: Dataset split to validate on
            
        Returns:
            Validation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        print(f"{Colors.info(f'Running validation on {split} set...')}")
        results = self.model.val(data=dataset_yaml, split=split)
        
        return results
