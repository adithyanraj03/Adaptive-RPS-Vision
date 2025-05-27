#!/usr/bin/env python3
"""
Custom training example for AdaptiveRPS-Vision.

This script demonstrates how to train a custom model with
specific configurations and hyperparameters.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.core.trainer import RPSTrainer
from adaptive_rps.utils.colors import Colors, print_header
from config.model_config import ModelConfig
from config.dataset_config import DatasetConfig


def create_custom_config():
    """Create custom training configuration."""
    config = ModelConfig(
        model_name="yolov8s.pt",  # Use small model for better accuracy
        epochs=150,
        batch_size=32,
        image_size=640,
        device="auto",
        
        # Custom optimization settings
        learning_rate=0.002,
        final_lr_factor=0.05,
        momentum=0.95,
        weight_decay=0.001,
        
        # Advanced training schedule
        warmup_epochs=5.0,
        warmup_momentum=0.9,
        patience=25,
        
        # Loss weights for better performance
        box_loss_weight=8.0,
        cls_loss_weight=0.6,
        dfl_loss_weight=1.8,
        
        # Export settings
        export_formats=['onnx', 'torchscript', 'engine'],
        dynamic_export=False,
        simplify_onnx=True
    )
    
    return config


def create_dataset_config():
    """Create dataset configuration."""
    config = DatasetConfig(
        dataset_root="./datasets/custom_rps_dataset",
        dataset_name="custom_rock_paper_scissors",
        dataset_version="v2.0",
        
        # Enhanced augmentation
        enable_augmentation=True,
        augmentation_probability=0.7,
        
        # Custom splits
        validation_split=0.15,
        test_split=0.15
    )
    
    return config


def main():
    """Run custom training example."""
    print_header("CUSTOM TRAINING EXAMPLE")
    
    # Create configurations
    model_config = create_custom_config()
    dataset_config = create_dataset_config()
    
    print(f"{Colors.BOLD}Custom Training Configuration:{Colors.ENDC}")
    print(f"├─ Model: {Colors.CYAN}{model_config.model_name}{Colors.ENDC}")
    print(f"├─ Epochs: {Colors.CYAN}{model_config.epochs}{Colors.ENDC}")
    print(f"├─ Batch Size: {Colors.CYAN}{model_config.batch_size}{Colors.ENDC}")
    print(f"├─ Learning Rate: {Colors.CYAN}{model_config.learning_rate}{Colors.ENDC}")
    print(f"├─ Patience: {Colors.CYAN}{model_config.patience}{Colors.ENDC}")
    print(f"└─ Export Formats: {Colors.CYAN}{model_config.export_formats}{Colors.ENDC}")
    
    # Create dataset YAML
    dataset_yaml = dataset_config.create_yaml_config("custom_dataset.yaml")
    print(f"\n{Colors.info('Dataset configuration created')}")
    
    # Initialize trainer
    trainer = RPSTrainer(model_config)
    
    # Custom run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"custom_training_{timestamp}"
    
    try:
        print_header("STARTING CUSTOM TRAINING")
        
        # Train with custom settings
        results = trainer.train(
            dataset_yaml=dataset_yaml,
            output_dir="./runs/custom",
            run_name=run_name,
            save_period=10,  # Save more frequently
            validate=True
        )
        
        print_header("TRAINING COMPLETED")
        print(f"{Colors.success('Custom training completed successfully!')}")
        
        # Export model in multiple formats
        print(f"\n{Colors.info('Exporting model in multiple formats...')}")
        export_paths = trainer.export_model()
        
        for format_name, path in export_paths.items():
            print(f"├─ {format_name.upper()}: {Colors.CYAN}{path}{Colors.ENDC}")
        
        # Run final validation
        print(f"\n{Colors.info('Running final validation...')}")
        val_results = trainer.validate(dataset_yaml, split='test')
        
        print_header("CUSTOM TRAINING SUMMARY")
        print(f"{Colors.success('All training steps completed successfully!')}")
        print(f"Model saved to: {Colors.CYAN}{trainer.model_save_path}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.error('Custom training failed:')}")
        print(f"{Colors.FAIL}{str(e)}{Colors.ENDC}")
        raise


if __name__ == "__main__":
    main()
