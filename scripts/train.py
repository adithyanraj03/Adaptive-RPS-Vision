#!/usr/bin/env python3
"""
Training script for AdaptiveRPS-Vision model.

This script provides a comprehensive training pipeline for YOLOv8-based
rock-paper-scissors detection model with configurable parameters.
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.core.trainer import RPSTrainer
from adaptive_rps.utils.colors import Colors, print_header
from config.model_config import ModelConfig
from config.dataset_config import DatasetConfig


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AdaptiveRPS-Vision model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--dataset_yaml",
        type=str,
        help="Path to dataset YAML file (will be created if not provided)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, or device ID)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt",
        help="YOLO model to use"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0005,
        help="Weight decay"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./runs",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--name", 
        type=str,
        help="Name for this training run"
    )
    parser.add_argument(
        "--save_period", 
        type=int, 
        default=20,
        help="Save model every N epochs"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Flags
    parser.add_argument(
        "--no_export", 
        action="store_true",
        help="Skip model export after training"
    )
    parser.add_argument(
        "--no_validation", 
        action="store_true",
        help="Skip validation during training"
    )
    parser.add_argument(
        "--resume", 
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    return parser.parse_args()


def create_configs(args: argparse.Namespace) -> tuple[ModelConfig, DatasetConfig]:
    """Create configuration objects from command line arguments."""
    
    # Create model config
    model_config = ModelConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        learning_rate=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay
    )
    
    # Create dataset config
    dataset_config = DatasetConfig(
        dataset_root=args.dataset_path
    )
    
    return model_config, dataset_config


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Show training header
    print_header("ADAPTIVERPSVISION MODEL TRAINING")
    
    # Create configurations
    model_config, dataset_config = create_configs(args)
    
    # Print configuration
    print(f"{Colors.BOLD}Training Configuration:{Colors.ENDC}")
    print(f"├─ Dataset path:       {Colors.CYAN}{args.dataset_path}{Colors.ENDC}")
    print(f"├─ Model:              {Colors.CYAN}{args.model}{Colors.ENDC}")
    print(f"├─ Epochs:             {Colors.CYAN}{args.epochs}{Colors.ENDC}")
    print(f"├─ Batch size:         {Colors.CYAN}{args.batch_size}{Colors.ENDC}")
    print(f"├─ Image size:         {Colors.CYAN}{args.image_size}{Colors.ENDC}")
    print(f"├─ Device:             {Colors.CYAN}{args.device}{Colors.ENDC}")
    print(f"├─ Learning rate:      {Colors.CYAN}{args.lr}{Colors.ENDC}")
    print(f"├─ Patience:           {Colors.CYAN}{args.patience}{Colors.ENDC}")
    print(f"└─ Random seed:        {Colors.CYAN}{args.seed}{Colors.ENDC}")
    
    # Create dataset YAML if not provided
    dataset_yaml = args.dataset_yaml
    if dataset_yaml is None:
        print(f"\n{Colors.info('Creating dataset configuration...')}")
        dataset_yaml = dataset_config.create_yaml_config("dataset_config.yaml")
        print(f"Dataset config saved to: {Colors.CYAN}{dataset_yaml}{Colors.ENDC}")
    
    # Generate run name if not provided
    run_name = args.name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"
    
    # Create trainer
    try:
        trainer = RPSTrainer(model_config=model_config)
        
        # Start training
        print_header("STARTING TRAINING")
        print(f"{Colors.info('Training started at:')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = trainer.train(
            dataset_yaml=dataset_yaml,
            output_dir=args.output_dir,
            run_name=run_name,
            resume=args.resume,
            save_period=args.save_period,
            validate=not args.no_validation
        )
        
        print_header("TRAINING COMPLETED")
        print(f"{Colors.success('Training completed successfully!')}")
        
        # Export model if requested
        if not args.no_export:
            print(f"\n{Colors.info('Exporting model...')}")
            trainer.export_model()
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n{Colors.BOLD}Final Metrics:{Colors.ENDC}")
            if 'metrics/mAP50(B)' in metrics:
                print(f"├─ mAP50:      {Colors.GREEN}{metrics['metrics/mAP50(B)']:.4f}{Colors.ENDC}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"└─ mAP50-95:   {Colors.GREEN}{metrics['metrics/mAP50-95(B)']:.4f}{Colors.ENDC}")
        
        print(f"\n{Colors.success('Model saved to:')} {Colors.CYAN}{trainer.model_save_path}{Colors.ENDC}")
        
    except Exception as e:
        print(f"\n{Colors.error('Training failed:')}")
        print(f"{Colors.FAIL}{str(e)}{Colors.ENDC}")
        raise
    
    print(f"\n{Colors.header('Training completed successfully!')}")


if __name__ == "__main__":
    main()
