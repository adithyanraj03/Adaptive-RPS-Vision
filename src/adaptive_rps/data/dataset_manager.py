"""
Dataset management utilities for AdaptiveRPS-Vision.

This module provides tools for dataset creation, validation,
augmentation, and management for training RPS detection models.
"""

import os
import cv2
import json
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging


class DatasetManager:
    """
    Comprehensive dataset management for RPS detection.
    
    Handles dataset creation, validation, augmentation, and splitting
    for training robust gesture detection models.
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize dataset manager.
        
        Args:
            dataset_root: Root directory for the dataset
        """
        self.dataset_root = Path(dataset_root)
        self.logger = logging.getLogger(__name__)
        
        # Dataset structure
        self.splits = ['train', 'valid', 'test']
        self.class_names = ['rock', 'paper', 'scissors']
        
        # Statistics
        self.dataset_stats = {
            'total_images': 0,
            'class_distribution': defaultdict(int),
            'split_distribution': defaultdict(lambda: defaultdict(int))
        }
        
        # Augmentation settings
        self.augmentation_config = {
            'brightness_range': (-0.2, 0.2),
            'contrast_range': (0.8, 1.2),
            'saturation_range': (0.8, 1.2),
            'rotation_range': (-15, 15),
            'scale_range': (0.9, 1.1),
            'flip_probability': 0.5,
            'noise_probability': 0.3
        }
    
    def create_dataset_structure(self):
        """Create the standard dataset directory structure."""
        self.logger.info(f"Creating dataset structure at {self.dataset_root}")
        
        for split in self.splits:
            for subfolder in ['images', 'labels']:
                folder_path = self.dataset_root / split / subfolder
                folder_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {folder_path}")
        
        self.logger.info("Dataset structure created successfully")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity and structure.
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating dataset...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check directory structure
        for split in self.splits:
            images_dir = self.dataset_root / split / 'images'
            labels_dir = self.dataset_root / split / 'labels'
            
            if not images_dir.exists():
                validation_results['errors'].append(f"Missing images directory: {images_dir}")
                validation_results['is_valid'] = False
            
            if not labels_dir.exists():
                validation_results['errors'].append(f"Missing labels directory: {labels_dir}")
                validation_results['is_valid'] = False
        
        # Validate image-label pairs
        for split in self.splits:
            images_dir = self.dataset_root / split / 'images'
            labels_dir = self.dataset_root / split / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            # Get image and label files
            image_files = set(f.stem for f in images_dir.glob('*.jpg') | images_dir.glob('*.png'))
            label_files = set(f.stem for f in labels_dir.glob('*.txt'))
            
            # Check for missing labels
            missing_labels = image_files - label_files
            if missing_labels:
                validation_results['warnings'].append(
                    f"Missing labels in {split}: {len(missing_labels)} files"
                )
            
            # Check for orphaned labels
            orphaned_labels = label_files - image_files
            if orphaned_labels:
                validation_results['warnings'].append(
                    f"Orphaned labels in {split}: {len(orphaned_labels)} files"
                )
            
            # Validate label format
            invalid_labels = self._validate_label_files(labels_dir)
            if invalid_labels:
                validation_results['errors'].extend(invalid_labels)
                validation_results['is_valid'] = False
        
        # Calculate statistics
        validation_results['statistics'] = self._calculate_dataset_statistics()
        
        self.logger.info(f"Dataset validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def _validate_label_files(self, labels_dir: Path) -> List[str]:
        """
        Validate YOLO format label files.
        
        Args:
            labels_dir: Directory containing label files
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(
                            f"Invalid format in {label_file}:{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Validate class ID
                        if class_id < 0 or class_id >= len(self.class_names):
                            errors.append(
                                f"Invalid class ID in {label_file}:{line_num} - {class_id}"
                            )
                        
                        # Validate coordinates (should be normalized 0-1)
                        for coord_name, coord_val in [('x_center', x_center), ('y_center', y_center), 
                                                     ('width', width), ('height', height)]:
                            if coord_val < 0 or coord_val > 1:
                                errors.append(
                                    f"Invalid {coord_name} in {label_file}:{line_num} - {coord_val}"
                                )
                    
                    except ValueError as e:
                        errors.append(f"Invalid number format in {label_file}:{line_num} - {e}")
            
            except Exception as e:
                errors.append(f"Error reading {label_file}: {e}")
        
        return errors
    
    def _calculate_dataset_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int),
            'split_distribution': defaultdict(lambda: {'images': 0, 'annotations': 0}),
            'image_dimensions': [],
            'annotations_per_image': []
        }
        
        for split in self.splits:
            images_dir = self.dataset_root / split / 'images'
            labels_dir = self.dataset_root / split / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            # Count images
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            stats['split_distribution'][split]['images'] = len(image_files)
            stats['total_images'] += len(image_files)
            
            # Analyze labels
            for image_file in image_files:
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if label_file.exists():
                    # Get image dimensions
                    try:
                        img = cv2.imread(str(image_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            stats['image_dimensions'].append((w, h))
                    except Exception:
                        pass
                    
                    # Count annotations
                    try:
                        with open(label_file, 'r') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                        stats['annotations_per_image'].append(len(lines))
                        stats['split_distribution'][split]['annotations'] += len(lines)
                        stats['total_annotations'] += len(lines)
                        
                        # Count classes
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 1:
                                try:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < len(self.class_names):
                                        class_name = self.class_names[class_id]
                                        stats['class_distribution'][class_name] += 1
                                except ValueError:
                                    pass
                    
                    except Exception:
                        pass
        
        # Calculate additional statistics
        if stats['image_dimensions']:
            widths, heights = zip(*stats['image_dimensions'])
            stats['image_size_stats'] = {
                'mean_width': np.mean(widths),
                'mean_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights)
            }
        
        if stats['annotations_per_image']:
            stats['annotation_stats'] = {
                'mean_annotations_per_image': np.mean(stats['annotations_per_image']),
                'min_annotations_per_image': min(stats['annotations_per_image']),
                'max_annotations_per_image': max(stats['annotations_per_image'])
            }
        
        return stats
    
    def split_dataset(
        self,
        source_dir: str,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Split dataset into train/validation/test sets.
        
        Args:
            source_dir: Source directory containing images and labels
            train_ratio: Ratio for training set
            valid_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducible splits
        """
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.logger.info(f"Splitting dataset with ratios - Train: {train_ratio}, Valid: {valid_ratio}, Test: {test_ratio}")
        
        source_path = Path(source_dir)
        images_dir = source_path / 'images'
        labels_dir = source_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            raise ValueError(f"Source directory must contain 'images' and 'labels' subdirectories")
        
        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        # Filter files that have corresponding labels
        valid_files = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_files.append(img_file.stem)
        
        self.logger.info(f"Found {len(valid_files)} valid image-label pairs")
        
        # Shuffle files
        random.seed(random_seed)
        random.shuffle(valid_files)
        
        # Calculate split indices
        n_files = len(valid_files)
        train_end = int(n_files * train_ratio)
        valid_end = train_end + int(n_files * valid_ratio)
        
        splits_data = {
            'train': valid_files[:train_end],
            'valid': valid_files[train_end:valid_end],
            'test': valid_files[valid_end:]
        }
        
        # Create dataset structure
        self.create_dataset_structure()
        
        # Copy files to respective splits
        for split_name, file_list in splits_data.items():
            self.logger.info(f"Copying {len(file_list)} files to {split_name} set")
            
            split_images_dir = self.dataset_root / split_name / 'images'
            split_labels_dir = self.dataset_root / split_name / 'labels'
            
            for file_stem in file_list:
                # Copy image
                for ext in ['.jpg', '.png']:
                    src_img = images_
