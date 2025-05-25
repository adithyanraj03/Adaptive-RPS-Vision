# Developer Guide

## Project Architecture

### Overview
AdaptiveRPS-Vision follows a modular architecture with clear separation of concerns:

```
adaptive-rps-vision/
├── config/          # Configuration management
├── src/             # Source code
│   └── adaptive_rps/
│       ├── core/    # Core detection and training
│       ├── ai/      # Game AI and RL components
│       ├── gui/     # User interface
│       ├── utils/   # Utility functions
│       └── data/    # Data management
├── scripts/         # Entry point scripts
├── tests/          # Unit tests
└── docs/           # Documentation
```

### Design Patterns

#### Configuration Pattern
All configurable parameters are centralized in the `config/` directory:
- `ModelConfig`: Training and inference parameters
- `DatasetConfig`: Dataset paths and structure

#### Factory Pattern
Model loading and initialization use factory patterns for flexibility:
```
def create_detector(model_path, device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return RPSDetector(model_path, device=device)
```

#### Observer Pattern
The GUI uses Qt's signal-slot mechanism for loose coupling:
```
class DetectorWidget(QWidget):
    detection_ready = pyqtSignal(dict)
    
    def on_detection(self, result):
        self.detection_ready.emit(result)
```

#### Strategy Pattern
The GameAI uses multiple strategies with dynamic weighting:
```
class GameAI:
    def __init__(self):
        self.strategies = {
            'counter_frequency': FrequencyStrategy(),
            'counter_pattern': PatternStrategy(),
            'counter_transition': TransitionStrategy()
        }
```

## Core Components

### Detection Engine

#### RPSDetector Class
The main detection class wraps YOLOv8 functionality:

```
class RPSDetector:
    def __init__(self, model_path, confidence_threshold=0.25, device='auto'):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_names = {0: 'rock', 1: 'paper', 2: 'scissors'}
    
    def detect(self, image):
        results = self.model(image, conf=self.confidence_threshold)
        return self._process_results(results)
```

**Key Features:**
- Configurable confidence thresholds
- Device-agnostic inference
- Batch processing support
- Real-time performance optimization

#### Training Pipeline

```
class RPSTrainer:
    def train(self, dataset_yaml, **kwargs):
        # Initialize model
        model = YOLO(self.config.model_name)
        
        # Configure training parameters
        train_args = self._prepare_training_args(kwargs)
        
        # Start training
        results = model.train(**train_args)
        
        # Post-process results
        return self._process_training_results(results)
```

### AI System

#### Adaptive Learning
The GameAI implements multiple learning strategies:

1. **Frequency Analysis**: Tracks move distribution
2. **Pattern Recognition**: Detects sequential patterns
3. **Transition Learning**: Models move-to-move transitions
4. **Performance Adaptation**: Adjusts strategy weights based on success

```
def prepare_next_move(self):
    # Collect predictions from different strategies
    predictions = {}
    for name, strategy in self.strategies.items():
        predictions[name] = strategy.predict(self.user_history)
    
    # Weight predictions and select best
    return self._weighted_selection(predictions)
```

#### Strategy Implementation

**Frequency Strategy:**
```
class FrequencyStrategy:
    def predict(self, history):
        if not history:
            return random.choice(['rock', 'paper', 'scissors'])
        
        # Count frequencies
        counts = Counter(history)
        most_frequent = counts.most_common(1)
        
        # Return counter-move
        return self.winning_moves[most_frequent]
```

**Pattern Strategy:**
```
class PatternStrategy:
    def predict(self, history):
        if len(history)  List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """Detect hand gestures in an image.
    
    Args:
        image: Input image array
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of (class_name, confidence, bbox) tuples
        
    Raises:
        DetectionError: If detection fails
    """
```

#### Import Organization
```
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from PyQt5.QtWidgets import QMainWindow

# Local imports
from adaptive_rps.core.detector import RPSDetector
from adaptive_rps.utils.colors import Colors
```

### Testing Strategy

#### Unit Tests
```
import pytest
from adaptive_rps.core.detector import RPSDetector

class TestRPSDetector:
    @pytest.fixture
    def detector(self):
        return RPSDetector("path/to/test/model.pt")
    
    def test_initialization(self, detector):
        assert detector.confidence_threshold == 0.25
        assert detector.device in ['cpu', 'cuda']
    
    def test_detection_output_format(self, detector):
        # Mock image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(image)
        
        assert isinstance(results, list)
        for result in results:
            assert len(result) == 3  # class, confidence, bbox
```

#### Integration Tests
```
def test_training_pipeline():
    """Test complete training pipeline"""
    config = ModelConfig(epochs=1, batch_size=2)
    trainer = RPSTrainer(config)
    
    # Use minimal test dataset
    results = trainer.train("test_dataset.yaml")
    
    assert results is not None
    assert os.path.exists(trainer.model_save_path)
```

#### GUI Tests
```
def test_main_window_initialization(qtbot):
    """Test main window creation"""
    window = MainWindow()
    qtbot.addWidget(window)
    
    assert window.stacked_widget.count() == 2
    assert window.detector_screen is not None
    assert window.game_screen is not None
```

### Performance Optimization

#### Profiling
```
import cProfile
import pstats

def profile_detection():
    """Profile detection performance"""
    detector = RPSDetector("model.pt")
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(100):
        detector.detect(image)
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(10)
```

#### Memory Optimization
```
import gc
import torch

def optimize_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
```

### Deployment Considerations

#### Docker Deployment
```
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/run_app.py"]
```

#### Model Optimization
```
def optimize_model_for_deployment(model_path):
    """Optimize model for production deployment"""
    model = YOLO(model_path)
    
    # Export to ONNX for faster inference
    model.export(format='onnx', dynamic=False, simplify=True)
    
    # Export to TensorRT for NVIDIA GPUs
    if torch.cuda.is_available():
        model.export(format='engine', device=0)
```

## Extending the System

### Adding New Gestures

1. **Update Configuration:**
```
# config/model_config.py
class ModelConfig:
    def __post_init__(self):
        self.class_names = {
            0: 'rock', 
            1: 'paper', 
            2: 'scissors',
            3: 'thumbs_up',  # New gesture
            4: 'peace'       # New gesture
        }
        self.num_classes = len(self.class_names)
```

2. **Update Game Logic:**
```
# src/adaptive_rps/ai/game_ai.py
class GameAI:
    def __init__(self):
        self.winning_moves = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock',
            'thumbs_up': 'thumbs_down',  # Define new relationships
            'peace': 'war'
        }
```

### Adding New AI Strategies

```
class CustomStrategy:
    def __init__(self):
        self.name = "custom_strategy"
        self.weight = 0.2
    
    def predict(self, history):
        """Implement custom prediction logic"""
        # Your custom strategy here
        return predicted_move
    
    def update(self, actual_move, predicted_move):
        """Update strategy based on performance"""
        success = self.evaluate_prediction(actual_move, predicted_move)
        self.adapt_parameters(success)

# Register new strategy
game_ai.strategies['custom'] = CustomStrategy()
```

### Custom Training Callbacks

```
class CustomCallback:
    def on_epoch_end(self, trainer, epoch, metrics):
        """Called at the end of each epoch"""
        if metrics['mAP50'] > 0.95:
            print("High accuracy achieved!")
            # Could trigger early stopping or model saving
    
    def on_batch_end(self, trainer, batch_idx, loss):
        """Called after each batch"""
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss:.4f}")

# Use custom callback
trainer = RPSTrainer(config)
trainer.add_callback(CustomCallback())
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and Clone:**
```
git fork https://github.com/original/adaptive-rps-vision.git
git clone https://github.com/yourusername/adaptive-rps-vision.git
```

2. **Create Feature Branch:**
```
git checkout -b feature/new-detection-algorithm
```

3. **Development Cycle:**
```
# Make changes
git add .
git commit -m "feat: add new detection algorithm"

# Run tests
pytest tests/

# Check code style
black src/
flake8 src/

# Push changes
git push origin feature/new-detection-algorithm
```

4. **Create Pull Request:**
- Provide clear description of changes
- Include test results
- Update documentation if necessary
- Ensure all CI checks pass

### Issue Reporting

When reporting issues, include:
- System information (OS, Python version, GPU)
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs and error messages
- Minimal code example if applicable

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Backward compatibility maintained
