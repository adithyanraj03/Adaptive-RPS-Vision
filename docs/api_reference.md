# API Reference

## Core Classes

### RPSDetector
Main detection class for rock-paper-scissors gesture recognition.

```
from adaptive_rps.core.detector import RPSDetector

detector = RPSDetector(model_path="./models/best.pt")
results = detector.detect(image)
```

#### Methods

##### `__init__(model_path, confidence_threshold=0.25, device='auto')`
Initialize the detector.

**Parameters:**
- `model_path` (str): Path to the trained model file
- `confidence_threshold` (float): Minimum confidence for detections (0.0-1.0)
- `device` (str): Device to run inference on ('auto', 'cpu', 'cuda')

##### `detect(image, return_confidence=True)`
Detect gestures in an image.

**Parameters:**
- `image` (np.ndarray): Input image array
- `return_confidence` (bool): Whether to return confidence scores

**Returns:**
- `list`: List of (class_name, confidence, bbox) tuples

##### `set_confidence_threshold(threshold)`
Update the confidence threshold.

**Parameters:**
- `threshold` (float): New confidence threshold (0.0-1.0)

### RPSTrainer
Training class for model training and evaluation.

```
from adaptive_rps.core.trainer import RPSTrainer
from config.model_config import ModelConfig

config = ModelConfig(epochs=100, batch_size=16)
trainer = RPSTrainer(model_config=config)
results = trainer.train(dataset_yaml="dataset.yaml")
```

#### Methods

##### `__init__(model_config)`
Initialize the trainer.

**Parameters:**
- `model_config` (ModelConfig): Configuration object for training

##### `train(dataset_yaml, output_dir="./runs", run_name=None, **kwargs)`
Train the model.

**Parameters:**
- `dataset_yaml` (str): Path to dataset YAML configuration
- `output_dir` (str): Output directory for results
- `run_name` (str): Name for this training run
- `**kwargs`: Additional training parameters

**Returns:**
- Training results object

##### `export_model(formats=['onnx', 'torchscript'])`
Export model to different formats.

**Parameters:**
- `formats` (list): List of export formats

**Returns:**
- `dict`: Dictionary of export paths

### GameAI
Adaptive AI opponent that learns from player patterns.

```
from adaptive_rps.ai.game_ai import GameAI

ai = GameAI()
ai.update("rock")  # Update with user's move
ai_move = ai.prepare_next_move()  # Get AI's next move
```

#### Methods

##### `__init__()`
Initialize the game AI.

##### `update(user_move)`
Update AI knowledge with user's move.

**Parameters:**
- `user_move` (str): User's move ('rock', 'paper', 'scissors')

##### `prepare_next_move()`
Prepare AI's next move based on learned patterns.

**Returns:**
- `str`: AI's next move ('rock', 'paper', 'scissors')

##### `get_next_move()`
Get the already-prepared next move.

**Returns:**
- `str`: AI's next move

##### `get_statistics()`
Get AI learning statistics.

**Returns:**
- `dict`: Statistics about AI's learning and performance

##### `reset()`
Reset AI for a new game session.

## Configuration Classes

### ModelConfig
Configuration for model training and inference.

```
from config.model_config import ModelConfig

config = ModelConfig(
    model_name="yolov8n.pt",
    epochs=100,
    batch_size=16,
    learning_rate=0.001
)
```

#### Attributes

**Model Settings:**
- `model_name` (str): Base model to use
- `num_classes` (int): Number of classes (default: 3)
- `class_names` (dict): Mapping of class IDs to names

**Training Settings:**
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `image_size` (int): Input image size
- `device` (str): Training device

**Optimization Settings:**
- `optimizer` (str): Optimizer type
- `learning_rate` (float): Initial learning rate
- `momentum` (float): Momentum factor
- `weight_decay` (float): Weight decay factor

### DatasetConfig
Configuration for dataset management.

```
from config.dataset_config import DatasetConfig

config = DatasetConfig(
    dataset_root="./datasets/rps_dataset",
    train_images="train/images",
    val_images="valid/images"
)
```

#### Methods

##### `create_yaml_config(output_path=None)`
Create YAML configuration file for YOLO training.

**Parameters:**
- `output_path` (str): Output path for YAML file

**Returns:**
- `str`: Path to created YAML file

## Utility Classes

### Colors
Terminal color utilities for better output formatting.

```
from adaptive_rps.utils.colors import Colors

print(Colors.success("Training completed!"))
print(Colors.error("Error occurred"))
print(Colors.info("Information message"))
```

#### Class Methods

##### `success(text)`
Return green colored text for success messages.

##### `error(text)`
Return red colored text for error messages.

##### `info(text)`
Return cyan colored text for info messages.

##### `warning(text)`
Return yellow colored text for warning messages.

### CameraManager
Camera management utilities.

```
from adaptive_rps.utils.camera import CameraManager

with CameraManager(camera_id=0) as camera:
    ret, frame = camera.read_frame()
```

#### Methods

##### `__init__(camera_id=0, width=640, height=480)`
Initialize camera manager.

##### `open_camera()`
Open the camera device.

**Returns:**
- `bool`: True if successful, False otherwise

##### `read_frame()`
Read a frame from the camera.

**Returns:**
- `tuple`: (success, frame) where success is bool and frame is np.ndarray

##### `release()`
Release the camera resource.

### ImageProcessor
Image processing utilities.

```
from adaptive_rps.utils.image_utils import ImageProcessor

# Resize image maintaining aspect ratio
resized = ImageProcessor.resize_maintain_aspect_ratio(image, (640, 640))

# Draw detection box
image_with_box = ImageProcessor.draw_detection_box(
    image, (x1, y1, x2, y2), "rock", 0.95
)
```

#### Static Methods

##### `resize_maintain_aspect_ratio(image, target_size, fill_color=(114, 114, 114))`
Resize image maintaining aspect ratio with padding.

##### `draw_detection_box(image, box, class_name, confidence, color=(0, 255, 0), thickness=2)`
Draw detection box with label on image.

##### `get_class_color(class_name)`
Get BGR color for a class name.

##### `create_move_visualization(move, size=(360, 480), background_color=(45, 45, 45))`
Create visual representation of a move.

##### `save_image(image, filepath)`
Save image to file.

## GUI Classes

### MainWindow
Main application window with detector and game modes.

```
from adaptive_rps.gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
window = MainWindow(
    model_path="./models/best.pt",
    camera_id=0,
    confidence_threshold=0.25
)
window.show()
```

## Data Types

### Detection Result
```
{
    'class_name': str,      # Detected class ('rock', 'paper', 'scissors')
    'confidence': float,    # Detection confidence (0.0-1.0)
    'bbox': tuple,         # Bounding box (x1, y1, x2, y2)
    'timestamp': datetime  # Detection timestamp
}
```

### Game Statistics
```
{
    'total_games': int,
    'user_move_frequency': dict,
    'strategy_weights': dict,
    'strategy_performance': dict,
    'recent_patterns': dict,
    'transition_matrix': dict
}
```

### Training Results
```
{
    'final_metrics': dict,
    'training_time': float,
    'best_epoch': int,
    'model_path': str,
    'export_paths': dict
}
```

## Error Handling

### Common Exceptions

#### `ModelLoadError`
Raised when model cannot be loaded.

#### `CameraError`
Raised when camera cannot be accessed.

#### `DetectionError`
Raised when detection fails.

#### `TrainingError`
Raised when training encounters an error.

### Example Error Handling
```
try:
    detector = RPSDetector("./models/best.pt")
    results = detector.detect(image)
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except DetectionError as e:
    print(f"Detection failed: {e}")
```

## Performance Considerations

### Optimization Tips
1. **Batch Processing**: Process multiple images together when possible
2. **Model Size**: Use smaller models (YOLOv8n) for real-time applications
3. **Image Size**: Reduce input size for faster inference
4. **Device Management**: Use GPU when available, fallback to CPU gracefully

### Memory Management
- Release camera resources when done
- Clear detection history periodically
- Use context managers for resource management

### Threading Considerations
- GUI updates must happen on main thread
- Use Qt signals for cross-thread communication
- Consider using QTimer for periodic updates
