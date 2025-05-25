# Usage Guide

## Quick Start

### 1. Training a Model
```
python scripts/train.py --dataset_path ./datasets/your_dataset --epochs 100
```

### 2. Running the Application
```
python scripts/run_app.py --model_path ./models/best.pt
```

## Detailed Usage

### Training Process

#### Preparing Your Dataset
Your dataset should follow this structure:
```
datasets/
└── rps_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

#### Training Command Options
```
python scripts/train.py \
    --dataset_path ./datasets/rps_dataset \
    --epochs 100 \
    --batch_size 16 \
    --image_size 640 \
    --lr 0.001 \
    --device auto \
    --patience 15
```

**Parameters:**
- `--dataset_path`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--image_size`: Input image size (default: 640)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use ('auto', 'cpu', 'cuda', or device ID)
- `--patience`: Early stopping patience (default: 15)

#### Training Output
Training creates:
- `runs/train_TIMESTAMP/weights/best.pt` - Best model weights
- `runs/train_TIMESTAMP/weights/last.pt` - Latest model weights
- Training plots and metrics in the run directory

### Using the Application

#### Detector Mode
1. **Real-time Detection**: Shows live camera feed with gesture detection
2. **Statistics**: Displays detection counts and performance metrics
3. **History**: Shows recent detections with timestamps
4. **Controls**: Save screenshots, generate reports, toggle CPU/GPU mode

#### Game Mode
1. **Setup**: 10-round game against adaptive AI
2. **Gameplay**: 
   - 5-second countdown per round
   - Show your gesture when countdown reaches 0
   - AI learns from your patterns
3. **Results**: Score tracking and round history

### Application Interface

#### Main Controls
- **Start Game Mode**: Switch to competitive gameplay
- **Save Screenshot**: Capture current frame with detections
- **Save Report**: Generate detailed session report
- **Toggle CPU Mode**: Switch between GPU and CPU inference

#### Game Controls
- **Next Round**: Proceed to next round (enabled after AI analysis)
- **Restart Game**: Start a new 10-round game
- **Exit Game Mode**: Return to detector mode

### Command Line Options

#### Training Script
```
python scripts/train.py --help
```

#### Application Runner
```
python scripts/run_app.py --help
```

### Performance Optimization

#### GPU Optimization
- Use batch size 16-32 for optimal GPU utilization
- Enable mixed precision training for faster convergence
- Monitor GPU memory usage during training

#### CPU Optimization
- Reduce batch size to 8 or lower
- Use smaller image size (320 or 416)
- Limit number of CPU threads

### Data Management

#### Dataset Preparation
1. Collect diverse hand gesture images
2. Annotate using tools like LabelImg or Roboflow
3. Split into train/validation/test sets (70/20/10)
4. Ensure class balance across splits

#### Model Management
- Models are saved in `models/` directory
- Keep backups of best-performing models
- Version control model configurations

## Tips and Best Practices

### Training Tips
1. **Data Quality**: Use high-quality, diverse images
2. **Augmentation**: Enable data augmentation for better generalization
3. **Validation**: Monitor validation metrics to prevent overfitting
4. **Early Stopping**: Use patience parameter to avoid overtraining

### Deployment Tips
1. **Model Size**: Use YOLOv8n for real-time inference
2. **Hardware**: GPU recommended for smooth performance
3. **Camera**: Ensure good lighting and camera positioning
4. **Calibration**: Test with different hand sizes and skin tones

### Game Strategy
The AI learns from your playing patterns:
- **Frequency Analysis**: Counters your most-played moves
- **Pattern Recognition**: Detects sequences in your moves
- **Transition Learning**: Learns what move follows which move
- **Adaptive Weighting**: Adjusts strategy based on success rate

## Troubleshooting

### Performance Issues
- Reduce image size for faster inference
- Use CPU mode if GPU causes issues
- Close other applications to free up resources

### Detection Issues
- Ensure good lighting conditions
- Position hand clearly in camera view
- Avoid cluttered backgrounds
- Check model confidence threshold

### Game Issues
- Make gestures clearly during countdown
- Keep hand in camera view when timer reaches 0
- Ensure stable camera positioning
