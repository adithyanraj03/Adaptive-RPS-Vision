# Installation Guide

## Prerequisites

### System Requirements
- Windows 10/11 (64-bit)
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)
- Webcam for real-time detection

### Hardware Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **GPU**: NVIDIA GTX 1060 or better (for GPU acceleration)
- **Storage**: 5GB free space
- **Camera**: USB webcam with 720p resolution minimum

## Installation Steps

### 1. Clone the Repository
```
git clone https://github.com/yourusername/adaptive-rps-vision.git
cd adaptive-rps-vision
```

### 2. Create Virtual Environment
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Install Additional GPU Support (Optional)
For NVIDIA GPU acceleration:
```
# Install CUDA-enabled PyTorch (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation
```
python scripts/verify_installation.py
```

## Troubleshooting

### Common Issues

#### CUDA Issues
If you encounter CUDA-related errors:
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Use CPU mode if GPU issues persist

#### Camera Issues
- Ensure camera permissions are granted
- Try different camera IDs (0, 1, 2...)
- Check camera compatibility with OpenCV

#### Model Loading Issues
- Verify model file exists in `models/` directory
- Check file permissions
- Re-download model if corrupted

### Environment Variables
Set these environment variables for optimal performance:
```
# Windows
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=4

# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```

## Next Steps
After installation, proceed to the [Usage Guide](usage.md) to learn how to use the system.
