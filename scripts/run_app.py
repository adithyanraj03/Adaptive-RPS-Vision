#!/usr/bin/env python3
"""
Main application entry point for AdaptiveRPS-Vision.

This script launches the complete GUI application with both detection
and game modes.
"""

import sys
import argparse
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.gui.main_window import MainWindow


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AdaptiveRPS-Vision application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/best.pt",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cpu, cuda)"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("AdaptiveRPS-Vision")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    try:
        window = MainWindow(
            model_path=args.model_path,
            camera_id=args.camera_id,
            confidence_threshold=args.confidence,
            device=args.device
        )
        window.show()
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
