#!/usr/bin/env python3
"""
Basic detection example for AdaptiveRPS-Vision.

This script demonstrates how to use the RPSDetector for real-time
gesture detection with a webcam feed.
"""

import cv2
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rps.core.detector import RPSDetector
from adaptive_rps.utils.colors import Colors


def main():
    """Run basic detection example."""
    print(f"{Colors.header('AdaptiveRPS-Vision Basic Detection Example')}")
    
    # Initialize detector
    model_path = "./models/best.pt"
    try:
        detector = RPSDetector(model_path, confidence_threshold=0.25)
        print(f"{Colors.success('Model loaded successfully')}")
    except Exception as e:
        print(f"{Colors.error(f'Failed to load model: {e}')}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{Colors.error('Could not open camera')}")
        return
    
    print(f"{Colors.info('Press q to quit, s to save screenshot')}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for natural view
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            detections = detector.detect(frame)
            
            # Draw detections
            for class_name, confidence, bbox in detections:
                x1, y1, x2, y2 = bbox
                color = detector.get_class_color(class_name)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"Detected: {class_name} (confidence: {confidence:.2f})")
            
            # Display frame
            cv2.imshow('AdaptiveRPS-Vision Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_screenshot_{frame_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"{Colors.success(f'Screenshot saved: {filename}')}")
                frame_count += 1
    
    except KeyboardInterrupt:
        print(f"\n{Colors.info('Interrupted by user')}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{Colors.success('Detection example completed')}")


if __name__ == "__main__":
    main()
