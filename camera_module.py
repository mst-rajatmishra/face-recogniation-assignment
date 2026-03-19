"""
Camera Module

Responsibilities:
- Initialize webcam
- Capture frames continuously
- Provide frames to the main loop

Pipeline Step: Camera Initialization → Frame Capture → Frame Processing
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraModule:
    """Handles webcam initialization and continuous frame capture."""
    
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize the camera module.
        
        Args:
            camera_index: Index of the camera to use (default: 0 for built-in webcam)
            frame_width: Width of captured frames
            frame_height: Height of captured frames
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = None
        self.is_initialized = False
        
    def initialize_camera(self) -> bool:
        """
        Initialize the webcam for video capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
            
        Pipeline: Camera Device Access → Configuration → Ready State
        """
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False
            
            # Set frame dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Test capture to verify camera is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("Error: Could not capture test frame from camera")
                self.cap.release()
                return False
            
            self.is_initialized = True
            print(f"Camera initialized successfully (Resolution: {test_frame.shape[1]}x{test_frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the webcam.
        
        Returns:
            Captured frame as numpy array, or None if capture failed
            
        Pipeline: Frame Request → Camera Capture → Frame Return
        """
        if not self.is_initialized or self.cap is None:
            print("Error: Camera not initialized")
            return None
        
        try:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("Error: Could not capture frame")
                return None
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def get_camera_info(self) -> Tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            Tuple of (width, height)
        """
        if not self.is_initialized or self.cap is None:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def release_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.is_initialized = False
            print("Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is properly released."""
        self.release_camera()
