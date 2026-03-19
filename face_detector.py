"""
Face Detector Module

Responsibilities:
- Detect face locations in frames using face_recognition.face_locations
- Provide face bounding boxes for recognition and rendering

Pipeline Step: Frame Input → Face Detection → Location Output
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """Detects face locations in video frames for recognition pipeline."""
    
    def __init__(self, model: str = "hog", upscale_factor: int = 1):
        """
        Initialize the face detector.
        
        Args:
            model: Face detection model ('hog' for CPU, 'cnn' for GPU)
            upscale_factor: Factor to upscale image for better detection
        """
        self.model = model
        self.upscale_factor = upscale_factor
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face locations in the given frame.
        
        Args:
            frame: Input frame in BGR format (from OpenCV)
            
        Returns:
            List of face bounding boxes as (top, right, bottom, left) tuples
            
        Pipeline: Frame Input → Color Conversion → Face Detection → Location Return
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            # Returns list of tuples: (top, right, bottom, left)
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model=self.model,
                number_of_times_to_upsample=self.upscale_factor
            )
            
            return face_locations
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def detect_faces_with_confidence(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect face locations with confidence scores (if available).
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            List of tuples containing (face_location, confidence_score)
            Note: face_recognition library doesn't provide confidence scores
                  This method is for future compatibility
        """
        face_locations = self.detect_faces(frame)
        # face_recognition doesn't provide confidence scores, so we return 1.0 for all
        return [(location, 1.0) for location in face_locations]
    
    def get_largest_face(self, face_locations: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest face from detected faces based on area.
        
        Args:
            face_locations: List of face bounding boxes
            
        Returns:
            Bounding box of the largest face, or None if no faces detected
        """
        if not face_locations:
            return None
        
        # Calculate area for each face and return the largest
        largest_face = max(face_locations, key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
        return largest_face
    
    def filter_faces_by_size(self, face_locations: List[Tuple[int, int, int, int]], 
                           min_size: int = 50) -> List[Tuple[int, int, int, int]]:
        """
        Filter out faces that are too small.
        
        Args:
            face_locations: List of face bounding boxes
            min_size: Minimum width/height in pixels
            
        Returns:
            Filtered list of face locations
        """
        filtered_faces = []
        
        for (top, right, bottom, left) in face_locations:
            width = right - left
            height = bottom - top
            
            if width >= min_size and height >= min_size:
                filtered_faces.append((top, right, bottom, left))
        
        return filtered_faces
    
    def get_face_center(self, face_location: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get the center point of a face bounding box.
        
        Args:
            face_location: Face bounding box as (top, right, bottom, left)
            
        Returns:
            Center point as (x, y)
        """
        top, right, bottom, left = face_location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        return (center_x, center_y)
    
    def get_face_roi(self, frame: np.ndarray, face_location: Tuple[int, int, int, int], 
                    padding: int = 20) -> np.ndarray:
        """
        Extract the Region of Interest (ROI) for a face.
        
        Args:
            frame: Input frame
            face_location: Face bounding box
            padding: Extra padding around the face
            
        Returns:
            Face ROI as numpy array
        """
        top, right, bottom, left = face_location
        
        # Add padding
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        # Extract ROI
        face_roi = frame[top:bottom, left:right]
        return face_roi
