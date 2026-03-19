"""
Utils Module

Responsibilities:
- Draw bounding boxes
- Draw names above boxes
- Provide helper functions for rendering

Pipeline Step: Recognition Results → Visual Rendering → Display Output
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceRenderer:
    """Handles drawing of bounding boxes and names on video frames."""
    
    def __init__(self, box_thickness: int = 2, font_scale: float = 0.6, 
                 font_thickness: int = 2):
        """
        Initialize the face renderer.
        
        Args:
            box_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text
            font_thickness: Thickness of text
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors in BGR format (OpenCV uses BGR)
        self.colors = {
            'recognized': (0, 255, 0),    # Green
            'unknown': (0, 0, 255),       # Red
            'text': (255, 255, 255),      # White
            'text_bg': (0, 0, 0)          # Black
        }
    
    def draw_bounding_box(self, frame: np.ndarray, 
                          face_location: Tuple[int, int, int, int],
                          is_recognized: bool = True) -> np.ndarray:
        """
        Draw a bounding box around a detected face.
        
        Args:
            frame: Input frame
            face_location: Face bounding box as (top, right, bottom, left)
            is_recognized: True for recognized faces (green), False for unknown (red)
            
        Returns:
            Frame with bounding box drawn
            
        Pipeline: Face Location → Box Drawing → Frame Return
        """
        top, right, bottom, left = face_location
        
        # Choose color based on recognition status
        color = self.colors['recognized'] if is_recognized else self.colors['unknown']
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, self.box_thickness)
        
        return frame
    
    def draw_name(self, frame: np.ndarray, 
                  face_location: Tuple[int, int, int, int],
                  name: str, is_recognized: bool = True) -> np.ndarray:
        """
        Draw name above the bounding box.
        
        Args:
            frame: Input frame
            face_location: Face bounding box as (top, right, bottom, left)
            name: Name to display
            is_recognized: True for recognized faces, False for unknown
            
        Returns:
            Frame with name drawn
        """
        top, right, bottom, left = face_location
        
        # Choose colors
        box_color = self.colors['recognized'] if is_recognized else self.colors['unknown']
        text_color = self.colors['text']
        bg_color = self.colors['text_bg']
        
        # Get text size
        text = name.upper() if is_recognized else "UNKNOWN"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # Calculate text position (above the bounding box)
        text_x = left
        text_y = top - 10  # 10 pixels above the box
        
        # Draw background rectangle for better text visibility
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            bg_color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness
        )
        
        # Draw a small line connecting text to box
        cv2.line(
            frame,
            (left + text_width // 2, text_y + baseline),
            (left + text_width // 2, top),
            box_color,
            1
        )
        
        return frame
    
    def draw_face_info(self, frame: np.ndarray,
                      face_location: Tuple[int, int, int, int],
                      name: str, confidence: Optional[float] = None) -> np.ndarray:
        """
        Draw complete face information (bounding box + name + confidence).
        
        Args:
            frame: Input frame
            face_location: Face bounding box
            name: Recognized name
            confidence: Optional confidence score (0.0 to 1.0)
            
        Returns:
            Frame with complete face information drawn
        """
        is_recognized = name != "Unknown"
        
        # Draw bounding box
        frame = self.draw_bounding_box(frame, face_location, is_recognized)
        
        # Prepare display text
        display_text = name
        if confidence is not None and is_recognized:
            display_text += f" ({confidence:.2f})"
        
        # Draw name
        frame = self.draw_name(frame, face_location, display_text, is_recognized)
        
        return frame
    
    def draw_multiple_faces(self, frame: np.ndarray,
                           face_locations: List[Tuple[int, int, int, int]],
                           names: List[str],
                           confidences: Optional[List[float]] = None) -> np.ndarray:
        """
        Draw multiple faces on the same frame.
        
        Args:
            frame: Input frame
            face_locations: List of face bounding boxes
            names: List of corresponding names
            confidences: Optional list of confidence scores
            
        Returns:
            Frame with all faces drawn
        """
        if len(face_locations) != len(names):
            print("Warning: Number of face locations doesn't match number of names")
            return frame
        
        for i, (face_location, name) in enumerate(zip(face_locations, names)):
            confidence = confidences[i] if confidences and i < len(confidences) else None
            frame = self.draw_face_info(frame, face_location, name, confidence)
        
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Input frame
            fps: Frames per second value
            
        Returns:
            Frame with FPS drawn
        """
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw in top-right corner
        cv2.putText(
            frame,
            fps_text,
            (frame.shape[1] - 150, 30),
            self.font,
            self.font_scale,
            self.colors['text'],
            self.font_thickness
        )
        
        return frame
    
    def draw_status(self, frame: np.ndarray, status_text: str) -> np.ndarray:
        """
        Draw status text at the bottom of the frame.
        
        Args:
            frame: Input frame
            status_text: Status message to display
            
        Returns:
            Frame with status drawn
        """
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            status_text, self.font, self.font_scale, self.font_thickness
        )
        
        # Position at bottom center
        text_x = (frame.shape[1] - text_width) // 2
        text_y = frame.shape[0] - 20
        
        # Draw background
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + 10),
            self.colors['text_bg'],
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            status_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.colors['text'],
            self.font_thickness
        )
        
        return frame
    
    def resize_frame(self, frame: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """
        Resize frame by scale factor.
        
        Args:
            frame: Input frame
            scale_factor: Scale factor (1.0 = original size)
            
        Returns:
            Resized frame
        """
        if scale_factor == 1.0:
            return frame
        
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def add_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """
        Add timestamp to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with timestamp added
        """
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cv2.putText(
            frame,
            timestamp,
            (10, frame.shape[0] - 10),
            self.font,
            0.5,
            self.colors['text'],
            1
        )
        
        return frame
