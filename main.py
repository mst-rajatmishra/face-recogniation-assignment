"""
Main Application - Student Face Recognition System

Responsibilities:
- Initialize all modules
- Main application loop
- Process frames continuously
- Coordinate face detection and recognition
- Display live camera preview with results

Pipeline: Camera → Frame Capture → Face Detection → Face Encoding → Face Matching → Bounding Box Rendering → Display
"""

import cv2
import time
import numpy as np
from typing import List, Tuple

# Import custom modules
from dataset_loader import DatasetLoader
from camera_module import CameraModule
from face_detector import FaceDetector
from recognizer import FaceRecognizer
from utils import FaceRenderer


class FaceRecognitionApp:
    """Main application class for student face recognition system."""
    
    def __init__(self, dataset_path: str = "dataset", recognition_interval: int = 3):
        """
        Initialize the face recognition application.
        
        Args:
            dataset_path: Path to the dataset folder
            recognition_interval: Perform recognition every N frames (optimization)
        """
        self.dataset_path = dataset_path
        self.recognition_interval = recognition_interval
        
        # Initialize modules
        self.dataset_loader = DatasetLoader(dataset_path)
        self.camera = CameraModule()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.renderer = FaceRenderer()
        
        # Application state
        self.is_running = False
        self.frame_count = 0
        self.last_recognition_frame = -recognition_interval
        self.last_face_locations = []
        self.last_recognized_names = []
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def initialize(self) -> bool:
        """
        Initialize all components of the application.
        
        Returns:
            True if initialization successful, False otherwise
            
        Pipeline: Dataset Loading → Camera Initialization → Ready State
        """
        print("=== Student Face Recognition System ===")
        print("Initializing application...")
        
        # Load dataset
        known_encodings, known_names = self.dataset_loader.load_dataset()
        if not known_encodings:
            print("Error: No face encodings loaded from dataset!")
            print("Please add student images to the 'dataset' folder.")
            return False
        
        # Initialize recognizer with known faces
        self.face_recognizer.load_known_faces(known_encodings, known_names)
        
        # Initialize camera
        if not self.camera.initialize_camera():
            print("Error: Failed to initialize camera!")
            return False
        
        print("Application initialized successfully!")
        print(f"Loaded {len(known_names)} students for recognition")
        print(f"Recognition will run every {self.recognition_interval} frames")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the recognition pipeline.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with bounding boxes and names
            
        Pipeline: Frame Input → Face Detection → Recognition (if needed) → Rendering
        """
        self.frame_count += 1
        
        # Detect faces in current frame
        face_locations = self.face_detector.detect_faces(frame)
        
        if not face_locations:
            # No faces detected, return original frame
            return self.renderer.draw_status(frame, "No faces detected")
        
        # Perform recognition only on specified intervals for optimization
        if self.frame_count - self.last_recognition_frame >= self.recognition_interval:
            # Time to perform recognition
            recognized_names = self.face_recognizer.recognize_faces(frame, face_locations)
            
            # Store results for reuse in skipped frames
            self.last_face_locations = face_locations
            self.last_recognized_names = recognized_names
            self.last_recognition_frame = self.frame_count
            
            print(f"Frame {self.frame_count}: Detected {len(face_locations)} faces, "
                  f"recognized: {recognized_names}")
        else:
            # Reuse previous recognition results
            recognized_names = self.last_recognized_names
            # Use current face locations but previous names
            # This handles slight movement between recognition frames
        
        # Draw bounding boxes and names
        processed_frame = self.renderer.draw_multiple_faces(
            frame, 
            face_locations, 
            recognized_names
        )
        
        # Draw FPS
        processed_frame = self.renderer.draw_fps(processed_frame, self.current_fps)
        
        # Draw status
        status_text = f"Faces: {len(face_locations)} | Recognition every {self.recognition_interval} frames"
        processed_frame = self.renderer.draw_status(processed_frame, status_text)
        
        return processed_frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        # Update FPS every second
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """
        Main application loop.
        
        Pipeline: Continuous Frame Capture → Processing → Display
        """
        if not self.initialize():
            print("Failed to initialize application. Exiting...")
            return
        
        self.is_running = True
        print("\nStarting face recognition...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while self.is_running:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    print("Error: Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Student Face Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting application...")
                    self.is_running = False
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"face_recognition_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame as {filename}")
                
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.is_running = False
        self.camera.release_camera()
        cv2.destroyAllWindows()
        print("Application closed")
    
    def get_system_info(self) -> dict:
        """
        Get system information and statistics.
        
        Returns:
            Dictionary with system information
        """
        camera_info = self.camera.get_camera_info()
        recognizer_stats = self.face_recognizer.get_recognition_stats()
        
        return {
            "camera_resolution": camera_info,
            "recognizer_stats": recognizer_stats,
            "recognition_interval": self.recognition_interval,
            "dataset_path": self.dataset_path
        }


def main():
    """Main entry point for the application."""
    print("Student Face Recognition System")
    print("=" * 40)
    
    # Create and run application
    app = FaceRecognitionApp(
        dataset_path="dataset",
        recognition_interval=3  # Perform recognition every 3 frames for optimization
    )
    
    # Print system info
    info = app.get_system_info()
    print(f"Camera Resolution: {info['camera_resolution']}")
    print(f"Dataset Path: {info['dataset_path']}")
    print(f"Recognition Interval: {info['recognition_interval']} frames")
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
