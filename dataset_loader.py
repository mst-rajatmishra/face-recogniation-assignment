"""
Dataset Loader Module

Responsibilities:
- Load student images from the dataset folder
- Extract face encodings using face_recognition
- Store names and encodings in memory

Pipeline Step: Dataset Loading → Face Encoding Generation
"""

import os
import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict


class DatasetLoader:
    """Loads student images and extracts face encodings for recognition."""
    
    def __init__(self, dataset_path: str = "dataset"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset folder containing student images
        """
        self.dataset_path = dataset_path
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_dataset(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all student images from dataset folder and extract face encodings.
        
        Returns:
            Tuple of (face_encodings, face_names)
            
        Pipeline: Image Loading → Face Detection → Face Encoding → Storage
        """
        print("Loading dataset...")
        
        # Check if dataset folder exists
        if not os.path.exists(self.dataset_path):
            print(f"Dataset folder '{self.dataset_path}' not found!")
            return [], []
        
        # Get all image files from dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(self.dataset_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"No image files found in '{self.dataset_path}' folder!")
            return [], []
        
        print(f"Found {len(image_files)} images in dataset")
        
        # Process each image
        for image_file in image_files:
            # Extract student name from filename (remove extension)
            student_name = os.path.splitext(image_file)[0]
            
            # Load image
            image_path = os.path.join(self.dataset_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not load image {image_file}")
                continue
                
            # Convert BGR to RGB for face_recognition library
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations in the image
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                print(f"Warning: No face detected in {image_file}")
                continue
                
            if len(face_locations) > 1:
                print(f"Warning: Multiple faces detected in {image_file}, using first face")
            
            # Extract face encoding for the first face found
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            
            # Store encoding and name
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(student_name)
            
            print(f"Loaded: {student_name}")
        
        print(f"Successfully loaded {len(self.known_face_encodings)} face encodings")
        return self.known_face_encodings, self.known_face_names
    
    def get_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get loaded face encodings and names.
        
        Returns:
            Tuple of (face_encodings, face_names)
        """
        return self.known_face_encodings, self.known_face_names
    
    def add_student(self, image_path: str, student_name: str) -> bool:
        """
        Add a new student to the dataset.
        
        Args:
            image_path: Path to the student's image
            student_name: Name of the student
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return False
                
            # Extract encoding and store
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(student_name)
            
            return True
            
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
