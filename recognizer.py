"""
Face Recognizer Module

Responsibilities:
- Compare detected faces with known encodings
- Return recognized student names
- Handle unknown faces

Pipeline Step: Face Locations → Face Encoding → Face Matching → Name Output
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict, Optional


class FaceRecognizer:
    """Recognizes faces by comparing with known face encodings."""
    
    def __init__(self, tolerance: float = 0.6):
        """
        Initialize the face recognizer.
        
        Args:
            tolerance: How much distance between faces to consider it a match
                      Lower is stricter, 0.6 is typical default
        """
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_faces(self, encodings: List[np.ndarray], names: List[str]):
        """
        Load known face encodings and names for recognition.
        
        Args:
            encodings: List of face encodings
            names: List of corresponding names
            
        Pipeline: Known Faces Storage → Ready for Recognition
        """
        if len(encodings) != len(names):
            raise ValueError("Number of encodings must match number of names")
        
        self.known_face_encodings = encodings
        self.known_face_names = names
        print(f"Loaded {len(self.known_face_encodings)} known faces for recognition")
    
    def recognize_faces(self, frame: np.ndarray, 
                       face_locations: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Recognize faces in the given frame based on detected face locations.
        
        Args:
            frame: Input frame in BGR format
            face_locations: List of face bounding boxes from face detector
            
        Returns:
            List of recognized names (or "Unknown" for unrecognized faces)
            
        Pipeline: Frame Input → Face Encoding → Comparison → Name Assignment
        """
        if not face_locations:
            return []
        
        if not self.known_face_encodings:
            print("Warning: No known faces loaded for recognition")
            return ["Unknown"] * len(face_locations)
        
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face encodings for detected faces
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # If no encodings found (shouldn't happen if face_locations exist)
            if not face_encodings:
                return ["Unknown"] * len(face_locations)
            
            # Compare each face encoding with known faces
            recognized_names = []
            
            for face_encoding in face_encodings:
                # See if the face is a match for the known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                
                # Find the best match
                name = "Unknown"
                
                # If we found at least one match
                if True in matches:
                    # Find indices of all matches
                    match_indices = [i for i, match in enumerate(matches) if match]
                    
                    # Use the first match (could be enhanced to use face distance)
                    first_match_index = match_indices[0]
                    name = self.known_face_names[first_match_index]
                
                recognized_names.append(name)
            
            return recognized_names
            
        except Exception as e:
            print(f"Error recognizing faces: {e}")
            return ["Unknown"] * len(face_locations)
    
    def recognize_faces_with_distance(self, frame: np.ndarray, 
                                    face_locations: List[Tuple[int, int, int, int]]) -> List[Tuple[str, float]]:
        """
        Recognize faces and return confidence distances.
        
        Args:
            frame: Input frame in BGR format
            face_locations: List of face bounding boxes
            
        Returns:
            List of tuples (name, distance) where distance is face distance
        """
        if not face_locations or not self.known_face_encodings:
            return [("Unknown", 1.0)] * len(face_locations)
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                return [("Unknown", 1.0)] * len(face_locations)
            
            results = []
            
            for face_encoding in face_encodings:
                # Calculate face distances to all known faces
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                # Find the best match (minimum distance)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                
                # Check if the best match is within tolerance
                if min_distance <= self.tolerance:
                    name = self.known_face_names[best_match_index]
                else:
                    name = "Unknown"
                
                results.append((name, min_distance))
            
            return results
            
        except Exception as e:
            print(f"Error recognizing faces with distance: {e}")
            return [("Unknown", 1.0)] * len(face_locations)
    
    def get_recognition_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded known faces.
        
        Returns:
            Dictionary with recognition statistics
        """
        return {
            "total_known_faces": len(self.known_face_encodings),
            "unique_names": len(set(self.known_face_names)),
            "tolerance": self.tolerance
        }
    
    def update_tolerance(self, new_tolerance: float):
        """
        Update the recognition tolerance.
        
        Args:
            new_tolerance: New tolerance value (0.0 to 1.0)
        """
        if 0.0 <= new_tolerance <= 1.0:
            self.tolerance = new_tolerance
            print(f"Updated recognition tolerance to {new_tolerance}")
        else:
            print("Error: Tolerance must be between 0.0 and 1.0")
    
    def add_known_face(self, encoding: np.ndarray, name: str):
        """
        Add a new known face to the recognizer.
        
        Args:
            encoding: Face encoding to add
            name: Name associated with the face
        """
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        print(f"Added new known face: {name}")
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face by name.
        
        Args:
            name: Name of the face to remove
            
        Returns:
            True if face was removed, False if not found
        """
        if name not in self.known_face_names:
            return False
        
        # Find all indices with this name
        indices_to_remove = [i for i, known_name in enumerate(self.known_face_names) if known_name == name]
        
        # Remove in reverse order to maintain indices
        for index in reversed(indices_to_remove):
            del self.known_face_encodings[index]
            del self.known_face_names[index]
        
        print(f"Removed {len(indices_to_remove)} instances of {name}")
        return True
