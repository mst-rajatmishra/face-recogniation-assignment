# Student Face Recognition System

An offline Python proof-of-concept application for real-time student face recognition using the built-in webcam.

## Features

- **Real-time face detection** using OpenCV and face_recognition library
- **Multiple face recognition** simultaneously
- **Modular architecture** with separate components for each task
- **Optimized performance** with recognition every 3 frames
- **Visual feedback** with color-coded bounding boxes (green for recognized, red for unknown)
- **Live camera preview** with student names displayed above detected faces
- **Scalable design** supporting up to 40+ students

## Tech Stack

- **Python 3.7+**
- **OpenCV** - Camera capture and image processing
- **face_recognition** - Face detection and recognition (based on dlib)
- **numpy** - Numerical operations

## Project Structure

```
project/
├── main.py              # Main application loop and coordination
├── camera_module.py     # Webcam initialization and frame capture
├── face_detector.py     # Face detection in video frames
├── recognizer.py        # Face recognition and matching
├── dataset_loader.py    # Load student images and extract encodings
├── utils.py            # Drawing utilities for bounding boxes and text
├── requirements.txt    # Python dependencies
└── dataset/            # Folder containing student images
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Note on Windows:** The `face_recognition` library requires CMake and Visual Studio Build Tools. If you encounter installation issues:

   ```bash
   # Install CMake first
   pip install cmake
   
   # Then install the requirements
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Dataset

Add student images to the `dataset/` folder:
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- File name should be the student's first name (e.g., `alice.jpg`, `bob.png`)
- Each image should contain one clear face
- Recommended size: 200x200 pixels or larger

Example dataset structure:
```
dataset/
├── alice.jpg
├── bob.png
├── charlie.jpg
├── diana.jpeg
└── eve.jpg
```

### 2. Run the Application

```bash
python main.py
```

### 3. Controls

- **'q'** - Quit the application
- **'s'** - Save current frame as image

## Recognition Pipeline

The application follows this processing pipeline:

1. **Camera Initialization** → Setup webcam for video capture
2. **Frame Capture** → Continuously capture video frames
3. **Face Detection** → Locate faces in each frame using HOG/CNN
4. **Face Encoding** → Generate facial embeddings for detected faces
5. **Face Matching** → Compare with known student encodings
6. **Bounding Box Rendering** → Draw boxes and names on frame
7. **Display** → Show processed frame with results

## Performance Optimization

- **Frame Skipping**: Recognition performed every 3 frames instead of every frame
- **Result Caching**: Previous recognition results reused for skipped frames
- **HOG Model**: Uses CPU-optimized HOG model for face detection (configurable to CNN)

## Visual Indicators

- **Green bounding box** - Recognized student
- **Red bounding box** - Unknown person
- **Name text** - Student name above green boxes, "UNKNOWN" above red boxes
- **FPS counter** - Top-right corner
- **Status bar** - Bottom showing face count and recognition interval

## Module Responsibilities

### dataset_loader.py
- Load student images from dataset folder
- Extract face encodings using face_recognition
- Store names and encodings in memory

### camera_module.py
- Initialize webcam using cv2.VideoCapture(0)
- Capture frames continuously
- Handle camera errors and cleanup

### face_detector.py
- Detect face locations using face_recognition.face_locations
- Support for HOG (CPU) and CNN (GPU) models
- Filter small faces and get largest face

### recognizer.py
- Compare detected faces with known encodings
- Return recognized student names or "Unknown"
- Configurable tolerance for face matching

### utils.py
- Draw bounding boxes with color coding
- Display names above boxes with background
- Add FPS counter and status information

### main.py
- Coordinate all modules
- Main application loop
- Handle keyboard input and cleanup

## Configuration

You can modify these parameters in `main.py`:

```python
app = FaceRecognitionApp(
    dataset_path="dataset",        # Path to student images
    recognition_interval=3        # Recognition every N frames
)
```

And in `face_detector.py`:

```python
detector = FaceDetector(
    model="hog",                  # "hog" for CPU, "cnn" for GPU
    upscale_factor=1              # Upscale factor for better detection
)
```

## Limitations

- **No attendance logging** - This is a proof-of-concept only
- **No database** - All data stored in memory
- **No GUI framework** - Simple camera preview window
- **Lighting dependent** - Performance varies with lighting conditions
- **Single face per image** - Dataset images should contain one face

## Troubleshooting

### Camera not found
- Ensure built-in webcam is connected and not used by other applications
- Try changing camera index in `camera_module.py` (0, 1, 2...)

### No faces detected
- Check lighting conditions
- Ensure faces are clearly visible and facing the camera
- Try adjusting recognition tolerance in `recognizer.py`

### Installation issues
- Install Visual Studio Build Tools on Windows
- Ensure CMake is installed: `pip install cmake`
- Update pip: `python -m pip install --upgrade pip`

## Example Output

When running, you'll see:
- Live camera feed with bounding boxes around detected faces
- Student names displayed above recognized faces
- Green boxes for known students, red for unknown
- Real-time FPS counter and status information

## Future Enhancements

This proof-of-concept could be extended with:
- Attendance logging system
- Database integration
- Web-based GUI
- Multiple camera support
- Face registration interface
- Advanced recognition algorithms
