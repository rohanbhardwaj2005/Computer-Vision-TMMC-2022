# TMMC-2022 Vehicle Wheel Detection System

A real-time computer vision system that detects vehicles and their wheels using PyTorch YOLOv5 and OpenCV. The system can process live webcam feeds and video files to identify cars/trucks and locate their wheels with high accuracy.

## 🎯 Project Overview

This project combines deep learning and traditional computer vision techniques to:
1. Detect vehicles (cars and trucks) in real-time using YOLOv5
2. Isolate vehicle regions from the frame
3. Detect wheels using Hough Circle Transform
4. Track wheel positions relative to a reference point (e.g., conveyor belt system)

### Use Cases
- Quality control in automotive manufacturing
- Vehicle inspection systems
- Automated measurement on conveyor belts
- Real-time vehicle analysis

## 🚀 Features

- **Real-time Detection**: Process live webcam feeds with minimal latency
- **Video Processing**: Batch process pre-recorded videos
- **Hybrid Approach**: Combines deep learning (YOLOv5) with classical CV (Hough Circles)
- **Distance Measurement**: Calculate wheel position relative to reference points
- **Automated Capture**: Trigger screenshots when wheels reach specific positions

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for YOLOv5 model
- **YOLOv5**: State-of-the-art object detection
- **OpenCV**: Image processing and computer vision operations
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Google Colab**: Development and execution environment

## 📋 Prerequisites

```python
Python 3.7+
torch>=1.7.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
IPython
ipywidgets
Pillow
```

## 💻 Installation

### For Google Colab (Recommended)

1. Open the notebook in Google Colab
2. All dependencies are pre-installed or will be installed automatically
3. Mount Google Drive if processing video files:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### For Local Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-wheel-detection.git
cd vehicle-wheel-detection

# Install dependencies
pip install torch torchvision
pip install opencv-python numpy matplotlib ipython ipywidgets pillow

# Install YOLOv5
pip install yolov5
```

## 🎮 Usage

### 1. Real-time Webcam Detection

```python
import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start video stream
video_stream()

# Process frames in real-time
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break
    
    img = js_to_image(js_reply["img"])
    results = model(img)
    
    # Process detections...
```

### 2. Video File Processing

```python
# Path to your video file
video_path = "/path/to/your/video.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Process video
while cap.isOpened():
    is_success, frame = cap.read()
    if not is_success:
        break
    
    results = model(frame)
    # Process and save results...
```

### 3. Single Image Detection

```python
# Load image
img = cv2.imread('path/to/image.jpg')

# Run detection
results = model(img)

# Display results
results.show()
```

## 🔧 Configuration

### Detection Parameters

```python
# Vehicle classes to detect (COCO dataset)
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 7

# Wheel detection parameters
RESIZE_CONSTANT = 16  # Upscaling factor for better circle detection
HOUGH_MIN_DIST = 400  # Minimum distance between circle centers
HOUGH_PARAM1 = 200    # Edge detection threshold
HOUGH_PARAM2 = 70     # Circle detection threshold

# Reference positions
METAL_BAR_X = 290     # Reference point for distance measurement
CAPTURE_THRESHOLD = 30 # Pixel threshold for capture trigger
```

### Image Enhancement Settings

```python
# Brightness and contrast adjustment
ALPHA = 3      # Contrast control (1.0-3.0)
BETA = 0.3     # Brightness control (-100 to 100)

# Median blur kernel size
BLUR_KERNEL = 5

# Canny edge detection
CANNY_THRESHOLD1 = 75
CANNY_THRESHOLD2 = 100
```

## 📊 How It Works

### Detection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Input Frame (Webcam/Video)                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. YOLOv5 Vehicle Detection (PyTorch)                       │
│    - Detects cars (class 2) and trucks (class 7)           │
│    - Returns bounding boxes [x1, y1, x2, y2, conf, class]  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Region of Interest (ROI) Extraction                      │
│    - Crop lower half of vehicle (where wheels are)         │
│    - Upscale for better detection                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Image Preprocessing (OpenCV)                             │
│    - Convert to HSV color space                            │
│    - Enhance contrast and brightness                       │
│    - Apply median blur                                     │
│    - Canny edge detection                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Wheel Detection (Hough Circle Transform)                 │
│    - Detect circular shapes (wheels)                       │
│    - Filter by position (rightmost wheel)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Distance Calculation & Overlay                           │
│    - Calculate distance from reference point               │
│    - Draw bounding boxes and annotations                   │
│    - Trigger capture if at target position                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **PyTorch YOLOv5 Integration**
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
results = model(img)
```
- Uses pre-trained YOLOv5s model from PyTorch Hub
- Trained on COCO dataset (80 object classes)
- Real-time inference with GPU acceleration support

#### 2. **Vehicle Detection**
```python
for i in results.xyxy[0]:
    if i[5] == 2 or i[5] == 7:  # Car or Truck
        x_min, y_min, x_max, y_max = int(i[0]), int(i[1]), int(i[2]), int(i[3])
```
- Filters for car (class 2) and truck (class 7) detections
- Extracts bounding box coordinates

#### 3. **Wheel Detection**
```python
circles = cv2.HoughCircles(
    img_edge,
    cv2.HOUGH_GRADIENT,
    1.1,
    minDist=400,
    param1=200,
    param2=70
)
```
- Uses Hough Circle Transform for circular wheel detection
- Applies on preprocessed/edge-detected images

## 📈 Performance

- **Frame Rate**: ~15-30 FPS (depending on hardware)
- **Detection Accuracy**: High for vehicles in good lighting
- **Wheel Detection**: Works best with clear, circular wheels
- **GPU Acceleration**: Supported via PyTorch CUDA

## 🔍 Example Results

The system outputs:
- **Blue bounding boxes**: Detected vehicles
- **Green rectangles**: All detected circular objects
- **Yellow rectangles**: Confirmed wheel detections
- **Console output**: Distance measurements and capture triggers

## 🐛 Known Limitations

1. **Lighting Sensitivity**: Wheel detection may struggle in poor lighting
2. **Occlusion**: Partially hidden wheels may not be detected
3. **Wheel Design**: Works best with standard circular wheels
4. **Camera Angle**: Requires side view for optimal wheel detection
5. **Processing Speed**: Real-time performance depends on hardware
