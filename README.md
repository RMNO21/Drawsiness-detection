# Drowsiness Detection System

Real-time eye detection and drowsiness monitoring using a laptop webcam. Built with Python, OpenCV, dlib, MediaPipe, and PySide6.

## Features

- **Real-time eye tracking** via MediaPipe Face Mesh (468 landmarks)
- **Eye Aspect Ratio (EAR)** computation with temporal smoothing
- **Adaptive threshold calibration** for different users
- **Multi-level warnings**: Warning → Alarm → Emergency
- **Audio alerts** with configurable volume
- **Blink detection** with frequency tracking
- **Yawning detection** via mouth aspect ratio
- **Live EAR graph** dashboard
- **Statistics tracking** with CSV export
- **Dark modern GUI** built with PySide6

## Demo

| State | Description |
|-------|-------------|
| NORMAL | Eyes open, monitoring active |
| WARNING | Eyes closed > 0.8s |
| ALARM | Eyes closed > 1.5s |
| EMERGENCY | Eyes closed > 2.5s — "WAKE UP!" overlay |

## Installation

### Prerequisites

- Python 3.10+
- Webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/RMNO21/Drawsiness-detection.git
cd Drawsiness-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

1. Launch the application with `python main.py`
2. Click **Start** to begin monitoring
3. Position your face in the camera view
4. The system will calibrate automatically (~1 second)
5. Monitor your eye state in real-time

### Controls

| Button | Key | Action |
|--------|-----|--------|
| Start | F5 | Begin monitoring |
| Stop | F6 | Stop monitoring |
| Pause | F7 | Pause/resume |

## Configuration

Edit `settings.json` to customize behavior:

```json
{
  "camera_index": 0,
  "ear_threshold": 0.21,
  "drowsiness_warning_sec": 0.8,
  "drowsiness_alarm_sec": 1.5,
  "drowsiness_emergency_sec": 2.5,
  "alarm_volume": 0.7
}
```

## Project Structure

```
.
├── main.py              # Entry point
├── config.py            # Configuration management
├── camera.py            # Webcam capture
├── face_detector.py     # MediaPipe face detection
├── eye_detector.py      # EAR computation & state tracking
├── alarm.py             # Audio alert system
├── gui.py               # PySide6 GUI
├── ear_graph.py         # EAR history chart widget
├── statistics.py        # Stats tracking & CSV export
├── logger.py            # Logging configuration
├── settings.json        # Default settings
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── assets/
    └── models/          # Shape predictor model
```

## How It Works

### Eye Aspect Ratio (EAR)

The system computes EAR using six reference points per eye:

```
EAR = (v1 + v2 + v3) / (2.0 * h)
```

Where v1-v3 are vertical distances and h is horizontal distance. When eyes close, EAR drops below the threshold.

### Detection Pipeline

1. **Frame capture** from webcam
2. **Face detection** via YuNet + dlib 68-point landmarks
3. **Landmark extraction** with EMA smoothing
4. **EAR computation** for both eyes
5. **State classification**: Open / Closed / Blinking
6. **Drowsiness scoring** based on closure duration
7. **Alarm triggering** at configured thresholds

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
dlib>=19.24.0
PySide6>=6.5.0
psutil>=5.9.0
pygame>=2.5.0
```

## License

MIT License - see [LICENSE](LICENSE) for details.
