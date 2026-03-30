# ReID-TrackNet: Real-Time Multi-Object Tracking with Re-Identification

ReID-TrackNet is a real-time multi-object tracking system that combines object detection, motion prediction, and appearance-based re-identification to achieve stable identity tracking across video frames.

---

## Features

- ✅ Real-time human detection using YOLOv8
- ✅ Multi-object tracking with Kalman Filter
- ✅ Hungarian Algorithm for optimal assignment
- ✅ Appearance modeling using ReID (OSNet)
- ✅ Identity stability with Exponential Moving Average (EMA)
- ✅ Robust tracking under occlusion
- ✅ Trajectory visualization
- ✅ Video output saving

---

## Pipeline

Detection → Feature Extraction → Motion Prediction → Data Association → Track Management

---

## Evaluation Results

| Metric    | Score |
|----------|------|
| MOTA     | 0.96 |
| IDF1     | 0.64 |
| Precision| 1.00 |
| Recall   | 0.99 |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Key Techniques

- **YOLOv8** for object detection
- **Kalman Filter** for motion prediction
- **Hungarian Algorithm** for optimal assignment
- **Cosine Similarity** for feature matching
- **OSNet (ReID)** for identity preservation
- **Exponential Moving Average (EMA)** for stable embeddings

---

## Applications

- Surveillance systems
- Autonomous driving perception
- Crowd monitoring
- Smart city analytics

---
