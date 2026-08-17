# D. Player & Ball Tracking System Design

## Overview

This document defines the tracking layer of the football analysis system.

The purpose of this module is to transform frame-by-frame object detections from the YOLO detection model into continuous player and ball identities.

The Tracking System provides:

- Stable Player IDs
- Ball tracking
- Player trajectories
- Movement speed
- Distance covered
- Position history
- Data required for tactical analysis


---

# 1. Goal

Track all players and the ball throughout a football match video.

The system should maintain consistent identities between frames.

Example:

YOLO Detection:

Frame 1:
Player → (120,300)

Frame 2:
Player → (125,305)

Tracking Output:

Player ID 10
Frame 1 → (120,300)
Frame 2 → (125,305)

---

# 2. System Inputs

The Tracking System receives data from the Computer Vision Layer.

## Player Detection Input

Source:
- YOLO Object Detection Model

Format:

```json
{
  "class": "player",
  "bbox": [x, y, width, height],
  "confidence": 0.96
}
Ball Detection Input
Format:
{
  "class": "ball",
  "bbox": [x, y, width, height],
  "confidence": 0.88
}
Frame Information
Format:
{
  "frame_id": 120,
  "timestamp": 4.0,
  "fps": 30
}
3. Tracking Algorithm
Selected Algorithm
ByteTrack
Reasons:
High speed
Suitable for football videos
Works well with multiple objects
Does not require additional Re-ID training
Suitable for real-time processing
4. Tracking Pipeline
Video Input

↓

Frame Extraction

↓

YOLO Detection

↓

Detection Filtering

↓

ByteTrack

↓

Object ID Assignment

↓

Trajectory Generation

↓

Feature Extraction

↓

Database Storage

↓

Analysis Engine
5. Tracking Output
The tracker generates unique IDs for every player.
Example:
{
  "frame_id":120,
  "players":[
    {
      "player_id":10,
      "bbox":[530,280,60,120],
      "confidence":0.95
    }
  ],
  "ball":{
    "x":600,
    "y":310,
    "confidence":0.88
  }
}
6. Player Trajectory System
Each player's movement history is stored.
Example:
{
  "player_id":10,
  "trajectory":[
    {
      "frame":1,
      "x":120,
      "y":300
    },
    {
      "frame":2,
      "x":125,
      "y":305
    }
  ]
}
Trajectory data is used for:
Distance calculation
Speed calculation
Heatmaps
Tactical positioning
Formation detection
7. Movement Feature Extraction
The Tracking System calculates physical features.
Distance Covered
Formula:
distance =
sqrt((x2-x1)^2 + (y2-y1)^2)
Output:
Total distance covered per player
Speed Calculation
Formula:
speed =
distance / time
Used for:
Sprint detection
Running intensity
Injury risk analysis
Acceleration
Formula:
acceleration =
(speed2-speed1) / time
Used for:
Sprint load
Fatigue estimation
8. Coordinate Transformation
Problem
YOLO outputs pixel coordinates.
Example:
(600,400) pixels
Football analysis requires real-world pitch coordinates.
Example:
(35 meters,20 meters)
Solution
Use Homography Transformation.
Pipeline:
Camera Coordinates

↓

Homography Matrix

↓

Pitch Coordinates

↓

Real World Measurements
This enables:
Accurate distance calculation
Tactical zones
Heatmaps
9. Database Integration
Tracking output is stored in:
PlayerTracking
Schema:
PlayerTracking

- player_id
- frame_id
- pitch_x_meter
- pitch_y_meter
- speed
- distance
Relationship:
Match

↓

Frame

↓

PlayerDetection

↓

PlayerTracking

↓

Event

↓

PlayerMetric
10. Tracking Quality Metrics
The system evaluates tracking performance using:
Metric	Goal
ID Switches	Minimize identity changes
Tracking Accuracy	Maximize correct tracking
Missing Frames	Minimize lost objects
FPS Performance	Maintain processing speed

11. Implementation Stack
Technologies:
YOLOv8
+
ByteTrack
+
OpenCV
+
NumPy
+
Homography
+
PostgreSQL
12. Module Structure
Recommended implementation:
tracking/

├── tracker.py
├── trajectory.py
├── speed.py
├── homography.py
└── utils.py
13. Future Extensions
Possible improvements:
Player Re-Identification
Team Assignment
Tactical Zone Detection
Automatic Possession Detection
Advanced Movement Analysis