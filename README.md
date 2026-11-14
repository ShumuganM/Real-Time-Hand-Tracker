# Real-Time Hand & Finger Tracker

A real-time computer vision application built with **Python**, **OpenCV**, and **MediaPipe** that detects 21 3D hand landmarks per hand, identifies individual fingers (`Thumb`, `Index`, `Middle`, `Ring`, `Pinky`), and determines finger states (raised vs. folded) at 30+ FPS.

## Features
- **21-Point Hand Skeleton:** Maps wrist, knuckles, joints, and fingertips in real time for up to two hands.
- **Individual Finger Identification:** Color-codes and labels all five fingertips with live handedness (`Left` vs. `Right`) detection.
- **Geometric State Detection:** Compares normalized joint coordinates to detect which fingers are raised or folded into a fist.

## Setup & Usage
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt