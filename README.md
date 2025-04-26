# Real-Time Object Presence Detection – ML Intern Task
Author: Vikrant Tyagi
Date: April 26, 2025
Company: Samajh.ai (ML Engineer Intern Evaluation)

# Objective
Build a real-time video analytics system to detect:

✅ Missing Object Detection – Detect when an object disappears from the scene.

✅ New Object Placement Detection – Detect when a new object appears.

🛠# Tech Stack
Language: Python 3.10

Model: YOLOv5s (Pretrained)

Libraries: PyTorch, OpenCV, Pandas, NumPy

Environment: Docker (optional for reproducibility)

# Setup Instructions
1. Clone this repo:
git clone https://github.com/Vikrant_Tyagi/repo-name.git
cd repo-name
2. Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
pip install -r requirements.txt
4. Run the code
python main.py
To stop the video, press q in the video window.


# Sample Outputs
FPS Achieved: ~15 FPS (CPU)

Hardware: Intel i5, 8GB RAM, CPU-only

Example Labels:

"New: Bag"

"Missing: Bottle"


# Optimizations
Lightweight model (yolov5s) for speed

Manual FPS calculation using time

Skipped video writing to avoid FPS drop during testing

# Possible Improvements
Add object tracking (e.g., DeepSORT)

Use GPU if available

Use object count or class-wise change alerts

