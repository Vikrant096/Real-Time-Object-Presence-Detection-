#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
print(torch.__version__)


# In[ ]:


import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # Confidence threshold

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Setup video
video_path = 'input/input_video.mp4'
cap = cv2.VideoCapture(video_path)
prev_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    dets = []
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append(([x1, y1, x2 - x1, y2 - y1], conf, str(int(cls))))

    tracks = tracker.update_tracks(dets, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        current_ids.add(track_id)

        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Detection logic
    missing = prev_ids - current_ids
    new = current_ids - prev_ids

    for mid in missing:
        print(f'Object ID {mid} missing!')

    for nid in new:
        print(f'New object ID {nid} placed!')

    prev_ids = current_ids

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

