import cv2
import os
import math


# input video path
video_path = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\videos\sample\videoplayback.mp4"

# output folder
output_folder = "../dataset/frames/sample_video"

os.makedirs(output_folder, exist_ok=True)

# open video
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0
frame_skip = 5

import math

# get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

target_frames = 30

step = max(1, total_frames // target_frames)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame is None:
        continue

    if frame_count % step == 0 and saved_count < target_frames:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Total frames in video: {total_frames}")
print(f"Frames saved: {saved_count}")
cap.release()

print(f"Total frames read: {frame_count}")
print(f"Total frames saved: {saved_count}")