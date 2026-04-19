import cv2
import os
import mediapipe as mp

# ==============================
# PATHS (adjusted to your structure)
# ==============================

real_video_path = "../dataset/videos/real"
fake_video_path = "../dataset/videos/fake"

save_real = "../dataset/faces/real"
save_fake = "../dataset/faces/fake"

os.makedirs(save_real, exist_ok=True)
os.makedirs(save_fake, exist_ok=True)

# ==============================
# FACE DETECTOR
# ==============================

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ==============================
# FUNCTION
# ==============================

def process_videos(video_folder, save_folder):

    count = 0

    print("Scanning folder:", video_folder)

    files = os.listdir(video_folder)

    if len(files) == 0:
        print("❌ No files found!")
        return

    for video_file in files:
        print("\nProcessing video:", video_file)

        video_full_path = os.path.join(video_folder, video_file)

        cap = cv2.VideoCapture(video_full_path)

        print("Opened:", video_full_path)
        print("Is opened:", cap.isOpened())

        if not cap.isOpened():
            print("❌ Cannot open video:", video_file)
            continue

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("❌ Cannot read frame / video ended")
                break

            frame_count += 1

            # sample every 5 frames
            if frame_count % 5 != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_detector.process(frame_rgb)

            if results.detections:
                for detection in results.detections:

                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    face = frame[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face = cv2.resize(face, (224, 224))

                    save_path = os.path.join(save_folder, f"{count}.jpg")
                    cv2.imwrite(save_path, face)

                    count += 1

        cap.release()

    print(f"✅ Total faces saved in {save_folder}: {count}")

# ==============================
# RUN
# ==============================

print("\n--- STARTING FACE EXTRACTION ---\n")

real_video_path = "../dataset/videos/real"
fake_video_path = "../dataset/videos/fake"

print("Extracting REAL faces...")
process_videos(real_video_path, save_real)

print("\nExtracting FAKE faces...")
process_videos(fake_video_path, save_fake)

print("\n✅ DONE")