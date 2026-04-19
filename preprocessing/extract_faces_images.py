import cv2
import os
import mediapipe as mp

# ==============================
# PATHS (IMAGE DATASET)
# ==============================

real_path = "../dataset/Train/Real"
fake_path = "../dataset/Train/Fake"

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

def process_images(folder_path, save_folder):

    count = 0

    files = os.listdir(folder_path)
    print("Total images in", folder_path, ":", len(files))

    for img_file in files:

        img_path = os.path.join(folder_path, img_file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_detector.process(img_rgb)

        if results.detections:
            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))

                save_path = os.path.join(save_folder, f"{count}.jpg")
                cv2.imwrite(save_path, face)

                count += 1

    print(f"✅ Saved {count} faces to {save_folder}")

# ==============================
# RUN
# ==============================

print("Extracting REAL faces...")
process_images(real_path, save_real)

print("\nExtracting FAKE faces...")
process_images(fake_path, save_fake)

print("\n✅ DONE")