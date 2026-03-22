import os
import cv2
import mediapipe as mp

input_folder = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\frames\sample_video"

frame_files = os.listdir(input_folder)

print("Files found:", len(frame_files))

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

for file_name in frame_files:
    print("Processing:", file_name)

    frame_path = os.path.join(input_folder, file_name)
    image = cv2.imread(frame_path)

    if image is None:
        print("❌ Image not loaded")
        continue

    h, w, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        print("✅ Face detected")

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            face = image[y:y2, x:x2]

            print("👉 Cropped face shape:", face.shape)

            output_folder = r"C:\Users\Prasad\Desktop\DeepShield-AI\dataset\faces\sample_video"
            os.makedirs(output_folder, exist_ok=True)

            face_filename = os.path.join(output_folder, f"face_{file_name}")
            cv2.imwrite(face_filename, face)

            print("💾 Face saved")

    else:
        print("❌ No face detected")