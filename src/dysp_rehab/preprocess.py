import cv2
import os
import numpy as np
import mediapipe as mp

# ==========================================
# 1. CONFIGURATION (Thesis Specifics)
# ==========================================

# Standard MediaPipe Imports
mp_face_mesh = mp.solutions.face_mesh

# The specific 21 landmarks for Dysphagia
SWALLOW_LANDMARKS = [
    1,
    4,
    6,
    10,
    33,
    263,
    152,
    172,
    136,
    150,
    176,
    379,
    395,
    365,
    397,
    234,
    127,
    93,
    454,
    356,
    323,
]

# Indices within the SUBSET list for normalization
NOSE_IDX_SUBSET = 0
LEFT_EYE_IDX_SUBSET = 4
RIGHT_EYE_IDX_SUBSET = 5


def normalize_spatial(landmarks_array):
    """
    Applies Thesis Normalization:
    1. Center on Nose
    2. Scale by Inter-ocular Distance
    """
    if len(landmarks_array) == 0:
        return landmarks_array, 1.0

    data = landmarks_array.copy()

    # 1. Calculate Scale (Eye Distance)
    p1 = data[:, LEFT_EYE_IDX_SUBSET, :2]
    p2 = data[:, RIGHT_EYE_IDX_SUBSET, :2]
    dist = np.linalg.norm(p1 - p2, axis=1)
    scale = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0

    # 2. Center Face (Translation Invariance)
    nose_ref = data[:, NOSE_IDX_SUBSET, :3]
    data[:, :, 0] -= nose_ref[:, 0, None]
    data[:, :, 1] -= nose_ref[:, 1, None]
    data[:, :, 2] -= nose_ref[:, 2, None]

    # 3. Apply Scale
    data /= scale

    return data, scale


def extract_landmarks_from_video(video_path):
    raw_landmarks = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"   ⚠️ Error: Could not open video")
        return None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Extract ONLY the 21 relevant points
                frame_points = []
                for idx in SWALLOW_LANDMARKS:
                    frame_points.append([lm[idx].x, lm[idx].y, lm[idx].z])

                raw_landmarks.append(frame_points)

    cap.release()

    raw_arr = np.array(raw_landmarks)

    if len(raw_arr) > 0:
        norm_arr, scale_factor = normalize_spatial(raw_arr)
        return norm_arr
    else:
        return np.empty((0, 21, 3))


def process_dataset(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    print(f"🔍 Scanning for videos in: {os.path.abspath(raw_dir)}")

    count = 0
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                vid_path = os.path.join(root, file)

                # Create unique filename (e.g., Patients_Sub_01_Video.npy)
                rel_path = os.path.relpath(root, raw_dir)
                safe_prefix = rel_path.replace(os.sep, "_").replace(" ", "")
                safe_name = os.path.splitext(file)[0].replace(" ", "")

                output_name = f"{safe_prefix}_{safe_name}.npy"
                save_path = os.path.join(processed_dir, output_name)

                if os.path.exists(save_path):
                    print(f"⏩ Skipping {output_name} (Exists)")
                    continue

                print(f"🎬 Processing: {output_name}...")
                try:
                    landmarks = extract_landmarks_from_video(vid_path)
                    if landmarks is not None and len(landmarks) > 0:
                        np.save(save_path, landmarks)
                        print(f"   ✅ Saved {landmarks.shape}")
                        count += 1
                    else:
                        print(f"   ⚠️ No face detected.")
                except Exception as e:
                    print(f"   ❌ Error: {e}")

    print(f"\n🎉 Processing Complete. Total videos processed: {count}")


if __name__ == "__main__":
    # --- ROBUST PATH CONFIGURATION ---
    # This ensures it finds the 'Data' folder regardless of where you run the command
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Points to medical-ai-core/AI_Dysphagia/Data
    RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "..", "AI_Dysphagia", "Data")
    PROCESSED_DATA_PATH = os.path.join(
        SCRIPT_DIR, "..", "..", "AI_Dysphagia", "Processed"
    )

    print(f"🎯 Target Input: {os.path.abspath(RAW_DATA_PATH)}")
    print(f"🎯 Target Output: {os.path.abspath(PROCESSED_DATA_PATH)}")

    process_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)
