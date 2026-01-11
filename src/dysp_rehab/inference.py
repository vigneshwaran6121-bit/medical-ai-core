import torch
import cv2
import numpy as np
import mediapipe as mp
import os
from lstm_model import DysphagiaLSTM

# ==========================================
# 1. SETUP
# ==========================================
# Must match your training config
SEQUENCE_LENGTH = 150
CLASS_MAP = {
    0: "Chin Down",
    1: "Chin Tuck",
    2: "Head Tilt",
    3: "Head Turn",
    4: "Neck Extension",
}

# The 21 Landmarks
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
NOSE_IDX = 0  # Index 0 in the subset list
LEFT_EYE = 4
RIGHT_EYE = 5


def normalize_realtime(landmarks):
    """Same math as your preprocessing, but for a single frame"""
    # 1. Scale
    p1 = np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y])
    p2 = np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
    dist = np.linalg.norm(p1 - p2)
    scale = dist if dist > 0 else 1.0

    # 2. Center
    nose = np.array(
        [landmarks[NOSE_IDX].x, landmarks[NOSE_IDX].y, landmarks[NOSE_IDX].z]
    )

    normalized = []
    for i in range(len(landmarks)):
        lx, ly, lz = landmarks[i].x, landmarks[i].y, landmarks[i].z
        normalized.append(
            [(lx - nose[0]) / scale, (ly - nose[1]) / scale, (lz - nose[2]) / scale]
        )

    return normalized


def run_inference():
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DysphagiaLSTM(num_classes=5).to(device)

    # Load weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "dysphagia_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("✅ Model Loaded Successfully!")

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False
    )

    # Buffer to store the last 150 frames
    frame_buffer = []
    prediction_text = "Waiting for data..."
    confidence_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        if results.multi_face_landmarks:
            # Get raw landmarks
            lm = results.multi_face_landmarks[0].landmark
            subset = [lm[i] for i in SWALLOW_LANDMARKS]

            # Normalize
            norm_frame = normalize_realtime(subset)
            frame_buffer.append(norm_frame)

            # Keep buffer at fixed size
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            # Draw Face Mesh
            for pt in subset:
                cv2.circle(
                    frame,
                    (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])),
                    2,
                    (0, 255, 255),
                    -1,
                )

            # --- PREDICTION LOGIC ---
            # Predict every 10 frames (to avoid flickering)
            if len(frame_buffer) == SEQUENCE_LENGTH and len(frame_buffer) % 10 == 0:
                # Prepare Input: (1, 150, 63)
                input_data = np.array(frame_buffer).reshape(1, SEQUENCE_LENGTH, -1)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                    class_id = predicted.item()
                    conf_score = confidence.item() * 100

                    prediction_text = CLASS_MAP.get(class_id, "Unknown")
                    confidence_text = f"Conf: {conf_score:.1f}%"

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (640, 60), (40, 40, 40), -1)

        color = (0, 255, 0) if "Waiting" not in prediction_text else (200, 200, 200)
        cv2.putText(
            frame,
            f"AI Detects: {prediction_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        cv2.putText(
            frame,
            confidence_text,
            (450, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            1,
        )

        cv2.imshow("Dysphagia AI Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
