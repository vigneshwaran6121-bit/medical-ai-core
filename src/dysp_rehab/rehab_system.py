import torch
import cv2
import numpy as np
import mediapipe as mp
import os
from lstm_model import DysphagiaLSTM

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map ID to Name (Must match your training)
CLASS_MAP = {0: "ChinDown", 1: "ChinTuck", 2: "HeadTilt", 3: "HeadTurn", 4: "Neck"}

# 21 Landmarks (Subset)
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
NOSE_IDX = 0
LEFT_EYE = 4
RIGHT_EYE = 5


class RehabSystem:
    def __init__(self):
        self.load_model()
        self.load_expert_data()
        self.init_mediapipe()

    def load_model(self):
        print("🧠 Loading AI Model...")
        self.model = DysphagiaLSTM(num_classes=len(CLASS_MAP)).to(DEVICE)

        # Robust path finding
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "dysphagia_model.pth")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
            print("✅ AI Ready.")
        else:
            print("❌ Error: Model not found. Run train.py first.")
            exit()

    def load_expert_data(self):
        """Loads the Expert .npy files for scoring comparison"""
        print("📚 Loading Expert Reference Data...")
        self.expert_refs = {}

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Pointing to the Processed folder we made in Phase 2
        processed_dir = os.path.join(script_dir, "..", "AI_Dysphagia", "Processed")

        if not os.path.exists(processed_dir):
            print(f"⚠️ Warning: processed folder not found at {processed_dir}")
            return

        # Load one reference file per class (taking the first one found)
        for filename in os.listdir(processed_dir):
            if "expert" in filename.lower():
                for cls_id, cls_name in CLASS_MAP.items():
                    if (
                        cls_name.lower() in filename.lower()
                        and cls_id not in self.expert_refs
                    ):
                        # Load and store the simplified reference path
                        data = np.load(os.path.join(processed_dir, filename))
                        # Flatten to 1D for simple distance checking: (150, 63)
                        self.expert_refs[cls_id] = data.reshape(data.shape[0], -1)
                        print(f"   🔹 Loaded Reference for: {cls_name}")

    def init_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False
        )

    def normalize(self, landmarks):
        """Standard Normalization (Scale & Center)"""
        p1 = np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y])
        p2 = np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
        dist = np.linalg.norm(p1 - p2)
        scale = dist if dist > 0 else 1.0

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

    def calculate_score(self, patient_sequence, class_id):
        """
        Compares Patient Motion vs Expert Motion
        Simple Euclidean Distance approach (Fast & Robust)
        """
        if class_id not in self.expert_refs:
            return 0.0  # No reference available

        expert_seq = self.expert_refs[class_id]

        # We compare the last N frames to the Expert's last N frames
        # This checks if the "holding position" matches
        compare_len = min(len(patient_sequence), len(expert_seq))

        pat_flat = np.array(patient_sequence[-compare_len:]).reshape(compare_len, -1)
        exp_flat = expert_seq[-compare_len:]

        # Calculate distance (lower is better)
        dist = np.mean(np.linalg.norm(pat_flat - exp_flat, axis=1))

        # Convert distance to a Score (0-100%)
        # Empirical logic: Distance of 0.0 = 100%, Distance of 0.2 = 0%
        score = max(0, 100 - (dist * 400))
        return score

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_buffer = []

        current_action = "Waiting..."
        current_score = 0.0
        feedback = "Get Ready"

        print("📷 Starting Camera...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                subset = [lm[i] for i in SWALLOW_LANDMARKS]

                # 1. Process & Normalize
                norm_frame = self.normalize(subset)
                frame_buffer.append(norm_frame)
                if len(frame_buffer) > SEQUENCE_LENGTH:
                    frame_buffer.pop(0)

                # 2. AI Inference (Every 5 frames)
                if len(frame_buffer) == SEQUENCE_LENGTH and len(frame_buffer) % 5 == 0:
                    input_data = np.array(frame_buffer).reshape(1, SEQUENCE_LENGTH, -1)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(
                        DEVICE
                    )

                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        conf, pred = torch.max(probs, 1)

                        class_id = pred.item()
                        current_action = CLASS_MAP.get(class_id, "Unknown")

                        # 3. Calculate Score (Math)
                        if conf.item() > 0.6:  # Only score if AI is confident
                            current_score = self.calculate_score(frame_buffer, class_id)

                            # Generate Feedback
                            if current_score > 80:
                                feedback = "Excellent!"
                            elif current_score > 50:
                                feedback = "Good, improve range"
                            else:
                                feedback = "Try harder / Move more"
                        else:
                            current_score = 0
                            feedback = "Adjusting..."

                # --- VISUALIZATION ---
                # Draw Landmarks
                for pt in subset:
                    cv2.circle(
                        frame,
                        (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])),
                        2,
                        (0, 255, 255),
                        -1,
                    )

            # Dashboard
            cv2.rectangle(frame, (0, 0), (1280, 80), (30, 30, 30), -1)

            # Action Text
            cv2.putText(
                frame,
                f"Action: {current_action}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # Score Bar
            cv2.rectangle(frame, (400, 20), (800, 50), (50, 50, 50), -1)
            bar_width = int((current_score / 100) * 400)
            bar_color = (0, 255, 0) if current_score > 70 else (0, 165, 255)
            cv2.rectangle(frame, (400, 20), (400 + bar_width, 50), bar_color, -1)
            cv2.putText(
                frame,
                f"{int(current_score)}%",
                (820, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Feedback Text
            cv2.putText(
                frame, feedback, (950, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2
            )

            cv2.imshow("Dysphagia Rehab System (AI + Math)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = RehabSystem()
    system.run()
