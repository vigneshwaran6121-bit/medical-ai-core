import cv2
import numpy as np
import mediapipe as mp
import torch
import os
import time
import math
from lstm_model import DysphagiaLSTM

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
WINDOW_NAME = "AI Dysphagia Rehabilitation System (Thesis Final)"
WIDTH, HEIGHT = 1280, 720  # The target resolution we want
SEQUENCE_LENGTH = 150
AI_CHECK_INTERVAL = 30

# Colors
C_BG = (20, 20, 20)
C_ACCENT = (0, 255, 255)  # Cyan
C_OK = (0, 255, 127)  # Green
C_WARN = (0, 165, 255)  # Orange
C_ERR = (0, 0, 255)  # Red
C_TXT = (255, 255, 255)

# Exercise Mapping
EXERCISES = {
    0: {"name": "Chin Down", "key": "ChinDown"},
    1: {"name": "Chin Tuck", "key": "ChinTuck"},
    2: {"name": "Head Tilt", "key": "HeadTilt"},
    3: {"name": "Head Turn", "key": "HeadTurn"},
    4: {"name": "Neck Extension", "key": "Neck"},
}

# 21 Landmarks
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


# ==========================================
# 2. MATH & AI ENGINE (FINAL FIX)
# ==========================================
class AIEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.expert_data = {}
        self.load_model()
        self.load_expert_cache()

    def load_model(self):
        try:
            self.model = DysphagiaLSTM(num_classes=5).to(self.device)
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "dysphagia_model.pth"
            )
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print("✅ AI Model Loaded.")
        except Exception as e:
            print(f"❌ AI Model Error: {e}")

    def load_expert_cache(self):
        print("\n🔎 LOOKING FOR EXPERT DATA...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        processed_dir = os.path.join(
            script_dir, "..", "..", "AI_Dysphagia", "Processed"
        )

        if not os.path.exists(processed_dir):
            print(f"❌ ERROR: Processed folder missing at {processed_dir}")
            return

        found_count = 0
        for f in os.listdir(processed_dir):
            # Create a "clean" version of filename: lowercase, no spaces, no underscores
            clean_name = f.lower().replace(" ", "").replace("_", "")

            # We look for files containing 'expert'
            if "expert" in clean_name:
                for id, info in EXERCISES.items():
                    # clean the key too: "Head Tilt" -> "headtilt"
                    clean_key = info["key"].lower().replace(" ", "")

                    # Fuzzy Match: Does "headtilt" exist inside "expert...headtilt..."?
                    if clean_key in clean_name:
                        try:
                            data = np.load(os.path.join(processed_dir, f))
                            # Take the last 150 frames (most relevant part) to store
                            data = data.reshape(data.shape[0], -1)
                            self.expert_data[id] = data
                            print(f"   🔹 MATCHED: {info['name']} -> {f}")
                            found_count += 1
                        except:
                            pass

        if found_count == 0:
            print("❌ WARNING: No Expert files matched! Score will be 0%.")
            print("   (Ensure you ran preprocess.py on the 'Expert' folder)")
        else:
            print(f"✅ Success: Loaded {found_count} expert references.\n")

    def normalize(self, landmarks):
        p1 = np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y])
        p2 = np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y])
        scale = np.linalg.norm(p1 - p2)
        if scale == 0:
            scale = 1.0
        nose = np.array(
            [landmarks[NOSE_IDX].x, landmarks[NOSE_IDX].y, landmarks[NOSE_IDX].z]
        )
        norm = []
        for p in landmarks:
            norm.append(
                [
                    (p.x - nose[0]) / scale,
                    (p.y - nose[1]) / scale,
                    (p.z - nose[2]) / scale,
                ]
            )
        return norm

    def predict_live(self, buffer):
        if self.model is None or len(buffer) != SEQUENCE_LENGTH:
            return -1, 0.0
        inp = np.array(buffer).reshape(1, SEQUENCE_LENGTH, -1)
        tensor = torch.tensor(inp, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
            return pred.item(), conf.item()

    def calculate_dtw_score(self, patient_seq, exercise_id):
        if exercise_id not in self.expert_data:
            print(f"⚠️ No Expert Data for ID {exercise_id}")
            return 0.0

        expert = self.expert_data[exercise_id]

        # Compare the active tail of the movement
        L = min(len(patient_seq), len(expert))
        if L < 10:
            return 0.0

        p_flat = np.array(patient_seq[-L:]).reshape(L, -1)
        e_flat = expert[-L:]

        dist = np.mean(np.linalg.norm(p_flat - e_flat, axis=1))

        # Calibration: 0.0 dist = 100%, 0.4 dist = 0%
        score = max(0, 100 - (dist * 250))
        return score


# ==========================================
# 3. MAIN APPLICATION
# ==========================================
class ThesisApp:
    def __init__(self):
        # FIX 1: Explicitly set window size so it doesn't shrink
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

        self.cap = cv2.VideoCapture(0)
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False
        )

        self.ai = AIEngine()
        self.state = "LOGIN"
        self.user_name = ""
        self.input_buf = ""
        self.selected_ex_id = 0

        self.frame_buffer = []
        self.full_recording = []
        self.live_feedback = "Waiting..."
        self.live_score_color = C_TXT
        self.final_score = 0.0

    def draw_ui_header(self, img):
        cv2.rectangle(img, (0, 0), (WIDTH, 50), (30, 30, 30), -1)
        cv2.putText(
            img,
            f"DYSPHAGIA AI | User: {self.user_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            C_ACCENT,
            1,
        )
        if self.state != "LOGIN":
            ex_name = EXERCISES[self.selected_ex_id]["name"]
            cv2.putText(
                img,
                f"Exercise: {ex_name}",
                (WIDTH - 400, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                C_TXT,
                1,
            )

    def process(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)

        # --- CRITICAL FIX: RESIZE CAMERA TO MATCH UI ---
        # This ensures the camera image is exactly 1280x720, so our UI math works.
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        canvas = frame.copy()

        # ---------------- LOGIN STATE ----------------
        if self.state == "LOGIN":
            canvas.fill(20)  # Dark Background
            # Boxes are drawn based on WIDTH (1280), so they will now be centered!
            cv2.rectangle(
                canvas,
                (WIDTH // 2 - 200, HEIGHT // 2 - 40),
                (WIDTH // 2 + 200, HEIGHT // 2 + 40),
                C_ACCENT,
                2,
            )
            cv2.putText(
                canvas,
                "ENTER PATIENT NAME:",
                (WIDTH // 2 - 180, HEIGHT // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                C_TXT,
                2,
            )
            cv2.putText(
                canvas,
                self.input_buf + "|",
                (WIDTH // 2 - 180, HEIGHT // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                C_TXT,
                2,
            )
            cv2.putText(
                canvas,
                "Press ENTER to Start",
                (WIDTH // 2 - 100, HEIGHT // 2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 150),
                1,
            )

        # ---------------- MENU STATE ----------------
        elif self.state == "MENU":
            canvas.fill(20)
            self.draw_ui_header(canvas)
            y = 150
            for id, info in EXERCISES.items():
                col = C_ACCENT if id == self.selected_ex_id else (60, 60, 60)
                cv2.rectangle(
                    canvas,
                    (WIDTH // 2 - 250, y),
                    (WIDTH // 2 + 250, y + 60),
                    col,
                    -1 if id == self.selected_ex_id else 2,
                )
                txt_col = (0, 0, 0) if id == self.selected_ex_id else C_TXT
                cv2.putText(
                    canvas,
                    info["name"],
                    (WIDTH // 2 - 230, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    txt_col,
                    2,
                )
                y += 80
            cv2.putText(
                canvas,
                "Select: W/S | Confirm: ENTER",
                (WIDTH // 2 - 150, HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                C_TXT,
                1,
            )

        # ---------------- RECORD/CALIB STATE ----------------
        elif self.state in ["CALIB", "RECORD"]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                subset = [lm[i] for i in SWALLOW_LANDMARKS]
                norm = self.ai.normalize(subset)

                # Draw Landmarks
                for p in subset:
                    # Draw points scaled to the NEW frame size
                    cv2.circle(
                        canvas, (int(p.x * WIDTH), int(p.y * HEIGHT)), 2, C_ACCENT, -1
                    )

                if self.state == "RECORD":
                    self.frame_buffer.append(norm)
                    self.full_recording.append(norm)
                    if len(self.frame_buffer) > SEQUENCE_LENGTH:
                        self.frame_buffer.pop(0)

                    if (
                        len(self.frame_buffer) == SEQUENCE_LENGTH
                        and len(self.full_recording) % AI_CHECK_INTERVAL == 0
                    ):
                        pred_id, conf = self.ai.predict_live(self.frame_buffer)
                        if pred_id == self.selected_ex_id:
                            self.live_feedback = f"✅ Correct: {EXERCISES[pred_id]['name']} ({int(conf*100)}%)"
                            self.live_score_color = C_OK
                        else:
                            detected = EXERCISES.get(pred_id, {"name": "Unknown"})[
                                "name"
                            ]
                            self.live_feedback = f"⚠️ WRONG! Detected: {detected}"
                            self.live_score_color = C_ERR

            self.draw_ui_header(canvas)
            if self.state == "CALIB":
                # Calibration Overlay
                cv2.rectangle(
                    canvas,
                    (WIDTH // 2 - 300, HEIGHT // 2 - 100),
                    (WIDTH // 2 + 300, HEIGHT // 2 + 100),
                    (0, 0, 0, 150),
                    -1,
                )
                cv2.putText(
                    canvas,
                    "Position Face in Center",
                    (WIDTH // 2 - 180, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    C_TXT,
                    2,
                )
                cv2.putText(
                    canvas,
                    "Press 'S' to Start Recording",
                    (WIDTH // 2 - 200, HEIGHT // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    C_OK,
                    2,
                )
            else:
                # Live AI Feedback
                cv2.rectangle(canvas, (0, HEIGHT - 60), (WIDTH, HEIGHT), (0, 0, 0), -1)
                cv2.putText(
                    canvas,
                    self.live_feedback,
                    (50, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.live_score_color,
                    2,
                )
                cv2.putText(
                    canvas,
                    "Press 'S' to Stop & Score",
                    (WIDTH - 350, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 150, 150),
                    1,
                )

        # ---------------- REPORT STATE ----------------
        elif self.state == "REPORT":
            canvas.fill(30)
            self.draw_ui_header(canvas)
            cv2.circle(canvas, (WIDTH // 4, HEIGHT // 2), 120, C_ACCENT, 2)
            cv2.putText(
                canvas,
                f"{int(self.final_score)}%",
                (WIDTH // 4 - 60, HEIGHT // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                C_ACCENT,
                4,
            )
            cv2.putText(
                canvas,
                "Similarity Score",
                (WIDTH // 4 - 70, HEIGHT // 2 + 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                C_TXT,
                1,
            )

            f_color = C_OK if self.final_score > 75 else C_WARN
            f_text = (
                "Excellent Performance!"
                if self.final_score > 75
                else "Needs Improvement."
            )
            cv2.putText(
                canvas,
                "ANALYSIS REPORT",
                (WIDTH // 2, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                C_TXT,
                2,
            )
            cv2.putText(
                canvas,
                f"AI Status: {f_text}",
                (WIDTH // 2, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                f_color,
                2,
            )
            cv2.putText(
                canvas,
                "Press 'M' for Menu | 'Q' to Quit",
                (WIDTH // 2, HEIGHT - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                1,
            )

        return canvas

    def run(self):
        while True:
            canvas = self.process()
            if canvas is None:
                break

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if self.state == "LOGIN":
                if key == 13:
                    self.user_name = self.input_buf if self.input_buf else "Guest"
                    self.state = "MENU"
                elif key == 8:
                    self.input_buf = self.input_buf[:-1]
                elif 32 <= key <= 126:
                    self.input_buf += chr(key)

            elif self.state == "MENU":
                if key == ord("w"):
                    self.selected_ex_id = max(0, self.selected_ex_id - 1)
                elif key == ord("s"):
                    self.selected_ex_id = min(4, self.selected_ex_id + 1)
                elif key == 13:
                    self.state = "CALIB"

            elif self.state == "CALIB":
                if key == ord("s"):
                    self.state = "RECORD"
                    self.frame_buffer = []
                    self.full_recording = []
                    self.live_feedback = "Initializing AI..."

            elif self.state == "RECORD":
                if key == ord("s"):
                    self.final_score = self.ai.calculate_dtw_score(
                        self.full_recording, self.selected_ex_id
                    )
                    self.state = "REPORT"

            elif self.state == "REPORT":
                if key == ord("m"):
                    self.state = "MENU"

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = ThesisApp()
    app.run()
