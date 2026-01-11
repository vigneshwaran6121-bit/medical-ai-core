import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Rehab System", layout="wide")

# =========================
# SESSION STATE INIT
# =========================
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"
if "selected_exercise" not in st.session_state:
    st.session_state["selected_exercise"] = None
if "run_exercise" not in st.session_state:
    st.session_state["run_exercise"] = False

# =========================
# CONFIG & CONSTANTS
# =========================
# Assuming you have 5 videos. For testing, we use the one you provided for all options.
# Replace these filenames with your actual 5 expert video files.
EXERCISES = {
    "Exercise 1: Arm Raise": "WIN_20250823_16_19_36_Pro.mp4",
    "Exercise 2: Elbow Flex": "WIN_20250823_16_19_36_Pro.mp4",
    "Exercise 3: Shoulder Abduction": "WIN_20250823_16_19_36_Pro.mp4",
    "Exercise 4: Wrist Flexion": "WIN_20250823_16_19_36_Pro.mp4",
    "Exercise 5: Neck Rotation": "WIN_20250823_16_19_36_Pro.mp4",
}

DTW_WINDOW = 60
ALPHA = 0.15
DTW_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]


# =========================
# HELPER FUNCTIONS
# =========================
def extract_pose(res):
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    pts = np.array([[lm[i].x, lm[i].y] for i in DTW_JOINTS])
    center = (pts[-1] + pts[-2]) / 2
    pts = pts - center
    scale = np.linalg.norm(pts[1] - pts[2]) + 1e-6
    pts = pts / scale
    return pts.flatten()


def resize_frame(img, width, height):
    return cv2.resize(img, (width, height))


def draw_skeleton_on_black(landmarks, w, h):
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    if landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            black_frame,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec((255, 255, 255), 2, 2),
        )
    return black_frame


# =========================
# PAGE 1: LOGIN
# =========================
def show_login():
    st.title("🏥 AI Rehabilitation System")
    st.markdown("### Welcome to your recovery journey")

    col1, col2 = st.columns([1, 2])
    with col1:
        user_input = st.text_input("Username")
        if st.button("Login"):
            if user_input:
                st.session_state["username"] = user_input
                st.session_state["page"] = "select"
                st.rerun()
            else:
                st.warning("Please enter a username")

        st.markdown("---")
        if st.button("Continue as Guest"):
            st.session_state["username"] = "Guest"
            st.session_state["page"] = "select"
            st.rerun()


# =========================
# PAGE 2: SELECTION
# =========================
def show_selection():
    st.title(f"Hello, {st.session_state['username']}")
    st.subheader("Select an Exercise")

    # We use a standard selectbox for web navigation, but we can bind keys if needed.
    # For a web app, clicking is usually better than 'w'/'s' keys.
    options = list(EXERCISES.keys())
    choice = st.selectbox("Choose your therapy module:", options)

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back to Login"):
            st.session_state["page"] = "login"
            st.rerun()
    with col2:
        if st.button("Start Exercise ➡", type="primary"):
            st.session_state["selected_exercise"] = EXERCISES[choice]
            st.session_state["page"] = "app"
            st.rerun()


# =========================
# PAGE 3: MAIN APP
# =========================
def show_app():
    # Header
    st.markdown(f"### Performing: {st.session_state['selected_exercise']}")
    if st.button("Stop & Return to Menu"):
        st.session_state["page"] = "select"
        st.rerun()

    # Placeholders for the video stream
    frame_placeholder = st.empty()

    # Init Logic
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap_e = cv2.VideoCapture(st.session_state["selected_exercise"])
    cap_p = cv2.VideoCapture(0)  # 0 is Webcam

    if not cap_p.isOpened():
        st.error("Cannot access webcam.")
        return

    # Storage vars
    seq_e, seq_p = [], []
    curve_e, curve_p = [], []
    ema_e, ema_p = 0.0, 0.0

    # Layout Dimensions (Fixed for the constructed image)
    TOTAL_W, TOTAL_H = 1280, 720

    # Left Side (60% width) -> Split 30:30 means two equal halves of the 60%
    LEFT_W = int(TOTAL_W * 0.6)
    SKEL_W = int(LEFT_W / 2)  # Width of one skeleton window
    SKEL_H = TOTAL_H  # Full height

    # Right Side (40% width)
    RIGHT_W = TOTAL_W - LEFT_W

    # Right Top (60% height) -> Real Videos
    RIGHT_TOP_H = int(TOTAL_H * 0.6)
    REAL_VID_W = int(RIGHT_W / 2)  # Split real videos side by side

    # Right Bottom (40% height) -> Graph
    RIGHT_BOT_H = TOTAL_H - RIGHT_TOP_H

    while cap_p.isOpened():
        ret_e, frame_e = cap_e.read()
        ret_p, frame_p = cap_p.read()

        # Loop expert video
        if not ret_e:
            cap_e.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_e, frame_e = cap_e.read()

        if not ret_p:
            st.error("Webcam disconnected")
            break

        # Flip patient and convert
        frame_p = cv2.flip(frame_p, 1)

        # Process Pose
        frame_e_rgb = cv2.cvtColor(frame_e, cv2.COLOR_BGR2RGB)
        frame_p_rgb = cv2.cvtColor(frame_p, cv2.COLOR_BGR2RGB)

        res_e = pose.process(frame_e_rgb)
        res_p = pose.process(frame_p_rgb)

        # -----------------------------------
        # 1. CREATE SKELETON VIEWS (Left 60%)
        # -----------------------------------
        skel_e_img = draw_skeleton_on_black(res_e.pose_landmarks, SKEL_W, SKEL_H)
        skel_p_img = draw_skeleton_on_black(res_p.pose_landmarks, SKEL_W, SKEL_H)

        # Add labels
        cv2.putText(
            skel_e_img,
            "Expert Skeleton",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            skel_p_img,
            "Patient Skeleton",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # -----------------------------------
        # 2. CREATE REAL VIDEO VIEWS (Right Top)
        # -----------------------------------
        real_e_resized = resize_frame(frame_e, REAL_VID_W, RIGHT_TOP_H)
        real_p_resized = resize_frame(frame_p, REAL_VID_W, RIGHT_TOP_H)

        # -----------------------------------
        # 3. DTW & FEEDBACK LOGIC
        # -----------------------------------
        dtw_score = 0.0
        feedback_text = "Aligning..."
        feedback_color = (200, 200, 200)

        pe = extract_pose(res_e)
        pp = extract_pose(res_p)

        if pe is not None and pp is not None:
            seq_e.append(pe)
            seq_p.append(pp)
            seq_e = seq_e[-DTW_WINDOW:]
            seq_p = seq_p[-DTW_WINDOW:]

            d, _ = fastdtw(seq_e, seq_p, dist=euclidean)
            dtw_score = 1 / (1 + d / len(seq_e))

            # Smoothing for graph
            if len(seq_e) > 1:
                motion_e = np.clip(np.linalg.norm(seq_e[-1] - seq_e[-2]), 0, 1.2)
                motion_p = np.clip(np.linalg.norm(seq_p[-1] - seq_p[-2]), 0, 1.2)
            else:
                motion_e, motion_p = 0.0, 0.0

            ema_e = ALPHA * motion_e + (1 - ALPHA) * ema_e
            ema_p = ALPHA * motion_p + (1 - ALPHA) * ema_p

            curve_e.append(ema_e)
            curve_p.append(ema_p)
            curve_e = curve_e[-RIGHT_W:]  # Width of graph area
            curve_p = curve_p[-RIGHT_W:]

            # Feedback Logic
            if dtw_score > 0.8:
                feedback_text = "Excellent!"
                feedback_color = (0, 255, 0)
            elif dtw_score > 0.5:
                feedback_text = "Good"
                feedback_color = (0, 255, 255)
            else:
                feedback_text = "Improve"
                feedback_color = (0, 0, 255)

        # -----------------------------------
        # 4. CREATE GRAPH/PANEL (Right Bottom)
        # -----------------------------------
        panel = np.zeros((RIGHT_BOT_H, RIGHT_W, 3), dtype=np.uint8)

        # Draw text
        cv2.putText(
            panel,
            f"Score: {dtw_score:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            panel,
            f"{feedback_text}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            feedback_color,
            3,
        )

        # Draw curves
        center_y = int(RIGHT_BOT_H / 2) + 20
        for i in range(1, len(curve_e)):
            start_pt = (i - 1, int(center_y - curve_e[i - 1] * 100))
            end_pt = (i, int(center_y - curve_e[i] * 100))
            # Check bounds to avoid crash
            if 0 <= start_pt[0] < RIGHT_W and 0 <= end_pt[0] < RIGHT_W:
                cv2.line(panel, start_pt, end_pt, (0, 0, 255), 2)

        for i in range(1, len(curve_p)):
            start_pt = (i - 1, int(center_y - curve_p[i - 1] * 100))
            end_pt = (i, int(center_y - curve_p[i] * 100))
            if 0 <= start_pt[0] < RIGHT_W and 0 <= end_pt[0] < RIGHT_W:
                cv2.line(panel, start_pt, end_pt, (255, 0, 0), 2)

        # -----------------------------------
        # 5. STITCH EVERYTHING
        # -----------------------------------
        # Left Block (60% W)
        left_block = np.hstack((skel_e_img, skel_p_img))

        # Right Top Block
        right_top_block = np.hstack((real_e_resized, real_p_resized))

        # Right Block (40% W)
        right_block = np.vstack((right_top_block, panel))

        # Final Assembly
        final_layout = np.hstack((left_block, right_block))

        # Convert to RGB for Streamlit
        final_rgb = cv2.cvtColor(final_layout, cv2.COLOR_BGR2RGB)

        # Display in Streamlit
        frame_placeholder.image(final_rgb, channels="RGB", use_container_width=True)

    cap_e.release()
    cap_p.release()


# =========================
# MAIN ROUTER
# =========================
if st.session_state["page"] == "login":
    show_login()
elif st.session_state["page"] == "select":
    show_selection()
elif st.session_state["page"] == "app":
    show_app()
