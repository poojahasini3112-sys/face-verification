import os
import sys
import time
import urllib.request
import cv2
import numpy as np

# Landmark indices for EAR calculation (standard 468/478 mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """Calculates Eye Aspect Ratio (EAR) to detect eye closure."""
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
    d_v1 = np.linalg.norm(pts[1] - pts[5])
    d_v2 = np.linalg.norm(pts[2] - pts[4])
    d_h = np.linalg.norm(pts[0] - pts[3])
    if d_h == 0:
        return 0.0
    return float((d_v1 + d_v2) / (2.0 * d_h))

def _ensure_landmarker_task() -> str:
    """Ensures face_landmarker.task model file is downloaded for MediaPipe Tasks."""
    local_task = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if os.path.exists(local_task) and os.path.getsize(local_task) > 100000:
        return local_task

    print("Downloading MediaPipe face landmarker model (~3.7MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, local_task)
    return local_task

class FaceDetectorLiveness:
    """
    Unified liveness detector supporting:
    1. MediaPipe Tasks 1.0+ (FaceLandmarker with blendshapes & EAR)
    2. Legacy MediaPipe solutions (0.10.x)
    3. OpenCV Haar Cascade fallback
    """
    def __init__(self):
        self.mode = "none"
        self.landmarker = None
        self.legacy_mesh = None

        # 1. Try MediaPipe Tasks 1.0+
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            task_file = _ensure_landmarker_task()
            base_opts = mp_python.BaseOptions(model_asset_path=task_file)
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
            self.mp = mp
            self.mode = "tasks"
            return
        except Exception as e:
            pass

        # 2. Try MediaPipe legacy solutions
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.legacy_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mode = "legacy"
                return
        except Exception:
            pass

        # 3. Fallback to OpenCV Haar Cascade
        self.mode = "opencv"
        cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_frame(self, frame_bgr):
        """
        Returns: (face_found: bool, is_blinking: bool, avg_ear: float)
        """
        h, w, _ = frame_bgr.shape

        if self.mode == "tasks":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_image)

            if not res.face_landmarks or len(res.face_landmarks) == 0:
                return False, False, 0.30

            landmarks = res.face_landmarks[0]
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Also check blendshape blink score if available
            blink_blendshape = False
            if res.face_blendshapes and len(res.face_blendshapes) > 0:
                for b in res.face_blendshapes[0]:
                    if b.category_name in ("eyeBlinkLeft", "eyeBlinkRight") and b.score > 0.45:
                        blink_blendshape = True

            is_closed = (avg_ear < 0.20) or blink_blendshape
            return True, is_closed, avg_ear

        elif self.mode == "legacy":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.legacy_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return False, False, 0.30

            landmarks = res.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            return True, (avg_ear < 0.20), avg_ear

        else:
            # OpenCV mode
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                return False, False, 0.30
            return True, False, 0.28


def capture_live_face_with_liveness(
    output_path: str = "temp_live_face.jpg", 
    timeout_seconds: int = 15,
    mock_image_path: str = None
):
    """
    Opens webcam with anti-spoofing challenge (Natural Blink).
    Captures live frame once verified.
    """
    # If mock image is provided (for automated testing without webcam)
    if mock_image_path and os.path.exists(mock_image_path):
        print(f"  [Mock Mode] Simulating live capture from: {mock_image_path}")
        img = cv2.imread(mock_image_path)
        cv2.imwrite(output_path, img)
        return True, output_path

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [Warning] Webcam could not be opened. Check webcam connection or privacy settings.")
        return False, None

    detector = FaceDetectorLiveness()
    blink_detected = False
    eye_was_closed = False
    start_time = time.time()
    saved_frame = None

    print("\n--- LOOK AT THE WEBCAM AND BLINK TO VERIFY LIVENESS ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for intuitive user interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        elapsed = time.time() - start_time
        remaining = max(0, int(timeout_seconds - elapsed))

        if elapsed > timeout_seconds:
            break

        face_found, is_closed, ear_val = detector.process_frame(frame)

        status_text = "Please center face and BLINK your eyes"
        status_color = (0, 165, 255) # Orange

        if face_found:
            if is_closed:
                eye_was_closed = True
            elif eye_was_closed and not is_closed:
                # User closed eyes then opened them -> genuine live blink!
                blink_detected = True
                eye_was_closed = False
                saved_frame = frame.copy()
                status_text = "Liveness Verified! Capturing photo..."
                status_color = (0, 255, 0) # Green

        # Draw Border Security Kiosk HUD
        # Header bar
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, "BORDER SECURITY E-GATE - BIOMETRIC VERIFICATION", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status footer
        cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"STATUS: {status_text}", (15, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"Timeout: {remaining}s", (w - 150, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Oval framing guide for passenger
        center = (w // 2, h // 2)
        axes = (w // 6, h // 4)
        guide_color = (0, 255, 0) if blink_detected else (0, 165, 255) if face_found else (100, 100, 100)
        cv2.ellipse(frame, center, axes, 0, 0, 360, guide_color, 2)

        cv2.imshow("Border Checkpoint - Biometric Liveness Verification", frame)

        if blink_detected:
            cv2.waitKey(600)
            break

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if blink_detected and saved_frame is not None:
        cv2.imwrite(output_path, saved_frame)
        return True, output_path
    else:
        return False, None