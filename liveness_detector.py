import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


def get_ear(eye_landmarks, img_w, img_h):
  """Calculate Eye Aspect Ratio (EAR) to detect closed eyes."""
  coords = np.array(
      [[int(p.x * img_w), int(p.y * img_h)] for p in eye_landmarks]
  )

  # Vertical eye distances
  v1 = dist.euclidean(coords[1], coords[5])
  v2 = dist.euclidean(coords[2], coords[4])

  # Horizontal eye distance
  h = dist.euclidean(coords[0], coords[3])

  if h == 0:
    return 0.0
  ear = (v1 + v2) / (2.0 * h)
  return ear


def capture_live_face_with_liveness(
    output_path: str = 'live_selfie.jpg',
    ear_threshold: float = 0.22,
    blink_consecutive_frames: int = 2,
):
  """Opens webcam, verifies user blinks (live human), and captures the selfie."""
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(
      max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
  )

  # MediaPipe landmark indices for left and right eyes
  LEFT_EYE = [33, 160, 158, 133, 153, 144]
  RIGHT_EYE = [362, 385, 387, 263, 373, 380]

  cap = cv2.VideoCapture(0)
  blink_counter = 0
  total_blinks = 0
  captured = False

  print(
      '[INFO] Starting live capture. Please look at the camera and blink to'
      ' verify liveness...'
  )

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    display_text = 'Look at the camera & BLINK'
    text_color = (0, 165, 255)  # Orange

    if results.multi_face_landmarks:
      landmarks = results.multi_face_landmarks[0].landmark

      left_eye_points = [landmarks[idx] for idx in LEFT_EYE]
      right_eye_points = [landmarks[idx] for idx in RIGHT_EYE]

      left_ear = get_ear(left_eye_points, w, h)
      right_ear = get_ear(right_eye_points, w, h)
      avg_ear = (left_ear + right_ear) / 2.0

      if avg_ear < ear_threshold:
        blink_counter += 1
      else:
        if blink_counter >= blink_consecutive_frames:
          total_blinks += 1
          print(f'[INFO] Blink detected! Total blinks: {total_blinks}')
        blink_counter = 0

      if total_blinks >= 1:
        display_text = 'Liveness Confirmed! Capturing...'
        text_color = (0, 255, 0)  # Green
        cv2.imwrite(output_path, frame)
        captured = True

    # Render instructions on frame
    cv2.putText(
        frame,
        display_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
    )
    cv2.putText(
        frame,
        f'Blinks: {total_blinks}',
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imshow('Checkpoint Face Verification', frame)

    # Press 'q' to exit early or wait 1 second after liveness confirmation
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or (captured and total_blinks >= 1):
      cv2.waitKey(500)
      break

  cap.release()
  cv2.destroyAllWindows()

  return {
      'liveness_verified': captured,
      'total_blinks': total_blinks,
      'live_photo_path': output_path if captured else None,
  }