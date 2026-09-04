"""Module 4: Face Verification & Identity Screening Service

Smart India Hackathon - AI-Based Fake Identity & Document Screening System
"""

import json
import os
import time
import cv2
from deepface import DeepFace
import numpy as np

# =====================================================================
# 1. DOCUMENT FACE EXTRACTION (Multi-Backend with Auto-Fallback)
# =====================================================================


def extract_document_face(
    doc_image_path: str, output_path: str = 'extracted_doc_face.jpg'
) -> dict:
  """Detects and crops the face using MTCNN with fallback."""
  if not os.path.exists(doc_image_path):
    return {
        'status': 'ERROR',
        'message': f'File not found: {doc_image_path}',
        'face_path': None,
    }

  img = cv2.imread(doc_image_path)
  if img is None:
    return {
        'status': 'ERROR',
        'message': 'Unable to decode image file',
        'face_path': None,
    }

  h, w, _ = img.shape

  # Try backends in order of reliability
  for backend in ['mtcnn', 'retinaface', 'skip']:
    try:
      face_objs = DeepFace.extract_faces(
          img_path=doc_image_path,
          detector_backend=backend,
          enforce_detection=False,
      )

      if face_objs and len(face_objs) > 0:
        area = face_objs[0].get('facial_area', {})
        x = area.get('x', 0)
        y = area.get('y', 0)
        box_w = area.get('w', w)
        box_h = area.get('h', h)

        if box_w <= 0 or box_h <= 0:
          x, y, box_w, box_h = 0, 0, w, h

        pad_x = int(0.15 * box_w)
        pad_y = int(0.15 * box_h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + box_w + pad_x)
        y2 = min(h, y + box_h + pad_y)

        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
          cv2.imwrite(output_path, cropped)
        else:
          cv2.imwrite(output_path, img)

        return {
            'status': 'SUCCESS',
            'face_path': output_path,
            'bounding_box': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
            },
            'detector_used': backend,
        }
    except Exception:
      continue

  # Fallback to original image if detectors fail
  cv2.imwrite(output_path, img)
  return {
      'status': 'SUCCESS',
      'face_path': output_path,
      'bounding_box': {'x1': 0, 'y1': 0, 'x2': w, 'y2': h},
      'detector_used': 'fallback_full_image',
  }


# =====================================================================
# 2. LIVE WEBCAM CAPTURE & STABILITY CHECK
# =====================================================================


def capture_live_face_from_camera(
    output_path: str = 'live_selfie.jpg', timeout_seconds: int = 15
) -> dict:
  """Opens webcam and captures the passenger photo."""
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    return {
        'status': 'ERROR',
        'captured': False,
        'message': 'Webcam could not be opened',
        'live_photo_path': None,
    }

  print('[INFO] Webcam started. Look directly at the camera...')
  start_time = time.time()
  stable_frames = 0
  captured = False

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
      print('[WARN] Camera capture timed out.')
      break

    cv2.putText(
        frame,
        'Look at camera [Press SPACE to capture]',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f'Auto-capture in: {max(0, int(timeout_seconds - elapsed))}s',
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.imshow('Checkpoint Passenger Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32 or key == ord(' '):  # SPACE bar
      cv2.imwrite(output_path, frame)
      captured = True
      break
    elif key == ord('q'):
      break

    stable_frames += 1
    if stable_frames > 90:  # ~3 seconds
      cv2.imwrite(output_path, frame)
      captured = True
      break

  cap.release()
  cv2.destroyAllWindows()

  return {
      'status': 'SUCCESS' if captured else 'FAILED',
      'captured': captured,
      'live_photo_path': output_path if captured else None,
  }


# =====================================================================
# 3. 1:1 FACE VERIFICATION & SIMILARITY
# =====================================================================


def compare_faces(
    doc_face_path: str, live_face_path: str, model_name: str = 'VGG-Face'
) -> dict:
  """Compares document photo against live selfie using DeepFace."""
  for backend in ['mtcnn', 'retinaface', 'skip']:
    try:
      result = DeepFace.verify(
          img1_path=doc_face_path,
          img2_path=live_face_path,
          model_name=model_name,
          detector_backend=backend,
          distance_metric='cosine',
          enforce_detection=False,
      )

      cosine_distance = float(result.get('distance', 1.0))
      similarity_percentage = max(
          0.0, min(100.0, round((1.0 - cosine_distance) * 100, 2))
      )
      is_match = bool(result.get('verified', False))

      risk_score = 0 if is_match else 40

      return {
          'status': 'SUCCESS',
          'is_match': is_match,
          'similarity_percentage': similarity_percentage,
          'cosine_distance': round(cosine_distance, 4),
          'model_name': model_name,
          'detector_backend': backend,
          'risk_score': risk_score,
      }
    except Exception:
      continue

  return {
      'status': 'ERROR',
      'message': 'Face comparison failed',
      'is_match': False,
      'similarity_percentage': 0.0,
      'risk_score': 50,
  }


# =====================================================================
# 4. MAIN PIPELINE INTERFACE
# =====================================================================


def run_module4_pipeline(
    doc_image_path: str,
    live_image_path: str = None,
    use_live_camera: bool = False,
) -> dict:
  """Runs the complete Module 4 Face Verification workflow."""
  start_t = time.time()

  response = {
      'module': 'Module 4: Face Verification',
      'status': 'FAIL',
      'doc_face_extraction': None,
      'face_matching': None,
      'risk_score': 0,
      'execution_time_seconds': 0.0,
  }

  # Step 1: Crop Document Face
  doc_face_res = extract_document_face(doc_image_path)
  response['doc_face_extraction'] = doc_face_res

  if doc_face_res['status'] != 'SUCCESS':
    response['error'] = doc_face_res['message']
    response['risk_score'] = 50
    response['execution_time_seconds'] = round(time.time() - start_t, 2)
    return response

  # Step 2: Handle Live Photo
  if use_live_camera:
    cam_res = capture_live_face_from_camera()
    if not cam_res['captured']:
      response['error'] = 'Live camera capture failed or was cancelled.'
      response['risk_score'] = 45
      response['execution_time_seconds'] = round(time.time() - start_t, 2)
      return response
    target_live_path = cam_res['live_photo_path']
  else:
    if not live_image_path or not os.path.exists(live_image_path):
      response['error'] = f'Live image path is invalid: {live_image_path}'
      response['risk_score'] = 50
      response['execution_time_seconds'] = round(time.time() - start_t, 2)
      return response
    target_live_path = live_image_path

  # Step 3: Match Faces
  match_res = compare_faces(
      doc_face_path=doc_face_res['face_path'], live_face_path=target_live_path
  )

  response['face_matching'] = match_res
  response['risk_score'] = match_res['risk_score']
  response['status'] = 'SUCCESS' if match_res['is_match'] else 'FLAGGED'
  response['execution_time_seconds'] = round(time.time() - start_t, 2)

  return response


# =====================================================================
# 5. TEST RUNNER
# =====================================================================
if __name__ == '__main__':
  print('=' * 60)
  print('   AI BORDER CHECKPOINT - MODULE 4 TEST RUNNER')
  print('=' * 60)

  SAMPLE_PASSPORT = 'sample_passport.jpg'
  SAMPLE_SELFIE = 'sample_selfie.jpg'

  ENABLE_WEBCAM_TEST = False

  if not ENABLE_WEBCAM_TEST and not os.path.exists(SAMPLE_PASSPORT):
    print(f'[!] Missing test image: "{SAMPLE_PASSPORT}"')
    print('    Please ensure your test passport image is in this folder.')
  else:
    print('[INFO] Running Module 4 verification...')
    output = run_module4_pipeline(
        doc_image_path=SAMPLE_PASSPORT,
        live_image_path=SAMPLE_SELFIE,
        use_live_camera=ENABLE_WEBCAM_TEST,
    )

    print('\n' + '=' * 60)
    print('   VERIFICATION OUTPUT (JSON)')
    print('=' * 60)
    print(json.dumps(output, indent=4))