import os
import cv2


def extract_document_face(
    doc_image_path: str, output_path: str = 'extracted_doc_face.jpg'
):
  """Detects and crops the face from an ID card or passport image."""
  # Load the image
  img = cv2.imread(doc_image_path)
  if img is None:
    return {'status': 'ERROR', 'message': 'Failed to read document image'}

  # Convert to grayscale for Haar Cascade
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Load OpenCV's built-in face detector
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
  )

  # Detect faces
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
  )

  if len(faces) == 0:
    return {
        'status': 'ERROR',
        'message': 'No face found on document',
        'face_path': None,
    }

  # Take the most prominent face (largest area)
  x, y, w, h = max(faces, key=lambda item: item[2] * item[3])

  # Add slight padding around the face
  padding_x = int(0.15 * w)
  padding_y = int(0.15 * h)
  y1 = max(0, y - padding_y)
  y2 = min(img.shape[0], y + h + padding_y)
  x1 = max(0, x - padding_x)
  x2 = min(img.shape[1], x + w + padding_x)

  cropped_face = img[y1:y2, x1:x2]

  # Save cropped face
  cv2.imwrite(output_path, cropped_face)

  return {
      'status': 'SUCCESS',
      'face_path': output_path,
      'bbox': [int(x1), int(y1), int(x2), int(y2)],
  }