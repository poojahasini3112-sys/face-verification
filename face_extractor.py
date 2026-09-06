import os
import cv2
import numpy as np

def extract_face_from_document(doc_image_path: str, output_crop_path: str = "temp_doc_face.jpg") -> np.ndarray:
    """
    Detects and crops the portrait photo from an identity document.
    """
    image = cv2.imread(doc_image_path)
    if image is None:
        raise ValueError(f"Could not load image from {doc_image_path}")

    # Locate Haar Cascade XML with robust fallbacks
    cascade_paths = [
        getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
    ]
    face_cascade = None
    for cp in cascade_paths:
        if os.path.exists(cp):
            fc = cv2.CascadeClassifier(cp)
            if not fc.empty():
                face_cascade = fc
                break

    if face_cascade is None or face_cascade.empty():
        # Auto-download cascade file if missing from OpenCV wheel
        import urllib.request
        fallback_file = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        try:
            urllib.request.urlretrieve(url, fallback_file)
            face_cascade = cv2.CascadeClassifier(fallback_file)
        except Exception:
            pass

    if face_cascade is None or face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar Cascade face detector.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Fallback to lower minNeighbors if initial detection finds nothing
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 50)
        )

    if len(faces) == 0:
        raise ValueError("No face detected on the identity document. Ensure good lighting and orientation.")

    # Pick the largest detected face (the main portrait photo)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Add a 15% padding around the face for better embedding quality
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    
    h_img, w_img, _ = image.shape
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    cropped_face = image[y1:y2, x1:x2]
    
    # Save the cropped face for inspection
    cv2.imwrite(output_crop_path, cropped_face)
    return cropped_face