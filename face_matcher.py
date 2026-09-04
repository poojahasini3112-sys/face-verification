from deepface import DeepFace


def compare_faces(
    doc_face_path: str,
    live_face_path: str,
    model_name: str = 'ArcFace',
    threshold: float = 0.60,
):
  """Compares two face images and returns similarity metrics."""
  try:
    result = DeepFace.verify(
        img1_path=doc_face_path,
        img2_path=live_face_path,
        model_name=model_name,
        distance_metric='cosine',
        enforce_detection=False,
    )

    cosine_distance = result.get('distance', 1.0)
    # Cosine similarity score (0 to 100%)
    similarity_score = max(0.0, min(100.0, round((1 - cosine_distance) * 100, 2)))
    is_match = bool(result.get('verified', False))

    # Calculate module-specific risk contribution
    risk_score = 0 if is_match else 40

    return {
        'status': 'SUCCESS',
        'is_match': is_match,
        'similarity_percentage': similarity_score,
        'cosine_distance': round(cosine_distance, 4),
        'model_used': model_name,
        'risk_score': risk_score,
    }

  except Exception as e:
    return {
        'status': 'ERROR',
        'message': str(e),
        'is_match': False,
        'similarity_percentage': 0.0,
        'risk_score': 50,
    }