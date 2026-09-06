from deepface import DeepFace

def verify_faces(doc_face_path: str, live_face_path: str, model_name: str = "Facenet"):
    """
    Compares two face images using 1:1 biometric matching.
    Supported models: 'Facenet', 'VGG-Face', 'ArcFace'.
    """
    try:
        # DeepFace handles alignment, normalization, and distance computation.
        # Since images may already be tightly cropped face portraits, enforce_detection is set to False
        # to prevent false negative rejections on pre-cropped images.
        result = DeepFace.verify(
            img1_path=doc_face_path,
            img2_path=live_face_path,
            model_name=model_name,
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False
        )

        distance = float(result["distance"])
        threshold = float(result["threshold"])
        is_match = bool(result["verified"])

        # Convert cosine distance into an intuitive 0-100% confidence score
        if distance <= threshold:
            # Scale similarity from 70% to 100% when under threshold
            similarity_pct = round((1.0 - (distance / threshold)) * 30 + 70, 2)
        else:
            # Scale down towards 0%
            similarity_pct = round(max(0.0, (1.0 - distance) * 70), 2)

        # Risk level determination for Border Security Screening
        if similarity_pct >= 75:
            risk_level = "LOW"
            decision = "CLEARED"
        elif similarity_pct >= 60:
            risk_level = "MEDIUM"
            decision = "SECONDARY_INSPECTION_RECOMMENDED"
        else:
            risk_level = "HIGH"
            decision = "IMPOSTER_ALERT_DETAIN"

        return {
            "is_match": is_match,
            "similarity_score": similarity_pct,
            "distance": round(distance, 4),
            "threshold": round(threshold, 4),
            "risk_level": risk_level,
            "decision": decision,
            "model_used": model_name
        }

    except Exception as e:
        return {
            "is_match": False,
            "similarity_score": 0.0,
            "error": str(e)
        }