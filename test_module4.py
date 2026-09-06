import os
import sys

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from face_extractor import extract_face_from_document
from liveness_capture import capture_live_face_with_liveness
from face_matcher import verify_faces

def run_face_verification_pipeline(passport_image_path: str, mock_live_image: str = None):
    print("====================================================")
    print("      BORDER SECURITY BIOMETRIC VERIFICATION        ")
    print("====================================================")

    # 1. Extract face from the passport/ID
    print("\n[Step 1/3] Extracting portrait photo from document...")
    doc_face_crop = "extracted_doc_face.jpg"
    try:
        extract_face_from_document(passport_image_path, doc_face_crop)
        print("  [PASS] Document face successfully isolated and saved.")
    except Exception as e:
        print(f"  [FAIL] Error extracting face from document: {e}")
        return {
            "verdict": "ERROR_DOCUMENT_EXTRACTION_FAILED",
            "error": str(e),
            "liveness_verified": False,
            "face_match": False
        }

    # 2. Perform live webcam capture + liveness check
    print("\n[Step 2/3] Initiating live face capture with anti-spoofing...")
    live_face_file = "live_captured_face.jpg"
    liveness_passed, live_path = capture_live_face_with_liveness(
        live_face_file, 
        timeout_seconds=15,
        mock_image_path=mock_live_image
    )

    if not liveness_passed:
        print("  [FAIL] LIVENESS FAILED! Possible photo spoof or user timed out.")
        return {
            "verdict": "REJECTED_SPOOF_ATTEMPT",
            "liveness_verified": False,
            "face_match": False
        }
    print("  [PASS] Liveness confirmed via natural blink.")

    # 3. Compare the two faces
    print("\n[Step 3/3] Running 1:1 facial biometric matching...")
    verification_result = verify_faces(doc_face_crop, live_face_file, model_name="Facenet")

    # 4. Final Assessment
    is_match = verification_result.get("is_match", False)
    similarity = verification_result.get("similarity_score", 0.0)
    risk_level = verification_result.get("risk_level", "HIGH" if not is_match else "LOW")
    decision = verification_result.get("decision", "IMPOSTER_ALERT_DETAIN" if not is_match else "CLEARED")

    print("\n---------------- FINAL BIOMETRIC VERDICT ----------------")
    print(f"Liveness Check   : {'PASSED [OK]' if liveness_passed else 'FAILED [X]'}")
    print(f"Match Status     : {'MATCH [OK]' if is_match else 'NO MATCH [X]'}")
    print(f"Similarity Score : {similarity}%")
    print(f"Distance Score   : {verification_result.get('distance')} (Threshold: {verification_result.get('threshold')})")
    print(f"Risk Assessment  : {risk_level} RISK")
    print(f"Officer Action   : {decision}")

    if is_match and liveness_passed:
        print("RESULT: PASSENGER CLEARED")
    else:
        print("RESULT: IDENTITY FRAUD / IMPOSTER ALERT")
    print("---------------------------------------------------------")

    return {
        "verdict": "PASSENGER_CLEARED" if (is_match and liveness_passed) else "IDENTITY_FRAUD_ALERT",
        "liveness_verified": liveness_passed,
        "face_match": is_match,
        "similarity_score": similarity,
        "risk_level": risk_level,
        "officer_action": decision
    }

if __name__ == "__main__":
    sample_doc = "sample_passport.jpg"
    mock_mode = False
    mock_img = None

    # CLI Arguments Support:
    # python test_module4.py --mock
    # python test_module4.py <doc_path> <live_path>
    if len(sys.argv) > 1:
        if "--mock" in sys.argv:
            mock_mode = True
        else:
            sample_doc = sys.argv[1]
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
                mock_img = sys.argv[2]

    if not os.path.exists(sample_doc):
        print(f"Error: Scanned document '{sample_doc}' not found in folder.")
    else:
        if mock_mode:
            print("[INFO] Running in mock validation mode (simulating live feed using document face).")
            mock_img = sample_doc
        run_face_verification_pipeline(sample_doc, mock_live_image=mock_img)