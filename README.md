# AegisPass AI - Module 4: Biometric Face Verification & Anti-Spoofing

Part of the **AI-Based Border Control & Document Screening System** for Smart India Hackathon.

## 📌 Features
1. **Document Face Extraction (`face_extractor.py`)**: Automatically detects, pads (+15%), and crops portrait photos from scanned passports, visas, and IDs.
2. **Anti-Spoofing & Liveness (`liveness_capture.py`)**: Enforces active blink verification using **MediaPipe FaceLandmarker Blendshapes** (`eyeBlinkLeft`, `eyeBlinkRight`) and Eye Aspect Ratio (EAR) to reject printed photo and screen spoofing.
3. **1:1 Biometric Verification (`face_matcher.py`)**: Deep learning face embeddings using **DeepFace (FaceNet)** with Cosine Similarity scoring and Risk Level categorization (`LOW`, `MEDIUM`, `HIGH`).
4. **End-to-End Orchestrator (`test_module4.py`)**: Runs the full pipeline from document scan to clearance verdict.

---

## 🚀 How to Clone & Run

### 1. Clone the Repository
```bash
git clone https://github.com/poojahasini3112-sys/face-verification.git
cd face-verification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Verification Pipeline

**Live Webcam Mode (Interactive Kiosk):**
```bash
python test_module4.py
```

**Fast Mock Mode (Offline validation without webcam):**
```bash
python test_module4.py --mock
```

**Custom Document vs Photo Comparison:**
```bash
python test_module4.py sample_passport.jpg your_photo.jpg
```
