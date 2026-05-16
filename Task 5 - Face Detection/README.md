# Task 5 — 🎭 FaceVision AI — Face Detection & Recognition System

> **CodSoft AI Internship | Sreeja Pollreddy | BY25RY287818**

A comprehensive AI-powered face detection and recognition application built with Python. It leverages multiple detection backends (Haar Cascades, OpenCV DNN, MediaPipe), **ArcFace** embeddings for recognition, and **DeepFace** for facial attribute analysis — all wrapped in a polished Streamlit interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Multi-Method Detection** | Compare Haar Cascade, DNN SSD, and MediaPipe detectors side-by-side with performance benchmarks |
| 👤 **Face Registration** | Build a persistent face database with multiple images per person |
| 🎯 **ArcFace Recognition** | Identify registered individuals using state-of-the-art ArcFace embeddings |
| 🧬 **Attribute Analysis** | Estimate age, gender, dominant emotion, and ethnicity |
| ⚡ **Face Verification** | Compare two images to verify whether they show the same person |
| 🗺️ **468-Point Face Mesh** | Detailed facial landmark visualization via MediaPipe |

---

## 🛠️ Tech Stack

- **Face Detection**: OpenCV Haar Cascades, OpenCV DNN (SSD), MediaPipe
- **Face Recognition**: DeepFace with ArcFace model
- **Face Analysis**: DeepFace (age, gender, emotion, race)
- **Landmark Detection**: MediaPipe Face Mesh (468 points)
- **Frontend**: Streamlit with custom glassmorphic UI
- **Visualization**: Plotly (radar charts, bar charts)

---

## 📁 Project Structure

```
Task 5 - Face Detection/
├── app.py                 # Main Streamlit application
├── face_detector.py       # Multi-backend face detection engine
├── face_recognizer.py     # ArcFace-based face recognition & verification
├── face_analyzer.py       # Facial attribute analysis (age/gender/emotion)
├── utils.py               # Drawing utilities, image processing helpers
├── styles.py              # Custom CSS for premium UI
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
├── models/                # Auto-downloaded detection models
└── database/
    └── known_faces/       # Registered face images (organized by person)
```

---

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> **Note**: On first run, pre-trained models (SSD face detector, ArcFace, DeepFace analysis models) will be downloaded automatically. This may take a few minutes depending on your connection.

---

## 🔬 How It Works

### Detection Pipeline

1. **Haar Cascade** — Classical ML approach using Viola-Jones algorithm with cascaded classifiers trained on positive/negative face samples. Fast but sensitive to orientation.
2. **DNN SSD** — A Single Shot MultiBox Detector (SSD) with a ResNet-10 backbone, trained on face data. Runs inference via OpenCV's DNN module for high accuracy.
3. **MediaPipe** — Google's lightweight ML pipeline optimized for mobile/edge inference. Uses a BlazeFace-based architecture for robust real-time detection.

### Recognition Pipeline

- Faces are encoded into 512-dimensional embeddings using **ArcFace** (Additive Angular Margin Loss), a metric-learning approach that maximizes inter-class variance in hyperspherical embedding space.
- Recognition is performed via cosine distance comparison against the registered face database.

### Analysis Pipeline

- **Age**: Regression model estimating apparent age
- **Gender**: Binary classification (Man/Woman)
- **Emotion**: 7-class classification (angry, disgust, fear, happy, sad, surprise, neutral)
- **Ethnicity**: Multi-class estimation

---

## 📸 Usage Guide

1. **Home** — Overview of system capabilities and database stats.
2. **Face Detection** — Upload an image → select method → view results with bounding boxes, confidence scores, and optional face mesh overlay.
3. **Face Registry** — Enter a name → upload/capture a photo → register. Multiple photos per person improve accuracy.
4. **Recognition** — Upload an image → the system identifies any registered faces with similarity scores.
5. **Face Analysis** — Upload an image → view estimated age, gender, emotion radar chart, and ethnicity.
6. **Face Compare** — Upload two images → get a same-person verification with similarity percentage.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing, Haar/DNN detection |
| `mediapipe` | Face detection & 468-point mesh |
| `deepface` | ArcFace recognition, facial analysis |
| `streamlit` | Interactive web interface |
| `plotly` | Performance & emotion visualizations |
| `tf-keras` | Backend for DeepFace models |

---

## 📜 License

Part of the CodSoft AI Virtual Internship Program.

---
