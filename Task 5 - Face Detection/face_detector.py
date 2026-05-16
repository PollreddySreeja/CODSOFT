"""
FaceVision AI - Face Detection Engine
Supports multiple detection backends: Haar Cascade, OpenCV DNN (SSD), and MediaPipe.
"""

import cv2
import numpy as np
import os
import urllib.request
import time


class FaceDetector:
    """
    Multi-backend face detector supporting Haar Cascades, OpenCV DNN (SSD),
    and MediaPipe face detection.
    """

    def __init__(self):
        # Haar Cascade (built into OpenCV)
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.haar_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # DNN model (loaded lazily)
        self._dnn_net = None
        self._model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

        # MediaPipe (loaded lazily)
        self._mp_face_detection = None
        self._mp_face_mesh = None

    # ── DNN Model Management ──────────────────────────────────────────────────

    def _ensure_dnn_model(self):
        """Download and load the SSD face detection model if not already loaded."""
        if self._dnn_net is not None:
            return

        os.makedirs(self._model_dir, exist_ok=True)
        prototxt = os.path.join(self._model_dir, 'deploy.prototxt')
        caffemodel = os.path.join(self._model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        if not os.path.exists(prototxt):
            print("[FaceDetector] Downloading SSD prototxt...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt',
                prototxt
            )

        if not os.path.exists(caffemodel):
            print("[FaceDetector] Downloading SSD Caffe model (~10MB)...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/opencv/opencv_3rdparty/'
                'dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
                caffemodel
            )

        self._dnn_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print("[FaceDetector] SSD model loaded successfully.")

    def _ensure_mediapipe(self):
        """Import MediaPipe lazily."""
        if self._mp_face_detection is None:
            import mediapipe as mp
            self._mp_face_detection = mp.solutions.face_detection
            self._mp_face_mesh = mp.solutions.face_mesh

    # ── Detection Methods ─────────────────────────────────────────────────────

    def detect_haar(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces using Haar Cascade classifier.
        Classic ML approach — fast but less accurate.
        """
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        elapsed = time.perf_counter() - start

        results = []
        for (x, y, w, h) in faces:
            # Detect eyes within the face region for validation
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.haar_eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            results.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'confidence': None,
                'eyes_detected': len(eyes),
                'method': 'Haar Cascade'
            })

        return results, elapsed

    def detect_dnn(self, image, confidence_threshold=0.5):
        """
        Detect faces using OpenCV's DNN module with a pre-trained SSD model.
        Deep learning approach — more accurate and robust.
        """
        self._ensure_dnn_model()
        start = time.perf_counter()

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()
        elapsed = time.perf_counter() - start

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                results.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence,
                    'method': 'DNN SSD'
                })

        return results, elapsed

    def detect_mediapipe(self, image, confidence_threshold=0.5):
        """
        Detect faces using Google's MediaPipe Face Detection.
        Lightweight ML approach — great balance of speed and accuracy.
        """
        self._ensure_mediapipe()
        start = time.perf_counter()

        with self._mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence_threshold
        ) as face_detection:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_results = face_detection.process(rgb)

        elapsed = time.perf_counter() - start
        results = []

        if mp_results.detections:
            h, w = image.shape[:2]
            for detection in mp_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = min(int(bbox.width * w), w - x)
                bh = min(int(bbox.height * h), h - y)

                # Extract keypoints
                keypoints = {}
                for kp_id, kp in enumerate(detection.location_data.relative_keypoints):
                    keypoints[kp_id] = (int(kp.x * w), int(kp.y * h))

                results.append({
                    'bbox': (x, y, bw, bh),
                    'confidence': float(detection.score[0]),
                    'keypoints': keypoints,
                    'method': 'MediaPipe'
                })

        return results, elapsed

    # ── Landmark Detection ────────────────────────────────────────────────────

    def get_landmarks(self, image, max_faces=10):
        """
        Detect detailed facial landmarks using MediaPipe Face Mesh.
        Returns up to 468 landmark points per face.
        """
        self._ensure_mediapipe()
        start = time.perf_counter()

        with self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(rgb)

        elapsed = time.perf_counter() - start
        landmarks_list = []

        if mp_results.multi_face_landmarks:
            h, w = image.shape[:2]
            for face_landmarks in mp_results.multi_face_landmarks:
                points = []
                for lm in face_landmarks.landmark:
                    points.append((int(lm.x * w), int(lm.y * h)))
                landmarks_list.append(points)

        return landmarks_list, elapsed

    # ── Multi-Method Detection ────────────────────────────────────────────────

    def detect_all_methods(self, image, conf_threshold=0.5):
        """
        Run all three detection methods and return comparative results.
        Useful for benchmarking and method comparison.
        """
        results = {}

        try:
            haar_faces, haar_time = self.detect_haar(image)
            results['Haar Cascade'] = {'faces': haar_faces, 'time': haar_time}
        except Exception as e:
            results['Haar Cascade'] = {'faces': [], 'time': 0, 'error': str(e)}

        try:
            dnn_faces, dnn_time = self.detect_dnn(image, conf_threshold)
            results['DNN SSD'] = {'faces': dnn_faces, 'time': dnn_time}
        except Exception as e:
            results['DNN SSD'] = {'faces': [], 'time': 0, 'error': str(e)}

        try:
            mp_faces, mp_time = self.detect_mediapipe(image, conf_threshold)
            results['MediaPipe'] = {'faces': mp_faces, 'time': mp_time}
        except Exception as e:
            results['MediaPipe'] = {'faces': [], 'time': 0, 'error': str(e)}

        return results
