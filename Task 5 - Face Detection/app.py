"""
FaceVision AI — Face Detection & Recognition System
Built with OpenCV DNN, MediaPipe, and DeepFace (ArcFace).
Streamlit-based interactive interface.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import tempfile

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_analyzer import FaceAnalyzer
from utils import (
    load_image, cv2_to_pil, resize_image, draw_detections,
    draw_landmarks, draw_landmarks_mesh, save_temp_image,
    get_face_crop, format_time, COLORS_BGR
)
from styles import CUSTOM_CSS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVision AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Singletons ────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    return FaceDetector()

@st.cache_resource
def get_recognizer():
    return FaceRecognizer()

@st.cache_resource
def get_analyzer():
    return FaceAnalyzer()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎭 FaceVision AI")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Face Detection", "📹 Video Detection",
         "👤 Face Registry", "🎯 Recognition", "🧬 Face Analysis",
         "⚡ Face Compare"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.75rem;'>"
        "FaceVision AI v1.0<br>OpenCV • MediaPipe • ArcFace</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def render_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">FaceVision AI</div>
        <div class="hero-subtitle">
            A powerful face detection and recognition system powered by
            Haar Cascades, Deep Neural Networks (SSD), MediaPipe, and ArcFace embeddings.
            Detect, recognize, and analyze faces with state-of-the-art accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    recognizer = get_recognizer()
    stats = recognizer.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("3", "Detection Models"),
        ("ArcFace", "Recognition Engine"),
        (str(stats['total_people']), "Registered Faces"),
        ("468", "Facial Landmarks"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Features grid
    st.markdown('<div class="section-header">✨ Capabilities</div>', unsafe_allow_html=True)
    features = [
        ("🔍", "Multi-Method Detection",
         "Compare Haar Cascades, DNN SSD, and MediaPipe detectors side by side."),
        ("👤", "Face Registration",
         "Build a face database — register people with multiple photos for robust matching."),
        ("🎯", "ArcFace Recognition",
         "Identify registered individuals using state-of-the-art ArcFace embeddings."),
        ("🧬", "Attribute Analysis",
         "Estimate age, gender, dominant emotion, and ethnicity from facial features."),
        ("⚡", "Face Comparison",
         "Verify whether two photos show the same person with a similarity score."),
        ("🗺️", "468-Point Face Mesh",
         "Visualize detailed facial landmarks with a futuristic mesh overlay."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 3].markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FACE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def render_detection():
    st.markdown('<div class="section-header">🔍 Face Detection Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload an image and compare three detection methods in real-time.</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                key="det_upload")
    if uploaded is None:
        st.info("👆 Upload an image to begin face detection.")
        return

    image = load_image(uploaded)
    image = resize_image(image, 1024)
    detector = get_detector()

    # Controls
    col_a, col_b, col_c = st.columns(3)
    method = col_a.selectbox("Detection Method",
                             ["All Methods (Compare)", "Haar Cascade", "DNN SSD", "MediaPipe"])
    confidence = col_b.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    show_landmarks = col_c.checkbox("Show Face Mesh Landmarks", value=False)

    if st.button("🚀 Detect Faces", use_container_width=True):
        with st.spinner("Running detection..."):
            if method == "All Methods (Compare)":
                _render_comparison(image, detector, confidence, show_landmarks)
            else:
                _render_single_detection(image, detector, method, confidence, show_landmarks)


def _render_single_detection(image, detector, method, conf, show_landmarks):
    if method == "Haar Cascade":
        faces, elapsed = detector.detect_haar(image)
    elif method == "DNN SSD":
        faces, elapsed = detector.detect_dnn(image, conf)
    else:
        faces, elapsed = detector.detect_mediapipe(image, conf)

    result_img = draw_detections(image, faces)

    if show_landmarks:
        landmarks, lm_time = detector.get_landmarks(image)
        result_img = draw_landmarks_mesh(result_img, landmarks)
        elapsed += lm_time

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card"><div class="metric-value">{len(faces)}</div>
        <div class="metric-label">Faces Detected</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-value">{format_time(elapsed)}</div>
        <div class="metric-label">Processing Time</div></div>""", unsafe_allow_html=True)
    fps = 1.0 / elapsed if elapsed > 0 else 0
    c3.markdown(f"""<div class="metric-card"><div class="metric-value">{fps:.1f}</div>
        <div class="metric-label">Effective FPS</div></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown("**Original**")
    col1.image(cv2_to_pil(image), use_container_width=True)
    col2.markdown(f"**{method} Result**")
    col2.image(cv2_to_pil(result_img), use_container_width=True)

    if faces:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("**🔎 Detected Face Crops**")
        crop_cols = st.columns(min(len(faces), 5))
        for i, face in enumerate(faces):
            crop = get_face_crop(image, face['bbox'])
            if crop.size > 0:
                crop_cols[i % len(crop_cols)].image(cv2_to_pil(crop), caption=f"Face {i+1}",
                                                     use_container_width=True)


def _render_comparison(image, detector, conf, show_landmarks):
    results = detector.detect_all_methods(image, conf)

    # Metrics row
    cols = st.columns(3)
    for col, (name, data) in zip(cols, results.items()):
        n = len(data['faces'])
        t = format_time(data['time'])
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n}</div>
            <div class="metric-label">{name} — {t}</div>
        </div>""", unsafe_allow_html=True)

    # Side-by-side images
    img_cols = st.columns(3)
    colors = [COLORS_BGR['primary'], COLORS_BGR['success'], COLORS_BGR['secondary']]
    for i, (col, (name, data)) in enumerate(zip(img_cols, results.items())):
        vis = draw_detections(image, data['faces'], color=colors[i])
        if show_landmarks:
            lms, _ = detector.get_landmarks(image)
            vis = draw_landmarks_mesh(vis, lms)
        col.markdown(f"**{name}**")
        col.image(cv2_to_pil(vis), use_container_width=True)

    # Performance chart
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("**📊 Performance Comparison**")
    names = list(results.keys())
    times = [results[n]['time'] * 1000 for n in names]
    counts = [len(results[n]['faces']) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=times, name="Time (ms)",
                         marker_color=['#00d4ff', '#00ff88', '#7b2fff']))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Milliseconds"
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FACE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
def render_registry():
    st.markdown('<div class="section-header">👤 Face Registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Register faces to build your recognition database.</div>',
                unsafe_allow_html=True)

    recognizer = get_recognizer()
    tab1, tab2 = st.tabs(["📥 Register New Face", "📋 Registered Faces"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            name = st.text_input("Person's Name", placeholder="e.g., Sreeja")
            uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"],
                                        key="reg_upload")
            camera_img = st.camera_input("Or capture from camera", key="reg_camera")

        source_img = uploaded or camera_img
        with col2:
            if source_img:
                image = load_image(source_img)
                image = resize_image(image, 640)

                # Show detection preview
                detector = get_detector()
                faces, _ = detector.detect_dnn(image, 0.5)
                preview = draw_detections(image, faces)
                st.image(cv2_to_pil(preview), caption=f"{len(faces)} face(s) detected",
                         use_container_width=True)

        if st.button("✅ Register Face", use_container_width=True, disabled=not (name and source_img)):
            if name and source_img:
                image = load_image(source_img)
                image = resize_image(image, 640)
                path = recognizer.register_face(image, name.strip())
                st.success(f"✅ **{name}** registered successfully!")
                st.balloons()

    with tab2:
        people = recognizer.get_registered_people()
        if not people:
            st.info("No faces registered yet. Go to the Register tab to add some!")
            return

        st.markdown(f"**{len(people)} registered people** with "
                    f"**{sum(len(v) for v in people.values())} total images**")

        cols = st.columns(4)
        for i, (person_name, images) in enumerate(people.items()):
            with cols[i % 4]:
                img = cv2.imread(images[0])
                if img is not None:
                    st.image(cv2_to_pil(img), use_container_width=True)
                st.markdown(f"""<div class="person-card">
                    <div class="person-name">{person_name}</div>
                    <div class="person-count">{len(images)} image(s)</div>
                </div>""", unsafe_allow_html=True)

                if st.button(f"🗑️ Delete", key=f"del_{person_name}"):
                    recognizer.delete_person(person_name)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
def render_recognition():
    st.markdown('<div class="section-header">🎯 Face Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Identify registered faces in uploaded images using ArcFace embeddings.</div>',
                unsafe_allow_html=True)

    recognizer = get_recognizer()
    people = recognizer.get_registered_people()

    if not people:
        st.warning("⚠️ No faces in the database! Register faces in the **Face Registry** first.")
        return

    st.markdown(f'<span class="result-badge badge-info">Database: {len(people)} people registered</span>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image to identify", type=["jpg", "jpeg", "png"],
                                key="recog_upload")
    if uploaded is None:
        return

    image = load_image(uploaded)
    image = resize_image(image, 800)
    temp_path = save_temp_image(image, "recog")

    if st.button("🔍 Identify Faces", use_container_width=True):
        with st.spinner("Running ArcFace recognition..."):
            result = recognizer.recognize_face(temp_path)

        if 'error' in result:
            st.error(f"Recognition error: {result['error']}")
        elif result.get('matches'):
            matches = result['matches']
            st.markdown(f'<span class="result-badge badge-success">'
                        f'{len(matches)} face(s) identified</span>', unsafe_allow_html=True)

            det = get_detector()
            faces, _ = det.detect_dnn(image, 0.4)

            # Assign names to detected faces based on matches
            for i, face in enumerate(faces):
                if i < len(matches):
                    face['name'] = matches[i]['name']
                    face['confidence'] = matches[i]['similarity'] / 100.0

            vis = draw_detections(image, faces, color=COLORS_BGR['success'])
            st.image(cv2_to_pil(vis), caption="Recognition Result", use_container_width=True)

            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
            for match in matches:
                c1, c2 = st.columns([1, 3])
                c1.markdown(f"### {match['name']}")
                sim = match['similarity']
                color = "#00ff88" if sim > 60 else "#ff9500" if sim > 40 else "#ff453a"
                c2.markdown(f"""
                <div class="similarity-bar-bg">
                    <div class="similarity-bar-fill" style="width:{sim:.0f}%">{sim:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No matches found in the database.")

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FACE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def render_analysis():
    st.markdown('<div class="section-header">🧬 Face Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Analyze age, gender, emotion, and ethnicity from facial features.</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="ana_upload")
    if uploaded is None:
        st.info("👆 Upload an image to analyze facial attributes.")
        return

    image = load_image(uploaded)
    image = resize_image(image, 800)

    actions = st.multiselect("Select analyses", ["age", "gender", "emotion", "race"],
                             default=["age", "gender", "emotion", "race"])

    if st.button("🧬 Analyze Faces", use_container_width=True):
        temp_path = save_temp_image(image, "analysis")
        analyzer = get_analyzer()

        with st.spinner("Analyzing facial attributes..."):
            results = analyzer.analyze(temp_path, actions)

        if isinstance(results, dict) and 'error' in results:
            st.error(f"Analysis error: {results['error']}")
        elif isinstance(results, list):
            st.markdown(f'<span class="result-badge badge-success">'
                        f'{len(results)} face(s) analyzed</span>', unsafe_allow_html=True)

            for idx, face_result in enumerate(results):
                st.markdown(f'<div class="styled-divider"></div>', unsafe_allow_html=True)
                st.markdown(f"### Face {idx + 1}")

                summary = analyzer.get_dominant_attributes(face_result)

                # Attribute cards
                attr_cols = st.columns(4)
                attr_data = [
                    ("🎂", "Age", str(summary.get('age', '—')), "years"),
                    ("⚧", "Gender", summary.get('gender', '—'), ""),
                    ("😊", "Emotion", summary.get('emotion', '—'), ""),
                    ("🌍", "Ethnicity", summary.get('ethnicity', '—'), ""),
                ]
                for col, (icon, label, value, unit) in zip(attr_cols, attr_data):
                    col.markdown(f"""<div class="attr-card">
                        <div style="font-size:1.6rem">{icon}</div>
                        <div class="attr-value">{value}</div>
                        <div class="attr-label">{label} {unit}</div>
                    </div>""", unsafe_allow_html=True)

                # Emotion radar chart
                if 'emotion' in face_result:
                    emotions = face_result['emotion']
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=list(emotions.values()),
                        theta=[e.capitalize() for e in emotions.keys()],
                        fill='toself',
                        fillcolor='rgba(0,212,255,0.15)',
                        line=dict(color='#00d4ff', width=2),
                        marker=dict(size=6, color='#00d4ff')
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(visible=True, range=[0, 100],
                                            gridcolor='rgba(255,255,255,0.1)'),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        ),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=380, margin=dict(l=60, r=60, t=30, b=30),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FACE COMPARE
# ══════════════════════════════════════════════════════════════════════════════
def render_compare():
    st.markdown('<div class="section-header">⚡ Face Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Verify if two photos show the same person using ArcFace verification.</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📸 Image 1**")
        img1_file = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="cmp1")
        if img1_file:
            img1 = load_image(img1_file)
            img1 = resize_image(img1, 500)
            st.image(cv2_to_pil(img1), use_container_width=True)
    with col2:
        st.markdown("**📸 Image 2**")
        img2_file = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="cmp2")
        if img2_file:
            img2 = load_image(img2_file)
            img2 = resize_image(img2, 500)
            st.image(cv2_to_pil(img2), use_container_width=True)

    if img1_file and img2_file:
        if st.button("⚡ Compare Faces", use_container_width=True):
            path1 = save_temp_image(img1, "cmp1")
            path2 = save_temp_image(img2, "cmp2")
            recognizer = get_recognizer()

            with st.spinner("Running ArcFace verification..."):
                result = recognizer.verify_faces(path1, path2)

            if 'error' in result:
                st.error(f"Verification error: {result['error']}")
            else:
                verified = result['verified']
                sim = result['similarity']
                badge_class = "badge-success" if verified else "badge-danger"
                label = "✅ SAME PERSON" if verified else "❌ DIFFERENT PEOPLE"

                st.markdown(f"""
                <div style="text-align:center;margin:24px 0;">
                    <span class="result-badge {badge_class}" style="font-size:1.2rem;padding:12px 32px;">
                        {label}
                    </span>
                </div>
                <div class="similarity-bar-bg" style="max-width:600px;margin:20px auto;">
                    <div class="similarity-bar-fill" style="width:{max(sim,5):.0f}%">{sim:.1f}% Similar</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class="metric-card"><div class="metric-value">{sim:.1f}%</div>
                    <div class="metric-label">Similarity</div></div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="metric-card"><div class="metric-value">{result['distance']:.4f}</div>
                    <div class="metric-label">Distance</div></div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="metric-card"><div class="metric-value">{result['threshold']:.4f}</div>
                    <div class="metric-label">Threshold</div></div>""", unsafe_allow_html=True)

            for p in [path1, path2]:
                if os.path.exists(p):
                    os.remove(p)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def render_video():
    st.markdown('<div class="section-header">📹 Video Face Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a video to detect faces frame-by-frame using deep learning.</div>',
                unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_upload"
    )
    if uploaded_video is None:
        st.info("👆 Upload a video file (MP4, AVI, MOV, MKV, or WebM) to begin.")
        return

    # Controls
    col_a, col_b, col_c = st.columns(3)
    method = col_a.selectbox("Detection Method", ["DNN SSD", "Haar Cascade", "MediaPipe"], key="vid_method")
    confidence = col_b.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="vid_conf")
    skip_frames = col_c.slider("Process every N-th frame", 1, 10, 1, 1, key="vid_skip",
                                help="Higher = faster processing but less smooth. 1 = every frame.")

    # Preview original
    st.video(uploaded_video)

    if st.button("🚀 Process Video", use_container_width=True):
        detector = get_detector()

        # Save uploaded video to temp file
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.temp')
        os.makedirs(temp_dir, exist_ok=True)
        input_path = os.path.join(temp_dir, f"input_{int(time.time())}.mp4")
        output_path = os.path.join(temp_dir, f"output_{int(time.time())}.mp4")

        with open(input_path, 'wb') as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file. Please try a different format.")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Limit resolution for performance
        max_dim = 720
        scale = 1.0
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        progress_bar = st.progress(0, text="Processing video...")
        status_text = st.empty()
        frame_preview = st.empty()

        total_faces_detected = 0
        frames_processed = 0
        start_time = time.perf_counter()
        last_detections = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))

            # Detect faces (skip frames for speed)
            if frame_idx % skip_frames == 0:
                try:
                    if method == "Haar Cascade":
                        last_detections, _ = detector.detect_haar(frame)
                    elif method == "DNN SSD":
                        last_detections, _ = detector.detect_dnn(frame, confidence)
                    else:
                        last_detections, _ = detector.detect_mediapipe(frame, confidence)
                except Exception:
                    last_detections = []
                frames_processed += 1

            # Draw detections on every frame (using last results)
            from utils import draw_detections as dd, cv2_to_pil as c2p, COLORS_BGR as CB
            annotated = dd(frame, last_detections, color=CB['primary'])
            total_faces_detected += len(last_detections)
            writer.write(annotated)

            # Update progress
            progress = min((frame_idx + 1) / max(total_frames, 1), 1.0)
            progress_bar.progress(progress, text=f"Processing frame {frame_idx + 1}/{total_frames}")

            # Show preview every 15 frames
            if frame_idx % 15 == 0:
                frame_preview.image(c2p(annotated), caption=f"Frame {frame_idx + 1}",
                                    use_container_width=True)

            frame_idx += 1

        cap.release()
        writer.release()
        elapsed = time.perf_counter() - start_time

        progress_bar.progress(1.0, text="✅ Processing complete!")
        frame_preview.empty()

        # Metrics
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{frame_idx}</div>'
                    f'<div class="metric-label">Total Frames</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{format_time(elapsed)}</div>'
                    f'<div class="metric-label">Total Time</div></div>', unsafe_allow_html=True)
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{avg_fps:.1f}</div>'
                    f'<div class="metric-label">Processing FPS</div></div>', unsafe_allow_html=True)
        avg_faces = total_faces_detected / max(frames_processed, 1)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{avg_faces:.1f}</div>'
                    f'<div class="metric-label">Avg Faces/Frame</div></div>', unsafe_allow_html=True)

        # Show processed video
        st.markdown("**🎬 Processed Video**")
        st.video(output_path)

        # Download button
        with open(output_path, 'rb') as vf:
            st.download_button(
                label="📥 Download Processed Video",
                data=vf,
                file_name="facevision_detected.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        # Cleanup input temp
        if os.path.exists(input_path):
            os.remove(input_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page_map = {
    "🏠 Home":           render_home,
    "🔍 Face Detection": render_detection,
    "📹 Video Detection": render_video,
    "👤 Face Registry":  render_registry,
    "🎯 Recognition":    render_recognition,
    "🧬 Face Analysis":  render_analysis,
    "⚡ Face Compare":   render_compare,
}

page_map[page]()
