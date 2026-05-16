"""
Utility functions for FaceVision AI - drawing, image processing, and visualization helpers.
"""

import cv2
import numpy as np
from PIL import Image
import io
import time


# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    'primary':    (0, 212, 255),    # Cyan
    'secondary':  (138, 43, 226),   # Purple
    'success':    (0, 255, 136),    # Green
    'warning':    (255, 193, 7),    # Amber
    'danger':     (255, 69, 58),    # Red
    'info':       (52, 152, 219),   # Blue
    'white':      (255, 255, 255),
    'dark':       (20, 20, 40),
}

# BGR versions for OpenCV
COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in COLORS.items()}


def load_image(uploaded_file):
    """Load an uploaded file into OpenCV format (BGR numpy array)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return image


def cv2_to_pil(image):
    """Convert OpenCV BGR image to PIL RGB image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(pil_image):
    """Convert PIL RGB image to OpenCV BGR image."""
    rgb = np.array(pil_image)
    if len(rgb.shape) == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_image(image, max_dim=1024):
    """Resize image if larger than max_dim while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_fancy_bbox(image, bbox, label="", confidence=None, color=COLORS_BGR['primary'], thickness=2):
    """
    Draw a stylish bounding box with corner accents and optional label.
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    overlay = image.copy()

    # Semi-transparent fill
    cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, image, 0.92, 0, image)

    # Main rectangle
    cv2.rectangle(image, (x, y), (x2, y2), color, thickness)

    # Corner accents (L-shaped brackets)
    corner_len = min(w, h) // 4
    corner_thickness = thickness + 1

    # Top-left
    cv2.line(image, (x, y), (x + corner_len, y), color, corner_thickness)
    cv2.line(image, (x, y), (x, y + corner_len), color, corner_thickness)
    # Top-right
    cv2.line(image, (x2, y), (x2 - corner_len, y), color, corner_thickness)
    cv2.line(image, (x2, y), (x2, y + corner_len), color, corner_thickness)
    # Bottom-left
    cv2.line(image, (x, y2), (x + corner_len, y2), color, corner_thickness)
    cv2.line(image, (x, y2), (x, y2 - corner_len), color, corner_thickness)
    # Bottom-right
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, corner_thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, corner_thickness)

    # Label background and text
    if label or confidence is not None:
        text_parts = []
        if label:
            text_parts.append(label)
        if confidence is not None:
            text_parts.append(f"{confidence:.1%}")
        text = " | ".join(text_parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        # Label box above the bounding box
        label_y = max(y - 8, th + 8)
        cv2.rectangle(image, (x, label_y - th - 8), (x + tw + 12, label_y + 4), color, -1)
        cv2.putText(image, text, (x + 6, label_y - 2), font, font_scale,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image


def draw_detections(image, detections, color=COLORS_BGR['primary'], label_prefix="Face"):
    """Draw all detections on an image with fancy bounding boxes."""
    result = image.copy()
    for i, det in enumerate(detections):
        bbox = det['bbox']
        conf = det.get('confidence')
        label = f"{label_prefix} {i + 1}"
        if det.get('name'):
            label = det['name']
        result = draw_fancy_bbox(result, bbox, label=label, confidence=conf, color=color)
    return result


def draw_landmarks(image, landmarks_list, color=COLORS_BGR['primary'], radius=1):
    """Draw facial landmarks on the image."""
    result = image.copy()
    for landmarks in landmarks_list:
        for (px, py) in landmarks:
            cv2.circle(result, (px, py), radius, color, -1, cv2.LINE_AA)
    return result


def draw_landmarks_mesh(image, landmarks_list):
    """Draw facial landmarks with connecting mesh lines for a futuristic look."""
    result = image.copy()
    overlay = result.copy()

    # Key landmark indices for face mesh connections (simplified)
    jaw_line = list(range(0, 17))
    left_eyebrow = list(range(17, 22))
    right_eyebrow = list(range(22, 27))

    for landmarks in landmarks_list:
        # Draw all points
        for (px, py) in landmarks:
            cv2.circle(overlay, (px, py), 1, COLORS_BGR['primary'], -1, cv2.LINE_AA)

        # Draw connections between nearby points for mesh effect
        pts = np.array(landmarks)
        if len(pts) > 1:
            for i in range(len(pts)):
                for j in range(i + 1, min(i + 5, len(pts))):
                    dist = np.linalg.norm(pts[i] - pts[j])
                    if dist < 30:
                        alpha_color = COLORS_BGR['primary']
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[j]),
                                 alpha_color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    return result


def create_comparison_grid(images, titles, cols=3):
    """Create a comparison grid of images with titles."""
    rows = (len(images) + cols - 1) // cols
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    grid = np.zeros((rows * (max_h + 40), cols * max_w, 3), dtype=np.uint8)
    grid[:] = (20, 20, 40)  # Dark background

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        y_offset = r * (max_h + 40) + 40
        x_offset = c * max_w

        # Resize to fit
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Center in cell
        y_start = y_offset + (max_h - new_h) // 2
        x_start = x_offset + (max_w - new_w) // 2
        grid[y_start:y_start + new_h, x_start:x_start + new_w] = resized

        # Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(title, font, 0.6, 1)
        tx = x_offset + (max_w - tw) // 2
        cv2.putText(grid, title, (tx, r * (max_h + 40) + 28),
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return grid


def save_temp_image(image, prefix="temp"):
    """Save image to a temporary file and return the path."""
    import tempfile
    import os
    temp_dir = os.path.join(os.path.dirname(__file__), '.temp')
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"{prefix}_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(path, image)
    return path


def get_face_crop(image, bbox, padding=0.2):
    """Crop a face from the image with padding."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Add padding
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)

    return image[y1:y2, x1:x2]


def format_time(seconds):
    """Format processing time for display."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"
