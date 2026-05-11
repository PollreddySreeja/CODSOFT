"""
Neural Image Captioner – Streamlit Web Application
===================================================

A premium interface for generating natural-language captions
from images using deep learning.

Two captioning backends:
  1. Custom ResNet-50 + LSTM-Attention (if trained checkpoint exists)
  2. BLIP (Salesforce) via HuggingFace Transformers (always available)

Run:
    streamlit run app.py
"""

import os
import io
import time
import base64
import requests
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config

# ─────────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Image Captioner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Custom CSS – dark premium theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* global overrides */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #121228 40%, #0d1b2a 100%);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    border-right: 1px solid rgba(100,100,255,0.15);
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }

/* hero */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: -1px;
    animation: glow 3s ease-in-out infinite alternate;
}
@keyframes glow {
    from { filter: drop-shadow(0 0 8px rgba(102,126,234,0.3)); }
    to   { filter: drop-shadow(0 0 20px rgba(118,75,162,0.5)); }
}
.hero-sub {
    font-size: 1.1rem;
    color: #8888bb;
    text-align: center;
    margin-top: -8px;
    margin-bottom: 2rem;
}

/* caption display card */
.caption-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.caption-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
}
.caption-text {
    font-size: 1.35rem;
    font-weight: 500;
    color: #e8e8ff;
    line-height: 1.7;
    letter-spacing: 0.3px;
}
.caption-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #667eea;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* beam result card */
.beam-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.3s ease;
}
.beam-card:hover {
    border-color: rgba(102,126,234,0.4);
    background: rgba(102,126,234,0.05);
    transform: translateX(4px);
}
.beam-caption {
    color: #d0d0f0;
    font-size: 1.0rem;
    font-weight: 400;
}
.beam-score {
    color: #667eea;
    font-size: 0.85rem;
    font-weight: 600;
}

/* stats */
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.75rem;
    color: #8888bb;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* upload zone */
.upload-zone {
    border: 2px dashed rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    background: rgba(102,126,234,0.03);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: rgba(102,126,234,0.6);
    background: rgba(102,126,234,0.06);
}

/* image container */
.img-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #d0d0f0;
    margin: 2rem 0 1rem;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(102,126,234,0.2);
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Model loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_blip_model():
    """Load BLIP captioning model from HuggingFace."""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_NAME)
    model.eval()
    return processor, model


@st.cache_resource
def load_custom_model():
    """Load custom ResNet+LSTM model if checkpoint exists."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    vocab_path = config.VOCAB_PATH

    if not os.path.exists(ckpt_path) or not os.path.exists(vocab_path):
        return None, None

    from src.vocabulary import Vocabulary
    from src.caption_model import ImageCaptioner

    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)

    model = ImageCaptioner(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        attention_dim=config.ATTENTION_DIM,
        decoder_dim=config.HIDDEN_DIM,
        encoder_dim=config.ENCODER_DIM,
        encoded_size=config.ENCODED_SIZE,
        dropout=0.0,
    ).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


# ─────────────────────────────────────────────────────────────
#  Captioning functions
# ─────────────────────────────────────────────────────────────
def caption_with_blip(image, processor, model, max_length=60,
                      num_beams=5, num_return=1):
    """Generate captions using BLIP."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return,
            early_stopping=True,
        )
    captions = []
    for ids in output_ids:
        cap = processor.decode(ids, skip_special_tokens=True)
        captions.append(cap)
    return captions


def caption_with_custom(image, model, vocab, beam_width=5):
    """Generate captions using custom model."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    img_tensor = transform(image).to(config.DEVICE)

    # greedy
    greedy_cap, greedy_attn = model.generate_caption(img_tensor, vocab)

    # beam search
    beam_results = model.beam_search_caption(
        img_tensor, vocab, beam_width=beam_width
    )

    return greedy_cap, greedy_attn, beam_results


# ─────────────────────────────────────────────────────────────
#  Attention visualisation
# ─────────────────────────────────────────────────────────────
def create_attention_overlay(image, attention_maps, caption_words, encoded_size=7):
    """Create attention heatmap overlay for each word."""
    image = image.resize((224, 224))
    img_array = np.array(image)

    num_words = min(len(caption_words), len(attention_maps), 12)
    if num_words == 0:
        return None

    cols = min(4, num_words)
    rows = (num_words + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.patch.set_facecolor("#0a0a1a")

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(num_words):
        ax = axes[i]
        attn = attention_maps[i]
        if isinstance(attn, torch.Tensor):
            attn = attn.numpy()
        attn = attn.reshape(encoded_size, encoded_size)

        # upsample attention to image size
        attn_img = Image.fromarray(attn).resize((224, 224), Image.BILINEAR)
        attn_array = np.array(attn_img)

        ax.imshow(img_array)
        ax.imshow(attn_array, alpha=0.55, cmap="magma")
        ax.set_title(caption_words[i], fontsize=11, fontweight="bold",
                     color="#e8e8ff", pad=6)
        ax.axis("off")

    for j in range(num_words, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=1.0)
    return fig


def create_full_attention_heatmap(image, attention_maps, encoded_size=7):
    """Create a single aggregated attention heatmap."""
    image = image.resize((224, 224))
    img_array = np.array(image)

    # average all attention maps
    all_attn = []
    for attn in attention_maps:
        if isinstance(attn, torch.Tensor):
            attn = attn.numpy()
        all_attn.append(attn.reshape(encoded_size, encoded_size))

    avg_attn = np.mean(all_attn, axis=0)
    attn_img = Image.fromarray(avg_attn).resize((224, 224), Image.BILINEAR)
    attn_array = np.array(attn_img)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor("#0a0a1a")
    ax.imshow(img_array)
    ax.imshow(attn_array, alpha=0.5, cmap="inferno")
    ax.set_title("Aggregated Attention", fontsize=14, fontweight="bold",
                 color="#e8e8ff")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  Image loading helpers
# ─────────────────────────────────────────────────────────────
def load_image_from_url(url):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


# demo images (publicly available, royalty-free)
DEMO_IMAGES = [
    {
        "name": "🏖️ Beach Sunset",
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=640",
        "desc": "A serene beach at sunset"
    },
    {
        "name": "🐕 Dog in Park",
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=640",
        "desc": "A happy dog playing in a park"
    },
    {
        "name": "🏙️ City Skyline",
        "url": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=640",
        "desc": "Modern city skyline at dusk"
    },
    {
        "name": "🍕 Food Plate",
        "url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=640",
        "desc": "A delicious pizza on a plate"
    },
    {
        "name": "⛰️ Mountain Lake",
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
        "desc": "Mountain landscape with a lake"
    },
]


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # model selection
        custom_model, custom_vocab = load_custom_model()
        custom_available = custom_model is not None

        model_options = ["BLIP (Pre-trained)"]
        if custom_available:
            model_options.append("Custom ResNet+LSTM")

        selected_model = st.selectbox(
            "Captioning Model",
            model_options,
            help="BLIP is always available. Custom model requires training first."
        )

        st.markdown("---")
        st.markdown("### 🎛️ Generation Settings")

        num_beams = st.slider("Beam Width", 1, 10, 5,
                              help="Higher = more diverse candidates but slower")
        max_length = st.slider("Max Caption Length", 10, 80, 50)

        if selected_model == "BLIP (Pre-trained)":
            num_return = st.slider("Number of Captions", 1, 5, 3)
        else:
            num_return = num_beams

        st.markdown("---")
        st.markdown("### 📊 Architecture")

        with st.expander("Model Details", expanded=False):
            if selected_model == "BLIP (Pre-trained)":
                st.markdown("""
                **BLIP** (Bootstrapping Language-Image Pre-training)
                - Vision Transformer encoder
                - Multimodal mixture of encoder-decoder
                - Pre-trained on 14M image-text pairs
                - Fine-tuned for captioning
                """)
            else:
                st.markdown(f"""
                **Custom Architecture**
                - Encoder: ResNet-50 (ImageNet)
                - Attention: Bahdanau (Additive)
                - Decoder: LSTM ({config.HIDDEN_DIM}d hidden)
                - Embedding: {config.EMBED_DIM}d
                - Vocabulary: {len(custom_vocab)} words
                """)

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#555; font-size:0.75rem;'>"
            "Built with PyTorch & Streamlit<br>"
            "ResNet-50 + LSTM + Attention</div>",
            unsafe_allow_html=True
        )

    return selected_model, num_beams, max_length, num_return, custom_model, custom_vocab


# ─────────────────────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────────────────────
def main():
    # hero header
    st.markdown('<h1 class="hero-title">Neural Image Captioner</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">'
        'Deep learning meets language — generating rich descriptions '
        'from visual content using ResNet-50 encoder with LSTM attention decoder'
        '</p>',
        unsafe_allow_html=True
    )

    # sidebar config
    (selected_model, num_beams, max_length,
     num_return, custom_model, custom_vocab) = render_sidebar()

    # ── image input section ──────────────────────────────────
    st.markdown('<div class="section-header">📸 Input Image</div>',
                unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Paste URL", "Demo Gallery"],
        horizontal=True, label_visibility="collapsed"
    )

    image = None

    if input_method == "Upload Image":
        uploaded = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supports JPEG, PNG, WebP, BMP formats"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    elif input_method == "Paste URL":
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/photo.jpg"
        )
        if url:
            image = load_image_from_url(url)

    else:  # Demo Gallery
        cols = st.columns(len(DEMO_IMAGES))
        for i, demo in enumerate(DEMO_IMAGES):
            with cols[i]:
                if st.button(demo["name"], key=f"demo_{i}", use_container_width=True):
                    with st.spinner("Loading demo image..."):
                        image = load_image_from_url(demo["url"])
                        if image:
                            st.session_state["demo_image"] = image

        if "demo_image" in st.session_state:
            image = st.session_state["demo_image"]

    if image is None:
        st.info("👆 Upload an image, paste a URL, or pick a demo to get started.")
        return

    # ── display image & generate ─────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # image stats
        w, h = image.size
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{w}×{h}</div>'
                    f'<div class="stat-label">Resolution</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value">'
                    f'{image.mode}</div><div class="stat-label">Color Mode</div></div>',
                    unsafe_allow_html=True)
        size_kb = len(image.tobytes()) // 1024
        c3.markdown(f'<div class="stat-box"><div class="stat-value">{size_kb}KB</div>'
                    f'<div class="stat-label">Size</div></div>',
                    unsafe_allow_html=True)

    with col_result:
        generate_btn = st.button("🚀 Generate Caption", type="primary",
                                 use_container_width=True)

        if generate_btn:
            with st.spinner("Analysing image..."):
                start_time = time.time()

                if selected_model == "BLIP (Pre-trained)":
                    processor, blip_model = load_blip_model()
                    captions = caption_with_blip(
                        image, processor, blip_model,
                        max_length=max_length,
                        num_beams=num_beams,
                        num_return=num_return
                    )
                    elapsed = time.time() - start_time

                    # primary caption
                    st.markdown(
                        f'<div class="caption-card">'
                        f'<div class="caption-label">Generated Caption</div>'
                        f'<div class="caption-text">"{captions[0]}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # beam candidates
                    if len(captions) > 1:
                        st.markdown(
                            '<div class="section-header">🔍 Alternative Captions</div>',
                            unsafe_allow_html=True
                        )
                        for j, cap in enumerate(captions[1:], 2):
                            st.markdown(
                                f'<div class="beam-card">'
                                f'<div class="beam-score">Candidate #{j}</div>'
                                f'<div class="beam-caption">"{cap}"</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                else:
                    # custom model
                    greedy_cap, greedy_attn, beam_results = caption_with_custom(
                        image, custom_model, custom_vocab, beam_width=num_beams
                    )
                    elapsed = time.time() - start_time

                    st.markdown(
                        f'<div class="caption-card">'
                        f'<div class="caption-label">Generated Caption (Greedy)</div>'
                        f'<div class="caption-text">"{greedy_cap}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    if beam_results:
                        st.markdown(
                            '<div class="section-header">🔍 Beam Search Results</div>',
                            unsafe_allow_html=True
                        )
                        for j, result in enumerate(beam_results):
                            score_pct = min(100, max(0, (result["score"] + 10) * 5))
                            st.markdown(
                                f'<div class="beam-card">'
                                f'<div class="beam-score">#{j+1} · '
                                f'Score: {result["score"]:.3f}</div>'
                                f'<div class="beam-caption">"{result["caption"]}"</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    # attention visualisation
                    if greedy_attn:
                        st.markdown(
                            '<div class="section-header">'
                            '🔥 Attention Heatmaps</div>',
                            unsafe_allow_html=True
                        )

                        tab1, tab2 = st.tabs([
                            "Per-Word Attention", "Aggregated Heatmap"
                        ])
                        with tab1:
                            fig = create_attention_overlay(
                                image, greedy_attn, greedy_cap.split()
                            )
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)

                        with tab2:
                            fig2 = create_full_attention_heatmap(
                                image, greedy_attn
                            )
                            if fig2:
                                st.pyplot(fig2, use_container_width=True)
                                plt.close(fig2)

                # inference time
                st.markdown(
                    f'<div class="stat-box" style="margin-top:16px;">'
                    f'<div class="stat-value">{elapsed:.2f}s</div>'
                    f'<div class="stat-label">Inference Time</div></div>',
                    unsafe_allow_html=True
                )

    # ── architecture section ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>',
                unsafe_allow_html=True)

    arch_col1, arch_col2, arch_col3 = st.columns(3)

    with arch_col1:
        st.markdown("""
        <div class="stat-box">
        <div class="stat-value" style="font-size:1.4rem;">ResNet-50</div>
        <div class="stat-label">Image Encoder</div>
        <div style="color:#8888bb; font-size:0.8rem; margin-top:8px;">
        Pre-trained on ImageNet · Extracts 2048-d spatial features
        from a 7×7 grid (49 feature vectors per image)
        </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_col2:
        st.markdown("""
        <div class="stat-box">
        <div class="stat-value" style="font-size:1.4rem;">Bahdanau</div>
        <div class="stat-label">Attention Mechanism</div>
        <div style="color:#8888bb; font-size:0.8rem; margin-top:8px;">
        Additive attention with learnable gate · Produces per-word
        heatmaps showing which image regions inform each word
        </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_col3:
        st.markdown("""
        <div class="stat-box">
        <div class="stat-value" style="font-size:1.4rem;">LSTM</div>
        <div class="stat-label">Caption Decoder</div>
        <div style="color:#8888bb; font-size:0.8rem; margin-top:8px;">
        512-d hidden state · Teacher forcing during training ·
        Beam search with length penalty for inference
        </div>
        </div>
        """, unsafe_allow_html=True)

    # pipeline diagram
    st.markdown("""
    <div style="text-align:center; margin:30px 0; padding:20px;
                background:rgba(255,255,255,0.02); border-radius:12px;
                border:1px solid rgba(255,255,255,0.06);">
    <span style="color:#667eea; font-weight:600;">Input Image</span>
    <span style="color:#555; margin:0 12px;">→</span>
    <span style="color:#764ba2; font-weight:600;">ResNet-50 Encoder</span>
    <span style="color:#555; margin:0 12px;">→</span>
    <span style="color:#f093fb; font-weight:600;">Spatial Features (7×7×2048)</span>
    <span style="color:#555; margin:0 12px;">→</span>
    <span style="color:#667eea; font-weight:600;">Attention + LSTM</span>
    <span style="color:#555; margin:0 12px;">→</span>
    <span style="color:#764ba2; font-weight:600;">Caption Tokens</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
