"""
Configuration module for the Image Captioning system.
Centralizes all hyperparameters and path settings so nothing
is scattered across files.
"""

import os
import torch

# ── paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
FLICKR8K    = os.path.join(DATA_DIR, "flickr8k")
IMAGE_DIR   = os.path.join(FLICKR8K, "Images")
CAPTION_FILE = os.path.join(FLICKR8K, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VOCAB_PATH  = os.path.join(CHECKPOINT_DIR, "vocabulary.pkl")

# ── encoder (ResNet-50) ─────────────────────────────────────
ENCODER_DIM   = 2048        # output channels of ResNet-50 layer4
ENCODED_SIZE  = 7           # spatial size after adaptive pool (7×7)

# ── decoder ──────────────────────────────────────────────────
EMBED_DIM     = 300         # word-embedding dimension
ATTENTION_DIM = 256         # attention hidden layer size
HIDDEN_DIM    = 512         # LSTM hidden state size
DROPOUT       = 0.5

# ── vocabulary ───────────────────────────────────────────────
MIN_WORD_FREQ = 3           # words appearing fewer times are <UNK>
MAX_CAPTION_LEN = 52        # includes <start> and <end> tokens
PAD_TOKEN   = "<pad>"
START_TOKEN = "<start>"
END_TOKEN   = "<end>"
UNK_TOKEN   = "<unk>"

# ── training ─────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 4e-4
ENCODER_LR    = 1e-4        # lower LR for fine-tuning encoder
EPOCHS        = 25
GRAD_CLIP     = 5.0
TEACHER_FORCE = 0.75        # probability of teacher forcing
FINE_TUNE_ENCODER = False   # set True after decoder converges

# ── beam search ──────────────────────────────────────────────
BEAM_WIDTH    = 5
LENGTH_PENALTY = 0.7        # penalize short captions during beam

# ── device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── pre-trained model (for demo / comparison) ───────────────
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ── image transforms ────────────────────────────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
