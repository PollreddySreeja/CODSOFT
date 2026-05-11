"""
Evaluation script – computes BLEU scores on the validation set
and generates sample captions with attention visualisation.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import config
from src.vocabulary import Vocabulary
from src.caption_model import ImageCaptioner
from src.dataset import Flickr8kDataset, get_eval_transform


def load_model(checkpoint_path, vocab):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
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
    return model


def compute_bleu(model, dataset, vocab, num_samples=500):
    """
    Compute BLEU-1 through BLEU-4 on a subset of the validation data.
    """
    references_corpus = []
    hypotheses_corpus = []
    transform = get_eval_transform()

    # group captions by image
    from collections import defaultdict
    img_captions = defaultdict(list)
    for i in range(len(dataset)):
        img_name = dataset.images[i]
        caption = dataset.captions[i]
        img_captions[img_name].append(caption.lower().split())

    sampled = list(img_captions.keys())[:num_samples]
    smooth = SmoothingFunction().method1

    for img_name in sampled:
        img_path = os.path.join(config.IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).to(config.DEVICE)

        caption, _ = model.generate_caption(image, vocab)
        hypothesis = caption.split()

        refs = img_captions[img_name]
        references_corpus.append(refs)
        hypotheses_corpus.append(hypothesis)

    # compute BLEU scores
    bleu1 = corpus_bleu(references_corpus, hypotheses_corpus,
                        weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references_corpus, hypotheses_corpus,
                        weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references_corpus, hypotheses_corpus,
                        weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references_corpus, hypotheses_corpus,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}


def visualise_attention(image_path, caption_words, attention_maps, save_path=None):
    """
    Create a grid showing the original image alongside attention
    heatmaps for each generated word.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    num_words = min(len(caption_words), len(attention_maps))
    cols = 5
    rows = (num_words + 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(num_words):
        ax = axes[i + 1]
        attn = attention_maps[i].numpy()
        grid_size = int(np.sqrt(len(attn)))
        attn = attn.reshape(grid_size, grid_size)
        attn = np.array(Image.fromarray(attn).resize((224, 224), Image.BILINEAR))

        ax.imshow(image)
        ax.imshow(attn, alpha=0.6, cmap="jet")
        ax.set_title(caption_words[i], fontsize=9, fontweight="bold")
        ax.axis("off")

    # hide unused axes
    for j in range(num_words + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(" ".join(caption_words), fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[Eval] Attention map saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioner")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    vocab = Vocabulary.load(config.VOCAB_PATH)
    model = load_model(args.checkpoint, vocab)

    print("\n[Eval] Computing BLEU scores...")
    val_dataset = Flickr8kDataset(
        config.CAPTION_FILE, config.IMAGE_DIR, vocab,
        max_len=config.MAX_CAPTION_LEN, split="val",
        transform=get_eval_transform()
    )

    scores = compute_bleu(model, val_dataset, vocab, args.num_samples)

    print("\n" + "=" * 40)
    print(" BLEU Score Results")
    print("=" * 40)
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    print("=" * 40)

    # generate a few sample captions with attention
    print("\n[Eval] Generating sample captions with attention maps...\n")
    transform = get_eval_transform()
    sample_images = val_dataset.images[:5]

    for img_name in sample_images:
        img_path = os.path.join(config.IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).to(config.DEVICE)

        caption, attn_maps = model.generate_caption(image_tensor, vocab)
        print(f"  {img_name}: {caption}")

        save_path = os.path.join(config.CHECKPOINT_DIR, f"attn_{img_name}")
        visualise_attention(img_path, caption.split(), attn_maps, save_path)


if __name__ == "__main__":
    main()
