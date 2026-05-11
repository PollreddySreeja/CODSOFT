"""
Training script for the Image Captioning model.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 64 --fine_tune

Downloads Flickr8k automatically if not present, builds vocabulary,
and trains the ResNet-50 + LSTM-Attention model with teacher forcing.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

import config
from src.vocabulary import Vocabulary
from src.dataset import Flickr8kDataset, get_eval_transform
from src.caption_model import ImageCaptioner


def setup_data():
    """Download Flickr8k from Kaggle if not already present."""
    if os.path.exists(config.CAPTION_FILE):
        print("[Data] Flickr8k already present, skipping download.")
        return

    print("[Data] Downloading Flickr8k dataset...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            "adityajn105/flickr8k", path=config.FLICKR8K, unzip=True
        )
        print("[Data] Download complete.")
    except Exception as e:
        print(f"[Data] Auto-download failed: {e}")
        print("[Data] Please manually download Flickr8k from:")
        print("       https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print(f"       and extract to: {config.FLICKR8K}")
        sys.exit(1)


def build_vocabulary():
    """Build or load vocabulary from captions."""
    if os.path.exists(config.VOCAB_PATH):
        return Vocabulary.load(config.VOCAB_PATH)

    import pandas as pd
    df = pd.read_csv(config.CAPTION_FILE)
    df.columns = [c.strip() for c in df.columns]
    captions = df["caption"].tolist()

    vocab = Vocabulary(min_freq=config.MIN_WORD_FREQ)
    vocab.build(captions)
    vocab.save(config.VOCAB_PATH)
    return vocab


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    """Single training epoch with teacher forcing."""
    model.train()
    epoch_loss = 0.0

    for images, captions, lengths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        predictions, alphas, sorted_idx = model(images, captions, lengths)

        # sort targets to match model output order
        sorted_captions = captions[sorted_idx]
        sorted_lengths = lengths[sorted_idx]
        targets = sorted_captions[:, 1:]  # skip <start>

        # pack predictions and targets (ignore padding)
        decode_lengths = (sorted_lengths - 1).tolist()
        max_decode = max(decode_lengths)

        # flatten for cross-entropy
        pred_flat = predictions[:, :max_decode, :].reshape(-1, predictions.size(2))
        tgt_flat = targets[:, :max_decode].reshape(-1)

        loss = criterion(pred_flat, tgt_flat)

        # doubly stochastic attention regularisation
        # encourages the model to attend to every pixel equally over time
        alpha_reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = loss + 1.0 * alpha_reg

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validation loop."""
    model.eval()
    val_loss = 0.0

    for images, captions, lengths in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        predictions, alphas, sorted_idx = model(images, captions, lengths)

        sorted_captions = captions[sorted_idx]
        sorted_lengths = lengths[sorted_idx]
        targets = sorted_captions[:, 1:]

        decode_lengths = (sorted_lengths - 1).tolist()
        max_decode = max(decode_lengths)

        pred_flat = predictions[:, :max_decode, :].reshape(-1, predictions.size(2))
        tgt_flat = targets[:, :max_decode].reshape(-1)

        loss = criterion(pred_flat, tgt_flat)
        val_loss += loss.item()

    return val_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Image Captioner")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"[Train] Using device: {device}")

    # prepare data
    setup_data()
    vocab = build_vocabulary()

    train_dataset = Flickr8kDataset(
        config.CAPTION_FILE, config.IMAGE_DIR, vocab,
        max_len=config.MAX_CAPTION_LEN, split="train"
    )
    val_dataset = Flickr8kDataset(
        config.CAPTION_FILE, config.IMAGE_DIR, vocab,
        max_len=config.MAX_CAPTION_LEN, split="val",
        transform=get_eval_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Train] Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"[Train] Vocabulary size: {len(vocab)}")

    # build model
    model = ImageCaptioner(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        attention_dim=config.ATTENTION_DIM,
        decoder_dim=config.HIDDEN_DIM,
        encoder_dim=config.ENCODER_DIM,
        encoded_size=config.ENCODED_SIZE,
        dropout=config.DROPOUT,
        fine_tune_encoder=args.fine_tune,
    ).to(device)

    # separate param groups for encoder and decoder
    decoder_params = list(model.decoder.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": args.lr},
        {"params": encoder_params, "lr": config.ENCODER_LR},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.word2idx[vocab.pad_token]
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Train] Resumed from epoch {start_epoch}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # training loop
    print(f"\n{'='*60}")
    print(f" Training Image Captioner — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.GRAD_CLIP
        )
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": len(vocab),
            }, save_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": len(vocab),
            }, ckpt_path)

    print("\n[Train] Training complete!")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
