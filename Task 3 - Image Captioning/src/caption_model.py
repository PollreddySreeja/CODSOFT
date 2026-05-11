"""
End-to-end image captioning model.

Ties the encoder (ResNet-50) and decoder (LSTM + attention)
together and adds convenience methods for:
  • caption generation with greedy decoding
  • beam search with length penalty
  • returning attention maps for visualisation
"""

import torch
import torch.nn as nn

from .encoder import ImageEncoder
from .decoder import CaptionDecoder
from .beam_search import beam_search_decode


class ImageCaptioner(nn.Module):
    """
    Full image-captioning pipeline:
        Image → ResNet Encoder → Spatial Features → Attention Decoder → Caption
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = 300,
        attention_dim: int = 256,
        decoder_dim: int = 512,
        encoder_dim: int = 2048,
        encoded_size: int = 7,
        dropout: float   = 0.5,
        fine_tune_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            encoded_size=encoded_size,
            fine_tune=fine_tune_encoder,
        )
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            dropout=dropout,
        )

    def forward(self, images, captions, caption_lengths):
        """Training forward pass with teacher forcing."""
        features = self.encoder(images)
        predictions, alphas, sorted_idx = self.decoder(
            features, captions, caption_lengths
        )
        return predictions, alphas, sorted_idx

    # ── greedy decoding ──────────────────────────────────────
    @torch.no_grad()
    def generate_caption(
        self,
        image: torch.Tensor,
        vocabulary,
        max_len: int = 50,
    ):
        """
        Greedy decode: pick the highest-probability word at each step.
        Returns the generated caption string and the attention weights
        for each word.
        """
        self.eval()
        device = image.device

        # encode
        features = self.encoder(image.unsqueeze(0))   # (1, px, enc_dim)

        # init decoder
        h, c = self.decoder.init_hidden_state(features)

        word_idx = vocabulary.word2idx[vocabulary.start_token]
        word = torch.tensor([word_idx], device=device)

        caption_words = []
        attention_maps = []

        for _ in range(max_len):
            emb = self.decoder.embedding(word)            # (1, emb)
            context, alpha = self.decoder.attention(features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([emb, context], dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))

            logits = self.decoder.fc_out(h)               # (1, vocab)
            word_idx = logits.argmax(dim=1)

            token = vocabulary.idx2word[word_idx.item()]
            if token == vocabulary.end_token:
                break

            caption_words.append(token)
            attention_maps.append(alpha.squeeze(0).cpu())

            word = word_idx

        caption = " ".join(caption_words)
        return caption, attention_maps

    # ── beam search ──────────────────────────────────────────
    @torch.no_grad()
    def beam_search_caption(
        self,
        image: torch.Tensor,
        vocabulary,
        beam_width: int = 5,
        max_len: int = 50,
        length_penalty: float = 0.7,
    ):
        """
        Beam search decoding – returns top-k caption candidates
        along with their scores and attention maps.
        """
        self.eval()
        features = self.encoder(image.unsqueeze(0))

        results = beam_search_decode(
            decoder=self.decoder,
            encoder_out=features,
            vocabulary=vocabulary,
            beam_width=beam_width,
            max_len=max_len,
            length_penalty=length_penalty,
        )
        return results
