"""
Caption decoder – LSTM with Bahdanau attention.

At each time-step the decoder:
  1. Embeds the previous word (or <start> at t=0).
  2. Computes attention over the encoder feature grid.
  3. Concatenates the embedding with the attention context.
  4. Feeds that through the LSTM cell.
  5. Projects the hidden state to vocabulary logits.

The attention weights are returned at every step so we can
build per-word heatmaps later for visualisation.
"""

import torch
import torch.nn as nn

from .attention import BahdanauAttention


class CaptionDecoder(nn.Module):
    """
    LSTM decoder with soft attention for image caption generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        attention_dim: int = 256,
        decoder_dim: int = 512,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.vocab_size    = vocab_size
        self.encoder_dim   = encoder_dim
        self.decoder_dim   = decoder_dim

        # word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout   = nn.Dropout(dropout)

        # attention module
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        # LSTM cell: input = [embedding ; context]
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # initialisation layers – learn initial h0, c0 from mean encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # gating scalar β – controls how much the context contributes
        self.f_beta  = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # output projection to vocab
        self.fc_out = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    # ── weight initialisation ────────────────────────────────
    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc_out.weight.data.uniform_(-0.1, 0.1)
        self.fc_out.bias.data.fill_(0)

    # ── initial hidden / cell state from encoder features ────
    def init_hidden_state(self, encoder_out: torch.Tensor):
        """
        Compute h0 and c0 by taking the mean of encoder features
        across all spatial locations.
        """
        mean_features = encoder_out.mean(dim=1)          # (B, enc_dim)
        h0 = self.init_h(mean_features)                  # (B, dec_dim)
        c0 = self.init_c(mean_features)                  # (B, dec_dim)
        return h0, c0

    # ── forward pass (teacher-forcing) ───────────────────────
    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        """
        Parameters
        ----------
        encoder_out     : (batch, num_pixels, encoder_dim)
        captions        : (batch, max_len)  – encoded caption indices
        caption_lengths : (batch,)          – true lengths (incl. <end>)

        Returns
        -------
        predictions : (batch, max_len-1, vocab_size)  – logits
        alphas      : (batch, max_len-1, num_pixels)  – attention maps
        sorted_idx  : (batch,) – indices used to sort by length
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # sort by decreasing caption length (needed for pack_padded later)
        caption_lengths, sorted_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sorted_idx]
        captions    = captions[sorted_idx]

        # embed all words at once
        embeddings = self.embedding(captions)              # (B, max_len, emb)

        # initial LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # we decode starting from <start> so we skip the last token
        decode_len = (caption_lengths - 1).tolist()
        max_decode = max(decode_len)

        predictions = torch.zeros(batch_size, max_decode, self.vocab_size).to(encoder_out.device)
        alphas      = torch.zeros(batch_size, max_decode, num_pixels).to(encoder_out.device)

        for t in range(max_decode):
            # only process sequences that haven't ended yet
            active = sum([1 for l in decode_len if t < l])

            # attention
            context, alpha = self.attention(
                encoder_out[:active], h[:active]
            )

            # gating the context
            gate    = self.sigmoid(self.f_beta(h[:active]))     # (active, enc_dim)
            context = gate * context

            # LSTM step
            lstm_input = torch.cat(
                [embeddings[:active, t, :], context], dim=1
            )                                                    # (active, emb+enc)
            h_new, c_new = self.lstm_cell(lstm_input, (h[:active], c[:active]))

            # update full tensors
            h = h.clone()
            c = c.clone()
            h[:active] = h_new
            c[:active] = c_new

            # project to vocabulary
            preds = self.fc_out(self.dropout(h_new))            # (active, vocab)
            predictions[:active, t, :] = preds
            alphas[:active, t, :]      = alpha

        return predictions, alphas, sorted_idx
