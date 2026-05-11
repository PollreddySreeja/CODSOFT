"""
Bahdanau (additive) attention mechanism.

At each decoding step the attention module takes:
  • the encoder output (all spatial feature vectors)
  • the decoder's previous hidden state

and produces:
  • a context vector (weighted sum of encoder features)
  • attention weights (one per spatial location → interpretable heatmap)

This is the key component that lets the model "look" at different
parts of the image while generating each word of the caption.
"""

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Additive attention: score(s_t, h_i) = v^T · tanh(W_s · s_t + W_h · h_i)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Parameters
        ----------
        encoder_dim   : feature vector size from encoder (2048 for ResNet-50)
        decoder_dim   : hidden state size of the LSTM decoder
        attention_dim : size of the intermediate attention space
        """
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.score_layer  = nn.Linear(attention_dim, 1)
        self.softmax      = nn.Softmax(dim=1)
        self.relu         = nn.ReLU()

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor
    ):
        """
        Parameters
        ----------
        encoder_out    : (batch, num_pixels, encoder_dim)
        decoder_hidden : (batch, decoder_dim)

        Returns
        -------
        context : (batch, encoder_dim) – weighted feature vector
        alpha   : (batch, num_pixels)  – attention weights for visualisation
        """
        # project both into the shared attention space
        enc_proj = self.encoder_proj(encoder_out)          # (B, px, att)
        dec_proj = self.decoder_proj(decoder_hidden)        # (B, att)
        dec_proj = dec_proj.unsqueeze(1)                    # (B, 1, att)

        # additive scoring
        combined = self.relu(enc_proj + dec_proj)           # (B, px, att)
        scores   = self.score_layer(combined).squeeze(2)    # (B, px)

        # normalise to get attention distribution
        alpha = self.softmax(scores)                        # (B, px)

        # context is the weighted sum of encoder features
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, enc_dim)

        return context, alpha
