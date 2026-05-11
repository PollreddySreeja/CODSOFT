"""
Encoder module – extracts spatial feature maps from images using
a pre-trained ResNet-50 backbone.

We strip the last two layers (avg-pool + fc) so the output is a
grid of 2048-d feature vectors, one per spatial location.  An
adaptive pool lets us fix the grid to a chosen size (default 7×7)
regardless of input resolution.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    ResNet-50 encoder that outputs spatial features
    of shape (batch, encoded_size*encoded_size, encoder_dim).
    """

    def __init__(self, encoded_size: int = 7, fine_tune: bool = False):
        super().__init__()

        # load ResNet-50 with ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # keep everything except avg-pool and fc
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # adaptive pool to get fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        # batch-norm to stabilize features going into decoder
        self.bn = nn.BatchNorm2d(2048)

        # freeze or unfreeze backbone weights
        self.set_fine_tune(fine_tune)

    def set_fine_tune(self, fine_tune: bool):
        """
        Toggle gradient computation for the backbone.
        When fine_tune is False only BN layers are frozen entirely;
        when True we unfreeze layer3 and layer4 (deeper features).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        if fine_tune:
            # unfreeze later layers only – they capture higher-level features
            for child in list(self.backbone.children())[5:]:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (batch, 3, H, W)

        Returns
        -------
        features : (batch, num_pixels, 2048)
            where num_pixels = encoded_size * encoded_size
        """
        features = self.backbone(images)          # (B, 2048, h, w)
        features = self.adaptive_pool(features)   # (B, 2048, enc, enc)
        features = self.bn(features)

        # flatten spatial dims and move channels to last axis
        batch = features.size(0)
        features = features.view(batch, 2048, -1) # (B, 2048, enc*enc)
        features = features.permute(0, 2, 1)      # (B, enc*enc, 2048)

        return features
