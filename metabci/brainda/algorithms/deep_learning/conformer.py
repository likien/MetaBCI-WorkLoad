# -*- coding: utf-8 -*-
#
# Date: 2024/7/28
# License: GPLv3
"""
EEG Conformer.
Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

"""

import math

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from .base import SkorchNet


class PatchEmbedding(nn.Module):
    """
    Convolution module: use conv to capture local features, instead of position embedding.
    """

    def __init__(self, emb_size=40, channels=32, num_kernel=40, kernel_size=(1, 25), pool_kernel=(1, 75),
                 pool_stride=(1, 15), dropout=0.5):
        super().__init__()

        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, num_kernel, kernel_size),
            nn.Conv2d(num_kernel, num_kernel, (channels, 1)),
            nn.BatchNorm2d(num_kernel),
            nn.ELU(),
            nn.AvgPool2d(pool_kernel, pool_stride),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(num_kernel, emb_size, (1, 1)),
            # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallow_net(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, num_heads, emb_size, drop_p=0.5, forward_drop_p=0.5):
        super().__init__(
            *[TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads, drop_p=drop_p,
                                      forward_drop_p=forward_drop_p) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, middle_size, emb_size, n_classes, cls_type="avg_pool", dropout=0.5):
        super().__init__()

        self.cls_type = cls_type

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(middle_size, middle_size // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size // 2, middle_size // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size // 4, n_classes)
        )

    def forward(self, x):
        if self.cls_type == "avg_pool":
            out = self.clshead(x)
        else:
            x = x.contiguous().view(x.size(0), -1)
            out = self.fc(x)

        return out


@SkorchNet
class Conformer(nn.Module):
    """
    Conformer is a neural network model designed for EEG signal classification tasks,
    integrating convolutional layers for local feature extraction and transformer
    layers for capturing long-range dependencies.

    The Conformer model combines the strengths of convolutional neural networks (CNNs)
    and transformers, leveraging CNNs for spatial feature extraction and transformers
    for sequence modeling.

    Created on: 2024-07-28

    Parameters
    ----------
    num_channels: int
        Number of channels in the input signal.
    n_samples: int
        Number of sampling points in the input signal.
    n_classes: int
        The number of classes for classification.
    num_conv_kernel: int, optional
        Number of convolution kernels (default is 40).
    emb_size: int, optional
        Dimension of the embeddings (default is 40).
    kernel_size: tuple, optional
        Size of the convolution kernel (default is (1, 25)).
    depth: int, optional
        Number of transformer layers (default is 6).
    num_heads: int, optional
        Number of attention heads (default is 10).
    dropout: float, optional
        Dropout rate (default is 0.5).
    cls_type: str, optional
        Type of classification head ('avg_pool' or 'fc', default is 'avg_pool').

    Attributes
    ----------
    patch_embedding: torch.nn.Module
        Module for extracting patch embeddings using convolution.
    transformer: torch.nn.Sequential
        Transformer module consisting of multiple layers.
    fc: torch.nn.Sequential
        Classification head for producing final class scores.

    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> num_classes = 2
    >>> X = torch.randn(16, 32, 200)
    >>> model = Conformer(num_channels=32, n_samples=200, emb_size=40, depth=4, n_classes=num_classes)
    >>> output = model(X)

    References
    ----------
    .. [1] Y. Song, Q. Zheng, B. Liu and X. Gao, "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, 2023, doi: 10.1109/TNSRE.2022.3230250.
    """

    def __init__(self, num_channels, n_samples, n_classes, num_conv_kernel=40, emb_size=40, kernel_size=(1, 25),
                 depth=6, num_heads=10, dropout=0.5, cls_type="avg_pool"):
        super().__init__()
        pool_kernel = (1, 75)
        pool_stride = (1, 15)

        self.patch_embedding = PatchEmbedding(emb_size=emb_size, channels=num_channels, num_kernel=num_conv_kernel,
                                              kernel_size=kernel_size, pool_kernel=pool_kernel, pool_stride=pool_stride,
                                              dropout=dropout)
        self.transformer = TransformerEncoder(depth=depth, num_heads=num_heads, emb_size=emb_size)

        with torch.no_grad():
            fake_input = torch.zeros((1, 1, num_channels, n_samples))
            fake_output = self.transformer(self.patch_embedding(fake_input))
            middle_size = fake_output.shape[-1] * fake_output.shape[-2]

        self.fc = ClassificationHead(middle_size=middle_size, emb_size=emb_size, n_classes=n_classes, cls_type=cls_type,
                                     dropout=dropout)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        out = self.patch_embedding(x)
        out = self.transformer(out)
        out = self.fc(out)

        return out
