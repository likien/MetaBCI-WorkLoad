# -*- coding: utf-8 -*-
#
# Date: 2024/7/28
# License: MIT License
"""
SimpleViT.
Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

"""
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from .base import SkorchNet


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


@SkorchNet
class SimpleViT(nn.Module):
    """
    SimpleViT is a Vision Transformer model designed for image classification tasks,
    leveraging the power of transformer architecture to capture spatial dependencies
    within images.

    SimpleViT divides the input image into patches, applies positional embeddings,
    and processes the patches through multiple transformer layers. The output is
    then classified using a linear head.

    Created on: 2024-07-28

    Parameters
    ----------
    image_size: tuple
        Size of the input image (height, width).
    patch_size: tuple
        Size of each patch (height, width).
    num_classes: int
        The number of classes for classification.
    dim: int, optional
        Dimension of the patch embeddings (default is 64).
    depth: int, optional
        Number of transformer layers (default is 2).
    heads: int, optional
        Number of attention heads (default is 4).
    mlp_dim: int, optional
        Dimension of the MLP (FeedForward) layer (default is 64).
    channels: int, optional
        Number of input channels (default is 3).
    dim_head: int, optional
        Dimension of each attention head (default is 64).

    Attributes
    ----------
    to_patch_embedding: torch.nn.Sequential
        Embedding layer to convert image patches into embeddings.
    pos_embedding: torch.Tensor
        Positional embeddings for the patches.
    transformer: torch.nn.Module
        Transformer module consisting of multiple layers.
    pool: str
        Pooling method to aggregate patch embeddings.
    to_latent: torch.nn.Identity
        Identity layer before the final classification.
    linear_head: torch.nn.Linear
        Linear layer for the final classification.

    Examples
    ----------
    >>> # X size: [batch size, number of channels, number of sample points]
    >>> X = torch.randn(16, 32, 200)
    >>> num_classes = 3
    >>> model = SimpleViT(image_size=(1, 32), patch_size=(1, 1), num_classes=num_classes, dim=20, depth=2, heads=4, mlp_dim=40, channels=200, dim_head=64)
    >>> output = model(X)

    References
    ----------
    .. [1] Beyer L, Zhai X, Kolesnikov A. Better plain ViT baselines for ImageNet-1k[J]. arXiv preprint arXiv:2205.01580, 2022.
    """

    def __init__(self, image_size, patch_size, num_classes, dim=64, depth=2, heads=4, mlp_dim=64, channels=3,
                 dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        img = img.unsqueeze(dim=1)
        img = img.permute(0, 3, 1, 2)

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
