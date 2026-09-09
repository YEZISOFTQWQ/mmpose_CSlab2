"""Strided Transformer backbone for temporal 2D-to-3D pose lifting.

Implements the VTE + STE split of Li et al., "Exploiting Temporal Contexts
with Strided Transformer for 3D Human Pose Estimation" (T-PAMI 2022). The
backbone returns full-sequence VTE features and one target-frame STE feature,
so a companion head can apply the paper's full-to-single supervision.
"""

import torch
from torch import Tensor, nn

from mmpose.models.backbones.base_backbone import BaseBackbone
from mmpose.registry import MODELS


class VanillaTransformerBlock(nn.Module):
    """Pre-norm temporal self-attention followed by an MLP."""

    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        attended = self.norm1(x)
        x = x + self.attn(attended, attended, attended, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class StridedTransformerBlock(nn.Module):
    """Attention plus strided convolutional FFN that shortens time length."""

    def __init__(self, embed_dim, num_heads, feedforward_dim, stride, dropout):
        super().__init__()
        self.stride = stride
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cffn = nn.Sequential(
            nn.Conv1d(embed_dim, feedforward_dim, kernel_size=1),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(feedforward_dim, embed_dim, kernel_size=stride,
                      stride=stride), nn.Dropout(dropout))
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] % self.stride:
            raise ValueError(
                f'Temporal length {x.shape[1]} is not divisible by stride '
                f'{self.stride}')
        attended = self.norm1(x)
        x = x + self.attn(attended, attended, attended, need_weights=False)[0]
        normalized = self.norm2(x).transpose(1, 2)
        return (self.pool(x.transpose(1, 2)) + self.cffn(normalized)).transpose(1, 2)


@MODELS.register_module()
class StridedTransformerBackbone(BaseBackbone):
    """Lift a fixed temporal window of 2D joints into temporal features.

    Args:
        in_channels: Flattened joint feature channels, normally ``17 * 2``.
        keypoint_dim: Features per joint. v3 uses ``(x, y)``; the planned
            occlusion-aware v4 uses ``(x, y, confidence, mask)``.
        seq_len: Fixed odd input window. ``9`` is used for the first H36M
            fps10 experiment; ``27`` with strides ``(3, 3, 3)`` matches the
            paper's illustrative compression pattern.
        strides: Per-STE temporal compression factors. Their product must
            equal ``seq_len`` so STE yields exactly the center-pose feature.
    """

    def __init__(self,
                 in_channels=34,
                 keypoint_dim=2,
                 seq_len=9,
                 embed_dim=256,
                 num_heads=8,
                 feedforward_dim=512,
                 num_vte_layers=3,
                 strides=(3, 3),
                 dropout=0.25):
        super().__init__()
        if seq_len % 2 != 1:
            raise ValueError('seq_len must be odd for a centered target frame')
        product = 1
        for stride in strides:
            product *= stride
        if product != seq_len:
            raise ValueError(
                f'Product of STE strides {tuple(strides)} must equal seq_len '
                f'{seq_len}, got {product}')
        self.causal = False
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.keypoint_dim = keypoint_dim
        if in_channels % keypoint_dim:
            raise ValueError(
                f'in_channels={in_channels} is not divisible by '
                f'keypoint_dim={keypoint_dim}')
        self.embed_dim = embed_dim
        self.pose_embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.vte_position = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.vte = nn.ModuleList([
            VanillaTransformerBlock(embed_dim, num_heads, feedforward_dim,
                                    dropout) for _ in range(num_vte_layers)
        ])
        lengths = []
        length = seq_len
        for stride in strides:
            lengths.append(length)
            length //= stride
        self.ste_position = nn.ParameterList([
            nn.Parameter(torch.zeros(1, length, embed_dim)) for length in lengths
        ])
        self.ste = nn.ModuleList([
            StridedTransformerBlock(embed_dim, num_heads, feedforward_dim,
                                    stride, dropout)
            for stride in strides
        ])
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.vte_position, std=.02)
        for position in self.ste_position:
            nn.init.trunc_normal_(position, std=.02)

    def forward(self, x: Tensor):
        """Forward 2D joints to VTE and final STE features.

        MMPose's pose-lifting packer stores a sequence as flattened
        ``(B, K*C, T)``. Accept that native representation as well as the
        explicit ``(B, K, C, T)`` representation used in the paper.
        """
        if x.ndim == 3:
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f'Expected {self.in_channels} flattened channels, got '
                    f'{tuple(x.shape)}')
            x = x.reshape(x.shape[0], -1, self.keypoint_dim, x.shape[-1])
        if x.ndim != 4 or x.shape[-1] != self.seq_len:
            raise ValueError(
                f'Expected (B, K, C, {self.seq_len}), got {tuple(x.shape)}')
        tokens = x.permute(0, 3, 1, 2).reshape(x.shape[0], self.seq_len, -1)
        # BatchNorm1d works on channel dimension, not temporal token dimension.
        embedded = self.pose_embed[0](tokens)
        embedded = self.pose_embed[1](embedded.transpose(1, 2)).transpose(1, 2)
        x = self.pose_embed[3](self.pose_embed[2](embedded)) + self.vte_position
        for block in self.vte:
            x = block(x)
        full_sequence = x.transpose(1, 2)
        for position, block in zip(self.ste_position, self.ste):
            x = block(x + position)
        return full_sequence, x.transpose(1, 2)
