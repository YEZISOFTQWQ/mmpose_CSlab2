"""Dual-scale regression head for full-to-single Strided Transformer training."""

import torch
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.models.heads.base_head import BaseHead
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy


@MODELS.register_module()
class FullToSingleRegressionHead(BaseHead):
    """Predict all VTE frames and the STE target frame with two MPJPE losses."""

    def __init__(self,
                 in_channels=256,
                 num_joints=17,
                 loss=dict(type='MPJPELoss'),
                 sequence_loss=dict(type='MPJPELoss'),
                 sequence_loss_weight=1.0,
                 decoder=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_joints = num_joints
        self.sequence_loss_weight = sequence_loss_weight
        self.single_norm = nn.BatchNorm1d(in_channels)
        self.single_conv = nn.Conv1d(in_channels, num_joints * 3, 1)
        self.sequence_norm = nn.BatchNorm1d(in_channels)
        self.sequence_conv = nn.Conv1d(in_channels, num_joints * 3, 1)
        self.loss_module = MODELS.build(loss)
        self.sequence_loss_module = MODELS.build(sequence_loss)
        self.decoder = KEYPOINT_CODECS.build(decoder) if decoder else None

    def _single_coords(self, feats) -> Tensor:
        x = self.single_conv(self.single_norm(feats[-1]))
        return x.squeeze(-1).reshape(-1, self.num_joints, 3)

    def _sequence_coords(self, feats) -> Tensor:
        x = self.sequence_conv(self.sequence_norm(feats[0]))
        return x.transpose(1, 2).reshape(x.shape[0], -1, self.num_joints, 3)

    def forward(self, feats) -> Tensor:
        return self._single_coords(feats)

    def loss(self, feats, batch_data_samples, train_cfg=None):
        single_pred = self._single_coords(feats)
        sequence_pred = self._sequence_coords(feats)
        single_label = torch.cat([
            sample.gt_instance_labels.lifting_target_label
            for sample in batch_data_samples])
        single_weight = torch.cat([
            sample.gt_instance_labels.lifting_target_weight
            for sample in batch_data_samples])
        sequence_label = torch.cat([
            sample.gt_instance_labels.lifting_sequence_label
            for sample in batch_data_samples])
        sequence_weight = torch.cat([
            sample.gt_instance_labels.lifting_sequence_weight
            for sample in batch_data_samples])
        if sequence_pred.shape != sequence_label.shape:
            raise ValueError(
                f'Sequence prediction {tuple(sequence_pred.shape)} does not '
                f'match label {tuple(sequence_label.shape)}')
        loss_single = self.loss_module(single_pred, single_label,
                                       single_weight.unsqueeze(-1))
        loss_sequence = self.sequence_loss_module(
            sequence_pred, sequence_label, sequence_weight.unsqueeze(-1))
        mpjpe = keypoint_mpjpe(to_numpy(single_pred), to_numpy(single_label),
                               to_numpy(single_weight) > 0)
        return dict(loss_pose3d=loss_single,
                    loss_pose3d_sequence=loss_sequence * self.sequence_loss_weight,
                    mpjpe=torch.tensor(mpjpe, device=single_label.device))

    def predict(self, feats, batch_data_samples, test_cfg=None):
        coords = self._single_coords(feats)
        target_root = batch_data_samples[0].metainfo.get('target_root', None)
        if target_root is not None:
            target_root = torch.stack([
                torch.from_numpy(sample.metainfo['target_root'])
                for sample in batch_data_samples])
        else:
            target_root = torch.stack([
                torch.empty((0), dtype=torch.float32)
                for _ in batch_data_samples])
        return self.decode((coords, target_root))
