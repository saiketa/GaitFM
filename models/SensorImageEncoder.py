import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SumToOneTemporalEmbedding(nn.Module):

    def __init__(self, segment_stride=10, use_sum_to_one=True):
        super().__init__()
        self.segment_stride = int(segment_stride)
        self.use_sum_to_one = bool(use_sum_to_one)
        if self.segment_stride <= 0:
            raise ValueError("segment_stride must be positive.")
        self.weight = nn.Parameter(torch.zeros(self.segment_stride * 3, 1, self.segment_stride))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            for out_idx in range(self.segment_stride * 3):
                self.weight[out_idx, 0, out_idx % self.segment_stride] = 1.0

    def _same_pad(self, length):
        out_len = int(math.ceil(length / float(self.segment_stride)))
        pad_total = max((out_len - 1) * self.segment_stride + self.segment_stride - length, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return pad_left, pad_right

    def _conv_weight(self):
        if not self.use_sum_to_one:
            return self.weight
        weight = self.weight.abs()
        return weight / (weight.sum(dim=(1, 2), keepdim=True) + 1e-6)

    def forward(self, x):
        pad_left, pad_right = self._same_pad(x.shape[-1])
        x = F.pad(x, (pad_left, pad_right))
        return F.conv1d(x, self._conv_weight(), stride=self.segment_stride, padding=0)


class SensorImageEncoder(nn.Module):

    def __init__(self, segment_stride=10, image_size=240, use_sum_to_one=True):
        super().__init__()
        self.segment_stride = int(segment_stride)
        self.image_size = int(image_size)
        self.temporal_embed = SumToOneTemporalEmbedding(segment_stride, use_sum_to_one)
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _minmax(self, x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, T, C], got {tuple(x.shape)}")
        bsz, seq_len, channels = x.shape
        x = x.float().transpose(1, 2).contiguous()
        x = x.view(bsz * channels, 1, seq_len)
        x = self.temporal_embed(x)

        token_len = x.shape[-1]
        x = x.view(bsz, channels, 3, self.segment_stride, token_len)
        x = x.permute(0, 2, 4, 1, 3).contiguous()
        x = x.view(bsz, 3, token_len, channels * self.segment_stride)
        x = x.permute(0, 1, 3, 2).contiguous()

        if self.image_size > 0:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self._minmax(x)
        return (x - self.image_mean) / self.image_std
