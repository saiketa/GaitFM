import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SensorImageEncoder import SensorImageEncoder
from models import MAEViT


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(getattr(configs, "seq_len", 250))
        self.mask_ratio = float(getattr(configs, "mae_mask_ratio", 0.75))
        self.norm_pix_loss = bool(int(getattr(configs, "mae_norm_pix_loss", 1)))

        image_size = int(getattr(configs, "gait_image_size", 224))
        segment_stride = int(getattr(configs, "gait_segment_stride", 10))
        use_sum_to_one = bool(int(getattr(configs, "gait_sum_to_one", 1)))
        self.max_channels = int(getattr(configs, "gait_max_channels", 6))
        self.drop_gyro_prob = float(os.environ.get("GAITFM_PRETRAIN_DROP_GYRO_PROB", "0.5"))

        self.image_encoder = SensorImageEncoder(
            segment_stride=segment_stride,
            image_size=image_size,
            use_sum_to_one=use_sum_to_one,
        )

        mae_name = getattr(configs, "mae_model", "mae_vit_tiny_patch16")
        if not hasattr(MAEViT, mae_name):
            raise ValueError(f"Unknown internal MAE model '{mae_name}'")
        self.mae = getattr(MAEViT, mae_name)(
            img_size=image_size,
            norm_pix_loss=self.norm_pix_loss,
        )
        checkpoint_path = getattr(configs, "mae_checkpoint_path", None) or os.path.join(
            "checkpoints", "backbones", "mae_tiny_400e.pth.tar"
        )
        self._load_mae_checkpoint(checkpoint_path)
        print(
            "GaitFMPretrain | mae={0} image_size={1} max_channels={2} "
            "mask_ratio={3} norm_pix_loss={4} drop_gyro_prob={5}".format(
                mae_name,
                image_size,
                self.max_channels,
                self.mask_ratio,
                int(self.norm_pix_loss),
                self.drop_gyro_prob,
            )
        )

    def _load_mae_checkpoint(self, checkpoint_path):
        if checkpoint_path is None or str(checkpoint_path).strip().lower() in {"", "none", "random", "scratch"}:
            print("MAE checkpoint disabled; using randomly initialized MAE backbone.")
            return
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MAE checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        state = {
            k.replace("module.", "").removeprefix("model."): v
            for k, v in state.items()
        }
        missing, unexpected = self.mae.load_state_dict(state, strict=False)
        print(f"Loaded MAE weights from {path} | missing={len(missing)} unexpected={len(unexpected)}")

    def _target_for_loss(self, images):
        target = self.mae.patchify(images.detach())
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        return target

    def _masked_patch_metrics(self, target_images, pred, mask):
        target = self._target_for_loss(target_images)
        patch_mse = ((pred - target) ** 2).mean(dim=-1)
        patch_mae = (pred - target).abs().mean(dim=-1)
        denom = mask.sum().clamp_min(1.0)
        mse = (patch_mse * mask).sum() / denom
        mae = (patch_mae * mask).sum() / denom
        return mse, mae

    def _fit_channels(self, x, channel_valid_mask=None):
        channels = x.shape[-1]
        if channels == self.max_channels:
            if channel_valid_mask is None:
                channel_valid_mask = torch.ones(x.shape[0], channels, dtype=torch.bool, device=x.device)
            return x, channel_valid_mask
        if channels > self.max_channels:
            x = x[..., : self.max_channels]
            if channel_valid_mask is not None:
                channel_valid_mask = channel_valid_mask[:, : self.max_channels]
            else:
                channel_valid_mask = torch.ones(x.shape[0], self.max_channels, dtype=torch.bool, device=x.device)
            return x, channel_valid_mask
        pad = self.max_channels - channels
        x = F.pad(x, (0, pad), mode="constant", value=0.0)
        if channel_valid_mask is None:
            channel_valid_mask = torch.zeros(x.shape[0], self.max_channels, dtype=torch.bool, device=x.device)
            channel_valid_mask[:, :channels] = True
        else:
            channel_valid_mask = F.pad(channel_valid_mask.to(dtype=torch.bool), (0, pad), mode="constant", value=False)
        return x, channel_valid_mask

    def _maybe_drop_gyro(self, x, channel_valid_mask):
        if (not self.training) or self.drop_gyro_prob <= 0 or x.shape[-1] < 6:
            return x, channel_valid_mask
        valid = channel_valid_mask.to(device=x.device, dtype=torch.bool)
        has_acc_gyro = valid[:, :6].all(dim=1)
        if not has_acc_gyro.any():
            return x, channel_valid_mask
        drop = (torch.rand(x.shape[0], device=x.device) < self.drop_gyro_prob) & has_acc_gyro
        if not drop.any():
            return x, channel_valid_mask
        x = x.clone()
        valid = valid.clone()
        x[drop, :, 3:6] = 0.0
        valid[drop, 3:6] = False
        return x, valid

    def forward(self, x_enc, padding_mask=None, x_mark_enc=None, x_dec=None, mask=None, channel_valid_mask=None):
        if x_enc.ndim != 3:
            raise ValueError(f"Expected input [B,T,C], got {tuple(x_enc.shape)}")
        if x_enc.shape[1] != self.seq_len:
            x_enc = nn.functional.interpolate(
                x_enc.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        if channel_valid_mask is not None:
            channel_valid_mask = channel_valid_mask.to(device=x_enc.device, dtype=torch.bool)
        x_enc, channel_valid_mask = self._fit_channels(x_enc, channel_valid_mask=channel_valid_mask)
        x_enc, channel_valid_mask = self._maybe_drop_gyro(x_enc, channel_valid_mask)

        images = self.image_encoder(x_enc)
        target_images = images.detach()
        latent, mae_mask, ids_restore = self.mae.forward_encoder(images, self.mask_ratio)
        pred = self.mae.forward_decoder(latent, ids_restore)
        loss = self.mae.forward_loss(target_images, pred, mae_mask)
        mse, mae = self._masked_patch_metrics(target_images, pred, mae_mask)
        return {
            "loss": loss,
            "mse": mse.detach(),
            "mae": mae.detach(),
            "mask_ratio": mae_mask.float().mean().detach(),
        }

    def reconstruction_loss(self, x_enc, channel_valid_mask=None):
        return self.forward(x_enc, channel_valid_mask=channel_valid_mask)
