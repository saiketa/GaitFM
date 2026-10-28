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
        image_size = int(getattr(configs, "gait_image_size", 224))
        segment_stride = int(getattr(configs, "gait_segment_stride", 10))
        use_sum_to_one = bool(int(getattr(configs, "gait_sum_to_one", 1)))
        self.max_channels = int(getattr(configs, "gait_max_channels", 6))
        self.mask_mode = os.environ.get("GAITFM_CHANNEL_MASK_MODE", "random_acc_only").strip().lower()
        self.mask_prob = float(os.environ.get("GAITFM_CHANNEL_MASK_PROB", "0.5"))
        self.apply_eval_mask = bool(int(os.environ.get("GAITFM_CHANNEL_MASK_APPLY_EVAL", "0")))
        self.pooling = str(getattr(configs, "mae_finetune_pooling", "cls")).lower()

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
            norm_pix_loss=bool(int(getattr(configs, "mae_norm_pix_loss", 1))),
        )
        checkpoint_path = getattr(configs, "mae_checkpoint_path", None)
        if checkpoint_path is None:
            checkpoint_path = os.path.join("checkpoints", "backbones", "mae_tiny_400e.pth.tar")
        self._load_mae_checkpoint(checkpoint_path)

        hidden_size = int(self.mae.norm.normalized_shape[0])
        dropout = float(getattr(configs, "dropout", 0.5))
        classifier_hidden = int(getattr(configs, "classifier_hidden_dim", max(64, hidden_size // 2)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, int(configs.num_class)),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "GaitFMFullFinetune | mae={0} image_size={1} token_pooling={2} "
            "max_channels={3} channel_mask={4} prob={5} trainable params: {6}/{7} ({8:.2f}%)".format(
                mae_name,
                image_size,
                self.pooling,
                self.max_channels,
                self.mask_mode,
                self.mask_prob,
                trainable,
                total,
                100.0 * trainable / max(total, 1),
            )
        )

    def _load_mae_checkpoint(self, checkpoint_path):
        if checkpoint_path is None or str(checkpoint_path).strip().lower() in {"", "none", "random", "scratch"}:
            print("MAE-Lite checkpoint disabled; using randomly initialized MAE backbone.")
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
        print(f"Loaded MAE-Lite weights from {path} | missing={len(missing)} unexpected={len(unexpected)}")

    def _encode_images(self, pixel_values):
        tokens = self.mae.patch_embed(pixel_values)
        tokens = tokens + self.mae.pos_embed[:, 1:, :]
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        for block in self.mae.blocks:
            tokens = block(tokens)
        tokens = self.mae.norm(tokens)
        if self.pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        return tokens[:, 0, :]

    def _fit_channels(self, x):
        channels = x.shape[-1]
        if channels == self.max_channels:
            return x
        if channels > self.max_channels:
            return x[..., : self.max_channels]
        return F.pad(x, (0, self.max_channels - channels), mode="constant", value=0.0)

    def _zero_gyro(self, x):
        if x.shape[-1] >= 6:
            x = x.clone()
            x[..., 3:6] = 0.0
        return x

    def _zero_acc(self, x):
        if x.shape[-1] >= 3:
            x = x.clone()
            x[..., :3] = 0.0
        return x

    def _random_channel_zero(self, x):
        if x.shape[-1] <= 1 or self.mask_prob <= 0:
            return x
        keep = torch.rand(x.shape[0], x.shape[-1], device=x.device) >= self.mask_prob
        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            keep[all_dropped, torch.randint(0, x.shape[-1], (int(all_dropped.sum()),), device=x.device)] = True
        return x * keep[:, None, :].to(dtype=x.dtype)

    def _random_modality_zero(self, x):
        if x.shape[-1] < 6 or self.mask_prob <= 0:
            return x
        out = x.clone()
        drop_gyro = torch.rand(x.shape[0], device=x.device) < self.mask_prob
        out[drop_gyro, :, 3:6] = 0.0
        return out

    def _apply_channel_mask(self, x):
        mode = self.mask_mode
        if mode in {"", "none", "off", "no"}:
            return x
        if (not self.training) and (not self.apply_eval_mask):
            return x
        if mode == "acc_only":
            return self._zero_gyro(x)
        if mode == "gyro_only":
            return self._zero_acc(x)
        if mode == "random_channel":
            return self._random_channel_zero(x)
        if mode == "random_modality":
            return self._random_modality_zero(x)
        if mode == "random_acc_only":
            if x.shape[-1] < 6 or self.mask_prob <= 0:
                return x
            out = x.clone()
            use_acc_only = torch.rand(x.shape[0], device=x.device) < self.mask_prob
            out[use_acc_only, :, 3:6] = 0.0
            return out
        raise ValueError(
            "Unsupported GAITFM_CHANNEL_MASK_MODE='{0}'. Expected one of: "
            "none, acc_only, gyro_only, random_channel, random_modality, random_acc_only.".format(mode)
        )

    def forward(self, x_enc, padding_mask=None, x_mark_enc=None, x_dec=None, channel_valid_mask=None):
        x_enc = self._fit_channels(x_enc)
        x_enc = self._apply_channel_mask(x_enc)
        pixel_values = self.image_encoder(x_enc)
        features = self._encode_images(pixel_values)
        return self.classifier(features.float())
