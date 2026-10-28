
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import mobilised_regression as base


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class PatchResNetRegressor(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(ResBlock(in_channels, 64), ResBlock(64, 128), ResBlock(128, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )

    def forward(self, tokens):
        return self.head(self.net(tokens.transpose(1, 2))).squeeze(-1)


class MobiliseReferencePatchResNet(base.MobiliseReferenceRegressorBase):
    def __init__(self, args):
        super().__init__(args)
        hidden = int(self.encoder.mae.norm.normalized_shape[0])
        self.head = PatchResNetRegressor(hidden, args.dropout)

    def encode_patch_tokens(self, x):
        x = self.encoder._fit_channels(x)
        x = self.encoder._apply_channel_mask(x)
        images = self.encoder.image_encoder(x)
        tokens = self.encoder.mae.patch_embed(images)
        tokens = tokens + self.encoder.mae.pos_embed[:, 1:, :]
        cls_token = self.encoder.mae.cls_token + self.encoder.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        for block in self.encoder.mae.blocks:
            tokens = block(tokens)
        tokens = self.encoder.mae.norm(tokens)
        return tokens[:, 1:, :]

    def forward(self, x):
        tokens = self.encode_patch_tokens(x)
        return self.head(tokens.float())


if __name__ == "__main__":
    base.main(MobiliseReferencePatchResNet)
