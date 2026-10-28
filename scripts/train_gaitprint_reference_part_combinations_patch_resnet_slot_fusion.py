
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gaitprint_part_combination_regression as base


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class SlotFusionPatchResNetHead(nn.Module):
    def __init__(self, embedding_dim, num_parts, hidden_dim, dropout):
        super().__init__()
        self.num_parts = int(num_parts)
        self.part_dim = 128
        self.feature_extractor = nn.Sequential(
            ResBlock(embedding_dim, 64),
            ResBlock(64, self.part_dim),
            ResBlock(self.part_dim, self.part_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        fusion_dim = self.num_parts * self.part_dim + self.num_parts
        self.regressor = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, patch_tokens):
        return self.feature_extractor(patch_tokens.transpose(1, 2))

    def forward(self, slots, slot_mask):
        fusion = torch.cat([slots.reshape(slots.shape[0], -1), slot_mask], dim=-1)
        return self.regressor(fusion).squeeze(-1)


class GaitPrintPartCombinationPatchResNetSlotFusion(base.GaitPrintPartCombinationRegressorBase):
    def __init__(self, args, num_parts):
        super().__init__(args, num_parts)
        embedding_dim = int(self.encoder.mae.norm.normalized_shape[0])
        self.num_parts = int(num_parts)
        self.part_embed = nn.Embedding(self.num_parts, 128)
        self.head = SlotFusionPatchResNetHead(embedding_dim, self.num_parts, args.hidden_dim, args.dropout)

    def encode_patch_tokens(self, x):
        x = self.encoder._fit_channels(x)
        x = self.encoder._apply_channel_mask(x)
        images = self.encoder.image_encoder(x)
        tokens = self.encoder.mae.patch_embed(images)
        tokens = tokens + self.encoder.mae.pos_embed[:, 1:, :]
        cls_token = self.encoder.mae.cls_token + self.encoder.mae.pos_embed[:, :1, :]
        tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for block in self.encoder.mae.blocks:
            tokens = block(tokens)
        return self.encoder.mae.norm(tokens)[:, 1:, :]

    def forward(self, x, part_ids, part_mask):
        batch, max_parts, seq_len, channels = x.shape
        flat_mask = part_mask.reshape(-1).bool()
        flat_x = x.reshape(batch * max_parts, seq_len, channels)
        flat_ids = part_ids.reshape(-1)

        features = x.new_zeros((batch * max_parts, 128))
        if flat_mask.any():
            valid_tokens = self.encode_patch_tokens(flat_x[flat_mask])
            valid_features = self.head.encode(valid_tokens.float())
            valid_features = valid_features + self.part_embed(flat_ids[flat_mask])
            features[flat_mask] = valid_features.to(dtype=features.dtype)
        features = features.reshape(batch, max_parts, 128)

        slots = x.new_zeros((batch, self.num_parts, 128))
        slot_mask = x.new_zeros((batch, self.num_parts))
        rows = torch.arange(batch, device=x.device)
        valid_mask = part_mask.bool()
        for position in range(max_parts):
            active = valid_mask[:, position]
            if not active.any():
                continue
            ids = part_ids[:, position].clamp(min=0, max=self.num_parts - 1)
            slots[rows[active], ids[active]] = features[active, position]
            slot_mask[rows[active], ids[active]] = 1.0
        return self.head(slots.float(), slot_mask.float())


if __name__ == "__main__":
    base.main(GaitPrintPartCombinationPatchResNetSlotFusion)
