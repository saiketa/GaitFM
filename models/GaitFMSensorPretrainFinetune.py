import os
from pathlib import Path

import torch

from models.GaitFMFullFinetune import Model as _DirectFinetuneModel


class Model(_DirectFinetuneModel):

    def __init__(self, configs):
        super().__init__(configs)
        checkpoint = getattr(configs, "gaitfm_pretrain_checkpoint", "")
        if not checkpoint:
            raise ValueError(
                "gaitfm_pretrain_checkpoint is required for GaitFMSensorPretrainFinetune."
            )
        self._load_sensor_pretrained_encoder(checkpoint)

    def _load_sensor_pretrained_encoder(self, checkpoint):
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"GaitFM sensor pretrain checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        compatible = {
            k: v
            for k, v in state.items()
            if k.startswith("image_encoder.") or k.startswith("mae.")
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(
            "Loaded GaitFM sensor-pretrained encoder from {0} | "
            "tensors={1} missing={2} unexpected={3}".format(
                path, len(compatible), len(missing), len(unexpected)
            )
        )
