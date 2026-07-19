#!/usr/bin/env python3
"""Run a small end-to-end CUDA smoke test through DVLT's VGGT-Omega wrapper."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from accelerate import Accelerator

from dvlt.common.constants import DataField, PredictionField
from dvlt.model.vggt_omega.model import VGGTOmega


def find_checkpoint() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("VGGT_OMEGA_CHECKPOINT"),
        repo_root / "checkpoints" / "vggt_omega_1b_512.pt",
        repo_root.parent / "checkpoints" / "vggt_omega_1b_512.pt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "VGGT-Omega checkpoint not found. Set VGGT_OMEGA_CHECKPOINT or place "
        "vggt_omega_1b_512.pt in ./checkpoints or ../checkpoints."
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA-capable GPU")

    checkpoint = find_checkpoint()
    accelerator = Accelerator(mixed_precision="bf16")
    model = VGGTOmega()
    model.load_pretrained(str(checkpoint), strict=True)
    model.setup_test(accelerator)

    # Two small, patch-aligned views exercise aggregation, camera/depth heads,
    # and DVLT's standardized post-processing without a costly 512px run.
    images = torch.rand(1, 2, 3, 64, 64, device=accelerator.device)
    batch = {DataField.IMAGES: images}
    with torch.inference_mode(), accelerator.autocast():
        predictions = model.predict(batch, accelerator)

    cameras = predictions[PredictionField.CAMERAS]
    depths = predictions[PredictionField.DEPTHS]
    points = predictions[PredictionField.WORLD_POINTS]
    assert len(cameras) == 1
    assert tuple(depths.shape) == (1, 2, 64, 64)
    assert tuple(points.shape) == (1, 2, 64, 64, 3)
    assert torch.isfinite(depths).all()
    assert torch.isfinite(points).all()

    print(
        {
            "checkpoint": str(checkpoint),
            "device": str(accelerator.device),
            "depth_shape": tuple(depths.shape),
            "points_shape": tuple(points.shape),
            "status": "ok",
        }
    )


if __name__ == "__main__":
    main()
