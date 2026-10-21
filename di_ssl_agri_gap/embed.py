"""Stage 7: extract frozen features once per encoder.

The encoder is frozen and never touched again. Features are deterministic --
resize to 256 (a no-op on the cached tree) then a centre crop, with no
augmentation -- so they can be cached and reused by all four probes and by the
drift analysis.

Feature composition is `[CLS || mean(patch_tokens)]`, which is DINOv2's own
linear-evaluation representation and consistently beats CLS alone at fine
granularity -- precisely where this study's signal lives.

`lightly_train.embed` is deliberately not used: it returns a pooled embedding
only, and the patch-token mean is needed here.

The exported checkpoint holds the **EMA teacher** (verified: 176/176 tensors
match the teacher, 0/176 the student -- see docs/lightlytrain-findings.md),
which is the backbone DINOv2's own evaluations use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from di_ssl_agri_gap.enums import Split
from di_ssl_agri_gap.utils import load_settings, seed_rng

IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class ImageListDataset(Dataset):
    """Deterministic centre-cropped image loading, in fixed manifest order."""

    def __init__(self, root: Path, rel_paths: list[str]) -> None:
        from torchvision.transforms import v2

        self.root = root
        self.rel_paths = rel_paths
        self.transform = v2.Compose(
            [
                v2.Resize(256),
                v2.CenterCrop(IMAGE_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.root / self.rel_paths[index]) as img:
            return self.transform(img.convert("RGB"))


def load_backbone(checkpoint: Path | None, device: torch.device) -> torch.nn.Module:
    """Load an exported backbone, or Meta's off-the-shelf weights when None."""
    from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage

    if checkpoint is None:
        model = DINOv2ViTPackage.get_model("vits14")
    else:
        model = DINOv2ViTPackage.get_model("vits14")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"state dict mismatch loading {checkpoint}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected "
                f"(e.g. missing={missing[:3]}, unexpected={unexpected[:3]})"
            )
    return model.to(device).eval()


@torch.inference_mode()
def extract(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total: int,
) -> np.ndarray:
    """Return an (N, 2*embed_dim) fp16 array of [CLS || mean(patch)] features."""
    out: np.ndarray | None = None
    cursor = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            tokens = model.forward_features(batch)

        # DINOv2 returns a dict; fall back to raw tensors for robustness.
        if isinstance(tokens, dict):
            cls = tokens["x_norm_clstoken"]
            patch = tokens["x_norm_patchtokens"].mean(dim=1)
        else:
            cls, patch = tokens[:, 0], tokens[:, 1:].mean(dim=1)

        feats = torch.cat([cls, patch], dim=1).float().cpu().numpy()
        if out is None:
            out = np.empty((total, feats.shape[1]), dtype=np.float16)
        out[cursor : cursor + len(feats)] = feats.astype(np.float16)
        cursor += len(feats)

        if cursor % 51_200 < len(feats):
            print(f"    {cursor}/{total}", flush=True)

    assert out is not None and cursor == total, f"extracted {cursor}, expected {total}"
    return out


def embed_encoder(
    encoder_id: str,
    checkpoint: Path | None,
    batch_size: int = 512,
    num_workers: int = 32,
) -> Path:
    """Extract and cache features for every split. Returns the cache directory."""
    cfg = load_settings()
    seed_rng(strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(checkpoint, device)

    manifest = pd.read_parquet(cfg.manifest_path, columns=["rel_path_256", "split"])
    image_root = cfg.image_root()
    cache_dir = cfg.data_root / "features" / encoder_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEncoder : {encoder_id}")
    print(f"Weights : {checkpoint or 'Meta DINOv2 ViT-S/14 (off-the-shelf)'}")
    print(f"Images  : {image_root}")
    print(f"Cache   : {cache_dir}", flush=True)

    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        target = cache_dir / f"{split.value}.npy"
        if target.exists():
            print(f"  {split.value}: cached, skipping")
            continue

        # Sorted so row i of the cache always maps to row i of the manifest
        # subset -- the alignment every probe and the drift analysis rely on.
        rows = sorted(
            manifest.loc[manifest["split"] == split.value, "rel_path_256"].tolist()
        )
        loader = DataLoader(
            ImageListDataset(image_root, rows),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        print(f"  {split.value}: {len(rows)} images", flush=True)
        feats = extract(model, loader, device, len(rows))
        np.save(target, feats)
        print(f"  {split.value}: wrote {feats.shape} {feats.dtype}", flush=True)

    return cache_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="exported backbone; omit for the off-the-shelf Meta encoder",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args(argv)

    cache = embed_encoder(
        args.encoder_id, args.checkpoint, args.batch_size, args.num_workers
    )
    print(f"\nFeatures cached at {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
