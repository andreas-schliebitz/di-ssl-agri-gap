"""Does DINOv2's augmentation recipe destroy the colour cues fine-grained
botany depends on?

The learning-rate sweep ruled out the obvious explanation for why adapting a
pretrained DINOv2 on iNatAg-mini makes it *worse* than not adapting at all:

    base_lr   peak_lr   species top-1   vs off-the-shelf (54.45%)
    4e-3      2.0e-3        10.78%           -43.67
    4e-4      2.0e-4        33.64%           -20.81
    1e-4      5.0e-5        36.32%           -18.13
    4e-5      2.0e-5        38.63%           -15.82

Lower is monotonically better but plainly asymptotic (+22.9, +2.7, +2.3 per
reduction), and the 100-epoch run at the *default* rate landed at 39.81% -- so
every configuration converges to 38-40% regardless of rate or schedule length.
That is not overshooting; the objective is settling into a representation that is
genuinely worse for fine-grained discrimination than Meta's.

The hypothesis tested here: DINOv2 optimises *augmentation invariance*, and its
default recipe asks the model to ignore colour --

    color_jitter: brightness 0.8, contrast 0.8, saturation 0.4, hue 0.2, p=0.8
    random_gray_scale: 0.2
    solarize (global view 1): p=0.2

Flower and leaf colour is a primary cue for separating 2,958 plant species.
Training for invariance to it should hurt species far more than crop/weed, which
matches the granularity-dependent damage observed (-14.6 species, -1.4 crop/weed).

Two runs, both at the sweep's best rate (base 4e-5) so the only change is the
augmentation:

  A  softened photometrics -- grayscale and solarize off, jitter roughly halved
  B  A, plus student_freeze_backbone_steps=2500 so the randomly initialised
     22.4M-parameter DINO head warms up against a frozen backbone instead of
     pushing noise gradients into pretrained weights from step 0

The bar is the un-adapted encoder: 54.45% species top-1.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from di_ssl_agri_gap.enums import Arm, Granularity
from di_ssl_agri_gap.lr_sweep import (
    BATCH_SIZE,
    OFFTHESHELF_SPECIES_TOP1,
    scaled_schedule,
)
from di_ssl_agri_gap.pretrain import RunSpec, run_pretrain
from di_ssl_agri_gap.utils import load_settings, print_gpu_info

BEST_BASE_LR = 4e-5

# Colour-preserving photometrics. Geometry (multi-crop, flips, resize) and blur
# are left untouched: they are not suspected, and changing them too would make
# the result uninterpretable.
SOFT_PHOTOMETRIC: dict[str, Any] = {
    "color_jitter": {
        "prob": 0.4,
        "strength": 0.5,
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.05,
    },
    "random_gray_scale": 0.0,
    "global_view_1": {"solarize": None},
}

CONFIGURATIONS: list[dict[str, Any]] = [
    {
        "name": "soft_aug",
        "transform_args": SOFT_PHOTOMETRIC,
        "method_args": {},
    },
    {
        "name": "soft_aug_freeze",
        "transform_args": SOFT_PHOTOMETRIC,
        "method_args": {"student_freeze_backbone_steps": 2500},
    },
]


def run(
    epochs: int = 10,
    devices: int = 2,
    num_workers: int = 48,
    seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    from di_ssl_agri_gap.embed import embed_encoder
    from di_ssl_agri_gap.probe import probe_encoder

    cfg = load_settings()
    train_dir = cfg.shm_root / "img256_train"
    n_train = int(
        (
            pd.read_parquet(cfg.manifest_path, columns=["split"])["split"] == "train"
        ).sum()
    )
    schedule = scaled_schedule((n_train // BATCH_SIZE) * epochs)

    rows = []
    for config in CONFIGURATIONS:
        suffix = f"_ep{epochs}_lr{BEST_BASE_LR:g}_{config['name']}"
        spec = RunSpec(arm=Arm.JOINT, seed=seed, suffix=suffix)
        exported = (
            cfg.out_dir / spec.encoder_id / "exported_models" / "exported_last.pt"
        )
        if exported.exists() and not force:
            print(f"\n=== {config['name']}: encoder exists, probing only ===")
        else:
            exported = run_pretrain(
                spec=spec,
                data=train_dir,
                n_images=n_train,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                devices=devices,
                num_workers=num_workers,
                overwrite=True,
                base_lr=BEST_BASE_LR,
                checkpoint_every_n_epochs=2,
                extra_method_args=schedule | config["method_args"],
                transform_args=config["transform_args"],
            )

        embed_encoder(spec.encoder_id, exported, batch_size=512, num_workers=32)
        result = probe_encoder(
            spec.encoder_id, seed=seed, granularities=(Granularity.SPECIES,)
        )
        test = result[result["split"] == "test"].iloc[0]
        top1 = float(test["accuracy_micro"])

        rows.append(
            {
                "config": config["name"],
                "base_lr": BEST_BASE_LR,
                "epochs": epochs,
                "encoder_id": spec.encoder_id,
                "species_top1": top1,
                "species_f1": float(test["f1_macro"]),
                "vs_offtheshelf": top1 - OFFTHESHELF_SPECIES_TOP1,
                "vs_default_aug": top1 - 0.386306,  # base 4e-5, default recipe
            }
        )
        print(
            f"\n>>> {config['name']}: species top1={top1:.4f}  "
            f"vs off-the-shelf {rows[-1]['vs_offtheshelf']:+.4f}  "
            f"vs default aug {rows[-1]['vs_default_aug']:+.4f}\n",
            flush=True,
        )
        out = cfg.data_root / "results" / "aug_test.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print_gpu_info()
    frame = run(
        epochs=args.epochs,
        devices=args.devices,
        num_workers=args.num_workers,
        seed=args.seed,
        force=args.force,
    )

    print("\n" + "=" * 72)
    print(f"off-the-shelf species top-1 : {OFFTHESHELF_SPECIES_TOP1:.4f}")
    print(f"default recipe @ base 4e-5   : 0.3863")
    print(frame.to_string(index=False))

    winners = frame[frame["vs_offtheshelf"] > 0]
    if winners.empty:
        best = frame.loc[frame["species_top1"].idxmax()]
        print(
            f"\nNeither configuration cleared the un-adapted encoder. Best was "
            f"{best['config']} at {best['species_top1']:.4f} "
            f"({best['vs_default_aug']:+.4f} vs the default recipe).\n"
            "If the improvement over the default recipe is small, augmentation is "
            "not the dominant cause and adaptation on this corpus is a dead end -- "
            "switch to from-scratch pre-training."
        )
    else:
        best = winners.loc[winners["species_top1"].idxmax()]
        print(
            f"\nHYPOTHESIS CONFIRMED: {best['config']} reaches "
            f"{best['species_top1']:.4f} ({best['vs_offtheshelf']:+.4f} vs "
            "off-the-shelf). Re-run the joint arm for 100 epochs with this recipe."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
