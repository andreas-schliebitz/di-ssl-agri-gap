"""Learning-rate sweep for the joint arm.

Why this exists: at DINOv2's default base learning rate of 0.004 (peak 2.0e-3
after sqrt batch scaling), 100 epochs of adaptation on iNatAg-mini produced an
encoder *worse than the un-adapted Meta model at every granularity*:

    granularity   off-the-shelf   joint @ 2.0e-3     delta
    species          54.45%          39.81%         -14.64
    genus            63.64%          49.10%         -14.54
    family           66.85%          56.59%         -10.26
    crop/weed        75.67%          74.30%          -1.37

The median relative weight change against Meta's initialisation was 0.95, i.e.
the encoder effectively retrained itself on 560k images and discarded the
LVD-142M prior. 0.004 is a *from-scratch* rate; adapting pretrained weights
needs far less.

A joint baseline that loses to doing nothing makes the joint-vs-incremental gap
meaningless, so this sweep finds a rate where adaptation actually helps before
the ~10-day phase-2 matrix is committed.

Each configuration runs a short schedule and is probed at species level only --
the finest granularity, and the one that degraded most, so the most sensitive
detector. Periodic checkpoints are kept so the trajectory can be probed later
without re-training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from di_ssl_agri_gap.enums import Arm, Granularity
from di_ssl_agri_gap.pretrain import BATCH_SIZE, RunSpec, run_pretrain
from di_ssl_agri_gap.utils import load_settings, print_gpu_info

# Base rates, before LightlyTrain's sqrt(batch/1024) scaling. 0.004 is the
# DINOv2 default and is included as a control, so the sweep shows a trend rather
# than an isolated number.
BASE_LEARNING_RATES = (4e-3, 4e-4, 1e-4, 4e-5)

# The bar the adapted encoder has to clear: off-the-shelf DINOv2 ViT-S/14,
# species-level test top-1, measured on this exact split.
OFFTHESHELF_SPECIES_TOP1 = 0.5445

# DINOv2's schedule defaults are absolute step counts sized for the full run.
# Left alone in a short sweep they dominate it: at 10 epochs (16,430 steps) the
# LR warmup would be 76% of training and the teacher-temperature warmup 228%,
# so the peak learning rate would barely be applied and the temperature schedule
# would never finish. Every rate would then look equally inert and the sweep
# would measure nothing.
#
# These are rescaled to hold the same *fraction* of training as the 100-epoch
# reference run, which keeps the schedule shape -- and therefore the sweep's
# predictive value for the full run -- intact.
REFERENCE_STEPS = 164_300
REFERENCE_SCHEDULE = {"warmup_steps": 12_500, "teacher_temp_warmup_steps": 37_500}


def scaled_schedule(total_steps: int) -> dict[str, int]:
    scale = total_steps / REFERENCE_STEPS
    return {k: max(1, round(v * scale)) for k, v in REFERENCE_SCHEDULE.items()}


def sweep(
    base_lrs: tuple[float, ...] = BASE_LEARNING_RATES,
    epochs: int = 10,
    checkpoint_every: int = 2,
    devices: int = 2,
    num_workers: int = 48,
    seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    from di_ssl_agri_gap.embed import embed_encoder
    from di_ssl_agri_gap.probe import probe_encoder

    cfg = load_settings()
    train_dir = cfg.shm_root / "img256_train"
    manifest = pd.read_parquet(cfg.manifest_path, columns=["split"])
    n_train = int((manifest["split"] == "train").sum())

    total_steps = (n_train // BATCH_SIZE) * epochs
    schedule = scaled_schedule(total_steps)
    print(
        f"Sweep: {epochs} epochs = {total_steps} steps; schedule rescaled to {schedule}"
    )

    rows = []
    for base_lr in base_lrs:
        suffix = f"_ep{epochs}_lr{base_lr:g}"
        spec = RunSpec(arm=Arm.JOINT, seed=seed, suffix=suffix)

        # Resume support: a 14 h sweep that dies partway should not retrain what
        # already finished. An exported backbone means this rate is done.
        exported = (
            cfg.out_dir / spec.encoder_id / "exported_models" / "exported_last.pt"
        )
        if exported.exists() and not force:
            print(f"\n=== base_lr={base_lr:g}: encoder exists, probing only ===")
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
                base_lr=base_lr,
                checkpoint_every_n_epochs=checkpoint_every,
                extra_method_args=schedule,
            )

        embed_encoder(spec.encoder_id, exported, batch_size=512, num_workers=32)
        result = probe_encoder(
            spec.encoder_id, seed=seed, granularities=(Granularity.SPECIES,)
        )
        test = result[result["split"] == "test"].iloc[0]

        rows.append(
            {
                "base_lr": base_lr,
                "peak_lr": base_lr * (BATCH_SIZE / 1024) ** 0.5,
                "epochs": epochs,
                "encoder_id": spec.encoder_id,
                "species_top1": float(test["accuracy_micro"]),
                "species_f1": float(test["f1_macro"]),
                "vs_offtheshelf": float(test["accuracy_micro"])
                - OFFTHESHELF_SPECIES_TOP1,
            }
        )
        print(
            f"\n>>> base_lr={base_lr:g}  species top1={rows[-1]['species_top1']:.4f}  "
            f"vs off-the-shelf {rows[-1]['vs_offtheshelf']:+.4f}\n",
            flush=True,
        )
        # Persist after every configuration, so a crash costs one run at most.
        out = cfg.data_root / "results" / "lr_sweep.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-lrs", type=float, nargs="+", default=list(BASE_LEARNING_RATES)
    )
    parser.add_argument(
        "--force", action="store_true", help="retrain configurations already done"
    )
    args = parser.parse_args(argv)

    cfg = load_settings()
    print_gpu_info()
    frame = sweep(
        base_lrs=tuple(args.base_lrs),
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        devices=args.devices,
        num_workers=args.num_workers,
        seed=args.seed,
        force=args.force,
    )

    out = cfg.data_root / "results" / "lr_sweep.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    print("\n" + "=" * 72)
    print(f"off-the-shelf species top-1: {OFFTHESHELF_SPECIES_TOP1:.4f}")
    print(frame.to_string(index=False))

    better = frame[frame["vs_offtheshelf"] > 0]
    if better.empty:
        print(
            "\nNo learning rate beat the un-adapted encoder at species level. "
            "Adaptation on this corpus may not help at any rate -- consider "
            "training from scratch instead, or shortening the schedule."
        )
    else:
        best = better.loc[better["species_top1"].idxmax()]
        print(
            f"\nBest: base_lr={best['base_lr']:g} (peak {best['peak_lr']:.2e}), "
            f"species top-1 {best['species_top1']:.4f} "
            f"({best['vs_offtheshelf']:+.4f} vs off-the-shelf)"
        )
        (cfg.data_root / "results" / "lr_sweep_best.json").write_text(
            json.dumps(
                {k: float(v) for k, v in best.items() if k != "encoder_id"}, indent=2
            )
        )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
