"""Stage 6: DINOv2 self-supervised pre-training.

Wraps `lightly_train.pretrain` so that the joint arm (phase 1) and the chained
incremental arm (phase 2) go through one code path with identical
hyperparameters. The only thing that differs between arms is which images are
handed to `data` and whether `checkpoint` points at a previous chunk.

Why DINOv2 rather than DINOv3: LightlyTrain 0.17.0 cannot train a DINOv3
architecture with an EMA-teacher objective. `_methods/dinov2/dinov2.py` requires
a `DINOv2ViTModelWrapper`, so `method="dinov2"` accepts `dinov2/*` models only;
`dinov3/*` is reachable solely via distillation from a frozen Meta teacher. A
frozen teacher is a fixed target, which would mute exactly the representation
drift this study is trying to measure.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Must be set before the CUDA caching allocator initialises, hence module level:
# DDP re-executes this module for rank 1, so both ranks pick it up.
#
# At 128/GPU the DINO loss allocates a 1280 x 65536 logits tensor per crop group
# and the allocator fragments badly -- a failed run showed 33.4 GB allocated but
# a further 3.4 GB reserved-and-unusable, then OOM'd asking for 2.4 GB with only
# 1.9 GB free. Expandable segments let the allocator grow existing segments
# instead of stranding memory in fixed-size blocks.
# torch 2.9 renamed the variable; both are set so the setting survives either
# direction of a version change.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from di_ssl_agri_gap.enums import Arm, Ordering
from di_ssl_agri_gap.utils import file_sha256, git_sha, load_settings, print_gpu_info

# Adaptation from Meta's pretrained weights was measured and abandoned: across
# seven configurations spanning 100x in learning rate and two augmentation
# recipes, the adapted encoder never came within 14 points of the un-adapted one
# at species level (best 40.40% vs 54.45%). 560k images at ~190 per class is
# simply too small for DINOv2 self-distillation to improve on a 142M-image prior.
#
# From-scratch removes the confound entirely: there is no prior to forget, so
# DINOv2's default hyperparameters are appropriate, and the joint-vs-incremental
# gap becomes a statement purely about data ordering -- which is the paper's
# actual question. Absolute numbers will be far lower; the off-the-shelf row
# becomes context rather than a headroom denominator.
MODEL = "dinov2/vits14-notpretrained"
METHOD = "dinov2"

# Short tag distinguishing encoder ids and result files across the two regimes.
MODEL_TAG = "vits14scratch" if "notpretrained" in MODEL else "vits14"
EPOCHS = 100
PRECISION = "bf16-mixed"
SEED = 42

# Global batch, i.e. 96 per GPU across 2 devices.
#
# Measured peaks for ViT-S/14 with the DINOv2 multi-crop (2x224 + 8x98):
#
#   128/GPU -> ~31.7 GB allocated, ~38.4 GB reserved: 94% of the card
#    96/GPU -> ~29.0 GB
#    64/GPU -> ~19.7 GB
#
# 128/GPU (global 256) is NOT reliably safe. The iBOT loss masks patches with
# mask_probability=0.5 and mask_ratio 0.1-0.5, so the log_softmax allocation in
# `dinov2_loss.forward_masked` is variable-sized per batch. At 94% utilisation an
# OOM becomes a coin flip on the mask draw: one 100-epoch run survived, a later
# identically-configured run died at step 0 with only 489 MB fragmented. Across
# seven long runs that is not a risk worth taking.
#
# 96/GPU leaves ~10 GB of headroom to absorb the largest draw. Any batch size is
# defensible provided the joint and incremental arms share it.
BATCH_SIZE = 192

# The joint arm's total optimizer steps. Every incremental configuration must
# spend the same number across its chunks, which is what makes the comparison a
# statement about data ordering rather than about compute. Derived, not
# hardcoded, so it stays correct if the batch size changes again.
JOINT_TRAIN_IMAGES = 420630


def steps_for(n_images: int, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> int:
    """Optimizer steps for one run. Lightning drops the final partial batch."""
    return (n_images // batch_size) * epochs


EXPECTED_JOINT_STEPS = steps_for(JOINT_TRAIN_IMAGES)


@dataclass(frozen=True)
class RunSpec:
    """Identifies one pre-training run and produces its name and tags."""

    arm: Arm
    ordering: Ordering | None = None
    n_chunks: int = 1
    chunk_idx: int = 0
    seed: int = SEED
    suffix: str = ""

    @property
    def encoder_id(self) -> str:
        if self.arm is Arm.JOINT:
            return f"joint_{MODEL_TAG}_ep{EPOCHS}_seed{self.seed}{self.suffix}"
        if self.arm is Arm.OFFTHESHELF:
            return "offtheshelf_vits14"
        return (
            f"{self.ordering}_n{self.n_chunks}_chunk{self.chunk_idx}"
            f"_{MODEL_TAG}_seed{self.seed}"
        )

    @property
    def tags(self) -> dict[str, str]:
        return {
            "arm": str(self.arm),
            "ordering": str(self.ordering) if self.ordering else "none",
            "n_chunks": str(self.n_chunks),
            "chunk_idx": str(self.chunk_idx),
            "seed": str(self.seed),
            "encoder_id": self.encoder_id,
            "model": MODEL,
            "method": METHOD,
            "git_sha": git_sha(),
        }


def run_pretrain(
    spec: RunSpec,
    data: Path | Sequence[Path],
    n_images: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    devices: int = 2,
    num_workers: int = 32,
    checkpoint: Path | None = None,
    overwrite: bool = False,
    extra_method_args: dict[str, Any] | None = None,
    resume_interrupted: bool = False,
    base_lr: float | None = None,
    checkpoint_every_n_epochs: int | None = None,
    transform_args: dict[str, Any] | None = None,
    ddp_timeout_hours: int = 4,
) -> Path:
    """Run one pre-training stage and return the exported backbone path.

    `resume_interrupted` continues a crashed run from `out/<id>/checkpoints/last.ckpt`,
    restoring optimizer and scheduler state. Every other parameter must be
    unchanged from the original invocation, which is why the supervisor re-issues
    the same command line rather than rebuilding it.
    """
    import lightly_train

    from di_ssl_agri_gap.utils import seed_rng

    # Establish the RNG/determinism state this stage needs rather than inheriting
    # whatever ran before it in the same process. `probe_encoder` enables strict
    # determinism globally and does not restore it, so a driver that interleaves
    # probing and pre-training (the LR sweep, and phase 2 if probes are run
    # between chunks) would otherwise carry it in here -- and DINOv2's multi-crop
    # has no deterministic bicubic-upsample backward, so training dies at step 0
    # with `upsample_bicubic2d_aa_backward_out_cuda does not have a deterministic
    # implementation`.
    seed_rng(spec.seed, strict=False)

    cfg = load_settings()
    out_dir = cfg.out_dir / spec.encoder_id
    expected_steps = steps_for(n_images, epochs, batch_size)

    tags = spec.tags | {
        "n_images": str(n_images),
        "epochs": str(epochs),
        "batch_size": str(batch_size),
        "expected_steps": str(expected_steps),
        "manifest_sha256": file_sha256(cfg.manifest_path)[:16],
    }

    print(f"\n{'=' * 72}\n{spec.encoder_id}\n{'=' * 72}")
    print(f"  images   : {n_images}")
    print(f"  steps    : {expected_steps} ({epochs} epochs @ batch {batch_size})")
    init = (
        str(checkpoint)
        if checkpoint
        else f"{MODEL} "
        + ("(random)" if "notpretrained" in MODEL else "(Meta LVD-142M weights)")
    )
    print(f"  init     : {init}")
    print(f"  out      : {out_dir}", flush=True)

    # DINOv2's default base lr of 0.004 is a *from-scratch* rate. Applied to
    # Meta-pretrained weights it caused catastrophic forgetting: after 100 epochs
    # the encoder scored below the un-adapted model at every granularity
    # (species 39.81% vs 54.45%), with a median relative weight change of 0.95.
    # Adapting a pretrained encoder needs a far smaller rate.
    optim_args = {"lr": base_lr} if base_lr is not None else None
    if base_lr is not None:
        # LightlyTrain multiplies by sqrt(global_batch / reference_batch_size).
        effective = base_lr * (batch_size / 1024) ** 0.5
        tags["base_lr"] = str(base_lr)
        tags["effective_peak_lr"] = f"{effective:.3e}"
        print(f"  lr       : base {base_lr:g} -> peak {effective:.3e}")

    callbacks: dict[str, Any] | None = None
    if checkpoint_every_n_epochs:
        # save_top_k=-1 keeps every periodic checkpoint, so the degradation
        # trajectory can be probed after the fact instead of inferred.
        callbacks = {
            "model_checkpoint": {
                "every_n_epochs": checkpoint_every_n_epochs,
                "save_top_k": -1,
                "save_last": True,
            }
        }

    # NCCL's default collective timeout is 10 minutes, which is not enough at a
    # chunk boundary in the incremental arm: rank 0 has to load a ~700 MB
    # checkpoint and index tens of thousands of file paths off NFS while rank 1
    # is already blocked on a broadcast. A real run died exactly this way --
    # "Watchdog caught collective operation timeout: WorkNCCL(OpType=BROADCAST)"
    # -- taking down the process group and leaving zombie CUDA contexts behind.
    #
    # find_unused_parameters mirrors what LightlyTrain sets itself; some methods
    # in the same code path legitimately have unused parameters.
    strategy: Any = "auto"
    if devices > 1:
        from datetime import timedelta

        # MUST come from `pytorch_lightning`, not `lightning.pytorch`. Both are
        # installed and they define *distinct* Strategy classes, so LightlyTrain's
        # pydantic `is-instance[Strategy]` check (it imports from
        # pytorch_lightning.strategies.strategy) rejects the lightning.pytorch one.
        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(hours=ddp_timeout_hours),
        )
        tags["ddp_timeout_hours"] = str(ddp_timeout_hours)

    loggers: dict[str, Any] = {"jsonl": {}}
    if cfg.use_mlflow:
        loggers["mlflow"] = {
            "experiment_name": cfg.experiment_name,
            "run_name": spec.encoder_id,
            "tags": tags,
        }

    lightly_train.pretrain(
        out=str(out_dir),
        data=str(data) if isinstance(data, (str, Path)) else [str(p) for p in data],
        model=MODEL,
        method=METHOD,
        method_args=extra_method_args,
        epochs=epochs,
        batch_size=batch_size,
        devices=devices,
        num_workers=num_workers,
        strategy=strategy,
        precision=PRECISION,
        seed=spec.seed,
        checkpoint=str(checkpoint) if checkpoint else None,
        overwrite=overwrite,
        resume_interrupted=resume_interrupted,
        optim_args=optim_args,
        transform_args=transform_args,
        callbacks=callbacks,
        loggers=loggers,
        # Do NOT pass trainer_args={"max_steps": ...} here. Under DDP it
        # deadlocks: the two ranks evaluate the stop condition at different
        # points around the distributed metric reduction in
        # `lightly_train/_methods/method.py`, so one rank exits the loop and
        # calls destroy_process_group while the other blocks forever on the
        # sync barrier inside training_step. Bound runs by `epochs` instead --
        # which is also required for the LR and teacher-temperature schedules
        # to complete. To benchmark, shrink the dataset, not the step count.
    )

    exported = out_dir / "exported_models" / "exported_last.pt"
    if not exported.exists():
        raise FileNotFoundError(f"expected an exported backbone at {exported}")

    (out_dir / "run_spec.json").write_text(json.dumps(tags, indent=2))
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="base learning rate before sqrt batch scaling (DINOv2 default 0.004, "
        "which is a from-scratch rate and forgets when adapting pretrained weights)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="keep a checkpoint every N epochs so the trajectory can be probed",
    )
    parser.add_argument("--run-suffix", default="", help="disambiguate sweep runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a crashed run from out/<id>/checkpoints/last.ckpt "
        "(all other arguments must match the original invocation)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="pre-train on the first N train images only (smoke test)",
    )
    args = parser.parse_args(argv)

    import pandas as pd

    cfg = load_settings()
    print_gpu_info()

    train_dir = cfg.shm_root / "img256_train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"{train_dir} does not exist -- run stage_shm.py first. Pre-training "
            "must read the train-only view so val/test are never seen."
        )

    manifest = pd.read_parquet(cfg.manifest_path, columns=["rel_path_256", "split"])
    n_train = int((manifest["split"] == "train").sum())

    if args.subset:
        rows = manifest.loc[manifest["split"] == "train", "rel_path_256"].head(
            args.subset
        )
        data: Any = [train_dir / r for r in rows]
        n_images = len(rows)
        spec = RunSpec(arm=Arm.JOINT, seed=args.seed, suffix=args.run_suffix)
        print(f"SMOKE TEST on {n_images} images (use --overwrite to rerun)")
    else:
        data = train_dir
        n_images = n_train
        spec = RunSpec(arm=Arm.JOINT, seed=args.seed, suffix=args.run_suffix)

        steps = steps_for(n_images, args.epochs, args.batch_size)
        if n_images != JOINT_TRAIN_IMAGES:
            print(
                f"WARNING: train split has {n_images} images, expected "
                f"{JOINT_TRAIN_IMAGES}"
            )
        if steps != EXPECTED_JOINT_STEPS:
            print(
                f"WARNING: {steps} steps != the {EXPECTED_JOINT_STEPS} the "
                "incremental arm is budgeted against"
            )

    import time

    started = time.perf_counter()
    exported = run_pretrain(
        spec=spec,
        data=data,
        n_images=n_images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=args.devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        resume_interrupted=args.resume,
        base_lr=args.lr,
        checkpoint_every_n_epochs=args.checkpoint_every,
    )
    elapsed = time.perf_counter() - started
    print(f"\nExported backbone: {exported}")

    if args.subset:
        report_throughput(
            out_dir=cfg.out_dir / spec.encoder_id,
            wall_seconds=elapsed,
            ran_steps=steps_for(n_images, args.epochs, args.batch_size),
            full_steps=steps_for(n_train, EPOCHS, args.batch_size),
            batch_size=args.batch_size,
        )
    return 0


def report_throughput(
    out_dir: Path,
    wall_seconds: float,
    ran_steps: int,
    full_steps: int,
    batch_size: int,
) -> None:
    """Project full-run wall clock from a short epoch-bounded run.

    Two estimates are printed. The wall-clock one includes fixed startup (model
    build, DDP handshake, dataset indexing) and so over-estimates. The
    steady-state one uses LightlyTrain's own `profiling/batch_time`, which
    excludes startup and is the number to trust for a long run.
    """
    import json

    print(
        f"\nBENCHMARK  {ran_steps} steps in {wall_seconds:.0f}s wall "
        f"({wall_seconds / ran_steps:.3f}s/step incl. startup)"
    )
    print(
        f"           naive projection: {full_steps} steps ~ "
        f"{full_steps * wall_seconds / ran_steps / 3600:.1f} h"
    )

    metrics = out_dir / "metrics.jsonl"
    if not metrics.exists():
        return
    times = [
        json.loads(line)["profiling/batch_time"]
        for line in metrics.read_text().splitlines()
        if "profiling/batch_time" in line
    ]
    if len(times) < 5:
        return
    # Drop the first few steps: they carry cudnn autotuning and worker warmup.
    steady = sorted(times[3:])[len(times[3:]) // 2]
    print(
        f"           steady state: {steady:.3f}s/step "
        f"({batch_size / steady:.0f} img/s)"
    )
    print(
        f"           STEADY PROJECTION: {full_steps} steps ~ "
        f"{full_steps * steady / 3600:.1f} h"
    )


if __name__ == "__main__":
    raise SystemExit(main())
