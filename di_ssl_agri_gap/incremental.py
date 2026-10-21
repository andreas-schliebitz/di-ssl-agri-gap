"""Stage 10: chained data-incremental pre-training (phase 2).

N sequential `pretrain` calls over disjoint chunks of the same frozen train
split, each seeded from the previous chunk's exported backbone.

Compute matching falls out for free. Each chunk holds 1/N of the train split, so
100 epochs over it costs EXPECTED_JOINT_STEPS/N steps and N chunks total exactly
the joint arm's budget (219,000 at batch 192). Every image is seen exactly 100
times in both arms -- the cleanest possible matching. It is asserted, not assumed.

Chunk 0 passes `checkpoint=None`, so the model is randomly initialised exactly as
the joint arm was, making the resulting gap attributable to data ordering.

Two documented consequences of `checkpoint=` restoring weights only:
  * warmup and the cosine cycle restart at every chunk (the naive sequential
    behaviour we deliberately accept), so the measured gap is an upper bound on
    the pure data-ordering effect; and
  * the export is the EMA teacher, so each chunk's student is seeded from the
    previous chunk's teacher and a fresh teacher is initialised from it. The EMA
    accumulator does not cross a chunk boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from di_ssl_agri_gap.enums import Arm, Ordering
from di_ssl_agri_gap.pretrain import (
    BATCH_SIZE,
    EPOCHS,
    EXPECTED_JOINT_STEPS,
    RunSpec,
    run_pretrain,
    steps_for,
)
from di_ssl_agri_gap.stream import build_stream, describe
from di_ssl_agri_gap.utils import load_settings, print_gpu_info


def checkpoint_for(run_dir: Path) -> Path:
    """The artefact to hand the next chunk via `checkpoint=`.

    It must be the full Lightning checkpoint, NOT `exported_models/exported_last.pt`.
    LightlyTrain's docs show the exported file being passed to `checkpoint=`, but
    `Checkpoint.from_path` requires a dict with a `state_dict` key and the export
    is a bare state dict, so it dies with `KeyError: 'state_dict'` -- after the
    previous chunk has already spent its full training budget.

    Using `last.ckpt` is also the better choice scientifically: it carries the
    student *and* the EMA teacher plus both projection heads, so the momentum
    encoder and head survive a chunk boundary. Only the optimizer state and LR
    schedulers restart.
    """
    path = run_dir / "checkpoints" / "last.ckpt"
    if not path.exists():
        raise FileNotFoundError(
            f"expected a Lightning checkpoint at {path}; the chain cannot continue"
        )
    return path


def run_arm(
    ordering: Ordering,
    n_chunks: int,
    seed: int = 42,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    devices: int = 2,
    num_workers: int = 48,
    overwrite: bool = False,
    tolerance: float = 0.02,
) -> list[Path]:
    """Run one (ordering, N) configuration end to end. Returns exported backbones."""
    cfg = load_settings()
    chunks = build_stream(ordering, n_chunks, seed)

    planned = [steps_for(len(c), epochs, batch_size) for c in chunks]
    total = sum(planned)
    print(f"\n{'#' * 72}")
    print(f"# {ordering} | N={n_chunks} | seed={seed}")
    print(f"{'#' * 72}")
    print(describe(chunks, ordering).to_string(index=False))
    print(f"\nsteps per chunk : {planned}")
    print(f"total steps     : {total}  (joint arm: {EXPECTED_JOINT_STEPS})")

    # The seasonal arm drops rows with no usable date, so exact equality is not
    # achievable there; a small tolerance keeps the check meaningful without
    # being brittle.
    drift = abs(total - EXPECTED_JOINT_STEPS) / EXPECTED_JOINT_STEPS
    if drift > tolerance:
        raise AssertionError(
            f"{ordering} N={n_chunks} would spend {total} steps against the joint "
            f"arm's {EXPECTED_JOINT_STEPS} ({drift:.1%} off); the comparison would "
            "no longer be compute-matched"
        )
    print(f"compute match   : {drift:.3%} off -- OK")

    exported: list[Path] = []
    previous: Path | None = None
    for chunk in chunks:
        spec = RunSpec(
            arm=Arm.INCREMENTAL,
            ordering=ordering,
            n_chunks=n_chunks,
            chunk_idx=chunk.index,
            seed=seed,
        )

        # Resume support. The full matrix is ~10 days across 28 chunk runs; a
        # death partway must not restart the chain from chunk 0.
        run_dir = cfg.out_dir / spec.encoder_id
        exported_backbone = run_dir / "exported_models" / "exported_last.pt"
        if exported_backbone.exists() and not overwrite:
            print(f"\n=== {spec.encoder_id}: already trained, reusing ===", flush=True)
            previous = checkpoint_for(run_dir)
            exported.append(exported_backbone)
            continue

        # Three states, not two: a directory can exist without an exported
        # backbone, which means an earlier attempt died partway (the NCCL abort
        # left exactly this). LightlyTrain refuses a non-empty `out`, so such a
        # directory has to be overwritten rather than resumed -- its partial
        # checkpoints are not a valid initialisation for anything.
        chunk_overwrite = overwrite or run_dir.exists()
        if run_dir.exists() and not overwrite:
            print(
                f"\n=== {spec.encoder_id}: incomplete previous attempt, retraining ===",
                flush=True,
            )

        run_pretrain(
            spec=spec,
            data=[Path(p) for p in chunk.paths],
            n_images=len(chunk),
            epochs=epochs,
            batch_size=batch_size,
            devices=devices,
            num_workers=num_workers,
            checkpoint=previous,
            overwrite=chunk_overwrite,
        )
        previous = checkpoint_for(run_dir)
        exported.append(exported_backbone)

    record = cfg.out_dir / f"chain_{ordering}_n{n_chunks}_seed{seed}.json"
    record.write_text(
        json.dumps(
            {
                "ordering": str(ordering),
                "n_chunks": n_chunks,
                "seed": seed,
                "steps_per_chunk": planned,
                "total_steps": total,
                "joint_steps": EXPECTED_JOINT_STEPS,
                "encoders": [str(p) for p in exported],
            },
            indent=2,
        )
    )
    print(f"\nChain recorded -> {record}")
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ordering", type=str, choices=[str(o) for o in Ordering], required=True
    )
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and verify the stream, print the plan, train nothing",
    )
    args = parser.parse_args(argv)

    ordering = Ordering(args.ordering)
    if args.dry_run:
        chunks = build_stream(ordering, args.n_chunks, args.seed)
        planned = [steps_for(len(c), args.epochs, args.batch_size) for c in chunks]
        print(describe(chunks, ordering).to_string(index=False))
        print(f"\nsteps per chunk: {planned}")
        print(f"total: {sum(planned)} vs joint {EXPECTED_JOINT_STEPS}")
        return 0

    print_gpu_info()
    run_arm(
        ordering=ordering,
        n_chunks=args.n_chunks,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=args.devices,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
