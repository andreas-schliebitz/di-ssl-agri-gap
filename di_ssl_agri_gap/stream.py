"""Stage 9: build the data-incremental stream (phase 2).

Continuum's `InstanceIncremental` supplies the scenario semantics, but its
default partitioner is not used. Two properties of
`continuum/scenarios/instance_incremental.py` make it unsuitable for a
compute-matched comparison:

  * `random_seed` has no effect on the partition. `self._random_state` is
    constructed *after* `_setup()` runs, and `_split_dataset()` never shuffles --
    it slices each class in the dataset's existing order. Changing the seed
    changes nothing.

  * Every per-class remainder lands in task 0. It computes
    `n = floor(count / nb_tasks)`, writes only `[task*n : (task+1)*n]`, and
    leaves the tail at its zero-initialised value. At N=8 that makes task 0
    roughly 22% oversized, which would silently break compute matching and hand
    the first chunk an advantage.

So the task-id array is computed here (seeded, balanced) and handed to Continuum
through `InMemoryDataset`, which `_setup` consumes via its documented
`elif t is not None` branch. The scenario, its iteration semantics and its
citation all remain Continuum's.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from di_ssl_agri_gap.enums import Ordering, Split
from di_ssl_agri_gap.splits import balanced_task_ids
from di_ssl_agri_gap.utils import load_settings


@dataclass(frozen=True)
class Chunk:
    """One task in the stream."""

    index: int
    paths: list[str]
    species_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def n_species(self) -> int:
        return int(np.unique(self.species_ids).size)


def seasonal_task_ids(months: np.ndarray, n_tasks: int) -> np.ndarray:
    """Cut the hemisphere-corrected calendar into N equal-count blocks.

    Ordering is by month-of-year, ignoring the year. Absolute-date ordering was
    measured and rejected: 91.5% of iNatAg-mini falls in 2020-2025 and 50.9% in
    2024 alone, so equal-count date blocks collapse into a single year after the
    first boundary and the arm degenerates into IID.

    Ties within a month are broken deterministically by position, so the split
    is reproducible without needing a seed.

    Rows with month <= 0 (no usable date) get task id -1 and are excluded by the
    caller.
    """
    task_ids = np.full(len(months), -1, dtype=np.int64)
    usable = np.nonzero(months > 0)[0]
    # Stable sort keeps same-month samples in manifest order.
    order = usable[np.argsort(months[usable], kind="stable")]
    # array_split handles a non-divisible count by making the earliest blocks at
    # most one larger, so chunk sizes differ by at most one.
    for task, block in enumerate(np.array_split(order, n_tasks)):
        task_ids[block] = task
    return task_ids


def build_task_ids(
    manifest: pd.DataFrame, ordering: Ordering, n_tasks: int, seed: int
) -> np.ndarray:
    if ordering is Ordering.IID:
        return balanced_task_ids(
            manifest["species_id"].to_numpy(), n_tasks, random_seed=seed
        )
    if ordering is Ordering.SEASONAL:
        return seasonal_task_ids(manifest["month"].to_numpy(), n_tasks)
    raise ValueError(f"unknown ordering: {ordering}")


def build_stream(
    ordering: Ordering,
    n_tasks: int,
    seed: int = 42,
    image_root: Path | None = None,
) -> list[Chunk]:
    """Partition the train split into `n_tasks` chunks via Continuum."""
    from continuum import InstanceIncremental
    from continuum.datasets import InMemoryDataset
    from continuum.tasks import TaskType

    cfg = load_settings()
    root = image_root or (cfg.shm_root / "img256")
    manifest = (
        pd.read_parquet(
            cfg.manifest_path, columns=["rel_path_256", "split", "species_id", "month"]
        )
        .query(f"split == '{Split.TRAIN.value}'")
        .sort_values("rel_path_256")
        .reset_index(drop=True)
    )

    task_ids = build_task_ids(manifest, ordering, n_tasks, seed)
    keep = task_ids >= 0
    dropped = int((~keep).sum())

    paths = np.array([str(root / r) for r in manifest["rel_path_256"]])
    species = manifest["species_id"].to_numpy()

    dataset = InMemoryDataset(
        paths[keep], species[keep], task_ids[keep], data_type=TaskType.IMAGE_PATH
    )
    # nb_tasks=None routes _setup through its `elif t is not None` branch, which
    # uses our task ids verbatim instead of re-partitioning.
    scenario = InstanceIncremental(dataset, nb_tasks=None)

    chunks = [
        Chunk(
            index=i,
            paths=[str(p) for p in taskset._x],
            species_ids=np.asarray(taskset._y),
        )
        for i, taskset in enumerate(scenario)
    ]

    if dropped:
        print(f"  excluded {dropped} row(s) with no usable date ({ordering})")
    verify_stream(chunks, manifest, dropped)
    return chunks


def verify_stream(chunks: list[Chunk], manifest: pd.DataFrame, dropped: int) -> None:
    """Assert the partition invariants the comparison depends on."""
    sizes = [len(c) for c in chunks]
    total = sum(sizes)
    expected = len(manifest) - dropped
    if total != expected:
        raise AssertionError(f"chunks cover {total} rows, expected {expected}")

    seen: set[str] = set()
    for chunk in chunks:
        overlap = seen & set(chunk.paths)
        if overlap:
            raise AssertionError(
                f"chunk {chunk.index} overlaps earlier chunks on {len(overlap)} paths"
            )
        seen.update(chunk.paths)

    spread = max(sizes) - min(sizes)
    if spread > len(chunks):
        raise AssertionError(
            f"chunk sizes differ by {spread} (max {max(sizes)}, min {min(sizes)}); "
            "an unbalanced partition breaks compute matching"
        )


def describe(chunks: list[Chunk], ordering: Ordering) -> pd.DataFrame:
    rows = []
    for c in chunks:
        rows.append(
            {
                "ordering": str(ordering),
                "n_chunks": len(chunks),
                "chunk_idx": c.index,
                "n_images": len(c),
                "n_species": c.n_species,
                "species_coverage": round(c.n_species / 2958, 4),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_settings()
    frames = []
    for ordering in Ordering:
        for n in args.n_chunks:
            print(f"\n=== {ordering} N={n} ===")
            chunks = build_stream(ordering, n, args.seed)
            frame = describe(chunks, ordering)
            print(frame.to_string(index=False))
            frames.append(frame)

    out = cfg.data_root / "results" / "stream_coverage.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print(f"\nCoverage -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
