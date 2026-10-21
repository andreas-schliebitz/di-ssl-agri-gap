"""Seeded, stratified train/val/test splitting.

Returns index arrays rather than dataset objects: the split is persisted as a
column in labels.parquet and is never re-derived at runtime, so that every run
can prove which partition it used via the manifest checksum.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def stratified_split(
    targets: np.ndarray,
    lengths: tuple[float, float, float] = (0.75, 0.15, 0.10),
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test, stratified on `targets`.

    Args:
        targets: integer class label per sample, shape (n,).
        lengths: (train, val, test) fractions, must sum to 1.
        random_seed: controls both splitting passes.

    Returns:
        Three arrays of indices into `targets`.

    Raises:
        AssertionError: if any class fails to appear in all three splits. With
            2,958 species at a minimum of 50 images each this is comfortable,
            but the assertion is the guard that keeps the four nested probes
            well-posed -- a species missing from `test` would silently reduce
            the effective class count and inflate macro-averaged metrics.
    """
    assert len(lengths) == 3, "expected exactly (train, val, test) fractions"
    assert (
        abs(sum(lengths) - 1.0) <= 1e-6
    ), f"fractions must sum to 1, got {sum(lengths)}"

    targets = np.asarray(targets)
    indices = np.arange(len(targets))
    train_frac, val_frac, test_frac = lengths

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=targets,
        shuffle=True,
        random_state=random_seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac / (train_frac + val_frac),
        stratify=targets[train_val_idx],
        shuffle=True,
        random_state=random_seed,
    )

    n_classes = np.unique(targets).size
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        got = np.unique(targets[idx]).size
        assert got == n_classes, (
            f"split '{name}' covers {got}/{n_classes} classes; "
            "stratification failed -- some class is too small to divide"
        )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def balanced_task_ids(
    targets: np.ndarray,
    n_tasks: int,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Assign each sample to one of `n_tasks` chunks, balanced within each class.

    This replaces Continuum's `InstanceIncremental` default partitioner, which
    has two properties that are wrong for a compute-matched comparison
    (`continuum/scenarios/instance_incremental.py`):

      * `random_seed` has no effect on the partition. `self._random_state` is
        constructed *after* `_setup()` runs and `_split_dataset()` never
        shuffles, so it slices each class in the dataset's existing order.
      * Every per-class remainder lands in task 0. It computes
        `n = floor(count / n_tasks)`, writes only `[task*n : (task+1)*n]`, and
        leaves the tail at its zero-initialised value. At N=8 that makes task 0
        roughly 22% oversized.

    Here, each class is shuffled with a seeded RNG and dealt round-robin, so
    per-class chunk sizes differ by at most one and the seed genuinely controls
    the partition. The result is fed back to Continuum as an explicit task-id
    array, which is a documented input path.

    The round-robin counter is carried *across* classes rather than reset per
    class. Resetting would hand every class's remainder to task 0 and reproduce
    a milder form of the very bias described above: with 2,958 species at N=2,
    the ~344 odd-count classes each contribute one extra image to chunk 0, for a
    172-image skew. Carrying the offset rotates remainders across tasks, which
    bounds the global spread at one image.

    Returns:
        int array of task ids in [0, n_tasks), shape (n,).
    """
    assert n_tasks >= 1, "n_tasks must be positive"
    targets = np.asarray(targets)
    rng = np.random.default_rng(random_seed)
    task_ids = np.empty(len(targets), dtype=np.int64)

    offset = 0
    for cls in np.unique(targets):
        (positions,) = np.nonzero(targets == cls)
        if len(positions) < n_tasks:
            raise ValueError(
                f"class {cls} has {len(positions)} samples but {n_tasks} tasks "
                "were requested; some chunk would be empty for this class"
            )
        shuffled = rng.permutation(positions)
        task_ids[shuffled] = (offset + np.arange(len(shuffled))) % n_tasks
        offset = (offset + len(shuffled)) % n_tasks

    return task_ids
