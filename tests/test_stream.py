"""Phase-2 stream invariants, including the Continuum work-arounds.

The compute-matched comparison rests on these: disjoint chunks that exactly
cover the train split, sized within one image of each other, and a seasonal arm
that actually carries distribution shift rather than being IID in disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from di_ssl_agri_gap.enums import Ordering, Split
from di_ssl_agri_gap.pretrain import EXPECTED_JOINT_STEPS, steps_for
from di_ssl_agri_gap.stream import build_stream, seasonal_task_ids
from di_ssl_agri_gap.utils import load_settings

N_CHUNKS = [2, 4, 8]


@pytest.fixture(scope="module")
def train_manifest() -> pd.DataFrame:
    cfg = load_settings()
    if not cfg.manifest_path.exists():
        pytest.skip("manifest not built yet")
    return pd.read_parquet(
        cfg.manifest_path, columns=["rel_path_256", "split", "species_id", "month"]
    ).query(f"split == '{Split.TRAIN.value}'")


def test_continuum_accepts_explicit_task_ids():
    """The documented `elif t is not None` branch, which we depend on.

    Continuum's own partitioner ignores `random_seed` and dumps every per-class
    remainder into task 0, so we compute task ids ourselves and pass them in.
    """
    from continuum import InstanceIncremental
    from continuum.datasets import InMemoryDataset
    from continuum.tasks import TaskType

    x = np.array([f"/tmp/{i}.jpg" for i in range(12)])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    t = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    scenario = InstanceIncremental(
        InMemoryDataset(x, y, t, data_type=TaskType.IMAGE_PATH), nb_tasks=None
    )
    assert scenario.nb_tasks == 3
    assert [len(ts) for ts in scenario] == [4, 4, 4]


@pytest.mark.parametrize("ordering", list(Ordering))
@pytest.mark.parametrize("n", N_CHUNKS)
def test_chunks_are_disjoint_and_balanced(ordering, n, train_manifest):
    chunks = build_stream(ordering, n, seed=42)
    assert len(chunks) == n

    seen: set[str] = set()
    for chunk in chunks:
        assert not seen & set(chunk.paths), f"chunk {chunk.index} overlaps"
        seen.update(chunk.paths)

    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= n, sizes


@pytest.mark.parametrize("n", N_CHUNKS)
def test_iid_chunks_cover_the_whole_train_split(n, train_manifest):
    chunks = build_stream(Ordering.IID, n, seed=42)
    assert sum(len(c) for c in chunks) == len(train_manifest)


@pytest.mark.parametrize("n", N_CHUNKS)
def test_iid_chunks_contain_every_species(n):
    """IID means same label space throughout -- that is what makes it IID."""
    for chunk in build_stream(Ordering.IID, n, seed=42):
        assert chunk.n_species == 2958, (n, chunk.index, chunk.n_species)


@pytest.mark.parametrize("n", [4, 8])
def test_seasonal_chunks_genuinely_lose_species(n):
    """If the seasonal arm covered every species it would carry no shift.

    A chunk missing species is the mechanism the arm exists to create; without
    it there is nothing for the granularity hypothesis to bite on.
    """
    coverage = [c.n_species for c in build_stream(Ordering.SEASONAL, n, seed=42)]
    assert max(coverage) < 2958, coverage


def test_seasonal_blocks_are_contiguous_in_the_calendar():
    months = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    ids = seasonal_task_ids(months, 3)
    for task in range(3):
        block = months[ids == task]
        assert block.max() - block.min() <= 1, block
    # Later chunks must hold later months.
    means = [months[ids == t].mean() for t in range(3)]
    assert means == sorted(means)


def test_seasonal_excludes_rows_without_a_usable_date():
    months = np.array([1, -1, 5, -1, 9, 12])
    ids = seasonal_task_ids(months, 2)
    assert (ids[months <= 0] == -1).all()
    assert set(ids[months > 0]) == {0, 1}


@pytest.mark.parametrize("ordering", list(Ordering))
@pytest.mark.parametrize("n", N_CHUNKS)
def test_every_configuration_is_compute_matched(ordering, n):
    """Total optimizer steps must match the joint arm to within 2%.

    Exact equality is impossible for the seasonal arm, which drops the 0.39% of
    rows with no usable date.
    """
    chunks = build_stream(ordering, n, seed=42)
    total = sum(steps_for(len(c)) for c in chunks)
    drift = abs(total - EXPECTED_JOINT_STEPS) / EXPECTED_JOINT_STEPS
    assert drift <= 0.02, f"{ordering} N={n}: {total} vs {EXPECTED_JOINT_STEPS}"


def test_stream_is_reproducible_under_the_same_seed():
    a = build_stream(Ordering.IID, 4, seed=42)
    b = build_stream(Ordering.IID, 4, seed=42)
    assert [c.paths for c in a] == [c.paths for c in b]


def test_stream_changes_with_the_seed():
    a = build_stream(Ordering.IID, 4, seed=42)
    c = build_stream(Ordering.IID, 4, seed=7)
    assert [x.paths for x in a] != [x.paths for x in c]
