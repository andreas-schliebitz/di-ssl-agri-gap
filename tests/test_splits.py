"""Split and stream-partition invariants.

These are the checks that keep the joint-vs-incremental comparison honest: if a
partition is unbalanced or a class goes missing from a split, the measured gap
stops being about data ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

from di_ssl_agri_gap.splits import balanced_task_ids, stratified_split


@pytest.fixture
def targets() -> np.ndarray:
    # 300 classes with deliberately uneven, mostly odd counts.
    return np.repeat(np.arange(300), [50 + (i % 7) for i in range(300)])


def test_split_fractions_and_disjointness(targets):
    train, val, test = stratified_split(targets, (0.75, 0.15, 0.10), 42)
    total = len(targets)
    assert len(train) + len(val) + len(test) == total
    assert not (set(train) & set(val)) and not (set(val) & set(test))
    assert not set(train) & set(test)
    assert abs(len(train) / total - 0.75) < 0.01
    assert abs(len(val) / total - 0.15) < 0.01
    assert abs(len(test) / total - 0.10) < 0.01


def test_every_class_survives_every_split(targets):
    n_classes = np.unique(targets).size
    for idx in stratified_split(targets, (0.75, 0.15, 0.10), 42):
        assert np.unique(targets[idx]).size == n_classes


def test_split_is_deterministic_under_reseed(targets):
    a = stratified_split(targets, (0.75, 0.15, 0.10), 42)
    b = stratified_split(targets, (0.75, 0.15, 0.10), 42)
    c = stratified_split(targets, (0.75, 0.15, 0.10), 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_split_rejects_fractions_that_do_not_sum_to_one(targets):
    with pytest.raises(AssertionError):
        stratified_split(targets, (0.75, 0.15, 0.20), 42)


@pytest.mark.parametrize("n_tasks", [2, 3, 4, 8])
def test_task_partition_is_globally_balanced(targets, n_tasks):
    """Global spread must be <= 1.

    Resetting the round-robin per class would instead hand every class's
    remainder to task 0. On the real data that is a 172-image skew at N=2 --
    a milder form of the Continuum bias this function exists to avoid.
    """
    ids = balanced_task_ids(targets, n_tasks, 42)
    sizes = np.bincount(ids, minlength=n_tasks)
    assert sizes.max() - sizes.min() <= 1, sizes


@pytest.mark.parametrize("n_tasks", [2, 3, 4, 8])
def test_task_partition_is_balanced_within_each_class(targets, n_tasks):
    ids = balanced_task_ids(targets, n_tasks, 42)
    for cls in np.unique(targets):
        per = np.bincount(ids[targets == cls], minlength=n_tasks)
        assert per.max() - per.min() <= 1, (cls, per)


def test_task_partition_covers_everything_exactly_once(targets):
    ids = balanced_task_ids(targets, 4, 42)
    assert len(ids) == len(targets)
    assert set(np.unique(ids)) == {0, 1, 2, 3}


def test_task_partition_respects_the_seed(targets):
    """Continuum's own `random_seed` does not affect its partition; ours must."""
    a = balanced_task_ids(targets, 4, 42)
    b = balanced_task_ids(targets, 4, 42)
    c = balanced_task_ids(targets, 4, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_task_partition_refuses_more_tasks_than_samples():
    with pytest.raises(ValueError, match="tasks were requested"):
        balanced_task_ids(np.repeat(np.arange(5), 3), 8, 42)
