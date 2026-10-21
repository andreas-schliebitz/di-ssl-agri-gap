"""Checks against the real, built manifest.

These pin the numbers the whole study is calibrated on. If any of them drifts,
the source data changed and every downstream count needs revisiting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from di_ssl_agri_gap.enums import Granularity, Split
from di_ssl_agri_gap.manifest import add_month
from di_ssl_agri_gap.utils import load_settings

EXPECTED_ROWS = 560841  # 560,846 images less the 5 with no annotation record


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    cfg = load_settings()
    if not cfg.manifest_path.exists():
        pytest.skip("manifest not built yet; run di_ssl_agri_gap.manifest")
    return pd.read_parquet(cfg.manifest_path)


def test_row_count(manifest):
    assert len(manifest) == EXPECTED_ROWS


def test_class_counts_per_granularity(manifest):
    for granularity in Granularity:
        assert (
            manifest[granularity.label_column].nunique() == granularity.num_classes
        ), granularity


def test_label_ids_are_contiguous_from_zero(manifest):
    for granularity in Granularity:
        ids = manifest[granularity.label_column]
        assert ids.min() == 0
        assert ids.max() == granularity.num_classes - 1


def test_split_proportions(manifest):
    counts = manifest["split"].value_counts(normalize=True)
    assert abs(counts[Split.TRAIN.value] - 0.75) < 0.005
    assert abs(counts[Split.VAL.value] - 0.15) < 0.005
    assert abs(counts[Split.TEST.value] - 0.10) < 0.005


def test_every_class_appears_in_every_split(manifest):
    """Nested hierarchy: stratifying on species must cover all four levels."""
    for split in Split:
        subset = manifest[manifest["split"] == split.value]
        for granularity in Granularity:
            assert (
                subset[granularity.label_column].nunique() == granularity.num_classes
            ), (split, granularity)


def test_paths_are_unique(manifest):
    assert not manifest["rel_path"].duplicated().any()
    assert not manifest["rel_path_256"].duplicated().any()


def test_resized_paths_are_all_jpeg(manifest):
    """Everything is re-encoded to JPEG, whatever the source format was."""
    assert manifest["rel_path_256"].str.endswith(".jpg").all()


def test_hierarchy_is_consistent_within_the_manifest(manifest):
    """Each species maps to exactly one genus, family and crop/weed class."""
    grouped = manifest.groupby("species_id")[["genus_id", "family_id", "cropweed_id"]]
    assert (grouped.nunique() == 1).all().all()


def test_annotation_join_coverage(manifest):
    """Pinned: the photo_id join is near-total but not complete.

    annotations.json holds 641,951 records for 560,846 images, so a positional
    or count-based join would be silently wrong for 100 species.

    These are per-image counts. Counting annotation *records* instead gives
    slightly higher numbers, because the file contains duplicate photo_ids --
    another reason the join must be keyed rather than counted.
    """
    assert int((manifest["month"] > 0).sum()) == 558643
    assert int(manifest["latitude"].notna().sum()) == 555091
    # Latitude and longitude are always present or absent together.
    assert (manifest["latitude"].notna() == manifest["longitude"].notna()).all()


def test_seasonal_months_are_in_range(manifest):
    months = manifest["month"]
    assert months.isin(list(range(1, 13)) + [-1]).all()


def test_hemisphere_flip_is_an_involution():
    """Shifting by six months twice returns the raw calendar month."""
    raw_months = [1, 7, 3]
    frame = pd.DataFrame(
        {
            "observed_on": [f"2024-{m:02d}-15" for m in raw_months],
            "latitude": [-33.0, -33.0, 51.0],
        }
    )
    once = add_month(frame.copy())["month"]
    # Feed the shifted months back in at the same latitudes.
    twice = add_month(
        pd.DataFrame(
            {
                "observed_on": [f"2024-{m:02d}-15" for m in once],
                "latitude": frame["latitude"],
            }
        )
    )["month"]
    assert twice.tolist() == raw_months


def test_hemisphere_flip_only_moves_southern_rows():
    frame = pd.DataFrame(
        {
            "observed_on": ["2024-01-15", "2024-01-15", "2024-01-15"],
            "latitude": [-33.0, 51.0, np.nan],
        }
    )
    months = add_month(frame)["month"].tolist()
    assert months[0] == 7  # southern January -> July
    assert months[1] == 1  # northern unchanged
    assert months[2] == 1  # unknown latitude keeps the raw month
