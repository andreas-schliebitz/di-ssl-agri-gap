"""Enumerations shared across both experimental phases."""

from __future__ import annotations

from enum import StrEnum


class Granularity(StrEnum):
    """The four nested probe targets.

    The hierarchy is strictly nested (species -> genus -> family), with
    crop/weed a separate binary partition over species. Stratifying a split at
    SPECIES therefore guarantees every genus and family appears in every split,
    which is why one split serves all four probes.
    """

    SPECIES = "species"
    GENUS = "genus"
    FAMILY = "family"
    CROPWEED = "cropweed"

    @property
    def label_column(self) -> str:
        """Column in labels.parquet holding this granularity's integer target."""
        return f"{self.value}_id"

    @property
    def num_classes(self) -> int:
        """Class count, verified against the AgML metadata for iNatAg-mini."""
        return {
            Granularity.SPECIES: 2958,
            Granularity.GENUS: 1171,
            Granularity.FAMILY: 215,
            Granularity.CROPWEED: 2,
        }[self]


class Arm(StrEnum):
    """Experimental arm.

    OFFTHESHELF is the un-adapted Meta DINOv2 encoder, probed without any
    pre-training on iNatAg-mini. It supplies the headroom denominator: the gap
    between JOINT and INCREMENTAL is only interesting relative to how much
    adaptation buys in the first place.
    """

    OFFTHESHELF = "offtheshelf"
    JOINT = "joint"
    INCREMENTAL = "incremental"


class Ordering(StrEnum):
    """How the data-incremental stream is ordered (phase 2).

    IID is Continuum's InstanceIncremental semantics: same classes throughout,
    only the instances differ. SEASONAL orders by hemisphere-corrected
    month-of-year, which induces real phenological distribution shift.

    Absolute-date ordering was measured and rejected: 91.5% of iNatAg-mini falls
    in 2020-2025 and 50.9% in 2024 alone, so equal-count date blocks collapse
    into a single year after the first boundary.
    """

    IID = "iid"
    SEASONAL = "seasonal"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
