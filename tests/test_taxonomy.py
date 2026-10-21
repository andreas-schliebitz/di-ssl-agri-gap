"""Taxonomy joins and the edge cases that silently corrupt the family probe."""

from __future__ import annotations

import pytest

from di_ssl_agri_gap.enums import Granularity
from di_ssl_agri_gap.taxonomy import (
    FAMILY_REMAP,
    Taxon,
    build_label_maps,
    derive_genus,
    load_taxonomy,
    verify_taxonomy,
)


@pytest.fixture(scope="module")
def taxonomy() -> dict[str, Taxon]:
    return load_taxonomy()


def test_counts_match_the_verified_dataset(taxonomy):
    assert verify_taxonomy(taxonomy) == {
        "species": 2958,
        "genus": 1171,
        "family": 215,
        "crop_species": 1986,
        "weed_species": 972,
    }


def test_granularity_class_counts_agree_with_taxonomy(taxonomy):
    """The enum hardcodes class counts; they must track the real taxonomy."""
    assert Granularity.SPECIES.num_classes == len(taxonomy)
    assert Granularity.GENUS.num_classes == len({t.genus for t in taxonomy.values()})
    assert Granularity.FAMILY.num_classes == len({t.family for t in taxonomy.values()})
    assert Granularity.CROPWEED.num_classes == 2


def test_no_species_is_left_in_the_unknown_family(taxonomy):
    """An 'Unknown' family would become a junk 216th class in the family probe."""
    assert not [s for s, t in taxonomy.items() if t.family.lower() == "unknown"]


def test_abrus_pecatorius_is_remapped_to_fabaceae(taxonomy):
    """The sole Unknown, a misspelling of *Abrus precatorius*.

    Its genus-mate carries Fabaceae, which is also the correct family for the
    real taxon.
    """
    assert "abrus_pecatorius" in FAMILY_REMAP
    assert taxonomy["abrus_pecatorius"].family == "Fabaceae"
    assert taxonomy["abrus_precatorius"].family == "Fabaceae"


@pytest.mark.parametrize(
    ("species", "genus"),
    [
        # Four tokens and a U+00D7 multiplication sign.
        ("musa_acuminata_×_balbisiana", "musa"),
        # Three tokens, trinomial.
        ("portulaca_pilosa_pilosa", "portulaca"),
        ("abelmoschus_esculentus", "abelmoschus"),
    ],
)
def test_genus_derivation_handles_awkward_species_names(species, genus):
    assert derive_genus(species) == genus


def test_awkward_species_are_actually_present(taxonomy):
    """Guard the parametrised cases above against silently vanishing."""
    assert "musa_acuminata_×_balbisiana" in taxonomy
    assert "portulaca_pilosa_pilosa" in taxonomy


def test_label_maps_are_contiguous_and_deterministic(taxonomy):
    maps = build_label_maps(taxonomy)
    for level, mapping in maps.items():
        assert sorted(mapping.values()) == list(range(len(mapping))), level
    # Fixed polarity so the binary probe's positive class never flips.
    assert maps["cropweed"] == {"crop": 0, "weed": 1}
    assert build_label_maps(taxonomy) == maps


def test_hierarchy_is_nested(taxonomy):
    """Each genus sits in exactly one family, so the tree is well defined.

    The hierarchical error-distance metric assumes this: a genus spanning two
    families would make 'same family, different genus' ambiguous.
    """
    genus_to_families: dict[str, set[str]] = {}
    for taxon in taxonomy.values():
        genus_to_families.setdefault(taxon.genus, set()).add(taxon.family)
    ambiguous = {g: f for g, f in genus_to_families.items() if len(f) > 1}
    assert not ambiguous, f"genera spanning multiple families: {ambiguous}"
