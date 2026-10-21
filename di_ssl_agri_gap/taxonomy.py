"""The four-level taxonomy for iNatAg-mini.

Family labels exist nowhere in the dataset directory: they are carried only by
AgML's bundled metadata asset, keyed by species. Species comes from the
directory name, genus is derived from it, and crop/weed is redundantly encoded
both in AgML and in the on-disk subdirectory (we cross-check the two).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ASSET_NAME = "iNatAg-mini_public_datasources.json"

# `abrus_pecatorius` is the only species AgML files under family "Unknown". It
# is a misspelling of *Abrus precatorius*, and the genus's other species carries
# Fabaceae -- which is also correct for the real taxon. Remapping keeps the
# family probe a clean 215-way problem instead of 215 real classes plus a
# one-species junk class.
UNKNOWN_FAMILY = "Unknown"
FAMILY_REMAP: dict[str, str] = {"abrus_pecatorius": "Fabaceae"}

# Verified against the AgML asset on 2026-08-07.
EXPECTED_SPECIES = 2958
EXPECTED_GENERA = 1171
EXPECTED_FAMILIES = 215
EXPECTED_CROP_SPECIES = 1986
EXPECTED_WEED_SPECIES = 972


@dataclass(frozen=True)
class Taxon:
    species: str
    genus: str
    family: str
    cropweed: str


@cache
def agml_asset_path() -> Path:
    """Locate AgML's bundled metadata asset inside the installed package."""
    import agml

    path = Path(agml.__file__).parent / "_assets" / ASSET_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"AgML metadata asset not found at {path}. Family labels are only "
            "available from AgML; install agml==0.8.0."
        )
    return path


def derive_genus(species: str) -> str:
    """Genus is the first underscore-delimited token of the species name.

    Correct for both awkward cases in this dataset:
    `musa_acuminata_×_balbisiana` (four tokens, non-ASCII) -> `musa`, and
    `portulaca_pilosa_pilosa` (three tokens) -> `portulaca`.
    """
    return species.split("_")[0]


def load_taxonomy(asset_path: Path | None = None) -> dict[str, Taxon]:
    """Build the species -> Taxon map from AgML's metadata.

    Raises:
        ValueError: if any species still resolves to an unknown family after
            the remap, since that would corrupt the family probe silently.
    """
    path = asset_path or agml_asset_path()
    raw = json.loads(path.read_text())

    taxonomy: dict[str, Taxon] = {}
    for key, entry in raw.items():
        species = key.split("/", 1)[1]
        family = FAMILY_REMAP.get(species, entry["extra_metadata"]["family_name"])
        if family.lower() == UNKNOWN_FAMILY.lower():
            raise ValueError(
                f"species {species!r} has family 'Unknown' and no remap entry; "
                "add one to FAMILY_REMAP rather than letting it become a class"
            )
        # AgML stores crop/weed as a single-entry class dict, e.g. {"0": "crop"}.
        cropweed = next(iter(entry["classes"].values())).lower()
        if cropweed not in ("crop", "weed"):
            raise ValueError(f"species {species!r} has unexpected class {cropweed!r}")

        taxonomy[species] = Taxon(
            species=species,
            genus=derive_genus(species),
            family=family,
            cropweed=cropweed,
        )

    return taxonomy


def build_label_maps(
    taxonomy: dict[str, Taxon],
) -> dict[str, dict[str, int]]:
    """Assign contiguous integer ids per level, sorted for determinism."""
    return {
        "species": {n: i for i, n in enumerate(sorted(taxonomy))},
        "genus": {
            n: i for i, n in enumerate(sorted({t.genus for t in taxonomy.values()}))
        },
        "family": {
            n: i for i, n in enumerate(sorted({t.family for t in taxonomy.values()}))
        },
        # Fixed order: crop=0, weed=1, so the binary probe's positive class is
        # stable across runs and matches the directory naming.
        "cropweed": {"crop": 0, "weed": 1},
    }


def verify_taxonomy(taxonomy: dict[str, Taxon]) -> dict[str, int]:
    """Check the taxonomy against the counts the experiment design assumes."""
    counts = {
        "species": len(taxonomy),
        "genus": len({t.genus for t in taxonomy.values()}),
        "family": len({t.family for t in taxonomy.values()}),
        "crop_species": sum(t.cropweed == "crop" for t in taxonomy.values()),
        "weed_species": sum(t.cropweed == "weed" for t in taxonomy.values()),
    }
    expected = {
        "species": EXPECTED_SPECIES,
        "genus": EXPECTED_GENERA,
        "family": EXPECTED_FAMILIES,
        "crop_species": EXPECTED_CROP_SPECIES,
        "weed_species": EXPECTED_WEED_SPECIES,
    }
    mismatches = {k: (counts[k], v) for k, v in expected.items() if counts[k] != v}
    if mismatches:
        raise ValueError(
            "taxonomy does not match the verified iNatAg-mini counts "
            f"(got, expected): {mismatches}"
        )
    return counts
