"""Stage 3: build labels.parquet, the single source of truth for every run.

One row per image, carrying all four taxonomic targets, the frozen split
assignment, and the observation metadata that phase 2's seasonal ordering needs.

Two joins happen here and both have sharp edges:

  * species -> family comes only from AgML (see `taxonomy`), and
  * image -> observation metadata must be keyed on `photo_id`.

`annotations.json` is a *superset* of the extracted images: 641,951 records
against 560,846 files, with 100 species carrying more records than they have
images. Anything positional or count-based is silently wrong for those species.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from di_ssl_agri_gap.enums import Split
from di_ssl_agri_gap.prepare import ANNOTATION_NAME, IMAGE_SUFFIXES
from di_ssl_agri_gap.splits import RANDOM_SEED, stratified_split
from di_ssl_agri_gap.taxonomy import build_label_maps, load_taxonomy, verify_taxonomy
from di_ssl_agri_gap.utils import file_sha256, load_settings

EXPECTED_IMAGES = 560846
EXPECTED_MATCHED = 560841
SPLIT_FRACTIONS = (0.75, 0.15, 0.10)


def _scan_species(species_dir_s: str) -> list[dict]:
    """Collect one record per image in a species directory, joined to its annotation."""
    species_dir = Path(species_dir_s)
    species = species_dir.name

    annotations: dict[str, dict] = {}
    ann_path = species_dir / ANNOTATION_NAME
    if ann_path.exists():
        for record in json.loads(ann_path.read_text()):
            annotations[str(record.get("photo_id"))] = record

    rows = []
    for image_path in sorted(species_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        photo_id = image_path.stem
        # The parent directory encodes crop/weed independently of AgML; kept so
        # the two sources can be cross-checked rather than trusted blindly.
        cropweed_dir = image_path.parent.name.lower()
        record = annotations.get(photo_id, {})
        latitude, longitude = record.get("latitude"), record.get("longitude")

        rows.append(
            {
                "rel_path": f"{species}/{cropweed_dir}/{image_path.name}",
                # Everything is re-encoded to JPEG during resizing, so the 256px
                # tree has a uniform extension regardless of the source format
                # (359,563 .jpeg / 200,147 .jpg / 1,136 .png).
                "rel_path_256": f"{species}/{cropweed_dir}/{photo_id}.jpg",
                "photo_id": photo_id,
                "species": species,
                "cropweed_dir": cropweed_dir,
                "matched": bool(record),
                "observed_on": record.get("observed_on") or None,
                "latitude": float(latitude) if latitude is not None else np.nan,
                "longitude": float(longitude) if longitude is not None else np.nan,
                "observer": record.get("user") or None,
            }
        )
    return rows


def scan_images(raw_dir: Path, workers: int) -> pd.DataFrame:
    species_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not species_dirs:
        raise FileNotFoundError(f"no species directories under {raw_dir}")

    with ProcessPoolExecutor(workers) as pool:
        chunks = pool.map(_scan_species, [str(p) for p in species_dirs], chunksize=8)
        rows = [row for chunk in chunks for row in chunk]

    return pd.DataFrame(rows)


def add_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the hemisphere-corrected month-of-year used by the seasonal arm.

    Southern-hemisphere observations are shifted by six months so that N/S
    seasons align phenologically. Without this, January in Norway and January in
    New Zealand land in the same chunk despite being opposite points in the
    growing cycle, which cancels out most of the shift the arm exists to create.

    Rows with no usable date get month = -1 and are excluded from the seasonal
    stream only; they remain in every other part of the pipeline.
    """
    parsed = pd.to_datetime(frame["observed_on"], errors="coerce", format="mixed")
    month = parsed.dt.month

    southern = frame["latitude"].notna() & (frame["latitude"] < 0)
    shifted = month.where(~southern, ((month + 6 - 1) % 12) + 1)

    frame["month"] = shifted.fillna(-1).astype(int)
    frame["year"] = parsed.dt.year.fillna(-1).astype(int)
    return frame


def build_manifest(
    raw_dir: Path, workers: int, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    taxonomy = load_taxonomy()
    verify_taxonomy(taxonomy)
    label_maps = build_label_maps(taxonomy)

    frame = scan_images(raw_dir, workers)
    if len(frame) != EXPECTED_IMAGES:
        raise ValueError(
            f"found {len(frame)} images, expected {EXPECTED_IMAGES}; "
            "the extracted tree does not match the verified snapshot"
        )

    unknown = set(frame["species"]) - set(taxonomy)
    if unknown:
        raise ValueError(
            f"{len(unknown)} species missing from AgML: {sorted(unknown)[:5]}"
        )

    # Cross-check AgML's crop/weed against the on-disk directory naming. These
    # are independent sources; disagreement means one of them is wrong and the
    # binary probe would be built on sand.
    frame["cropweed"] = frame["species"].map(lambda s: taxonomy[s].cropweed)
    disagree = frame.loc[frame["cropweed"] != frame["cropweed_dir"], "species"].unique()
    if len(disagree):
        raise ValueError(
            f"AgML and directory naming disagree on crop/weed for "
            f"{len(disagree)} species: {sorted(disagree)[:5]}"
        )

    frame["genus"] = frame["species"].map(lambda s: taxonomy[s].genus)
    frame["family"] = frame["species"].map(lambda s: taxonomy[s].family)
    for level in ("species", "genus", "family", "cropweed"):
        frame[f"{level}_id"] = frame[level].map(label_maps[level]).astype("int32")

    frame = add_month(frame)

    # Drop images with no annotation record. Verified to be exactly 5 of
    # 560,846; a larger number means the photo_id join has broken.
    n_matched = int(frame["matched"].sum())
    if n_matched != EXPECTED_MATCHED:
        raise ValueError(
            f"annotation join matched {n_matched} images, expected "
            f"{EXPECTED_MATCHED}; the photo_id join is wrong"
        )
    dropped = frame.loc[~frame["matched"], "rel_path"].tolist()
    frame = frame.loc[frame["matched"]].reset_index(drop=True)
    print(f"Dropped {len(dropped)} unmatched image(s): {dropped}")

    train_idx, val_idx, test_idx = stratified_split(
        frame["species_id"].to_numpy(), SPLIT_FRACTIONS, seed
    )
    frame["split"] = pd.NA
    frame.loc[train_idx, "split"] = Split.TRAIN.value
    frame.loc[val_idx, "split"] = Split.VAL.value
    frame.loc[test_idx, "split"] = Split.TEST.value
    assert frame["split"].notna().all(), "some row was left unassigned by the split"

    # photo_id is unique per species but not globally, so the 256px path (which
    # embeds species and crop/weed) must still be unique across the whole tree.
    if frame["rel_path_256"].duplicated().any():
        dupes = frame.loc[frame["rel_path_256"].duplicated(), "rel_path_256"].head()
        raise ValueError(f"re-encoding would collide on: {dupes.tolist()}")

    columns = [
        "rel_path",
        "rel_path_256",
        "photo_id",
        "split",
        "species_id",
        "genus_id",
        "family_id",
        "cropweed_id",
        "species",
        "genus",
        "family",
        "cropweed",
        "observed_on",
        "month",
        "year",
        "latitude",
        "longitude",
        "observer",
    ]
    return frame[columns].sort_values("rel_path").reset_index(drop=True)


def summarise(frame: pd.DataFrame) -> None:
    print(f"\nRows: {len(frame)}")
    print(
        "Classes  :",
        {
            lvl: int(frame[f"{lvl}_id"].nunique())
            for lvl in ("species", "genus", "family", "cropweed")
        },
    )
    counts = frame["split"].value_counts()
    print("Split    :", {k: int(v) for k, v in counts.items()})
    print("Fractions:", {k: round(v / len(frame), 4) for k, v in counts.items()})

    for split in (Split.TRAIN, Split.VAL, Split.TEST):
        sub = frame[frame["split"] == split.value]
        cover = {
            lvl: int(sub[f"{lvl}_id"].nunique())
            for lvl in ("species", "genus", "family", "cropweed")
        }
        print(f"  {split.value:5s} coverage: {cover}")

    usable = int((frame["month"] > 0).sum())
    print(f"\nSeasonal-usable rows: {usable} ({100 * usable / len(frame):.2f}%)")
    print(
        "Month histogram:",
        frame.loc[frame["month"] > 0, "month"].value_counts().sort_index().to_dict(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    cfg = load_settings()
    frame = build_manifest(cfg.raw_dir, args.workers, args.seed)
    summarise(frame)

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.manifest_path, index=False)

    taxonomy = load_taxonomy()
    cfg.label_maps_path.write_text(json.dumps(build_label_maps(taxonomy), indent=2))

    print(f"\nManifest  : {cfg.manifest_path}")
    print(f"SHA-256   : {file_sha256(cfg.manifest_path)}")
    print(f"Label maps: {cfg.label_maps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
