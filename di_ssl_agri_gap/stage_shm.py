"""Stage 5: copy the 256px tree into /dev/shm and build the train-only view.

Reading 560k small files off NFS starves the DataLoader; /dev/shm has 504 GB
free and the resized tree is ~22 GB, so the whole dataset lives in RAM for the
duration of a session.

`img256_train/` contains hardlinks to exactly the train-split rows and nothing
else. This is the leakage guard: the SSL encoder is pointed at that directory,
so val and test images are unreachable during pre-training even as unlabelled
pixels. Hardlinks cost inodes, not bytes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from di_ssl_agri_gap.enums import Split
from di_ssl_agri_gap.utils import load_settings


def rsync_tree(src: Path, dst: Path) -> None:
    # No --info=progress2: it emits a carriage-return status line per file, which
    # for 560k files is tens of MB of output when stdout is not a terminal.
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--stats", f"{src}/", f"{dst}/"], check=True)


def build_train_view(manifest: pd.DataFrame, image_root: Path, train_dir: Path) -> int:
    """Hardlink the train-split rows into `train_dir`, rebuilt from scratch."""
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)

    rows = manifest.loc[manifest["split"] == Split.TRAIN.value, "rel_path_256"]
    linked = 0
    for rel in rows:
        src, dst = image_root / rel, train_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        linked += 1
    return linked


def verify_train_view(manifest: pd.DataFrame, train_dir: Path) -> None:
    """Assert the train view is exactly the train split -- no more, no less."""
    expected = set(manifest.loc[manifest["split"] == Split.TRAIN.value, "rel_path_256"])
    actual = {str(p.relative_to(train_dir)) for p in train_dir.rglob("*.jpg")}
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise AssertionError(
            f"train view mismatch: {len(missing)} missing, {len(extra)} extra "
            f"(e.g. missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]})"
        )

    leaked = (
        set(manifest.loc[manifest["split"] != Split.TRAIN.value, "rel_path_256"])
        & actual
    )
    if leaked:
        raise AssertionError(
            f"LEAKAGE: {len(leaked)} val/test images present in the train view"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-rsync",
        action="store_true",
        help="rebuild the train view only, assuming /dev/shm is already populated",
    )
    args = parser.parse_args(argv)

    cfg = load_settings()
    manifest = pd.read_parquet(cfg.manifest_path, columns=["rel_path_256", "split"])

    staged_images = cfg.shm_root / "img256"
    staged_train = cfg.shm_root / "img256_train"

    if not args.skip_rsync:
        free = shutil.disk_usage("/dev/shm").free
        need = sum(p.stat().st_size for p in cfg.img256_dir.rglob("*.jpg"))
        print(f"/dev/shm free: {free / 1e9:.0f} GB  |  need: {need / 1e9:.1f} GB")
        if need > free * 0.9:
            raise RuntimeError("not enough space in /dev/shm for the resized tree")
        print(f"Staging {cfg.img256_dir} -> {staged_images}", flush=True)
        rsync_tree(cfg.img256_dir, staged_images)

    print(f"\nBuilding train view at {staged_train}", flush=True)
    linked = build_train_view(manifest, staged_images, staged_train)
    verify_train_view(manifest, staged_train)

    n_train = int((manifest["split"] == Split.TRAIN.value).sum())
    print(f"Hardlinked {linked} train images (manifest says {n_train}) -- verified.")
    print(f"\nPre-train against: {staged_train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
