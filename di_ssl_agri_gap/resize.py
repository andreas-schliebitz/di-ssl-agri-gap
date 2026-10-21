"""Stage 4: re-encode every image to a short side of 256px.

The full-resolution tree is 254 GB across 560k files. Decoding that on every
epoch is the dominant cost in a multi-crop SSL loop, so it is paid once here.
The result is roughly 22 GB, which fits comfortably in /dev/shm.

256 is chosen so that the DINOv2 defaults work unmodified: global crops are
224 (16x14, a multiple of the ViT-S/14 patch size) and local crops 98 (7x14),
both obtainable from a 256-short-side source without upsampling artefacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile, ImageOps

from di_ssl_agri_gap.utils import load_settings

SHORT_SIDE = 256
JPEG_QUALITY = 95

# Pillow refuses very large images by default as a decompression-bomb guard;
# these are trusted and some legitimately exceed the limit.
Image.MAX_IMAGE_PIXELS = None


def _encode(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        # EXIF orientation must be applied before resizing, otherwise portrait
        # photos get cropped along the wrong axis downstream.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        width, height = img.size
        scale = SHORT_SIDE / min(width, height)
        # Always normalise to exactly 256 on the short side -- including the rare
        # upscale -- so CenterCrop(224) is valid for every image and the dataset
        # is dimensionally uniform.
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.resize(target, Image.Resampling.LANCZOS)

        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def resize_one(args: tuple[str, str, str, bool]) -> tuple[str, str]:
    """Resize one image.

    Returns `(rel_path, status)` where status is `""` on a clean decode,
    `"truncated"` when the source only decoded with truncation tolerance, and
    an error string otherwise.

    A small number of images (27 of 560,841, verified) carry truncated JPEG
    payloads. Extraction checked every file byte-for-byte against its archive's
    central directory, so the truncation is upstream -- they arrived incomplete
    from iNaturalist. They are decoded with the missing scanlines filled grey
    rather than dropped: removing rows would shift every index in the stratified
    split and invalidate the manifest checksum that downstream runs pin to.
    """
    src_s, dst_s, rel, force = args
    src, dst = Path(src_s), Path(dst_s)

    if not force and dst.exists() and dst.stat().st_size > 0:
        return (rel, "")

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        _encode(src, dst)
        return (rel, "")
    except OSError as exc:
        if "truncated" not in str(exc).lower():
            return (rel, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - collected, not raised
        return (rel, f"{type(exc).__name__}: {exc}")

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        _encode(src, dst)
        return (rel, "truncated")
    except Exception as exc:  # noqa: BLE001
        return (rel, f"{type(exc).__name__}: {exc}")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_settings()
    frame = pd.read_parquet(cfg.manifest_path, columns=["rel_path", "rel_path_256"])

    payload = [
        (
            str(cfg.raw_dir / row.rel_path),
            str(cfg.img256_dir / row.rel_path_256),
            row.rel_path_256,
            args.force,
        )
        for row in frame.itertuples(index=False)
    ]

    print(f"Source : {cfg.raw_dir}")
    print(f"Target : {cfg.img256_dir}")
    print(f"Images : {len(payload)}  |  workers: {args.workers}\n", flush=True)

    errors: list[tuple[str, str]] = []
    truncated: list[str] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (rel, status) in enumerate(
            pool.map(resize_one, payload, chunksize=64), 1
        ):
            if status == "truncated":
                truncated.append(rel)
            elif status:
                errors.append((rel, status))
            if n % 50_000 == 0 or n == len(payload):
                print(
                    f"  {n:7d}/{len(payload)}  |  {len(truncated)} truncated  |  "
                    f"{len(errors)} errors",
                    flush=True,
                )

    if truncated:
        record = cfg.data_root / "truncated_images.txt"
        record.write_text("\n".join(sorted(truncated)) + "\n")
        print(f"\n{len(truncated)} truncated source image(s) recovered -> {record}")

    if errors:
        print(f"\n{len(errors)} image(s) failed outright:")
        for rel, err in errors[:10]:
            print(f"  {rel}: {err}")
        return 1

    written = sum(1 for _ in cfg.img256_dir.rglob("*.jpg"))
    total = sum(p.stat().st_size for p in cfg.img256_dir.rglob("*.jpg"))
    print(f"\nWrote {written} images, {total / 1e9:.1f} GB")
    if written != len(payload):
        print(f"WARNING: expected {len(payload)} files on disk, found {written}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
