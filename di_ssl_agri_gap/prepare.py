"""Stage 1-2: extract the iNatAg-mini archives, verify, then optionally delete.

The source dataset is a HuggingFace snapshot: 2,958 per-species zip archives
totalling 237 GB compressed / 254.1 GB uncompressed, laid out as

    <species>.zip
      └── <species>/
            ├── annotations.json
            └── <crop|weed>/<photo_id>.{jpg,jpeg,png}

Extraction and deletion are deliberately separate phases. Deleting an archive
before its contents have been verified would cost a full re-download of the
snapshot, so `--delete-zips` refuses to run unless every archive verified clean.

This module imports only the standard library so that it can run before (or
alongside) the installation of the training stack.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
ANNOTATION_NAME = "annotations.json"

# Verified against the full snapshot on 2026-08-07. A mismatch here means the
# source data changed and every downstream count in the paper needs revisiting.
EXPECTED_ARCHIVES = 2958
EXPECTED_IMAGES = 560846


@dataclass(frozen=True)
class ArchiveReport:
    """Outcome of extracting and verifying a single species archive."""

    species: str
    n_images: int
    n_extracted: int
    n_size_mismatch: int
    n_missing: int
    uncompressed_bytes: int
    has_annotations: bool
    skipped: bool
    error: str

    @property
    def clean(self) -> bool:
        return (
            not self.error
            and self.has_annotations
            and self.n_missing == 0
            and self.n_size_mismatch == 0
            and self.n_extracted == self.n_images
        )


def _safe_members(zf: zipfile.ZipFile, species: str) -> list[zipfile.ZipInfo]:
    """Return the file members of `zf`, rejecting anything outside `species/`.

    Guards against zip-slip: a member named `../../etc/passwd` would otherwise
    escape the extraction root.
    """
    members = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = Path(info.filename).parts
        if not parts or parts[0] != species or ".." in parts:
            raise ValueError(f"unsafe member path in {species}.zip: {info.filename!r}")
        members.append(info)
    return members


def extract_archive(args: tuple[str, str, bool]) -> ArchiveReport:
    """Extract one archive and verify every member against its central directory.

    Verification compares the on-disk size of each extracted file with the
    `file_size` recorded in the archive, which catches truncated writes -- the
    realistic failure mode when writing 560k small files to NFS.
    """
    zip_path_s, out_root_s, force = args
    zip_path, out_root = Path(zip_path_s), Path(out_root_s)
    species = zip_path.stem

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = _safe_members(zf, species)
            images = [
                m for m in members if Path(m.filename).suffix.lower() in IMAGE_SUFFIXES
            ]
            has_ann = any(Path(m.filename).name == ANNOTATION_NAME for m in members)
            uncompressed = sum(m.file_size for m in members)

            # Resume support: if every member is already on disk at the right
            # size, skip the extraction entirely. Makes re-runs nearly free.
            if not force and _already_complete(members, out_root):
                return ArchiveReport(
                    species=species,
                    n_images=len(images),
                    n_extracted=len(images),
                    n_size_mismatch=0,
                    n_missing=0,
                    uncompressed_bytes=uncompressed,
                    has_annotations=has_ann,
                    skipped=True,
                    error="",
                )

            zf.extractall(out_root, members=members)

        missing, mismatch, extracted = 0, 0, 0
        for m in members:
            target = out_root / m.filename
            if not target.exists():
                missing += 1
                continue
            if target.stat().st_size != m.file_size:
                mismatch += 1
                continue
            if Path(m.filename).suffix.lower() in IMAGE_SUFFIXES:
                extracted += 1

        return ArchiveReport(
            species=species,
            n_images=len(images),
            n_extracted=extracted,
            n_size_mismatch=mismatch,
            n_missing=missing,
            uncompressed_bytes=uncompressed,
            has_annotations=has_ann,
            skipped=False,
            error="",
        )
    except Exception as exc:  # noqa: BLE001 - recorded per archive, not raised
        return ArchiveReport(
            species=species,
            n_images=0,
            n_extracted=0,
            n_size_mismatch=0,
            n_missing=0,
            uncompressed_bytes=0,
            has_annotations=False,
            skipped=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _already_complete(members: list[zipfile.ZipInfo], out_root: Path) -> bool:
    for m in members:
        target = out_root / m.filename
        try:
            if target.stat().st_size != m.file_size:
                return False
        except OSError:
            return False
    return True


def write_report(reports: list[ArchiveReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(ArchiveReport)] + ["clean"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in sorted(reports, key=lambda x: x.species):
            writer.writerow({**asdict(r), "clean": r.clean})


def summarise(reports: list[ArchiveReport]) -> dict[str, int]:
    return {
        "archives": len(reports),
        "clean": sum(r.clean for r in reports),
        "failed": sum(not r.clean for r in reports),
        "skipped": sum(r.skipped for r in reports),
        "images": sum(r.n_extracted for r in reports),
        "missing": sum(r.n_missing for r in reports),
        "size_mismatch": sum(r.n_size_mismatch for r in reports),
        "no_annotations": sum(not r.has_annotations for r in reports),
        "uncompressed_bytes": sum(r.uncompressed_bytes for r in reports),
    }


def delete_archives(zip_dir: Path, reports: list[ArchiveReport]) -> int:
    """Remove archives, but only when every single one verified clean.

    All-or-nothing on purpose: a partial delete would leave the snapshot in a
    state that is neither re-extractable nor re-downloadable incrementally.
    """
    unclean = [r.species for r in reports if not r.clean]
    if unclean:
        raise RuntimeError(
            f"refusing to delete: {len(unclean)} archive(s) did not verify, "
            f"e.g. {unclean[:5]}"
        )
    if len(reports) != EXPECTED_ARCHIVES:
        raise RuntimeError(
            f"refusing to delete: report covers {len(reports)} archives, "
            f"expected {EXPECTED_ARCHIVES}"
        )

    removed = 0
    for r in reports:
        target = zip_dir / f"{r.species}.zip"
        if target.exists():
            target.unlink()
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "INATAG_ZIP_DIR",
                "/cluster/user/aschlieb/asyncio/datasets/iNatAg-mini",
            )
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "DATA_ROOT",
                "/cluster/user/aschlieb/asyncio/datasets/iNatAg-mini-work",
            )
        ),
    )
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    parser.add_argument(
        "--force", action="store_true", help="re-extract even if files look complete"
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="delete source archives, only if every archive verified clean",
    )
    args = parser.parse_args(argv)

    raw_root = args.data_root / "raw"
    report_path = args.data_root / "extraction_report.csv"
    raw_root.mkdir(parents=True, exist_ok=True)

    zips = sorted(p for p in args.zip_dir.iterdir() if p.suffix == ".zip")
    if not zips:
        print(f"No archives found in {args.zip_dir}", file=sys.stderr)
        return 1

    free = shutil.disk_usage(args.data_root).free
    print(f"Archives      : {len(zips)} (expected {EXPECTED_ARCHIVES})")
    print(f"Destination   : {raw_root}")
    print(f"Free space    : {free / 1e12:.1f} TB")
    print(f"Workers       : {args.workers}")
    print(flush=True)

    payload = [(str(p), str(raw_root), args.force) for p in zips]
    reports: list[ArchiveReport] = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(extract_archive, item) for item in payload]
        for n, fut in enumerate(as_completed(futures), 1):
            reports.append(fut.result())
            if n % 100 == 0 or n == len(futures):
                done = sum(r.n_extracted for r in reports)
                bad = sum(not r.clean for r in reports)
                print(
                    f"  {n:5d}/{len(futures)} archives | {done:7d} images | "
                    f"{bad} failed",
                    flush=True,
                )

    write_report(reports, report_path)
    stats = summarise(reports)

    print(f"\nReport written to {report_path}")
    for k, v in stats.items():
        print(f"  {k:20s} {v}")

    if stats["images"] != EXPECTED_IMAGES:
        print(
            f"\nWARNING: extracted {stats['images']} images, "
            f"expected {EXPECTED_IMAGES}",
            file=sys.stderr,
        )

    if stats["failed"]:
        print(
            f"\n{stats['failed']} archive(s) failed verification. "
            "Inspect the report; not deleting anything.",
            file=sys.stderr,
        )
        return 1

    if args.delete_zips:
        removed = delete_archives(args.zip_dir, reports)
        print(f"\nDeleted {removed} source archives.")
    else:
        print(
            "\nAll archives verified clean. Re-run with --delete-zips to remove them."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
