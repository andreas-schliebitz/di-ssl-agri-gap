"""Result-completeness audit and MLflow resync.

The experiment matrix runs for over a week, so the failure mode that matters is
noticing at the *end* that something never got recorded. This enumerates the
encoders the design expects, reports what is missing, consolidates everything
into one CSV, and can re-push results that the tracking server rejected while it
was unreachable.

Local disk is the system of record. MLflow is a convenience mirror.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from di_ssl_agri_gap.enums import Granularity, Ordering, Split
from di_ssl_agri_gap.utils import load_settings

N_CHUNKS = (2, 4, 8)
SEED = 42


def expected_encoders(model_tag: str = "vits14scratch") -> list[str]:
    """Every encoder the design calls for, in report order."""
    ids = ["offtheshelf_vits14", f"joint_{model_tag}_ep100_seed{SEED}"]
    for ordering in Ordering:
        for n in N_CHUNKS:
            ids += [
                f"{ordering}_n{n}_chunk{i}_{model_tag}_seed{SEED}" for i in range(n)
            ]
    return ids


def audit(model_tag: str = "vits14scratch") -> pd.DataFrame:
    cfg = load_settings()
    results = cfg.data_root / "results"
    rows = []
    for encoder_id in expected_encoders(model_tag):
        probe = results / f"probe_{encoder_id}.parquet"
        preds = cfg.data_root / "predictions" / encoder_id
        feats = cfg.data_root / "features" / encoder_id
        trained = (
            encoder_id == "offtheshelf_vits14"
            or (
                cfg.out_dir / encoder_id / "exported_models" / "exported_last.pt"
            ).exists()
        )
        n_gran = 0
        if probe.exists():
            n_gran = int(
                pd.read_parquet(probe)
                .query(f"split == '{Split.TEST.value}'")["granularity"]
                .nunique()
            )
        rows.append(
            {
                "encoder_id": encoder_id,
                "trained": trained,
                "features": all((feats / f"{s.value}.npy").exists() for s in Split),
                "probed": probe.exists(),
                "granularities": n_gran,
                "complete": probe.exists() and n_gran == len(Granularity),
                "predictions": (
                    len(list(preds.glob("*.parquet"))) if preds.exists() else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def consolidate(model_tag: str = "vits14scratch") -> Path:
    """Merge every per-encoder probe result into one CSV."""
    cfg = load_settings()
    results = cfg.data_root / "results"
    frames = [pd.read_parquet(p) for p in sorted(results.glob("probe_*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no probe_*.parquet under {results}")
    frame = pd.concat(frames, ignore_index=True)
    target = results / "all_probe_results.csv"
    frame.to_csv(target, index=False)
    return target


def resync_mlflow() -> int:
    """Re-push results for encoders whose MLflow logging previously failed."""
    from di_ssl_agri_gap.probe import log_to_mlflow

    cfg = load_settings()
    pending = cfg.data_root / "results" / "mlflow_pending.txt"
    if not pending.exists():
        print("nothing pending")
        return 0
    ids = sorted(
        {
            line.split("\t")[0]
            for line in pending.read_text().splitlines()
            if line.strip()
        }
    )
    print(f"{len(ids)} encoder(s) queued for resync")
    still_failing = []
    for encoder_id in ids:
        probe = cfg.data_root / "results" / f"probe_{encoder_id}.parquet"
        if not probe.exists():
            print(f"  {encoder_id}: no local result to push, skipping")
            continue
        before = pending.read_text()
        log_to_mlflow(encoder_id, pd.read_parquet(probe), {})
        if pending.read_text() != before:
            still_failing.append(encoder_id)
        else:
            print(f"  {encoder_id}: pushed")
    # Rewrite the queue with only what is still broken.
    pending.write_text("".join(f"{i}\tretry\n" for i in still_failing))
    return 1 if still_failing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", default="vits14scratch")
    parser.add_argument(
        "--resync", action="store_true", help="re-push failed MLflow logs"
    )
    args = parser.parse_args(argv)

    if args.resync:
        return resync_mlflow()

    frame = audit(args.model_tag)
    print(frame.to_string(index=False))

    n_done = int(frame["complete"].sum())
    print(f"\n{n_done}/{len(frame)} encoders fully probed")

    missing = frame.loc[~frame["complete"] & frame["trained"], "encoder_id"].tolist()
    if missing:
        print(f"\nTRAINED BUT NOT PROBED ({len(missing)}): {missing}")
    untrained = frame.loc[~frame["trained"], "encoder_id"].tolist()
    if untrained:
        print(
            f"\nnot yet trained ({len(untrained)}): {untrained[:6]}"
            f"{' ...' if len(untrained) > 6 else ''}"
        )

    cfg = load_settings()
    pending = cfg.data_root / "results" / "mlflow_pending.txt"
    if pending.exists() and pending.read_text().strip():
        n = len(pending.read_text().strip().splitlines())
        print(f"\n{n} MLflow push(es) pending -- run with --resync")

    if any(frame["probed"]):
        print(f"\nConsolidated -> {consolidate(args.model_tag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
