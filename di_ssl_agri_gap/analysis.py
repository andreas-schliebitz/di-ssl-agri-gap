"""Stage 11: gap tables and the three mechanism metrics.

The gap table is the headline, but on its own it only says *that* incremental
pre-training costs something. These three say *what happens to the
representation*, which is the actual claim:

  1. Hierarchical error distance -- do species errors move further up the tree?
  2. Error containment -- do they stop staying inside the right genus/family?
  3. CKA drift -- does the representation move, and does that track the gap?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from di_ssl_agri_gap.enums import Granularity, Split
from di_ssl_agri_gap.utils import load_settings

# --------------------------------------------------------------------------- #
# 1 & 2: hierarchical error structure
# --------------------------------------------------------------------------- #


def hierarchical_error_metrics(
    predictions: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, float]:
    """Measure how far species-level mistakes travel up the taxonomy.

    Distance is 0 for a correct species, 1 for the right genus, 2 for the right
    family but wrong genus, and 3 for a different family.

    The iNatAg paper's own error analysis notes that species mistakes largely
    stay within the correct genus or family. If representation drift is the
    mechanism behind the gap, incremental pre-training should erode exactly
    that containment -- a falling `contained_in_genus` is the cleanest single
    piece of evidence for it, and it is not implied by the accuracy drop alone.
    """
    species_to_genus = manifest.drop_duplicates("species_id").set_index("species_id")[
        "genus_id"
    ]
    species_to_family = manifest.drop_duplicates("species_id").set_index("species_id")[
        "family_id"
    ]

    true_species = predictions["y_true"].to_numpy()
    pred_species = predictions["y_pred"].to_numpy()

    true_genus = species_to_genus.reindex(true_species).to_numpy()
    pred_genus = species_to_genus.reindex(pred_species).to_numpy()
    true_family = species_to_family.reindex(true_species).to_numpy()
    pred_family = species_to_family.reindex(pred_species).to_numpy()

    correct = true_species == pred_species
    same_genus = true_genus == pred_genus
    same_family = true_family == pred_family

    distance = np.where(
        correct, 0, np.where(same_genus, 1, np.where(same_family, 2, 3))
    )

    errors = ~correct
    n_err = int(errors.sum())
    return {
        "species_accuracy": float(correct.mean()),
        "mean_hierarchical_distance": float(distance.mean()),
        # Conditioned on being wrong -- this is the containment rate that drift
        # should degrade.
        "contained_in_genus": (
            float(same_genus[errors].mean()) if n_err else float("nan")
        ),
        "contained_in_family": (
            float(same_family[errors].mean()) if n_err else float("nan")
        ),
        "escaped_family": (
            float((~same_family)[errors].mean()) if n_err else float("nan")
        ),
        "n_errors": n_err,
    }


# --------------------------------------------------------------------------- #
# 3: representation drift
# --------------------------------------------------------------------------- #


def linear_cka(x: np.ndarray, y: np.ndarray, device: str = "cuda") -> float:
    """Linear CKA between two feature matrices over the same, aligned samples.

    Uses the Gram-free form: ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), which is
    O(n*d^2) rather than O(n^2*d) and therefore tractable on 56k x 768 without
    materialising a 56k x 56k Gram matrix.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.float32)
    b = torch.from_numpy(np.ascontiguousarray(y)).to(dev, torch.float32)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)

    cross = torch.linalg.matrix_norm(b.T @ a) ** 2
    norm_a = torch.linalg.matrix_norm(a.T @ a)
    norm_b = torch.linalg.matrix_norm(b.T @ b)
    return float((cross / (norm_a * norm_b)).item())


def drift_against(
    reference_id: str, encoder_id: str, split: Split = Split.TEST
) -> float:
    cfg = load_settings()
    root = cfg.data_root / "features"
    ref = np.load(root / reference_id / f"{split.value}.npy")
    cur = np.load(root / encoder_id / f"{split.value}.npy")
    if ref.shape != cur.shape:
        raise ValueError(f"feature shape mismatch: {ref.shape} vs {cur.shape}")
    return linear_cka(ref, cur)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def collect_probe_results(results_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(results_dir.glob("probe_*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no probe_*.parquet under {results_dir}")
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["split"] == Split.TEST.value].reset_index(drop=True)


# The canonical joint baseline. Ablation encoders (the abandoned adaptation LR
# sweep and augmentation tests) also begin with "joint", so matching on the
# prefix silently pulled 8 encoders into the baseline -- and because
# `pivot_table` aggregates with mean, the gap was measured against the *average*
# of 8 unrelated encoders including one that scored 10.78% at species level.
# The reference must be exactly one encoder.
JOINT_REFERENCE = "joint_vits14scratch_ep100_seed42"


def parse_encoder_id(encoder_id: str) -> dict[str, object]:
    """Recover arm/ordering/N/chunk from the encoder id used across all stages."""
    if encoder_id == JOINT_REFERENCE:
        return {"arm": "joint", "ordering": "none", "n_chunks": 1, "chunk_idx": 0}
    if encoder_id.startswith("joint"):
        # A pre-training ablation, not a comparison arm.
        return {"arm": "ablation", "ordering": "none", "n_chunks": 1, "chunk_idx": 0}
    if encoder_id.startswith("offtheshelf"):
        return {
            "arm": "offtheshelf",
            "ordering": "none",
            "n_chunks": 0,
            "chunk_idx": -1,
        }
    parts = encoder_id.split("_")
    return {
        "arm": "incremental",
        "ordering": parts[0],
        "n_chunks": int(parts[1].lstrip("n")),
        "chunk_idx": int(parts[2].replace("chunk", "")),
    }


def build_gap_table(
    test_results: pd.DataFrame, reference: str = JOINT_REFERENCE
) -> pd.DataFrame:
    """Join every encoder against the joint baseline at the same granularity."""
    meta = pd.DataFrame(
        [parse_encoder_id(e) for e in test_results["encoder_id"]],
        index=test_results.index,
    )
    frame = pd.concat([test_results, meta], axis=1)

    ref = frame[frame["encoder_id"] == reference]
    if ref.empty:
        raise ValueError(f"reference encoder {reference!r} has no probe results")
    # One row per granularity, or the join below fans out and the mean of several
    # encoders silently becomes the baseline.
    if ref["granularity"].duplicated().any():
        raise ValueError(
            f"reference {reference!r} has duplicate granularity rows: "
            f"{ref['granularity'].tolist()}"
        )
    joint = ref.set_index("granularity")[["accuracy_micro", "f1_macro"]].rename(
        columns={"accuracy_micro": "joint_top1", "f1_macro": "joint_f1"}
    )
    before = len(frame)
    frame = frame.join(joint, on="granularity")
    assert len(frame) == before, "join fanned out; the baseline is not unique"
    frame["gap_top1"] = frame["joint_top1"] - frame["accuracy_micro"]
    frame["gap_f1"] = frame["joint_f1"] - frame["f1_macro"]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=JOINT_REFERENCE)
    parser.add_argument("--skip-drift", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_settings()
    results_dir = cfg.data_root / "results"
    test = collect_probe_results(results_dir)
    gaps = build_gap_table(test)

    # Mechanism metrics need species-level predictions per encoder.
    manifest = pd.read_parquet(
        cfg.manifest_path, columns=["species_id", "genus_id", "family_id"]
    )
    rows = []
    for encoder_id in sorted(gaps["encoder_id"].unique()):
        pred_path = (
            cfg.data_root
            / "predictions"
            / encoder_id
            / f"{Granularity.SPECIES}.parquet"
        )
        if not pred_path.exists():
            continue
        record: dict[str, object] = {"encoder_id": encoder_id}
        record |= parse_encoder_id(encoder_id)
        record |= hierarchical_error_metrics(pd.read_parquet(pred_path), manifest)
        if not args.skip_drift and encoder_id != args.reference:
            try:
                record["cka_vs_joint"] = drift_against(args.reference, encoder_id)
            except FileNotFoundError:
                record["cka_vs_joint"] = float("nan")
        rows.append(record)

    mechanism = pd.DataFrame(rows)

    gaps.to_parquet(results_dir / "gap_table.parquet", index=False)
    if not mechanism.empty:
        mechanism.to_parquet(results_dir / "mechanism.parquet", index=False)

    print("\n=== gap vs joint (test top-1) ===")
    pivot = gaps.pivot_table(
        index=["arm", "ordering", "n_chunks", "chunk_idx"],
        columns="granularity",
        values="gap_top1",
    )
    print(pivot.to_string())

    if not mechanism.empty:
        print("\n=== mechanism (species-level errors) ===")
        cols = [
            "encoder_id",
            "species_accuracy",
            "mean_hierarchical_distance",
            "contained_in_genus",
            "contained_in_family",
        ]
        if "cka_vs_joint" in mechanism:
            cols.append("cka_vs_joint")
        print(mechanism[cols].to_string(index=False))

    print(f"\nWritten to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
