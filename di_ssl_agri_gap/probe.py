"""Stage 8: four linear probes on cached frozen features.

One encoder, four nested targets: species (2,958), genus (1,171), family (215)
and crop/weed (2). The whole point of the study is that a single averaged number
hides how differently these behave, so every probe reports the same metric set
and nothing is collapsed.

Features are cached and GPU-resident (roughly 860 MB in fp16 for all 560k rows
at 768-d), so a probe trains in seconds and a six-point LR sweep across four
granularities is minutes rather than hours.

One metric subtlety drives a lot of this file: torchmetrics'
`MulticlassAccuracy` defaults to `average="macro"`. With 2,958 classes at 50-200
images each, macro accuracy is *not* top-1 and differs substantially from it.
The iNatAg reference numbers are overall (micro) accuracy, so both are logged.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from di_ssl_agri_gap.enums import Granularity, Split
from di_ssl_agri_gap.utils import git_sha, load_settings, seed_rng

LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
EPOCHS = 40
BATCH_SIZE = 4096
WEIGHT_DECAY = 0.0


@dataclass
class ProbeResult:
    encoder_id: str
    granularity: str
    learning_rate: float
    split: str
    accuracy_micro: float
    accuracy_macro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    loss: float


class LinearProbe(nn.Module):
    """A single fully-connected layer over frozen features."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def load_features(cache_dir: Path, split: Split, device: torch.device) -> Tensor:
    array = np.load(cache_dir / f"{split.value}.npy")
    # Kept in fp32 on device: 420k x 768 fp32 is ~1.3 GB, which fits alongside a
    # probe and avoids repeated host-device transfer entirely.
    return torch.from_numpy(array).to(device, dtype=torch.float32)


def load_targets(
    manifest: pd.DataFrame, split: Split, granularity: Granularity, device: torch.device
) -> Tensor:
    # Sorted by rel_path_256 to match the extraction order in embed.py exactly.
    rows = manifest.loc[manifest["split"] == split.value].sort_values("rel_path_256")
    return torch.from_numpy(rows[granularity.label_column].to_numpy(dtype=np.int64)).to(
        device
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    num_classes: int,
    batch_size: int = 16384,
) -> tuple[dict[str, float], Tensor]:
    from torchmetrics.functional.classification import (
        multiclass_accuracy,
        multiclass_f1_score,
        multiclass_precision,
        multiclass_recall,
    )

    model.eval()
    logits = torch.cat(
        [
            model(features[i : i + batch_size])
            for i in range(0, len(features), batch_size)
        ]
    )
    preds = logits.argmax(dim=1)

    with _relaxed_determinism():
        metrics = {
            # average="micro" is top-1; the torchmetrics default of "macro" is not.
            "accuracy_micro": multiclass_accuracy(
                preds, targets, num_classes, average="micro"
            ).item(),
            "accuracy_macro": multiclass_accuracy(
                preds, targets, num_classes, average="macro"
            ).item(),
            "precision_macro": multiclass_precision(
                preds, targets, num_classes, average="macro"
            ).item(),
            "recall_macro": multiclass_recall(
                preds, targets, num_classes, average="macro"
            ).item(),
            "f1_macro": multiclass_f1_score(
                preds, targets, num_classes, average="macro"
            ).item(),
            "loss": F.cross_entropy(logits, targets).item(),
        }
    return metrics, preds


@contextmanager
def _relaxed_determinism():
    """Disable deterministic algorithms for the metric computation only.

    Under `torch.use_deterministic_algorithms(True)`, torchmetrics swaps
    `torch.bincount` for a fallback that materialises
    `arange(num_classes**2).repeat(len(x), 1)`
    (`torchmetrics/utilities/data.py:203`). At 2,958 species and 84,126 val rows
    that is an 84,126 x 8,749,764 tensor -- 5.5 TiB -- and OOMs instantly.

    Relaxing the flag here is safe: the metrics are computed from `preds`, which
    are already fixed by this point. Kernel-level nondeterminism affects the
    order of atomic adds, not the value of an integer count, so the metrics are
    deterministic either way. Probe *training*, where nondeterminism would
    actually change results, stays strict.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def train_probe(
    train_x: Tensor,
    train_y: Tensor,
    num_classes: int,
    learning_rate: float,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> nn.Module:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # `LinearProbe.__init__` draws from the *global* RNG, so seeding only the
    # shuffle generator leaves the initial weights different on every call.
    # Seeding globally here also fixes the init across the whole LR sweep, so a
    # sweep compares learning rates rather than lucky initialisations.
    torch.manual_seed(seed)
    model = LinearProbe(train_x.shape[1], num_classes).to(train_x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    steps = epochs * max(1, len(train_x) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator).to(train_x.device)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


def probe_encoder(
    encoder_id: str,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    epochs: int = EPOCHS,
    seed: int = 42,
    granularities: tuple[Granularity, ...] | None = None,
) -> pd.DataFrame:
    """Sweep LRs for all four granularities; select on val, report on test."""
    cfg = load_settings()
    seed_rng(seed, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = cfg.data_root / "features" / encoder_id
    manifest = pd.read_parquet(cfg.manifest_path)

    features = {s: load_features(cache_dir, s, device) for s in Split}
    print(f"\nEncoder {encoder_id}: feature dim {features[Split.TRAIN].shape[1]}")

    records: list[ProbeResult] = []
    predictions_dir = cfg.data_root / "predictions" / encoder_id
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for granularity in granularities or tuple(Granularity):
        targets = {s: load_targets(manifest, s, granularity, device) for s in Split}
        n_classes = granularity.num_classes
        observed = int(targets[Split.TRAIN].max().item()) + 1
        if observed != n_classes:
            raise ValueError(
                f"{granularity}: manifest has {observed} classes, expected {n_classes}"
            )

        best = None
        for lr in learning_rates:
            model = train_probe(
                features[Split.TRAIN],
                targets[Split.TRAIN],
                n_classes,
                lr,
                epochs,
                seed=seed,
            )
            val_metrics, _ = evaluate(
                model, features[Split.VAL], targets[Split.VAL], n_classes
            )
            records.append(
                ProbeResult(
                    encoder_id, str(granularity), lr, Split.VAL.value, **val_metrics
                )
            )
            print(
                f"  {granularity:9s} lr={lr:<7g} "
                f"val top1={val_metrics['accuracy_micro']:.4f} "
                f"macroF1={val_metrics['f1_macro']:.4f}",
                flush=True,
            )
            # Selection on top-1, the metric the iNatAg reference curve reports.
            if best is None or val_metrics["accuracy_micro"] > best[1]:
                best = (lr, val_metrics["accuracy_micro"], model)

        assert best is not None
        best_lr, _, best_model = best
        test_metrics, preds = evaluate(
            best_model, features[Split.TEST], targets[Split.TEST], n_classes
        )
        records.append(
            ProbeResult(
                encoder_id, str(granularity), best_lr, Split.TEST.value, **test_metrics
            )
        )
        print(
            f"  {granularity:9s} BEST lr={best_lr:<7g} "
            f"TEST top1={test_metrics['accuracy_micro']:.4f} "
            f"macroF1={test_metrics['f1_macro']:.4f}",
            flush=True,
        )

        # Per-image test predictions feed the hierarchical error analysis in
        # phase 2; recomputing them later would mean re-running every encoder.
        test_rows = (
            manifest.loc[manifest["split"] == Split.TEST.value]
            .sort_values("rel_path_256")
            .reset_index(drop=True)
        )
        pd.DataFrame(
            {
                "rel_path_256": test_rows["rel_path_256"],
                "y_true": targets[Split.TEST].cpu().numpy(),
                "y_pred": preds.cpu().numpy(),
            }
        ).to_parquet(predictions_dir / f"{granularity}.parquet", index=False)

    return pd.DataFrame([asdict(r) for r in records])


def persist_results(encoder_id: str, results: pd.DataFrame) -> Path:
    """Write probe results locally, in parquet and CSV, before anything remote.

    Parquet is the machine-readable record; the CSV mirror is deliberate
    redundancy -- it is readable without pyarrow and survives a parquet version
    mismatch, which matters when the runs producing it take days.
    """
    cfg = load_settings()
    out_dir = cfg.data_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"probe_{encoder_id}.parquet"
    results.to_parquet(target, index=False)
    results.to_csv(target.with_suffix(".csv"), index=False)
    return target


def log_to_mlflow(encoder_id: str, results: pd.DataFrame, tags: dict[str, str]) -> None:
    cfg = load_settings()
    if not cfg.use_mlflow:
        return
    try:
        _log_to_mlflow_impl(encoder_id, results, tags)
    except Exception as exc:  # noqa: BLE001
        # The local parquet/CSV is already written, so the *result* is safe. What
        # must not happen is a transient tracking-server failure aborting a batch
        # of evaluations and silently leaving later encoders unprobed. Record the
        # gap so it can be re-pushed with `mlflow_resync`.
        pending = cfg.data_root / "results" / "mlflow_pending.txt"
        pending.parent.mkdir(parents=True, exist_ok=True)
        with pending.open("a") as fh:
            fh.write(f"{encoder_id}\t{type(exc).__name__}: {exc}\n")
        print(
            f"WARNING: MLflow logging failed for {encoder_id} ({type(exc).__name__}). "
            f"Results are on disk; queued for resync in {pending}",
            flush=True,
        )


def _log_to_mlflow_impl(
    encoder_id: str, results: pd.DataFrame, tags: dict[str, str]
) -> None:
    cfg = load_settings()
    import mlflow

    mlflow.set_tracking_uri(str(__import__("os").environ["MLFLOW_TRACKING_URI"]))
    mlflow.set_experiment(cfg.experiment_eval_name)

    test = results[results["split"] == Split.TEST.value]
    for row in test.itertuples(index=False):
        with mlflow.start_run(run_name=f"{encoder_id}__{row.granularity}"):
            mlflow.set_tags(
                tags
                | {
                    "granularity": row.granularity,
                    "encoder_id": encoder_id,
                    "git_sha": git_sha(),
                }
            )
            mlflow.log_param("learning_rate", row.learning_rate)
            mlflow.log_param("probe_epochs", EPOCHS)
            mlflow.log_metrics(
                {
                    "test_accuracy_micro": row.accuracy_micro,
                    "test_accuracy_macro": row.accuracy_macro,
                    "test_precision_macro": row.precision_macro,
                    "test_recall_macro": row.recall_macro,
                    "test_f1_macro": row.f1_macro,
                    "test_loss": row.loss,
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tags", type=str, default="{}", help="JSON dict of MLflow tags"
    )
    args = parser.parse_args(argv)

    cfg = load_settings()
    results = probe_encoder(args.encoder_id, epochs=args.epochs, seed=args.seed)

    out_dir = cfg.data_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"probe_{args.encoder_id}.parquet"
    results.to_parquet(target, index=False)

    log_to_mlflow(args.encoder_id, results, json.loads(args.tags))

    print(f"\nResults -> {target}")
    print(results[results["split"] == "test"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
