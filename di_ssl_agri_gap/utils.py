"""Seeding, logging, environment and provenance helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Paths and tracking configuration, read from the gitignored .env."""

    zip_dir: Path
    data_root: Path
    shm_root: Path
    out_dir: Path
    log_dir: Path
    use_mlflow: bool
    experiment_name: str
    experiment_eval_name: str

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def img256_dir(self) -> Path:
        return self.data_root / "img256"

    @property
    def img256_train_dir(self) -> Path:
        return self.data_root / "img256_train"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "labels.parquet"

    @property
    def label_maps_path(self) -> Path:
        return self.data_root / "label_maps.json"

    def image_root(self, staged: bool = True) -> Path:
        """Where to read 256px images from.

        Prefers the /dev/shm copy when it exists, since reading 560k small files
        straight off NFS is the bottleneck this staging step exists to remove.
        """
        staged_dir = self.shm_root / "img256"
        if staged and staged_dir.exists():
            return staged_dir
        return self.img256_dir


@cache
def load_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    def _path(key: str, default: str) -> Path:
        return Path(os.environ.get(key, default)).expanduser()

    return Settings(
        zip_dir=_path(
            "INATAG_ZIP_DIR", "/cluster/user/aschlieb/asyncio/datasets/iNatAg-mini"
        ),
        data_root=_path(
            "DATA_ROOT", "/cluster/user/aschlieb/asyncio/datasets/iNatAg-mini-work"
        ),
        shm_root=_path("SHM_ROOT", "/dev/shm/inatag-mini"),
        out_dir=_path("OUT_DIR", str(REPO_ROOT / "out")),
        log_dir=_path("LOG_DIR", str(REPO_ROOT / "logs")),
        use_mlflow=os.environ.get("USE_MLFLOW", "false").lower() == "true",
        experiment_name=os.environ.get("EXPERIMENT_NAME", "di-ssl-agri-gap-pretrain"),
        experiment_eval_name=os.environ.get(
            "EXPERIMENT_EVAL_NAME", "di-ssl-agri-gap-eval"
        ),
    )


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def seed_rng(seed: int = RANDOM_SEED, strict: bool = False) -> None:
    """Seed every RNG, optionally demanding bitwise-deterministic kernels.

    `strict=False` is mandatory for the DINOv2 pre-training stage: several
    multi-crop and attention kernels have no deterministic implementation, so
    `use_deterministic_algorithms(True)` either raises or falls back to
    pathologically slow paths. Pre-training is therefore seed-controlled but not
    bitwise reproducible, and that limitation is recorded with each run.

    `strict=True` is used for linear probing, which runs on cached features and
    is cheap enough to make full determinism free.
    """
    import torch
    from lightning.pytorch import seed_everything

    seed_everything(seed=seed, workers=True)

    if strict:
        # Must be set before the first CUBLAS handle is created.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=True)


@cache
def git_sha(short: bool = False) -> str:
    """Current commit, suffixed `-dirty` when the tree has uncommitted changes."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        sha = subprocess.check_output(
            args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def create_loggers(
    experiment_name: str,
    run_name: str,
    tags: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Build a CSV logger, plus an MLflow logger when USE_MLFLOW is set.

    MLflow credentials are read from the environment by the mlflow client
    itself (MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD), so they are
    never passed as arguments and never end up in a logged hyperparameter dict.
    """
    from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

    cfg = settings or load_settings()
    loggers: dict[str, Any] = {
        "csv": CSVLogger(
            save_dir=str(cfg.log_dir / "csv" / experiment_name), name=run_name
        )
    }

    if not cfg.use_mlflow:
        return loggers

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("USE_MLFLOW=true but MLFLOW_TRACKING_URI is unset")

    loggers["mlflow"] = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags={k: str(v) for k, v in (tags or {}).items()} | {"git_sha": git_sha()},
        log_model=False,
    )
    return loggers


def lightly_train_loggers(
    experiment_name: str, run_name: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Logger config in the shape `lightly_train.pretrain(loggers=...)` expects."""
    cfg = settings or load_settings()
    loggers: dict[str, Any] = {"jsonl": {}}
    if cfg.use_mlflow:
        loggers["mlflow"] = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        }
    return loggers


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def print_gpu_info() -> None:
    import torch

    if not torch.cuda.is_available():
        print("No CUDA devices available.")
        return
    print(f"\nCUDA {torch.version.cuda} | {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(
            f"  GPU {i}: {p.name} | {p.total_memory / 1e9:.0f} GB | "
            f"sm_{p.major}{p.minor} | {p.multi_processor_count} SMs"
        )
    print(flush=True)


def get_parameter_stats(module: Any, prefix: str = "") -> dict[str, int]:
    return {
        f"{prefix}trainable_params": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
        f"{prefix}frozen_params": sum(
            p.numel() for p in module.parameters() if not p.requires_grad
        ),
    }


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Checksum a file so a run can prove which manifest/split it consumed."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
