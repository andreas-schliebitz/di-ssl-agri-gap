"""Orchestration entry point.

Stages are separate subcommands rather than one pipeline, because the expensive
ones run for days and need to be launched, monitored and resumed independently.
The usual order:

    prepare        extract the archives, verify, optionally delete them
    manifest       build labels.parquet (taxonomy join + frozen split)
    resize         re-encode to 256px short side
    stage          copy into /dev/shm and build the train-only view
    pretrain       joint SSL pre-training                     (phase 1)
    evaluate       extract features + four linear probes
    incremental    chained SSL pre-training over N chunks     (phase 2)
    analyse        gap tables and the three mechanism metrics

`evaluate` is the one to run after every encoder, joint or incremental.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from di_ssl_agri_gap.enums import Ordering
from di_ssl_agri_gap.utils import load_settings


def _evaluate(args: argparse.Namespace) -> int:
    """Extract features for one encoder, then probe all four granularities."""
    from di_ssl_agri_gap.embed import embed_encoder
    from di_ssl_agri_gap.probe import log_to_mlflow, persist_results, probe_encoder

    cfg = load_settings()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    embed_encoder(args.encoder_id, checkpoint, args.batch_size, args.num_workers)

    results = probe_encoder(args.encoder_id, seed=args.seed)
    persist_results(args.encoder_id, results)
    log_to_mlflow(args.encoder_id, results, json.loads(args.tags))
    print(results[results["split"] == "test"].to_string(index=False))
    return 0


def _evaluate_all(args: argparse.Namespace) -> int:
    """Evaluate every encoder that has an exported backbone, plus off-the-shelf."""
    from di_ssl_agri_gap.embed import embed_encoder
    from di_ssl_agri_gap.probe import log_to_mlflow, persist_results, probe_encoder

    cfg = load_settings()
    out_dir = cfg.data_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path | None]] = [("offtheshelf_vits14", None)]
    for run_dir in sorted(cfg.out_dir.iterdir()):
        exported = run_dir / "exported_models" / "exported_last.pt"
        if exported.exists():
            targets.append((run_dir.name, exported))

    print(f"{len(targets)} encoder(s) to evaluate")
    failures: list[str] = []
    for encoder_id, checkpoint in targets:
        marker = out_dir / f"probe_{encoder_id}.parquet"
        if marker.exists() and not args.force:
            print(f"  {encoder_id}: already done, skipping")
            continue
        try:
            embed_encoder(encoder_id, checkpoint, args.batch_size, args.num_workers)
            results = probe_encoder(encoder_id, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            # One bad encoder must not cost the other 27 their evaluation.
            print(f"ERROR evaluating {encoder_id}: {type(exc).__name__}: {exc}")
            failures.append(encoder_id)
            continue
        persist_results(encoder_id, results)
        log_to_mlflow(encoder_id, results, {})

    if failures:
        print(f"\n{len(failures)} encoder(s) failed evaluation: {failures}")
        return 1
    return 0


def _incremental(args: argparse.Namespace) -> int:
    from di_ssl_agri_gap.incremental import run_arm

    orderings = [Ordering(args.ordering)] if args.ordering else list(Ordering)
    for ordering in orderings:
        for n in args.n_chunks:
            run_arm(
                ordering=ordering,
                n_chunks=n,
                seed=args.seed,
                devices=args.devices,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, module in [
        ("prepare", "di_ssl_agri_gap.prepare"),
        ("manifest", "di_ssl_agri_gap.manifest"),
        ("resize", "di_ssl_agri_gap.resize"),
        ("stage", "di_ssl_agri_gap.stage_shm"),
        ("pretrain", "di_ssl_agri_gap.pretrain"),
        ("stream", "di_ssl_agri_gap.stream"),
        ("analyse", "di_ssl_agri_gap.analysis"),
    ]:
        p = sub.add_parser(name, help=f"run {module} (pass --help to it via --)")
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(_delegate=module)

    ev = sub.add_parser("evaluate", help="extract features + probe one encoder")
    ev.add_argument("--encoder-id", required=True)
    ev.add_argument("--checkpoint", default=None)
    ev.add_argument("--batch-size", type=int, default=512)
    ev.add_argument("--num-workers", type=int, default=32)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--tags", default="{}")
    ev.set_defaults(_run=_evaluate)

    ea = sub.add_parser("evaluate-all", help="probe every exported encoder")
    ea.add_argument("--batch-size", type=int, default=512)
    ea.add_argument("--num-workers", type=int, default=32)
    ea.add_argument("--seed", type=int, default=42)
    ea.add_argument("--force", action="store_true")
    ea.set_defaults(_run=_evaluate_all)

    inc = sub.add_parser("incremental", help="chained pre-training (phase 2)")
    inc.add_argument("--ordering", choices=[str(o) for o in Ordering], default=None)
    inc.add_argument("--n-chunks", type=int, nargs="+", default=[2, 4, 8])
    inc.add_argument("--seed", type=int, default=42)
    inc.add_argument("--devices", type=int, default=2)
    inc.add_argument("--num-workers", type=int, default=48)
    inc.add_argument("--overwrite", action="store_true")
    inc.set_defaults(_run=_incremental)

    args = parser.parse_args(argv)

    if hasattr(args, "_delegate"):
        import importlib

        module = importlib.import_module(args._delegate)
        rest = [a for a in args.rest if a != "--"]
        return int(module.main(rest) or 0)
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())
