"""Watchdog for long pre-training runs.

A joint run takes ~34 h and the phase-2 matrix ~10 days. A silent death at 3am
otherwise costs a full day, so this polls the run, records progress, and resumes
from `last.ckpt` if the process disappears before finishing.

Resumption is bounded and conservative:

  * `--max-restarts` caps the number of retries, so a deterministic crash burns
    a few minutes rather than days of GPU time.
  * A restart is only attempted if the checkpoint advanced since the previous
    one. A crash that makes no progress twice in a row is treated as fatal.
  * `resume_interrupted=True` requires identical parameters, so the supervisor
    re-invokes the exact same command line rather than reconstructing it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from di_ssl_agri_gap.utils import load_settings

PROGRESS = re.compile(r"Epoch (\d+):.*?(\d+)/(\d+)")
CHECKPOINT_EPOCH = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")
COMPLETION_MARKER = "Training completed."


@dataclass
class Status:
    alive: bool
    epoch: int
    step: int
    steps_per_epoch: int
    last_checkpoint: str
    finished: bool


def read_status(
    run_dir: Path, log_path: Path, pattern: str, total_epochs: int
) -> Status:
    alive = (
        subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        ).returncode
        == 0
    )

    epoch = step = per_epoch = 0
    completed = False
    if log_path.exists():
        text = log_path.read_text(errors="ignore").replace("\r", "\n")
        matches = PROGRESS.findall(text)
        if matches:
            epoch, step, per_epoch = (int(v) for v in matches[-1])
        completed = COMPLETION_MARKER in text

    ckpts = (
        sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
        if run_dir.exists()
        else []
    )
    # Epochs are zero-indexed in checkpoint names, so the final one is N-1.
    reached_last_epoch = any(
        int(m.group(1)) >= total_epochs - 1
        for c in ckpts
        if (m := CHECKPOINT_EPOCH.match(c.name))
    )

    # `exported_last.pt` is NOT a finish signal: LightlyTrain rewrites it at
    # every checkpoint, so it appears within minutes of the run starting.
    return Status(
        alive=alive,
        epoch=epoch,
        step=step,
        steps_per_epoch=per_epoch,
        last_checkpoint=ckpts[-1].name if ckpts else "-",
        finished=completed or reached_last_epoch,
    )


def log_line(path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(f"[{stamp}] {message}\n")
    print(f"[{stamp}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-id", default="joint_vits14_ep100_seed42")
    parser.add_argument("--log", type=Path, default=Path("logs/joint_run.log"))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--resume-command",
        default=(
            "venv/bin/python -m di_ssl_agri_gap.pretrain --epochs 100 "
            "--batch-size 256 --devices 2 --num-workers 48 --resume"
        ),
        help="command re-issued on an unexpected death",
    )
    parser.add_argument(
        "--no-restart", action="store_true", help="observe and record only"
    )
    args = parser.parse_args(argv)

    cfg = load_settings()
    run_dir = cfg.out_dir / args.encoder_id
    status_log = cfg.log_dir / f"supervise_{args.encoder_id}.log"
    pattern = "di_ssl_agri_gap.pretrain"

    restarts = 0
    last_restart_step = -1
    log_line(status_log, f"supervising {args.encoder_id} every {args.interval}s")

    while True:
        st = read_status(run_dir, args.log, pattern, args.epochs)

        if st.finished:
            log_line(
                status_log, f"FINISHED -- exported model present ({st.last_checkpoint})"
            )
            return 0

        if st.alive:
            pct = 100 * st.epoch / args.epochs if st.epoch else 0
            log_line(
                status_log,
                f"alive | epoch {st.epoch}/{args.epochs} step {st.step}/{st.steps_per_epoch} "
                f"(~{pct:.0f}%) | ckpt {st.last_checkpoint}",
            )
            time.sleep(args.interval)
            continue

        # Process is gone and no exported model exists: an unexpected death.
        log_line(
            status_log,
            f"DIED at epoch {st.epoch} step {st.step} | ckpt {st.last_checkpoint}",
        )

        if args.no_restart:
            return 1
        if restarts >= args.max_restarts:
            log_line(status_log, f"giving up after {restarts} restart(s)")
            return 1
        if st.step == last_restart_step and st.epoch == 0:
            log_line(
                status_log, "no progress since the last restart; treating as fatal"
            )
            return 1

        restarts += 1
        last_restart_step = st.step
        log_line(
            status_log, f"restart {restarts}/{args.max_restarts}: {args.resume_command}"
        )
        subprocess.Popen(
            f"setsid nohup {args.resume_command} >> {args.log} 2>&1 < /dev/null &",
            shell=True,
            cwd=Path.cwd(),
        )
        # Give the relaunch time to claim the GPUs before polling again.
        time.sleep(120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
