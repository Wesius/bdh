import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def locate_latest_metrics_file(metrics_dir: Path) -> Path:
    candidates = sorted(
        metrics_dir.glob("train_metrics_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No metrics JSON files found in {metrics_dir.resolve()}"
        )
    return candidates[0]


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def plot_metrics(records: List[Dict[str, Any]], title: str) -> None:
    if not records:
        print("No metric records to plot.")
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    window_losses = [r.get("loss_window_avg") for r in records]
    cumulative_losses = [r.get("loss_cumulative_avg") for r in records]
    grad_norms = [r.get("grad_norm") for r in records]
    lr_values = [r.get("learning_rate") for r in records]
    step_duration = [r.get("step_duration_sec") for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    fig.suptitle(title)

    ax_loss = axes[0, 0]
    ax_loss.plot(steps, losses, label="loss", alpha=0.6)
    if any(window_losses):
        ax_loss.plot(steps, window_losses, label="window avg", linewidth=2)
    if any(cumulative_losses):
        ax_loss.plot(steps, cumulative_losses, label="cumulative avg", linewidth=2)
    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.3)

    ax_grad = axes[1, 0]
    if any(grad_norms):
        ax_grad.plot(steps, grad_norms, color="tab:orange")
    ax_grad.set_ylabel("Grad Norm (L2)")
    ax_grad.set_xlabel("Step")
    ax_grad.grid(True, linestyle="--", alpha=0.3)

    ax_lr = axes[0, 1]
    if any(lr_values):
        ax_lr.plot(steps, lr_values, color="tab:green")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.grid(True, linestyle="--", alpha=0.3)

    ax_time = axes[1, 1]
    if any(step_duration):
        ax_time.plot(steps, step_duration, color="tab:red")
    ax_time.set_ylabel("Step Time (s)")
    ax_time.set_xlabel("Step")
    ax_time.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Plot training metrics recorded by train.py"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to metrics JSON file (defaults to latest in ./metrics)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metrics",
        help="Directory to search for metrics files when --file is not provided.",
    )
    args = parser.parse_args(argv)

    metrics_file = args.file
    if metrics_file is None:
        metrics_file = locate_latest_metrics_file(args.metrics_dir)
        print(f"Using latest metrics file: {metrics_file}")
    else:
        metrics_file = metrics_file.expanduser()

    data = load_metrics(metrics_file)
    metadata = data.get("metadata", {})
    records = data.get("metrics", [])

    title_bits = [
        metadata.get("run_id", "training run"),
        metadata.get("device"),
        metadata.get("dtype"),
    ]
    title = " | ".join(filter(None, title_bits))

    plot_metrics(records, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
