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


def plot_metrics(records: List[Dict[str, Any]], title: str, output_path: Path) -> None:
    if not records:
        print("No metric records to plot.")
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    window_losses = [r.get("loss_window_avg") for r in records]
    cumulative_losses = [r.get("loss_cumulative_avg") for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    ax.plot(steps, losses, label="loss", alpha=0.6)
    if any(window_losses):
        ax.plot(steps, window_losses, label="window avg", linewidth=2)
    if any(cumulative_losses):
        ax.plot(steps, cumulative_losses, label="cumulative avg", linewidth=2)
    ax.set_ylabel("Cross-Entropy")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory to store generated plots (defaults to ./plots).",
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

    output_dir = args.output_dir.expanduser()
    output_path = output_dir / f"{metrics_file.stem}.png"

    plot_metrics(records, title, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
