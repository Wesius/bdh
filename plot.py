import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


def locate_latest_metrics_file(metrics_dir: Path, model: str | None = None) -> Path:
    pattern = f"train_metrics_{model}_*.json" if model else "train_metrics_*.json"
    candidates = sorted(
        metrics_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        target = f"for model '{model}'" if model else ""
        raise FileNotFoundError(
            f"No metrics JSON files found {target} in {metrics_dir.resolve()}"
        )
    return candidates[0]


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def plot_comparison(
    runs: Sequence[Dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    if not runs:
        print("No metric records to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title)

    for run in runs:
        records = run["records"]
        label = run["label"]

        train_points = []
        for r in records:
            if not r.get("log_step"):
                continue
            loss_value = (
                r.get("train_loss_window_avg")
                or r.get("loss_window_avg")
                or r.get("train_loss")
                or r.get("loss")
            )
            if loss_value is None:
                continue
            train_points.append((r["step"], loss_value))

        val_points = [
            (r["step"], r.get("val_loss"))
            for r in records
            if r.get("log_step") and r.get("val_loss") is not None
        ]

        if train_points:
            steps, losses = zip(*train_points)
            ax.plot(steps, losses, label=f"{label} train", linewidth=2)

        if val_points:
            steps_val, losses_val = zip(*val_points)
            ax.plot(
                steps_val,
                losses_val,
                linestyle="--",
                linewidth=2,
                label=f"{label} val",
            )

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
        "--files",
        type=Path,
        nargs="*",
        help="Metrics JSON files to compare. If omitted, uses latest BDH and Transformer runs.",
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

    metrics_files: List[Path] = []
    if args.files:
        metrics_files = [path.expanduser() for path in args.files]
    else:
        for model in ("bdh", "transformer"):
            try:
                metrics_files.append(locate_latest_metrics_file(args.metrics_dir, model))
            except FileNotFoundError as exc:
                print(exc)
        if not metrics_files:
            raise FileNotFoundError(
                "No metrics files found. Provide --files or run training first."
            )

    runs: List[Dict[str, Any]] = []
    labels_for_output: List[str] = []
    for metrics_file in metrics_files:
        data = load_metrics(metrics_file)
        metadata = data.get("metadata", {})
        records = data.get("metrics", [])
        model_label = metadata.get("model_type", metrics_file.stem)
        run_id = metadata.get("run_id")
        label = model_label
        if run_id:
            label = f"{model_label} ({run_id})"
        print(f"Loaded {metrics_file} for {label}")
        runs.append({"records": records, "label": model_label})
        labels_for_output.append(model_label)

    title = " vs. ".join(labels_for_output)
    output_dir = args.output_dir.expanduser()
    if len(labels_for_output) == 1:
        output_name = metrics_files[0].stem
    else:
        unique_labels = "_".join(dict.fromkeys(labels_for_output))
        output_name = f"comparison_{unique_labels}"
    output_path = output_dir / f"{output_name}.png"

    plot_comparison(runs, title, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
