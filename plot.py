import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def extract_train_points(records: Sequence[Dict[str, Any]]) -> List[tuple[int, float]]:
    points: List[tuple[int, float]] = []
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
        points.append((int(r["step"]), float(loss_value)))
    return points


def extract_val_points(records: Sequence[Dict[str, Any]]) -> List[tuple[int, float]]:
    return [
        (int(r["step"]), float(r["val_loss"]))
        for r in records
        if r.get("log_step") and r.get("val_loss") is not None
    ]


def plot_series(
    runs: Sequence[Dict[str, Any]],
    title: str,
    metric: str,
    extractor: Callable[[Sequence[Dict[str, Any]]], List[tuple[int, float]]],
    output_path: Path,
) -> Optional[Path]:
    if not runs:
        print("No metric records to plot.")
        return None

    plotted = False
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title)

    for run in runs:
        records = run["records"]
        label = run["label"]
        points = extractor(records)
        if not points:
            continue
        steps, values = zip(*points)
        linestyle = "-" if metric == "train" else "--"
        ax.plot(steps, values, label=f"{label}", linewidth=2, linestyle=linestyle)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"No {metric} data available to plot.")
        return None

    ax.set_ylabel("Cross-Entropy")
    ax.set_xlabel("Step")
    ax.legend(title=f"{metric.capitalize()} loss")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {metric} plot to {output_path}")
    return output_path


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

    train_output = output_dir / f"{output_name}_train.png"
    val_output = output_dir / f"{output_name}_val.png"

    any_plotted = False
    if plot_series(runs, title + " (train)", "train", extract_train_points, train_output):
        any_plotted = True
    if plot_series(runs, title + " (val)", "val", extract_val_points, val_output):
        any_plotted = True

    if not any_plotted:
        raise SystemExit("No data was plotted. Check that metrics files contain log_step entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
