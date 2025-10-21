from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from datasets import load_dataset


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_tiny_shakespeare(data_dir: Path) -> Dict[str, Path | int]:
    """Download tiny Shakespeare if needed and return split paths/tokens."""

    data_dir = data_dir / "tinyshakespeare"
    ensure_directory(data_dir)
    input_path = data_dir / "input.txt"
    if not input_path.exists():
        data_url = (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt"
        )
        import requests

        response = requests.get(data_url, timeout=60)
        response.raise_for_status()
        input_path.write_text(response.text, encoding="utf-8")

    raw_bytes = input_path.read_bytes()
    cutoff = int(0.9 * len(raw_bytes))
    train_bytes = raw_bytes[:cutoff]
    val_bytes = raw_bytes[cutoff:]

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists():
        train_path.write_bytes(train_bytes)
    if not val_path.exists():
        val_path.write_bytes(val_bytes)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "train_tokens": len(train_bytes),
        "val_tokens": len(val_bytes),
        "languages": ["en"],
        "dataset_name": "tinyshakespeare",
    }


def prepare_translation_dataset(
    data_dir: Path,
    source_languages: Sequence[str],
    base_language: str = "en",
    max_train_examples: int | None = 100_000,
    max_val_examples: int | None = 4_000,
    seed: int = 1337,
) -> Dict[str, Path | int | Sequence[str]]:
    """Build cached translation corpora for the requested language pairs."""

    languages_sorted = tuple(sorted(source_languages))
    key = "_".join(languages_sorted)
    subdir = data_dir / f"wmt19_{key}_{base_language}_train{max_train_examples or 'all'}_val{max_val_examples or 'all'}"
    ensure_directory(subdir)

    split_info: Dict[str, Dict[str, Path | int]] = {}
    for split, max_examples in (("train", max_train_examples), ("validation", max_val_examples)):
        output_path = subdir / f"{split}.bin"
        if not output_path.exists():
            total_bytes = 0
            with output_path.open("wb") as outfile:
                for lang in source_languages:
                    dataset_id = f"{lang}-{base_language}"
                    dataset = load_dataset("wmt/wmt19", dataset_id, split=split)
                    num_samples = len(dataset)
                    take = None
                    if max_examples is not None and max_examples > 0:
                        take = min(num_samples, max_examples)
                        dataset = dataset.shuffle(seed=seed).select(range(take))
                    elif max_examples is not None and max_examples <= 0:
                        take = num_samples
                    for sample in dataset:
                        translation = sample.get("translation") or {}
                        src = translation.get(lang, "").strip()
                        tgt = translation.get(base_language, "").strip()
                        if not src or not tgt:
                            continue
                        text = (
                            f"<src:{lang}> {src}\n"
                            f"<tgt:{base_language}> {tgt}\n\n"
                        )
                        encoded = text.encode("utf-8")
                        outfile.write(encoded)
                        total_bytes += len(encoded)
            byte_count = total_bytes
        else:
            byte_count = output_path.stat().st_size
        split_info[split] = {"path": output_path, "tokens": byte_count}

    return {
        "train_path": split_info["train"]["path"],
        "val_path": split_info["validation"]["path"],
        "train_tokens": split_info["train"]["tokens"],
        "val_tokens": split_info["validation"]["tokens"],
        "languages": list(source_languages),
        "dataset_name": "wmt19",
        "base_language": base_language,
        "max_train_examples": max_train_examples,
        "max_val_examples": max_val_examples,
    }
