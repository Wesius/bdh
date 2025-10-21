# Copyright Pathway Technology, Inc.

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import VanillaTransformer, VanillaTransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
EVAL_ITERS = 20

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


class MetricsLogger:
    """Collects and persists training metrics to JSON on a fixed cadence."""

    def __init__(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self) -> None:
        if not self.records:
            return
        payload = {
            "metadata": self.metadata,
            "metrics": self.records,
        }
        serialized = json.dumps(payload, indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self.file_path.parent
        ) as tmp_file:
            tmp_file.write(serialized)
            tmp_name = tmp_file.name
        os.replace(tmp_name, self.file_path)


def compute_grad_norm(parameters) -> float:
    grads = [
        p.grad.detach().to(dtype=torch.float32).cpu()
        for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return 0.0
    stacked = torch.stack([g.norm(2) for g in grads])
    return float(torch.norm(stacked, 2).item())


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


@torch.no_grad()
def evaluate_split(model: nn.Module, split: str, num_iters: int = EVAL_ITERS) -> float:
    model.eval()
    losses = []
    for _ in range(num_iters):
        x, y = get_batch(split)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.detach())
    model.train()
    losses_tensor = torch.stack(losses)
    return float(losses_tensor.mean().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train BDH or a vanilla Transformer")
    parser.add_argument(
        "--model",
        choices=("bdh", "transformer"),
        default="bdh",
        help="Which architecture to train (default: bdh)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_data()

    if args.model == "bdh":
        model = bdh.BDH(BDH_CONFIG).to(device)
    else:
        transformer_config = VanillaTransformerConfig(
            vocab_size=BDH_CONFIG.vocab_size,
            block_size=BLOCK_SIZE,
            # Match BDH's parameter budget (~25.3M) with a more balanced GPT-style layout
            n_layer=8,
            n_head=8,
            n_embd=312,
            dropout=BDH_CONFIG.dropout,
            mlp_hidden_multiplier=14,
        )
        model = VanillaTransformer(transformer_config).to(device)

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"Training model: {args.model}")

    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%SZ")
    metrics_path = (
        Path(__file__).resolve().parent
        / "metrics"
        / f"train_metrics_{args.model}_{run_id}.json"
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_logger = MetricsLogger(
        metrics_path,
        metadata={
            "run_id": run_id,
            "started_at": run_started_at.isoformat(),
            "device": str(device),
            "dtype": dtype,
            "model_type": args.model,
            "max_iters": MAX_ITERS,
            "log_frequency": LOG_FREQ,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "trainable_params": trainable_params,
        },
    )

    x, y = get_batch("train")

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    tokens_seen = 0
    loss_window_total = 0.0
    loss_window_steps = 0
    loss_cumulative_total = 0.0
    latest_val_loss = None
    train_start_time = time.time()
    last_step_time = train_start_time

    for step in range(1, MAX_ITERS + 1):
        with ctx:
            logits, loss = model(x, y)

        train_loss = float(loss.detach().item())
        loss_window_total += train_loss
        loss_window_steps += 1
        loss_cumulative_total += train_loss

        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = compute_grad_norm(model.parameters())

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        current_time = time.time()
        step_duration = current_time - last_step_time
        last_step_time = current_time
        wall_time_sec = current_time - train_start_time

        tokens_seen += tokens_per_step
        current_lr = optimizer.param_groups[0]["lr"]
        window_avg_loss = loss_window_total / loss_window_steps
        cumulative_avg_loss = loss_cumulative_total / step
        window_index = (step - 1) // LOG_FREQ

        is_log_step = step % LOG_FREQ == 0 or step == MAX_ITERS
        val_loss = None
        if is_log_step:
            val_loss = evaluate_split(model, "val")
            latest_val_loss = val_loss

        metrics_logger.add(
            {
                "step": step,
                "train_loss": train_loss,
                "train_loss_window_avg": window_avg_loss,
                "train_loss_cumulative_avg": cumulative_avg_loss,
                "val_loss": val_loss if val_loss is not None else latest_val_loss,
                "learning_rate": current_lr,
                "grad_norm": grad_norm,
                "tokens_processed": tokens_seen,
                "tokens_in_step": tokens_per_step,
                "step_duration_sec": step_duration,
                "wall_time_sec": wall_time_sec,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_index": window_index,
                "log_step": is_log_step,
                "model_type": args.model,
            }
        )

        if is_log_step:
            print(
                (
                    f"Step {step:5d}/{MAX_ITERS} "
                    f"train_loss={train_loss:.4f} "
                    f"window_avg_loss={window_avg_loss:.4f} "
                    f"cumulative_avg_loss={cumulative_avg_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"grad_norm={grad_norm:.4f} "
                    f"lr={current_lr:.6f} "
                    f"tokens_seen={tokens_seen} "
                    f"step_time={step_duration:.4f}s"
                )
            )
            metrics_logger.flush()
            loss_window_total = 0.0
            loss_window_steps = 0

        x, y = get_batch("train")

    metrics_logger.flush()
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
