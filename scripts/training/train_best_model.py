"""
Train the supervised rumor detector and save the best checkpoint to best_model/.

This is a local-friendly replacement for the original server script: paths,
device, model name, seeds, and hyperparameters are configurable from the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "[URL]", str(text))
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"[HASHTAG] \1", text)
    return re.sub(r"\s+", " ", text).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


def collate_fn(batch):
    inputs = {k: torch.stack([item[0][k] for item in batch]) for k in batch[0][0]}
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


@dataclass
class EpochMetric:
    epoch: int
    train_loss: float
    val_accuracy: float
    elapsed_seconds: float


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            all_probs.extend(probs.tolist())
            all_preds.extend(probs.argmax(dim=-1).tolist())
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels, all_probs


def train(args) -> dict:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    metrics_path = save_dir / "training_metrics.json"

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)
    print(f"Save dir: {save_dir}", flush=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    train_texts = [clean_text(t) for t in train_df["text"]]
    val_texts = [clean_text(t) for t in val_df["text"]]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    ).to(device)

    train_set = RumorDataset(train_texts, train_df["label"].tolist(), tokenizer, args.max_len)
    val_set = RumorDataset(val_texts, val_df["label"].tolist(), tokenizer, args.max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[EpochMetric] = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.time()
        model.train()
        total_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs).logits
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if args.log_every and step % args.log_every == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={total_loss / step:.4f}",
                    flush=True,
                )

        acc, preds, labels, _ = evaluate_model(model, val_loader, device)
        metric = EpochMetric(
            epoch=epoch,
            train_loss=total_loss / max(1, len(train_loader)),
            val_accuracy=acc,
            elapsed_seconds=time.time() - epoch_started_at,
        )
        history.append(metric)
        print(
            f"Epoch {epoch}/{args.epochs} | Loss: {metric.train_loss:.4f} "
            f"| Val Acc: {acc:.4f} | {metric.elapsed_seconds:.1f}s",
            flush=True,
        )

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            stale_epochs = 0
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  -> saved best checkpoint ({best_acc:.4f})", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epochs.", flush=True)
                break

    report = classification_report(labels, preds, output_dict=True)
    result = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "total_seconds": time.time() - started_at,
        "hyperparameters": vars(args),
        "history": [asdict(item) for item in history],
        "final_classification_report": report,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Best Val Acc: {best_acc:.4f} at epoch {best_epoch}", flush=True)
    print(f"Metrics written to {metrics_path}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train the best local rumor detector.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "rumer2026"))
    parser.add_argument("--save-dir", default=str(PROJECT_ROOT / "best_model"))
    parser.add_argument("--model-name", default="roberta-large")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
