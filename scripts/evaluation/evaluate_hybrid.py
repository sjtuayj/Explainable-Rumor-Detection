"""
Evaluate supervised predictions and optional SJTU LLM review for uncertain cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rumor_detection.llm_client import DEFAULT_LLM_MODEL, get_llm_client


load_dotenv()


SYSTEM_PROMPT = """You are classifying tweets from a rumor detection dataset.

Label meanings:
- 0: NOT a rumor. The tweet is an opinion, reaction, joke, question, general discussion, or an official/confirmed update.
- 1: RUMOR. The tweet makes a specific check-worthy factual claim about an event that appears unverified at posting time, speculative, disputed, or based on unclear sourcing.

Important:
- A rumor does not need to be false. It can be an unverified breaking-news claim.
- Treat unsupported claims about names, identities, causes, deaths, arrests, or what authorities supposedly said as rumor.
- Treat personal reactions and commentary without a concrete factual claim as not rumor.

Respond with ONLY a single digit: 0 or 1. Nothing else."""


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "[URL]", str(text))
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"[HASHTAG] \1", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_label(answer: str) -> int:
    match = re.search(r"\b[01]\b", answer or "")
    return int(match.group(0)) if match else -1


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def few_shot_examples(data_dir: Path, per_class: int, seed: int):
    if per_class <= 0:
        return []
    rows = load_rows(data_dir / "train.csv")
    rng = random.Random(seed)
    examples = []
    for label in (0, 1):
        candidates = [row for row in rows if row["label"] == label and len(row["text"]) <= 260]
        examples.extend(rng.sample(candidates, k=min(per_class, len(candidates))))
    rng.shuffle(examples)
    return examples


def llm_classify(client, text: str, model: str, examples, max_retries: int = 2) -> int:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": f"Tweet: {example['text']}\nLabel:"},
                {"role": "assistant", "content": str(example["label"])},
            ]
        )
    messages.append({"role": "user", "content": f"Tweet: {text}\nLabel:"})

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8,
                temperature=0,
            )
            return parse_label(response.choices[0].message.content.strip())
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
    print(f"LLM error: {last_error}", flush=True)
    return -1


def predict_supervised(model_dir: Path, texts, batch_size: int, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    preds = []
    probs = []
    cleaned = [clean_text(text) for text in texts]
    with torch.no_grad():
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu()
            probs.extend(batch_probs.tolist())
            preds.extend(batch_probs.argmax(dim=-1).tolist())
    return preds, probs


def summarize(name: str, labels, preds) -> dict:
    acc = accuracy_score(labels, preds)
    print(f"\n=== {name} ===", flush=True)
    print(f"Accuracy: {acc:.4f}", flush=True)
    print("Confusion matrix [[TN, FP], [FN, TP]]:", flush=True)
    print(confusion_matrix(labels, preds), flush=True)
    print(classification_report(labels, preds), flush=True)
    return {
        "accuracy": acc,
        "classification_report": classification_report(labels, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    cache = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache[str(item["index"])] = item
    return cache


def append_cache(cache_path: Path, item: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run(args):
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    rows = load_rows(data_dir / "val.csv")
    if args.num_samples:
        rng = random.Random(args.seed)
        indexed = list(enumerate(rows))
        indexed = rng.sample(indexed, k=min(args.num_samples, len(indexed)))
    else:
        indexed = list(enumerate(rows))

    indices = [idx for idx, _ in indexed]
    texts = [row["text"] for _, row in indexed]
    labels = [row["label"] for _, row in indexed]

    supervised_preds, probs = predict_supervised(model_dir, texts, args.batch_size, device)
    supervised_summary = summarize("Supervised", labels, supervised_preds)

    reviewed_preds = list(supervised_preds)
    uncertain = []
    for local_pos, (idx, row) in enumerate(indexed):
        prob0, prob1 = probs[local_pos]
        max_prob = max(prob0, prob1)
        margin = abs(prob1 - prob0)
        if max_prob < args.max_prob_threshold or margin < args.margin_threshold:
            uncertain.append((local_pos, idx, row, max_prob, margin))

    print(
        f"\nUncertain samples selected for LLM review: {len(uncertain)}/{len(indexed)} "
        f"(max_prob<{args.max_prob_threshold} or margin<{args.margin_threshold})",
        flush=True,
    )

    llm_items = []
    if args.review_limit is not None:
        uncertain = uncertain[: args.review_limit]

    if args.use_llm and uncertain:
        if args.rpm > 10:
            raise ValueError("SJTU API limit is 10 requests per minute; use --rpm <= 10.")
        examples = few_shot_examples(data_dir, args.few_shot_per_class, args.seed)
        client = get_llm_client()
        cache_path = output_dir / "llm_review_cache.jsonl"
        cache = load_cache(cache_path)
        last_request_at = None
        min_interval = 60.0 / args.rpm

        for count, (local_pos, idx, row, max_prob, margin) in enumerate(uncertain, start=1):
            key = str(idx)
            cached = cache.get(key)
            if cached:
                llm_pred = cached["llm_pred"]
            else:
                if last_request_at is not None:
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                last_request_at = time.monotonic()
                llm_pred = llm_classify(client, row["text"], args.llm_model, examples)
                cached = {
                    "index": idx,
                    "text": row["text"],
                    "true": row["label"],
                    "supervised_pred": supervised_preds[local_pos],
                    "probabilities": probs[local_pos],
                    "max_prob": max_prob,
                    "margin": margin,
                    "llm_pred": llm_pred,
                    "llm_model": args.llm_model,
                }
                append_cache(cache_path, cached)
                cache[key] = cached

            if llm_pred in (0, 1):
                reviewed_preds[local_pos] = llm_pred
            llm_items.append(cached)
            print(
                f"[{count}/{len(uncertain)}] idx={idx} true={row['label']} "
                f"supervised={supervised_preds[local_pos]} llm={llm_pred} "
                f"max_prob={max_prob:.3f} margin={margin:.3f}",
                flush=True,
            )

    hybrid_summary = summarize("Hybrid", labels, reviewed_preds)
    result = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "sample_indices": indices,
        "thresholds": {
            "max_prob": args.max_prob_threshold,
            "margin": args.margin_threshold,
        },
        "use_llm": args.use_llm,
        "llm_model": args.llm_model,
        "few_shot_examples": args.few_shot_per_class * 2,
        "supervised": supervised_summary,
        "hybrid": hybrid_summary,
        "uncertain_count": len(uncertain),
        "llm_reviewed_count": len(llm_items),
        "llm_items": llm_items,
    }
    output_path = output_dir / "hybrid_eval.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate supervised and hybrid rumor detection.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "rumer2026"))
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "best_model"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval_outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prob-threshold", type=float, default=0.65)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model", default=os.environ.get("SJTU_CLASSIFY_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--few-shot-per-class", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--review-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
