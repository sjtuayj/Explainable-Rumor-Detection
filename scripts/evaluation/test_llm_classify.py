"""
测试纯LLM分类准确率（小样本），看看能否超过传统ML
"""
import argparse
import csv
import importlib.util
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rumor_detection.llm_client import DEFAULT_LLM_MODEL, get_llm_client

DATA_DIR = os.environ.get(
    "RUMOR_DATA_DIR",
    str(PROJECT_ROOT / 'rumer2026'),
)
LLM_MODEL = os.environ.get("SJTU_CLASSIFY_MODEL", DEFAULT_LLM_MODEL)
_client = None

SYSTEM_PROMPT = """You are classifying tweets from a rumor detection dataset.

Label meanings:
- 0: NOT a rumor. The tweet is an opinion, reaction, joke, question, general discussion, or an official/confirmed update.
- 1: RUMOR. The tweet makes a specific check-worthy factual claim about an event that appears unverified at posting time, speculative, disputed, or based on unclear sourcing.

Important:
- A rumor does not need to be false. It can be an unverified breaking-news claim.
- Treat unsupported claims about names, identities, causes, deaths, arrests, or what authorities supposedly said as rumor.
- Treat personal reactions and commentary without a concrete factual claim as not rumor.

Respond with ONLY a single digit: 0 or 1. Nothing else."""


def parse_label(answer):
    match = re.search(r"\b[01]\b", answer or "")
    return int(match.group(0)) if match else -1


def get_client():
    global _client
    if _client is None:
        _client = get_llm_client()
    return _client


def llm_classify(text, model=LLM_MODEL, max_retries=2, few_shot_examples=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shot_examples or []:
        messages.extend([
            {"role": "user", "content": f"Tweet: {example['text']}\nLabel:"},
            {"role": "assistant", "content": str(example['label'])},
        ])
    messages.append({"role": "user", "content": f"Tweet: {text}\nLabel:"})

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8,
                temperature=0,
            )
            ans = resp.choices[0].message.content.strip()
            return parse_label(ans)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    print(f"Error: {last_error}", flush=True)
    return -1


def wait_for_rate_limit(last_request_at, rpm):
    if last_request_at is None:
        return time.monotonic()
    min_interval = 60.0 / rpm
    elapsed = time.monotonic() - last_request_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.monotonic()


def load_validation_rows(data_dir):
    val_path = os.path.join(data_dir, 'val.csv')
    with open(val_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['label'] = int(row['label'])
    return rows


def load_few_shot_examples(data_dir, per_class, seed):
    if per_class <= 0:
        return []
    train_path = os.path.join(data_dir, 'train.csv')
    with open(train_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['label'] = int(row['label'])
    rng = random.Random(seed)
    examples = []
    for label in (0, 1):
        candidates = [row for row in rows if row['label'] == label and len(row['text']) <= 260]
        examples.extend(rng.sample(candidates, k=min(per_class, len(candidates))))
    rng.shuffle(examples)
    return examples


def check_prerequisites(data_dir, require_train=False):
    missing = []
    if not os.path.exists(os.path.join(data_dir, 'val.csv')):
        missing.append(f"validation file not found: {os.path.join(data_dir, 'val.csv')}")
    if require_train and not os.path.exists(os.path.join(data_dir, 'train.csv')):
        missing.append(f"training file not found: {os.path.join(data_dir, 'train.csv')}")
    if not os.getenv("SJTU_API_KEY"):
        missing.append("SJTU_API_KEY is not set")
    if importlib.util.find_spec("openai") is None:
        missing.append("openai is not installed; run `pip install -r requirements.txt`")
    if missing:
        raise RuntimeError("; ".join(missing))


def test_llm(n=50, model=LLM_MODEL, rpm=10, seed=42, data_dir=DATA_DIR, few_shot_per_class=0):
    if rpm > 10:
        raise ValueError("SJTU API limit is 10 requests per minute; use --rpm <= 10.")
    check_prerequisites(data_dir, require_train=few_shot_per_class > 0)
    rows = load_validation_rows(data_dir)
    sample = random.Random(seed).sample(rows, k=min(n, len(rows)))
    few_shot_examples = load_few_shot_examples(data_dir, few_shot_per_class, seed)

    correct = 0
    total = 0
    skipped = 0
    tp = tn = fp = fn = 0
    last_request_at = None
    for idx, row in enumerate(sample, start=1):
        last_request_at = wait_for_rate_limit(last_request_at, rpm)
        pred = llm_classify(row['text'], model=model, few_shot_examples=few_shot_examples)
        if pred == -1:
            skipped += 1
            print(f"[{idx}/{len(sample)}] skipped true={row['label']}", flush=True)
            continue
        total += 1
        is_correct = pred == row['label']
        correct += int(is_correct)
        if pred == 1 and row['label'] == 1:
            tp += 1
        elif pred == 0 and row['label'] == 0:
            tn += 1
        elif pred == 1 and row['label'] == 0:
            fp += 1
        elif pred == 0 and row['label'] == 1:
            fn += 1
        mark = "OK" if is_correct else "ERR"
        print(f"[{idx}/{len(sample)}] {mark} pred={pred} true={row['label']}", flush=True)

    acc = correct / total if total > 0 else 0
    print(f"Model: {model}", flush=True)
    print(f"Few-shot examples: {len(few_shot_examples)}", flush=True)
    print(f"Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}", flush=True)
    print(f"LLM accuracy on {total} samples: {acc:.4f} ({correct}/{total}), skipped={skipped}", flush=True)
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SJTU OpenAI-compatible LLM classification.")
    parser.add_argument("-n", "--num-samples", type=int, default=50)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--few-shot-per-class", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    test_llm(
        n=args.num_samples,
        model=args.model,
        rpm=args.rpm,
        seed=args.seed,
        data_dir=args.data_dir,
        few_shot_per_class=args.few_shot_per_class,
    )


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f"Cannot run SJTU LLM evaluation: {e}", file=sys.stderr)
        sys.exit(1)
