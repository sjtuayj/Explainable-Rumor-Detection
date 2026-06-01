"""
最终冲刺90%：twitter-roberta-base 更精细调参
- 更小batch size + gradient accumulation
- 更多epoch with cosine scheduler
- 数据增强：随机删词
"""
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))
SAVE_DIR = os.environ.get("RUMOR_MODEL_DIR", str(PROJECT_ROOT / "best_model"))
DEVICE = torch.device('cuda:0')

# Fix seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(text):
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def augment_text(text, p=0.1):
    """随机删词增强"""
    words = text.split()
    if len(words) <= 3:
        return text
    new_words = [w for w in words if random.random() > p]
    return ' '.join(new_words) if new_words else text


class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = augment_text(text)
        enc = self.tokenizer(text, truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    inputs = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
    labels = torch.stack([b[1] for b in batch])
    return inputs, labels


def train_one(model_name, lr, batch_size, epochs, max_len, seed, accum_steps=1):
    set_seed(seed)
    print(f'\n=== {model_name} lr={lr} bs={batch_size} seed={seed} ===')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    train_texts = [clean_text(t) for t in train_df['text']]
    val_texts = [clean_text(t) for t in val_df['text']]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    train_set = RumorDataset(train_texts, train_df['label'].tolist(), tokenizer, max_len, augment=True)
    val_set = RumorDataset(val_texts, val_df['label'].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc = 0
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                logits = model(**inputs).logits
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            patience = 0
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f'  -> Best! ({acc:.4f})')
        else:
            patience += 1
            if patience >= 5:
                print('Early stopping')
                break

    print(f'Best: {best_acc:.4f}')
    return best_acc


if __name__ == '__main__':
    results = {}
    # Run multiple seeds and configs
    configs = [
        ('cardiffnlp/twitter-roberta-base', 2e-5, 8, 20, 160, 42, 2),
        ('cardiffnlp/twitter-roberta-base', 1.5e-5, 16, 20, 160, 42, 1),
        ('cardiffnlp/twitter-roberta-base', 2e-5, 16, 20, 160, 123, 1),
        ('cardiffnlp/twitter-roberta-base', 2e-5, 16, 20, 160, 456, 1),
    ]
    for model_name, lr, bs, ep, ml, seed, accum in configs:
        key = f'{model_name}_lr{lr}_bs{bs}_seed{seed}'
        results[key] = train_one(model_name, lr, bs, ep, ml, seed, accum)

    print('\n=== Summary ===')
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f'{k}: {v:.4f}')
