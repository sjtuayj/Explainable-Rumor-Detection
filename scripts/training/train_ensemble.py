"""
Ensemble: 训练3个不同seed的roberta-large，保存各自checkpoint
然后用投票集成
"""
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))
BASE_SAVE_DIR = os.environ.get("RUMOR_ENSEMBLE_DIR", str(PROJECT_ROOT / "ensemble_models"))
DEVICE = torch.device('cuda:0')

def set_seed(seed):
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

class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                   max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

def collate_fn(batch):
    inputs = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
    labels = torch.stack([b[1] for b in batch])
    return inputs, labels

def train_one(model_name, lr, seed, save_dir, epochs=15):
    set_seed(seed)
    print(f'\n=== seed={seed} ===')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    train_texts = [clean_text(t) for t in train_df['text']]
    val_texts = [clean_text(t) for t in val_df['text']]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    train_set = RumorDataset(train_texts, train_df['label'].tolist(), tokenizer)
    val_set = RumorDataset(val_texts, val_df['label'].tolist(), tokenizer)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=16, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps),
                                                 num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc = 0
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)
            loss = criterion(model(**inputs).logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                all_preds.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            patience = 0
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            patience += 1
            if patience >= 5:
                print('Early stopping')
                break
    print(f'Best: {best_acc:.4f}')
    return best_acc

def ensemble_eval():
    """加载所有checkpoint，投票集成"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    val_texts = [clean_text(t) for t in val_df['text']]
    y_val = val_df['label'].tolist()

    model_dirs = [os.path.join(BASE_SAVE_DIR, d) for d in sorted(os.listdir(BASE_SAVE_DIR))
                  if os.path.isdir(os.path.join(BASE_SAVE_DIR, d))]

    all_logits = []
    for mdir in model_dirs:
        print(f'Loading {mdir}...')
        tokenizer = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to(DEVICE)
        model.eval()
        inputs = tokenizer(val_texts, truncation=True, padding='max_length',
                           max_length=128, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu()
        all_logits.append(logits)
        del model
        torch.cuda.empty_cache()

    # Average logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = avg_logits.argmax(dim=-1).tolist()
    acc = accuracy_score(y_val, preds)
    print(f'\nEnsemble Accuracy: {acc:.4f}')
    print(classification_report(y_val, preds))

    # Also save best_model as the ensemble result
    # Copy best single model for inference
    return acc

if __name__ == '__main__':
    model_name = 'roberta-large'
    seeds = [42, 123, 456]
    for seed in seeds:
        save_dir = os.path.join(BASE_SAVE_DIR, f'seed_{seed}')
        train_one(model_name, lr=1e-5, seed=seed, save_dir=save_dir)

    print('\n=== Ensemble Evaluation ===')
    ensemble_eval()
