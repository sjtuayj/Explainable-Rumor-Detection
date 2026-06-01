"""
DeBERTa-v3-base微调 + 多种优化策略，目标90%
- 更长max_len (推文可能有url等)
- label smoothing
- 多轮学习率搜索
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))
SAVE_DIR = os.environ.get("RUMOR_MODEL_DIR", str(PROJECT_ROOT / "best_model"))
BATCH_SIZE = 16
EPOCHS = 15
MAX_LEN = 160
DEVICE = torch.device('cuda:0')


def clean_text(text):
    """清洗推文"""
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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


def run_experiment(model_name, lr):
    print(f'\n=== {model_name} lr={lr} ===')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    train_texts = [clean_text(t) for t in train_df['text']]
    val_texts = [clean_text(t) for t in val_df['text']]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    train_set = RumorDataset(train_texts, train_df['label'].tolist(), tokenizer, MAX_LEN)
    val_set = RumorDataset(val_texts, val_df['label'].tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)

    # Label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
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
                logits = model(**inputs).logits
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            patience = 0
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f'  -> Best! ({acc:.4f})')
        else:
            patience += 1
            if patience >= 4:
                print('Early stopping')
                break

    print(f'Best: {best_acc:.4f}')
    return best_acc


if __name__ == '__main__':
    results = {}
    for model_name in ['cardiffnlp/twitter-roberta-base', 'microsoft/deberta-base']:
        for lr in [1e-5, 2e-5]:
            acc = run_experiment(model_name, lr)
            results[f'{model_name}_lr{lr}'] = acc

    print('\n=== Summary ===')
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f'{k}: {v:.4f}')
