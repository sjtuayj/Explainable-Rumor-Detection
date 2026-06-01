"""
优化分类器：尝试多种模型和特征组合，目标90%
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))


def preprocess(text):
    text = text.lower()
    # 保留hashtags作为特征
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' MENTION ', text)
    text = re.sub(r'#(\w+)', r'HASHTAG_\1', text)
    return text


def train_and_eval():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    X_train = train_df['text'].apply(preprocess)
    X_val = val_df['text'].apply(preprocess)
    y_train = train_df['label']
    y_val = val_df['label']

    # TF-IDF with bigrams and more features
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=30000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    models = {
        'LR': LogisticRegression(max_iter=2000, C=1.0),
        'LR_C10': LogisticRegression(max_iter=2000, C=10.0),
        'LR_C01': LogisticRegression(max_iter=2000, C=0.1),
        'SVM': LinearSVC(max_iter=5000, C=1.0),
        'SVM_C10': LinearSVC(max_iter=5000, C=10.0),
    }

    best_acc = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_val_vec)
        acc = accuracy_score(y_val, pred)
        print(f'{name}: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print(f'\nBest: {best_name} = {best_acc:.4f}')
    print(classification_report(y_val, best_model.predict(X_val_vec)))

    # Save best
    save_path = str(PROJECT_ROOT / 'lr_model.pkl')
    joblib.dump({'model': best_model, 'vectorizer': vectorizer}, save_path)
    print(f'Best model saved to {save_path}')
    return best_acc


if __name__ == '__main__':
    train_and_eval()
