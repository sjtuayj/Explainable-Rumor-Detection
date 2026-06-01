"""
谣言检测分类器训练脚本
方案：BiGRU + 逻辑回归，取最优
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))

def train_lr():
    """训练逻辑回归模型"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    X_train = train_df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    X_val = val_df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    y_train = train_df['label']
    y_val = val_df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_pred)
    print(f'Logistic Regression Val Acc: {val_acc:.4f}')
    print(classification_report(y_val, val_pred))

    save_path = str(PROJECT_ROOT / 'lr_model.pkl')
    joblib.dump({'model': model, 'vectorizer': vectorizer}, save_path)
    print(f'模型已保存为 {save_path}')
    return val_acc

if __name__ == '__main__':
    train_lr()
