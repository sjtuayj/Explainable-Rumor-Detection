"""
进一步优化：加入手工特征 + 更强的模型组合
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import joblib
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get("RUMOR_DATA_DIR", str(PROJECT_ROOT / "rumer2026"))


def extract_manual_features(texts):
    """手工特征"""
    features = []
    for text in texts:
        f = []
        f.append(len(text))  # 长度
        f.append(text.count('!'))  # 感叹号
        f.append(text.count('?'))  # 问号
        f.append(text.count('#'))  # hashtag数
        f.append(text.count('@'))  # mention数
        f.append(1 if 'http' in text else 0)  # 含链接
        f.append(len(re.findall(r'[A-Z]{2,}', text)))  # 全大写词数
        f.append(text.count('"') + text.count("'"))  # 引号
        # 情绪词
        emo_words = ['breaking', 'confirmed', 'just', 'omg', 'wow', 'unbelievable',
                     'shocking', 'rumor', 'hoax', 'fake', 'false', 'true', 'real',
                     'official', 'report', 'source', 'alleged', 'claim']
        text_lower = text.lower()
        f.append(sum(1 for w in emo_words if w in text_lower))
        features.append(f)
    return np.array(features)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' MENTION ', text)
    text = re.sub(r'#(\w+)', r' HASHTAG_\1 \1 ', text)
    return text


def train():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    X_train_text = train_df['text'].apply(preprocess)
    X_val_text = val_df['text'].apply(preprocess)
    y_train = train_df['label']
    y_val = val_df['label']

    # TF-IDF: word + char ngrams
    word_vec = TfidfVectorizer(
        stop_words='english', max_features=30000,
        ngram_range=(1, 3), min_df=2, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        max_features=30000, min_df=2, sublinear_tf=True,
    )

    X_train_word = word_vec.fit_transform(X_train_text)
    X_val_word = word_vec.transform(X_val_text)
    X_train_char = char_vec.fit_transform(X_train_text)
    X_val_char = char_vec.transform(X_val_text)

    # 手工特征
    X_train_manual = extract_manual_features(train_df['text'])
    X_val_manual = extract_manual_features(val_df['text'])

    # 合并
    X_train_all = hstack([X_train_word, X_train_char, X_train_manual])
    X_val_all = hstack([X_val_word, X_val_char, X_val_manual])

    # 尝试多个C值
    for C in [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]:
        model = LogisticRegression(max_iter=3000, C=C)
        model.fit(X_train_all, y_train)
        acc = accuracy_score(y_val, model.predict(X_val_all))
        print(f'LR C={C}: {acc:.4f}')

    for C in [0.5, 1.0, 5.0, 10.0]:
        model = LinearSVC(max_iter=5000, C=C)
        model.fit(X_train_all, y_train)
        acc = accuracy_score(y_val, model.predict(X_val_all))
        print(f'SVM C={C}: {acc:.4f}')

    # Train best and save
    best_model = LogisticRegression(max_iter=3000, C=10.0)
    best_model.fit(X_train_all, y_train)
    best_acc = accuracy_score(y_val, best_model.predict(X_val_all))
    print(f'\nFinal: {best_acc:.4f}')
    print(classification_report(y_val, best_model.predict(X_val_all)))

    save_path = str(PROJECT_ROOT / 'lr_model.pkl')
    joblib.dump({
        'model': best_model,
        'word_vec': word_vec,
        'char_vec': char_vec,
    }, save_path)
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    train()
