"""
谣言检测模块：RoBERTa分类器 + SJTU OpenAI兼容API生成判断依据
"""
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from .llm_client import DEFAULT_LLM_MODEL, get_llm_client
except ImportError:
    import sys

    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rumor_detection.llm_client import DEFAULT_LLM_MODEL, get_llm_client

MODEL_DIR = os.environ.get("RUMOR_MODEL_DIR",
    str(PROJECT_ROOT / 'best_model'))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def clean_text(text):
    """与训练时一致的预处理"""
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RumourDetector:
    def __init__(self, model_dir=MODEL_DIR, llm_model=DEFAULT_LLM_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        self.model.eval()
        self.llm_model = llm_model
        self._llm_client = None

    def classify(self, text: str) -> int:
        """返回 0(非谣言) 或 1(谣言)"""
        cleaned = clean_text(text)
        inputs = self.tokenizer(cleaned, truncation=True, padding=True,
                                max_length=128, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return int(logits.argmax(dim=-1).item())

    def classify_batch(self, texts):
        """批量分类"""
        cleaned = [clean_text(t) for t in texts]
        inputs = self.tokenizer(cleaned, truncation=True, padding=True,
                                max_length=128, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).cpu().tolist()

    def explain(self, text: str, label: int) -> str:
        """用 SJTU OpenAI兼容API生成判断依据"""
        label_str = "谣言" if label == 1 else "非谣言"
        prompt = f"""你是一个谣言检测专家。以下推文被判定为{label_str}（label={label}）。
请用2-3句话简要解释为什么这条推文被判定为{label_str}。分析要点包括：信息来源可靠性、表述是否客观、是否有情绪化用语、是否可验证等。

推文内容："{text}"

判断依据："""
        try:
            if self._llm_client is None:
                self._llm_client = get_llm_client()
            resp = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful rumor detection expert."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[生成失败: {e}]"

    def detect(self, text: str) -> dict:
        """完整检测：返回 label 和 explanation"""
        label = self.classify(text)
        explanation = self.explain(text, label)
        return {"label": label, "explanation": explanation}


def main():
    detector = RumourDetector()
    test_text = "BREAKING: Scientists confirm the moon is made of cheese #fakenews"
    result = detector.detect(test_text)
    print(f"Text: {test_text}")
    print(f"Label: {result['label']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == '__main__':
    main()
