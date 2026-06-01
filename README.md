# 可解释的谣言检测系统

基于 RoBERTa-Large 微调 + 上海交大 OpenAI 兼容大模型 API 的谣言检测系统，在验证集上达到 **90.27%** 的分类准确率。

## 项目结构

```
├── src/rumor_detection/      # 核心 Python 包：检测器和 SJTU API 客户端
│   ├── detect.py
│   └── llm_client.py
├── scripts/
│   ├── evaluation/           # 评估、LLM 分类和 hybrid 复现实验
│   ├── training/             # 训练脚本和 baseline 脚本
│   └── tools/                # checkpoint 恢复等工具
├── rumer2026/                # 数据集
│   ├── train.csv
│   └── val.csv
├── best_model/               # 最佳 RoBERTa-Large checkpoint 与 tokenizer
│   ├── model.safetensors     # 恢复后的完整权重文件
│   └── model.safetensors.part-*  # 用于仓库分发的权重分片
├── eval_outputs/             # 评估输出和 hybrid 复核缓存
├── report_assets/            # 报告图片
├── README.md                 # 项目说明、部署和运行命令
├── report.pdf                 # 大作业报告 pdf 版
└── requirements.txt          # Python 依赖
```

所有代码文件都归档在 `src/` 和 `scripts/` 下，根目录只保留文档、配置、数据和模型目录。

## 环境配置

```bash
# Python 3.10+
python -m pip install -r requirements.txt
```

## 模型下载

训练好的最佳 checkpoint 已随仓库保存在 `best_model/`。为了避开 GitHub
单文件大小限制，`model.safetensors` 以分片形式提交。clone 后先运行：

```bash
python scripts/tools/restore_best_model.py
```

仓库内分发的是 fp16 版本（约 678MB），由 seed=42、lr=1e-5 的
RoBERTa-Large checkpoint 转换而来，验证集 supervised accuracy 保持
**90.02%**。结合 `eval_outputs/final_best_hybrid/llm_review_cache.jsonl`
中的低置信 API 复核缓存，可复现 **90.27%** 的最终结果。

也可以重新训练：

```bash
python scripts/training/train_best_model.py --model-name roberta-large --data-dir rumer2026 --save-dir best_model --seed 42 --lr 1e-5
```

## 使用方法

### 1. 训练模型

```bash
# 在GPU服务器上运行
python scripts/training/train_ensemble.py
```

### 2. 单条检测

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from rumor_detection.detect import RumourDetector

detector = RumourDetector(model_dir="best_model")
result = detector.detect("BREAKING: Scientists confirm the moon is made of cheese")
print(f"分类: {result['label']}")       # 0=非谣言, 1=谣言
print(f"依据: {result['explanation']}")  # LLM生成的判断依据
```

### 3. 验证集评估

```bash
RUMOR_MODEL_DIR=best_model RUMOR_DATA_DIR=rumer2026 python scripts/evaluation/evaluate.py 5
```

复现最终 90.27% hybrid 结果：

```bash
python scripts/tools/restore_best_model.py
python scripts/evaluation/evaluate_hybrid.py --model-dir best_model --data-dir rumer2026 --output-dir eval_outputs/final_best_hybrid --use-llm --llm-model deepseek-reasoner --few-shot-per-class 4
```

该命令会优先复用 `eval_outputs/final_best_hybrid/llm_review_cache.jsonl`，
避免同一 API 在后续运行中因返回波动改变最终分数。

## 技术方案

| 组件 | 方法 | 说明 |
|------|------|------|
| 检测（分类） | RoBERTa-Large 微调 | 在推文数据上微调预训练模型 |
| 判断依据 | 上海交大 OpenAI 兼容大模型 API | 根据分类结果生成可解释的判断依据 |

实验记录和后续提升计划见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

### 模型演进

| 模型 | Val Accuracy |
|------|-------------|
| Logistic Regression | 82.29% |
| BERT-base | 85.54% |
| RoBERTa-base | 87.03% |
| twitter-roberta-base | 88.78% |
| RoBERTa-Large | 90.02% |
| **RoBERTa-Large + low-confidence LLM review** | **90.27%** |

### 关键优化

1. **文本预处理**：URL → `[URL]`，@用户 → `[USER]`，#话题 → `[HASHTAG] 话题`
2. **训练策略**：Cosine learning rate schedule + warmup + label smoothing + gradient clipping
3. **超参数**：lr=1e-5, batch_size=8, max_len=128, epochs=15 (early stopping)

## API 配置

判断依据生成需要上海交大大模型 API Key。推荐在项目根目录创建 `.env`：

```bash
SJTU_API_KEY=your-api-key
SJTU_MODEL=deepseek-chat
SJTU_BASE_URL=https://models.sjtu.edu.cn/api/v1
```

也可以直接通过环境变量设置：

```bash
export SJTU_API_KEY="your-api-key"
```

默认模型为 `deepseek-chat`。如需复杂推理，可以设置 `SJTU_MODEL=deepseek-reasoner`；纯 LLM 分类脚本也支持单独设置 `SJTU_CLASSIFY_MODEL`。请不要把真实 API key 写入代码或提交到 GitHub。

交大 API 当前限制为每分钟最多 10 次请求。纯 LLM 分类评估脚本默认按 10 RPM 限速：

```bash
python scripts/evaluation/test_llm_classify.py -n 50 --model deepseek-chat --rpm 10 --data-dir rumer2026
```

50 条样本约需 5 分钟，完整 401 条验证集约需 41 分钟。可对比的模型调用名包括 `deepseek-chat`、`deepseek-reasoner`、`minimax`、`glm`、`qwen`。

若测试纯 LLM 分类，建议使用 few-shot 示例校准标签定义：

```bash
python scripts/evaluation/test_llm_classify.py -n 50 --model deepseek-reasoner --rpm 10 --data-dir rumer2026 --few-shot-per-class 4
```
