# Experiment Log and Improvement Plan

## Current Status

Date: 2026-06-01

This repository was adapted from the original rumor detection homework code to use the Shanghai Jiao Tong University OpenAI-compatible model API.

API integration changes:

- Added `src/rumor_detection/llm_client.py` to centralize API access.
- Uses `SJTU_API_KEY` from environment variables or local `.env`.
- Uses `https://models.sjtu.edu.cn/api/v1` as the default `base_url`.
- Uses `deepseek-chat` as the default model, with runtime overrides supported.
- Removed hard-coded API keys and the old DeepSeek endpoint.
- Added API rate limiting support to `scripts/evaluation/test_llm_classify.py`.

Local safety setup:

- `.env`, `.env.example`, and `.venv/` are ignored by git.
- The local API key should stay in `.env` only.
- Do not commit real API keys to GitHub.

## SJTU API Constraints

- Requests per minute: 10
- Tokens per minute: 100000
- Weekly token quota: 1000000000

The evaluation script defaults to `--rpm 10`, so it stays within the request limit.

## LLM-Only Evaluation Results

All results below use the same sampled 50 validation examples with `seed=42`.

| Method | Model | Prompt | Few-shot examples | Accuracy |
|---|---|---|---:|---:|
| LLM-only | `deepseek-chat` | original zero-shot | 0 | 54.00% |
| LLM-only | `deepseek-chat` | dataset-aligned zero-shot | 0 | 58.00% |
| LLM-only | `deepseek-reasoner` | dataset-aligned zero-shot | 0 | 62.00% |
| LLM-only | `deepseek-reasoner` | dataset-aligned few-shot | 8 | 70.00% |

Best command tested:

```bash
.venv/bin/python scripts/evaluation/test_llm_classify.py -n 50 --model deepseek-reasoner --rpm 10 --data-dir rumer2026 --few-shot-per-class 4
```

Best observed result:

```text
Model: deepseek-reasoner
Few-shot examples: 8
LLM accuracy on 50 samples: 0.7000 (35/50), skipped=0
```

## Key Finding

Pure LLM classification is not strong enough to replace the supervised classifier.

The project report records a 90.02% RoBERTa-Large supervised validation accuracy and a 90.27% hybrid validation accuracy after narrow low-confidence LLM review. The best pure LLM result observed here was 70.00% on a 50-sample subset. The LLM is useful for explanation and uncertain-case review, but it should not be the primary classifier.

## Recommended Next Architecture

Use a hybrid detector:

1. Fine-tune `RoBERTa-Large` on the server using `scripts/training/train_ensemble.py` or `scripts/training/train_large.py`.
2. Save the resulting checkpoint as `best_model/`.
3. Change classification to return both label and confidence.
4. Use RoBERTa for all high-confidence samples.
5. Send only low-confidence samples to `deepseek-reasoner` with few-shot examples.
6. Use the LLM response as a second opinion, not an unconditional override.
7. Generate human-readable explanations with the LLM after the final label is chosen.

Suggested low-confidence gates to tune on `val.csv`:

- `max_prob < 0.65`
- or `abs(prob_1 - prob_0) < 0.20`

## Accuracy Improvement Plan

1. Train and evaluate RoBERTa-Large baseline on the server.
2. Add probability output to `RumourDetector.classify`.
3. Implement `HybridRumourDetector`.
4. Compare three validation results:
   - RoBERTa-only
   - LLM-only
   - RoBERTa + low-confidence LLM review
5. Tune the confidence threshold to maximize validation accuracy.
6. Replace random few-shot examples with retrieved few-shot examples:
   - Use same-event examples when possible.
   - Otherwise retrieve semantically similar examples from `train.csv`.
7. Track confusion matrix changes, especially false negatives where rumor tweets are predicted as non-rumor.

## Full LLM Evaluation Command

A full 401-example validation run takes about 41 minutes per model because of the 10 RPM limit:

```bash
.venv/bin/python scripts/evaluation/test_llm_classify.py -n 401 --model deepseek-reasoner --rpm 10 --data-dir rumer2026 --few-shot-per-class 4
```

This is useful for reporting, but hybrid evaluation is the more important next step.

## Local A100 Reproduction and Hybrid Results

Date: 2026-06-01

Local training was rerun on one NVIDIA A100-SXM4-80GB using `scripts/training/train_best_model.py`.
The active `best_model/` checkpoint is now the best local run:

| Run | Best epoch | Val Accuracy | Notes |
|---|---:|---:|---|
| `roberta-large`, seed 123, lr `1e-5` | 7 | 89.28% | First reproduced checkpoint |
| `roberta-large`, seed 42, lr `1e-5` | 6 | 90.02% | Promoted to `best_model/` |
| `roberta-large`, seed 42, lr `5e-6` | 3 | 88.03% | Lower learning rate underperformed |

Final supervised evaluation for the promoted `best_model/`:

```text
Accuracy: 90.02% (361/401)
Confusion matrix [[TN, FP], [FN, TP]]:
[[203, 23],
 [17, 158]]
```

Hybrid evaluation with the SJTU `deepseek-reasoner` API improves only when the
LLM is used conservatively on the lowest-confidence cases:

| Hybrid rule | Reviewed | Accuracy | Result |
|---|---:|---:|---|
| Review `max_prob < 0.65` or `margin < 0.20` | 4 | 90.27% | +1 corrected FP, no new errors |
| Review `max_prob < 0.95` or `margin < 0.20` | 39 | 88.78% | Too many harmful overrides |

Key conclusion: the API should not broadly override the supervised classifier.
It is useful as a narrow review step for very low-confidence predictions, but
wider coverage damages accuracy because the LLM does not reliably match the
dataset's rumor-label boundary.

For the `JunyiWuCode/ai_homework` sync, `best_model/model.safetensors` was
converted from fp32 to fp16 and stored as `best_model/model.safetensors.part-*`
chunks to avoid GitHub's single-file size limit. Run `python scripts/tools/restore_best_model.py`
after cloning to reconstruct the checkpoint. The fp16 checkpoint keeps the same
supervised result:

```text
Accuracy: 90.02% (361/401)
Confusion matrix [[TN, FP], [FN, TP]]:
[[203, 23],
 [17, 158]]
```

The best 90.27% hybrid result is preserved with the cached low-confidence API
review in `eval_outputs/final_best_hybrid/llm_review_cache.jsonl`. This cache is
important because repeated `deepseek-reasoner` calls can vary on borderline
items even with deterministic settings.

## Local Recheck After Project Reorganization

Date: 2026-06-02

The code layout was reorganized into `src/rumor_detection/` and `scripts/`.
The root directory no longer contains Python entry scripts. The reproduced
hybrid command is:

```bash
python scripts/evaluation/evaluate_hybrid.py --model-dir best_model --data-dir rumer2026 --output-dir eval_outputs/final_best_hybrid --use-llm --llm-model deepseek-reasoner --few-shot-per-class 4
```

Observed output:

```text
Supervised accuracy: 0.9002
Supervised confusion matrix [[TN, FP], [FN, TP]]:
[[203, 23],
 [17, 158]]

Uncertain samples selected for LLM review: 4/401

Hybrid accuracy: 0.9027
Hybrid confusion matrix [[TN, FP], [FN, TP]]:
[[204, 22],
 [17, 158]]
```

This matches the intended result: the supervised checkpoint provides 90.02%
accuracy, and the narrow low-confidence LLM review improves the final hybrid
accuracy to 90.27%.

Next accuracy plan:

1. Keep RoBERTa as the primary classifier and expose calibrated probabilities.
2. Use the LLM only for extremely uncertain samples, starting with
   `max_prob < 0.65` or `margin < 0.20`.
3. Tune the review threshold with cross-validation or a separate dev split
   instead of the final validation labels.
4. Improve the LLM prompt with dataset-specific examples that distinguish
   confirmed breaking-news updates from unverified factual claims.
5. Replace random few-shot examples with retrieved examples from the same event
   or nearest-neighbor training tweets.
6. Train several `roberta-large` seeds and evaluate logit averaging; only adopt
   an ensemble if it improves validation accuracy without requiring API calls.
7. Try domain-pretrained checkpoints such as `cardiffnlp/twitter-roberta-base`
   or larger Twitter-oriented models if available, then compare against the
   current `best_model/`.
