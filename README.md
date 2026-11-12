# Kaggle Credit Card Fraud Detection — Reproducible OOF Ensemble

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Public%20LB%200.95298-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e11)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)

An end-to-end, reproducible solution for Kaggle's **Playground Series S6E11 — Predicting Credit Card Fraud**. The project combines same-fold out-of-fold (OOF) predictions from a diverse model bank with a second target-first language model, then produces a deterministic rank ensemble.

> Verified Kaggle submission: **0.95298 ROC AUC** (submission `#58229122`), improving the previous version from **0.95222** by **+0.00076**.

## Highlights

- Audited a **201-member** public OOF stack spanning 12 model sources.
- Added a diverse target-first language model; its rank correlation with the existing LM view was about 0.98, providing complementary signal.
- Used strict same-fold predictions for model evaluation and fixed blend weights before submission.
- Recomputed the wide-stack OOF AUC locally: **0.952565**.
- Rebuilt the scored submission byte-for-byte (`SHA256 f449932b...fcdf`).
- Persisted configuration, metrics, hashes, environment versions, and the official Kaggle receipt.

## Method

```text
Competition data + CC0 public OOF assets
                  │
          SHA256 and schema checks
                  │
     201-member same-fold wide stack
       (trees, GLMs, MLPs, GPT-style LM)
                  │
        0.7 × wide stack + 0.3 × Part 9
                  │
      deterministic rank normalization
                  │
           Kaggle submission.csv
```

The final prediction is:

```python
rank(0.7 * rank(wide_lm_stack_test) + 0.3 * rank(part9_test))
```

The 0.7/0.3 weights are inherited from the preceding OOF-backed stack and were not retuned on this repository's submission. The final blend's source notebook reports nested OOF AUC **0.952527**; the directly available 201-member wide-stack OOF array scores **0.952565** when recomputed locally. This distinction is retained in the artifacts instead of treating a public-leaderboard score as validation.

## Results

| Version | Validation evidence | Public LB |
|---|---:|---:|
| Previous public rank blend | Public solution blend | 0.95222 |
| 201-member wide stack + second LM | Wide-stack OOF 0.952565; final nested OOF 0.952527 | **0.95298** |

Metric: ROC AUC. The competition's public leaderboard is a subset of the hidden test labels, so small score differences should be interpreted with caution.

## Repository structure

```text
.
├── configs/v2_part12.json            # immutable run configuration and source hashes
├── src/build_v2_part12.py             # validation, deterministic blend, artifacts
├── scripts/download_assets.sh         # Kaggle data and CC0 artifact downloader
├── artifacts/v2_part12/
│   ├── metrics.json                   # local OOF metrics and checks
│   └── manifest.json                  # file hashes and environment versions
├── logs/submission_receipt.json       # official Kaggle score and submission id
└── requirements.txt
```

Large competition files, prediction arrays, and generated submissions are intentionally excluded from Git.

## Reproduce

Prerequisites:

- Python 3.11+
- A Kaggle account that has accepted the [competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules)
- Kaggle authentication configured with `kaggle auth login` or an access token

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_assets.sh
python src/build_v2_part12.py configs/v2_part12.json
```

The generated file is:

```text
submissions/s6e11_v2_part12_20261111.csv
```

Optional submission:

```bash
kaggle competitions submit playground-series-s6e11 \
  -f submissions/s6e11_v2_part12_20261111.csv \
  -m "Reproduced OOF wide-stack + second LM rank ensemble"
```

## Reproducibility and leakage controls

- All supervised stack inputs are out-of-fold predictions on the same five stratified folds.
- The build fails on input hash, row count, ID order, non-finite value, or final submission hash mismatch.
- Test-only public prediction files are not used to estimate local blend weights.
- Source licenses and notebook versions are recorded in the run configuration.
- Raw Kaggle data, credentials, tokens, and local environment files are excluded from Git.

## Credits and data licenses

The ensemble builds on public Kaggle work, especially:

- [S6E11 Part 12 — Wide Stack + 2nd LM](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298)
- [S6E11 Fraud Series Artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts)
- [S6E11 Target-First LM Predictions](https://www.kaggle.com/datasets/megayak/s6e11-target-first-lm)

The two prediction datasets are published under **CC0-1.0**. Competition data remains subject to Kaggle's competition rules. Code in this repository is released under the MIT License.

## 中文简介

本项目围绕信用卡欺诈二分类任务，完成了公开高分方案调研、同折 OOF 证据审计、201 模型宽栈融合、语言模型多样性引入、可复现提交构建与线上评分闭环。最终 Kaggle 公榜 ROC AUC 为 **0.95298**，相较上一版提升 **0.00076**；仓库保留配置、哈希、OOF 指标和官方提交回执，可作为数据科学求职项目展示。
