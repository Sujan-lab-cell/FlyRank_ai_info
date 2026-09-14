# Google Search Ranking & Discoverability Capstone

## ML-CAP-01 — Refresh / Content Opportunity Scoring

This capstone was completed as part of the **FlyRank Machine Learning Internship**.

The project focuses on using historical search-performance signals to identify webpages that may experience meaningful search-performance decline and help SEO/content teams **prioritize pages for human review**.

The system is designed as a **decision-support workflow**, not an autonomous content optimization system.

---

## 1. Research Question

> **Can historical search-performance signals be used to identify webpages that are most likely to experience meaningful performance decline, so that SEO and content teams can prioritize refresh reviews?**

The practical goal is to answer:

> **Which webpages should an SEO/content team review first when many pages may be experiencing search-performance decline?**

---

## 2. Problem Framing

### Unit of Analysis

The unit of analysis is a **webpage/content item**.

### Decision

Prioritize webpages that may be experiencing meaningful search-performance decline.

### Model Output

The model produces a score that can be used to rank eligible webpages.

### Intended Users

- SEO analysts
- Content strategists
- Editors

### Intended Action

The output helps teams decide:

1. Which pages should be reviewed first?
2. Which supporting signals should be investigated?
3. Whether a refresh or other content action should be considered.

The model recommends **where to look**.

A human decides **what to do**.

---

## 3. Data

The project uses the **FlyRank ML Internship dataset**.

### Feature Window

**February–April 2026**

The feature cutoff is **April 30, 2026**.

### Outcome Window

**May 1–31, 2026**

The May outcome is used only to define the target and is not used as a predictive feature.

### Evaluation Population

After applying the eligibility criteria:

- **16,513 eligible webpages**
- **36 clients**

### Eligibility

```text
impressions_total >= 1000
AND
april_clicks >= 10
```

---

## 4. Target Definition

The target represents meaningful decline in May clicks relative to April.

```python
decline = (may_clicks < 0.8 * april_clicks).astype(int)
```

- `1` = meaningful click decline
- `0` = no meaningful decline according to the defined threshold

This is a predefined operational definition for the capstone, not a universal definition of search-performance decline.

---

## 5. Features

The final model uses **9 historical features**.

| Feature | Description |
|---|---|
| `impressions_total` | Total impressions during February–April |
| `clicks_total` | Total clicks during February–April |
| `april_impressions` | April impressions |
| `april_clicks` | April clicks |
| `feb_clicks` | February clicks |
| `momentum` | April clicks relative to February clicks |
| `ctr` | Click-through rate |
| `active_days` | Number of days with impressions |
| `weighted_position` | Impression-weighted search position |

### Feature Definitions

```text
momentum = april_clicks / (feb_clicks + 1.0)
ctr = clicks_total / impressions_total × 100
```

`active_days` represents the number of days with impressions.

`weighted_position` is based on impression-weighted search position. A search position of `0` is treated as missing.

---

## 6. Excluded Information

The following were intentionally excluded from predictive features:

- May performance values
- Target-derived fields
- `trend_direction`
- `trend_pct`
- Future-period information
- Client identifiers as predictive features
- Content identifiers as predictive features
- Sparse session signals
- Scroll signals
- AI referral signals

Identifiers were used only for grouping, joining, validation, and deterministic tie-breaking.

---

## 7. Baseline

A transparent rule-based baseline was created.

A page is flagged when:

```text
april_clicks < march_clicks
```

Flagged pages are ranked by:

```text
april_impressions
```

| Metric | Baseline |
|---|---:|
| Precision@10 | 0.400 |
| Precision@20 | 0.370 |
| Precision@50 | **0.392** |
| Precision@100 | 0.388 |

---

## 8. Models Compared

Eight model options were evaluated:

1. Baseline
2. Logistic Regression
3. Decision Tree
4. HistGradientBoosting
5. Random Forest
6. LightGBM
7. CatBoost
8. XGBoost

No hyperparameter search was performed for the additional boosting models. The purpose was a controlled **model-family comparison**.

---

## 9. Validation Methodology

A **5-fold GroupKFold** validation strategy was used, grouped by client.

This prevents pages from the same client appearing in both training and validation within a fold.

Confirmed client overlap per fold:

```text
[0, 0, 0, 0, 0]
```

The same evaluation population, folds, features, target, ranking policy, and tie-breaking rule were used across the model comparison.

---

## 10. Evaluation Metric

The primary metric is **Precision@50 (P@50)**.

```text
Precision@50 =
Number of actual decline pages in top 50
/
50
```

Additional metrics:

- Precision@10
- Precision@20
- Precision@100

---

## 11. Ranking Policy

Predictions are ranked using:

1. `model_score` descending
2. `april_clicks` descending
3. `content_hash_id` ascending

This deterministic tie-breaking makes rankings reproducible.

---

## 12. Model Results

| Model | P@10 | P@20 | P@50 | P@100 |
|---|---:|---:|---:|---:|
| Baseline | 0.400 | 0.370 | **0.392** | 0.388 |
| Logistic Regression | 0.380 | 0.400 | **0.424** | 0.438 |
| Decision Tree | 0.400 | 0.390 | **0.324** | 0.346 |
| HistGradientBoosting | 0.440 | 0.420 | **0.436** | 0.442 |
| LightGBM | 0.450 | 0.425 | **0.438** | 0.444 |
| CatBoost | 0.450 | 0.430 | **0.440** | 0.445 |
| XGBoost | 0.450 | 0.430 | **0.442** | 0.446 |
| **Random Forest** | **0.460** | **0.430** | **0.444** | **0.448** |

---

## 13. Final Model

### Random Forest

Random Forest achieved the highest Precision@50:

```text
Random Forest P@50 = 0.444
Baseline P@50      = 0.392
```

Improvement:

```text
0.444 - 0.392 = 0.052
```

Therefore, Random Forest improved Precision@50 by **5.2 percentage points** over the baseline.

Random Forest was retained as the final capstone model.

---

## 14. Random Forest Feature Importance

The final Random Forest importance ranking was calculated using:

```python
rf.feature_importances_
```

Average importance across the five training folds:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `momentum` | 0.2899 |
| 2 | `april_impressions` | 0.2203 |
| 3 | `impressions_total` | 0.1921 |
| 4 | `ctr` | 0.0634 |
| 5 | `feb_clicks` | 0.0633 |
| 6 | `weighted_position` | 0.0590 |
| 7 | `clicks_total` | 0.0540 |
| 8 | `april_clicks` | 0.0512 |
| 9 | `active_days` | 0.0067 |

These are **model-derived associations**, not causal explanations.

---

## 15. Action Playbook

The model output is converted into a practical review queue.

| Reason | Action | Priority |
|---|---|---|
| Model signal only | Manual investigation | Medium |
| Decline + low visibility | SEO/content review | High |
| No strong signal | Monitor | Monitor |

Final queue:

- **723** model-signal-only pages
- **249** decline-and-low-visibility pages
- **15,541** monitor pages

The model does not automatically decide the final content action.

---

## 16. Human Review Workflow

```text
Model ranking
      ↓
Human review
      ↓
Check evidence
      ↓
Choose action
      ↓
Human approval
      ↓
Manual execution
```

Reviewers should consider:

1. Content quality
2. Content relevance
3. Search performance
4. Search visibility
5. Business importance
6. Evidence supporting a refresh
7. Whether the observed decline is meaningful and persistent

---

## 17. What Should NOT Be Automated

This system should not automatically:

- edit content
- publish content
- delete pages
- redirect pages
- modify canonical tags
- change titles
- change meta descriptions
- change H1 headings
- modify internal links
- claim a root cause
- predict Google's ranking algorithm
- guarantee traffic recovery
- assume a high model score requires immediate action

> **The model recommends where to look. A human decides what to do.**

---

## 18. Limitations

### Observational Data

The analysis uses historical observational data. Results show measured associations and model performance, not causal relationships.

### Generalization

Client-grouped validation reduces client-overlap leakage, but further validation on unseen clients and future time periods would strengthen confidence.

### Target Definition

The decline target is based on:

```text
May clicks < 80% of April clicks
```

Different thresholds may produce different results.

### Model Performance

Random Forest achieved **P@50 = 0.444**. This is a benchmark for this evaluation setup, not an SLA or guarantee.

### Feature Importance

Random Forest feature importance is model-derived and should not be interpreted as causal evidence.

### Human Review

The model cannot independently determine why a page declined or which content change will recover performance.

---

## 19. Reproducibility

Main notebook:

```text
work/notebooks/capstone.ipynb
```

Supporting artifacts:

```text
work/
├── notebooks/
├── figures/
└── outputs/
```

The repository contains the methodology, model evaluation, figures, results, recommendations, and reproducibility information needed to understand the workflow.

---

## 20. Public-Safety Rules

The public project intentionally avoids exposing:

- client names
- client domains
- private URLs
- private search queries
- credentials
- raw data exports
- sensitive client information

---

## 21. Key Finding

> **Under 5-fold client-grouped validation, Random Forest achieved a Precision@50 of 0.444 for the defined May 2026 decline target, compared with 0.392 for the transparent baseline.**

This provides **directional evidence** that historical search-performance signals can help prioritize potentially declining pages for human review.

It does not establish causation, guarantee future performance, or predict Google's ranking algorithm.

---

## 22. Final Recommendation

```text
Historical search signals
          ↓
Random Forest scoring
          ↓
Rank webpages
          ↓
Prioritize top candidates
          ↓
Human investigation
          ↓
Evidence-based action
```

Random Forest is retained because it achieved the best measured Precision@50 among the tested model families.

The system should be used as a **human-in-the-loop prioritization tool**, rather than an autonomous content optimization system.

---

## 23. Deployed Research Paper

https://sujan-lab-cell.github.io/flyrank-ml-internship/

---

## 24. GitHub Repository

https://github.com/Sujan-lab-cell/flyrank-ml-internship

Final capstone commit:

```text
377c48d
Complete ML-CAP-01 capstone model comparison and paper
```

---

## 25. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**.

Data credit: **FlyRank AI**

https://flyrank.ai

---

## Final Summary

This capstone developed and evaluated a machine-learning workflow for prioritizing webpages that may experience meaningful search-performance decline.

Eight model families were compared under the same client-grouped validation methodology. Random Forest achieved the highest **Precision@50 of 0.444**, compared with **0.392 for the baseline**, representing a **5.2 percentage-point improvement**.

The resulting system is intended to support SEO and content teams by identifying where human investigation should begin, while keeping final content decisions under human control.
