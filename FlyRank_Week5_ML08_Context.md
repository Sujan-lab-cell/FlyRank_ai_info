# FlyRank ML Internship — Week 5 (ML-08) Context

## Project goal
Build a supervised ML priority-scoring system for the content opportunity / refresh-review lane. The model ranks pages for human review; it does not automatically decide to refresh pages or guarantee SEO improvement.

## Workflow
1. You + ChatGPT: decide methodology, logic, interpretation, wording.
2. Codex/Antigravity: implement code.
3. You + ChatGPT: review and understand results.
Do not let the coding agent randomly choose methodology.

## Official Week 5 task
- Code: ML-08
- Week: 5
- Phase: Build
- Track: ML
- Workload: 6h
- Notebook: `work/notebooks/w05_model.ipynb`
- Deliverable: executed and committed notebook.
- Agent instruction: read `skills/README.md`, then load `training-honest-models` + `flyrank/flyrank-data`.

## Real warehouse data
Week 5 uses the real FlyRank warehouse, not synthetic data or the earlier starter CSV as a substitute.
- Daily period: Feb 1–May 31, 2026
- 39,308,592 daily rows
- 407,121 distinct content pages
- 70 clients
- Daily columns: `gsc_clicks`, `gsc_impressions`, `gsc_avg_position`
- After aggregation/eligibility: 16,513 pages, 36 clients
- Target 0: 9,640
- Target 1: 6,873
- Decline base rate: 41.62%
- Duplicate count: 0
- Final feature missingness: 0

The first eager warehouse processing approach caused Colab RAM problems. Final processing was optimized with Polars lazy scanning/streaming and only required columns.

## Target
Actual Week 5 target:
```python
decline = (may_clicks < 0.8 * april_clicks).astype(int)
```
- 1 = May clicks were at least 20% lower than April
- 0 = May clicks did not fall by 20% or more

Eligibility:
- `impressions_total >= 1000`
- `april_clicks >= 10`

The earlier Weeks 2–4 proxy:
```python
is_declining_label = (trend_direction == "down").astype(int)
```
is NOT the Week 5 final target.

## Nine pre-May features
1. `impressions_total`
2. `clicks_total`
3. `april_impressions`
4. `april_clicks`
5. `feb_clicks`
6. `momentum`
7. `ctr`
8. `active_days`
9. `weighted_position`

Only information available before May may be used as model input. May clicks/impressions create the target only.

Client/content IDs are grouping/joining fields, never predictive features. Sparse sessions/scroll/AI-referral signals were dropped.

## Validation
Use 5-fold `GroupKFold` by client.
- 5 folds
- 0 client overlap between train/validation

The final eligible data was explicitly sorted before GroupKFold:
```python
elig_df = elig_df.sort(["client_hash_id", "content_hash_id"])
```
This made fold assignment reproducible because streaming aggregation did not guarantee row order.

## Baseline
Frozen Week 5 baseline:
```python
baseline_flag = april_clicks < march_clicks
baseline_score = april_impressions if baseline_flag else 0
```
Ranking:
1. score descending
2. April clicks descending
3. `content_hash_id` ascending

The baseline is a simple rule-based reference, not the actual future outcome.

## Models
Compare:
- Baseline
- Logistic Regression — simple/interpretable linear model, with StandardScaler fit only on training folds
- Decision Tree — `max_depth=3`
- Random Forest — `n_estimators=100`, `max_depth=5`

Do not select a model because it is more complex. Select based on validation evidence.

## Evaluation
Primary metric: **Precision@50** because human review capacity is limited.
Also report Precision@10, @20, @100.

All models use the same data, folds, metrics, review depths, and deterministic tie-breaking.

## FINAL CANONICAL RESULTS

| Model | Precision@10 | Precision@20 | Precision@50 | Precision@100 |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 0.4000 | 0.3700 | 0.3920 | 0.3880 |
| Logistic Regression | 0.3800 | 0.3700 | 0.4240 | 0.4380 |
| Decision Tree | 0.4000 | 0.3900 | 0.3240 | 0.3460 |
| Random Forest | **0.4600** | **0.4300** | **0.4440** | **0.4480** |

Random Forest is best at the primary metric: **P@50 = 0.4440**.
Baseline: **0.3920**.
Improvement: **+5.2 percentage points**.
Practical interpretation: about 22 actual declining pages in the top 50 vs about 20/50 for the baseline.

This is a ranking improvement, not a guarantee of SEO performance.

## Fold-level Precision@50

| Fold | Baseline | Logistic Regression | Decision Tree | Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| Fold 0 | 0.3200 | 0.3800 | 0.3600 | 0.4400 |
| Fold 1 | 0.3800 | 0.3800 | 0.3600 | 0.3200 |
| Fold 2 | 0.4200 | 0.4200 | 0.3600 | 0.4600 |
| Fold 3 | 0.4200 | 0.5600 | 0.4800 | 0.6400 |
| Fold 4 | 0.4400 | 0.3600 | 0.4200 | 0.3600 |
| Mean | **0.3920** | **0.4240** | **0.3240** | **0.4440** |

Random Forest did not win every fold; it had the highest overall mean.

## Logistic Regression supplementary coefficients

| Feature | Mean Standardized Coef |
| :--- | :---: |
| `momentum` | -0.6698 |
| `april_clicks` | -0.3214 |
| `ctr` | +0.2105 |
| `april_impressions` | +0.1852 |
| `weighted_position` | +0.1428 |
| `impressions_total` | +0.1120 |
| `clicks_total` | -0.0984 |
| `feb_clicks` | -0.0762 |
| `active_days` | -0.0410 |

Positive = increases predicted decline probability; negative = decreases it, holding other standardized features constant. These coefficients describe Logistic Regression only and are not causal evidence.

## Section 4: OOF error analysis
Use out-of-fold Random Forest predictions because RF was the best model. Ranking/Top-K is primary; a 0.5 probability threshold is supplementary.

Threshold confusion counts:
- TN: 8,972 (54.33%)
- TP: 4,217 (25.54%)
- FN: 2,656 (16.08%)
- FP: 668 (4.05%)
- Total: 16,513

RF Top-50 fold results:
- Fold 0: TP 22, FP 28, P@50 .44; baseline TP 16, FP 34, .32
- Fold 1: TP 16, FP 34, .32; baseline TP 19, FP 31, .38
- Fold 2: TP 23, FP 27, .46; baseline TP 21, FP 29, .42
- Fold 3: TP 32, FP 18, .64; baseline TP 21, FP 29, .42
- Fold 4: TP 18, FP 32, .36; baseline TP 21, FP 29, .42
- RF mean TP 22.2, FP 27.8, P@50 .444
- Baseline mean TP 19.6, FP 30.4, P@50 .392

Observed error patterns:
- FP: associated with lower pre-May momentum and poorer search positions; some were ranked high-risk but did not decline.
- FN: more commonly associated with higher pre-May momentum and stronger traffic; some historically healthy pages still declined.
These are observed associations, not causal explanations.

## Error-group median profile

| Group | momentum | april_clicks | april_impressions | impressions_total | clicks_total | ctr | active_days | weighted_position |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.985 | 42 | 1840 | 5420 | 124 | 2.24% | 88 | 14.82 |
| TP | 0.621 | 28 | 1420 | 4210 | 82 | 1.95% | 82 | 18.45 |
| FP | 0.684 | 24 | 1680 | 4890 | 76 | 1.55% | 81 | 19.12 |
| TN | 1.142 | 54 | 2120 | 6150 | 158 | 2.57% | 89 | 12.65 |
| FN | 1.185 | 58 | 2250 | 6480 | 164 | 2.53% | 89 | 13.10 |

## Business interpretation
- FP = human reviewers may spend time checking pages that ultimately did not decline.
- FN = genuinely declining pages may be missed.
- Goal is better prioritization when review capacity is limited, not perfect prediction.

## Future improvements
Potential leakage-safe signals if available:
- content freshness
- search intent
- pre-May position volatility

These are ideas, not guaranteed improvements.

Human review remains necessary for content quality, search intent, and whether a refresh is appropriate.

## Scientific honesty rules
Prefer: observed, measured, associated with, suggests, descriptive, directional, decision-support, prioritization.

Avoid unsupported causal claims such as:
- old content causes traffic decline
- model predicts Google's proprietary algorithm
- refreshing guarantees traffic recovery
- a feature causes decline
- model guarantees SEO improvement

The project predicts the defined May decline target using historical signals. It does not predict Google's proprietary ranking algorithm.

## Rejected approaches
1. Mapping starter CSV fields into Week 5 months was rejected.
2. Synthetic daily data generated with Poisson sampling was rejected because results were inconsistent.
3. `trend_direction` as the Week 5 target was rejected; it is only the earlier starter-data proxy.
4. Future May clicks/impressions as features are forbidden because they cause leakage.

## Final notebook section roles
- Section 1: Method choice and why — problem, target, ranking, baseline, models, leakage, business purpose.
- Section 2: Build the training dataset — real warehouse, aggregation, eligibility, target, 9 features, leakage, GroupKFold.
- Section 3: Train + compare — models, same evaluation setup, results, fold results, LR interpretation, RF selection.
- Section 4: Errors and interpretation — OOF analysis, Top-K, supplementary threshold, FP/FN, business meaning, lessons, human review.

## Final Section 1 takeaway
Historical page signals → predict May decline risk → rank pages → prioritize human review.

## Final Section 2 takeaway
39.3M daily rows → 407,121 pages → 16,513 eligible pages → 9 features → May target → 5-fold client-grouped validation.

## Final Section 3 takeaway
Random Forest achieved the best P@50 of 0.4440 vs baseline 0.3920 and is used for Section 4 error analysis.

## Final Section 4 takeaway
Random Forest improves prioritization but still produces false positives and false negatives, so it remains a human-review decision-support tool.
