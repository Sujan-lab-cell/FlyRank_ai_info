# FlyRank ML Internship — Week 6

## ML-09: Validation and Research Claim Audit

**Week:** 6  
**Phase:** Build+  
**Workload:** 5 hours  
**Track:** Machine Learning

---

## 1. Overview

Week 6 focused on auditing the Week 5 machine learning model instead of only looking at its performance score.

The main goal was to check whether the validation setup was honest, whether any features contained future information, and whether the final claims matched the evidence.

The audit followed four main ideas:

- Review methodology, not just results
- Use an honest validation split
- Check for data leakage
- Rewrite claims so they do not go beyond the evidence

---

## 2. Research Paper Findings and Methodology Questions

I reviewed two findings from FlyRank's research paper, **The State of AI-Driven SEO, March 2026**, and considered what methodology questions should be asked before interpreting the findings too strongly.

### Finding 1 — Engagement and Visibility Move Together

The paper reports that pages with high scroll depth and high engagement have higher Health Scores, with an observed difference of about **11.2 Health Score points** between the strongest and weakest bucket.

**My methodology question:**

Because scroll depth is already one of the components used to calculate Health Score, could part of this relationship come from the metric construction itself?

I would want to check the relationship using an outcome that does not directly include scroll depth.

This does not mean the finding is wrong. It means the strength of the relationship should be interpreted carefully.

### Finding 2 — The Freshness Multiplier

The paper reports that **365+ day-old content refreshed within 30 days** showed a **3.2× Health Score boost** and **57× more impressions**.

**My methodology question:**

How were refreshed pages selected, and was there a comparable control group or time-aware validation design?

Pages chosen for refresh may already differ from pages that were not refreshed.

I would want to know whether the comparison supports a refresh effect or only shows an observed difference between the two groups.

These questions are meant to make the findings more rigorous, not to reject them.

---

## 3. My Model Under an Honest Split

I wanted to check how much the validation method affects my model's reported performance.

For this comparison, I used the same dataset, features, and Random Forest configuration, and changed only the way the data was split.

### Validation Setup

Both experiments used:

- **Eligible population:** 16,513 pages across 36 clients
- **Target:** `may_clicks < 0.8 * april_clicks`
- **Decline base rate:** 41.62%
- **Features:** 9 pre-May historical features
- **Model:** Random Forest with `n_estimators=100`, `max_depth=5`, `random_state=42`
- **Primary metric:** Precision@50
- **Ranking:** predicted score descending with deterministic tie-breaking

### Before vs. After

| Validation setup | Split type | Precision@50 | Client overlap |
|---|---|---:|---:|
| **Before** | 5-Fold Random Row Split (`KFold`) | **0.9960** | **28 clients shared** |
| **After** | 5-Fold `GroupKFold` by client | **0.4440** | **0 clients shared** |

### What I learned

With a random row split, pages from the same client can appear in both training and validation folds. This can make the validation score look overly optimistic because the model sees other pages from the same clients during training.

The random split produced a very high Precision@50 of **0.9960**, but it had **28 clients shared** between training and validation.

With `GroupKFold`, all pages belonging to a client stay in the same fold. This resulted in **zero client overlap** and provides a more conservative test of generalization to clients that were not seen during training.

Under this setup, Precision@50 was **0.4440**.

The large difference between the two validation setups shows that the choice of validation strategy has a major effect on the measured performance. For this evaluation, I use the **0.4440 client-grouped result** as the more conservative result.

---

## 4. Leakage Audit

### What I checked

Before trusting the Week 5 model results, I checked whether any feature was accidentally using information from the future.

**Data leakage** happens when information that would not be available at prediction time gets into the model features. This can make a model look much better during validation than it would in a real prediction setting.

### Temporal Boundary

- **Prediction cutoff:** April 30, 2026
- **Feature window:** February 1 – April 30, 2026
- **Target window:** May 1 – May 31, 2026
- **Target:** `may_clicks < 0.8 * april_clicks`

The rule used was:

> **Features can use information available before May. May data can only be used to create the target.**

### Model Features

The Random Forest uses these 9 features:

1. `impressions_total` — total impressions from Feb–Apr
2. `clicks_total` — total clicks from Feb–Apr
3. `april_impressions` — impressions during April
4. `april_clicks` — clicks during April
5. `feb_clicks` — clicks during February
6. `momentum` — April clicks relative to February clicks
7. `ctr` — pre-May click-through rate
8. `active_days` — number of active days during Feb–Apr
9. `weighted_position` — impression-weighted position during Feb–Apr

The audit confirmed that all 9 features use data available on or before **April 30, 2026**.

**Result: No future information was found in the model features.**

### Target and Future Outcome Data

The May data was kept separate from the model features.

- `may_clicks` — May clicks
- `may_impressions` — May impressions
- `decline` — target defined as `may_clicks < 0.8 * april_clicks`

The audit confirmed that:

- `may_clicks` is not included in the feature matrix `X`
- `may_clicks` is used strictly to create the target `y`
- `may_impressions` is not included in `X`

### Identifiers

I also checked:

- `client_hash_id`
- `content_hash_id`

These are not model features.

`client_hash_id` is used for `GroupKFold`, while `content_hash_id` is used for joining, sorting, and deterministic tie-breaking.

### Other Potentially Risky Fields

The audit also checked fields such as:

- `trend_direction`
- `trend_pct`
- other target-derived or future fields

These were excluded from the model feature matrix.

### Compact Leakage Audit

| Feature / Field | Role | Data window | Available before May? | Decision |
|---|---|---|---|---|
| `impressions_total` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `clicks_total` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `april_impressions` | Model feature | Apr 2026 | Yes | **APPROVED** |
| `april_clicks` | Model feature | Apr 2026 | Yes | **APPROVED** |
| `feb_clicks` | Model feature | Feb 2026 | Yes | **APPROVED** |
| `momentum` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `ctr` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `active_days` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `weighted_position` | Model feature | Feb–Apr 2026 | Yes | **APPROVED** |
| `decline` | Target | May 2026 | Target only | **ISOLATED TO Y** |
| `may_clicks` | Future outcome | May 2026 | No | **EXCLUDED FROM X** |
| `may_impressions` | Future outcome | May 2026 | No | **EXCLUDED FROM X** |
| `trend_direction` / `trend_pct` | Outcome-derived | Post-May / outcome | No | **EXCLUDED FROM X** |
| `client_hash_id` | Identifier | Static pseudonym | Yes | **GROUPING ONLY** |
| `content_hash_id` | Identifier | Static pseudonym | Yes | **TIE-BREAK ONLY** |

### Leakage Audit Result

The audit confirmed:

1. All 9 model features use only information available on or before April 30, 2026.
2. `may_clicks` is used only to create the target and is not included in the feature matrix.
3. Future outcome fields and target-derived fields are excluded from `X`.
4. Client and content identifiers are not passed to the model as features.

**VERDICT: LEAKAGE AUDIT PASSED** ✅

---

## 5. Claim Rewrite

### What the Validation Results Actually Support

The random row split produced a very high **Precision@50 of 0.9960**. However, this split had **28 clients shared** between the training and validation folds.

Because pages from the same client could appear in both sets, this result may give an overly optimistic view of how well the model generalizes to new clients.

I then evaluated the same model using **5-fold GroupKFold by client**. This resulted in **0 client overlap** between training and validation sets, and Precision@50 was **0.4440**.

The difference between the two validation setups was **55.20 percentage points**. This shows that the choice of validation strategy has a large effect on the measured performance.

For this reason, I use the **0.4440 Precision@50 from the client-grouped validation** as the more conservative result for this evaluation.

### Key Evidence

- **Population:** 16,513 eligible pages across 36 clients
- **Decline base rate:** 41.62%
- **Metric:** Precision@50, not accuracy
- **Validation:** 5-fold GroupKFold by client
- **Client overlap:** 0
- **Model features:** 9 pre-May features
- **Leakage audit:** Passed
- **Feature cutoff:** April 30, 2026

### What I Should Not Claim

I should not claim that the model will achieve **44.4% Precision@50 in production**, because this result comes from one dataset and one validation setup.

I should also not claim that the model proves that any feature causes traffic decline or that refreshing a page will recover lost traffic. The analysis provides **observational and predictive evidence**, not causal evidence.

The **0.9960 random-split result** should also not be presented as the model's reliable performance because the random split allowed client overlap.

### Naive Claim vs. Safer Claim

| Naive claim | Safer claim |
|---|---|
| "The model achieves 99.6% accuracy." | **The model measured 0.444 Precision@50 under client-grouped validation.** |
| "The model reliably predicts decline for all clients." | **The model provides directional evidence for ranking potentially declining pages.** |
| "The model proves what causes traffic decline." | **The results show predictive associations, not causation.** |
| "The model can automatically decide which pages to refresh." | **The model can support human review by prioritizing pages for further investigation.** |

### Final Safe Claim

> **Under 5-fold client-grouped validation, the Random Forest measured a Precision@50 of 0.444 for the defined May 2026 decline target. This result provides directional evidence that the model can help rank potentially declining pages for human review, while its performance on unseen clients requires further validation.**

---

## 6. Self-check

Before submitting, confirm each line honestly:

- [x] Every section above is filled — markdown thinking AND the code that backs it
- [x] The notebook runs top to bottom with no errors (Runtime → Run all)
- [x] No client names, URLs, or private queries anywhere
- [x] My claims use careful words: observed, measured, directional, decision-support
- [x] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.

---

## Final Takeaway

Week 6 helped me understand that getting a high model score is not enough. The validation strategy, feature timing, leakage checks, and wording of the final claim all matter.

The most important lesson from this audit was the difference between the random row split (**0.9960 Precision@50**) and the client-grouped split (**0.4440 Precision@50**). Using a grouped split gave a more conservative evaluation and made the model's limitations clearer.

The final model should therefore be viewed as a **decision-support tool for prioritizing pages for human review**, rather than as an automatic or causal system.
