# FlyRank ML Internship — Week 4 (ML-06 + ML-07) Context

## Week 4 Goal

Week 4 focused on **signal auditing and building a transparent baseline action score** before moving into learned ML models.

The key idea was:

> A useful signal does not automatically become a decision rule.

I first tested whether important signals showed useful patterns in the data, then built a simple, transparent baseline for prioritizing pages for human review.

---

# Working Workflow

1. **You + ChatGPT:** decide methodology, logic, interpretation, and scientific wording.
2. **Codex / Antigravity:** implement the notebook/code.
3. **You + ChatGPT:** review results and understand what they mean.

Do not let the coding agent randomly choose signals or methodology.

---

# Week 4 Notebooks

- **ML-06:** `work/notebooks/w04_signal_audit.ipynb`
- **ML-07:** `work/notebooks/w04_baseline_score.ipynb`

ML-06 = Signal Audit  
ML-07 = Baseline Action Score and Top-10/Top-20 Review

---

# ML-06 — Signal Audit

## Purpose

The purpose of ML-06 was to test whether several signals had useful relationships with search performance and decline.

The analysis was **descriptive**, not causal.

The tests were intended to answer questions such as:

- Does staleness appear related to engagement?
- Does search position appear related to CTR?
- Does impressions volume show a consistent relationship with decline?
- Does CTR show a consistent relationship with decline?

---

# Section 1 — Feature Distributions

Important distribution observations:

### `impressions_90d`
- Median: **731**
- Mean: **5,200.37**
- p99: **73,505.83**
- Heavy/right-skewed distribution

### `clicks_90d`
- Median: **1**
- Mean: **16.10**
- p99: **253.01**
- Heavy/right-skewed distribution

### `content_age_days`
- Median: **236**
- Mean: **256.17**
- p99: **537**

### `days_since_last_update`
- Median: **20**
- Mean: **46.10**
- p99: **106**
- Maximum: **373**

### `ctr`
- Median: **0.07**
- Mean: **0.51**
- p99: **8.33**

### `avg_position`
- Median: **10.80**
- Mean: **16.34**
- p99: **69.90**
- `avg_position == 0`: **1,205 rows**

These distributions show that several search-performance variables are strongly skewed, so averages alone can be misleading.

---

# ML-06 Signal Tests

## Test 1 — Staleness → Engagement

Engagement was measured using a weighted engagement rate:

```text
engaged_sessions_90d / sessions_90d
```

The numerator and denominator were summed before calculating the rate rather than averaging page-level percentages.

Observed results:

| Staleness | Engagement Rate | n |
| :--- | ---: | ---: |
| 0–30d | 3.00% | 20,480 |
| 31–90d | 2.98% | 175 |
| 91–180d | 2.41% | 9,171 |
| 181+d | 2.70% | 174 |

**Verdict: MIXED**

The relationship was not consistent enough across buckets to treat staleness alone as a strong decision rule.

---

## Test 2 — Search Position → CTR

CTR was calculated using total clicks divided by total impressions.

Observed results:

| Position Group | Weighted CTR | n |
| :--- | ---: | ---: |
| top_3 | 0.49% | 1,116 |
| page_1 | 0.35% | 11,814 |
| striking | 0.35% | 7,304 |
| page_3_5 | 0.15% | 7,242 |
| deep | 0.04% | 1,319 |

**Verdict: CONFIRMED**

The data shows a clear directional pattern: pages in stronger search positions had higher observed CTR.

This supports using search position/CTR as useful supporting signals.

It does NOT prove that improving position will cause the CTR change.

---

## Test 3 — Longer Content → Search Demand

Median impressions by word-count bucket:

| Word Count | Median Impressions | n |
| :--- | ---: | ---: |
| <1000 | 4 | 973 |
| 1000–2000 | 172 | 3,780 |
| 2000–3500 | 997 | 11,263 |
| 3500+ | 1,340 | 6,285 |

**Verdict: CONFIRMED**

Longer-content buckets had higher median impressions in this dataset.

This is an association, not evidence that increasing word count causes more impressions.

This test was later considered less action-oriented and was dropped from the recommended final ML-06 set.

---

## Test 4 — Impressions → Decline

Observed decline rates:

| Impressions Bucket | Decline Rate | n |
| :--- | ---: | ---: |
| 0–100 | 38.92% | 8,006 |
| 101–1,000 | 60.28% | 8,485 |
| 1,001–10,000 | 62.03% | 9,907 |
| 10,000+ | 52.36% | 3,602 |

**Verdict: MIXED**

The decline rate did not increase consistently across all buckets.

Therefore, impressions should not be treated as a standalone decline rule.

---

## Test 5 — CTR → Decline

Observed decline rates:

| CTR Bucket | Decline Rate | n |
| :--- | ---: | ---: |
| <0.05% | 51.52% | — |
| 0.05–0.20% | 62.39% | — |
| 0.20–0.50% | 56.82% | — |
| 0.50–1.00% | 51.31% | — |
| >1.00% | 44.23% | — |

**Verdict: MIXED**

The pattern was not consistent enough to use CTR alone as a decline rule.

---

# Recommended Final ML-06 Tests

The recommended tests to keep were:

1. **Staleness → Engagement — MIXED**
2. **Position → CTR — CONFIRMED**
3. **Impressions → Decline — MIXED**
4. **CTR → Decline — MIXED**

The **Word Count → Impressions** test was dropped from the final recommended set because it was less action-oriented and conceptually overlapped with visibility/impressions.

---

# ML-06 Section 3 — Flag-Linked Staleness Test

This was especially important because it directly tests the assumption behind FlyRank's refresh/staleness flag.

The pages were divided into:

- **Not stale:** fewer than 91 days since last update
- **Stale:** 91 days or more since last update

Results:

| Group | Pages | Decline Rate |
| :--- | ---: | ---: |
| Not stale | 20,655 | 51.20% |
| Stale | 9,345 | 60.85% |

Difference:

**+9.65 percentage points**

---

# Interpretation of the Staleness Test

The stale group had a higher observed decline rate than the not-stale group.

This provides **descriptive support** for the assumption behind the refresh flag.

However:

> This does not mean that being stale causes a page to decline.

A page may not have been updated because:
- it is already accurate
- it is evergreen
- it is performing well
- there may be no reason to update it

Therefore, staleness should be treated as a **human-review signal**, not an automatic reason to refresh a page.

---

# ML-06 Practical Takeaway

No single signal showed a strong and consistent pattern across all buckets.

The staleness test was interesting because pages stale for 91+ days had a **60.85% observed decline rate**, compared with **51.20%** for recently updated pages.

Search position showed a clearer relationship with CTR.

Overall conclusion:

> Signals are better used together as supporting information for human review rather than as automatic decisions.

This became the foundation for ML-07's transparent baseline action score.

---

# ML-07 — Baseline Action Score and Review

## Purpose

ML-07 created a simple baseline before learned ML models.

The baseline should be:

- transparent
- easy to explain
- deterministic
- leakage-safe
- useful for comparison with later ML models

The baseline is a **reference point to beat**, not the final ML model.

---

# ML-07 Required Work

The assignment required three main things:

### 1. Check two signals

Test two signals using visible bucket tables with sample sizes.

At least one signal should be connected to a real FlyRank flag, such as:

- staleness → refresh flag
- CTR/position → CTR-fix logic
- volume → quick-win logic

Use one-word verdicts such as:

- CONFIRMED
- OPPOSITE
- MIXED
- FALSE

A negative result is still useful because it prevents an unsupported rule from being used.

### 2. Encode one transparent rule

The rule should produce:

- score
- one reason code
- action label

Then write the ranked queue to:

`work/outputs/baseline_action_score.csv`

### 3. Review the top pages

Review the Top-10/Top-20 queue.

For each page, explain:

- action
- why it was ranked
- what would make the recommendation wrong

---

# Final ML-07 Baseline Implementation

The final implemented starter-data baseline was:

```python
baseline_flag = clicks_last_30d < clicks_prev_30d
```

Scoring:

```python
baseline_score = impressions_last_30d
```

for flagged pages; otherwise score = 0.

Reason code:

```text
click_decline
```

for flagged pages; otherwise:

```text
no_decline_detected
```

Action:

```text
refresh_review
```

for flagged pages; otherwise:

```text
monitor
```

Ranking:

1. score descending
2. deterministic content ID tie-break

Output:

`work/outputs/baseline_action_score.csv`

Metrics output:

`work/outputs/baseline_metrics.json`

---

# Final ML-07 Baseline Results

Dataset:

- Rows: **30,000**
- Flagged: **6,806**
- Flagged percentage: **22.69%**
- Not flagged: **23,194**
- Base rate: **54.21%**

Precision results:

| Review Depth | Precision |
| :--- | ---: |
| Precision@10 | **0.800** |
| Precision@20 | **0.650** |
| Precision@50 | **0.680** |
| Precision@100 | **0.620** |
| Precision@500 | **0.548** |

The important Week 4 baseline number used for the later comparison was:

**Precision@50 = 0.680**

Note:

This starter-data baseline is separate from the final Week 5 warehouse baseline. Do not mix the two result sets.

---

# ML-07 Top Review

The Top-20 pages were reviewed.

Common weak-pick patterns included:

- recently updated pages
- small click drops
- deep search positions
- old/evergreen pages

These examples helped show why a simple rule can produce false positives.

A page being flagged does not automatically mean it should be refreshed.

---

# ML-07 Leakage Check

Baseline inputs were:

- `clicks_last_30d`
- `clicks_prev_30d`
- `impressions_last_30d`

Forbidden fields included:

- `trend_direction`
- `trend_pct`
- `is_declining_label`
- `content_id`
- `client_id`

The leakage checks passed.

No future-window or label-derived information was used in the final baseline score.

---

# Important Week 4 Distinction

There are two different datasets/results that must not be mixed:

## Weeks 1–4 starter snapshot

- 30,000 rows
- used for early discovery, framing, signal audit, and ML-07 starter baseline
- ML-07 P@50 = **0.680**

## Week 5 real warehouse modeling dataset

- 16,513 eligible pages
- actual Feb–Apr features + May target
- Week 5 baseline P@50 = **0.3920**
- Random Forest P@50 = **0.4440**

These are different evaluation setups and should not be compared as if they were the same experiment.

---

# Week 4 Scientific Honesty Rules

Use wording such as:

- observed
- measured
- associated with
- descriptive support
- directional
- mixed
- confirmed
- supporting signal
- human review

Avoid:

- causes
- guarantees
- proves SEO improvement
- guarantees a refresh will recover traffic
- predicts Google's proprietary algorithm

---

# Final Week 4 Story

The Week 4 progression was:

**ML-06: Test signals**
→ identify which signals show useful patterns

**ML-06: Flag-linked staleness test**
→ stale pages had 60.85% decline vs 51.20% for not-stale pages

**ML-07: Build transparent baseline**
→ use a simple click-decline rule and rank flagged pages

**ML-07: Evaluate**
→ starter baseline P@50 = 0.680

**ML-07: Review weak picks**
→ simple rules can still flag pages that do not need action

**ML-07: Leakage check**
→ confirm baseline uses allowed information only

This created the transparent baseline needed before moving to Week 5 learned models.

---

# Week 4 Key Numbers

## ML-06

- Not stale: **51.20% decline**
- Stale: **60.85% decline**
- Staleness difference: **+9.65 percentage points**
- Position → CTR: **CONFIRMED**
- Staleness → Engagement: **MIXED**
- Impressions → Decline: **MIXED**
- CTR → Decline: **MIXED**

## ML-07

- Starter rows: **30,000**
- Flagged: **6,806**
- Flagged: **22.69%**
- Base rate: **54.21%**
- P@10: **0.800**
- P@20: **0.650**
- P@50: **0.680**
- P@100: **0.620**
- P@500: **0.548**

---

# Relationship to Week 5

Week 4 established the baseline and the principle that:

> **A simple transparent rule should be evaluated before adding model complexity.**

Week 5 then moved to the real warehouse setup and compared:

- the frozen Week 5 baseline
- Logistic Regression
- Decision Tree
- Random Forest

using the actual May decline target and client-grouped validation.

Do not compare the Week 4 starter P@50 of **0.680** directly with the Week 5 P@50 values because they use different datasets, targets, and evaluation setups.
