# FlyRank ML Internship — Week 3 (ML-04 + ML-05) Context

## Week 3 Goal

Week 3 focused on building a reliable **Search Intelligence data contract**, checking the real warehouse data, creating the first leakage-safe features, and deliberately testing for data leakage.

The main principle was:

> Before training a model, first prove that the data, unit of analysis, features, target, and missingness rules are correct.

---

# Working Workflow

1. **You + ChatGPT:** decide methodology, logic, interpretation, and scientific wording.
2. **Codex / Antigravity:** implement the code.
3. **You + ChatGPT:** review the results and understand what they mean.

Do not let the coding agent randomly choose methodology.

For Week 3, the coding agent should first read:

`skills/README.md`

and then load the relevant FlyRank data skill.

---

# Week 3 Notebooks

- **ML-04:** `work/notebooks/w03_data_contract.ipynb`
- **ML-05:** `work/notebooks/w03_feature_leakage_check.ipynb`

ML-04 = Search Intelligence Data Contract  
ML-05 = Feature + Leakage Check

---

# ML-04 — Search Intelligence Data Contract

## Official Purpose

ML-04 focused on:

1. Describing the lane's data slice.
2. Proving important data facts using real warehouse queries.
3. Building the first 5 features.
4. Deliberately triggering a leakage test.
5. Removing the leakage.
6. Checking missingness and data quality.

The goal was to create a trustworthy foundation before moving to signal analysis and the baseline.

---

# Warehouse

The FlyRank warehouse is:

`FlyRank/internship-warehouse`

It is a gated Hugging Face dataset.

Required affiliation:

`FlyRank ML Internship 2026`

Important warehouse tables:

| Table | Rows | Purpose |
| :--- | ---: | :--- |
| `dim_clients` | 104 | Client information |
| `dim_content` | 519,606 | Content/page information |
| `fact_content_daily_performance` | 78,835,655 | Daily page performance |
| `fact_content_daily_performance_sample` | 11,694,072 | June 2026 final-month sample |
| `fact_content_query_90d` | 2,414,248 | Query-level 90-day context |

Daily performance grain:

**report_date × client × content**

Warehouse date range:

**January 27, 2025 – June 30, 2026**

The `_sample` table represents **June 2026 exactly**. It is not a random sample.

---

# Week 3 Data Contract

## Unit of Analysis

The starter modeling unit is:

> **One unique content item (`content_id`) belonging to one client (`client_id`).**

The starter CSV is a fixed trailing-90-day snapshot.

The warehouse contains a daily panel, which can later be aggregated into the required modeling windows.

---

# Important Data Gotchas

## Percentage Columns

Several columns are stored as percentage-point values multiplied by 100.

Examples:

- `ctr = 0.76` means **0.76%**
- `engagement_rate`
- `scroll_rate`
- `ai_traffic_pct`
- `trend_pct`

Do not interpret `0.76` as 76%.

---

## Average Position

`avg_position = 0` means **no position data**.

Starter-data count:

**1,205 rows**

Zero should therefore not automatically be treated as a real search position.

---

## Scroll and AI Traffic

`scroll_rate` and `ai_traffic_pct` can exceed 100 in the supplied data.

These should not automatically be treated as impossible values without understanding the source definition.

---

## Trend Fields

The following are target-derived / outcome-related fields and must not be used as model features:

- `trend_direction`
- `trend_pct`

These fields describe the observed trend and can leak the outcome.

---

## IDs

IDs such as:

- `client_id`
- `content_id`

are pseudonymous identifiers.

They are useful for:

- grouping
- joining
- splitting

but should **never be predictive features**.

---

## Missingness

Missingness is related to `content_type`.

Blindly doing:

```python
fillna(0)
```

can accidentally inject category information into the model.

Instead, use explicit missingness indicators such as:

```text
has_word_count
has_search_volume
```

where appropriate.

---

## Unbalanced Panel

Different clients have different history depths.

The client start date can be checked using:

`dim_clients.gsc_data_start`

Per-client windows are preferred when historical depth differs.

---

## GA4 Data

GA4 columns may be zero-filled before the client's:

`ga4_data_start`

`ga4_data_available` can also be NULL.

Use explicit checks such as:

```sql
IS TRUE
```

or:

```sql
IS NOT TRUE
```

rather than assuming NULL means the same thing as zero.

---

## Query-Level Data

The query table repeats client-level/content-level context on each query row.

When aggregating query data, use:

```sql
ANY_VALUE()
```

for repeated context.

Do **not** use:

```sql
SUM()
```

on values that are repeated for every query.

Otherwise, values can be multiplied incorrectly.

---

# Development vs Sealed Data

The warehouse guidance was:

- Use **March 2026** for development.
- Keep **June 2026** sealed.

June is the final month represented by the `_sample` table.

The reason is to prevent accidentally using a later period while developing.

---

# ML-04 Warehouse Setup Result

The warehouse setup was successfully verified.

Loaded data:

```text
dim_clients              104 rows
dim_content           519,606 rows
fact_daily_sample    11,694,072 rows
fact_query_90d        2,414,248 rows
```

This confirmed that the real warehouse was accessible and that the required tables existed.

---

# ML-05 — Feature + Leakage Check

## Purpose

ML-05 tested whether the proposed features were safe to use for modeling.

The workflow was:

1. Define the starter target.
2. Check missingness.
3. Build honest features.
4. Measure a baseline feature model.
5. Deliberately introduce an obvious leakage feature.
6. Confirm that leakage produces an unrealistically strong result.
7. Remove the leakage.
8. Freeze the final feature set.

---

# Starter Target Used in ML-05

For the starter-data feature/leakage exercise:

```python
df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)
```

Distribution:

- `1` = **16,262**
- `0` = **13,738**
- Total = **30,000**

This was a **starter-data proxy target**.

Important:

It is not the actual Week 5 May target.

The actual Week 5 target was later rebuilt from the real warehouse:

```python
may_clicks < 0.8 * april_clicks
```

---

# Missingness Audit

Important missingness results:

```text
search_volume    2468
word_count       7699
cpc              2468
avg_position        0
```

Zero-position rows:

```text
1205
```

This reinforced the need to understand missingness rather than blindly filling everything with zero.

---

# Honest Features

The first leakage-safe feature set included:

- `impressions_90d`
- `ctr`
- `avg_position`
- `days_since_last_update`
- `has_word_count`
- `has_search_volume`

These were chosen because they were available independently of the target and represented useful historical page/search signals.

---

# Honest Feature Model Result

A Random Forest using the honest features achieved:

**ROC-AUC = 0.695**

This showed that the historical features contained useful signal, while still being far from perfect prediction.

---

# Deliberate Leakage Test

To prove that leakage matters, a deliberately invalid feature was created:

```python
leak_feature = y
```

This means the feature directly contained the target.

The model achieved:

**ROC-AUC = 1.000**

This was intentionally done as a sanity check.

The result demonstrates:

> If the target is allowed into the features, the model can appear artificially perfect.

Therefore, the leakage feature was removed.

---

# Final ML-05 Feature Matrix

The final feature matrix contained:

**23 features**

The final check confirmed:

**0 excluded fields overlapping with the final model features.**

---

# Fields Explicitly Excluded

The following were excluded from the final feature matrix:

### Target/trend-derived

- `trend_pct`
- `trend_direction`
- `is_declining_label`

### Target-overlapping historical comparison fields

Some last/previous-30-day comparison fields were excluded because they overlapped with the target definition and could leak outcome information.

### Identifiers

- `content_id`
- `client_id`

IDs are context only.

---

# Core Week 3 Lesson: Leakage

Data leakage happens when the model receives information that would not actually be available at the time the prediction is supposed to be made.

Example:

If we want to predict future decline, we cannot give the model the future decline itself or a field derived from it.

The deliberate test:

```text
leak_feature = target
```

produced:

**ROC-AUC = 1.000**

This is a clear warning sign, not a successful model.

---

# Week 3 Key Principles

## 1. Data first

Do not train a model before understanding:

- data source
- grain
- time range
- missingness
- column meanings
- client history

## 2. IDs are not features

Pseudonymous IDs can help with grouping and validation but should not be used as predictive inputs.

## 3. Missingness can contain information

Missing values may be related to content type or data availability.

Blind zero-filling can create unintended signals.

## 4. Time matters

For future prediction, features must come from information available before the decision point.

## 5. Leakage can make a bad model look excellent

A ROC-AUC of 1.000 caused by directly using the target is not useful.

## 6. Features should have a real meaning

Each feature should represent information that could realistically be known when making the prediction.

---

# Relationship to Week 4

Week 3 established the trustworthy data foundation.

The progression was:

**ML-04**
→ understand warehouse + data contract

**ML-05**
→ audit missingness + build features + prove leakage risk

**ML-06**
→ audit signals and test their observed relationships

**ML-07**
→ build a transparent baseline action score

**ML-08 / Week 5**
→ build and compare learned models using the real warehouse and future May target

---

# Important Distinction: Starter Data vs Week 5 Warehouse

The starter dataset used in Weeks 1–4 had:

- 30,000 rows
- a fixed trailing-90-day snapshot
- `trend_direction`
- `trend_pct`

The Week 5 modeling setup was rebuilt from the real daily warehouse because the final future-looking target required actual February–April features and May outcomes.

Therefore:

**Do not mix the Week 3/4 starter proxy target with the Week 5 May target.**

---

# Key Week 3 Numbers

## Warehouse

- `dim_clients`: **104**
- `dim_content`: **519,606**
- Daily performance: **78,835,655**
- June sample: **11,694,072**
- Query 90d: **2,414,248**

## ML-05 Starter Data

- Rows: **30,000**
- Target 1: **16,262**
- Target 0: **13,738**
- Honest-feature ROC-AUC: **0.695**
- Deliberate leakage ROC-AUC: **1.000**
- Final feature matrix: **23 features**
- Zero-position rows: **1,205**

---

# Scientific Honesty Rules

Use:

- observed
- measured
- descriptive
- associated with
- signal
- leakage-safe
- available before the decision point

Avoid:

- causes
- guarantees
- proves
- predicts Google's proprietary algorithm
- guarantees SEO improvement

The data contract and feature audit establish what information is available and safe to use. They do not prove causal relationships.

---

# Final Week 3 Takeaway

Week 3 was about making the modeling foundation trustworthy.

The key chain was:

**Real warehouse → understand data grain → audit missingness → define safe features → deliberately test leakage → remove leakage → freeze the feature set**

Only after this foundation was established did the project move into signal analysis and baseline scoring in Week 4.
