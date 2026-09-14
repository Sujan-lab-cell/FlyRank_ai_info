# FlyRank Machine Learning Internship

## Overview

This repository contains my complete work and learning journey during the **FlyRank Machine Learning Internship**.

The internship provided practical experience in applying machine learning to **real-world Google Search ranking and discoverability data**, with an emphasis on data analysis, leakage prevention, model validation, research thinking, decision-support systems, and communicating results honestly.

The internship culminated in a **Google Search Ranking & Discoverability Capstone**, where I developed a machine-learning workflow to prioritize webpages that may require SEO/content review.

---

## Internship Objectives

The main objectives of the internship were to:

- Work with real-world search-performance data.
- Understand and audit large datasets.
- Identify meaningful ML signals.
- Avoid target leakage and invalid features.
- Build transparent machine-learning baselines.
- Compare multiple ML models.
- Use appropriate validation strategies.
- Evaluate models using business-relevant metrics.
- Translate model predictions into actionable recommendations.
- Understand limitations and avoid unsupported causal claims.
- Produce a reproducible research paper.
- Communicate technical findings to both technical and non-technical audiences.

---

# Internship Journey

## Week 1–2 — Foundation & Project Setup

The initial phase focused on understanding the internship workflow, repository structure, available data, project requirements, and the overall machine-learning problem.

The work established the foundation for later stages involving:

- Dataset exploration
- Feature understanding
- ML methodology
- Reproducibility
- Git/GitHub workflow
- Public-safe reporting

---

# Week 3 — Data Audit & Leakage Prevention

The first major analytical phase focused on understanding the available FlyRank search data and identifying reliable features.

### Dataset Scale

The warehouse contained approximately:

| Dataset | Records |
|---|---:|
| Daily performance data | 78.8M |
| Clients | 104 |
| Content | 519,606 |
| Sampled data | 11.7M |
| Query data | 2.4M |

The starter dataset contained approximately **30,000 rows across 32 clients and 44 columns**.

### Important Data Findings

The analysis examined:

- Missing values
- Zero-position records
- Search impressions
- Clicks
- CTR
- Search position
- Content age
- Update recency
- Word count
- Search volume

### Leakage Investigation

One of the most important lessons was understanding **target leakage**.

A deliberate experiment showed:

> Using the target itself as a feature produced ROC-AUC = **1.000**.

After removing the leakage, the honest Random Forest result was:

> ROC-AUC = **0.695**

This demonstrated why features must represent information that would actually be available **before the outcome occurs**.

### Final Honest Features

The early analysis used signals such as:

- `impressions_90d`
- `ctr`
- `avg_position`
- `days_since_last_update`
- `has_word_count`
- `has_search_volume`

IDs were retained only for:

- Joining
- Grouping
- Splitting
- Deterministic tie-breaking

They were not treated as predictive signals.

---

# Week 4 — Search Signal Audit

The next phase focused on understanding which historical signals were associated with search-performance behavior.

Several hypotheses were tested.

### Key Findings

#### 1. Staleness and Engagement

Pages that had not been updated for longer periods showed a higher observed decline rate.

| Group | Decline Rate |
|---|---:|
| Not stale | 51.20% |
| Stale | 60.85% |

Difference:

> **+9.65 percentage points**

This was treated as an **observed association**, not proof that stale content causes decline.

#### 2. Search Position and CTR

The analysis found a clear relationship between search position and CTR, supporting the use of search visibility-related signals in later modeling.

#### 3. Content Length and Search Demand

Longer content showed an association with search demand.

This was treated as an **association rather than causation**.

#### 4. Impressions and Decline

The relationship between impressions and decline was mixed.

#### 5. CTR and Decline

CTR also showed a mixed relationship with decline.

These results reinforced the importance of validating signals instead of assuming that a commonly used SEO metric is automatically predictive.

---

# Week 5 — Machine Learning Capstone Modeling

The main capstone modeling work focused on predicting meaningful search-performance decline.

## Research Question

> **Which webpages should an SEO/content team review first when many pages may be experiencing search-performance decline?**

The system was designed as a **decision-support tool**, not an autonomous SEO system.

## Target Definition

The May decline target was defined as:

```text
May clicks < 80% of April clicks
```

A page was labeled as declining when May clicks fell below 80% of April clicks.

## Eligibility

Pages were included when:

```text
Total impressions >= 1000
AND
April clicks >= 10
```

This resulted in:

- **16,513 eligible pages**
- **36 clients**
- **41.62% decline base rate**

---

# Features

The final model used nine historical features:

1. Impressions — February to April
2. Clicks — February to April
3. April impressions
4. April clicks
5. February clicks
6. Momentum
7. CTR
8. Active days
9. Weighted search position

The features were based on information available **before the May outcome**.

May performance was used only to construct the target.

---

# Validation Strategy

A major part of the internship was learning why validation strategy matters.

The final model used:

> **5-fold GroupKFold by client**

This prevents the same client from appearing in both training and validation folds.

### Validation Check

Client overlap between training and validation:

> **0 clients**

This provides a more honest estimate of performance when the goal is to generalize to unseen clients.

---

# Baseline

A simple rule-based baseline was created before evaluating the ML models.

The baseline identified pages where:

```text
April clicks < March clicks
```

Flagged pages were then ranked using April impressions.

This provided a transparent benchmark against which ML models could be compared.

---

# Model Comparison

Several model families were evaluated using the same feature set, evaluation population, validation strategy, metric, and tie-breaking policy.

| Model | P@50 |
|---|---:|
| Baseline | **0.392** |
| Logistic Regression | 0.424 |
| Decision Tree | 0.324 |
| HistGradientBoosting | 0.436 |
| LightGBM | 0.438 |
| CatBoost | 0.440 |
| XGBoost | 0.442 |
| Random Forest | **0.444** |

The **Random Forest** was selected as the final model.

## Final Model Result

**Random Forest P@50: 0.444**

**Baseline P@50: 0.392**

**Improvement: +5.2 percentage points**

This result provides directional evidence that the model can improve top-50 prioritization under client-grouped validation. It is not treated as a production guarantee or SLA.

---

# Why Precision@K?

The practical question was not:

> "Can we classify every webpage perfectly?"

Instead:

> "If an SEO/content team can only investigate a limited number of pages, which pages should they look at first?"

Therefore, **Precision@K** was particularly useful.

- P@10 — quality of the top 10 recommendations
- P@20 — quality of the top 20
- P@50 — quality of the top 50
- P@100 — quality of the top 100

This connects ML evaluation directly to a realistic human-review workflow.

---

# Week 6 — Validation & Research Claim Audit

Week 6 focused on making the project more scientifically honest.

## Random Split vs Client-Grouped Validation

An earlier random split produced:

> P@50 ≈ **0.996**

After changing to client-grouped validation:

> P@50 = **0.444**

Difference:

> **-55.2 percentage points**

The key lesson was that random row-level validation can provide an overly optimistic estimate when related pages from the same clients appear across train and validation sets.

The grouped result was therefore used for the final capstone claim.

## Leakage Audit

The final model used:

- 9 features
- 0 leaked future/target fields
- 0 IDs as predictive features

Feature cutoff:

> **April 30, 2026**

The May outcome was kept separate from the feature matrix.

The leakage audit passed.

## Research Finding Review

The internship also involved reviewing research findings and questioning whether apparently strong relationships could support the proposed conclusions.

Two important findings were examined:

### Engagement and Visibility

A research finding suggested that high engagement and high scroll depth were associated with higher Health Scores.

Because scroll depth could be part of the Health Score itself, the relationship was treated carefully to avoid circular reasoning.

### Freshness

Another finding reported a strong improvement for older content that had been refreshed.

The analysis highlighted the need to consider:

- Selection effects
- Comparable controls
- Time-aware validation
- Pre-existing differences between refreshed and non-refreshed pages

The project therefore avoided claiming that refreshing content causes a particular increase.

---

# Week 7 — Content Action Playbook

The model was translated into a practical review queue.

The guiding principle was:

> **The model recommends where to look. A human decides what to do.**

## Final Queue

The eligible population contained:

> **16,513 pages**

| Recommendation | Pages |
|---|---:|
| Model signal only / manual investigation | 723 |
| Decline + low visibility / SEO content review | 249 |
| Monitor | 15,541 |

## Reason Codes

### `model_signal_only`

The model identified the page as worth investigating, but additional evidence is required.

### `decline_and_low_visibility`

The page showed the model signal and weak search visibility based on the defined weighted-position threshold.

### `monitor`

The page did not meet the stronger review conditions and should continue to be monitored.

---

# Human-in-the-Loop Workflow

```text
Model Ranking
      ↓
Human Review
      ↓
Check Evidence
      ↓
Choose Action
      ↓
Human Approval
      ↓
Manual Execution
```

The system does **not** automatically modify webpages.

## What the System Should NOT Automate

The project explicitly avoided automatically:

- Editing content
- Publishing content
- Deleting pages
- Redirecting URLs
- Changing titles
- Changing meta descriptions
- Changing H1s
- Changing canonical tags
- Changing internal links
- Making high-stakes claims
- Claiming the root cause of decline
- Predicting Google algorithm changes
- Assuming traffic recovery

A high model score alone should never trigger a website change.

---

# Week 8 — ML-11 Research Paper

The final research paper was developed around the capstone work.

The paper includes:

1. Title and Abstract
2. Introduction / Problem Statement
3. Data
4. Methodology
5. Results
6. Limitations & Honest Framing
7. Ranked Recommendations
8. Reproducibility
9. Acknowledgments & Data Credit

The paper was deployed using GitHub Pages.

### Research Paper

https://sujan-lab-cell.github.io/flyrank-ml-internship/

---

# ML-12 — Tell the Story

The final stage focused on communicating the project clearly.

The capstone notebook contains:

- 5-minute demo outline
- Main research question
- Methodology
- Main result
- Precision@K chart
- Recommendation
- Short social-media post
- Employer-facing project summary

The main story is:

> Historical search-performance signals can help prioritize potentially declining webpages for human review, with the Random Forest achieving a Precision@50 of 0.444 compared with 0.392 for the baseline under client-grouped validation.

---

# Final Capstone

## Google Search Ranking & Discoverability Capstone

The capstone combines the internship's major lessons:

```text
Real Search Data
       ↓
Data Audit
       ↓
Leakage Prevention
       ↓
Signal Analysis
       ↓
Baseline
       ↓
Machine Learning
       ↓
Client-Grouped Validation
       ↓
Error / Claim Analysis
       ↓
Action Playbook
       ↓
Research Paper
       ↓
Human Decision Support
```

---

# Technology Stack

### Programming
- Python

### Data Processing
- Pandas
- DuckDB

### Machine Learning
- Scikit-learn
- Random Forest
- Logistic Regression
- Decision Tree
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

### Visualization
- Matplotlib

### Data / Dataset Access
- Hugging Face

### Development & Version Control
- Git
- GitHub
- Jupyter Notebook

### Deployment
- GitHub Pages

---

# Repository Structure

```text
flyrank-ml-internship/
│
├── docs/
│   ├── index.html
│   └── figures/
│       ├── ml_precision_at_k_comparison.png
│       └── ml_random_forest_grouped_fold_p50.png
│
├── work/
│   ├── notebooks/
│   │   └── capstone.ipynb
│   │
│   ├── figures/
│   │
│   └── outputs/
│       └── ml_action_playbook_metrics.json
│
├── submission/
│   └── paper_url.txt
│
└── README.md
```

---

# Reproducibility

The project was designed around reproducible ML practices.

Important principles included:

- Fixed feature definitions
- Explicit target definition
- Explicit eligibility rules
- Deterministic ranking
- Client-grouped validation
- Leakage checks
- Consistent evaluation population
- Reproducible metrics
- Version-controlled notebooks
- Public-safe outputs

The ranking tie-break policy used:

```text
model_score DESC
April clicks DESC
content_hash_id ASC
```

This ensures deterministic ordering.

---

# Data Privacy & Public Safety

Because the underlying dataset contains sensitive SEO/client information, the public repository does **not** expose:

- Client names
- Client domains
- Private URLs
- Private search queries
- Credentials
- Raw data exports
- Authentication tokens
- Other identifying client information

The project communicates methodology and aggregate results while keeping sensitive information private.

---

# Key Lessons Learned

### 1. Data understanding comes before modeling

A model cannot fix a misunderstood dataset.

### 2. Leakage can make a model look perfect

A score of 1.000 is not automatically a success.

### 3. Validation strategy matters

The same model can look dramatically different under random and client-grouped validation.

### 4. Baselines are important

A complex ML model should demonstrate value over a simple rule.

### 5. Business metrics matter

Precision@K was more useful for this problem than focusing only on overall classification accuracy.

### 6. Correlation is not causation

Observed relationships should not automatically become causal recommendations.

### 7. Models should support people

The final system recommends pages for investigation rather than automatically changing websites.

### 8. Honest communication is part of ML

A strong ML project should clearly communicate:

- What was measured
- What worked
- What did not work
- What assumptions were made
- What cannot yet be concluded

---

# Final Results

| Metric | Result |
|---|---:|
| Eligible pages | 16,513 |
| Clients | 36 |
| Decline base rate | 41.62% |
| Validation | 5-fold GroupKFold |
| Client overlap | 0 |
| Features | 9 |
| Final model | Random Forest |
| Random Forest P@50 | **0.444** |
| Baseline P@50 | **0.392** |
| Improvement | **+5.2 pp** |

---

# Final Deliverables

### GitHub Repository

https://github.com/Sujan-lab-cell/flyrank-ml-internship

### Capstone Notebook

https://github.com/Sujan-lab-cell/flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb

### Research Paper

https://sujan-lab-cell.github.io/flyrank-ml-internship/

---

# Acknowledgments

**Built on the FlyRank ML Internship dataset.**

Data credit: https://flyrank.ai

I would like to thank the **FlyRank team** for providing this internship opportunity and for creating a practical learning experience. I’m grateful for the guidance, resources, and real-world data that helped me apply my machine learning knowledge to a meaningful problem.

---

## Internship Summary

The FlyRank Machine Learning Internship gave me hands-on experience in taking an ML problem from **raw data to a validated model, practical recommendations, and a publicly deployed research paper**.

The most important takeaway was not simply achieving a higher model score, but learning how to build an ML solution that is:

**Data-aware → Leakage-safe → Properly validated → Interpretable → Actionable → Honest → Reproducible**

This repository represents my complete internship journey and final capstone work.
