# FlyRank Machine Learning Internship

## Internship Progress & Work Summary

This repository contains my work for the **FlyRank Machine Learning Internship**, covering the ML problem-framing, data understanding, feature engineering, data-contract validation, and leakage/privacy-check stages.

The main objective of the internship is not simply to train a machine learning model, but to build an ML workflow where the **business decision, data, target, features, evaluation metric, and validation strategy are clearly defined and defensible**.

My selected problem direction is:

> **Lane 2 — Refresh / Content Opportunity Scoring**

The goal is to build a decision-support system that can rank webpages according to how much they appear to deserve human content review.

The system is intended to help an SEO/content team answer:

> **"Which pages should we investigate first?"**

The model is not intended to automatically rewrite pages or guarantee that a refresh will improve performance. Its purpose is to provide a **priority score and ranked review queue** for human decision-making.


---

# Table of Contents

1. [Internship Objective](#internship-objective)
2. [Selected Lane](#selected-lane)
3. [Overall Progress](#overall-progress)
4. [Week 1 — Discovery and First ML Results](#week-1--discovery-and-first-ml-results)
   - [ML-01: First Look and Discovery](#ml-01-first-look-and-discovery)
   - [Starter Dataset](#starter-dataset)
   - [Baseline vs Learned Model](#baseline-vs-learned-model)
   - [My Additional Analysis](#my-additional-analysis)
5. [Week 2 — Problem and ML Task Framing](#week-2--problem-and-ml-task-framing)
   - [ML-02: Research Question](#ml-02-research-question)
   - [ML-03: ML Task Framing](#ml-03-ml-task-framing)
   - [Decision and User of the Output](#decision-and-user-of-the-output)
   - [Target and Label](#target-and-label)
   - [Evaluation Metric](#evaluation-metric)
   - [Error Costs](#error-costs)
6. [Week 3 — Data Contract and Feature/Leakage Validation](#week-3--data-contract-and-featureleakage-validation)
   - [ML-04: Data Contract](#ml-04-data-contract)
   - [Warehouse Access](#warehouse-access)
   - [Data Grain Verification](#data-grain-verification)
   - [Feature, Label, Context and Excluded Fields](#feature-label-context-and-excluded-fields)
   - [ML-05: Feature Vector](#ml-05-feature-vector)
   - [Missingness and Availability Checks](#missingness-and-availability-checks)
   - [Leakage Experiment](#leakage-experiment)
   - [ROC-AUC Results](#roc-auc-results)
   - [Final Leakage Removal](#final-leakage-removal)
7. [Features Used and Why](#features-used-and-why)
8. [Important Data Gotchas Identified](#important-data-gotchas-identified)
9. [What Was Required vs What I Added](#what-was-required-vs-what-i-added)
10. [Key Results](#key-results)
11. [Notebook Structure](#notebook-structure)
12. [Repository Structure](#repository-structure)
13. [Current Status](#current-status)
14. [Lessons Learned](#lessons-learned)
15. [Next Steps](#next-steps)


---

# Internship Objective

The internship follows a decision-first approach to machine learning.

The main principle I followed is:

> **A model is not the final goal. A better decision is the goal.**

Instead of starting directly with model training, I worked through:

- Understanding the dataset.
- Understanding the business problem.
- Selecting an appropriate ML lane.
- Defining the decision the model should support.
- Defining who will use the output.
- Identifying the cost of incorrect predictions.
- Defining the ML task.
- Defining the target/label.
- Selecting an evaluation metric.
- Creating a data contract.
- Checking data availability.
- Building a feature vector.
- Identifying missing values.
- Checking temporal availability.
- Deliberately testing for data leakage.
- Removing leaked information.
- Verifying the final feature matrix.


---

# Selected Lane

## Lane 2 — Refresh / Content Opportunity Scoring

I selected **Lane 2: Refresh / Content Opportunity Scoring**.

The objective is to rank webpages based on their apparent priority for human review.

### Business question

> **Which pages should a content/SEO team investigate first?**

### Model output

The model should produce a score representing the priority of each page.

Pages can then be sorted from highest priority to lowest priority.

### Human action

The SEO/content team uses the ranked queue to decide which pages deserve investigation or refresh consideration.

### What the system does NOT claim

The system does not:

- Automatically rewrite webpages.
- Guarantee that refreshing a page will improve traffic.
- Claim to predict Google's proprietary ranking algorithm.
- Replace human content decisions.

The output is treated as **decision support**.


---

# Overall Progress

| Week | Work | Status | Main Outcome |
|---|---|---|---|
| Week 1 | ML-01 — First Look and Discovery | Completed | Understood dataset, baseline, learned model and initial patterns |
| Week 2 | ML-02 — Research Question | Completed | Selected Lane 2 and defined the business decision |
| Week 2 | ML-03 — ML Task Framing | Completed | Defined task type, target, metric, errors and unit of analysis |
| Week 3 | ML-04 — Data Contract | Completed | Defined data grain, time window, fields and verification checks |
| Week 3 | ML-05 — Feature Vector & Leakage Check | Completed / Updated | Built features, checked missingness and demonstrated/remediated leakage |

---

# Week 1 — Discovery and First ML Results

## ML-01: First Look and Discovery

The first stage was designed to provide a fast understanding of the complete ML workflow using the starter dataset.

I executed the starter notebook and studied:

- Dataset structure.
- Number of rows.
- Available features.
- Target/label behaviour.
- Baseline rule.
- Learned model.
- Model evaluation.
- Client-level holdout.
- Initial business interpretation.

The starter dataset contains approximately **30,000 rows** representing content items across multiple clients.


## Starter Dataset

The starter dataset contains:

- **30,000 rows**
- **44 columns**
- **32 clients**
- One row per pseudonymized content item.
- Trailing 90-day performance information.

Important groups of columns include:

- Keyword information.
- Content properties.
- Search performance.
- Analytics activity.
- Recent-period comparisons.
- Derived rates.
- Content-age/freshness information.
- Trend information.

The dataset is anonymized, so identifiers such as `content_id` and `client_id` are treated as identifiers rather than predictive features.


## Baseline vs Learned Model

The initial experiment compared a simple rule-based baseline with a learned Random Forest model.

### Baseline

The baseline Precision@50 was:

> **0.240**

### Random Forest

The learned Random Forest achieved approximately:

> **0.740 Precision@50**

This showed that the available features contained useful signal beyond the initial rule-based approach.

The learned model achieved roughly:

> **3.1× the baseline Precision@50**

The evaluation used a **client-holdout split**, which is important because the model should be evaluated on clients that were not used for training.


## My Additional Analysis

In addition to the starter notebook requirements, I performed my own analysis using:

- Content age.
- Trend direction.
- Content type.

I compared the percentage of pages showing decline across different content-age groups and content types.

### Content Age — Comparison

For comparison content:

| Content Age | Pages | % Declining |
|---|---:|---:|
| 91–180 days | 378 | 63.76% |
| 181–365 days | 319 | 49.53% |

There were no observations for the 0–90 or 365+ groups in this subset.

### Content Age — Feedly

| Content Age | Pages | % Declining |
|---|---:|---:|
| 0–90 days | 11 | 54.55% |
| 91–180 days | 617 | 21.56% |
| 181–365 days | 1,468 | 31.47% |

The 0–90 day Feedly group is very small, so it should not be interpreted strongly.

### Content Age — Keyword

| Content Age | Pages | % Declining |
|---|---:|---:|
| 0–90 days | 481 | 67.15% |
| 91–180 days | 10,785 | 64.86% |
| 181–365 days | 9,581 | 54.62% |
| 365+ days | 6,360 | 42.63% |

### Observation

The analysis suggested an association between content age and observed decline, particularly for keyword content.

However, I treated this as an **observed association rather than a causal relationship**.

I did not claim that older content causes traffic decline.


---

# Week 2 — Problem and ML Task Framing

# ML-02: Research Question

I selected **Lane 2 — Refresh / Content Opportunity Scoring**.

The research/business question was framed around prioritization rather than simply predicting a yes/no outcome.

### Research question

> **Can historical content, search, and performance signals be used to rank webpages according to their priority for human content review?**

The important distinction is that:

- Prediction is an intermediate ML operation.
- Ranking is the business output.
- Human review is the final decision.


# ML-03: ML Task Framing

The task was framed as:

> **Supervised priority scoring / ranking**

The underlying learning problem uses a binary classification-style target, but the business output is a ranking.

The model produces a score/probability for each page, and pages are ranked using that score.

### Task mapping

| Component | Definition |
|---|---|
| Business decision | Which page should be reviewed first? |
| Output | Priority score / ranked queue |
| ML formulation | Supervised scoring/ranking |
| Target | Declining-page proxy |
| Main evaluation | Precision@50 |
| Human action | Review high-priority pages |

The internship guidance emphasizes defining the decision before selecting the model, and matching the ML task to the actual business question.


# Decision and User of the Output

### Decision

The decision is:

> **Which webpages should receive human content review first?**

### User

The expected user is:

- SEO team.
- Content team.
- Human reviewer/editor.

### Action

The user can inspect the highest-ranked pages and decide whether a refresh or investigation is appropriate.


# Target and Label

For the starter/proxy setup, I used:

```python
is_declining_label = (trend_direction == "down").astype(int)
