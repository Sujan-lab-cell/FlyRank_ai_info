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


Therefore:

1 = page is classified as declining.
0 = page is not classified as declining.

The starter dataset contains:

16,262 declining pages
13,738 non-declining pages
30,000 total pages

The observed declining base rate is approximately:

54.2%

Important leakage rule

Because:

is_declining_label
        ↓
trend_direction
        ↓
trend_pct

the following fields must not be used as model features:

trend_direction
trend_pct

They contain information directly related to the target definition.

Evaluation Metric

The primary metric selected for the Lane 2 problem is:

Precision@50

This measures how many of the top 50 ranked pages are actually relevant/declining according to the chosen label.

This is more closely aligned with the business situation than simply maximizing overall classification accuracy because the content team has limited review capacity.

The objective is therefore:

Put the most useful pages near the top of the review queue.

Error Costs

Two major error types were identified.

False Positive

A healthy page is incorrectly ranked as needing review.

Cost:

Wasted editor time.
Unnecessary investigation.
Lower review efficiency.
False Negative

A genuinely declining/high-opportunity page is not ranked highly enough.

Cost:

Important opportunity may be missed.
Potentially valuable content may not receive timely review.

This makes ranking quality important because the team cannot manually investigate every page.

Week 3 — Data Contract and Feature/Leakage Validation

Week 3 focused on making the data and feature construction defensible before moving deeper into modeling.

ML-04: Data Contract

The data contract defines exactly:

What one row represents.
Which data sources are used.
Which time window is used.
What is being predicted/ranked.
Which fields are features.
Which fields are context only.
Which fields are deliberately excluded.
Unit of Analysis

For the business unit:

One unique content item (content_id) belonging to one client (client_id).

For the warehouse's raw daily performance table, the grain is:

report_date × client_id × content_id

The starter CSV is a snapshot containing one row per content item.

Development Time Window

For warehouse development, I followed the recommendation to work with a mid-panel month, such as:

March 2026

The final month was treated as a sealed/future evaluation period rather than being used casually during development.

This prevents accidentally developing against the future outcome window.

Warehouse Access

I successfully connected to the gated FlyRank warehouse using:

Hugging Face dataset access.
HF_TOKEN stored through Colab Secrets.
DuckDB.
Parquet files.

I did not hardcode the Hugging Face token into the notebook.

The successful warehouse setup returned:

Table	Rows
dim_clients	104
dim_content	519,606
fact_content_daily_performance_sample	11,694,072
fact_content_query_90d	2,414,248

The full daily performance table contains approximately 78.8 million rows.

The warehouse covers approximately:

2025-01-27 to 2026-06-30

The final-month sample represents June 2026 and should not be used for developing a past-to-future label.

The warehouse documentation specifically warns about history depth, GA4 availability, repeated query-table context, and time-window overlap.

Data Grain Verification

I verified the starter dataset grain.

The starter dataset contained:

30,000 total rows
30,000 unique content-client pairs
0 duplicate content-client pairs

This supports the intended starter-data grain of one content item for one client.

Feature, Label, Context and Excluded Fields
Features

Candidate predictive features included:

search_volume
competition
cpc
word_count
char_count
impressions_90d
clicks_90d
pageviews_90d
sessions_90d
engaged_sessions_90d
content_age_days
days_since_last_update
ctr
avg_position
engagement_rate
scroll_rate

For the focused leakage experiment, I used a smaller six-feature set.

Label
is_declining_label

The label was kept separate from model features.

Context

Contextual fields include:

content_id
client_id
content_type
main_intent

Identifiers are useful for:

Grouping.
Joining.
Client-level splitting.

They are not treated as predictive features.

Explicitly Excluded Fields

The following fields were deliberately excluded:

trend_direction
trend_pct
impressions_last_30d
clicks_last_30d
impressions_prev_30d
clicks_prev_30d
content_id
client_id
Why?

trend_direction and trend_pct are directly connected to the label and therefore create target leakage.

The recent-period comparison fields can overlap with or contribute to the outcome measurement window, so they require strict temporal alignment before being considered valid predictive features.

content_id and client_id are identifiers and should not be used as predictive model inputs.

ML-05: Feature Vector

The feature-vector stage converts the raw dataset into the input representation used by the ML model.

Conceptually:

Raw Dataset
     |
     v
Feature Selection
     |
     v
Missingness Handling
     |
     v
Categorical Encoding
     |
     v
Feature Matrix X
     |
     +---- Target y

The feature matrix is represented as:

X = model features
y = target label

The model learns relationships between X and y.

The target is not included inside X.

Missingness and Availability Checks

I explicitly investigated missing values instead of blindly replacing every missing value.

The missingness check produced:

Feature	Missing values
search_volume	2,468
word_count	7,699
cpc	2,468
avg_position	0

I also checked the special value:

avg_position == 0

There were:

1,205 rows

where avg_position was zero.

According to the dataset documentation, avg_position = 0 means no position data, not that the page ranked at position zero.

This is why a missingness/availability flag such as:

has_pos_data = (avg_position > 0).astype(int)

is useful.

Missingness Flags

I created availability indicators including:

has_word_count
has_search_volume
has_pos_data

These flags allow the model to distinguish:

"The value is actually zero"

from:

"The value was unavailable."

Focused Six-Feature Set

For the explicit leakage experiment, I used the following six features:

impressions_90d
ctr
avg_position
days_since_last_update
has_word_count
has_search_volume

These were selected as a compact set covering different types of information:

Feature	Signal
impressions_90d	Search exposure
ctr	Click-through efficiency
avg_position	Search ranking context
days_since_last_update	Content freshness
has_word_count	Content-data availability
has_search_volume	Keyword-data availability

These six were treated as a focused experimental feature set, not as a claim that they are the final or universally best features.

Leakage Experiment

One of the most important additional checks I performed was a deliberate leakage experiment.

What is data leakage?

Data leakage happens when information that would not legitimately be available at prediction time is given to the model.

This can make a model appear extremely accurate while actually being invalid.

There are two important forms considered here:

Target Leakage

The feature directly or indirectly contains the answer.

Temporal Leakage

A feature contains information from the future or from a period overlapping the outcome being predicted.

Deliberate Target Leakage

I intentionally created a leaked feature:

X_leak["leak_feature"] = y

Here:

leak_feature = is_declining_label

The model was therefore given the answer directly.

This was intentionally done to demonstrate how leakage artificially improves evaluation results.

ROC-AUC Results

The honest feature set produced:

ROC-AUC = 0.695

The deliberately leaked feature set produced:

ROC-AUC = 1.000

This was expected.

Honest model
ROC-AUC: 0.695

This indicates that the six selected features contain meaningful signal for distinguishing the two target classes.

Leaked model
ROC-AUC: 1.000

This does NOT mean that the real model achieved perfect performance.

It demonstrates that the evaluation becomes artificially perfect when the target is directly leaked into the features.

Understanding ROC-AUC

ROC-AUC evaluates how well a model ranks positive examples above negative examples across different classification thresholds.

A rough interpretation is:

ROC-AUC	General interpretation
0.50	Random-like ranking
0.60	Weak signal
~0.70	Useful signal
0.80+	Stronger discrimination
1.00	Perfect separation

ROC-AUC is different from accuracy.

It evaluates the quality of the model's ranking/discrimination across thresholds.

The main purpose of the Week 3 experiment was not to maximize ROC-AUC, but to demonstrate that leakage can create an unrealistically high score.

Final Leakage Removal

After the leakage experiment, the deliberately leaked field was removed.

The final verification checked:

"leak_feature" in X_clean.columns

The result was:

False

The final feature matrix therefore did not contain the deliberately leaked feature.

I also checked the final feature matrix against the excluded-field list.

The result was:

Total features in final matrix: 23
Excluded fields present in feature matrix count: 0

The assertion passed.

This provides an explicit verification that the excluded fields were not accidentally included in the final model matrix.

Important Data Gotchas Identified

During Weeks 1–3, I identified several important properties of the dataset that affect ML validity.

1. Rate columns are percentages multiplied by 100

For example:

ctr = 0.76

means:

0.76% CTR

and not 76%.

This applies to fields such as:

ctr
engagement_rate
scroll_rate
ai_traffic_pct
trend_pct
2. avg_position = 0 means no data

It does not mean:

rank = 0

Therefore position availability must be handled explicitly.

3. Missingness is informative

Missing values can depend on content_type.

Blindly doing:

fillna(0)

can accidentally turn missingness into a hidden category signal.

Therefore I introduced availability flags where appropriate.

4. IDs are not features

The following are identifiers:

content_id
client_id

They are useful for:

Grouping.
Joining.
Splitting.

They should not be used as predictive features.

5. Trend fields create leakage

Because:

trend_pct
     ↓
trend_direction
     ↓
is_declining_label

the trend fields must remain outside the feature matrix.

6. Recent-period features require time alignment

Fields such as:

impressions_last_30d
clicks_last_30d
impressions_prev_30d
clicks_prev_30d

are not automatically invalid.

However, they must be checked against the prediction date and outcome window.

A feature is only valid if it would actually be known at the time the decision is made.

7. The warehouse is an unbalanced panel

Different clients have different amounts of historical data.

Therefore global calendar windows cannot automatically be assumed to be valid for every client.

Client-level history availability must be checked.

8. GA4 availability must be checked explicitly

Some rows occur before a client's GA4 data became available.

Those rows may contain zero-filled GA4 values.

Therefore:

ga4_data_available

should be used rather than assuming zero means no engagement.

9. Query-level context should not be blindly summed

The query table contains per-content context repeated across query rows.

Repeated contextual values should be handled appropriately, for example using:

ANY_VALUE()

when appropriate, rather than incorrectly summing them.

What Was Required vs What I Added

One of the main goals of this repository is to clearly distinguish the internship requirements from additional work I performed.

Week 1
Required
Run the starter notebook.
Understand the dataset.
Run the baseline.
Run the learned model.
Compare results.
Make an initial discovery.
Additional work completed
Performed a custom analysis of content age.
Compared decline rates across content types.
Investigated different age tiers.
Reported sample-size limitations.
Avoided interpreting correlation as causation.
Documented the business meaning of the observed patterns.
Week 2
Required
Select a lane.
Write a research question.
Define the decision.
Define who acts on the output.
Identify error costs.
Define the ML task.
Define the target.
Define the evaluation metric.
Additional work completed
Clearly separated the business ranking objective from the underlying classification formulation.
Explained false-positive and false-negative costs.
Documented why Precision@50 is more relevant to the review-queue use case.
Distinguished scoring/ranking from simple yes/no prediction.
Documented what the model will and will not claim.
Week 3
Required
Create a data contract.
Define the row grain.
Identify the relevant tables.
Define the time window.
Define the target.
Identify features and excluded fields.
Verify data properties with queries.
Build a feature vector.
Check feature availability.
Investigate leakage.
Remove leaked information.
Additional work completed
Successfully connected to the full Hugging Face warehouse through DuckDB.
Verified warehouse table row counts.
Investigated missingness across important fields.
Explicitly checked avg_position == 0.
Created missingness/availability flags.
Documented why individual features were selected.
Performed a deliberate leakage experiment.
Measured honest ROC-AUC.
Measured leaked ROC-AUC.
Demonstrated artificial performance inflation caused by leakage.
Removed the leaked feature.
Programmatically verified that the leaked feature was gone.
Programmatically verified that excluded fields were absent from the final feature matrix.
Built a final feature matrix with numeric features, availability flags and categorical encoding.
Documented important dataset gotchas and temporal-alignment concerns.
Key Results
Week 1
Result	Value
Starter rows	30,000
Baseline Precision@50	0.240
Random Forest Precision@50	0.740
Approximate improvement	3.1×
Week 2
Component	Result
Selected lane	Lane 2
Problem	Refresh / Content Opportunity Scoring
Main output	Ranked review queue
Primary metric	Precision@50
Starter declining pages	16,262
Starter non-declining pages	13,738
Declining base rate	~54.2%
Week 3
Experiment	ROC-AUC
Honest six-feature set	0.695
Deliberately leaked feature	1.000

The 1.000 score is intentionally invalid and demonstrates target leakage.

The honest score of 0.695 is the meaningful result from this particular six-feature experiment.

Notebook Structure
Week 1
ML-01
notebooks/
└── 01_first_look_and_discovery.ipynb

Purpose:

Dataset discovery.
Baseline.
First learned model.
Initial analysis.
Week 2
ML-02
work/notebooks/
└── w01_research_question.ipynb

Purpose:

Research question.
Lane selection.
Business decision.
Decision-maker.
Error costs.
Initial ML framing.
ML-03
work/notebooks/
└── w02_ml_task_framing.ipynb

Purpose:

ML task definition.
Target.
Metric.
Unit of analysis.
Ranking/scoring formulation.
Week 3
ML-04
work/notebooks/
└── w03_data_contract.ipynb

Purpose:

Data contract.
Data grain.
Warehouse verification.
Feature/label/context definitions.
Time-window alignment.
Data availability.
ML-05

Feature-vector and leakage-validation work covering:

Feature construction.
Missingness checks.
Availability flags.
Honest feature evaluation.
Deliberate leakage.
Leakage removal.
Final feature-matrix verification.
Repository Structure

The project is organized around the internship workflow.

flyrank-ml-internship/
│
├── data/
│   └── raw/
│       └── content_refresh_anonymized.csv
│
├── notebooks/
│   ├── 01_first_look_and_discovery.ipynb
│   └── 02_your_first_readable_model.ipynb
│
├── work/
│   └── notebooks/
│       ├── w01_research_question.ipynb
│       ├── w02_ml_task_framing.ipynb
│       └── w03_data_contract.ipynb
│
├── scripts/
│
├── docs/
│   ├── data-dictionary.md
│   └── ml-core-foundation-framework.md
│
└── README.md
Current Status
Completed
 Week 1 discovery work.
 Starter dataset exploration.
 Baseline evaluation.
 Learned-model comparison.
 Custom content-age analysis.
 Content-type analysis.
 Lane selection.
 Research question.
 Business decision definition.
 User/action definition.
 Error-cost analysis.
 ML task framing.
 Target definition.
 Metric definition.
 Data contract.
 Data-grain verification.
 Warehouse access.
 Feature selection.
 Missingness analysis.
 Availability checks.
 Feature-vector construction.
 Leakage experiment.
 Honest ROC-AUC evaluation.
 Deliberate leaked ROC-AUC evaluation.
 Leakage removal.
 Final excluded-feature verification.
 Week 3 notebook updates and resubmission preparation.
Lessons Learned
1. Start with the decision, not the model

A machine learning model is useful only when its output changes or improves a real decision.

For this project, the important question is not simply:

"Can I predict decline?"

It is:

"Which pages should the content team investigate first?"

2. Ranking changes how success is measured

If a human team can only review 50 pages, the top of the ranking matters much more than the performance across every page.

That is why Precision@50 is important for the Lane 2 business objective.

3. Data leakage can make a bad model look perfect

The deliberate leakage experiment demonstrated this clearly.

Honest model → ROC-AUC 0.695
Leaked model → ROC-AUC 1.000

The perfect result was caused by giving the model information that directly represented the target.

Therefore:

A higher metric is not automatically a better model.

The first question must be:

"Is the evaluation honest?"

4. Missing values contain information

A missing value is not always equivalent to zero.

The reason why the value is missing can itself provide information.

This motivated the use of availability flags such as:

has_word_count
has_search_volume
has_pos_data
5. Time matters in ML

A feature can be statistically useful but still be invalid if it would not have been available when the decision was made.

Therefore every feature needs a clear answer to:

"Would I know this information at the prediction/decision moment?"

6. IDs should not become accidental features

Identifiers can allow a model to memorize clients or pages instead of learning generalizable patterns.

Therefore:

content_id
client_id

are used for grouping, joining and splitting rather than prediction.

Next Steps

The foundation established during Weeks 1–3 will be used for the following stages of the internship.

The next stages should build on the same:

Business decision.
Data contract.
Target definition.
Feature-availability rules.
Leakage rules.
Evaluation metric.
Client-aware validation strategy.

Future work should focus on:

Signal auditing.
Building a transparent rule-based baseline.
Creating a ranked review queue.
Training and comparing honest ML models.
Evaluating the model against the frozen baseline.
Performing robust validation.
Measuring whether the learned ranking provides useful decision support.
Documenting limitations and defensible claims.
Final Summary

During the first three weeks of the FlyRank ML Internship, I progressed from basic dataset discovery to a clearly defined ML problem and an explicitly validated feature pipeline.

The work followed this progression:

Dataset Discovery
       ↓
Initial ML Experiment
       ↓
Business Problem Selection
       ↓
Lane 2: Content Opportunity Scoring
       ↓
ML Task Framing
       ↓
Data Contract
       ↓
Feature Selection
       ↓
Missingness & Availability Checks
       ↓
Leakage Experiment
       ↓
Leakage Removal
       ↓
Final Feature Verification

The most important outcome is not simply a model score.

The main outcome is a defensible ML foundation in which:

The business decision is defined.
The model's user is defined.
The target is explicit.
The evaluation metric is explicit.
Features are documented.
Missingness is investigated.
Temporal availability is considered.
Identifiers are excluded from prediction.
Target-derived fields are excluded.
Leakage is deliberately tested.
Artificially inflated performance is demonstrated and rejected.
The final feature matrix is programmatically checked.

This provides the foundation for the next stages of the internship, where the focus can move from "Can I build a model?" to:

"Can I build an honest model that improves the content-review decision?"


### One important note

I deliberately wrote this README so your **extra work is visible**, rather than making it look like you only completed the mandatory notebook cells. In particular, your **warehouse connection, missingness investigation, `avg_position` check, deliberate leakage experiment, ROC-AUC comparison, and automated excluded-feature verification** are clearly identified as additional work.

Also, I have **not claimed that ML-05 is a separate official Week-3 notebook name** where the available assignment material doesn't explicitly establish that; I've described it as your Week 3 feature/leakage work. The internship's skill index does explicitly associate ML-05 with the leakage/warehouse work. :contentReference[oaicite:3]{index=3}

Increase memory for more relevant answers
Upgrade to expand the amount of detail ChatGPT can bring into responses from your saved files and past conversations.
Upgrade to Plus

it 
