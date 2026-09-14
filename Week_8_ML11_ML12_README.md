# Week 8 — ML-11 & ML-12: Ship the Paper + Tell the Story

This README is a personal reference for the final week of the FlyRank Machine Learning Internship.

## 1. Week 8 Overview

Week 8 had two assignments:

- **ML-11 — Ship the Paper**
- **ML-12 — Tell the Story**

The purpose was to turn the ML work completed during the internship into a public research paper and then communicate the same work clearly to different audiences.

The project is a **decision-support workflow** for prioritizing webpages that may be experiencing search-performance decline.

> **The model recommends where to look. A human decides what to do.**

---

# ML-11 — Ship the Paper

## 2. Project Question

> **Which webpages should an SEO/content team review first when many pages may be experiencing search-performance decline?**

The practical problem is that a content team may have thousands of webpages but limited time and people to review them.

So the ML task became a **prioritization problem**:

```text
Many webpages
     ↓
Historical search-performance data
     ↓
ML ranking model
     ↓
Rank potentially declining pages
     ↓
Human review
     ↓
Human decides what action to take
```

The system is **not** an automatic content optimizer.

---

## 3. Paper Structure

The deployed ML-11 paper contains the required sections:

1. Title + Abstract
2. Introduction / Problem Statement
3. Data
4. Methodology
5. Results
6. Limitations & Honest Framing
7. Ranked Recommendations
8. Reproducibility
9. Acknowledgments & Data Credit

Public paper:

https://sujan-lab-cell.github.io/flyrank-ml-internship/

Paper source:

```text
docs/index.html
```

Figures:

```text
docs/figures/
```

---

## 4. Abstract — Main Story

The paper asks whether historical search-performance signals can be used to rank webpages for human review of a defined May 2026 click-decline outcome.

The experiment used:

- **16,513 eligible pages**
- **36 clients**
- Historical data from **February–April 2026**
- A **May 2026** outcome period
- A **Random Forest classifier**
- **5-fold GroupKFold validation by client**
- A simple historical baseline

Main result:

```text
Random Forest Precision@50 = 0.444
Baseline Precision@50      = 0.392
Improvement                 = +5.2 percentage points
```

This provides **directional evidence** that machine learning can help prioritize pages for human review.

It does not prove causation, guaranteed traffic recovery, or automatic content improvement.

---

## 5. FlyRank Content Problem

The useful operational question is not only:

> "Did a page's performance decline?"

It is:

> **"Which pages should the content team look at first?"**

The model creates a ranked list so an SEO/content team can focus limited review time on higher-ranked pages.

A high-ranked page means:

> "This page may deserve attention."

It does **not** mean:

> "This page definitely has a problem."

And it does not mean:

> "Refresh this page immediately."

A human must investigate the page.

---

# 6. Data

## Data source

The project used the **FlyRank full warehouse release**.

Main sources:

- `fact_daily` — daily search-performance records
- `dim_content` — content-page information
- `dim_clients` — client/group information

## Date windows

### Feature/training history

```text
February 1 – April 30, 2026
```

### Outcome period

```text
May 1 – May 31, 2026
```

May data was used only to create the outcome label and was not used as a predictive feature.

---

## 7. Eligible Population

A page was included only when:

```text
impressions_total >= 1000
AND
april_clicks >= 10
```

Final population:

```text
16,513 pages
36 clients
```

---

# 8. Target Definition

A page was labeled as declining when May clicks were less than 80% of April clicks:

```python
decline = (may_clicks < 0.8 * april_clicks).astype(int)
```

Therefore:

```text
1 → declining
0 → not declining
```

This is a **defined experimental target**, not a claim that every positive page has a real-world SEO problem.

---

# 9. Model Features

The Random Forest used nine historical features:

1. `impressions_total`
2. `clicks_total`
3. `april_impressions`
4. `april_clicks`
5. `feb_clicks`
6. `momentum`
7. `CTR`
8. `active_days`
9. `weighted_position`

All predictive information came from before the May outcome period.

---

# 10. Excluded Information

### Future May metrics

Excluded because they would leak information from the outcome period.

### Target-related fields

`may_clicks` was used to create the target but was not used as a model feature.

### IDs

Client/content IDs were used for:

- grouping
- joining
- deterministic tie-breaking

They were not predictive features.

### Sparse signals

Sparse session, scroll, and AI-referral signals were excluded because their coverage was limited for the final modeling population.

---

# 11. Baseline

The simple baseline flagged a page when:

```text
april_clicks < march_clicks
```

Flagged pages were ranked using April impressions.

The purpose was to test whether the Random Forest added useful ranking signal beyond a simple historical rule.

---

# 12. Random Forest

The main ML model was a:

> **Random Forest classifier**

The model generated scores that were used to rank webpages.

The practical output was therefore:

```text
Which pages should be reviewed first?
```

rather than simply:

```text
Declining / Not declining
```

---

# 13. Validation

The final evaluation used:

> **5-fold GroupKFold by client**

Pages from the same client were kept together within a fold.

This means the same client did not appear in both training and validation for a fold.

The purpose was to test whether the approach could generalize beyond the clients used for training.

```text
Client overlap between train/validation = 0
```

---

# 14. Why Grouped Validation Matters

Earlier random KFold evaluation produced:

```text
P@50 = 0.996
```

After using client-grouped validation:

```text
Random Forest P@50 = 0.444
```

The difference was:

```text
-55.2 percentage points
```

The correct interpretation is that random page-level splitting allowed client overlap and gave an easier evaluation.

Client-grouped validation provides a more honest test of generalization to unseen clients.

We did **not** claim that the exact gap proves memorization, because the experiment alone does not establish that specific mechanism.

---

# 15. Leakage Audit

The Week 6 leakage audit found:

```text
Predictive features = 9
Future/target fields leaking into X = 0
IDs used as predictive features = 0
```

Feature cutoff:

```text
April 30, 2026
```

All predictive features ended at or before the cutoff.

May data was used only for the target.

**Leakage audit verdict: PASSED**

---

# 16. Precision@K

The main evaluation metric was:

> **Precision@K**

It asks:

> Among the top K pages ranked by the model, how many are actually positive according to the defined target?

For example:

```text
Precision@50 = 0.444
```

means that about 44.4% of the top 50 ranked pages were positive according to the defined May decline label.

It does **not** mean:

- 44.4% traffic recovery
- 44.4% SEO improvement
- 44.4% guaranteed decline
- 44.4% probability of decline

It is a ranking precision metric.

---

# 17. Final Results

| Model | P@10 | P@20 | P@50 | P@100 |
|---|---:|---:|---:|---:|
| Baseline | 0.400 | 0.370 | 0.392 | 0.388 |
| Logistic Regression | 0.380 | 0.370 | 0.424 | 0.438 |
| Decision Tree | 0.400 | 0.390 | 0.324 | 0.346 |
| **Random Forest** | **0.460** | **0.430** | **0.444** | **0.448** |

Key comparison:

```text
Random Forest P@50 = 0.444
Baseline P@50      = 0.392
```

Improvement:

```text
+5.2 percentage points
```

---

# 18. Honest Interpretation

The safest claim is:

> Under 5-fold client-grouped validation, the Random Forest measured a Precision@50 of 0.444 for the defined May 2026 decline target.

This provides:

> **Directional evidence that the model can help rank potentially declining pages for human review.**

It does not prove:

- causal SEO effects
- guaranteed traffic recovery
- Google algorithm behavior
- that every high-score page needs refreshing
- that the model will perform equally well on future clients

---

# 19. Action Playbook

The model output was converted into a content action playbook.

The intended workflow is:

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

The model does not automatically edit or publish content.

---

## Action Categories

### Model signal only

```text
723 pages
```

Reason:

```text
model_signal_only
```

Action:

```text
manual_investigation
```

Priority:

```text
Medium
```

---

### Decline + Low Visibility

```text
249 pages
```

Reason:

```text
decline_and_low_visibility
```

Action:

```text
seo_content_review
```

Priority:

```text
High
```

The low-visibility rule used:

```text
weighted_position > 15
```

This is described as **weak search visibility**, rather than making an overly broad claim about exact Google page position.

---

### Monitor

```text
15,541 pages
```

Reason:

```text
monitor
```

Action:

```text
monitor
```

Priority:

```text
Monitor
```

---

# 20. Important Staleness Limitation

The final action queue did **not** use a `decline_and_stale` category.

Why?

Because:

> `days_since_last_update` was not available in the final daily-performance feature dataset used for this queue.

We therefore did not invent a staleness signal or use momentum as a proxy for staleness.

This is an important project principle:

> **Do not invent a signal just because it would make the action playbook look better.**

---

# 21. Staleness Descriptive Finding

A separate analysis found:

```text
Not stale (<91 days)
20,655 pages
Decline rate = 51.20%

Stale (>=91 days)
9,345 pages
Decline rate = 60.85%
```

Difference:

```text
+9.65 percentage points
```

The stale group had a higher observed decline rate.

However:

> **This is descriptive association, not causal evidence.**

We cannot conclude that refreshing a page will cause its traffic to recover.

---

# 22. Human Review Rules

Reviewers should check:

- Content quality
- Content relevance
- Search performance
- Search visibility
- Business importance
- Supporting evidence

The system should **not** automatically:

- edit content
- publish content
- delete pages
- redirect pages
- change titles
- change meta descriptions
- change H1s
- change canonicals
- change internal links

A high model score alone is not enough to justify a content change.

---

# 23. Monitoring and Retraining

The governance approach is:

> **Monitor first, investigate second, revalidate before retraining.**

The P@50 result of:

```text
0.444
```

is a reference benchmark, not an SLA.

Possible reasons to consider retraining:

- persistent performance degradation
- sustained feature-distribution shifts
- changes in client population
- changes in data definitions
- pipeline changes
- loss of practical utility

A single unusual prediction should not automatically trigger retraining.

There should be no automatic:

- retraining
- deployment
- threshold changes
- action mapping changes
- website changes

---

# 24. Public-Safety

The public paper does not expose:

- client names
- client domains
- private URLs
- private search queries
- credentials
- raw production exports

Only aggregate/public-safe information is reported.

---

# 25. ML-11 Deployment Problem and Fix

The first GitHub Pages deployments failed with:

```text
No url found for submodule path 'Frontend' in .gitmodules
```

Investigation showed that `Frontend` was tracked as a Git submodule/gitlink:

```text
mode 160000
```

but the repository had no valid `.gitmodules` configuration.

The stale Git tracking entry was removed with:

```bash
git rm --cached Frontend
```

This removed the broken Git tracking reference while keeping the actual local `Frontend` directory.

Fix commit:

```text
36ba366e3b4425594c501e355f4795e44f6df489
```

Message:

```text
Remove stale Frontend submodule reference
```

GitHub Pages then deployed successfully.

---

# ML-12 — Tell the Story

## 26. Purpose

ML-12 was about communicating the completed work.

It required:

1. Connecting the paper's findings to the FlyRank content problem.
2. Adding a 5-minute demo outline to the final notebook.
3. Adding a short social post.
4. Adding a 3-sentence employer-facing summary.

ML-12 was **not a new ML experiment**.

---

# 27. 5-Minute Demo Outline

The final notebook contains:

### Question

Which webpages should an SEO/content team review first when many pages may be experiencing search-performance decline?

### Method

Use:

- Historical search-performance signals
- February–April 2026 features
- May 2026 decline target
- Random Forest
- 5-fold GroupKFold by client

Population:

```text
16,513 pages
36 clients
```

### One Chart

```text
work/figures/ml_precision_at_k_comparison.png
```

This compares Precision@K for the baseline and Random Forest.

### One Honest Result

```text
Random Forest P@50 = 0.444
Baseline P@50      = 0.392
```

Improvement:

```text
+5.2 percentage points
```

### One Recommendation

> Use the model as a decision-support prioritization tool for human review rather than autonomous content optimization.

---

# 28. Social Post

The notebook contains a shareable post explaining:

- the content prioritization problem
- historical search-performance data
- Random Forest
- client-grouped validation
- P@50 = 0.444 vs 0.392 baseline
- human-in-the-loop decision support

Core message:

> **The model helps identify where to look, while human review determines what action makes sense.**

---

# 29. Employer-Facing Summary

The notebook contains exactly three sentences covering:

1. **What I built:** a Random Forest-based content prioritization workflow.
2. **What data I used:** 16,513 eligible pages across 36 clients.
3. **What it showed:** P@50 = 0.444 versus 0.392 for the baseline, followed by conversion of the model output into a human-review action playbook.

---

# 30. Where ML-12 Lives

The ML-12 content was added to:

```text
work/notebooks/capstone.ipynb
```

The final structure is:

```text
Capstone
│
├── Abstract
├── Introduction / Problem Statement
├── 1. Question
├── 2. Data
├── 3. Methodology
├── 4. Results
├── 5. Limitations & Honest Framing
├── 6. Ranked Recommendations
├── 7. Artifacts the Paper Embeds
│
├── 8. ML-12 — Tell the Story
│   ├── 5-Minute Demo Outline
│   ├── Short Social Post
│   └── Employer-Facing Summary
│
└── Self-check
```

---

# 31. ML-12 Commit

ML-12 was committed and pushed with:

```text
071d8f871f86a20ccc8f3c503d993440af6f6973
```

Commit message:

```text
Complete ML-12 Tell the Story
```

Final Git status:

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

---

# 32. Submission

## ML-11

Paper URL:

```text
https://sujan-lab-cell.github.io/flyrank-ml-internship/
```

## ML-12

Repository URL:

```text
https://github.com/Sujan-lab-cell/flyrank-ml-internship
```

Direct capstone notebook link:

```text
https://github.com/Sujan-lab-cell/flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb
```

The notebook link can be provided as an additional deliverable link if the submission form allows multiple links.

---

# 33. Interview Explanation

If asked:

> **"Tell me about your FlyRank project."**

Answer:

> I worked on a machine-learning workflow to help SEO and content teams prioritize webpages that may be experiencing search-performance decline. I used historical search-performance data from 16,513 eligible pages across 36 clients, trained a Random Forest, and evaluated it using 5-fold client-grouped validation. The model achieved a Precision@50 of 0.444 compared with 0.392 for a simple baseline, and I converted the results into a human-review action playbook rather than automatically changing content.

---

# 34. What Week 8 Demonstrated

The important achievement was not just the model score.

The project became an **end-to-end ML workflow**:

1. Start with a real content problem.
2. Define a measurable target.
3. Use historical data.
4. Prevent future leakage.
5. Build a simple baseline.
6. Train an ML model.
7. Validate by client.
8. Measure Precision@K.
9. Rank pages for review.
10. Convert predictions into an action playbook.
11. Add human-review guardrails.
12. Publish the research paper.
13. Explain the work to social and employer audiences.

---

# 35. Final Mental Model

Remember the whole project as:

```text
             FLYRANK CONTENT PROBLEM
                      │
                      ▼
            Which pages need attention?
                      │
                      ▼
                 Historical Data
                      │
                      ▼
               Define ML Target
                      │
                      ▼
              Check for Leakage
                      │
                      ▼
             Build Baseline
                      │
                      ▼
               Train Model
                      │
                      ▼
        Client-Grouped Validation
                      │
                      ▼
             Precision@K
                      │
                      ▼
             Rank Webpages
                      │
                      ▼
           Human Review / Action
                      │
                      ▼
             Paper + Storytelling
```

## Most important numbers

```text
Eligible pages       = 16,513
Clients              = 36
Validation            = 5-fold GroupKFold by client

RF P@10              = 0.460
RF P@20              = 0.430
RF P@50              = 0.444
RF P@100             = 0.448

Baseline P@10        = 0.400
Baseline P@20        = 0.370
Baseline P@50        = 0.392
Baseline P@100       = 0.388

RF P@50 improvement  = +5.2 percentage points
```

## Words to use

- **observed**
- **measured**
- **directional evidence**
- **decision support**
- **potentially declining**
- **human review**

## Words to avoid

- guaranteed
- caused
- proves
- automatic optimization
- guaranteed traffic recovery
- predicts Google's algorithm

---

## Final one-line project summary

> **An ML-based decision-support workflow that ranks potentially declining webpages for human SEO/content review using historical search-performance data.**
