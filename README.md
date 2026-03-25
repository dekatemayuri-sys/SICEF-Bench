# SICEF-Bench

Benchmark dataset and evaluation code for **SICEF: An AI-Driven Framework 
for Automated Support Case Evaluation and Languishing Ticket Detection in 
Large-Scale Cloud Support Systems**.

Submitted to IEEE Transactions on Services Computing, 2026.  
Author: Mayuri Mahesh Dekate — Amazon Web Services, Seattle, WA  
ORCID: 0009-0003-2738-813X

---

## Overview

SICEF-Bench is a structured synthetic benchmark of 5,000 support cases 
across five cloud service domains, designed to evaluate support case 
stagnation detection systems under controlled, reproducible conditions.

**Five domains:** Compute Infrastructure, Networking Systems, Storage 
Services, Identity Management, Deployment Workflows (1,000 cases each).

**Stagnation prevalence:** ~31% of cases are labeled as stagnating.

---

## Repository Contents

| File | Description |
|------|-------------|
| `sicef_experiment.py` | Main benchmark generation and SCP-A evaluation script |
| `sicef_results.json` | Experimental results reported in the paper |
| `README.md` | This file |

---

## Reproducing Paper Results

### Requirements
```
pip install scikit-learn numpy
```

### Run
```
python sicef_experiment.py --seed 42
```

All results in the paper are reproducible with `seed=42`.

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ (threshold) | 0.30 | Stagnation detection threshold |
| k (window) | 3 | Rolling mean observation window |
| α | 0.40 | Semantic novelty weight |
| β | 0.35 | Coverage expansion weight |
| γ | 0.25 | Narrative advancement weight |

---

## Results Summary

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Inactivity Threshold (baseline) | 0.307 | 0.926 | 0.462 |
| SCP-A (proposed) | **0.404** | 0.783 | **0.533** |

SCP-A achieves **31.6% higher precision** than the production-standard 
inactivity threshold baseline (p < 0.001, McNemar's test).

---

## Citation

If you use SICEF-Bench in your research, please cite:

> M. M. Dekate, "SICEF: An AI-Driven Framework for Automated Support Case 
> Evaluation and Languishing Ticket Detection in Large-Scale Cloud Support 
> Systems," IEEE Transactions on Services Computing, 2026.
