# Competitive Pricing & Demand Estimation

A machine learning pipeline for customer demand estimation and revenue-maximizing dynamic pricing, extended to real-time competitive pricing using opponent modeling.

Built for Cornell Tech's Applied Data Science course (ORIE 5355).

---

## Overview

This project tackles a two-part pricing problem:

1. **Demand Estimation & Price Optimization** — Segment customers and find the revenue-maximizing price for each segment under inventory constraints
2. **Pricing Under Competition** — Adapt pricing strategy in real time by predicting a competitor's behavior and undercutting strategically

---

## Part 1: Demand Estimation & Price Optimization

### Approach

Rather than splitting customers by median covariate values (a naive baseline), we used a **decision tree to learn optimal customer segmentation splits**, then fit a **logistic regression model per leaf node** to estimate each segment's purchase probability curve.

This allows for data-driven customer groupings and segment-specific price-demand curves, compared to the arbitrary median-split approach.

### Inventory-Constrained Pricing via Dynamic Programming

To handle inventory constraints over time, we implemented **dynamic programming with backward induction**:

```
V(t, i) = max_p [ P(buy|p) * (p + V(t+1, i-1)) + (1 - P(buy|p)) * V(t+1, i) ]
```

Rather than optimizing each customer independently, the DP table accounts for the **opportunity cost** of selling now versus preserving inventory for future customers. We used a weighted average demand distribution across segments (weighted by leaf proportions) to represent expected future customers, with ternary search at inference time per arriving customer.

### Results

| Metric | Value |
|---|---|
| Best ROC-AUC (validation) | **0.941** |
| Projected revenue (50,000 customers) | **$2,022,972** |

---

## Part 2: Pricing Under Competition

### Opponent Modeling

In the competitive setting, the team modeled the opponent's pricing strategy using a **Kalman Filter** to estimate latent price signals from noisy observations, with **Thompson Sampling** for early-round exploration.

### Demand Estimation Under Competition

We extended the demand model to account for the competitor by calculating the conditional probability of a sale:

1. Compute customer purchase probability using XGBoost
2. Predict opponent's price via Kalman Filter
3. Use a **sigmoid on the price difference** to estimate probability of undercutting
4. Return: `P(sell) = P(customer buys) × P(we undercut opponent)`

The weights and probability structure for this demand distribution were a core part of our team's pricing strategy work.

---

## Tech Stack

- **Python** — NumPy, Pandas, Scikit-learn, XGBoost
- **Models** — Decision Tree, Logistic Regression, XGBoost, Kalman Filter
- **Techniques** — Dynamic Programming, Thompson Sampling, Ternary Search
  

> **Note:** Training data is not included as it is course property.

---

## Repository Structure

├── demand-pricing-1
  ├── create_model.ipynb       # Decision Tree + Logistic Regression
  ├── pricing_agent.py         # Price optimization + DP industry constrained pricing
├── demand-pricing-2
  ├── create_model.ipynb       # XGBoost training pipeline
  ├── pricing_agent.py         # Competition pricing agent (Kalman Filter + DP)
└── README.md

## Team

Anderson Lo, Tiffany Yu, Jay Huang, and Kai Zuang — Cornell Tech ORIE 5355, 2025.
