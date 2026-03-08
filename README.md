# Competitive Pricing & Demand Estimation

A machine learning pipeline for customer demand estimation and revenue-maximizing dynamic pricing, extended to real-time competitive pricing using opponent modeling.

Built for Cornell Tech's Applied Data Science course (ORIE 5355).

## Team

Built collaboratively with Anderson Lo, Jay Huang, and Kai Zuang 

---

## Overview

This project tackles a two-part pricing problem:

1. **Demand Estimation & Price Optimization** — Segment customers and find the revenue-maximizing price for each segment under inventory constraints
2. **Pricing Under Competition** — Adapt pricing strategy in real time by predicting a competitor's behavior and undercutting strategically

---

## Part 1: Demand Estimation & Price Optimization

### Approach

Rather than splitting customers by median covariate values, we used a **decision tree to learn optimal customer segmentation splits**, then fit a **logistic regression model per leaf node** to estimate each segment's purchase probability curve.

This pipeline allows for:
- Non-arbitrary, data-driven customer groupings
- Segment-specific price-demand curves
- Personalized revenue-maximizing prices per customer

We compared multiple approaches including single logistic regression with interaction terms, random forest + k-means segmentation, and our final decision tree + logistic regression method.

### Inventory-Constrained Pricing via Dynamic Programming

To handle inventory constraints over time, we implemented **dynamic programming with backward induction**:

```
V(t, i) = max_p [ P(buy|p) * (p + V(t+1, i-1)) + (1 - P(buy|p)) * V(t+1, i) ]
```

Rather than optimizing each customer independently, the DP table accounts for the **opportunity cost** of selling now versus preserving inventory for future customers. We used a weighted average demand distribution across all segments (weighted by leaf proportions) to represent expected future customers, then performed ternary search at inference time for each arriving customer.

**Optimizations to reduce timeout errors:**
- Caching demand probabilities by price
- Caching leaf assignments by covariate vector
- Adaptive tolerance (larger for DP generation, tighter for online pricing)

### Results

| Metric | Value |
|---|---|
| Best ROC-AUC (validation) | **0.941** |
| Projected revenue (50,000 customers) | **$2,022,972** |
| Hyperparameter search | Grid search over `min_samples_leaf` × `max_tree_depth` |

---

## Part 2: Pricing Under Competition

### Opponent Modeling with Kalman Filter

In the competitive setting, we modeled the opponent's pricing strategy using a **Kalman Filter**, which estimates a latent (unobservable) price signal from noisy observations. Inputs include customer covariates, changes in opponent inventory, and time until replenishment.

To address the **cold start problem**, we pre-trained a ridge regression model on our own XGBoost predictions to initialize Kalman Filter weights, then used **Thompson Sampling** for early-round exploration.

### Demand Estimation Under Competition

We extended the demand model to account for the competitor:

1. Compute customer purchase probability using XGBoost
2. Predict opponent's price via Kalman Filter
3. Use a **sigmoid function on the price difference** to estimate probability of undercutting the opponent
4. Return conditional probability: `P(sell) = P(customer buys) × P(we undercut opponent)`

### Pricing Strategy

- Bellman equation to find the revenue-maximizing price given current DP table state
- **Inventory multiplier** to prioritize selling before replenishment when opponent inventory is low
- Slight undercutting when our optimal price is close to the opponent's predicted price

### Model Switch: XGBoost

For Part 2, we migrated from logistic regression to **XGBoost**, enabling better demand prediction while incorporating additional competitive features (opponent price predictions, inventory state).

---

## Tech Stack

- **Python** — NumPy, Pandas, Scikit-learn, XGBoost
- **Models** — Decision Tree, Logistic Regression, XGBoost, Kalman Filter, Ridge Regression
- **Techniques** — Dynamic Programming, Thompson Sampling, Ternary Search, Hyperparameter Grid Search
  

> **Note:** Training data (`train_prices_decisions_2025.csv`) is not included as it is course property.
