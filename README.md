# GARCH-Filtered Volatility Forecasting with Machine Learning (XGBoost)

## DE–FR Coupled Power Market

**This is a re-upload of the project. This new upload is active work in progress. I would especially like to test it on another dataset that was not standartized/regularized or anyhow manipulated before.**

This project implements a **hybrid volatility forecasting and signal extraction framework** for 24-hour electricity futures in a coupled European power market (Germany–France).

The methodology combines:
- **Rolling GARCH model** to filter systematic volatility dynamics (market-neutrality)
- **Machine learning (XGBoost)** to model and forecast *residual volatility*
- **Rank-based validation and trading**

The framework is designed for **relative ranking and market-neutral trading**, not absolute volatility prediction.

> **Research-grade project!** Also beware that it is limited by the dataset size, therefore results obtained here might not generalize extremely well onto larger data/drastically different regimes.


---

## Data Source and Scope

The data comes from a past machine learning challenge hosted by ENS:
- https://challengedata.ens.fr/participants/challenges/97/

Due to licensing and registration requirements, the dataset is **not included** in this repository. It can be retrieved freely from the website after registration. 
All variables contained in the dataset, along with additional context, are described in the accompanying Jupyter notebook.

**Important clarification**  
The goals of this project **do not align with the original ML challenge**:
- Only **TRAIN data** is used
- A **rolling-window forecasting setup** is applied
- Full realized volatility is available for evaluation
- No challenge TEST data is used
- The original challenge focuses on **price explanation**, not forecasting

---

## Objective and Evaluation

- **Primary metric:** Spearman rank correlation  
- **Motivation:** The model is evaluated on its ability to *correctly rank relative volatility*, consistent with cross-sectional and market-neutral trading applications.

Model outputs are validated:
- **Statistically** (rank correlation, permutation tests)
- **Economically** (via a realistic trading strategy with costs)

---

## Project Status

This project is actively evolving. Potential future extensions include:
- Bayesian hyperparameter optimization
- Alternative ranking and loss metrics
- Additional validation and robustness tests

---

## 1. Economic Motivation

Electricity price volatility is driven by **structural system stress**, not only by past price dynamics.

Key volatility regimes include:
- Supply–demand imbalance (residual load stress)
- Cross-border congestion and flow pressure
- Fuel merit order shifts (gas–coal–carbon spreads)
- Renewable penetration regimes

These regimes are:
- Persistent
- Interpretable
- Economically meaningful

This makes them particularly suitable for **rank-based machine learning approaches**.

---

## 2. Model Architecture

### 2.1 GARCH Baseline

For each country (DE and FR):

- A rolling **GARCH(1,1)** model is fitted on past returns
- Rolling window size: **500 days**
  - Ensures at least ~300 observations
  - Produces stable volatility estimates
- One-step-ahead conditional volatility is forecast
- Models are estimated **independently** for Germany and France

This step captures **autoregressive volatility structure** and removes common dynamics.

---

### 2.2 ML on Residual Volatility

Machine learning is applied to **residual volatility**:


- Model: **XGBoost (gradient-boosted decision trees)**
- Training is performed in rolling windows
- Hyperparameters are selected empirically (see notebook)

#### Why XGBoost?

- Captures nonlinear regime behavior and threshold effects
- Well suited for power markets, where volatility reacts strongly under system stress
- Robust to heavy tails and outliers
- Performs well on small, noisy datasets **with proper regularization!**

Residuals are used **directly** for forecasting.  
A smoothed (e.g. 5-day rolling mean) residual was tested and improves ranking stability, but shifts the strategy toward trend-following, which may reduce economic relevance for day-ahead trading.

---

### 2.3 Rolling Window Design

- **Training window:** 280 days (~1 trading year)  
  Balances regime stability and adaptability
- **Prediction window:** 21 days  
  Monthly rebalancing horizon, less noisy than daily signals

Window lengths should generally be chosen to optimize the stability–responsiveness tradeoff.

---

### 2.4 Feature Engineering

From raw inputs (weather, fuel data, system variables), **low-noise structural regime features** are constructed. Examples include:

#### Volatility Persistence
- `vol_lag1`, `vol_lag3`, `vol_lag7`
- `vol_roll_std_7`, `vol_roll_std_30`

#### System Stress
- `DE_RESIDUAL_LOAD`, `FR_RESIDUAL_LOAD`
- `DE_RESIDUAL_STRESS`, `FR_RESIDUAL_STRESS`

#### Cross-Border Congestion
- `LOAD_IMBALANCE`
- `FLOW_PRESSURE`
- `TOTAL_FLOW`

#### Fuel & Merit Order
- `GAS_COAL_SPREAD`
- `CARBON_PRESSURE`
- `HIGH_GAS_REGIME`

#### Renewables Regime
- `REL_RENEWABLE`

Initially, feature selection was manual.  
An **automatic feature selection procedure** was later introduced, significantly reducing the feature set.

Key finding:
> Structural regime variables dominate predictive power; many exogenous drivers are implicitly absorbed by lagged volatility and slow-moving stress indicators.

All engineered features are checked to ensure **no forward-looking bias** or data leakage.

---

## 3. Validation Results

| Metric                     | Value |
|----------------------------|-------|
| Pooled IC, residuals, ML   | 0.32  |
| DE IC, residuals, ML       | 0.12  |    
| FR IC, residuals, ML       | 0.18  |
| DE IC, baseline GARCH      | 0.08  |
| FR IC, baseline GARCH      | 0.07  |
| DE rank autocorr.          | 0.78  |
| FR rank autocorr.          | 0.68  |

**Interpretation:**
- The model extracts statistically significant and persistent structure, tradeable outcome
- Performance is stable across time
- Additional details can be found in the notebook
---

## 4. Trading Strategy

Model predictions are translated into a **relative volatility spread strategy**:

- Cross-sectional normalization per day  
  → market neutrality across countries
- Rolling z-scores per country
- Threshold-based long/short signals  
  → trades only when signals are strong enough to overcome costs
- Hypothetical transaction costs are included

---

### Trading Results (With Costs)

| Metric               | Value |
|----------------------|-------|
| Sharpe Ratio         | 0.8   |
| Max Drawdown         |-13.8  |
| Avg Daily Turnover   | 0.48  |

Additional evaluation and details can be found in the notebook.

These results are consistent with a **realistic, market-neutral relative-volatility arbitrage strategy**, rather than a directional volatility bet.

---
