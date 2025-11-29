# Quantitative-Alpha

[![Achievement](https://img.shields.io/badge/Achievement-3rd%20Place%20%7C%20DTL%20Quant%20Challenge%20'24-gold)](https://github.com/yourusername/repo)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## Overview
This repository contains the source code and documentation for a **Quantitative Alpha-Model** developed for the DTL 2024 Quant Challenge. The strategy secured **3rd position among 1300+ participants** globally.

The model implements an **industry-neutral, multi-factor strategy** that fuses mean-reversion signals with volatility and liquidity constraints to generate a robust daily alpha vector.

## Strategy Logic
The alpha generation engine (`alphaFn`) constructs a composite signal based on three primary components:

### 1. Weighted Mean Reversion (Reversal)
The core signal exploits short-term overreaction by betting on mean reversion. It is mathematically defined as a negative summation of past returns, weighted by:
* **Volume Penalty:** Penalizes stocks with low liquidity to avoid high slippage.
* **Volatility Penalty:** Reduces exposure to high-variance stocks to mitigate tail risk.
* **Liquidity Boost:** Amplifies signals for stocks with high shares outstanding.

### 2. Momentum Damping
Incorporates a lagged closing price ratio to identify and dampen signals against strong opposing momentum trends.

### 3. Volatility-Adjusted Trend
Captures upward trend opportunities by analyzing the daily high-low spread relative to volume, isolating high-conviction price movements supported by market participation.

## 🛠️ Risk Management & Optimization

The raw signal undergoes rigorous post-processing to ensure deployability:

* **Industry Neutralization:**
    $$S_{neutral} = S_{raw} - \text{mean}(S_{raw} | \text{Industry}_k)$$
    * Removes systematic sector risk, isolating idiosyncratic stock performance (Alpha vs. Beta).

* **Turnover Control:**
    * Implements a non-linear penalty function based on the delta between the current signal and the previous day's alpha.
    * Reduces transaction costs by discouraging marginal portfolio rebalancing.

* **Signal Smoothing:**
    * Applies an Exponential Moving Average (EMA) decay factor (`alpha_decay = 0.6`) to stabilise the alpha vector over time.

## File Structure
* `main.py`: Core logic containing `initializeFn` (state setup) and `alphaFn` (signal generation loop).
* `report.pdf`: Detailed mathematical breakdown of the strategy, terms, and primitive variables.

## Performance
* **Rank:** 3rd / 1300+ Participants.
* **Key Metrics:** High Sharpe ratio achieved through composite signal blending and aggressive turnover penalisation.

---
*Author: Resnick Singh | IIT Bombay*
