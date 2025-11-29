# Allowed imports
# You can import individual modules from packages
# e.g.- from sklearn import linear_model
import datetime
import math
import random

import auto_ts
import autots
import cvxopt
import cvxpy
import darts
import keras
import lightgbm
import numpy as np
import pandas as pd
import prophet
import scipy
import sklearn
import sktime
import statsmodels
import tensorflow
import torch
import tsfresh
import xgboost

# You are not allowed to rename `initializeFn` and `alphaFn` or modify their function signatures.


# ==========================================
# Strategy Configuration
# ==========================================
PARAMS = {
    'lookback_days': 4,
    'momentum_lag': 2,
    # Penalties and Thresholds
    'vol_penalty_qt': 0.4,      # Volume quantile threshold
    'vol_penalty_val': 0.6,     # Penalty value if below threshold
    'volatility_qt': 0.9,       # Volatility quantile threshold
    'volatility_high_pen': 0.8, # Penalty for high volatility
    'volatility_low_pen': 3.0,  # Boost for low volatility
    'liquidity_qt': 0.6,        # Liquidity quantile threshold
    'liquidity_boost': 1.5,     # Boost for high liquidity
    # Weights
    'w_momentum': 0.2,
    'w_scaled_ret': 0.5,
    'w_vol_adj': 1.1,
    # Smoothing / Turnover
    'turnover_penalty_strength': 0.4,
    'alpha_decay': 0.6,         # Weight of previous alpha
    'signal_strength': 0.4      # Weight of new signal
}

def initializeFn(simulation):
    """Initialize data to be stored across simulation days"""
    N = simulation.N()
    # Initialize variable to store previous day's alpha as array of 0s
    simulation.my_data["prev_alpha"] = pd.Series(index=np.arange(N)).fillna(0)
    # Another example
    simulation.my_data["days_elapsed"] = 0


def alphaFn(di, simulation):
    """Computes alpha vector"""
    N = simulation.N()
    # # Alpha needs to be
    # # - A Pandas Series
    # # - Have length N
    # # - Have index 0, 1, ... N-1
    """
    Fields available: ['open', 'high', 'close', 'low', 'volume', 'sharesout', 'returns', 'industry']

    1. simulation.N()

    Purpose: Returns the total number of stocks in the simulation.
    Usage: N = simulation.N()
    Output: Integer value representing the number of stocks.

    2. simulation.oneDayData(field, di)

    Purpose: Retrieves the data for a particular field (or fields) for all stocks on day di.
    Usage: data = simulation.oneDayData("returns", 3)
    Input:
        field: String or list of strings representing the field(s) you want data for (e.g., "close", "volume", "returns").
        di: Integer representing the day number.
    Output: A Pandas DataFrame where rows represent stocks, and columns represent the data fields.

    3. simulation.fields()

    Purpose: Returns a list of available fields that can be accessed for each stock (e.g., open, close, volume, etc.).
    Usage: fields = simulation.fields()
    Output: List of strings representing the field names.

    4. simulation.pointData(field, di, stock_id)

    Purpose: Retrieves data for a specific field and stock on a particular day.
    Usage: data = simulation.pointData("close", 2, 10)
    Input:
        field: String representing the field to retrieve data from.
        di: Integer representing the day number.
        stock_id: Integer representing the stock ID.
    Output: A Pandas Series with the data for the given stock on the specified day.

    5. simulation.oneStockData(field, stock_id, start_di, end_di)

    Purpose: Retrieves the data for a specific field for one stock over a range of days.
    Usage: data = simulation.oneStockData("returns", 1, 5, 9)
    Input:
        field: String representing the field (e.g., "returns", "close").
        stock_id: Integer representing the stock ID.
        start_di: Integer representing the start day of the range.
        end_di: Integer representing the end day of the range.
    Output: A Pandas DataFrame or Series containing the data for that stock over the specified range of days.

    6. simulation.fieldData(field, start_di, end_di)

    Purpose: Retrieves the data for a particular field for all stocks between two days.
    Usage: data = simulation.fieldData("volume", 0, 2)
    Input:
        field: String representing the field (e.g., "volume").
        start_di: Integer representing the start day.
        end_di: Integer representing the end day.
    Output: A Pandas DataFrame with stock data for the specified time range.

    7. simulation.my_data

    Purpose: Stores user-defined variables that persist across simulation days. It acts like a dictionary, allowing users to store and retrieve custom data.
    Usage:
    Storing data: simulation.my_data["key"] = value
    Retrieving data: value = simulation.my_data["key"]
    Example: Used in the code to store prev_alpha and days_elapsed.
    
    """
    N = simulation.N()
    market_index = np.arange(N)
    
    # Initialize the raw signal accumulator
    raw_signal_accum = pd.Series(0.0, index=market_index)
    
    # ------------------------------------------------------------------
    # 1. Signal Generation Loop (Rolling Window)
    # ------------------------------------------------------------------
    # Limit lookback if we are at the very beginning of the simulation
    lookback = min(di + 1, PARAMS['lookback_days'])
    
    for day_lag in range(lookback):
        current_lag_idx = di - day_lag
        
        # Fetch Data
        r = simulation.oneDayData("returns", current_lag_idx)
        volume_df = simulation.oneDayData("volume", current_lag_idx)
        close_df = simulation.oneDayData("close", current_lag_idx)
        high_df = simulation.oneDayData("high", current_lag_idx)
        low_df = simulation.oneDayData("low", current_lag_idx)
        shares_df = simulation.oneDayData("sharesout", current_lag_idx)
        
        lag_momentum_idx = max(0, current_lag_idx - PARAMS['momentum_lag'])
        lagged_close_df = simulation.oneDayData("close", lag_momentum_idx)

        # Extract Series for easier math
        s_close = close_df["close"]
        s_volume = volume_df["volume"]
        s_shares = shares_df["sharesout"]
        s_returns = r["returns"].fillna(0)

        # --- Factor Calculation ---
        
        # Handle division by zero or missing data implicitly via Pandas
        momentum = s_close / lagged_close_df["close"]

        # Volume Penalty: Penalize low volume stocks
        vol_threshold = s_volume.quantile(PARAMS['vol_penalty_qt'])
        penalty_factor = np.where(s_volume < vol_threshold, 
                                  PARAMS['vol_penalty_val'], 
                                  1.0)

        # Adjusted Volume (Turnover Ratio proxy)
        adjusted_volume = s_volume / s_shares.replace(0, np.nan) # Avoid div/0

        # Volatility: Daily Range / Close
        volatility = (high_df["high"] - low_df["low"]) / s_close
        vol_qt_val = volatility.quantile(PARAMS['volatility_qt'])
        volatility_penalty = np.where(volatility > vol_qt_val, 
                                      PARAMS['volatility_high_pen'], 
                                      PARAMS['volatility_low_pen'])

        # Liquidity Adjustment: Boost high share-count stocks
        liq_threshold = s_shares.quantile(PARAMS['liquidity_qt'])
        liquidity_factor = np.where(s_shares > liq_threshold, 
                                    PARAMS['liquidity_boost'], 
                                    0.0)

        # Composite Returns Scaling
        scaled_returns = (s_returns * penalty_factor * liquidity_factor * volatility_penalty)

        # Accumulate Signal
        # Formula: -Momentum - Returns + (Volatility * Volume)
        daily_signal = (- PARAMS['w_momentum'] * momentum 
                        - PARAMS['w_scaled_ret'] * scaled_returns.fillna(0) 
                        + PARAMS['w_vol_adj'] * volatility * adjusted_volume)
        
        raw_signal_accum = raw_signal_accum.add(daily_signal, fill_value=0)

    # ------------------------------------------------------------------
    # 2. Industry Neutralization
    # ------------------------------------------------------------------
    industry_data = simulation.oneDayData("industry", di)
    
    if not industry_data.empty:
        # Group by industry and subtract the group mean from the signal
        # This ensures the strategy is dollar-neutral within sectors
        raw_signal_accum = raw_signal_accum.groupby(industry_data["industry"]).transform(lambda x: x - x.mean())
    
    # Fill any remaining NaNs (e.g., stocks with no industry)
    raw_signal_accum = raw_signal_accum.fillna(0)

    # ------------------------------------------------------------------
    # 3. Turnover Control & Smoothing
    # ------------------------------------------------------------------
    prev_alpha = simulation.my_data["prev_alpha"].fillna(0)
    
    # Calculate difference from previous position
    turnover_diff = (raw_signal_accum - prev_alpha).abs()
    
    # Dampen signal if it requires high turnover
    turnover_penalty = 1 / (1 + turnover_diff ** PARAMS['turnover_penalty_strength'])
    penalized_signal = raw_signal_accum * turnover_penalty
    
    # Exponential Moving Average (EMA) to smooth Alpha over time
    alpha = (penalized_signal * PARAMS['signal_strength'] + 
             prev_alpha * PARAMS['alpha_decay'])

    # ------------------------------------------------------------------
    # 4. State Update
    # ------------------------------------------------------------------
    simulation.my_data["prev_alpha"] = alpha
    simulation.my_data["days_elapsed"] += 1

    ###### DO NOT REMOVE THIS OR WRITE ANY OTHER RETURN STATEMENTS ######
    return alpha
    #####################################################################

