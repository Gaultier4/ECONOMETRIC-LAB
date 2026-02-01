"""
Data processing utilities.
Contains data loading, transformations, and filtering functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """
    Load data from uploaded file (CSV or Excel).
    Cached to avoid reloading on every interaction.
    """
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None


@st.cache_data
def load_default_data(filepath: str) -> pd.DataFrame:
    """Load default CSV dataset."""
    return pd.read_csv(filepath)


# =============================================================================
# TRANSFORMATIONS (Non-Destructive)
# =============================================================================

def apply_transformation(
    df: pd.DataFrame,
    column: str,
    transform_type: str,
    new_name: str = None,
    keep_original: bool = True,
    **kwargs
) -> tuple:
    """
    Apply transformation to a column, optionally keeping the original.
    
    Args:
        df: DataFrame to transform
        column: Column to transform
        transform_type: Type of transformation ('Log', 'Diff', 'SDiff', 'Lag', 'HP_Trend', 'HP_Cycle')
        new_name: Name for new column (auto-generated if None)
        keep_original: Whether to keep the original column
        **kwargs: Additional parameters (e.g., lag periods, HP lambda)
    
    Returns:
        Tuple of (new_dataframe, new_column_name, transformation_record)
    """
    df_new = df.copy()
    
    # Auto-generate new name if not provided
    if new_name is None:
        suffix_map = {
            "Log": "_Log",
            "Diff": "_Diff",
            "SDiff": "_SDiff12",
            "Lag": f"_Lag{kwargs.get('periods', 1)}",
            "HP_Trend": "_Trend",
            "HP_Cycle": "_Cycle"
        }
        new_name = f"{column}{suffix_map.get(transform_type, '_Trans')}"
    
    # Apply transformation
    if transform_type == "Log":
        if (df_new[column] <= 0).any():
            # Replace non-positive with NaN before log
            series = df_new[column].copy()
            series[series <= 0] = np.nan
            df_new[new_name] = np.log(series)
        else:
            df_new[new_name] = np.log(df_new[column])
    
    elif transform_type == "Diff":
        df_new[new_name] = df_new[column].diff()
    
    elif transform_type == "SDiff":
        periods = kwargs.get('periods', 12)
        df_new[new_name] = df_new[column].diff(periods)
    
    elif transform_type == "Lag":
        periods = kwargs.get('periods', 1)
        df_new[new_name] = df_new[column].shift(periods)
    
    elif transform_type in ["HP_Trend", "HP_Cycle"]:
        lamb = kwargs.get('lamb', 1600)
        trend, cycle = hp_filter(df_new[column], lamb)
        if transform_type == "HP_Trend":
            df_new[new_name] = trend
        else:
            df_new[new_name] = cycle
    
    # Remove original if requested
    if not keep_original:
        df_new = df_new.drop(columns=[column])
    
    # Create transformation record
    record = {
        'old_name': column,
        'new_name': new_name,
        'type': transform_type,
        'keep_original': keep_original,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'params': kwargs
    }
    
    return df_new, new_name, record


# =============================================================================
# HODRICK-PRESCOTT FILTER
# =============================================================================

def hp_filter(series: pd.Series, lamb: float = 1600) -> tuple:
    """
    Apply Hodrick-Prescott filter to extract trend and cycle.
    
    Args:
        series: Time series data
        lamb: Smoothing parameter (1600 for quarterly, 14400 for monthly, 100 for annual)
    
    Returns:
        Tuple of (trend, cycle) as pandas Series
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    y = series.dropna().values
    n = len(y)
    
    if n < 4:
        return series, pd.Series(0, index=series.index)
    
    # Build the HP filter matrix
    # Second difference matrix D
    e = np.ones(n)
    D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(n-2, n))
    
    # HP filter: (I + lambda * D'D)^-1 * y
    I = sparse.eye(n)
    A = I + lamb * (D.T @ D)
    
    trend_values = spsolve(A.tocsc(), y)
    cycle_values = y - trend_values
    
    # Reconstruct series with original index (for non-nan values)
    trend = series.copy()
    cycle = series.copy()
    
    non_nan_idx = series.dropna().index
    trend.loc[non_nan_idx] = trend_values
    cycle.loc[non_nan_idx] = cycle_values
    
    return trend, cycle


# =============================================================================
# LAG CREATION
# =============================================================================

def make_lags(df: pd.DataFrame, target_col: str, n_lags: int) -> pd.DataFrame:
    """
    Create lagged versions of a column.
    
    Returns:
        DataFrame with original data plus lag columns
    """
    df_lagged = df.copy()
    for i in range(1, n_lags + 1):
        df_lagged[f'{target_col}_lag_{i}'] = df_lagged[target_col].shift(i)
    return df_lagged


def prepare_ml_features(df: pd.DataFrame, target: str, predictors: list, n_lags: int = 1) -> tuple:
    """
    Prepare features for ML models with lags.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    df_prep = df.copy()
    
    # Add lags of target
    for i in range(1, n_lags + 1):
        df_prep[f'{target}_lag_{i}'] = df_prep[target].shift(i)
    
    # Drop NaN rows
    df_prep = df_prep.dropna()
    
    # Build feature list
    lag_cols = [f'{target}_lag_{i}' for i in range(1, n_lags + 1)]
    feature_cols = predictors + lag_cols
    
    X = df_prep[feature_cols]
    y = df_prep[target]
    
    return X, y, feature_cols


# =============================================================================
# FREQUENCY HANDLING
# =============================================================================

def infer_and_set_frequency(df: pd.DataFrame, date_col: str) -> tuple:
    """
    Parse date column, set as index, and infer frequency.
    
    Returns:
        Tuple of (processed_df, inferred_freq or None)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)
    
    inferred = pd.infer_freq(df.index)
    if inferred:
        df.index.freq = inferred
    
    return df, inferred


def get_lambda_for_frequency(freq: str) -> int:
    """
    Return recommended HP filter lambda based on frequency.
    """
    freq_map = {
        'Y': 100,      # Annual
        'A': 100,
        'Q': 1600,     # Quarterly
        'M': 14400,    # Monthly
        'W': 270400,   # Weekly (approx)
        'D': 129600000 # Daily (very high)
    }
    
    if freq is None:
        return 1600  # Default to quarterly
    
    for key in freq_map:
        if key in freq.upper():
            return freq_map[key]
    
    return 1600  # Default
