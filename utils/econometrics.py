"""
Econometrics utility functions.
Contains statistical tests, model diagnostics, and econometric analysis tools.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, acf, pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
import statsmodels.api as sm


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

def check_stationarity(series: pd.Series, test_type: str = 'ADF') -> dict:
    """
    Run stationarity test on a time series.
    
    Args:
        series: Time series data
        test_type: 'ADF' for Augmented Dickey-Fuller or 'KPSS'
    
    Returns:
        Dictionary with 'Statistic', 'p-value', and 'Conclusion'
    """
    series = series.dropna()
    if series.empty or len(series) < 10:
        return {'Conclusion': 'Error', 'p-value': np.nan, 'Statistic': np.nan}
    
    results = {}
    if test_type == 'ADF':
        res = adfuller(series, autolag='AIC')
        results['p-value'] = res[1]
        results['Statistic'] = res[0]
        results['Conclusion'] = 'Stationary' if res[1] < 0.05 else 'Non-Stationary'
    elif test_type == 'KPSS':
        res = kpss(series, regression='c', nlags="auto")
        results['p-value'] = res[1]
        results['Statistic'] = res[0]
        results['Conclusion'] = 'Non-Stationary' if res[1] < 0.05 else 'Stationary'
    return results


def batch_stationarity_test(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Run ADF and KPSS tests on multiple columns.
    
    Returns:
        DataFrame with test results for each variable
    """
    results = []
    for col in columns:
        s = df[col].dropna()
        adf = check_stationarity(s, 'ADF')
        kpss_res = check_stationarity(s, 'KPSS')
        
        stat_adf = adf['Conclusion'] == 'Stationary'
        stat_kpss = kpss_res['Conclusion'] == 'Stationary'
        
        if stat_adf and stat_kpss:
            synthesis, icon = "Stable", "✅"
        elif (not stat_adf) and (not stat_kpss):
            synthesis, icon = "Unstable", "❌"
        else:
            synthesis, icon = "Ambiguous", "⚠️"
        
        results.append({
            "Variable": col,
            "ADF p-val": adf['p-value'],
            "KPSS p-val": kpss_res['p-value'],
            "Result": f"{icon} {synthesis}"
        })
    
    return pd.DataFrame(results)


# =============================================================================
# POST-ESTIMATION DIAGNOSTICS
# =============================================================================

def run_ols_diagnostics(model_result) -> pd.DataFrame:
    """
    Run diagnostic tests on OLS residuals.
    
    Args:
        model_result: Fitted OLS model from statsmodels
    
    Returns:
        DataFrame with test results (Jarque-Bera, Ljung-Box, Breusch-Pagan)
    """
    residuals = model_result.resid
    
    diagnostics = []
    
    # Jarque-Bera (Normality)
    jb_stat, jb_pval, _, _ = jarque_bera(residuals)
    diagnostics.append({
        "Test": "Jarque-Bera (Normality)",
        "Statistic": round(jb_stat, 4),
        "P-Value": round(jb_pval, 4),
        "Conclusion": "✅ Normal" if jb_pval > 0.05 else "❌ Non-Normal"
    })
    
    # Durbin-Watson (Autocorrelation - quick check)
    dw = durbin_watson(residuals)
    # DW close to 2 = no autocorrelation, <1.5 or >2.5 indicates issues
    dw_ok = 1.5 < dw < 2.5
    diagnostics.append({
        "Test": "Durbin-Watson (Autocorr)",
        "Statistic": round(dw, 4),
        "P-Value": "N/A",
        "Conclusion": "✅ Pass" if dw_ok else "⚠️ Autocorrelation"
    })
    
    # Ljung-Box (Autocorrelation - formal test)
    try:
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pval = lb_result['lb_pvalue'].values[0]
        lb_stat = lb_result['lb_stat'].values[0]
        diagnostics.append({
            "Test": "Ljung-Box (Autocorr)",
            "Statistic": round(lb_stat, 4),
            "P-Value": round(lb_pval, 4),
            "Conclusion": "✅ No Autocorr" if lb_pval > 0.05 else "❌ Autocorrelation"
        })
    except Exception:
        diagnostics.append({
            "Test": "Ljung-Box (Autocorr)",
            "Statistic": "N/A",
            "P-Value": "N/A",
            "Conclusion": "⚠️ Could not compute"
        })
    
    # Breusch-Pagan (Heteroscedasticity)
    try:
        exog = model_result.model.exog
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, exog)
        diagnostics.append({
            "Test": "Breusch-Pagan (Hetero)",
            "Statistic": round(bp_stat, 4),
            "P-Value": round(bp_pval, 4),
            "Conclusion": "✅ Homoscedastic" if bp_pval > 0.05 else "❌ Heteroscedastic"
        })
    except Exception:
        diagnostics.append({
            "Test": "Breusch-Pagan (Hetero)",
            "Statistic": "N/A",
            "P-Value": "N/A",
            "Conclusion": "⚠️ Could not compute"
        })
    
    return pd.DataFrame(diagnostics)


def run_arima_diagnostics(model_result) -> pd.DataFrame:
    """
    Run diagnostic tests on ARIMA residuals.
    """
    residuals = model_result.resid
    
    diagnostics = []
    
    # Jarque-Bera
    jb_stat, jb_pval, _, _ = jarque_bera(residuals)
    diagnostics.append({
        "Test": "Jarque-Bera (Normality)",
        "Statistic": round(jb_stat, 4),
        "P-Value": round(jb_pval, 4),
        "Conclusion": "✅ Normal" if jb_pval > 0.05 else "❌ Non-Normal"
    })
    
    # Ljung-Box
    try:
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pval = lb_result['lb_pvalue'].values[0]
        lb_stat = lb_result['lb_stat'].values[0]
        diagnostics.append({
            "Test": "Ljung-Box (Autocorr)",
            "Statistic": round(lb_stat, 4),
            "P-Value": round(lb_pval, 4),
            "Conclusion": "✅ No Autocorr" if lb_pval > 0.05 else "❌ Autocorrelation"
        })
    except Exception:
        diagnostics.append({
            "Test": "Ljung-Box (Autocorr)",
            "Statistic": "N/A",
            "P-Value": "N/A",
            "Conclusion": "⚠️ Could not compute"
        })
    
    return pd.DataFrame(diagnostics)


# =============================================================================
# COINTEGRATION & CAUSALITY
# =============================================================================

def johansen_test(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    """
    Run Johansen cointegration test.
    
    Returns:
        DataFrame with trace statistics and critical values
    """
    data = df.dropna()
    cj = coint_johansen(data, det_order, k_ar_diff)
    
    results = []
    for i in range(len(cj.lr1)):
        results.append({
            "r": i,
            "Trace Stat": round(cj.lr1[i], 4),
            "95% CV": round(cj.cvt[i, 1], 4),
            "Significant": "Yes" if cj.lr1[i] > cj.cvt[i, 1] else "No"
        })
    
    return pd.DataFrame(results)


def granger_causality_matrix(df: pd.DataFrame, variables: list, maxlag: int = 3) -> list:
    """
    Run pairwise Granger causality tests.
    
    Returns:
        List of significant causal relationships
    """
    significant = []
    for y in variables:
        for x in variables:
            if x == y:
                continue
            try:
                data = df[[y, x]].dropna()
                res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                p_min = min([res[lag][0]['ssr_ftest'][1] for lag in res])
                if p_min < 0.05:
                    significant.append({
                        "Cause": x,
                        "Effect": y,
                        "Min P-Value": round(p_min, 4)
                    })
            except Exception:
                pass
    return significant


# =============================================================================
# VAR IMPULSE RESPONSE FUNCTIONS
# =============================================================================

def compute_irf(var_result, periods: int = 20, orth: bool = True) -> object:
    """
    Compute Impulse Response Functions from a fitted VAR model.
    
    Args:
        var_result: Fitted VAR model result
        periods: Number of periods for IRF
        orth: Use orthogonalized IRF
    
    Returns:
        IRF results object
    """
    irf = var_result.irf(periods=periods)
    return irf


def get_irf_data(irf_result, impulse: str, response: str) -> tuple:
    """
    Extract IRF data for a specific impulse-response pair.
    
    Returns:
        Tuple of (periods, irf_values, lower_ci, upper_ci)
    """
    # Get variable indices
    names = irf_result.model.names
    impulse_idx = names.index(impulse)
    response_idx = names.index(response)
    
    # Extract IRF values
    irf_vals = irf_result.irfs[:, response_idx, impulse_idx]
    periods = np.arange(len(irf_vals))
    
    # Confidence intervals if available
    try:
        lower = irf_result.ci[:, :, response_idx, impulse_idx][0]
        upper = irf_result.ci[:, :, response_idx, impulse_idx][1]
    except Exception:
        lower = None
        upper = None
    
    return periods, irf_vals, lower, upper


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_pred) -> tuple:
    """Calculate RMSE, MAE, MAPE."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape


def get_acf_pacf(series: pd.Series, nlags: int = 39) -> tuple:
    """
    Compute ACF and PACF values.
    
    Returns:
        Tuple of (acf_values, pacf_values)
    """
    clean = series.dropna()
    acf_vals = acf(clean, nlags=nlags)
    pacf_vals = pacf(clean, nlags=nlags)
    return acf_vals, pacf_vals


def granger_pvalue_matrix(df: pd.DataFrame, variables: list, maxlag: int = 3) -> pd.DataFrame:
    """
    Build a p-value matrix for Granger causality (for heatmap visualization).
    
    Returns:
        DataFrame where matrix[cause][effect] = minimum p-value
    """
    n = len(variables)
    pvals = np.ones((n, n))  # Default to 1 (not significant)
    
    for i, effect in enumerate(variables):
        for j, cause in enumerate(variables):
            if i == j:
                pvals[i, j] = np.nan  # Diagonal
                continue
            try:
                data = df[[effect, cause]].dropna()
                res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                p_min = min([res[lag][0]['ssr_ftest'][1] for lag in res])
                pvals[i, j] = p_min
            except Exception:
                pvals[i, j] = 1.0
    
    return pd.DataFrame(pvals, index=variables, columns=variables)


def auto_arima_search(series: pd.Series, p_range: range = range(4), d_range: range = range(3), q_range: range = range(4)) -> dict:
    """
    Grid search for best ARIMA parameters using AIC.
    
    Returns:
        Dict with 'order', 'aic', and 'results_df'
    """
    from statsmodels.tsa.arima.model import ARIMA
    import itertools
    
    series = series.dropna()
    results = []
    best_aic = np.inf
    best_order = (0, 0, 0)
    
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            aic = model.aic
            results.append({
                'p': p, 'd': d, 'q': q,
                'AIC': round(aic, 2),
                'BIC': round(model.bic, 2)
            })
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
        except Exception:
            pass
    
    results_df = pd.DataFrame(results).sort_values('AIC')
    
    return {
        'order': best_order,
        'aic': best_aic,
        'results_df': results_df
    }

