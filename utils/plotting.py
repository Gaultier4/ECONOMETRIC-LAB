"""
Plotting utilities.
Contains all Plotly chart generators with consistent styling and rangesliders.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

DEFAULT_LAYOUT = {
    "hovermode": "x unified",
    "template": "plotly_white",
    "margin": dict(l=40, r=40, t=50, b=40),
}

COLORS = {
    "stationary": "#27ae60",
    "non_stationary": "#c0392b",
    "neutral": "#7f8c8d",
    "primary": "#3498db",
    "secondary": "#e74c3c",
    "tertiary": "#9b59b6",
    "train": "#3498db",
    "test": "#2ecc71",
    "forecast": "#e74c3c",
}


# =============================================================================
# TIME SERIES PLOTS
# =============================================================================

def plot_time_series(
    df: pd.DataFrame,
    columns: list,
    title: str = "Time Series",
    colors: list = None,
    height: int = 400,
    show_rangeslider: bool = True,
    stationarity_status: dict = None
) -> go.Figure:
    """
    Create time series line plot with optional rangeslider.
    
    Args:
        df: DataFrame with datetime index
        columns: List of column names to plot
        title: Chart title
        colors: Optional list of colors for each series
        height: Chart height
        show_rangeslider: Enable x-axis rangeslider for zooming
        stationarity_status: Dict mapping column -> 'Stationary'/'Non-Stationary'
    """
    fig = go.Figure()
    
    if colors is None:
        color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], 
                       "#f39c12", "#1abc9c", "#34495e"]
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(columns))]
    
    for i, col in enumerate(columns):
        # Determine line color based on stationarity if provided
        line_color = colors[i]
        if stationarity_status and col in stationarity_status:
            if stationarity_status[col] == 'Stationary':
                line_color = COLORS["stationary"]
            elif stationarity_status[col] == 'Non-Stationary':
                line_color = COLORS["non_stationary"]
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            line=dict(color=line_color, width=2)
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        **DEFAULT_LAYOUT
    )
    
    if show_rangeslider:
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    
    return fig


def plot_series_comparison(
    original: pd.Series,
    transformed: pd.Series,
    original_label: str = "Original",
    transformed_label: str = "Transformed",
    title: str = "Before vs After",
    height: int = 400
) -> go.Figure:
    """
    Plot original and transformed series side by side.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=original.index,
        y=original,
        name=original_label,
        line=dict(color=COLORS["primary"], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=transformed.index,
        y=transformed,
        name=transformed_label,
        line=dict(color=COLORS["secondary"], width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    
    return fig


# =============================================================================
# CORRELATION HEATMAP
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, columns: list, title: str = "Correlation Matrix") -> go.Figure:
    """Create correlation heatmap."""
    corr = df[columns].corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        title=title,
        height=500,
        **DEFAULT_LAYOUT
    )
    
    return fig


# =============================================================================
# ACF / PACF PLOTS
# =============================================================================

def plot_acf_pacf(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    title: str = "Autocorrelation Analysis"
) -> go.Figure:
    """Create side-by-side ACF and PACF bar plots."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    
    nlags = len(acf_values)
    lags = np.arange(nlags)
    
    # ACF
    fig.add_trace(
        go.Bar(x=lags, y=acf_values, marker_color=COLORS["primary"], name="ACF"),
        row=1, col=1
    )
    
    # PACF
    fig.add_trace(
        go.Bar(x=lags, y=pacf_values, marker_color=COLORS["secondary"], name="PACF"),
        row=1, col=2
    )
    
    # Add confidence bands (approximate 95% CI = 1.96/sqrt(n))
    # We'll use a simple approximation
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=2)
    
    fig.update_layout(
        title=title,
        height=350,
        showlegend=False,
        **DEFAULT_LAYOUT
    )
    
    return fig


# =============================================================================
# FORECAST PLOTS
# =============================================================================

def plot_forecast(
    history: pd.Series,
    forecast: pd.Series,
    ci_lower: pd.Series = None,
    ci_upper: pd.Series = None,
    title: str = "Forecast",
    height: int = 450
) -> go.Figure:
    """
    Create forecast plot with confidence interval.
    """
    fig = go.Figure()
    
    # History
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history,
        name="History",
        line=dict(color=COLORS["train"], width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast,
        name="Forecast",
        line=dict(color=COLORS["forecast"], width=3)
    ))
    
    # Confidence interval
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index)[::-1],
            y=list(ci_upper) + list(ci_lower)[::-1],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    
    return fig


def plot_train_test_forecast(
    train: pd.Series,
    test: pd.Series,
    predictions: pd.Series,
    title: str = "Model Evaluation"
) -> go.Figure:
    """
    Plot train/test split with predictions overlay.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train.index, y=train,
        name="Train",
        line=dict(color=COLORS["train"], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=test.index, y=test,
        name="Test (Actual)",
        line=dict(color=COLORS["test"], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions.index, y=predictions,
        name="Predictions",
        line=dict(color=COLORS["forecast"], width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    
    return fig


# =============================================================================
# VAR IMPULSE RESPONSE PLOTS
# =============================================================================

def plot_irf(
    periods: np.ndarray,
    irf_values: np.ndarray,
    lower_ci: np.ndarray = None,
    upper_ci: np.ndarray = None,
    impulse: str = "Impulse",
    response: str = "Response",
    title: str = None
) -> go.Figure:
    """
    Plot Impulse Response Function.
    """
    if title is None:
        title = f"IRF: {impulse} → {response}"
    
    fig = go.Figure()
    
    # IRF line
    fig.add_trace(go.Scatter(
        x=periods,
        y=irf_values,
        name="IRF",
        line=dict(color=COLORS["primary"], width=2)
    ))
    
    # Confidence bands
    if lower_ci is not None and upper_ci is not None:
        fig.add_trace(go.Scatter(
            x=list(periods) + list(periods)[::-1],
            y=list(upper_ci) + list(lower_ci)[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Periods",
        yaxis_title="Response",
        height=350,
        **DEFAULT_LAYOUT
    )
    
    return fig


def plot_irf_grid(irf_result, figsize: tuple = (900, 800)) -> go.Figure:
    """
    Create grid of all IRF combinations.
    """
    names = irf_result.model.names
    n = len(names)
    
    fig = make_subplots(
        rows=n, cols=n,
        subplot_titles=[f"{imp} → {res}" for imp in names for res in names],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    for i, impulse in enumerate(names):
        for j, response in enumerate(names):
            irf_vals = irf_result.irfs[:, j, i]
            periods = np.arange(len(irf_vals))
            
            fig.add_trace(
                go.Scatter(x=periods, y=irf_vals, line=dict(color=COLORS["primary"], width=1.5)),
                row=j+1, col=i+1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=j+1, col=i+1)
    
    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        showlegend=False,
        title="Impulse Response Functions",
        **DEFAULT_LAYOUT
    )
    
    return fig


# =============================================================================
# DIAGNOSTICS VISUALIZATION
# =============================================================================

def style_diagnostics_table(df: pd.DataFrame) -> str:
    """
    Apply color styling to diagnostics DataFrame for display.
    Returns HTML string.
    """
    def color_conclusion(val):
        if '✅' in str(val):
            return 'background-color: #d4edda; color: #155724'
        elif '❌' in str(val):
            return 'background-color: #f8d7da; color: #721c24'
        elif '⚠️' in str(val):
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    styled = df.style.applymap(color_conclusion, subset=['Conclusion'])
    return styled.to_html()


# =============================================================================
# HP FILTER VISUALIZATION
# =============================================================================

def plot_hp_decomposition(
    original: pd.Series,
    trend: pd.Series,
    cycle: pd.Series,
    title: str = "HP Filter Decomposition"
) -> go.Figure:
    """
    Plot HP filter decomposition (original, trend, cycle).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Original vs Trend", "Cycle (Detrended)"),
        vertical_spacing=0.12
    )
    
    # Original and trend
    fig.add_trace(
        go.Scatter(x=original.index, y=original, name="Original", 
                   line=dict(color=COLORS["neutral"], width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=trend.index, y=trend, name="Trend", 
                   line=dict(color=COLORS["primary"], width=2)),
        row=1, col=1
    )
    
    # Cycle
    fig.add_trace(
        go.Scatter(x=cycle.index, y=cycle, name="Cycle", 
                   line=dict(color=COLORS["secondary"], width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=500,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05, row=2, col=1)
    
    return fig


# =============================================================================
# SINGLE SERIES CHART (for grid layouts)
# =============================================================================

def plot_single_series(
    series: pd.Series,
    title: str,
    color: str = None,
    height: int = 250,
    show_rangeslider: bool = False
) -> go.Figure:
    """
    Create a compact single-series chart for grid layouts.
    """
    if color is None:
        color = COLORS["primary"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        mode='lines',
        line=dict(color=color, width=1.5),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=height,
        margin=dict(l=30, r=20, t=40, b=30),
        template="plotly_white",
        hovermode="x unified"
    )
    
    if show_rangeslider:
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.1)
    
    return fig


# =============================================================================
# ACF/PACF WITH SIGNIFICANCE BANDS
# =============================================================================

def plot_acf_pacf_compact(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    title: str = "ACF/PACF",
    height: int = 350,
    n_obs: int = 100
) -> go.Figure:
    """ACF/PACF with significance bands (95% CI)."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"), vertical_spacing=0.12)
    
    nlags = len(acf_values)
    lags = np.arange(nlags)
    
    # 95% confidence interval = 1.96 / sqrt(n)
    ci = 1.96 / np.sqrt(n_obs)
    
    # Color bars based on significance
    acf_colors = [COLORS["stationary"] if abs(v) > ci else COLORS["neutral"] for v in acf_values]
    pacf_colors = [COLORS["stationary"] if abs(v) > ci else COLORS["neutral"] for v in pacf_values]
    
    fig.add_trace(
        go.Bar(x=lags, y=acf_values, marker_color=acf_colors, showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=lags, y=pacf_values, marker_color=pacf_colors, showlegend=False),
        row=2, col=1
    )
    
    # Zero line
    fig.add_hline(y=0, line_color="black", row=1, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    
    # Significance bands
    fig.add_hline(y=ci, line_dash="dash", line_color="red", line_width=1, row=1, col=1)
    fig.add_hline(y=-ci, line_dash="dash", line_color="red", line_width=1, row=1, col=1)
    fig.add_hline(y=ci, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=-ci, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        template="plotly_white"
    )
    
    return fig


# =============================================================================
# GRANGER CAUSALITY HEATMAP
# =============================================================================

def plot_granger_heatmap(
    p_values: pd.DataFrame,
    title: str = "Granger Causality Matrix"
) -> go.Figure:
    """
    Create heatmap visualization of Granger causality p-values.
    Green = significant (p < 0.05), Red = not significant.
    
    Args:
        p_values: DataFrame where p_values[cause][effect] = p-value
    """
    # Create color scale: green for low p-values, red for high
    fig = go.Figure(data=go.Heatmap(
        z=p_values.values,
        x=p_values.columns,
        y=p_values.index,
        text=np.round(p_values.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[[0, "#27ae60"], [0.05, "#27ae60"], [0.05, "#f39c12"], [0.1, "#f39c12"], [0.1, "#e74c3c"], [1, "#e74c3c"]],
        zmin=0,
        zmax=0.2,
        colorbar=dict(title="P-Value", tickvals=[0, 0.05, 0.1, 0.2], ticktext=["0", "0.05", "0.1", "0.2+"])
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cause →",
        yaxis_title="← Effect",
        height=max(400, len(p_values) * 50),
        **DEFAULT_LAYOUT
    )
    
    return fig


# =============================================================================
# CORRELATION MATRIX (FULLSCREEN)
# =============================================================================

def plot_correlation_fullscreen(
    df: pd.DataFrame,
    columns: list,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """Create fullscreen correlation heatmap."""
    corr = df[columns].corr()
    n = len(columns)
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="equal"
    )
    
    fig.update_layout(
        title=title,
        height=max(600, n * 45),
        width=None,  # Let it fill container
        margin=dict(l=100, r=50, t=80, b=100),
        template="plotly_white"
    )
    
    # Larger font for readability
    fig.update_traces(textfont_size=11)
    
    return fig


# =============================================================================
# RESIDUALS PLOT
# =============================================================================

def plot_residuals(
    residuals: pd.Series,
    title: str = "Residuals over Time"
) -> go.Figure:
    """Plot residuals vs time with zero line."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=residuals.index,
        y=residuals,
        mode='lines+markers',
        line=dict(color=COLORS["primary"], width=1),
        marker=dict(size=3),
        name="Residuals"
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    # Add +/- 2 std bands
    std = residuals.std()
    fig.add_hline(y=2*std, line_dash="dot", line_color="orange", annotation_text="+2σ")
    fig.add_hline(y=-2*std, line_dash="dot", line_color="orange", annotation_text="-2σ")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Residual",
        height=350,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    
    return fig


# =============================================================================
# SINGLE TARGET IRF - VERTICAL LAYOUT (one per row)
# =============================================================================

def plot_irf_for_target(
    irf_result,
    target_variable: str,
    periods: int = 20
) -> go.Figure:
    """
    Plot IRF for a single target (response) variable from all impulses.
    Creates one subplot per row for better readability.
    """
    names = irf_result.model.names
    n_impulses = len(names)
    target_idx = names.index(target_variable)
    
    fig = make_subplots(
        rows=n_impulses, cols=1,
        subplot_titles=[f"Shock: {imp} → Response: {target_variable}" for imp in names],
        vertical_spacing=0.08
    )
    
    for i, impulse in enumerate(names):
        irf_vals = irf_result.irfs[:periods, target_idx, i]
        x = np.arange(len(irf_vals))
        
        fig.add_trace(
            go.Scatter(x=x, y=irf_vals, line=dict(color=COLORS["primary"], width=2),
                       fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'),
            row=i+1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i+1, col=1)
    
    fig.update_layout(
        title=f"Impulse Response Functions → {target_variable}",
        height=250 * n_impulses,
        showlegend=False,
        **DEFAULT_LAYOUT
    )
    fig.update_xaxes(title_text="Periods")
    
    return fig


# =============================================================================
# MODEL COMPARISON CHART
# =============================================================================

def plot_model_comparison(
    model_data: list,
    metric: str = "RMSE"
) -> go.Figure:
    """
    Create bar chart comparing model performance.
    
    Args:
        model_data: List of dicts with 'name', 'RMSE', 'MAE', 'MAPE'
        metric: Which metric to display ('RMSE', 'MAE', 'MAPE')
    """
    names = [m['name'] for m in model_data]
    values = [m.get(metric, 0) for m in model_data]
    
    # Color gradient based on performance (lower is better)
    colors = [COLORS["stationary"] if v == min(values) else COLORS["primary"] for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Model Comparison: {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        height=400,
        **DEFAULT_LAYOUT
    )
    
    return fig

