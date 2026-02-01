"""
Econometrics Lab v3.1 - Main Application
Streamlit econometrics tool with persistent state, improved visualization, and comprehensive diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go

# Stats & ML imports
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Local utils
from utils.econometrics import (
    check_stationarity, batch_stationarity_test, run_ols_diagnostics,
    run_arima_diagnostics, johansen_test, granger_causality_matrix,
    granger_pvalue_matrix, compute_irf, get_irf_data, calculate_metrics,
    get_acf_pacf, auto_arima_search
)
from utils.data_processing import (
    load_data, load_default_data, apply_transformation, hp_filter,
    make_lags, prepare_ml_features, infer_and_set_frequency, get_lambda_for_frequency
)
from utils.plotting import (
    plot_time_series, plot_series_comparison, plot_correlation_heatmap,
    plot_acf_pacf, plot_forecast, plot_train_test_forecast, plot_irf,
    plot_irf_grid, style_diagnostics_table, plot_hp_decomposition,
    plot_single_series, plot_acf_pacf_compact, plot_granger_heatmap,
    plot_correlation_fullscreen, plot_residuals, plot_irf_for_target,
    plot_model_comparison, COLORS
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Econometrics Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    h1 { color: #2C3E50; }
    h2 { color: #34495E; font-size: 1.5rem; border-bottom: 2px solid #ECF0F1; padding-bottom: 0.5rem; }
    h3 { color: #7F8C8D; font-size: 1.2rem; }
    .sidebar-info { background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 5px solid #3498db; font-size: 0.9em; }
    .diagnostic-pass { background-color: #d4edda; color: #155724; padding: 5px; border-radius: 3px; }
    .diagnostic-fail { background-color: #f8d7da; color: #721c24; padding: 5px; border-radius: 3px; }
    .stat-box { padding: 8px 12px; border-radius: 5px; margin: 2px 0; font-size: 0.85em; }
    .stat-pass { background-color: #d4edda; border-left: 4px solid #27ae60; }
    .stat-fail { background-color: #f8d7da; border-left: 4px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'active_tool': None,
        'tool_history': [],
        'model_history': [],
        # Persistent selections for each tool
        'plot_vars_selected': [],
        'acf_vars_selected': [],
        'corr_vars_selected': [],
        'granger_vars_selected': [],
        'coint_vars_selected': [],
        'stat_vars_selected': [],
        # Transform preview
        'preview_trans': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# UI HELPERS
# =============================================================================

def robust_multiselect(label, options, default=None, key=None, help=None, persist_key=None):
    """Multiselect with Select All/Deselect All and optional persistence."""
    k_all = f"btn_all_{key}"
    k_none = f"btn_none_{key}"
    
    # Use persistent key if provided
    if persist_key and persist_key in st.session_state and st.session_state[persist_key]:
        if key not in st.session_state or not st.session_state[key]:
            st.session_state[key] = st.session_state[persist_key]
    elif key not in st.session_state:
        st.session_state[key] = default if default else []

    def select_all(): 
        st.session_state[key] = list(options)
        if persist_key:
            st.session_state[persist_key] = list(options)
    def deselect_all(): 
        st.session_state[key] = []
        if persist_key:
            st.session_state[persist_key] = []

    c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
    with c1:
        st.markdown(f"<div style='padding-top:5px; font-weight:600'>{label}</div>", unsafe_allow_html=True)
    with c2:
        st.button("All", key=k_all, on_click=select_all, use_container_width=True)
    with c3:
        st.button("None", key=k_none, on_click=deselect_all, use_container_width=True)
    
    selected = st.multiselect(label, options, key=key, help=help, label_visibility="collapsed")
    if persist_key:
        st.session_state[persist_key] = selected
    return selected


def display_diagnostics(diagnostics_df):
    """Display diagnostics table."""
    st.markdown("#### 🔬 Residual Diagnostics")
    for _, row in diagnostics_df.iterrows():
        conclusion = row['Conclusion']
        css_class = "diagnostic-pass" if '✅' in conclusion else "diagnostic-fail" if '❌' in conclusion else ""
        
        cols = st.columns([3, 2, 2, 3])
        cols[0].write(row['Test'])
        cols[1].write(str(row['Statistic']))
        cols[2].write(str(row['P-Value']))
        cols[3].markdown(f"<span class='{css_class}'>{conclusion}</span>", unsafe_allow_html=True)


def navigate_to_tool(tool_name):
    """Set active tool."""
    st.session_state['active_tool'] = tool_name
    if tool_name not in st.session_state['tool_history']:
        st.session_state['tool_history'].append(tool_name)


def add_model_result(name, model_type, rmse, mae, mape):
    """Add model result to history."""
    st.session_state['model_history'].append({
        'name': name, 'type': model_type,
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })


def display_stationarity_metrics(series, var_name):
    """Display ADF and KPSS test results."""
    clean = series.dropna()
    if len(clean) < 10:
        st.warning("Not enough data for tests")
        return
    
    adf = check_stationarity(clean, 'ADF')
    kpss = check_stationarity(clean, 'KPSS')
    
    adf_pass = adf['Conclusion'] == 'Stationary'
    kpss_pass = kpss['Conclusion'] == 'Stationary'
    
    c1, c2 = st.columns(2)
    with c1:
        css = "stat-pass" if adf_pass else "stat-fail"
        emoji = "✅" if adf_pass else "❌"
        st.markdown(f"<div class='stat-box {css}'><b>ADF:</b> p={adf['p-value']:.4f} {emoji}</div>", unsafe_allow_html=True)
    with c2:
        css = "stat-pass" if kpss_pass else "stat-fail"
        emoji = "✅" if kpss_pass else "❌"
        st.markdown(f"<div class='stat-box {css}'><b>KPSS:</b> p={kpss['p-value']:.4f} {emoji}</div>", unsafe_allow_html=True)
    
    return adf_pass and kpss_pass


# =============================================================================
# SIDEBAR - DATA LOADING
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/graph.png", width=80)
    st.title("Data Control")
    
    with st.expander("📁 Data Import", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    df_raw = None
    if uploaded_file:
        df_raw = load_data(uploaded_file)
    elif os.path.exists("DATASET_VF.csv"):
        df_raw = load_default_data("DATASET_VF.csv")
        st.info("📊 Default dataset")
    
    if df_raw is None:
        st.warning("Upload a dataset or add DATASET_VF.csv")
        st.stop()
    
    with st.expander("📅 Date & Frequency", expanded=True):
        date_col = st.selectbox("Date Column", df_raw.columns)
        try:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col])
            df_raw = df_raw.sort_values(by=date_col)
            df_raw.set_index(date_col, inplace=True)
            
            inferred_freq = pd.infer_freq(df_raw.index)
            if inferred_freq:
                st.success(f"Freq: {inferred_freq}")
                df_raw.index.freq = inferred_freq
            else:
                freq_o = st.selectbox("Manual Freq", ["None", "D", "M", "Q", "Y"])
                if freq_o != "None":
                    df_raw = df_raw.asfreq(freq_o)
                    inferred_freq = freq_o
        except Exception as e:
            st.error(f"Error: {e}")
            inferred_freq = None

    if 'data' not in st.session_state or st.sidebar.button("🔄 Reset Data"):
        st.session_state['data'] = df_raw.copy()
        st.session_state['data_original'] = df_raw.copy()
        st.session_state['transform_history'] = []
        st.session_state['data_freq'] = inferred_freq
        st.rerun()


# =============================================================================
# MAIN LAYOUT
# =============================================================================

st.title("📊 Econometrics Lab")

df_curr = st.session_state['data']
numeric_cols = df_curr.select_dtypes(include=[np.number]).columns.tolist()

tabs = st.tabs(["📊 Data & Tests", "🔄 Transformations", "📈 Modeling", "🏆 Comparison"])


# =============================================================================
# TAB 1: DATA & TESTS
# =============================================================================

with tabs[0]:
    main_col, nav_col = st.columns([4, 1])
    
    with nav_col:
        st.markdown("### 🧭 Nav")
        tools = [("📈", "plots", "Series"), ("🔬", "stationarity", "Station."), ("🔗", "corr", "Correl."),
                 ("📊", "acf", "ACF/PACF"), ("🔍", "coint", "Coint."), ("⚡", "granger", "Granger")]
        
        for icon, key, label in tools:
            is_active = st.session_state.get('active_tool') == key
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                navigate_to_tool(key)
                st.rerun()
        
        st.divider()
        if st.button("🏠 Home", use_container_width=True):
            st.session_state['active_tool'] = None
            st.rerun()
    
    with main_col:
        active_tool = st.session_state.get('active_tool')
        
        # HOME
        if active_tool is None:
            st.markdown("## 🏠 Dashboard")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observations", len(df_curr))
            c2.metric("Variables", len(numeric_cols))
            c3.metric("Range", f"{df_curr.index[0].date()} → {df_curr.index[-1].date()}")
            c4.metric("Transforms", len(st.session_state.get('transform_history', [])))
            
            st.divider()
            st.markdown("### 🎯 Select a Tool")
            
            t1, t2, t3 = st.columns(3)
            if t1.button("📈 Time Series", use_container_width=True, type="primary"):
                navigate_to_tool("plots"); st.rerun()
            if t2.button("🔬 Stationarity", use_container_width=True, type="primary"):
                navigate_to_tool("stationarity"); st.rerun()
            if t3.button("🔗 Correlation", use_container_width=True, type="primary"):
                navigate_to_tool("corr"); st.rerun()
            
            t4, t5, t6 = st.columns(3)
            if t4.button("📊 ACF/PACF", use_container_width=True, type="primary"):
                navigate_to_tool("acf"); st.rerun()
            if t5.button("🔍 Cointegration", use_container_width=True, type="primary"):
                navigate_to_tool("coint"); st.rerun()
            if t6.button("⚡ Granger", use_container_width=True, type="primary"):
                navigate_to_tool("granger"); st.rerun()
        
        # TIME SERIES (3-col grid with stationarity)
        elif active_tool == 'plots':
            st.markdown("### 📈 Time Series Visualization")
            plot_vars = robust_multiselect("Select Series", numeric_cols, 
                                           default=numeric_cols[:3], key="ms_plot_vars",
                                           persist_key="plot_vars_selected")
            
            if plot_vars:
                n_cols = 3
                for i in range(0, len(plot_vars), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(cols):
                        if i + j < len(plot_vars):
                            var = plot_vars[i + j]
                            series = df_curr[var].dropna()
                            
                            if len(series) > 10:
                                adf = check_stationarity(series, 'ADF')
                                kpss = check_stationarity(series, 'KPSS')
                                is_stat = adf['Conclusion'] == 'Stationary' and kpss['Conclusion'] == 'Stationary'
                                color = COLORS["stationary"] if is_stat else COLORS["non_stationary"]
                                status = "✅" if is_stat else "❌"
                            else:
                                color, status = COLORS["neutral"], "⚠️"
                                adf, kpss = {'p-value': np.nan}, {'p-value': np.nan}
                            
                            with col:
                                fig = plot_single_series(series, f"{var} {status}", color=color)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Stationarity metrics below chart
                                st.markdown(f"<small><b>ADF p:</b> {adf['p-value']:.3f} | <b>KPSS p:</b> {kpss['p-value']:.3f}</small>", 
                                            unsafe_allow_html=True)
        
        # STATIONARITY SCANNER
        elif active_tool == 'stationarity':
            st.markdown("### 🔬 Stationarity Scanner")
            stat_vars = robust_multiselect("Variables", numeric_cols, default=numeric_cols[:5], 
                                           key="stat_sel", persist_key="stat_vars_selected")
            
            if st.button("Run Batch Tests", type="primary") and stat_vars:
                results_df = batch_stationarity_test(df_curr, stat_vars)
                st.dataframe(results_df, use_container_width=True)
        
        # CORRELATION (Fullscreen)
        elif active_tool == 'corr':
            st.markdown("### 🔗 Correlation Matrix")
            corr_vars = robust_multiselect("Variables", numeric_cols, default=numeric_cols[:8], 
                                           key="corr_sel", persist_key="corr_vars_selected")
            if corr_vars:
                fig = plot_correlation_fullscreen(df_curr, corr_vars)
                st.plotly_chart(fig, use_container_width=True)
        
        # ACF/PACF (2-column grid with significance)
        elif active_tool == 'acf':
            st.markdown("### 📊 ACF / PACF Analysis")
            st.caption("Green bars = significant at 95% CI | Red dashed = significance threshold")
            
            acf_vars = robust_multiselect("Variables", numeric_cols, default=numeric_cols[:4], 
                                          key="acf_sel", persist_key="acf_vars_selected")
            
            if acf_vars:
                n_cols = 2  # 2 columns for better readability
                for i in range(0, len(acf_vars), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(cols):
                        if i + j < len(acf_vars):
                            var = acf_vars[i + j]
                            series = df_curr[var].dropna()
                            n_obs = len(series)
                            acf_vals, pacf_vals = get_acf_pacf(series, nlags=20)
                            with col:
                                fig = plot_acf_pacf_compact(acf_vals, pacf_vals, title=var, n_obs=n_obs)
                                st.plotly_chart(fig, use_container_width=True)
        
        # COINTEGRATION
        elif active_tool == 'coint':
            st.markdown("### 🔍 Cointegration (Johansen)")
            coint_vars = robust_multiselect("System Variables", numeric_cols, 
                                            key="coint_sel", persist_key="coint_vars_selected")
            if len(coint_vars) > 1 and st.button("Run Johansen", type="primary"):
                try:
                    results_df = johansen_test(df_curr[coint_vars])
                    st.table(results_df)
                except Exception as e:
                    st.error(str(e))
        
        # GRANGER (Heatmap with clear labels)
        elif active_tool == 'granger':
            st.markdown("### ⚡ Granger Causality Matrix")
            st.info("**Reading:** Row variable is the **EFFECT**, Column variable is the **CAUSE**. Green = X causes Y (p<0.05)")
            
            gv = robust_multiselect("Variables", numeric_cols, default=numeric_cols[:5], 
                                    key="granger_sel", persist_key="granger_vars_selected")
            maxlag = st.slider("Max Lag", 1, 10, 3)
            
            if len(gv) > 1 and st.button("Compute Causality Matrix", type="primary"):
                with st.spinner("Computing..."):
                    pval_matrix = granger_pvalue_matrix(df_curr, gv, maxlag=maxlag)
                    fig = plot_granger_heatmap(pval_matrix, 
                                               title="Granger Causality (Columns CAUSE Rows)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Interpretation:** If cell (Row Y, Col X) is green, then **X Granger-causes Y**")


# =============================================================================
# TAB 2: TRANSFORMATIONS (with preview and stationarity color)
# =============================================================================

with tabs[1]:
    st.markdown("## 🔄 Transformations")
    
    if st.session_state.get('transform_history'):
        with st.expander(f"📜 History ({len(st.session_state['transform_history'])} transforms)"):
            for i, t in enumerate(reversed(st.session_state['transform_history'])):
                st.markdown(f"{i+1}. **{t['old_name']}** → **{t['new_name']}** (`{t['type']}`)")
            if st.button("⏮️ Undo Last"):
                last = st.session_state['transform_history'].pop()
                st.session_state['data'] = last['backup'].copy()
                st.rerun()
    
    st.divider()
    col_cfg, col_preview = st.columns([1, 2])
    
    with col_cfg:
        st.markdown("### ⚙️ Config")
        target_var = st.selectbox("Variable", numeric_cols, key='trans_var')
        trans_options = ["Log", "First Difference", "Seasonal Diff (12)", "Lag (1)", "HP Filter (Trend)", "HP Filter (Cycle)"]
        trans_op = st.selectbox("Transformation", trans_options, key='trans_op')
        
        suffix_map = {"Log": "_Log", "First Difference": "_Diff", "Seasonal Diff (12)": "_SDiff12", 
                      "Lag (1)": "_Lag1", "HP Filter (Trend)": "_Trend", "HP Filter (Cycle)": "_Cycle"}
        new_col_name = st.text_input("New Column", value=f"{target_var}{suffix_map.get(trans_op, '_Trans')}")
        
        if "HP Filter" in trans_op:
            hp_lambda = st.number_input("Lambda", value=get_lambda_for_frequency(st.session_state.get('data_freq')), min_value=1)
        else:
            hp_lambda = 1600
        
        keep_original = st.checkbox("Keep original", value=True)
        
        st.divider()
        if st.button("🔍 Preview", use_container_width=True):
            st.session_state['preview_trans'] = True
        
        if st.button("✅ APPLY", type="primary", use_container_width=True):
            backup = st.session_state['data'].copy()
            try:
                type_map = {"Log": "Log", "First Difference": "Diff", "Seasonal Diff (12)": "SDiff",
                            "Lag (1)": "Lag", "HP Filter (Trend)": "HP_Trend", "HP Filter (Cycle)": "HP_Cycle"}
                internal_type = type_map[trans_op]
                
                kwargs = {'periods': 12} if internal_type == "SDiff" else {'periods': 1} if internal_type == "Lag" else {}
                if internal_type in ["HP_Trend", "HP_Cycle"]:
                    kwargs['lamb'] = hp_lambda
                
                df_new, created_name, record = apply_transformation(
                    st.session_state['data'], target_var, internal_type,
                    new_name=new_col_name, keep_original=keep_original, **kwargs)
                
                record['backup'] = backup
                st.session_state['data'] = df_new
                st.session_state['transform_history'].append(record)
                st.session_state['preview_trans'] = False
                st.success(f"Created: {created_name}")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        
        if st.button("🔄 Reset All"):
            st.session_state['data'] = st.session_state['data_original'].copy()
            st.session_state['transform_history'] = []
            st.rerun()
    
    with col_preview:
        st.markdown("### 👁️ Preview")
        original = df_curr[target_var].dropna()
        
        if st.session_state.get('preview_trans', False):
            try:
                # Compute transformation
                if trans_op == "Log":
                    transformed = np.log(original.where(original > 0))
                elif trans_op == "First Difference":
                    transformed = original.diff()
                elif trans_op == "Seasonal Diff (12)":
                    transformed = original.diff(12)
                elif trans_op == "Lag (1)":
                    transformed = original.shift(1)
                else:  # HP Filter
                    trend, cycle = hp_filter(original, hp_lambda)
                    transformed = trend if "Trend" in trans_op else cycle
                
                transformed = transformed.dropna()
                
                # Stationarity check for coloring
                orig_stat = check_stationarity(original, 'ADF')['Conclusion'] == 'Stationary'
                trans_stat = check_stationarity(transformed, 'ADF')['Conclusion'] == 'Stationary' if len(transformed) > 10 else False
                
                orig_color = COLORS["stationary"] if orig_stat else COLORS["non_stationary"]
                trans_color = COLORS["stationary"] if trans_stat else COLORS["non_stationary"]
                
                # Plot comparison
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=original.index, y=original, name=f"{target_var} (Original)",
                                         line=dict(color=orig_color, width=2)))
                fig.add_trace(go.Scatter(x=transformed.index, y=transformed, name=f"{new_col_name} (Transformed)",
                                         line=dict(color=trans_color, width=2, dash='dot')))
                fig.update_layout(title="Before vs After (Color = Stationarity)", height=350,
                                  hovermode="x unified", template="plotly_white")
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Original:** {'✅ Stationary' if orig_stat else '❌ Non-Stationary'}")
                    st.metric("Mean", f"{original.mean():.3f}")
                with c2:
                    st.markdown(f"**Transformed:** {'✅ Stationary' if trans_stat else '❌ Non-Stationary'}")
                    st.metric("Mean", f"{transformed.mean():.3f}")
                
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            # Show current series
            is_stat = check_stationarity(original, 'ADF')['Conclusion'] == 'Stationary'
            color = COLORS["stationary"] if is_stat else COLORS["non_stationary"]
            fig = plot_single_series(original, f"{target_var} ({'✅' if is_stat else '❌'})", color=color, height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Click 'Preview' to see before/after comparison with stationarity colors")


# =============================================================================
# TAB 3: MODELING
# =============================================================================

with tabs[2]:
    st.markdown("## 📐 Modeling & Forecasting")
    
    mode = st.radio("Mode", ["📊 Evaluate", "🔮 Forecast"], horizontal=True, label_visibility="collapsed")
    is_forecast = "Forecast" in mode
    
    if not is_forecast:
        test_size = st.slider("Test Size", 0.05, 0.40, 0.20)
    else:
        f_h = st.slider("Horizon", 1, 60, 12)
    
    st.divider()
    col_param, col_res = st.columns([1, 2])
    
    with col_param:
        st.markdown("### Config")
        
        if is_forecast:
            m_type = st.selectbox("Model", ["ARIMA", "VAR", "Random Forest", "XGBoost"])
        else:
            m_type = st.selectbox("Model", ["OLS", "SARIMAX", "VAR", "VECM"])
        
        st.divider()
        run_btn = False
        
        # EVALUATE CONTROLS
        if not is_forecast and m_type == "OLS":
            y_ols = st.selectbox("Target", numeric_cols, key="ols_y")
            X_ols = robust_multiselect("Features", [c for c in numeric_cols if c != y_ols], key="ols_x")
            st.markdown(f"**Train:** {100-int(test_size*100)}% | **Test:** {int(test_size*100)}%")
            run_btn = st.button("Estimate", type="primary", use_container_width=True)
        
        elif not is_forecast and m_type == "SARIMAX":
            y_sari = st.selectbox("Target", numeric_cols, key="sari_y")
            x_sari = robust_multiselect("Exogenous", [c for c in numeric_cols if c != y_sari], key="sari_x")
            c1, c2, c3 = st.columns(3)
            p = c1.number_input("p", 0, 5, 1)
            d = c2.number_input("d", 0, 2, 1)
            q = c3.number_input("q", 0, 5, 1)
            
            if st.button("🔍 Auto (p,d,q)"):
                with st.spinner("Searching..."):
                    result = auto_arima_search(df_curr[y_sari])
                    st.session_state['auto_pdq'] = result['order']
                    st.success(f"Best: {result['order']}")
            
            if 'auto_pdq' in st.session_state and st.checkbox("Use auto"):
                p, d, q = st.session_state['auto_pdq']
            
            run_btn = st.button("Estimate", type="primary", use_container_width=True)
        
        elif not is_forecast and m_type == "VAR":
            v_sys = robust_multiselect("System", numeric_cols, key="var_sys")
            show_irf = st.checkbox("Show IRF", value=True)
            if show_irf and v_sys:
                irf_target = st.selectbox("IRF Target", v_sys, key="irf_tgt")
            run_btn = st.button("Estimate", type="primary", use_container_width=True)
        
        elif not is_forecast and m_type == "VECM":
            v_sys = robust_multiselect("System", numeric_cols, key="vecm_sys")
            r_rank = st.number_input("Coint Rank", 1, 5, 1)
            show_irf = st.checkbox("Show IRF", value=True)
            if show_irf and v_sys:
                irf_target = st.selectbox("IRF Target", v_sys, key="vecm_irf_tgt")
            run_btn = st.button("Estimate", type="primary", use_container_width=True)
        
        # FORECAST CONTROLS
        elif is_forecast and m_type == "ARIMA":
            f_var = st.selectbox("Series", numeric_cols, key="f_arima_var")
            c1, c2, c3 = st.columns(3)
            fp = c1.number_input("p", 0, 5, 1, key="fp")
            fd = c2.number_input("d", 0, 2, 0, key="fd")
            fq = c3.number_input("q", 0, 5, 1, key="fq")
            
            if st.button("🔍 Auto"):
                result = auto_arima_search(df_curr[f_var])
                st.session_state['f_auto'] = result['order']
                st.success(f"Best: {result['order']}")
            if 'f_auto' in st.session_state and st.checkbox("Use auto", key="use_f_auto"):
                fp, fd, fq = st.session_state['f_auto']
            
            run_btn = st.button("Forecast 🚀", type="primary", use_container_width=True)
        
        elif is_forecast and m_type == "VAR":
            f_sys = robust_multiselect("System", numeric_cols, key="f_var_sys")
            run_btn = st.button("Forecast 🚀", type="primary", use_container_width=True)
        
        elif is_forecast and m_type in ["Random Forest", "XGBoost"]:
            f_t = st.selectbox("Target", numeric_cols, key="f_ml_t")
            f_x = robust_multiselect("Predictors", [c for c in numeric_cols if c != f_t], key="f_ml_x")
            n_lags = st.number_input("Lags", 1, 10, 3)
            st.caption(f"**{m_type}** with recursive forecasting")
            run_btn = st.button("Forecast 🚀", type="primary", use_container_width=True)
    
    with col_res:
        st.markdown("### Results")
        
        # ===================== OLS =====================
        if not is_forecast and m_type == "OLS" and run_btn:
            data = df_curr[[y_ols] + X_ols].dropna() if X_ols else df_curr[[y_ols]].dropna()
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test
            
            y_all = data[y_ols]
            X_all = sm.add_constant(data[X_ols]) if X_ols else sm.add_constant(np.ones(len(data)))
            
            X_train, X_test = X_all.iloc[:n_train], X_all.iloc[n_train:]
            y_train, y_test = y_all.iloc[:n_train], y_all.iloc[n_train:]
            
            mod = sm.OLS(y_train, X_train).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            pred = mod.predict(X_test)
            
            rmse, mae, mape = calculate_metrics(y_test, pred)
            add_model_result(f"OLS: {y_ols}", "OLS", rmse, mae, mape)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAE", f"{mae:.4f}")
            c3.metric("MAPE", f"{mape:.2%}")
            
            display_diagnostics(run_ols_diagnostics(mod))
            
            fig = plot_train_test_forecast(y_train, y_test, pred)
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals
            resid = pd.Series(mod.resid, index=y_train.index)
            fig_r = plot_residuals(resid)
            st.plotly_chart(fig_r, use_container_width=True)
            
            with st.expander("📄 Full Summary"):
                st.code(mod.summary().as_text())
                st.download_button("⬇️ Download", mod.summary().as_text(), "ols_summary.txt")
        
        # ===================== SARIMAX =====================
        elif not is_forecast and m_type == "SARIMAX" and run_btn:
            cols = [y_sari] + (x_sari if x_sari else [])
            data = df_curr[cols].dropna()
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test
            
            endog_train = data[y_sari].iloc[:n_train]
            endog_test = data[y_sari].iloc[n_train:]
            exog_train = data[x_sari].iloc[:n_train] if x_sari else None
            exog_test = data[x_sari].iloc[n_train:] if x_sari else None
            
            try:
                mod = ARIMA(endog_train, exog=exog_train, order=(p, d, q)).fit()
                pred = mod.forecast(steps=n_test, exog=exog_test)
                
                rmse, mae, mape = calculate_metrics(endog_test, pred)
                add_model_result(f"ARIMA({p},{d},{q})", "ARIMA", rmse, mae, mape)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse:.4f}")
                c2.metric("MAE", f"{mae:.4f}")
                c3.metric("MAPE", f"{mape:.2%}")
                
                display_diagnostics(run_arima_diagnostics(mod))
                
                pred_series = pd.Series(pred, index=endog_test.index)
                fig = plot_train_test_forecast(endog_train, endog_test, pred_series)
                st.plotly_chart(fig, use_container_width=True)
                
                resid = pd.Series(mod.resid, index=endog_train.index)
                fig_r = plot_residuals(resid)
                st.plotly_chart(fig_r, use_container_width=True)
                
                with st.expander("📄 Full Summary"):
                    st.code(mod.summary().as_text())
                    st.download_button("⬇️ Download", mod.summary().as_text(), "arima_summary.txt")
            except Exception as e:
                st.error(str(e))
        
        # ===================== VAR =====================
        elif not is_forecast and m_type == "VAR" and run_btn:
            if len(v_sys) < 2:
                st.error("VAR needs ≥2 variables")
            else:
                data = df_curr[v_sys].dropna()
                n_test = int(len(data) * test_size)
                d_train, d_test = data.iloc[:-n_test], data.iloc[-n_test:]
                
                try:
                    res = VAR(d_train).fit(maxlags=10, ic='aic')
                    lag = res.k_ar
                    pred = res.forecast(d_train.values[-lag:], steps=n_test)
                    pred_df = pd.DataFrame(pred, index=d_test.index, columns=v_sys)
                    
                    st.markdown("#### Accuracy")
                    for col in v_sys:
                        r, m, mp = calculate_metrics(d_test[col], pred_df[col])
                        add_model_result(f"VAR: {col}", "VAR", r, m, mp)
                        st.write(f"**{col}**: RMSE={r:.3f}, MAPE={mp:.2%}")
                    
                    if show_irf:
                        st.markdown(f"#### IRF → {irf_target}")
                        irf = compute_irf(res, periods=20)
                        fig = plot_irf_for_target(irf, irf_target)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📄 Full Summary"):
                        st.code(res.summary().as_text())
                        st.download_button("⬇️ Download", res.summary().as_text(), "var_summary.txt")
                except Exception as e:
                    st.error(str(e))
        
        # ===================== VECM =====================
        elif not is_forecast and m_type == "VECM" and run_btn:
            if len(v_sys) < 2:
                st.error("VECM needs ≥2 variables")
            else:
                data = df_curr[v_sys].dropna()
                n_test = int(len(data) * test_size)
                d_train, d_test = data.iloc[:-n_test], data.iloc[-n_test:]
                
                try:
                    mod = VECM(d_train, coint_rank=r_rank, k_ar_diff=1)
                    res = mod.fit()
                    pred = res.predict(steps=n_test)
                    pred_df = pd.DataFrame(pred, index=d_test.index, columns=v_sys)
                    
                    st.markdown("#### Accuracy")
                    for col in v_sys:
                        r, m, mp = calculate_metrics(d_test[col], pred_df[col])
                        add_model_result(f"VECM: {col}", "VECM", r, m, mp)
                        st.write(f"**{col}**: RMSE={r:.3f}")
                    
                    if show_irf:
                        st.markdown(f"#### IRF → {irf_target}")
                        irf = res.irf(periods=20)
                        fig = plot_irf_for_target(irf, irf_target)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📄 Full Summary"):
                        st.code(res.summary().as_text())
                        st.download_button("⬇️ Download", res.summary().as_text(), "vecm_summary.txt")
                except Exception as e:
                    st.error(str(e))
        
        # ===================== ARIMA FORECAST =====================
        elif is_forecast and m_type == "ARIMA" and run_btn:
            s = df_curr[f_var].dropna()
            
            try:
                # Backtest
                n_back = min(f_h, len(s) // 4)
                train_bt = s.iloc[:-n_back]
                test_bt = s.iloc[-n_back:]
                
                mod_bt = ARIMA(train_bt, order=(fp, fd, fq)).fit()
                pred_bt = mod_bt.forecast(steps=n_back)
                rmse_bt, mae_bt, mape_bt = calculate_metrics(test_bt, pred_bt)
                
                st.markdown("#### 📊 Backtest")
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse_bt:.4f}")
                c2.metric("MAE", f"{mae_bt:.4f}")
                c3.metric("MAPE", f"{mape_bt:.2%}")
                add_model_result(f"ARIMA({fp},{fd},{fq}) Forecast", "ARIMA", rmse_bt, mae_bt, mape_bt)
                
                # Forecast
                mod = ARIMA(s, order=(fp, fd, fq)).fit()
                fcast = mod.get_forecast(steps=f_h)
                mean = fcast.predicted_mean
                ci = fcast.conf_int()
                
                freq = pd.infer_freq(s.index) or (s.index[1] - s.index[0])
                idx = pd.date_range(s.index[-1], periods=f_h+1, freq=freq)[1:]
                
                forecast_series = pd.Series(mean.values, index=idx)
                lower = pd.Series(ci.iloc[:, 0].values, index=idx)
                upper = pd.Series(ci.iloc[:, 1].values, index=idx)
                
                fig = plot_forecast(s, forecast_series, lower, upper)
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals
                resid = pd.Series(mod.resid, index=s.index)
                fig_r = plot_residuals(resid)
                st.plotly_chart(fig_r, use_container_width=True)
                
                with st.expander("📄 Summary & Export"):
                    st.code(mod.summary().as_text())
                    st.download_button("⬇️ Summary", mod.summary().as_text(), "arima_summary.txt")
                    st.download_button("⬇️ Forecast CSV", pd.DataFrame({'Forecast': forecast_series, 'Lower': lower, 'Upper': upper}).to_csv(), "forecast.csv")
                    
            except Exception as e:
                st.error(str(e))
        
        # ===================== VAR FORECAST =====================
        elif is_forecast and m_type == "VAR" and run_btn:
            if not f_sys or len(f_sys) < 2:
                st.error("VAR needs ≥2 variables")
            else:
                try:
                    df = df_curr[f_sys].dropna()
                    
                    # Backtest
                    n_back = min(f_h, len(df) // 4)
                    train_bt = df.iloc[:-n_back]
                    test_bt = df.iloc[-n_back:]
                    
                    mod_bt = VAR(train_bt).fit(maxlags=10, ic='aic')
                    pred_bt = mod_bt.forecast(train_bt.values[-mod_bt.k_ar:], steps=n_back)
                    
                    st.markdown("#### 📊 Backtest")
                    for i, col in enumerate(f_sys):
                        rmse, _, mape = calculate_metrics(test_bt[col], pred_bt[:, i])
                        add_model_result(f"VAR FC: {col}", "VAR", rmse, 0, mape)
                        st.write(f"**{col}**: RMSE={rmse:.3f}")
                    
                    # Full forecast
                    mod = VAR(df).fit(maxlags=10, ic='aic')
                    pred = mod.forecast(df.values[-mod.k_ar:], steps=f_h)
                    
                    freq = pd.infer_freq(df.index) or (df.index[1] - df.index[0])
                    idx = pd.date_range(df.index[-1], periods=f_h+1, freq=freq)[1:]
                    pred_df = pd.DataFrame(pred, index=idx, columns=f_sys)
                    
                    fig = go.Figure()
                    for col in f_sys:
                        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=f"{col} (History)"))
                        fig.add_trace(go.Scatter(x=idx, y=pred_df[col], name=f"{col} (Forecast)", line=dict(dash='dot')))
                    fig.update_layout(title="VAR Forecast", height=450)
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📄 Summary & Export"):
                        st.code(mod.summary().as_text())
                        st.download_button("⬇️ Summary", mod.summary().as_text(), "var_summary.txt")
                        st.download_button("⬇️ Forecast CSV", pred_df.to_csv(), "var_forecast.csv")
                        
                except Exception as e:
                    st.error(str(e))
        
        # ===================== ML FORECAST =====================
        elif is_forecast and m_type in ["Random Forest", "XGBoost"] and run_btn:
            if not f_x:
                st.error("ML needs predictors")
            else:
                try:
                    X, y, feature_cols = prepare_ml_features(df_curr, f_t, f_x, n_lags=n_lags)
                    
                    # Backtest via CV
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_rmse = []
                    
                    for train_idx, val_idx in tscv.split(X):
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model = RandomForestRegressor(n_estimators=100, random_state=42) if m_type == "Random Forest" else xgb.XGBRegressor(n_estimators=100, verbosity=0)
                        model.fit(X_tr, y_tr)
                        cv_rmse.append(np.sqrt(np.mean((y_val - model.predict(X_val))**2)))
                    
                    st.markdown("#### 📊 Backtest (CV)")
                    st.metric("CV RMSE", f"{np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
                    add_model_result(f"{m_type} FC", m_type, np.mean(cv_rmse), 0, 0)
                    
                    # Train final
                    final_model = RandomForestRegressor(n_estimators=100, random_state=42) if m_type == "Random Forest" else xgb.XGBRegressor(n_estimators=100, verbosity=0)
                    final_model.fit(X, y)
                    
                    # Recursive forecast
                    last_values = df_curr[f_t].dropna().tail(n_lags).tolist()
                    last_exog = df_curr[f_x].dropna().iloc[-1].values
                    
                    forecasts = []
                    for _ in range(f_h):
                        features = np.concatenate([last_exog, last_values[-n_lags:][::-1]]).reshape(1, -1)
                        pred = final_model.predict(features)[0]
                        forecasts.append(pred)
                        last_values.append(pred)
                    
                    freq = pd.infer_freq(df_curr.index) or (df_curr.index[1] - df_curr.index[0])
                    idx = pd.date_range(df_curr.index[-1], periods=f_h+1, freq=freq)[1:]
                    forecast_series = pd.Series(forecasts, index=idx)
                    
                    history = df_curr[f_t].dropna()
                    fig = plot_forecast(history, forecast_series, title=f"{m_type} Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.download_button("⬇️ Forecast CSV", pd.DataFrame({'Forecast': forecast_series}).to_csv(), f"{m_type.lower()}_forecast.csv")
                    
                except Exception as e:
                    st.error(str(e))


# =============================================================================
# TAB 4: COMPARISON
# =============================================================================

with tabs[3]:
    st.markdown("## 🏆 Model Comparison")
    
    if st.session_state['model_history']:
        model_df = pd.DataFrame(st.session_state['model_history'])
        st.dataframe(model_df, use_container_width=True)
        
        if len(st.session_state['model_history']) > 1:
            fig = plot_model_comparison(st.session_state['model_history'])
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🗑️ Clear"):
            st.session_state['model_history'] = []
            st.rerun()
    else:
        st.info("No models fitted yet.")


# =============================================================================
# SIDEBAR INFO
# =============================================================================

st.sidebar.divider()
st.sidebar.markdown(f"<div class='sidebar-info'>📊 {len(numeric_cols)} vars | 📅 {len(df_curr)} obs</div>", unsafe_allow_html=True)
st.sidebar.caption("Econometrics Lab v3.1")
