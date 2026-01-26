
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Econometrics & Stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, acf, pacf, ccf
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Configuration
st.set_page_config(
    page_title="Econometrics Data Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CSS STYLING ---
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Custom cards for clearer sections */
    .stCard {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Header branding */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    h1 { color: #2C3E50; }
    h2 { color: #34495E; font-size: 1.8rem; border-bottom: 2px solid #ECF0F1; padding-bottom: 0.5rem;}
    h3 { color: #7F8C8D; font-size: 1.4rem; }
    
    /* Sidebar info boxes */
    .sidebar-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING & UTILS ---
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

def check_stationarity(series, test_type='ADF'):
    series = series.dropna()
    if series.empty:
        return {'Conclusion': 'Error', 'p-value': np.nan, 'Test Statistic': 0}
    
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

def make_lags(df, target_col, lags):
    df_lagged = df.copy()
    for i in range(1, lags + 1):
        df_lagged[f'{target_col}_lag_{i}'] = df_lagged[target_col].shift(i)
    return df_lagged.dropna()

def plot_metric_card(label, value, delta=None, color="normal"):
    st.metric(label=label, value=value, delta=delta, delta_color=color)

def robust_multiselect(label, options, default=None, key=None, help=None):
    """
    Multiselect with Select All / Deselect All buttons (Optimized Layout).
    """
    k_all = f"btn_all_{key}"
    k_none = f"btn_none_{key}"
    
    if key not in st.session_state:
        st.session_state[key] = default if default else []

    def select_all(): st.session_state[key] = list(options)
    def deselect_all(): st.session_state[key] = []

    # Layout: Label (60%) | All (20%) | None (20%)
    c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
    
    with c1:
        st.markdown(f"<div style='padding-top:5px; font-weight:600'>{label}</div>", unsafe_allow_html=True)
    with c2:
        st.button("All", key=k_all, on_click=select_all, use_container_width=True)
    with c3:
        st.button("None", key=k_none, on_click=deselect_all, use_container_width=True)
        
    return st.multiselect(label, options, key=key, help=help, label_visibility="collapsed")

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/graph.png", width=80)
    st.title("Data Control")
    
    with st.expander("📁 Data Import", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df_raw = load_data(uploaded_file)
        
        with st.expander("📅 Date & Frequency", expanded=True):
            date_col = st.selectbox("Date Column", df_raw.columns)
            try:
                df_raw[date_col] = pd.to_datetime(df_raw[date_col])
                df_raw = df_raw.sort_values(by=date_col)
                df_raw.set_index(date_col, inplace=True)
                
                # Freq
                inferred_freq = pd.infer_freq(df_raw.index)
                if inferred_freq:
                    st.success(f"Freq: {inferred_freq}")
                    df_raw.index.freq = inferred_freq
                else:
                    freq_o = st.selectbox("Manual Freq", ["None", "D", "M", "Q", "Y", "H"])
                    if freq_o != "None":
                        df_raw = df_raw.asfreq(freq_o)
            except Exception as e:
                st.error(f"Date Error: {e}")

        # Session State Init with Backup
        if 'data' not in st.session_state or st.sidebar.button("🔄 Reset Data"):
             st.session_state['data'] = df_raw.copy()
             st.session_state['data_original'] = df_raw.copy()  # Backup
             st.session_state['transform_history'] = []  # Track transformations
             st.rerun()
             
    else:
        st.stop()

# --- MAIN LAYOUT ---
st.title("📊 Econometrics Lab")

# Tabs with Icons
tabs = st.tabs([
    "📊 Data Preview & Tests", 
    "🔄 Transformations", 
    "📈 Modeling & Forecasting"
])

df_curr = st.session_state['data']
# Update numeric_cols based on CURRENT data (after transformations)
numeric_cols = df_curr.select_dtypes(include=[np.number]).columns.tolist()
st.sidebar.markdown(f"<div class='sidebar-info'>Loaded <b>{len(numeric_cols)}</b> numeric variables.<br>Rows: {len(df_curr)}</div>", unsafe_allow_html=True)


# ================= TAB 1: DATA PREVIEW & TESTS =================
with tabs[0]:
    st.markdown("## 🏠 Data Overview & Testing Dashboard")
    
    # Data Summary (Always Visible)
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    col_sum1.metric("Observations", len(df_curr))
    col_sum2.metric("Variables", len(numeric_cols))
    col_sum3.metric("Date Range", f"{df_curr.index[0].date()} to {df_curr.index[-1].date()}")
    if st.session_state.get('transform_history'):
        col_sum4.metric("Transformations", len(st.session_state['transform_history']), delta="Applied")
    else:
        col_sum4.metric("Transformations", "0", delta="None")
    
    st.divider()
    st.markdown("### 🎯 Select a Tool")
    
    # === TILE-BASED NAVIGATION ===
    # Row 1
    tile_col1, tile_col2, tile_col3 = st.columns(3)
    
    with tile_col1:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>📈</h2>
                <h4 style='color:white; margin:10px 0'>Time Series Plots</h4>
                <p style='color:#f0f0f0'>Visualize series over time</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_plots", use_container_width=True):
                st.session_state['active_tool'] = 'plots'
    
    with tile_col2:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>🔬</h2>
                <h4 style='color:white; margin:10px 0'>Stationarity Tests</h4>
                <p style='color:#f0f0f0'>ADF & KPSS Testing</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_station", use_container_width=True):
                st.session_state['active_tool'] = 'stationarity'
    
    with tile_col3:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>🔗</h2>
                <h4 style='color:white; margin:10px 0'>Correlation Matrix</h4>
                <p style='color:#f0f0f0'>Heatmap & Relationships</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_corr", use_container_width=True):
                st.session_state['active_tool'] = 'correlation'
    
    # Row 2
    tile_col4, tile_col5, tile_col6 = st.columns(3)
    
    with tile_col4:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>📊</h2>
                <h4 style='color:white; margin:10px 0'>ACF / PACF</h4>
                <p style='color:#f0f0f0'>Autocorrelation Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_acf", use_container_width=True):
                st.session_state['active_tool'] = 'acf'
    
    with tile_col5:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #30cfd0 0%, #330867 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>🔍</h2>
                <h4 style='color:white; margin:10px 0'>Cointegration</h4>
                <p style='color:#f0f0f0'>Johansen Test</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_coint", use_container_width=True):
                st.session_state['active_tool'] = 'cointegration'
    
    with tile_col6:
        with st.container():
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius:10px; text-align:center'>
                <h2 style='color:white; margin:0'>⚡</h2>
                <h4 style='color:white; margin:10px 0'>Granger Causality</h4>
                <p style='color:#f0f0f0'>Causal Relationships</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open 👉", key="tile_granger", use_container_width=True):
                st.session_state['active_tool'] = 'granger'
    
    st.divider()
    
    # === TOOL CONTENT (Based on Selection) ===
    active_tool = st.session_state.get('active_tool', None)
    
    if active_tool == 'plots':
        st.markdown("### 📈 Time Series Visualization")
        plot_vars = robust_multiselect("Select Series", df_curr.columns, default=numeric_cols[:1], key="ms_plot_vars")
        
        if plot_vars:
            for v in plot_vars:
                clean = df_curr[v].dropna()
                if len(clean) > 5:
                    adf = check_stationarity(clean, 'ADF')
                    kpss_ = check_stationarity(clean, 'KPSS')
                    is_stat = (adf['Conclusion'] == 'Stationary') and (kpss_['Conclusion'] == 'Stationary')
                    color = "#27ae60" if is_stat else "#c0392b"
                    status = "STATIONARY" if is_stat else "NON-STATIONARY"
                else:
                    color = "#7f8c8d"
                    status = "No Data"

                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### {v}")
                c2.markdown(f"<h4 style='color:{color}; text-align:right'>{status}</h4>", unsafe_allow_html=True)
                
                fig = px.line(df_curr, y=v)
                fig.update_traces(line_color=color, line_width=2)
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Obs", len(clean))
                m2.metric("Mean", f"{clean.mean():.2f}")
                m3.metric("ADF p-val", f"{adf['p-value']:.3f}")
                m4.metric("KPSS p-val", f"{kpss_['p-value']:.3f}")
                st.divider()
    
    elif active_tool == 'stationarity':
        st.markdown("### 🔬 Stationarity Scanner")
        stat_vars = robust_multiselect("Select Variables", df_curr.columns, default=numeric_cols[:2], key="stat_sel_tool")
        if st.button("Run Batch Tests", use_container_width=True):
            res_data = []
            for v in stat_vars:
                s = df_curr[v].dropna()
                a = check_stationarity(s, 'ADF')
                k = check_stationarity(s, 'KPSS')
                
                stat_a = a['Conclusion'] == 'Stationary'
                stat_k = k['Conclusion'] == 'Stationary'
                
                if stat_a and stat_k: syn, icon = "Stable", "✅"
                elif (not stat_a) and (not stat_k): syn, icon = "Unstable", "❌"
                else: syn, icon = "Ambiguous", "⚠️"
                
                res_data.append({
                    "Variable": v,
                    "ADF": a['p-value'],
                    "KPSS": k['p-value'],
                    "Result": f"{icon} {syn}"
                })
            
            st.dataframe(pd.DataFrame(res_data).style.format({"ADF": "{:.3f}", "KPSS": "{:.3f}"}).background_gradient(subset=['ADF'], cmap='Reds'), use_container_width=True)
    
    elif active_tool == 'correlation':
        st.markdown("### 🔗 Correlation Matrix")
        corr_subset = robust_multiselect("Variables", df_curr.columns, default=numeric_cols[:5], key="ms_corr_tool")
        if corr_subset:
            st.plotly_chart(px.imshow(df_curr[corr_subset].corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
    
    elif active_tool == 'acf':
        st.markdown("### 📊 ACF / PACF Analysis")
        c_var = st.selectbox("Select Variable", df_curr.columns)
        clean = df_curr[c_var].dropna()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(x=np.arange(40), y=acf(clean, nlags=39), title="ACF"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(x=np.arange(40), y=pacf(clean, nlags=39), title="PACF"), use_container_width=True)
    
    elif active_tool == 'cointegration':
        st.markdown("### 🔍 Cointegration (Johansen Test)")
        coint_v = robust_multiselect("System Variables", df_curr.columns, key="coint_sel_tool")
        if len(coint_v) > 1 and st.button("Run Johansen", use_container_width=True):
             try:
                cj = coint_johansen(df_curr[coint_v].dropna(), 0, 1)
                dt = [{"r": i, "Trace": cj.lr1[i], "95%": cj.cvt[i,1], "Sig": "Yes" if cj.lr1[i] > cj.cvt[i,1] else "No"} for i in range(len(cj.lr1))]
                st.table(pd.DataFrame(dt))
             except Exception as e: st.error(str(e))
    
    elif active_tool == 'granger':
        st.markdown("### ⚡ Granger Causality Matrix")
        gv = robust_multiselect("Pairwise Variables", df_curr.columns, key="granger_sel_tool")
        if len(gv) > 1 and st.button("Run Causality Matrix", use_container_width=True):
            for y in gv:
                for x in gv:
                    if x == y: continue
                    try:
                        res = grangercausalitytests(df_curr[[y, x]].dropna(), maxlag=3, verbose=False)
                        p_min = min([res[i][0]['ssr_ftest'][1] for i in res])
                        if p_min < 0.05:
                            st.success(f"**{x}** causes **{y}** (p={p_min:.3f})")
                    except: pass
    
    else:
        st.info("👆 Click a tile above to start exploring your data!")

# ================= TAB 2: TRANSFORMATIONS =================
with tabs[1]:
    st.markdown("## 🔄 Data Transformations Studio")
    st.info("⚠️ Transformations **replace** the original series permanently. Use the backup/undo feature if needed.")
    
    # Transformation History Display
    if st.session_state.get('transform_history'):
        with st.expander(f"📜 Transformation History ({len(st.session_state['transform_history'])} applied)", expanded=False):
            for i, t in enumerate(reversed(st.session_state['transform_history'])):
                old = t.get('old_name', t.get('column', '?'))
                new = t.get('new_name', '?')
                st.markdown(f"{i+1}. **{old}** → **{new}** (`{t['type']}`) _{t.get('timestamp', 'N/A')}_")
            if st.button("⏮️ Undo Last Transformation"):
                if st.session_state['transform_history']:
                    last = st.session_state['transform_history'].pop()
                    st.session_state['data'] = last['backup'].copy()
                    st.success(f"Undid: {last['type']} on {last.get('old_name', last.get('column', '?'))}")
                    st.rerun()
    
    st.divider()
    
    # Transformation Controls
    col_cfg, col_preview = st.columns([1, 2])
    
    with col_cfg:
        st.markdown("### ⚙️ Configuration")
        
        target_var = st.selectbox("Select Variable", numeric_cols, key='transform_var')
        trans_op = st.selectbox("Transformation Type", options=["Log", "First Difference", "Seasonal Diff (12)", "Lag (1)"], key='transform_op')
        
        st.divider()
        st.markdown("### ⚡ Actions")
        
        if st.button("🔍 Preview Transformation", use_container_width=True):
            st.session_state['preview_trans'] = True
        
        st.markdown("---")
        
        if st.button("✅ APPLY & RENAME SERIES", type="primary", use_container_width=True):
            from datetime import datetime
            
            # Backup current state
            backup = st.session_state['data'].copy()
            old_name = target_var
            
            # Determine new column name and apply transformation
            try:
                if trans_op == "Log":
                    # Check for non-positive values
                    if (st.session_state['data'][target_var] <= 0).any():
                         st.warning(f"⚠️ **{target_var}** contains values <= 0. Log transformation created NaNs (gaps).")
                    
                    new_name = f"{target_var}_Log"
                    st.session_state['data'][new_name] = np.log(st.session_state['data'][target_var])
                    trans_label = "Log"
                elif trans_op == "First Difference":
                    new_name = f"{target_var}_Diff"
                    st.session_state['data'][new_name] = st.session_state['data'][target_var].diff()
                    trans_label = "Diff"
                elif trans_op == "Seasonal Diff (12)":
                    new_name = f"{target_var}_SDiff"
                    st.session_state['data'][new_name] = st.session_state['data'][target_var].diff(12)
                    trans_label = "Seasonal Diff"
                elif trans_op == "Lag (1)":
                    new_name = f"{target_var}_Lag1"
                    st.session_state['data'][new_name] = st.session_state['data'][target_var].shift(1)
                    trans_label = "Lag"
                
                # Remove old column (replace behavior)
                st.session_state['data'] = st.session_state['data'].drop(columns=[old_name])
                
                # Record in history
                st.session_state['transform_history'].append({
                    'old_name': old_name,
                    'new_name': new_name,
                    'type': trans_label,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'backup': backup
                })
                
                st.success(f"✅ **{old_name}** → **{new_name}** ({trans_label})")
                st.session_state['preview_trans'] = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Transformation failed: {e}")
        
        if st.button("🔄 Reset All Data", use_container_width=True):
            if st.session_state.get('data_original') is not None:
                st.session_state['data'] = st.session_state['data_original'].copy()
                st.session_state['transform_history'] = []
                st.success("Data reset to original state!")
                st.rerun()
    
    with col_preview:
        st.markdown("### 👁️ Preview")
        
        if st.session_state.get('preview_trans', False):
            original = df_curr[target_var].dropna()
            
            # Calculate transformed preview
            try:
                if trans_op == "Log":
                    # Check for non-positive values
                    if (original <= 0).any():
                         st.warning(f"⚠️ **{target_var}** contains values <= 0. Log transformation will appear discontinuous (NaNs).")
                    transformed = np.log(original)
                elif trans_op == "First Difference":
                    transformed = original.diff()
                elif trans_op == "Seasonal Diff (12)":
                    transformed = original.diff(12)
                elif trans_op == "Lag (1)":
                    transformed = original.shift(1)
                
                # Test stationarity for coloring
                transformed_clean = transformed.dropna()
                
                # Original stationarity
                if len(original) > 10:
                    adf_orig = check_stationarity(original, 'ADF')
                    kpss_orig = check_stationarity(original, 'KPSS')
                    is_stat_orig = (adf_orig['Conclusion'] == 'Stationary') and (kpss_orig['Conclusion'] == 'Stationary')
                    color_orig = "#27ae60" if is_stat_orig else "#c0392b"
                    status_orig = "✅ Stationary" if is_stat_orig else "❌ Non-Stationary"
                else:
                    color_orig = "#7f8c8d"
                    status_orig = "⚠️ Insufficient Data"
                
                # Transformed stationarity
                if len(transformed_clean) > 10:
                    adf_trans = check_stationarity(transformed_clean, 'ADF')
                    kpss_trans = check_stationarity(transformed_clean, 'KPSS')
                    is_stat_trans = (adf_trans['Conclusion'] == 'Stationary') and (kpss_trans['Conclusion'] == 'Stationary')
                    color_trans = "#27ae60" if is_stat_trans else "#c0392b"
                    status_trans = "✅ Stationary" if is_stat_trans else "❌ Non-Stationary"
                else:
                    color_trans = "#7f8c8d"
                    status_trans = "⚠️ Insufficient Data"
                
                # Side-by-side plot with stationarity colors
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=original.index, y=original, 
                    name=f"Original ({status_orig})", 
                    line=dict(color=color_orig, width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=transformed.index, y=transformed, 
                    name=f"After {trans_op} ({status_trans})", 
                    line=dict(color=color_trans, width=2, dash='dot')
                ))
                fig.update_layout(
                    title=f"Before vs After: {trans_op}",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats comparison with stationarity info
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Original** {status_orig}")
                    st.metric("Mean", f"{original.mean():.3f}")
                    st.metric("Std", f"{original.std():.3f}")
                    if len(original) > 10:
                        st.caption(f"ADF p-val: {adf_orig['p-value']:.3f}")
                        st.caption(f"KPSS p-val: {kpss_orig['p-value']:.3f}")
                with c2:
                    st.markdown(f"**After {trans_op}** {status_trans}")
                    st.metric("Mean", f"{transformed.mean():.3f}")
                    st.metric("Std", f"{transformed.std():.3f}")
                    if len(transformed_clean) > 10:
                        st.caption(f"ADF p-val: {adf_trans['p-value']:.3f}")
                        st.caption(f"KPSS p-val: {kpss_trans['p-value']:.3f}")
                    
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("👈 Click 'Preview Transformation' to see before/after comparison")
            
            # Show current series with stationarity color
            clean = df_curr[target_var].dropna()
            if len(clean) > 10:
                adf = check_stationarity(clean, 'ADF')
                kpss_ = check_stationarity(clean, 'KPSS')
                is_stat = (adf['Conclusion'] == 'Stationary') and (kpss_['Conclusion'] == 'Stationary')
                color = "#27ae60" if is_stat else "#c0392b"
                status = "✅ Stationary" if is_stat else "❌ Non-Stationary"
            else:
                color = "#7f8c8d"
                status = "⚠️ Insufficient Data"
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_curr.index, y=df_curr[target_var], 
                                     name=f"{target_var} ({status})",
                                     line=dict(color=color, width=2)))
            fig.update_layout(title=f"Current State: {target_var} - {status}", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if len(clean) > 10:
                c1, c2 = st.columns(2)
                c1.metric("ADF p-value", f"{adf['p-value']:.3f}")
                c2.metric("KPSS p-value", f"{kpss_['p-value']:.3f}")


# ================= TAB 3: MODELING & FORECASTING =================
with tabs[2]:
    st.markdown("## 📐 Modeling & Forecasting Studio")
    
    # Mode Selection
    mode = st.radio("Mode", ["📊 Evaluate (Train/Test)", "🔮 Forecast Future"], horizontal=True, label_visibility="collapsed")
    is_forecast = "Forecast" in mode
    
    # Global Test/Split (only for Evaluate)
    if not is_forecast:
        test_size = st.slider("Test Set Size", 0.05, 0.40, 0.20, help="Fraction of data reserved for validation.")
    else:
        f_h = st.slider("Forecast Horizon", 1, 60, 12, help="Steps ahead to predict.")
    
    st.divider()
    
    # Layout
    col_param, col_res = st.columns([1, 2])
    
    with col_param:
        st.markdown("### Model Config")
        if is_forecast:
            m_type = st.selectbox("Architecture", ["ARIMA", "VAR", "ML (RF/XGB)"])
        else:
            m_type = st.selectbox("Architecture", ["OLS", "SARIMAX", "VAR", "VECM"])
        st.divider()
        
        # SUB-CONTROLS FOR EVALUATE MODE
        if not is_forecast and m_type == "OLS":
            y_ols = st.selectbox("Target (Y)", df_curr.columns)
            X_ols = robust_multiselect("Features (X)", [c for c in df_curr.columns if c != y_ols], key="ms_ols_x")
            
            # Indicative Auto-Tune Button
            if st.button("Suggest Best Subset (AIC)"):
                with st.spinner("Optimizing..."):
                    curr = X_ols[:] if X_ols else [c for c in df_curr.columns if c != y_ols]
                    best = np.inf
                    loop = True
                    while loop:
                        loop = False
                        for v in curr:
                            test = [x for x in curr if x != v]
                            if not test: 
                                target_data = df_curr[[y_ols]].dropna()
                                m = sm.OLS(target_data, sm.add_constant(np.ones(len(target_data)))).fit()
                            else:
                                data_step = df_curr[[y_ols] + test].dropna()
                                m = sm.OLS(data_step[y_ols], sm.add_constant(data_step[test])).fit()
                            
                            if m.aic < best:
                                best = m.aic
                                curr = test
                                loop = True
                    st.session_state['best_ols_subset'] = curr
                    st.success(f"Suggestion: {curr}")
            
            # Use suggestion if implicit
            if 'best_ols_subset' in st.session_state and st.checkbox("Use Suggested Subset", value=True):
                 X_ols = st.session_state['best_ols_subset']

            run_btn = st.button("Estimate & Evaluate OLS", type="primary", use_container_width=True)
            
        elif not is_forecast and m_type == "SARIMAX":
            y_sari = st.selectbox("Target", df_curr.columns)
            x_sari = robust_multiselect("Exogenous", [c for c in df_curr.columns if c != y_sari], key="ms_sari_x")
            c1, c2, c3 = st.columns(3)
            p = c1.number_input("p", 0, 5, 1)
            d = c2.number_input("d", 0, 2, 1)
            q = c3.number_input("q", 0, 5, 1)
            
            # Indicative Auto-Tune
            if st.button("Suggest Best (p,d,q)"):
                with st.spinner("Grid Searching..."):
                    import itertools
                    s = df_curr[y_sari].dropna()
                    best_aic = np.inf
                    best_ord = (p,d,q)
                    for _p, _d, _q in itertools.product(range(3), range(2), range(3)):
                        try:
                            m = ARIMA(s, order=(_p,_d,_q)).fit()
                            if m.aic < best_aic: best_aic, best_ord = m.aic, (_p,_d,_q)
                        except: pass
                    st.session_state['best_arima'] = best_ord
                    st.success(f"Best: {best_ord} (AIC:{best_aic:.1f})")
            
            if 'best_arima' in st.session_state:
                st.caption(f"Suggested: {st.session_state['best_arima']}")

            run_btn = st.button("Estimate & Evaluate", type="primary", use_container_width=True)
            
        elif not is_forecast and m_type == "VAR":
            v_sys = robust_multiselect("System Variables", df_curr.columns, key="ms_var_sys")
            # VAR auto-selects lags usually, but we can offer a button to show the table
            if st.button("Show Lag Selection Table"):
                 d = df_curr[v_sys].dropna()
                 if not d.empty:
                     x = VAR(d).select_order(maxlags=10)
                     st.table(x.summary())
            
            run_btn = st.button("Estimate VAR", type="primary", use_container_width=True)
            
        elif not is_forecast and m_type == "VECM":
            v_sys = robust_multiselect("System Variables", df_curr.columns, key="ms_vecm_sys")
            r = st.number_input("Rank (r)", 1, 5, 1)
            if st.button("Suggest Rank"):
                cj = coint_johansen(df_curr[v_sys].dropna(), 0, 1)
                sug = sum(cj.lr1 > cj.cvt[:,1])
                st.success(f"Suggested Rank: {sug}")
            run_btn = st.button("Estimate VECM", type="primary", use_container_width=True)

        # FORECAST MODE CONTROLS
        elif is_forecast and m_type == "ARIMA":
            f_var = st.selectbox("Series", df_curr.columns, key="f_v_arima")
            st.info("💡 Tip: Use 'Suggest' for automatic parameter selection, or test stationarity first in Diagnostics tab.")
            c1, c2, c3 = st.columns(3)
            fp = c1.number_input("p", 0, 5, 1, key="f_p2")
            fd = c2.number_input("d", 0, 2, 0, key="f_d2", help="d=0 for stationary, d=1 for trending data")
            fq = c3.number_input("q", 0, 5, 1, key="f_q2")
            if st.button("Suggest Best (p,d,q)", key="suggest_arima_f"):
                with st.spinner("Grid Searching..."):
                    import itertools
                    s = df_curr[f_var].dropna()
                    best_aic = np.inf
                    best_ord = (fp,fd,fq)
                    for _p, _d, _q in itertools.product(range(3), range(2), range(3)):
                        try:
                            m = ARIMA(s, order=(_p,_d,_q)).fit()
                            if m.aic < best_aic: best_aic, best_ord = m.aic, (_p,_d,_q)
                        except: pass
                    st.session_state['f_arima_ord'] = best_ord
                    st.success(f"Best: {best_ord} (AIC:{best_aic:.1f})")
            
            if 'f_arima_ord' in st.session_state:
                st.caption(f"Suggested: {st.session_state['f_arima_ord']}")
            run_btn = st.button("Generate Forecast 🚀", type="primary", use_container_width=True)
            
        elif is_forecast and m_type == "VAR":
            f_sys = robust_multiselect("System", df_curr.columns, key="f_var_sys2")
            run_btn = st.button("Generate Forecast 🚀", type="primary", use_container_width=True)
            
        elif is_forecast and m_type == "ML (RF/XGB)":
            f_t = st.selectbox("Target", df_curr.columns, key="f_ml_t2")
            f_x = robust_multiselect("Predictors", [c for c in df_curr.columns if c!=f_t], key="f_ml_x2")
            run_btn = st.button("Generate Forecast 🚀", type="primary", use_container_width=True)

    with col_res:
        st.markdown("### 📝 Results")
        
        # EVALUATE MODE RESULTS
        if not is_forecast and m_type == "OLS" and run_btn:
             # Data Prep
             if X_ols:
                 data = df_curr[[y_ols] + X_ols].dropna()
                 y_all = data[y_ols]
                 X_all = sm.add_constant(data[X_ols])
             else:
                 data = df_curr[[y_ols]].dropna()
                 y_all = data[y_ols]
                 X_all = sm.add_constant(np.ones(len(data)))
            
             # Split
             n_test = int(len(data) * test_size)
             n_train = len(data) - n_test
             
             X_train, X_test = X_all.iloc[:n_train], X_all.iloc[n_train:]
             y_train, y_test = y_all.iloc[:n_train], y_all.iloc[n_train:]
             
             # Fit
             mod = sm.OLS(y_train, X_train).fit(cov_type='HAC', cov_kwds={'maxlags':1})
             pred = mod.predict(X_test)
             
             # Metrics
             rmse, mae, mape = calculate_metrics(y_test, pred)
             
             c1, c2, c3 = st.columns(3)
             c1.metric("RMSE (Test)", f"{rmse:.4f}")
             c2.metric("MAE (Test)", f"{mae:.4f}")
             c3.metric("MAPE (Test)", f"{mape:.2%}")
             
             st.info(f"R2 (Train): {mod.rsquared:.4f} | AIC: {mod.aic:.2f}")
             
             with st.expander("Full Summary"): st.code(mod.summary().as_text())

             # Visual
             fig = go.Figure()
             fig.add_trace(go.Scatter(x=y_train.index, y=y_train, name="Train", line_color="#3498db"))
             fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Test (Actual)", line_color="#2ecc71"))
             fig.add_trace(go.Scatter(x=y_test.index, y=pred, name="Test (Pred)", line_color="#e74c3c", line_dash="dot"))
             st.plotly_chart(fig, use_container_width=True)
        
        # SARIMAX
        elif not is_forecast and m_type == "SARIMAX" and run_btn:
            # Use suggested order if exists
            final_order = st.session_state.get('best_arima', (p,d,q))
            
            cols = [y_sari] + (x_sari if x_sari else [])
            data = df_curr[cols].dropna()
            
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test
            
            endog_train = data[y_sari].iloc[:n_train]
            endog_test = data[y_sari].iloc[n_train:]
            exog_train = data[x_sari].iloc[:n_train] if x_sari else None
            exog_test = data[x_sari].iloc[n_train:] if x_sari else None
            
            try:
                mod = ARIMA(endog_train, exog=exog_train, order=final_order).fit()
                pred = mod.forecast(steps=n_test, exog=exog_test)
                
                rmse, mae, mape = calculate_metrics(endog_test, pred)
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse:.4f}")
                c2.metric("MAE", f"{mae:.4f}")
                c3.metric("MAPE", f"{mape:.2%}")
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=endog_train.index, y=endog_train, name="Train"))
                fig.add_trace(go.Scatter(x=endog_test.index, y=endog_test, name="Test Actual"))
                fig.add_trace(go.Scatter(x=endog_test.index, y=pred, name="Forecast", line=dict(dash='dot')))
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Summary"): st.code(mod.summary().as_text())
                
            except Exception as e: st.error(str(e))
            
        # VAR
        elif not is_forecast and m_type == "VAR" and run_btn:
            data = df_curr[v_sys].dropna()
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test
            
            d_train = data.iloc[:n_train]
            d_test = data.iloc[n_train:]
            
            try:
                mod = VAR(d_train)
                res = mod.fit(maxlags=10, ic='aic')
                lag_order = res.k_ar
                
                # Forecast
                # VAR forecast needs last 'p' values from train to start
                pred_vals = res.forecast(d_train.values[-lag_order:], steps=n_test)
                pred_df = pd.DataFrame(pred_vals, index=d_test.index, columns=d_test.columns)
                
                # Metrics for first var (or average)
                st.write("### Accuracy on Test Set")
                for col in v_sys:
                    r, m, mp = calculate_metrics(d_test[col], pred_df[col])
                    st.write(f"**{col}**: RMSE={r:.3f}, MAPE={mp:.2%}")
                    
                st.subheader("Equations (Full Data Fit)")
                # Re-fit on full for equations
                res_full = VAR(data).fit(maxlags=10, ic='aic')
                params = res_full.params
                for col in params.columns:
                     terms = [f"{val:.2f}*{idx}" for idx, val in params[col].items() if abs(val)>0.001]
                     st.latex(f"{col}_t = {' + '.join(terms)}")
                     
            except Exception as e: st.error(str(e))

        # VECM
        elif not is_forecast and m_type == "VECM" and run_btn:
            data = df_curr[v_sys].dropna()
            n_test = int(len(data) * test_size)
            n_train = len(data) - n_test
            d_train, d_test = data.iloc[:n_train], data.iloc[n_train:]
            
            try:
                # VECM forecast is tricky in statsmodels, simpler to fit on train and predict
                mod = VECM(d_train, coint_rank=r, k_ar_diff=1)
                res = mod.fit()
                pred_vals = res.predict(steps=n_test)
                pred_df = pd.DataFrame(pred_vals, index=d_test.index, columns=d_test.columns)
                
                st.write("### Accuracy")
                for col in v_sys:
                    r_sq, m, mp = calculate_metrics(d_test[col], pred_df[col])
                    st.write(f"**{col}**: RMSE={r_sq:.3f}, MAPE={mp:.2%}")
                    
                with st.expander("Model Summary"): st.code(res.summary().as_text())
                
            except Exception as e: st.error(str(e))

        # ============ FORECAST MODE RESULTS ============
        elif is_forecast and m_type == "ARIMA" and run_btn:
            final_ord = st.session_state.get('f_arima_ord', (fp,fd,fq))
            s = df_curr[f_var].dropna()
            
            if len(s) < 10:
                st.error("Not enough data points. Need at least 10 observations.")
            else:
                try:
                    st.info(f"Fitting ARIMA{final_ord} on {len(s)} observations...")
                    mod = ARIMA(s, order=final_ord).fit()
                    
                    # Show model diagnostics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AIC", f"{mod.aic:.1f}")
                    c2.metric("BIC", f"{mod.bic:.1f}")
                    c3.metric("Log-Likelihood", f"{mod.llf:.1f}")
                    
                    fcast = mod.get_forecast(steps=f_h)
                    mean = fcast.predicted_mean
                    
                    # Warn if forecast is flat
                    if mean.std() < 1e-6:
                        st.warning("⚠️ Forecast is constant! Try: d=0 if data is stationary, or use 'Suggest' button.")
                    
                    freq = pd.infer_freq(s.index)
                    if not freq and len(s) > 2:
                        diff = s.index[1] - s.index[0]
                        idx = pd.date_range(s.index[-1], periods=f_h+1, freq=diff)[1:]
                    else:
                        idx = pd.date_range(s.index[-1], periods=f_h+1, freq=freq)[1:]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=s.index, y=s, name="History", line=dict(color='#3498db', width=2)))
                    fig.add_trace(go.Scatter(x=idx, y=mean, name="Forecast", line=dict(color='#e74c3c', width=3)))
                    lower = fcast.conf_int().iloc[:,0]
                    upper = fcast.conf_int().iloc[:,1]
                    fig.add_trace(go.Scatter(
                        x=list(idx) + list(idx)[::-1],
                        y=list(upper) + list(lower)[::-1],
                        fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'), name='95% CI'
                    ))
                    fig.update_layout(title=f"ARIMA{final_ord} Forecast", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 Forecast Values"):
                        forecast_df = pd.DataFrame({
                            'Forecast': mean.values,
                            'Lower 95%': lower.values,
                            'Upper 95%': upper.values
                        }, index=idx)
                        st.dataframe(forecast_df)
                        
                except Exception as e: 
                    st.error(f"Error: {e}")
                    st.info("💡 Tips: 1) Check stationarity in Diagnostics tab, 2) Use d=0 for stationary data, 3) Click 'Suggest' for auto-tuning")
                
        elif is_forecast and m_type == "VAR" and run_btn:
            if not f_sys or len(f_sys) < 2:
                st.error("VAR requires at least 2 variables.")
            else:
                try:
                    df = df_curr[f_sys].dropna()
                    st.info(f"Fitting VAR on {len(df)} observations...")
                    mod = VAR(df).fit(maxlags=10, ic='aic')
                    lag = mod.k_ar
                    st.success(f"Using {lag} lags (AIC-optimal)")
                    
                    pred = mod.forecast(df.values[-lag:], steps=f_h)
                    freq = pd.infer_freq(df.index)
                    if not freq and len(df) > 2:
                        diff = df.index[1] - df.index[0]
                        idx = pd.date_range(df.index[-1], periods=f_h+1, freq=diff)[1:]
                    else:
                        idx = pd.date_range(df.index[-1], periods=f_h+1, freq=freq)[1:]
                    
                    fig = go.Figure()
                    for i, col in enumerate(f_sys):
                        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=f"{col} (Hist)", line_width=2))
                        fig.add_trace(go.Scatter(x=idx, y=pred[:,i], name=f"{col} (Forecast)", line=dict(dash='dot', width=2)))
                    fig.update_layout(title="VAR System Forecast", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.error(f"Error: {e}")
                
        elif is_forecast and m_type == "ML (RF/XGB)" and run_btn:
            if not f_x:
                st.error("ML requires at least one predictor.")
            else:
                try:
                    df = df_curr[[f_t] + f_x].dropna()
                    X = df[f_x]
                    y_data = df[f_t]
                    
                    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_data)
                    
                    st.warning("⚠️ Naive ML Forecast: Uses last known predictor values (frozen). Not realistic for most scenarios.")
                    
                    last_row = X.iloc[-1].values.reshape(1, -1)
                    preds = [rf.predict(last_row)[0] for _ in range(f_h)]
                    
                    if len(set(np.round(preds, 4))) == 1:
                        st.error("Predictions are flat because predictors are constant. Use ARIMA/VAR instead.")
                    
                    freq = pd.infer_freq(df.index)
                    if not freq and len(df) > 2:
                        diff = df.index[1] - df.index[0]
                        idx = pd.date_range(df.index[-1], periods=f_h+1, freq=diff)[1:]
                    else:
                        idx = pd.date_range(df.index[-1], periods=f_h+1, freq=freq)[1:]
                        
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=y_data, name="History", line_width=2))
                    fig.add_trace(go.Scatter(x=idx, y=preds, name="Naive ML", line=dict(color="orange", width=3, dash='dot')))
                    fig.update_layout(title="ML Forecast (Naive)", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.error(f"Error: {e}")
