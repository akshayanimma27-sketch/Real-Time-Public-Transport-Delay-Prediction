import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toronto Bus Delay Forecasting",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #F8F7F4; }
    .block-container { padding: 2rem 2.5rem; }

    section[data-testid="stSidebar"] { background-color: #1A1A2E; }
    section[data-testid="stSidebar"] * { color: #E8E6E0 !important; }
    section[data-testid="stSidebar"] hr { border-color: #2E2E4E; }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8E6E0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card .label {
        font-size: 11px; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #888; margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2rem; font-weight: 700;
        color: #1A1A2E; line-height: 1;
    }
    .metric-card .sub { font-size: 12px; color: #AAA; margin-top: 4px; }

    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #1A1A2E;
        border-left: 3px solid #E63946;
        padding-left: 10px; margin: 1.5rem 0 1rem 0;
    }
    .result-box {
        background: #FFFFFF; border-radius: 14px;
        padding: 2rem; border: 1px solid #E8E6E0; text-align: center;
    }
    .result-big { font-size: 3.5rem; font-weight: 700; line-height: 1; }
    .result-label { font-size: 1rem; color: #888; margin-top: 6px; }

    .badge-low      { background:#D1FAE5; color:#065F46; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:700; }
    .badge-moderate { background:#FEF3C7; color:#B45309; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:700; }
    .badge-high     { background:#FDE8E8; color:#B91C1C; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:700; }
    .badge-severe   { background:#7F1D1D; color:#FEE2E2; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:700; }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Palette & theme ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})

BLUE  = '#457B9D'
RED   = '#E63946'
TEAL  = '#2A9D8F'
AMBER = '#F4A261'
DARK  = '#1D3557'


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    path = 'data/ttc-bus-delay-data-2022.csv'
    if not os.path.exists(path):
        st.error("❌ Dataset not found. Place `toronto_bus_delay_2022.csv` in the `data/` folder.")
        st.stop()
    df = pd.read_csv('data/ttc-bus-delay-data-2022.csv')
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    df = df.dropna(subset=['datetime', 'Min Delay'])
    df = df[(df['Min Delay'] >= 0) & (df['Min Delay'] <= 180)]
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour']      = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month']     = df['datetime'].dt.month
    df['day_name']  = df['datetime'].dt.day_name()
    return df

@st.cache_data
def load_ts():
    path = 'data/processed_timeseries.csv'
    if not os.path.exists(path):
        return None
    ts = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    ts.name = 'delay_minutes'
    return ts

@st.cache_resource
def load_models():
    sarima, scaler, lstm = None, None, None
    if os.path.exists('models/sarima_model.pkl'):
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        sarima = SARIMAXResults.load('models/sarima_model.pkl')
    if os.path.exists('models/lstm_scaler.pkl'):
        scaler = joblib.load('models/lstm_scaler.pkl')
    if os.path.exists('models/lstm_model.h5'):
        try:
            import tensorflow as tf
            lstm = tf.keras.models.load_model('models/lstm_model.h5')
        except Exception:
            lstm = None
    return sarima, scaler, lstm

def delay_badge(minutes):
    if minutes < 5:   return '<span class="badge-low">Low (&lt;5 min)</span>'
    if minutes < 15:  return '<span class="badge-moderate">Moderate (5–15 min)</span>'
    if minutes < 30:  return '<span class="badge-high">High (15–30 min)</span>'
    return '<span class="badge-severe">Severe (&gt;30 min)</span>'

def delay_color(minutes):
    if minutes < 5:  return TEAL
    if minutes < 15: return AMBER
    if minutes < 30: return RED
    return '#7F1D1D'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚌 TTC Bus Delay\n**Forecasting Dashboard**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Overview",
        "🔍  Delay Analysis",
        "🤖  Predict Delay",
        "📈  Model Comparison",
    ])
    st.markdown("---")
    st.markdown("**Dataset**\nToronto TTC Bus Delay 2022\nKaggle · 70,000+ incidents")
    st.markdown("**Models**\nARIMA · SARIMA · Prophet · LSTM")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    df = load_raw()

    st.markdown("# 🚌 Toronto TTC Bus Delay Forecasting")
    st.markdown("Time-series analysis and machine learning to predict bus delays across Toronto's transit network.")
    st.markdown("---")

    # KPI cards
    total_incidents = len(df)
    avg_delay       = df['Min Delay'].mean()
    max_delay       = df['Min Delay'].max()
    worst_route     = df.groupby('Route')['Min Delay'].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Incidents</div>
            <div class="value">{total_incidents:,}</div>
            <div class="sub">2022 full year</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Delay</div>
            <div class="value" style="color:{RED}">{avg_delay:.1f} min</div>
            <div class="sub">Per incident</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Max Recorded Delay</div>
            <div class="value">{max_delay:.0f} min</div>
            <div class="sub">Single incident</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Worst Avg Route</div>
            <div class="value" style="font-size:1.5rem">Route {worst_route}</div>
            <div class="sub">Highest mean delay</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Average Delay by Hour</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour')['Min Delay'].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hourly.index, hourly.values, color=RED, linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(hourly.index, hourly.values, alpha=0.1, color=RED)
        ax.axvspan(7, 9,   alpha=0.1, color='red',    label='AM Rush')
        ax.axvspan(16, 19, alpha=0.1, color='orange', label='PM Rush')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Delay (min)')
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">Delay by Day of Week</div>', unsafe_allow_html=True)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = df.groupby('day_name')['Min Delay'].mean().reindex(day_order)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [TEAL if d in ['Saturday','Sunday'] else RED for d in daily.index]
        ax.bar(daily.index, daily.values, color=colors, edgecolor='white', width=0.6)
        ax.axhline(avg_delay, color='gray', linestyle='--', linewidth=1, label=f'Avg {avg_delay:.1f} min')
        ax.set_ylabel('Avg Delay (min)')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)
    rush_am  = df[df['hour'].between(7, 9)]['Min Delay'].mean()
    rush_pm  = df[df['hour'].between(16, 19)]['Min Delay'].mean()
    off_peak = df[~df['hour'].between(7, 19)]['Min Delay'].mean()
    top_incident = df.groupby('Incident')['Min Delay'].mean().idxmax()

    fc1, fc2 = st.columns(2)
    findings = [
        ("🔴 AM Rush Hour", f"7–9 AM average delay is **{rush_am:.1f} min** — the worst window of the day"),
        ("🟠 PM Rush Hour", f"4–7 PM average delay is **{rush_pm:.1f} min** — nearly as bad as AM rush"),
        ("🟡 Off-Peak Relief", f"Off-peak average drops to **{off_peak:.1f} min** — significantly lower"),
        ("🟢 Top Incident Type", f"**{top_incident}** causes the longest average delays on TTC routes"),
    ]
    for i, (title, body) in enumerate(findings):
        with (fc1 if i % 2 == 0 else fc2):
            st.info(f"**{title}** — {body}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DELAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Delay Analysis":
    df = load_raw()

    st.markdown("# 🔍 Delay Analysis")
    st.markdown("Explore TTC delay patterns across routes, incidents, and time periods.")
    st.markdown("---")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        selected_month = st.selectbox("Month", ['All'] + list(range(1, 13)),
            format_func=lambda x: 'All Months' if x == 'All' else
            ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    with fc2:
        selected_day = st.selectbox("Day", ['All','Monday','Tuesday','Wednesday',
                                             'Thursday','Friday','Saturday','Sunday'])
    with fc3:
        top_n = st.slider("Top N Routes / Incidents", 5, 20, 10)

    # Apply filters
    dff = df.copy()
    if selected_month != 'All':
        dff = dff[dff['month'] == selected_month]
    if selected_day != 'All':
        dff = dff[dff['day_name'] == selected_day]

    st.markdown(f"**Showing {len(dff):,} incidents** after filters")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-title">Top {top_n} Routes by Avg Delay</div>', unsafe_allow_html=True)
        route_delay = dff.groupby('Route')['Min Delay'].mean().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.45)))
        colors = [RED if v >= route_delay.mean() else BLUE for v in route_delay.values]
        ax.barh(route_delay.index.astype(str)[::-1], route_delay.values[::-1],
                color=colors[::-1], edgecolor='white')
        ax.axvline(route_delay.mean(), color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Avg Delay (minutes)')
        for i, v in enumerate(route_delay.values[::-1]):
            ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown(f'<div class="section-title">Top {top_n} Incident Types by Avg Delay</div>', unsafe_allow_html=True)
        inc_delay = dff.groupby('Incident')['Min Delay'].mean().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.45)))
        colors = [RED if v >= inc_delay.mean() else TEAL for v in inc_delay.values]
        ax.barh(inc_delay.index[::-1], inc_delay.values[::-1],
                color=colors[::-1], edgecolor='white')
        ax.set_xlabel('Avg Delay (minutes)')
        for i, v in enumerate(inc_delay.values[::-1]):
            ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Monthly Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby('month')['Min Delay'].mean()
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(monthly.index, monthly.values,
                color=TEAL, linewidth=2.5, marker='s', markersize=7)
        ax.fill_between(monthly.index, monthly.values, alpha=0.1, color=TEAL)
        ax.set_xticks(monthly.index)
        ax.set_xticklabels([month_labels[m-1] for m in monthly.index], rotation=30)
        ax.set_ylabel('Avg Delay (minutes)')
        ax.set_title('Monthly Average Delay — 2022')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-title">Delay Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(dff['Min Delay'], bins=40, color=BLUE, edgecolor='white', alpha=0.85)
        ax.axvline(dff['Min Delay'].mean(),   color=RED,  linestyle='--', label=f'Mean {dff["Min Delay"].mean():.1f} min')
        ax.axvline(dff['Min Delay'].median(), color=TEAL, linestyle='-',  label=f'Median {dff["Min Delay"].median():.1f} min')
        ax.set_xlabel('Delay (minutes)')
        ax.set_ylabel('Number of Incidents')
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Heatmap — Hour vs Day of Week</div>', unsafe_allow_html=True)
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = df.pivot_table(values='Min Delay', index='day_name', columns='hour', aggfunc='mean')
    pivot = pivot.reindex(day_order)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap='RdYlGn_r', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Avg Delay (min)', 'shrink': 0.7})
    ax.set_title('Average Delay Heatmap — Hour of Day vs Day of Week', fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('')
    fig.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT DELAY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Predict Delay":
    st.markdown("# 🤖 Predict Bus Delay")
    st.markdown("Enter trip details to get a predicted delay using historical TTC patterns.")
    st.markdown("---")

    df = load_raw()

    with st.form("predict_form"):
        st.markdown('<div class="section-title">Trip Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            route = st.selectbox("Route", sorted(df['Route'].dropna().unique()))
            month = st.selectbox("Month", range(1, 13),
                format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                       'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
            day = st.selectbox("Day of Week",
                ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

        with c2:
            hour = st.slider("Hour of Day", 0, 23, 8)
            incident = st.selectbox("Incident Type",
                sorted(df['Incident'].dropna().unique()))

        with c3:
            direction = st.selectbox("Direction",
                sorted(df['Direction'].dropna().unique()))
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Prediction uses historical TTC patterns from 2022 data.")

        submitted = st.form_submit_button("🔍 Predict Delay", use_container_width=True)

    if submitted:
        # Build prediction from historical averages (pattern-based)
        day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                   'Friday':4,'Saturday':5,'Sunday':6}
        dow = day_map[day]

        # Route-specific average
        route_avg = df[df['Route'] == route]['Min Delay'].mean()
        if pd.isna(route_avg):
            route_avg = df['Min Delay'].mean()

        # Hour-based multiplier
        hourly_avg   = df.groupby('hour')['Min Delay'].mean()
        overall_avg  = df['Min Delay'].mean()
        hour_mult    = hourly_avg.get(hour, overall_avg) / overall_avg

        # Day-based multiplier
        daily_avg    = df.groupby('dayofweek')['Min Delay'].mean()
        day_mult     = daily_avg.get(dow, overall_avg) / overall_avg

        # Month-based multiplier
        monthly_avg  = df.groupby('month')['Min Delay'].mean()
        month_mult   = monthly_avg.get(month, overall_avg) / overall_avg

        # Incident-based multiplier
        incident_avg = df.groupby('Incident')['Min Delay'].mean()
        inc_val      = incident_avg.get(incident, overall_avg)
        inc_mult     = inc_val / overall_avg

        # Combine
        predicted = route_avg * hour_mult * day_mult * month_mult * (0.4 + 0.6 * inc_mult)
        predicted = float(np.clip(predicted, 0, 90))

        color  = delay_color(predicted)
        badge  = delay_badge(predicted)
        is_rush = (7 <= hour <= 9) or (16 <= hour <= 19)
        is_weekend = dow >= 5

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1.2, 1])
        with r2:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-big" style="color:{color}">{predicted:.1f} min</div>
                <div class="result-label">Predicted Delay</div>
                <br>{badge}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Contributing Factors</div>', unsafe_allow_html=True)

        fa1, fa2 = st.columns(2)
        factors = []
        if is_rush:
            factors.append(("🔴 Rush Hour", f"Hour {hour:02d}:00 falls in peak congestion — significantly increases delays"))
        if not is_weekend:
            factors.append(("🟠 Weekday", "Weekday TTC service carries higher passenger volume than weekends"))
        if month in [1, 2, 12]:
            factors.append(("🟡 Winter Month", "Winter months have historically higher delays due to weather"))
        if inc_mult > 1.2:
            factors.append(("🔴 High-Impact Incident", f"'{incident}' incidents average {inc_val:.1f} min — above the overall mean"))
        if hour_mult > 1.2:
            factors.append(("🟠 Peak Hour Effect", f"Hour {hour:02d}:00 has {hour_mult:.1f}× the baseline delay"))
        if not factors:
            factors.append(("🟢 Favourable Conditions", "Off-peak, weekend, or low-impact incident — expect below-average delays"))

        for i, (title, desc) in enumerate(factors):
            with (fa1 if i % 2 == 0 else fa2):
                st.warning(f"**{title}** — {desc}")

        st.markdown("---")
        st.markdown('<div class="section-title">Historical Comparison for This Route</div>', unsafe_allow_html=True)
        route_df = df[df['Route'] == route]
        route_hourly = route_df.groupby('hour')['Min Delay'].mean()

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(route_hourly.index, route_hourly.values,
                color=BLUE, linewidth=2, label=f'Route {route} avg', marker='o', markersize=4)
        ax.axvline(hour, color=RED, linestyle='--', linewidth=2, label=f'Selected hour ({hour:02d}:00)')
        ax.axhline(predicted, color=color, linestyle=':', linewidth=1.5,
                   label=f'Predicted: {predicted:.1f} min')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Delay (minutes)')
        ax.set_title(f'Route {route} — Historical Delay by Hour', fontweight='bold')
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Comparison":
    st.markdown("# 📈 Model Comparison")
    st.markdown("Compare ARIMA, SARIMA, Prophet, and LSTM performance on the TTC test set.")
    st.markdown("---")

    ts = load_ts()

    # Load results if available
    results_path = 'reports/final_model_comparison.csv'
    st.write("File exists:", os.path.exists(results_path))
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        results['RMSE'] = pd.to_numeric(results['RMSE'], errors='coerce')
        results['MAE']  = pd.to_numeric(results['MAE'],  errors='coerce')
        results = results.dropna()
    else:
        # Placeholder results for display before running notebooks
        results = pd.DataFrame({
            'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
            'RMSE':  [None, None, None, None],
            'MAE':   [None, None, None, None],
        })
        st.warning("⚠️ Run notebooks 02 and 03 first to generate model results. Showing placeholder structure.")

    if results['RMSE'].notna().any():
        best_model = results.loc[results['RMSE'].idxmin(), 'Model']

        # Metric cards
        m_cols = st.columns(len(results))
        colors_map = {'ARIMA': BLUE, 'SARIMA': TEAL, 'Prophet': AMBER, 'LSTM': RED}
        for col, (_, row) in zip(m_cols, results.iterrows()):
            with col:
                clr = colors_map.get(row['Model'], DARK)
                badge = "🏆 " if row['Model'] == best_model else ""
                st.markdown(f"""<div class="metric-card">
                    <div class="label">{badge}{row['Model']}</div>
                    <div class="value" style="color:{clr}">{row['RMSE']:.3f}</div>
                    <div class="sub">RMSE (minutes) · MAE: {row['MAE']:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">RMSE Comparison</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bar_colors = [RED if r['Model'] == best_model else BLUE
                          for _, r in results.iterrows()]
            bars = ax.bar(results['Model'], results['RMSE'],
                          color=bar_colors, edgecolor='white', width=0.5)
            ax.set_ylabel('RMSE (minutes)')
            ax.set_title('Lower is Better', fontsize=11, color='gray')
            for bar, v in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01, f'{v:.3f}',
                        ha='center', fontweight='bold', fontsize=11)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown('<div class="section-title">MAE Comparison</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(results['Model'], results['MAE'],
                          color=[TEAL if r['Model'] == best_model else AMBER
                                 for _, r in results.iterrows()],
                          edgecolor='white', width=0.5)
            ax.set_ylabel('MAE (minutes)')
            ax.set_title('Lower is Better', fontsize=11, color='gray')
            for bar, v in zip(bars, results['MAE']):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01, f'{v:.3f}',
                        ha='center', fontweight='bold', fontsize=11)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

    # Time series visualisation
    if ts is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">Hourly Delay Time Series — 2022</div>', unsafe_allow_html=True)

        n_days = st.slider("Days to display", 7, 90, 30)
        ts_show = ts[-24 * n_days:]

        fig, ax = plt.subplots(figsize=(14, 4))
        ts_show.plot(ax=ax, color=BLUE, linewidth=0.8, alpha=0.9)
        ax.set_title(f'TTC Hourly Mean Delay — Last {n_days} days of 2022',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Avg Delay (minutes)')
        ax.set_xlabel('Date')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        'Model':     ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
        'Type':      ['Statistical', 'Statistical', 'Statistical', 'Deep Learning'],
        'Seasonality': ['None', 'Daily (24hr)', 'Daily + Weekly + Yearly', 'Learned'],
        'Best For':  [
            'Simple baseline, no seasonal pattern',
            'Daily rush-hour TTC cycle',
            'Full year with holidays & seasons',
            'Complex non-linear patterns & incidents'
        ],
        'Training Time': ['Fast', 'Medium', 'Medium', 'Slow (use GPU)'],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Key Insight</div>', unsafe_allow_html=True)
    st.info("""
    **Did LSTM beat statistical models on TTC data?**

    - **SARIMA** performs strongly because TTC delays follow a very regular 24-hour pattern (rush hours every day)
    - **Prophet** excels on full-year data because it captures weekly patterns (weekday vs weekend) and yearly seasonality (winter vs summer)
    - **LSTM** gains an advantage when delays are caused by irregular incidents (mechanical failures, accidents) that break normal patterns
    - **Conclusion:** No single model dominates. A production system would combine Prophet (for regular patterns) with LSTM (for anomaly detection)
    """)
