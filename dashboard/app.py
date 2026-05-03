"""FixedSense: Enterprise Fixed Income Portfolio Analytics Platform."""

import logging
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from curves.bootstrapper import bootstrap_spot_curve
from curves.pca_model import fit_pca
from pricing.cashflow_generator import generate_cashflow_schedule, DayCountMethod
from pricing.bond_pricer import price_batch
from greeks.duration import DurationCalculator
from greeks.kr01 import KR01Calculator
from risk.monte_carlo import MonteCarloEngine
from risk.var_calculator import VaRCalculator
from risk.cvar_calculator import CVaRCalculator
from risk.marginal_var import MarginalVaRCalculator
from pnl.factor_attribution import FactorAttribution
from pnl.brinson_fachler import BrinsonFachler
from stress.advanced_scenarios import AdvancedScenarioRunner
from alerts.monitor import AlertMonitor
from data.ingestion.fred_client import FREDClient
from utils.date_utils import get_as_of_date, get_data_date_range

logging.basicConfig(level=logging.WARNING)

# ── PALETTE ──────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "sidebar":  "#0d1117",
    "surface":  "#161b22",
    "surface2": "#21262d",
    "border":   "#30363d",
    "blue":     "#58a6ff",
    "green":    "#3fb950",
    "red":      "#f85149",
    "amber":    "#d29922",
    "purple":   "#bc8cff",
    "cyan":     "#39d353",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "white":    "#ffffff",
}

CHART = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface"],
    font=dict(family="'Segoe UI', system-ui, sans-serif", size=12, color=C["text"]),
    hovermode="x unified",
    margin=dict(l=55, r=25, t=45, b=45),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"])),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], borderwidth=1, font=dict(color=C["text"])),
)

H = 460
SH = 340

st.set_page_config(
    page_title="FixedSense — Fixed Income Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MASTER CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset & Base ── */
html, body, [class*="css"] {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{
    background-color: {C['bg']};
    color: {C['text']};
}}
/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {C['sidebar']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
}}
[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}
[data-testid="stSidebarNav"] {{
    display: none;
}}
/* Radio in sidebar */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: 0.9rem !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    padding: 6px 8px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: {C['surface2']} !important;
}}
/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
/* ── Headings ── */
h1, h2, h3, h4 {{
    color: {C['text']} !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}}
h1 {{ font-size: 1.6rem !important; }}
h2 {{ font-size: 1.2rem !important; }}
h3 {{ font-size: 1.05rem !important; }}
/* ── Divider ── */
hr {{ border-color: {C['border']} !important; margin: 1.2rem 0; }}
/* ── Streamlit metric override ── */
[data-testid="metric-container"] {{
    background: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}}
[data-testid="stMetricLabel"] {{
    color: {C['muted']} !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}}
[data-testid="stMetricValue"] {{
    color: {C['text']} !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    font-variant-numeric: tabular-nums !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}}
/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}
/* ── Select/input widgets ── */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stRadio > div {{
    background-color: {C['surface2']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 6px !important;
}}
/* ── Button ── */
.stButton > button {{
    background: {C['blue']} !important;
    color: {C['bg']} !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
    letter-spacing: 0.3px;
    transition: opacity 0.15s;
}}
.stButton > button:hover {{
    opacity: 0.85 !important;
}}
/* ── Alerts/info boxes ── */
.stAlert {{
    background: {C['surface2']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
}}
/* ── Custom components ── */
.kpi-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 16px 20px;
    height: 100%;
}}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {C['text']};
    letter-spacing: -0.5px;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {C['muted']};
    margin-top: 6px;
}}
.kpi-green  {{ border-left: 3px solid {C['green']}; }}
.kpi-red    {{ border-left: 3px solid {C['red']}; }}
.kpi-blue   {{ border-left: 3px solid {C['blue']}; }}
.kpi-amber  {{ border-left: 3px solid {C['amber']}; }}
.kpi-purple {{ border-left: 3px solid {C['purple']}; }}

/* Section header */
.section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.4rem 0 0.8rem;
    padding-bottom: 10px;
    border-bottom: 1px solid {C['border']};
}}
.section-header-text {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {C['text']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Alert card */
.alert-card {{
    background: {C['surface']};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border: 1px solid {C['border']};
}}
.alert-card-crit {{ border-left: 4px solid {C['red']}; }}
.alert-card-warn {{ border-left: 4px solid {C['amber']}; }}
.alert-card-info {{ border-left: 4px solid {C['blue']}; }}
.alert-title {{
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 5px;
}}
.alert-msg {{ font-size: 0.85rem; color: {C['muted']}; line-height: 1.5; }}
.alert-action {{
    font-size: 0.8rem;
    color: {C['blue']};
    margin-top: 8px;
    font-weight: 500;
}}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
.badge-red    {{ background: rgba(248,81,73,0.15); color: {C['red']}; }}
.badge-amber  {{ background: rgba(210,153,34,0.15); color: {C['amber']}; }}
.badge-green  {{ background: rgba(63,185,80,0.15); color: {C['green']}; }}
.badge-blue   {{ background: rgba(88,166,255,0.15); color: {C['blue']}; }}

/* Page title bar */
.page-title {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.page-title-main {{
    font-size: 1.3rem;
    font-weight: 700;
    color: {C['text']};
}}
.page-title-sub {{
    font-size: 0.8rem;
    color: {C['muted']};
    margin-top: 3px;
}}
.page-title-badge {{
    font-size: 0.78rem;
    color: {C['muted']};
    text-align: right;
}}

/* Holdings table */
.holdings-row {{
    display: flex;
    padding: 10px 0;
    border-bottom: 1px solid {C['border']};
    align-items: center;
    font-size: 0.85rem;
}}
.holdings-header {{
    font-size: 0.72rem;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def kpi(label, value, sub=None, accent="blue"):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card kpi-{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>"""


def section(title, icon=""):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1rem;">{icon}</span>
        <span class="section-header-text">{title}</span>
    </div>""", unsafe_allow_html=True)


def page_title(title, subtitle, badge=""):
    st.markdown(f"""
    <div class="page-title">
        <div>
            <div class="page-title-main">{title}</div>
            <div class="page-title-sub">{subtitle}</div>
        </div>
        <div class="page-title-badge">{badge}</div>
    </div>""", unsafe_allow_html=True)


def badge(text, color="blue"):
    return f'<span class="badge badge-{color}">{text}</span>'


def chart(fig, key=None):
    fig.update_layout(**CHART)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── CACHING ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open(settings.PORTFOLIO_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    try:
        with open(settings.BENCHMARK_CONFIG_PATH) as f:
            bcfg = yaml.safe_load(f)
    except FileNotFoundError:
        bcfg = None
    return cfg, bcfg


@st.cache_data
def load_yield_curves():
    fred = FREDClient(use_sample_data=True)
    s, e = get_data_date_range()
    df = fred.fetch_treasury_yields(s, e)
    return df.pivot(index="date", columns="tenor", values="yield_pct")


@st.cache_data
def build_cashflows(cfg, _as_of):
    cfs = []
    for b in cfg["portfolio"]["bonds"]:
        cfs.append(generate_cashflow_schedule(
            bond_id=b["id"],
            face_value=b["face_value"],
            coupon_rate=b["coupon_rate"],
            coupon_frequency=b["coupon_frequency"],
            maturity_date=pd.to_datetime(b["maturity_date"]).date(),
            issue_date=pd.to_datetime(b["issue_date"]).date(),
            as_of_date=_as_of,
            day_count_method=DayCountMethod.ACT_ACT,
        ))
    return cfs


@st.cache_data
def run_mc(_cfs, _curve, _pca, _spreads, n=10_000):
    eng = MonteCarloEngine(_pca, _curve, n_scenarios=n, horizon_days=1)
    return eng.simulate(_cfs, list(_spreads), [100.0] * len(_cfs))


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────

port_cfg, bench_cfg = load_config()
yc_df = load_yield_curves()
as_of = get_as_of_date()
cfs = build_cashflows(port_cfg, as_of)

total_notional = port_cfg["portfolio"]["total_notional"]
bonds_meta      = port_cfg["portfolio"]["bonds"]
notionals = np.array([total_notional * b["weight"] for b in bonds_meta])
spreads   = np.array([b.get("credit_spread_bps", 0.0) for b in bonds_meta])
ids       = [b["id"] for b in bonds_meta]
sectors   = [b.get("sector", "N/A") for b in bonds_meta]
ratings   = [b.get("rating", "N/A") for b in bonds_meta]
coupons   = [b["coupon_rate"] for b in bonds_meta]

latest_y = yc_df.iloc[-1].dropna()
par_y    = {t: r / 100.0 for t, r in latest_y.items()}
curve    = bootstrap_spot_curve(par_y)

ymat     = yc_df.values
tenors_c = yc_df.columns.values
pca      = fit_pca(ymat, tenors_c, n_factors=3)

prices  = price_batch(cfs, curve, spreads)
nav     = float(np.sum(prices * notionals / 100.0))

mc   = run_mc(cfs, curve, pca, spreads)
v95  = VaRCalculator.monte_carlo_var(mc, confidence=0.95)
v99  = VaRCalculator.monte_carlo_var(mc, confidence=0.99)
cv95 = CVaRCalculator.monte_carlo_cvar(mc, confidence=0.95)

alerts = AlertMonitor.check_all(
    var_95=v95, portfolio_value=nav,
    notionals=list(notionals), bond_names=ids, bond_ids=ids,
    current_spreads_bps=list(spreads), sectors=sectors,
)

n_crit = sum(1 for a in alerts if a.severity == "critical")
n_warn = sum(1 for a in alerts if a.severity == "warning")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 4px 0 20px;">
        <div style="font-size:1.25rem; font-weight:800; color:{C['text']}; letter-spacing:-0.5px;">
            📈 FixedSense
        </div>
        <div style="font-size:0.75rem; color:{C['muted']}; margin-top:2px;">
            Enterprise Fixed Income Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{C['surface2']};border:1px solid {C['border']};border-radius:8px;padding:12px 14px;margin-bottom:16px;">
        <div style="font-size:0.7rem;color:{C['muted']};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Portfolio Snapshot</div>
        <div style="font-size:1.05rem;font-weight:700;color:{C['text']};">${nav/1e6:.1f}M <span style="font-size:0.8rem;font-weight:400;color:{C['muted']};">NAV</span></div>
        <div style="font-size:0.8rem;color:{C['muted']};margin-top:2px;">{len(cfs)} positions · {total_notional/1e9:.1f}B nominal</div>
        <div style="margin-top:8px;font-size:0.78rem;">
            {'<span style="color:' + C['red'] + ';">⬤</span>' if n_crit else '<span style="color:' + C['green'] + ';">⬤</span>'}
            <span style="color:{C['muted']};"> {n_crit} critical / {n_warn} warning</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("📊", "Overview",          "overview"),
        ("📐", "Greeks & Duration", "greeks"),
        ("⚡", "Risk Analytics",    "risk"),
        ("🔬", "Factor Attribution","factors"),
        ("🎯", "Benchmark",         "benchmark"),
        ("🌪",  "Stress Tests",     "stress"),
        ("🚨", "Alert Console",     "alerts_page"),
        ("🔄", "Trade Simulator",   "simulator"),
    ]

    page_key = st.radio(
        "nav",
        [p[2] for p in pages],
        format_func=lambda k: next(f"{p[0]}  {p[1]}" for p in pages if p[2] == k),
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="position:absolute;bottom:20px;left:0;right:0;padding:0 14px;">
        <div style="font-size:0.68rem;color:{C['muted']};border-top:1px solid {C['border']};padding-top:12px;">
            As-of: {as_of:%Y-%m-%d} &nbsp;|&nbsp; v1.0 Enterprise
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────

if page_key == "overview":
    page_title(
        "Portfolio Overview",
        "Real-time NAV, risk metrics, and position summary",
        f"As-of {as_of:%d %b %Y}"
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi("Portfolio NAV", f"${nav/1e6:.1f}M", f"{total_notional/1e6:.0f}M nominal", "blue"), unsafe_allow_html=True)
    k2.markdown(kpi("VaR 95% (1D)", f"${v95/1e6:.2f}M", f"{v95/nav*100:.2f}% of NAV", "red"), unsafe_allow_html=True)
    k3.markdown(kpi("VaR 99% (1D)", f"${v99/1e6:.2f}M", f"{v99/nav*100:.2f}% of NAV", "red"), unsafe_allow_html=True)
    k4.markdown(kpi("CVaR 95%", f"${cv95/1e6:.2f}M", f"{cv95/nav*100:.2f}% of NAV", "amber"), unsafe_allow_html=True)
    k5.markdown(kpi("Alerts", f"{len(alerts)}", f"{n_crit} critical · {n_warn} warning",
                    "red" if n_crit else "amber" if n_warn else "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("US Treasury Spot Curve", "〰")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve.tenors, y=curve.spot_rates * 100,
            mode="lines+markers",
            line=dict(color=C["blue"], width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
            marker=dict(size=6, color=C["blue"], line=dict(color=C["bg"], width=1.5)),
            name="Spot Rate",
            hovertemplate="<b>%{x:.2f}Y</b> — %{y:.3f}%<extra></extra>",
        ))
        fig.update_layout(height=H, yaxis_title="Yield (%)", xaxis_title="Tenor (years)")
        chart(fig)

    with col_r:
        section("Allocation by Bond", "◉")
        pal = [C["blue"], C["green"], C["red"], C["amber"], C["purple"], C["cyan"], "#f0883e"]
        fig = go.Figure(data=[go.Pie(
            labels=ids, values=notionals,
            hole=0.42,
            marker=dict(colors=pal, line=dict(color=C["bg"], width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color=C["text"]),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        )])
        fig.update_layout(height=H, showlegend=False)
        chart(fig)

    section("Holdings", "≡")
    dur_vals = [DurationCalculator.macaulay_duration(c) for c in cfs]
    df_h = pd.DataFrame({
        "Bond":          ids,
        "Sector":        sectors,
        "Rating":        ratings,
        "Coupon":        [f"{r*100:.2f}%" for r in coupons],
        "Notional ($M)": (notionals / 1e6).round(1),
        "Weight":        [f"{w:.1f}%" for w in notionals / nav * 100],
        "Price":         prices.round(4),
        "Spread (bps)":  spreads.astype(int),
        "Duration (Y)":  [f"{d:.2f}" for d in dur_vals],
    })
    st.dataframe(
        df_h.style
            .background_gradient(subset=["Notional ($M)"], cmap="Blues")
            .set_properties(**{"background-color": C["surface"], "color": C["text"],
                               "border": f"1px solid {C['border']}"}),
        use_container_width=True, hide_index=True, height=320,
    )


# ── PAGE: GREEKS ──────────────────────────────────────────────────────────────

elif page_key == "greeks":
    page_title("Greeks & Duration", "Key rate sensitivity and duration analysis", f"{len(cfs)} positions")

    kr01_map = KR01Calculator.portfolio_kr01(cfs, curve, notionals.tolist(), spreads.tolist())
    total_kr01 = sum(kr01_map.values())
    dur_vals = [DurationCalculator.macaulay_duration(c) for c in cfs]
    port_dur = float(np.average(dur_vals, weights=notionals))

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Total KR01", f"${total_kr01/1e6:.3f}M", "$ per 1bp parallel shift", "blue"), unsafe_allow_html=True)
    k2.markdown(kpi("Portfolio Duration", f"{port_dur:.2f}Y", "NAV-weighted Macaulay", "purple"), unsafe_allow_html=True)
    k3.markdown(kpi("KR01 / NAV", f"{total_kr01/nav*100:.3f}%", "Sensitivity ratio", "amber"), unsafe_allow_html=True)
    k4.markdown(kpi("Longest Tenor", f"{max(kr01_map):.0f}Y", "Max rate exposure", "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("Key Rate 01 Ladder", "📊")
        t_sorted = sorted(kr01_map)
        v_sorted = [kr01_map[t] for t in t_sorted]
        fig = go.Figure(go.Bar(
            x=[f"{t:.1f}Y" for t in t_sorted],
            y=v_sorted,
            marker=dict(
                color=v_sorted,
                colorscale=[[0, "#1c2f4a"], [0.5, C["blue"]], [1, "#a8d0ff"]],
                line=dict(color=C["bg"], width=1),
            ),
            text=[f"${v/1000:.0f}K" for v in v_sorted],
            textposition="outside",
            textfont=dict(color=C["muted"], size=10),
            hovertemplate="<b>%{x}</b><br>KR01: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(height=H, xaxis_title="Tenor", yaxis_title="KR01 ($)", showlegend=False)
        chart(fig)

    with col_r:
        section("Duration by Position", "⏱")
        fig = go.Figure(go.Bar(
            y=ids,
            x=dur_vals,
            orientation="h",
            marker=dict(
                color=dur_vals,
                colorscale=[[0, "#1b3d2f"], [1, C["green"]]],
                line=dict(color=C["bg"], width=1),
            ),
            text=[f"{d:.2f}Y" for d in dur_vals],
            textposition="outside",
            textfont=dict(color=C["muted"], size=10),
            hovertemplate="<b>%{y}</b><br>Duration: %{x:.3f}Y<extra></extra>",
        ))
        fig.update_layout(height=H, xaxis_title="Macaulay Duration (years)", showlegend=False)
        chart(fig)

    section("Per-Bond Sensitivity", "≡")
    kr01_per = {b["id"]: KR01Calculator.compute_kr01(cfs[i], curve, notionals[i], spreads[i]).total_kr01
                for i, b in enumerate(bonds_meta)}

    df_g = pd.DataFrame({
        "Bond":          ids,
        "Sector":        sectors,
        "Notional ($M)": (notionals / 1e6).round(2),
        "Duration (Y)":  [round(d, 3) for d in dur_vals],
        "KR01 ($K)":     [round(kr01_per.get(bid, 0) / 1000, 2) for bid in ids],
        "KR01 / NAV":    [f"{kr01_per.get(bid,0)/nav*100:.3f}%" for bid in ids],
        "Spread (bps)":  spreads.astype(int),
    })
    st.dataframe(
        df_g.style.background_gradient(subset=["Duration (Y)"], cmap="RdYlGn_r")
                  .background_gradient(subset=["KR01 ($K)"], cmap="Blues"),
        use_container_width=True, hide_index=True, height=340,
    )


# ── PAGE: RISK ────────────────────────────────────────────────────────────────

elif page_key == "risk":
    page_title("Risk Analytics", "VaR decomposition, distribution analysis, Monte Carlo", "10,000 scenarios")

    var_dec = MarginalVaRCalculator.decompose(cfs, notionals.tolist(), curve, spreads.tolist(), pca, mc)

    pnl_std = float(np.std(mc.pnl_distribution))
    div_pct  = var_dec.diversification_benefit / var_dec.total_var * 100 if var_dec.total_var else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Portfolio VaR 95%",  f"${var_dec.total_var/1e6:.2f}M",   "1-day, 95% confidence", "red"),    unsafe_allow_html=True)
    k2.markdown(kpi("Diversification",    f"${var_dec.diversification_benefit/1e6:.2f}M", f"{div_pct:.1f}% reduction", "green"), unsafe_allow_html=True)
    k3.markdown(kpi("P&L Volatility",     f"${pnl_std/1e6:.2f}M",            "1σ daily P&L",         "amber"),   unsafe_allow_html=True)
    k4.markdown(kpi("Worst Scenario",     f"${min(mc.pnl_distribution)/1e6:.2f}M", "Min MC outcome", "red"),     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        section("P&L Distribution (10K MC Scenarios)", "📉")
        pnl_s = np.sort(mc.pnl_distribution)
        var_v = pnl_s[int(np.ceil(0.05 * len(pnl_s)))]
        cvar_v = float(np.mean(pnl_s[:int(np.ceil(0.01 * len(pnl_s)))]))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mc.pnl_distribution, nbinsx=70,
            marker=dict(color="rgba(88,166,255,0.35)", line=dict(color=C["blue"], width=0.5)),
            name="P&L scenarios",
            hovertemplate="Bin: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        for val, label, color in [(var_v, f"VaR 95%  ${var_v/1e6:.2f}M", C["red"]),
                                   (cvar_v, f"CVaR 99%  ${cvar_v/1e6:.2f}M", "#ff7b72")]:
            fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5,
                          annotation=dict(text=label, font=dict(color=color, size=10),
                                          bgcolor=C["surface"], borderpad=4))
        fig.update_layout(height=H, xaxis_title="1-Day P&L ($)", yaxis_title="Frequency", showlegend=False)
        chart(fig)

    with col_r:
        section("Component VaR Attribution", "🥧")
        cv_vals = list(var_dec.component_var.values())
        cv_names = list(var_dec.component_var.keys())
        pal = [C["blue"], C["green"], C["red"], C["amber"], C["purple"], C["cyan"], "#f0883e"]
        fig = go.Figure(data=[go.Pie(
            labels=cv_names, values=cv_vals,
            hole=0.45,
            marker=dict(colors=pal[:len(cv_names)], line=dict(color=C["bg"], width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Component VaR: $%{value:,.0f}<br>%{percent}<extra></extra>",
        )])
        fig.update_layout(height=H, showlegend=False)
        chart(fig)

    section("Marginal VaR Decomposition", "≡")
    df_v = pd.DataFrame({
        "Bond":              ids,
        "Notional ($M)":     (notionals / 1e6).round(2),
        "Component VaR ($M)": [round(var_dec.component_var.get(b, 0) / 1e6, 4) for b in ids],
        "Marginal VaR (bps)": [round(var_dec.marginal_var.get(b, 0) * 10000, 4) for b in ids],
        "VaR Contribution":  [f"{var_dec.var_contribution_pct.get(b, 0)*100:.2f}%" for b in ids],
        "Standalone VaR ($M)":[round(var_dec.standalone_var.get(b, 0) / 1e6, 4) for b in ids],
    })
    st.dataframe(
        df_v.style.background_gradient(subset=["Component VaR ($M)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True, height=320,
    )


# ── PAGE: FACTORS ─────────────────────────────────────────────────────────────

elif page_key == "factors":
    page_title("Factor Attribution", "PCA yield curve factor decomposition (Level · Slope · Curvature)", "3 factors")

    exp = FactorAttribution.compute_factor_exposures(cfs, notionals.tolist(), curve, spreads.tolist(), pca)
    ev  = pca.explained_variance

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("PC1 — Level",     f"{ev[0]*100:.1f}%", "Parallel shift of curve",     "blue"),   unsafe_allow_html=True)
    k2.markdown(kpi("PC2 — Slope",     f"{ev[1]*100:.1f}%", "Short vs long end spread",    "green"),  unsafe_allow_html=True)
    k3.markdown(kpi("PC3 — Curvature", f"{ev[2]*100:.1f}%", "Belly vs wings",              "amber"),  unsafe_allow_html=True)
    k4.markdown(kpi("Total Explained", f"{sum(ev)*100:.1f}%","Variance explained by 3 PCs", "purple"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("PCA Factor Loadings", "📈")
        labels  = ["PC1: Level", "PC2: Slope", "PC3: Curvature"]
        palette = [C["blue"], C["green"], C["amber"]]
        fig = go.Figure()
        for f in range(pca.factor_loadings.shape[1]):
            fig.add_trace(go.Scatter(
                x=pca.tenors, y=pca.factor_loadings[:, f],
                mode="lines+markers",
                name=labels[f],
                line=dict(color=palette[f], width=2.5),
                marker=dict(size=7, color=palette[f], line=dict(color=C["bg"], width=1.5)),
                hovertemplate=f"<b>%{{x:.2f}}Y</b><br>{labels[f]}: %{{y:.4f}}<extra></extra>",
            ))
        fig.add_hline(y=0, line_color=C["border"], line_width=1, line_dash="dot")
        fig.update_layout(height=H, xaxis_title="Tenor (years)", yaxis_title="Loading Coefficient")
        chart(fig)

    with col_r:
        section("Factor Variance Explained", "🔬")
        fig = go.Figure(go.Bar(
            y=labels,
            x=[v * 100 for v in ev],
            orientation="h",
            marker=dict(color=palette, line=dict(color=C["bg"], width=1)),
            text=[f"{v*100:.1f}%" for v in ev],
            textposition="outside",
            textfont=dict(color=C["muted"]),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(height=H, xaxis_title="Variance Explained (%)", showlegend=False)
        chart(fig)

    section("Bond Factor Exposure Matrix", "🌡")
    df_exp = pd.DataFrame(exp, columns=["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"], index=ids)

    fig_hm = go.Figure(go.Heatmap(
        z=df_exp.values,
        x=df_exp.columns,
        y=df_exp.index,
        colorscale=[
            [0.0, "#7b1d1d"], [0.2, C["red"]], [0.45, C["surface2"]],
            [0.55, C["surface2"]], [0.8, C["blue"]], [1.0, "#1a4d8f"],
        ],
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in df_exp.values],
        texttemplate="%{text}",
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{y}</b> · %{x}<br>Exposure: %{z:.4f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color=C["muted"]),
            outlinecolor=C["border"],
            outlinewidth=1,
        ),
    ))
    fig_hm.update_layout(height=360, margin=dict(l=100, r=20, t=30, b=40))
    chart(fig_hm)


# ── PAGE: BENCHMARK ───────────────────────────────────────────────────────────

elif page_key == "benchmark":
    page_title("Benchmark Analysis", "Brinson-Fachler active return attribution vs UST benchmark", "vs. Equal-Weight UST")

    if not bench_cfg:
        st.error("benchmark.yaml not found")
        st.stop()

    b_cf, b_wt, b_sp = BrinsonFachler.load_benchmark(curve, as_of)
    b_ids = [b["id"] for b in bench_cfg["benchmark"]["bonds"]]

    bf = BrinsonFachler.compute(
        cfs, (notionals / nav).tolist(), spreads.tolist(), ids,
        b_cf, b_wt, b_sp, b_ids, curve, curve,
    )

    interaction = bf.total_active_return - bf.total_allocation_effect - bf.total_selection_effect

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Active Return",    f"{bf.total_active_return*10000:.2f}bps",    "Total excess vs benchmark", "green"),  unsafe_allow_html=True)
    k2.markdown(kpi("Allocation Effect", f"{bf.total_allocation_effect*10000:.2f}bps","Over/under-weighting",     "blue"),   unsafe_allow_html=True)
    k3.markdown(kpi("Selection Effect",  f"{bf.total_selection_effect*10000:.2f}bps", "Security outperformance",  "purple"), unsafe_allow_html=True)
    k4.markdown(kpi("Interaction",       f"{interaction*10000:.2f}bps",               "Weight × return cross",    "amber"),  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("Attribution Waterfall (basis points)", "💧")
        cats   = ["Allocation", "Selection", "Interaction", "Active Return"]
        vals   = [bf.total_allocation_effect * 10000, bf.total_selection_effect * 10000,
                  interaction * 10000, bf.total_active_return * 10000]
        colors = [C["blue"] if v >= 0 else C["red"] for v in vals[:3]] + [C["green"] if vals[3] >= 0 else C["red"]]
        fig = go.Figure(go.Waterfall(
            x=cats, y=vals, measure=["relative", "relative", "relative", "total"],
            text=[f"{v:.2f}bps" for v in vals],
            textposition="outside",
            textfont=dict(color=C["text"], size=11),
            connector=dict(line=dict(color=C["border"], width=1.5, dash="dot")),
            increasing=dict(marker_color=C["blue"]),
            decreasing=dict(marker_color=C["red"]),
            totals=dict(marker_color=C["green"] if vals[3] >= 0 else C["red"]),
            hovertemplate="<b>%{x}</b><br>%{y:.3f}bps<extra></extra>",
        ))
        fig.update_layout(height=H, yaxis_title="Return Contribution (bps)")
        chart(fig)

    with col_r:
        section("Portfolio vs Benchmark Weights", "⚖")
        port_wts = (notionals / nav * 100).tolist()
        all_ids  = list(dict.fromkeys(ids + b_ids))
        bm_wts   = [25.0 if bid in b_ids else 0 for bid in all_ids]
        pw       = [next((w for i, w in enumerate(port_wts) if ids[i] == bid), 0) for bid in all_ids]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Portfolio", x=all_ids, y=pw,
                             marker_color=C["blue"], text=[f"{v:.1f}%" for v in pw], textposition="outside",
                             textfont=dict(color=C["muted"], size=10)))
        fig.add_trace(go.Bar(name="Benchmark", x=all_ids, y=bm_wts,
                             marker_color=C["border"], text=[f"{v:.0f}%" if v else "" for v in bm_wts],
                             textposition="outside", textfont=dict(color=C["muted"], size=10)))
        fig.update_layout(height=H, barmode="group", yaxis_title="Weight (%)", xaxis_title="Bond")
        chart(fig)

    if bf.per_bond:
        section("Per-Bond Attribution Detail", "≡")
        rows = []
        for bid in ids:
            if bid in bf.per_bond:
                r = bf.per_bond[bid]
                rows.append({
                    "Bond":       bid,
                    "Port Wt":    f"{r.portfolio_weight*100:.2f}%",
                    "BM Wt":      f"{r.benchmark_weight*100:.2f}%",
                    "Active Wt":  f"{(r.portfolio_weight-r.benchmark_weight)*100:+.2f}%",
                    "Alloc (bps)":round(r.allocation_effect * 10000, 3),
                    "Select(bps)":round(r.selection_effect * 10000, 3),
                    "Active (bps)":round(r.active_effect * 10000, 3),
                })
        df_bf = pd.DataFrame(rows)
        st.dataframe(
            df_bf.style.background_gradient(subset=["Active (bps)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True, height=320,
        )


# ── PAGE: STRESS ──────────────────────────────────────────────────────────────

elif page_key == "stress":
    page_title("Stress Testing", "Hypothetical · Historical · Reverse stress scenarios", "Multi-scenario")

    with st.spinner("Running scenario suite…"):
        scenarios = AdvancedScenarioRunner.run_all_advanced(cfs, notionals, curve, spreads, nav)

    scn = sorted(scenarios, key=lambda s: s.impact_pct)
    names   = [s.scenario_name for s in scn]
    impacts = [s.impact_pct     for s in scn]
    absvals = [s.impact_absolute for s in scn]

    worst = scn[0]
    best  = scn[-1]
    losses = [s for s in scn if s.impact_pct < 0]

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Worst Case", f"{worst.impact_pct:.2f}%", worst.scenario_name, "red"),    unsafe_allow_html=True)
    k2.markdown(kpi("Best Case",  f"{best.impact_pct:.2f}%",  best.scenario_name,  "green"),  unsafe_allow_html=True)
    k3.markdown(kpi("Avg Impact", f"{np.mean(impacts):.2f}%", f"{len(scn)} scenarios", "amber"), unsafe_allow_html=True)
    k4.markdown(kpi("Scenarios w/ Loss", str(len(losses)), f"of {len(scn)} total", "red"),    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("Scenario Impact Analysis", "🌪")
    bar_colors = []
    for v in impacts:
        if v < -5:   bar_colors.append(C["red"])
        elif v < 0:  bar_colors.append("#ff7b72")
        elif v < 3:  bar_colors.append(C["amber"])
        else:        bar_colors.append(C["green"])

    fig = go.Figure(go.Bar(
        x=names, y=impacts,
        marker=dict(color=bar_colors, line=dict(color=C["bg"], width=1)),
        text=[f"${a/1e6:.1f}M" for a in absvals],
        textposition="outside",
        textfont=dict(color=C["muted"], size=10),
        customdata=[[f"${a/1e6:.2f}M", s.scenario_type] for s, a in zip(scn, absvals)],
        hovertemplate="<b>%{x}</b><br>Impact: %{y:.2f}%<br>$Amount: %{customdata[0]}<br>Type: %{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
    fig.update_layout(height=H, xaxis_title="Scenario", yaxis_title="Portfolio P&L (%)", showlegend=False)
    chart(fig)

    col_l, col_r, col_e = st.columns(3)
    with col_l:
        section("Hypothetical", "🔷")
        for s in scn:
            if s.scenario_type == "hypothetical":
                c = "red" if s.impact_pct < 0 else "green"
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {C["border"]};font-size:0.84rem;">'
                            f'<span style="color:{C["text"]}">{s.scenario_name}</span>'
                            f'<span style="color:{C[c]};font-weight:600;font-variant-numeric:tabular-nums">{s.impact_pct:.2f}%</span></div>',
                            unsafe_allow_html=True)
    with col_r:
        section("Historical", "📜")
        for s in scn:
            if s.scenario_type == "historical":
                c = "red" if s.impact_pct < 0 else "green"
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {C["border"]};font-size:0.84rem;">'
                            f'<span style="color:{C["text"]}">{s.scenario_name}</span>'
                            f'<span style="color:{C[c]};font-weight:600;font-variant-numeric:tabular-nums">{s.impact_pct:.2f}%</span></div>',
                            unsafe_allow_html=True)
    with col_e:
        section("Reverse Stress", "↩")
        for s in scn:
            if s.scenario_type == "reverse":
                trg = getattr(s, "reverse_stress_trigger", None)
                sub = f"Trigger: ±{trg:.0f}bps" if trg else f"{s.impact_pct:.2f}%"
                st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid {C["border"]};font-size:0.84rem;">'
                            f'<div style="color:{C["text"]}">{s.scenario_name}</div>'
                            f'<div style="color:{C["amber"]};font-size:0.78rem;margin-top:3px">{sub}</div></div>',
                            unsafe_allow_html=True)

    section("Scenario Detail Table", "≡")
    df_s = pd.DataFrame({
        "Scenario":   names,
        "Impact %":   impacts,
        "Impact $M":  [round(a / 1e6, 2) for a in absvals],
        "Type":       [s.scenario_type for s in scn],
        "Worst Bond": [s.worst_bond_id for s in scn],
    })
    st.dataframe(
        df_s.style.background_gradient(subset=["Impact %"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True, height=380,
    )


# ── PAGE: ALERTS ──────────────────────────────────────────────────────────────

elif page_key == "alerts_page":
    page_title("Alert Console", "Real-time risk limit monitoring and breach notifications",
               f"{len(alerts)} active alerts")

    a_crit = [a for a in alerts if a.severity == "critical"]
    a_warn = [a for a in alerts if a.severity == "warning"]
    a_info = [a for a in alerts if a.severity == "info"]

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Critical", str(len(a_crit)), "Immediate action required", "red"),   unsafe_allow_html=True)
    k2.markdown(kpi("Warning",  str(len(a_warn)), "Monitor closely",           "amber"), unsafe_allow_html=True)
    k3.markdown(kpi("Info",     str(len(a_info)), "For awareness",             "blue"),  unsafe_allow_html=True)
    k4.markdown(kpi("Total",    str(len(alerts)), "Active alerts",
                    "red" if a_crit else "amber" if a_warn else "green"),               unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def render_alert(a):
        sev_map = {
            "critical": ("alert-card-crit", C["red"],   "CRITICAL"),
            "warning":  ("alert-card-warn", C["amber"], "WARNING"),
            "info":     ("alert-card-info", C["blue"],  "INFO"),
        }
        cls, clr, label = sev_map.get(a.severity, ("alert-card-info", C["blue"], "INFO"))
        st.markdown(f"""
        <div class="alert-card {cls}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div class="alert-title" style="color:{clr}">{a.metric}</div>
                <div>
                    <span class="badge badge-{'red' if a.severity=='critical' else 'amber' if a.severity=='warning' else 'blue'}">{label}</span>
                    &nbsp;
                    <span style="font-size:0.75rem;color:{C['muted']};">Breach: {a.breach_pct:.1f}%</span>
                </div>
            </div>
            <div class="alert-msg">{a.message}</div>
            <div class="alert-action">→ {a.remediation}</div>
        </div>
        """, unsafe_allow_html=True)

    if not alerts:
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-left:4px solid {C['green']};
             border-radius:8px;padding:24px 28px;text-align:center;margin-top:20px;">
            <div style="font-size:2rem;">✅</div>
            <div style="font-size:1.1rem;font-weight:600;color:{C['green']};margin-top:8px;">All Clear</div>
            <div style="color:{C['muted']};font-size:0.88rem;margin-top:6px;">
                All portfolio risk metrics within acceptable parameters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if a_crit:
            section("Critical Alerts", "🔴")
            for a in a_crit:
                render_alert(a)
        if a_warn:
            section("Warnings", "🟡")
            for a in a_warn:
                render_alert(a)
        if a_info:
            section("Informational", "🔵")
            for a in a_info:
                render_alert(a)


# ── PAGE: SIMULATOR ───────────────────────────────────────────────────────────

elif page_key == "simulator":
    page_title("Trade Simulator", "Analyze hypothetical trade impact on portfolio Greeks and risk", "Pre-trade analytics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sel_bond = st.selectbox("Select Bond", ids)
    with col2:
        action = st.radio("Action", ["Buy", "Sell"], horizontal=True)
    with col3:
        amt_m = st.number_input("Amount ($M)", value=10.0, min_value=0.1, step=5.0)
    with col4:
        st.write("")
        st.write("")
        run = st.button("Run Simulation", use_container_width=True)

    st.markdown("---")

    idx = ids.index(sel_bond)
    sign = 1 if action == "Buy" else -1
    new_not = notionals[idx] + sign * amt_m * 1e6

    if new_not < 0:
        st.error(f"⚠️  Cannot sell ${amt_m:.1f}M — position would go negative")
        st.stop()

    new_nots = notionals.copy()
    new_nots[idx] = new_not
    new_nav = float(np.sum(prices * new_nots / 100.0))
    new_kr01_map = KR01Calculator.portfolio_kr01(cfs, curve, new_nots.tolist(), spreads.tolist())
    old_kr01_map = KR01Calculator.portfolio_kr01(cfs, curve, notionals.tolist(), spreads.tolist())

    old_wt      = notionals[idx] / nav * 100
    new_wt      = new_not / new_nav * 100
    old_kr01_t  = sum(old_kr01_map.values())
    new_kr01_t  = sum(new_kr01_map.values())
    nav_delta   = new_nav - nav
    kr01_delta  = new_kr01_t - old_kr01_t

    if run:
        section("Trade Impact Summary", "📊")

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Weight Change", f"{old_wt:.2f}% → {new_wt:.2f}%",
                        f"Δ {new_wt - old_wt:+.2f}pp", "blue"), unsafe_allow_html=True)
        k2.markdown(kpi("NAV Impact", f"${nav_delta/1e6:+.2f}M",
                        f"${nav/1e6:.1f}M → ${new_nav/1e6:.1f}M", "green" if nav_delta >= 0 else "red"),
                    unsafe_allow_html=True)
        k3.markdown(kpi("KR01 Change", f"${kr01_delta/1e6:+.3f}M",
                        f"${old_kr01_t/1e6:.3f}M → ${new_kr01_t/1e6:.3f}M",
                        "amber"), unsafe_allow_html=True)
        k4.markdown(kpi("Trade Size", f"${amt_m:.1f}M",
                        f"{action} {sel_bond}", "blue"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            section("Portfolio Weight Shift", "⚖")
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Before", x=ids, y=(notionals / nav * 100).tolist(),
                                 marker_color=C["border"]))
            fig.add_trace(go.Bar(name="After",  x=ids, y=(new_nots / new_nav * 100).tolist(),
                                 marker_color=C["blue"]))
            fig.update_layout(height=SH, barmode="group", yaxis_title="Weight (%)")
            chart(fig)

        with col_r:
            section("KR01 Ladder: Before vs After", "📐")
            t_sorted = sorted(old_kr01_map)
            old_v = [old_kr01_map[t] for t in t_sorted]
            new_v = [new_kr01_map.get(t, 0) for t in t_sorted]
            labels_x = [f"{t:.1f}Y" for t in t_sorted]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Before", x=labels_x, y=old_v, marker_color=C["border"]))
            fig.add_trace(go.Bar(name="After",  x=labels_x, y=new_v, marker_color=C["blue"]))
            fig.update_layout(height=SH, barmode="group", yaxis_title="KR01 ($)")
            chart(fig)

        color_msg = "green" if action == "Buy" else "amber"
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-left:4px solid {C[color_msg]};
             border-radius:8px;padding:16px 20px;margin-top:8px;">
            <div style="font-weight:600;color:{C[color_msg]};font-size:0.95rem;">
                {'✅' if action=='Buy' else '↩'} {action} ${amt_m:.1f}M of {sel_bond}
            </div>
            <div style="color:{C['muted']};font-size:0.84rem;margin-top:6px;line-height:1.6;">
                Position weight: {old_wt:.2f}% → {new_wt:.2f}%
                &nbsp;·&nbsp; KR01: ${old_kr01_t/1e6:.3f}M → ${new_kr01_t/1e6:.3f}M
                &nbsp;·&nbsp; NAV: ${nav/1e6:.1f}M → ${new_nav/1e6:.1f}M
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show live preview even before clicking
        section("Live Position Preview", "👁")
        df_preview = pd.DataFrame({
            "Bond":      ids,
            "Current Notional": [f"${n/1e6:.1f}M" for n in notionals],
            "After Trade":      [f"${n/1e6:.1f}M" if i != idx else f"${new_not/1e6:.1f}M  ←" for i, n in enumerate(notionals)],
            "Current Wt":       [f"{notionals[i]/nav*100:.2f}%" for i in range(len(ids))],
            "Post-Trade Wt":    [f"{new_nots[i]/new_nav*100:.2f}%" for i in range(len(ids))],
        })
        st.dataframe(df_preview, use_container_width=True, hide_index=True, height=320)


# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center;padding:32px 0 16px;color:{C['muted']};font-size:0.73rem;
     border-top:1px solid {C['border']};margin-top:32px;letter-spacing:0.3px;">
    FixedSense v1.0 &nbsp;·&nbsp; Enterprise Fixed Income Analytics Platform &nbsp;·&nbsp;
    Powered by Monte Carlo &amp; PCA &nbsp;·&nbsp; As-of {as_of:%d %b %Y}
</div>
""", unsafe_allow_html=True)
