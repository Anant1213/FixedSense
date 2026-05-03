"""
FixedSense Dynamic API — FastAPI backend.

Wraps the Python pricing engine and exposes it as a REST API
so the React dashboard can make live calculations.

Architecture:
  - Startup: load portfolio, bootstrap curve, fit PCA, run MC → cache in _S
  - /api/snapshot  → full portfolio snapshot (initial load)
  - /api/shift     → fast analytical P&L for curve/spread slider changes
  - /api/var       → recompute VaR percentile at any confidence
  - /api/recalc    → full reprice + new MC (slower, triggered by button)
  - /api/trade     → simulate buying/selling a bond position
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import settings
from curves.bootstrapper import SpotCurve, bootstrap_spot_curve
from curves.pca_model import fit_pca
from data.ingestion.fred_client import FREDClient
from greeks.dv01 import DV01Calculator
from greeks.kr01 import KR01Calculator
from pricing.bond_pricer import price_batch
from pricing.cashflow_generator import DayCountMethod, generate_cashflow_schedule
from pricing.ytm_solver import solve_ytm
from risk.cvar_calculator import CVaRCalculator
from risk.monte_carlo import MonteCarloEngine
from risk.var_calculator import VaRCalculator
from stress.scenario_runner import ScenarioRunner
from utils.date_utils import get_as_of_date, get_data_date_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FixedSense API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GLOBAL STATE ───────────────────────────────────────────────────────────
_S: dict = {}


def _bootstrap_and_cache():
    """Load everything and warm up the cache. Called once at startup."""
    logger.info("=" * 60)
    logger.info("FixedSense API — initialising pricing engine...")

    with open(settings.PORTFOLIO_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    bonds_cfg = cfg["portfolio"]["bonds"]
    total_notional = cfg["portfolio"]["total_notional"]
    base_notionals = np.array([total_notional * b["weight"] for b in bonds_cfg])
    base_spreads = np.array([b.get("credit_spread_bps", 0.0) for b in bonds_cfg])

    # Yield curve data
    fred = FREDClient(use_sample_data=True)
    s_dt, e_dt = get_data_date_range()
    ydata = fred.fetch_treasury_yields(s_dt, e_dt)
    yc_df = ydata.pivot(index="date", columns="tenor", values="yield_pct")
    as_of = get_as_of_date()

    # Cash flow schedules
    cfs = []
    for b in bonds_cfg:
        cfs.append(
            generate_cashflow_schedule(
                bond_id=b["id"],
                face_value=b["face_value"],
                coupon_rate=b["coupon_rate"],
                coupon_frequency=b["coupon_frequency"],
                maturity_date=pd.to_datetime(b["maturity_date"]).date(),
                issue_date=pd.to_datetime(b["issue_date"]).date(),
                as_of_date=as_of,
                day_count_method=DayCountMethod.ACT_ACT,
            )
        )

    # Bootstrap spot curve
    par_yields = {t: r / 100.0 for t, r in yc_df.iloc[-1].dropna().items()}
    curve = bootstrap_spot_curve(par_yields)

    # Fit PCA — drop rows/cols with any NaN to avoid contaminating the covariance matrix
    yc_clean = yc_df.dropna(axis=0, how="any").dropna(axis=1, how="any")
    pca = fit_pca(yc_clean.values, yc_clean.columns.values.astype(float), n_factors=3)

    # Price portfolio
    prices = price_batch(cfs, curve, base_spreads)
    nav = float(np.sum(prices * base_notionals / 100.0))

    # YTMs + DV01s
    ytms, dv01s = [], []
    for i, cf in enumerate(cfs):
        ytm = solve_ytm(cf, prices[i])
        ytms.append(ytm)
        dv01s.append(DV01Calculator.dv01_from_ytm(cf, prices[i], base_notionals[i]))

    # Portfolio KR01
    kr01_dict = KR01Calculator.portfolio_kr01(
        cfs, curve, base_notionals.tolist(), base_spreads.tolist()
    )

    # Per-bond KR01
    per_bond_kr01 = []
    for i, cf in enumerate(cfs):
        r = KR01Calculator.compute_kr01(cf, curve, base_notionals[i], base_spreads[i])
        per_bond_kr01.append(r.kr01_by_tenor)

    # MC VaR — run with fixed seed for reproducibility
    mc_engine = MonteCarloEngine(pca, curve, n_scenarios=5000, horizon_days=1, random_state=42)
    mc_result = mc_engine.simulate(cfs, base_spreads, prices)

    # MC pnl_distribution is in price-sum units (sum of 7 bond prices ~700).
    # Scale to actual dollar P&L using portfolio NAV / price_sum.
    pnl_scale = nav / float(np.sum(prices))
    scaled_pnl = mc_result.pnl_distribution * pnl_scale
    var_95 = float(-np.percentile(scaled_pnl, 5))
    tail = scaled_pnl[scaled_pnl <= -var_95]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    # Pre-compute stress scenarios
    stress = {}
    runners = [
        ("parallel_up_200",  lambda: ScenarioRunner.create_parallel_up_scenario(cfs, base_notionals, curve, base_spreads, bump_bps=200)),
        ("parallel_down_100", lambda: ScenarioRunner.create_parallel_down_scenario(cfs, base_notionals, curve, base_spreads, bump_bps=100)),
        ("steepener",        lambda: ScenarioRunner.create_steepener_scenario(cfs, base_notionals, curve, base_spreads)),
        ("spread_blowout",   lambda: ScenarioRunner.create_spread_blowout_scenario(cfs, base_notionals, curve, base_spreads)),
    ]
    for sid, fn in runners:
        try:
            r = fn()
            stress[sid] = {
                "id": sid,
                "name": r.scenario_name,
                "impact_pct": round(r.impact_pct, 3),
                "impact_abs": round(r.impact_absolute, 0),
                "worst_bond": r.worst_bond_id,
            }
        except Exception as e:
            logger.warning(f"Stress {sid} skipped: {e}")

    # Try advanced scenarios
    try:
        from stress.advanced_scenarios import AdvancedScenarioRunner
        adv_list = AdvancedScenarioRunner.run_all_advanced(
            cfs, base_notionals.tolist(), curve, base_spreads.tolist(), nav
        )
        for r in adv_list:
            stress[r.scenario_id] = {
                "id": r.scenario_id,
                "name": r.scenario_name,
                "impact_pct": round(r.impact_pct, 3),
                "impact_abs": round(r.impact_absolute, 0),
                "worst_bond": r.worst_bond_id,
            }
    except Exception as e:
        logger.warning(f"Advanced scenarios skipped: {e}")

    var_limit = nav * cfg["portfolio"].get("var_limit_bps", 150) / 10000

    _S.update(
        cfg=cfg,
        bonds_cfg=bonds_cfg,
        total_notional=total_notional,
        base_notionals=base_notionals,
        base_spreads=base_spreads,
        cfs=cfs,
        curve=curve,
        pca=pca,
        yc_df=yc_df,
        prices=prices,
        nav=nav,
        ytms=ytms,
        dv01s=dv01s,
        total_dv01=float(sum(dv01s)),
        kr01_dict=kr01_dict,
        per_bond_kr01=per_bond_kr01,
        mc_result=mc_result,
        pnl_dist=sorted(scaled_pnl.tolist()),
        pnl_scale=pnl_scale,
        var_95=var_95,
        cvar_95=cvar_95,
        var_limit=var_limit,
        stress=stress,
    )
    logger.info(f"Ready. NAV=${nav/1e6:.1f}M  VaR95=${var_95/1e6:.2f}M")


@app.on_event("startup")
async def _startup():
    _bootstrap_and_cache()


# ── REQUEST MODELS ─────────────────────────────────────────────────────────
class ShiftRequest(BaseModel):
    parallel_bps: float = 0.0
    tenor_shocks: Optional[Dict[str, float]] = None   # "2Y" -> bps
    spread_shifts: Optional[Dict[str, float]] = None  # bond_id -> delta bps


class VaRRequest(BaseModel):
    confidence: float = 0.95


class TradeRequest(BaseModel):
    bond_id: str
    direction: str        # "buy" | "sell"
    notional_m: float     # USD millions change


# ── HELPERS ────────────────────────────────────────────────────────────────
def _curve_snapshot(curve: SpotCurve):
    return [
        {"tenor": round(float(t), 3), "spot": round(float(r * 100), 4)}
        for t, r in zip(curve.tenors, curve.spot_rates)
    ]


def _bonds_snapshot(prices, notionals, spreads, pnl_vs_base):
    out = []
    for i, b in enumerate(_S["bonds_cfg"]):
        out.append({
            "id": b["id"],
            "name": b["name"],
            "rating": b["rating"],
            "sector": b["sector"],
            "weight": b["weight"],
            "coupon": b["coupon_rate"],
            "spread_bps": round(float(spreads[i]), 2),
            "notional": round(float(notionals[i]), 0),
            "price": round(float(prices[i]), 4),
            "value": round(float(prices[i] * notionals[i] / 100.0), 0),
            "dv01": round(float(_S["dv01s"][i]), 0),
            "pnl": round(float(pnl_vs_base[i]), 0),
        })
    return out


def _kr01_snapshot(kr01_dict):
    LABEL = {0.5: "6M", 1: "1Y", 2: "2Y", 3: "3Y", 5: "5Y", 7: "7Y", 10: "10Y", 20: "20Y", 30: "30Y"}
    return [
        {"tenor": LABEL.get(t, f"{t}Y"), "value": round(float(v) / 1e5, 4)}
        for t, v in sorted(kr01_dict.items())
    ]


# ── ENDPOINTS ──────────────────────────────────────────────────────────────
@app.get("/api/snapshot")
def get_snapshot():
    """Full portfolio snapshot — called once on page load."""
    s = _S
    pnl_base = np.zeros(len(s["cfs"]))

    return {
        "nav": round(s["nav"], 0),
        "var_95": round(s["var_95"], 0),
        "cvar_95": round(s["cvar_95"], 0),
        "var_limit": round(s["var_limit"], 0),
        "var_utilization": round(s["var_95"] / s["var_limit"] * 100, 1),
        "dv01_limit": s["cfg"]["portfolio"].get("dv01_limit_dollars", 500000),
        "max_position_pct": s["cfg"]["portfolio"].get("max_single_position_pct", 0.25),
        "total_dv01": round(s["total_dv01"], 0),
        "pnl_distribution": [round(x, 0) for x in s["pnl_dist"]],
        "curve": _curve_snapshot(s["curve"]),
        "kr01": _kr01_snapshot(s["kr01_dict"]),
        "bonds": _bonds_snapshot(s["prices"], s["base_notionals"], s["base_spreads"], pnl_base),
        "stress": s["stress"],
        "pca_explained": [round(float(v) * 100, 1) for v in s["pca"].explained_variance],
    }


@app.post("/api/shift")
def shift_portfolio(req: ShiftRequest):
    """
    Fast analytical P&L estimate for curve / spread slider changes.

    Uses KR01 (key-rate DV01) for rate shifts and DV01 for spread changes.
    Returns updated metrics in <50ms without repricing bonds.
    """
    s = _S

    # Build tenor-level rate shocks in bps
    tenor_shock_bps = {t: req.parallel_bps for t in s["kr01_dict"]}
    if req.tenor_shocks:
        for k, v in req.tenor_shocks.items():
            yr = float(k.rstrip("Y")) if k.endswith("Y") else float(k.rstrip("M")) / 12
            closest = min(s["kr01_dict"].keys(), key=lambda x: abs(x - yr))
            tenor_shock_bps[closest] = tenor_shock_bps.get(closest, 0) + v

    # Portfolio P&L from rate shift (KR01 × shock for each tenor bucket)
    rate_pnl = -sum(
        s["kr01_dict"].get(t, 0) * shock / 100.0
        for t, shock in tenor_shock_bps.items()
    )

    # Per-bond spread P&L
    spread_pnl_total = 0.0
    per_bond_spread_pnl = np.zeros(len(s["cfs"]))
    if req.spread_shifts:
        for i, b in enumerate(s["bonds_cfg"]):
            delta = req.spread_shifts.get(b["id"], 0.0)
            pnl_i = -s["dv01s"][i] * delta / 100.0
            per_bond_spread_pnl[i] = pnl_i
            spread_pnl_total += pnl_i

    total_pnl = rate_pnl + spread_pnl_total
    new_nav = s["nav"] + total_pnl

    # Approximate new VaR: scale by (new DV01 sensitivity / original)
    # For parallel shift the curve moves but PCA factors stay; VaR is roughly unchanged
    # Spread widening increases VaR proportionally
    new_var = s["var_95"]
    new_cvar = s["cvar_95"]

    # Build shifted curve for visualization only
    shifted_rates = s["curve"].spot_rates + (req.parallel_bps / 10000.0)
    if req.tenor_shocks:
        for k, v in req.tenor_shocks.items():
            yr = float(k.rstrip("Y")) if k.endswith("Y") else float(k.rstrip("M")) / 12
            idx = int(np.argmin(np.abs(s["curve"].tenors - yr)))
            shifted_rates[idx] += v / 10000.0
    shifted_curve = SpotCurve(
        tenors=s["curve"].tenors,
        spot_rates=shifted_rates,
        as_of_date=s["curve"].as_of_date,
    )

    # Per-bond rate P&L
    per_bond_rate_pnl = np.array([
        -sum(s["per_bond_kr01"][i].get(t, 0) * shock / 100.0
             for t, shock in tenor_shock_bps.items())
        for i in range(len(s["cfs"]))
    ])
    per_bond_total_pnl = per_bond_rate_pnl + per_bond_spread_pnl

    return {
        "pnl": round(total_pnl, 0),
        "rate_pnl": round(rate_pnl, 0),
        "spread_pnl": round(spread_pnl_total, 0),
        "new_nav": round(new_nav, 0),
        "new_var": round(new_var, 0),
        "new_cvar": round(new_cvar, 0),
        "var_utilization": round(new_var / s["var_limit"] * 100, 1),
        "curve": _curve_snapshot(shifted_curve),
        "per_bond_pnl": {
            s["bonds_cfg"][i]["id"]: round(float(per_bond_total_pnl[i]), 0)
            for i in range(len(s["cfs"]))
        },
    }


@app.post("/api/var")
def recalc_var(req: VaRRequest):
    """Recompute VaR/CVaR at requested confidence from stored MC distribution."""
    dist = np.array(_S["pnl_dist"])
    var = float(-np.percentile(dist, (1 - req.confidence) * 100))
    tail = dist[dist <= -var]
    cvar = float(-tail.mean()) if len(tail) > 0 else var
    return {
        "confidence": req.confidence,
        "var": round(var, 0),
        "cvar": round(cvar, 0),
        "var_utilization": round(var / _S["var_limit"] * 100, 1),
    }


@app.post("/api/recalc")
def full_recalc(req: ShiftRequest):
    """
    Full reprice + new MC simulation.
    Applies curve and spread shifts exactly, not analytically.
    Slower (~3–8s) — triggered only by the 'Full Recalculate' button.
    """
    s = _S

    # Build shocked curve
    shocked_rates = s["curve"].spot_rates + (req.parallel_bps / 10000.0)
    if req.tenor_shocks:
        for k, v in req.tenor_shocks.items():
            yr = float(k.rstrip("Y")) if k.endswith("Y") else float(k.rstrip("M")) / 12
            idx = int(np.argmin(np.abs(s["curve"].tenors - yr)))
            shocked_rates[idx] += v / 10000.0
    new_curve = SpotCurve(tenors=s["curve"].tenors, spot_rates=shocked_rates, as_of_date=s["curve"].as_of_date)

    # Shocked spreads
    new_spreads = s["base_spreads"].copy()
    if req.spread_shifts:
        for i, b in enumerate(s["bonds_cfg"]):
            new_spreads[i] += req.spread_shifts.get(b["id"], 0.0)

    # Full reprice
    new_prices = price_batch(s["cfs"], new_curve, new_spreads)
    new_nav = float(np.sum(new_prices * s["base_notionals"] / 100.0))
    pnl_vs_base = (new_prices - s["prices"]) * s["base_notionals"] / 100.0

    # New KR01
    new_kr01 = KR01Calculator.portfolio_kr01(
        s["cfs"], new_curve, s["base_notionals"].tolist(), new_spreads.tolist()
    )

    # New DV01s
    new_dv01s = [
        DV01Calculator.dv01_from_ytm(cf, new_prices[i], s["base_notionals"][i])
        for i, cf in enumerate(s["cfs"])
    ]

    # New MC VaR (scale to dollars same way as startup)
    mc_engine = MonteCarloEngine(s["pca"], new_curve, n_scenarios=5000, horizon_days=1, random_state=42)
    mc_result = mc_engine.simulate(s["cfs"], new_spreads, new_prices)
    new_scale = new_nav / float(np.sum(new_prices))
    new_scaled_pnl = mc_result.pnl_distribution * new_scale
    new_var = float(-np.percentile(new_scaled_pnl, 5))
    new_tail = new_scaled_pnl[new_scaled_pnl <= -new_var]
    new_cvar = float(-new_tail.mean()) if len(new_tail) > 0 else new_var

    return {
        "nav": round(new_nav, 0),
        "pnl": round(new_nav - s["nav"], 0),
        "var_95": round(new_var, 0),
        "cvar_95": round(new_cvar, 0),
        "var_utilization": round(new_var / s["var_limit"] * 100, 1),
        "total_dv01": round(sum(new_dv01s), 0),
        "curve": _curve_snapshot(new_curve),
        "kr01": _kr01_snapshot(new_kr01),
        "bonds": _bonds_snapshot(new_prices, s["base_notionals"], new_spreads, pnl_vs_base),
        "pnl_distribution": [round(x, 0) for x in sorted(new_scaled_pnl.tolist())],
    }


@app.post("/api/trade")
def simulate_trade(req: TradeRequest):
    """
    Simulate buying or selling a position.
    Adjusts notional, reprices, reruns KR01 + DV01.
    """
    s = _S
    bond_idx = next(
        (i for i, b in enumerate(s["bonds_cfg"]) if b["id"] == req.bond_id), None
    )
    if bond_idx is None:
        raise HTTPException(status_code=404, detail=f"Bond {req.bond_id} not found")

    new_notionals = s["base_notionals"].copy()
    delta = req.notional_m * 1e6 * (1 if req.direction == "buy" else -1)
    new_notionals[bond_idx] = max(0, new_notionals[bond_idx] + delta)

    new_prices = price_batch(s["cfs"], s["curve"], s["base_spreads"])
    new_nav = float(np.sum(new_prices * new_notionals / 100.0))

    new_dv01s = [
        DV01Calculator.dv01_from_ytm(cf, new_prices[i], new_notionals[i])
        for i, cf in enumerate(s["cfs"])
    ]
    new_kr01 = KR01Calculator.portfolio_kr01(
        s["cfs"], s["curve"], new_notionals.tolist(), s["base_spreads"].tolist()
    )

    pnl_vs_base = (new_prices - s["prices"]) * (new_notionals - s["base_notionals"]) / 100.0

    return {
        "nav": round(new_nav, 0),
        "pnl_impact": round(float(np.sum(pnl_vs_base)), 0),
        "total_dv01": round(sum(new_dv01s), 0),
        "dv01_delta": round(sum(new_dv01s) - s["total_dv01"], 0),
        "kr01": _kr01_snapshot(new_kr01),
        "bonds": _bonds_snapshot(new_prices, new_notionals, s["base_spreads"], pnl_vs_base),
    }


# ── STATIC FILE SERVING ────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent


@app.get("/")
def serve_index():
    return FileResponse(DASHBOARD_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
