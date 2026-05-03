# FixedSense — Complete Build Guide

> A production-grade fixed income portfolio risk management system.
> Use this document as a prompt for Claude Code to build the entire project layer by layer.

---

## Project Overview

FixedSense is a 6-layer fixed income risk system that mimics what runs on actual bank trading desks. It ingests bond market data, constructs yield curves, computes risk metrics (Greeks, VaR, CVaR), decomposes risk by factor and position, attributes daily P&L, runs stress tests, and surfaces everything through a real-time Streamlit dashboard with Kafka-based alerts.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Pandas, PyArrow, Boto3 (S3), Kafka (confluent-kafka), Streamlit, Plotly, Matplotlib, Docker, Airflow (optional scheduler)

---

## Directory Structure

```
fixedsense/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── config/
│   ├── settings.py              # All config — S3 buckets, API keys, thresholds
│   ├── portfolio.yaml           # Portfolio definition (bonds, weights, notionals)
│   └── scenarios.yaml           # Stress test scenario definitions
├── data/
│   ├── ingestion/
│   │   ├── fred_client.py       # FRED API client — treasury yields, economic data
│   │   ├── yahoo_client.py      # Yahoo Finance client — bond prices, ETF proxies
│   │   ├── data_validator.py    # Schema validation, null checks, outlier detection
│   │   └── scheduler.py         # Cron/Airflow DAG for nightly ingestion
│   ├── storage/
│   │   ├── s3_client.py         # S3 read/write with Parquet serialization
│   │   ├── partitioning.py      # Date-based partitioning logic
│   │   └── catalog.py           # Data catalog — track what's available per date
│   └── processing/
│       ├── cleaner.py           # Missing data imputation, business day alignment
│       ├── enricher.py          # Compute derived fields — mid prices, daily changes
│       └── pipeline.py          # Orchestrator — raw → processed → analytics
├── curves/
│   ├── bootstrapper.py          # Spot curve bootstrapping from par yields
│   ├── interpolator.py          # Cubic spline / Nelson-Siegel interpolation
│   ├── pca_model.py             # PCA decomposition of yield curve changes
│   └── forward_rates.py         # Implied forward rate computation
├── pricing/
│   ├── bond_pricer.py           # Full bond pricing from cash flows + spot curve
│   ├── ytm_solver.py            # Newton-Raphson YTM computation
│   └── cashflow_generator.py    # Generate cash flow schedules for any bond
├── greeks/
│   ├── duration.py              # Macaulay duration, modified duration
│   ├── convexity.py             # Second-order price sensitivity
│   ├── dv01.py                  # Dollar value of a basis point
│   ├── kr01.py                  # Key rate duration at each tenor point
│   └── spread_duration.py       # OAS spread duration for credit risk
├── risk/
│   ├── monte_carlo.py           # PCA-based Monte Carlo simulation engine
│   ├── var_calculator.py        # VaR — parametric, historical, Monte Carlo
│   ├── cvar_calculator.py       # CVaR / Expected Shortfall
│   ├── risk_decomposition.py    # Marginal, incremental, component VaR
│   └── backtester.py            # VaR backtesting — exception counting, traffic light
├── pnl/
│   ├── daily_pnl.py             # Total P&L computation
│   ├── attribution.py           # P&L attribution — carry, rate, spread, residual
│   └── waterfall.py             # Data prep for waterfall chart visualization
├── stress/
│   ├── historical_replay.py     # Replay 2008, COVID, 2022 on current portfolio
│   ├── hypothetical.py          # Custom shock scenarios
│   ├── correlated_shocks.py     # Multi-factor correlated stress
│   └── scenario_runner.py       # Orchestrator — run all scenarios, collect results
├── regulatory/
│   ├── expected_shortfall.py    # FRTB-compliant ES computation
│   ├── stressed_var.py          # Stressed VaR using crisis window data
│   └── liquidity_horizon.py     # Scaling ES by FRTB liquidity horizons
├── streaming/
│   ├── kafka_producer.py        # Publish risk events to Kafka
│   ├── kafka_consumer.py        # Consume and process risk events
│   ├── alert_engine.py          # VaR breach detection + alert routing
│   └── notifier.py              # Email / Slack notification dispatch
├── dashboard/
│   ├── app.py                   # Main Streamlit app — entry point
│   ├── pages/
│   │   ├── overview.py          # Portfolio summary — NAV, DV01, VaR, CVaR
│   │   ├── greeks.py            # Greeks dashboard — duration, KR01 chart
│   │   ├── risk.py              # VaR decomposition, Monte Carlo distribution
│   │   ├── pnl.py               # P&L attribution waterfall + time series
│   │   ├── stress.py            # Stress test results — scenario impact table
│   │   ├── trade_simulator.py   # What-if: add/remove bonds, see VaR impact
│   │   └── regulatory.py        # FRTB metrics — ES, sVaR, backtest results
│   ├── components/
│   │   ├── charts.py            # Plotly chart builders
│   │   ├── tables.py            # Styled dataframe renderers
│   │   └── sidebar.py           # Navigation + portfolio selector
│   └── styles/
│       └── theme.py             # Color palette, chart themes
└── tests/
    ├── test_bootstrapper.py
    ├── test_bond_pricer.py
    ├── test_greeks.py
    ├── test_monte_carlo.py
    ├── test_var.py
    ├── test_pnl_attribution.py
    └── test_stress.py
```

---

## Build Order

Build these layers in EXACTLY this order. Each layer depends on the ones above it.

---

## LAYER 0 — Config & Data Models

### `config/settings.py`

```
Central configuration. All magic numbers live here, nowhere else.

S3_BUCKET = "fixedsense-data"
S3_REGIONS:
  RAW = "raw/"
  PROCESSED = "processed/"
  ANALYTICS = "analytics/"

FRED_API_KEY = env("FRED_API_KEY")

FRED_SERIES:
  - DGS1MO   (1-month treasury yield)
  - DGS3MO   (3-month)
  - DGS6MO   (6-month)
  - DGS1     (1-year)
  - DGS2     (2-year)
  - DGS3     (3-year)
  - DGS5     (5-year)
  - DGS7     (7-year)
  - DGS10    (10-year)
  - DGS20    (20-year)
  - DGS30    (30-year)

TENOR_POINTS = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]  # in years

PCA_LOOKBACK_DAYS = 504    # ~2 years of trading days
PCA_NUM_FACTORS = 3        # level, slope, curvature

MONTE_CARLO_PATHS = 10000
MONTE_CARLO_HORIZON = 1    # days ahead (1-day VaR)

VAR_CONFIDENCE = 0.95
ES_CONFIDENCE = 0.975      # FRTB standard

RISK_FREE_RATE_SERIES = "DGS10"

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC_RISK = "fixedsense.risk.events"
KAFKA_TOPIC_ALERTS = "fixedsense.alerts"

ALERT_VAR_THRESHOLD_PCT = 0.80   # alert when VaR utilization > 80% of limit
```

### `config/portfolio.yaml`

```yaml
portfolio:
  name: "FixedSense Demo Portfolio"
  currency: "INR"
  total_notional: 100_00_00_000   # Rs 100 Crore

  bonds:
    - id: "GOI_10Y"
      name: "Government of India 10Y"
      issuer: "Government of India"
      rating: "Sovereign"
      coupon_rate: 0.0725          # 7.25%
      coupon_frequency: 2          # semi-annual
      face_value: 100
      maturity_date: "2033-06-15"
      issue_date: "2023-06-15"
      weight: 0.25                 # 25% of portfolio
      bond_type: "government"

    - id: "GOI_30Y"
      name: "Government of India 30Y"
      issuer: "Government of India"
      rating: "Sovereign"
      coupon_rate: 0.0754
      coupon_frequency: 2
      face_value: 100
      maturity_date: "2053-06-15"
      issue_date: "2023-06-15"
      weight: 0.20
      bond_type: "government"

    - id: "HDFC_5Y"
      name: "HDFC Corp 5Y AAA"
      issuer: "HDFC"
      rating: "AAA"
      coupon_rate: 0.0810
      coupon_frequency: 2
      face_value: 100
      maturity_date: "2028-03-15"
      issue_date: "2023-03-15"
      weight: 0.20
      bond_type: "corporate"
      credit_spread_bps: 55

    - id: "RIL_7Y"
      name: "Reliance 7Y AA+"
      issuer: "Reliance Industries"
      rating: "AA+"
      coupon_rate: 0.0835
      coupon_frequency: 2
      face_value: 100
      maturity_date: "2030-09-15"
      issue_date: "2023-09-15"
      weight: 0.15
      bond_type: "corporate"
      credit_spread_bps: 72

    - id: "SBI_3Y"
      name: "SBI 3Y AAA"
      issuer: "State Bank of India"
      rating: "AAA"
      coupon_rate: 0.0775
      coupon_frequency: 2
      face_value: 100
      maturity_date: "2026-12-15"
      issue_date: "2023-12-15"
      weight: 0.10
      bond_type: "corporate"
      credit_spread_bps: 40

    - id: "GOI_2Y"
      name: "Government of India 2Y"
      issuer: "Government of India"
      rating: "Sovereign"
      coupon_rate: 0.0680
      coupon_frequency: 2
      face_value: 100
      maturity_date: "2025-12-15"
      issue_date: "2023-12-15"
      weight: 0.10
      bond_type: "government"
```

### `config/scenarios.yaml`

```yaml
scenarios:
  historical:
    - name: "2008 Global Financial Crisis"
      id: "gfc_2008"
      start_date: "2008-09-01"
      end_date: "2008-11-30"
      description: "Lehman collapse, credit freeze, flight to quality"

    - name: "COVID March 2020"
      id: "covid_2020"
      start_date: "2020-03-01"
      end_date: "2020-03-31"
      description: "Pandemic panic, liquidity crisis, everything sold"

    - name: "2022 Rate Hike Cycle"
      id: "rate_hike_2022"
      start_date: "2022-01-01"
      end_date: "2022-12-31"
      description: "Aggressive Fed/RBI tightening, bond massacre"

  hypothetical:
    - name: "Parallel +200bps"
      id: "parallel_up_200"
      shocks:
        yield_curve_parallel_bps: 200
        credit_spread_bps: 0
      description: "Entire yield curve shifts up 200bps"

    - name: "Parallel -100bps"
      id: "parallel_down_100"
      shocks:
        yield_curve_parallel_bps: -100
        credit_spread_bps: 0
      description: "Rate cut scenario — curve drops 100bps"

    - name: "Curve Inversion"
      id: "curve_inversion"
      shocks:
        short_end_bps: 150        # 2Y and below
        long_end_bps: -50         # 10Y and above
        credit_spread_bps: 50
      description: "Short rates spike, long rates fall, spreads widen"

    - name: "Credit Spread Blowout"
      id: "spread_blowout"
      shocks:
        yield_curve_parallel_bps: 0
        credit_spread_bps_by_rating:
          AAA: 100
          AA: 200
          A: 350
          BBB: 500
      description: "Pure credit event — spreads explode, rates unchanged"

    - name: "Stagflation"
      id: "stagflation"
      shocks:
        yield_curve_parallel_bps: 300
        credit_spread_bps: 200
      description: "Rates spike AND spreads widen — worst case"
```

---

## LAYER 1 — Data Ingestion & S3 Data Lake

### `data/ingestion/fred_client.py`

**Purpose:** Fetch US Treasury yields from the FRED API (Federal Reserve Economic Data).

**Implementation Details:**
- Use the `fredapi` Python package (`pip install fredapi`)
- Fetch all 11 series listed in `FRED_SERIES` config
- For each series, pull daily data for the last `PCA_LOOKBACK_DAYS + 30` days (buffer for weekends/holidays)
- Return a single DataFrame with columns: `date`, `tenor` (in years), `yield` (as decimal, e.g. 0.05 for 5%)
- Handle missing data: FRED has gaps on holidays — forward-fill missing values
- Rate limit: FRED allows 120 requests/minute — add a 0.5s sleep between calls
- Cache responses locally to avoid re-fetching during the same day

```
Function: fetch_treasury_yields(start_date, end_date) -> pd.DataFrame
    Columns: [date, tenor, yield_pct]
    date: datetime
    tenor: float (years) — one of TENOR_POINTS
    yield_pct: float — e.g. 0.0725 for 7.25%

Function: fetch_single_series(series_id, start_date, end_date) -> pd.Series
    Wraps fredapi.Fred.get_series()
    Handles retries (3 attempts with exponential backoff)
```

**NOTE for Indian context:** FRED provides US Treasury data. For an Indian version, you would use RBI's FBIL (Financial Benchmarks India) data or CCIL data. For this project, we use US Treasury data as a proxy since it's freely available and the concepts are identical. Document this design decision in README.

### `data/ingestion/yahoo_client.py`

**Purpose:** Fetch bond ETF prices as proxies for bond market data.

**Implementation Details:**
- Use `yfinance` package
- Fetch these ETFs as bond market proxies:
  - `TLT` — iShares 20+ Year Treasury Bond ETF (long duration proxy)
  - `IEF` — iShares 7-10 Year Treasury Bond ETF (medium duration)
  - `SHY` — iShares 1-3 Year Treasury Bond ETF (short duration)
  - `LQD` — iShares Investment Grade Corporate Bond ETF (IG credit)
  - `HYG` — iShares High Yield Corporate Bond ETF (HY credit)
- Pull daily OHLCV data
- Compute daily returns: `(close_t / close_{t-1}) - 1`
- These returns are used to estimate credit spread volatilities and correlations

```
Function: fetch_bond_etf_data(tickers, start_date, end_date) -> pd.DataFrame
    Columns: [date, ticker, open, high, low, close, volume, daily_return]
```

### `data/ingestion/data_validator.py`

**Purpose:** Validate incoming data before storing.

**Checks to implement:**
1. **Schema validation** — correct column names, correct dtypes
2. **Null check** — no more than 5% nulls per column (else flag but don't reject)
3. **Range check** — yields between -2% and 25% (reject if outside)
4. **Staleness check** — most recent data point should be within 2 business days
5. **Monotonicity check** — on any single date, the yield curve should be checked for unusual inversions (warn but don't reject — inversions are rare but real)
6. **Duplicate check** — no duplicate date+tenor combinations

```
Function: validate_yield_data(df) -> ValidationResult
    Returns: ValidationResult(is_valid: bool, warnings: list[str], errors: list[str])
```

### `data/storage/s3_client.py`

**Purpose:** Read/write Parquet files to S3 with date partitioning.

**Implementation Details:**
- Use `boto3` for S3 operations and `pyarrow` for Parquet serialization
- Partition scheme: `s3://fixedsense-data/{zone}/type={data_type}/year={YYYY}/month={MM}/day={DD}/data.parquet`
- Write with snappy compression (fast, good ratio)
- Read with partition pruning — only read the dates you need
- Support both local filesystem (for dev/testing) and real S3

```
Function: write_parquet(df, zone, data_type, date) -> str (S3 path)
Function: read_parquet(zone, data_type, start_date, end_date) -> pd.DataFrame
Function: list_available_dates(zone, data_type) -> list[date]
```

**LOCAL DEV MODE:** For development without AWS, implement a `LocalStorage` class that uses the same interface but writes to `./data/local_lake/{zone}/...`. Use a factory pattern: `get_storage(mode="local"|"s3")`.

### `data/processing/pipeline.py`

**Purpose:** Orchestrate the full ingestion pipeline: fetch → validate → clean → enrich → store.

**Pipeline steps:**
1. Fetch raw data from FRED + Yahoo
2. Validate with `data_validator.py`
3. Write raw data to S3 `raw/` zone (immutable, append-only)
4. Clean: forward-fill missing dates, align to business day calendar, remove outliers (> 4 standard deviations from rolling mean)
5. Enrich: compute daily yield changes (absolute and relative), compute yield spreads (each tenor vs 2Y), compute rolling volatilities (21-day window)
6. Write cleaned+enriched data to `processed/` zone
7. Compute analytics views: current yield curve snapshot, historical yield curve matrix (for PCA), correlation matrix of daily changes
8. Write analytics to `analytics/` zone

```
Function: run_daily_pipeline(date=today) -> PipelineResult
    Runs the full pipeline for a given date
    Returns: PipelineResult(records_ingested, records_processed, errors)
```

---

## LAYER 2 — Yield Curve Engine

### `curves/bootstrapper.py`

**Purpose:** Bootstrap a spot (zero-coupon) yield curve from par yields.

**Algorithm:**

Given par yields at tenors [t1, t2, ..., tn]:

1. The spot rate at t1 equals the par rate at t1: `z(t1) = par(t1)`
2. For each subsequent tenor ti:
   - We know the par bond at ti has price = 100 (par) and coupon = par(ti)
   - Set up the equation:
     ```
     100 = sum(coupon / (1 + z(tj))^tj for all tj < ti) + (100 + coupon) / (1 + z(ti))^ti
     ```
   - All z(tj) for j < i are already known
   - Solve for z(ti)

**Implementation:**
- Use `scipy.optimize.brentq` to solve for each spot rate
- Store the spot curve as a dictionary: `{tenor: spot_rate}`
- Validate: spot curve should be smooth (no wild jumps between adjacent tenors)

```python
class SpotCurve:
    tenors: np.ndarray        # [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    spot_rates: np.ndarray    # corresponding zero-coupon rates
    as_of_date: date

    def rate_at(self, tenor: float) -> float:
        """Interpolated spot rate at any tenor"""

    def discount_factor(self, tenor: float) -> float:
        """DF = 1 / (1 + z(t))^t"""

    def forward_rate(self, t1: float, t2: float) -> float:
        """Implied forward rate between t1 and t2"""
```

```
Function: bootstrap_spot_curve(par_yields: dict[float, float], as_of_date: date) -> SpotCurve
    Input: {1: 0.068, 2: 0.070, 3: 0.071, 5: 0.073, ...}
    Output: SpotCurve object with interpolated rates at all tenor points
```

### `curves/interpolator.py`

**Purpose:** Interpolate spot rates between known tenor points.

**Implement two methods:**

1. **Cubic spline interpolation** — smooth, handles the curve well, use as default
   - Use `scipy.interpolate.CubicSpline`
   - Fit on (tenor, spot_rate) pairs
   - Extrapolate flat beyond 30Y (hold last rate constant)

2. **Nelson-Siegel model** (bonus, for interview depth):
   - Parametric model: `y(t) = β0 + β1 * (1-exp(-t/τ)) / (t/τ) + β2 * ((1-exp(-t/τ))/(t/τ) - exp(-t/τ))`
   - Fit 4 parameters (β0, β1, β2, τ) to observed spot rates using least squares
   - Advantage: smooth, economically interpretable (β0 = long-term level, β1 = slope, β2 = curvature)

```
Function: interpolate_cubic(tenors, rates, query_tenors) -> np.ndarray
Function: fit_nelson_siegel(tenors, rates) -> NelsonSiegelParams
Function: evaluate_nelson_siegel(params, query_tenors) -> np.ndarray
```

### `curves/pca_model.py`

**Purpose:** Decompose yield curve movements into 3 principal components (level, slope, curvature).

**This is the most important file in the entire project.**

**Algorithm:**

1. Load historical yield curve data: a matrix of shape `(N_days, N_tenors)` where each row is one day's yield curve
2. Compute daily changes: `dY[t] = Y[t] - Y[t-1]` — shape `(N_days-1, N_tenors)`
3. Compute covariance matrix of daily changes: shape `(N_tenors, N_tenors)`
4. Eigendecomposition: `cov_matrix = V @ diag(eigenvalues) @ V.T`
5. The first 3 eigenvectors (columns of V) are the factor loadings
6. The first 3 eigenvalues give the variance explained by each factor
7. Factor scores for any day: `scores = dY[t] @ V[:, :3]` — projects the curve change onto 3 dimensions

**Implementation:**

```python
class PCAModel:
    factor_loadings: np.ndarray   # shape (N_tenors, 3) — the 3 eigenvectors
    eigenvalues: np.ndarray       # shape (3,) — variance of each factor
    explained_variance: np.ndarray  # [0.85, 0.10, 0.03] — proportion explained
    mean_changes: np.ndarray      # shape (N_tenors,) — average daily change
    tenors: np.ndarray

    def fit(self, yield_curve_history: np.ndarray) -> None:
        """
        Input: (N_days, N_tenors) matrix of yield curves
        Computes: factor_loadings, eigenvalues, explained_variance
        """

    def project(self, daily_change: np.ndarray) -> np.ndarray:
        """
        Project a curve change vector onto 3 factors
        Input: (N_tenors,) vector of daily yield changes
        Output: (3,) factor scores [level_score, slope_score, curve_score]
        """

    def reconstruct(self, factor_scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct a curve change from factor scores
        Input: (3,) factor scores
        Output: (N_tenors,) reconstructed curve change
        """

    def simulate_scenarios(self, n_scenarios: int) -> np.ndarray:
        """
        Generate n random yield curve change scenarios
        Input: number of scenarios
        Output: (n_scenarios, N_tenors) matrix of simulated curve changes
        Uses: independent normal draws for each factor, scaled by sqrt(eigenvalue)
        """
```

**Critical detail for `simulate_scenarios`:**
```
For each scenario i:
    z = np.random.normal(0, 1, size=3)                    # 3 independent standard normals
    factor_scores = z * np.sqrt(eigenvalues[:3])           # scale by factor volatility
    curve_change = factor_loadings[:, :3] @ factor_scores  # reconstruct correlated changes
    scenarios[i] = curve_change
```

This automatically produces correlated tenor movements — the PCA structure handles it.

### `curves/forward_rates.py`

**Purpose:** Compute implied forward rates from the spot curve.

```
Function: forward_rate(spot_curve: SpotCurve, t1: float, t2: float) -> float
    """
    The implied rate for borrowing between time t1 and t2
    Formula: f(t1,t2) = ((1+z(t2))^t2 / (1+z(t1))^t1)^(1/(t2-t1)) - 1
    """

Function: forward_curve(spot_curve: SpotCurve, forward_start: float) -> dict[float, float]
    """
    Compute 1-year forward rates starting at each future point
    Returns: {1: f(0,1), 2: f(1,2), 3: f(2,3), ...}
    """
```

---

## LAYER 3 — Bond Pricing & Cash Flows

### `pricing/cashflow_generator.py`

**Purpose:** Generate the complete cash flow schedule for any bond.

```python
@dataclass
class CashFlow:
    date: date
    amount: float          # coupon or coupon + principal at maturity
    tenor: float           # years from today to this cash flow
    cf_type: str           # "coupon" or "principal+coupon"

class CashFlowSchedule:
    bond_id: str
    cash_flows: list[CashFlow]
    face_value: float
    coupon_rate: float

    @staticmethod
    def generate(bond: BondConfig, as_of_date: date) -> 'CashFlowSchedule':
        """
        Generate all future cash flows from as_of_date to maturity.
        For semi-annual bonds: coupon = face_value * coupon_rate / 2 every 6 months
        For annual bonds: coupon = face_value * coupon_rate every 12 months
        Last cash flow includes face_value + final coupon
        Tenors computed using actual/365 day count convention
        """
```

**Day count convention:** Use ACT/365 (actual days / 365). This is the most common in India. For US bonds, ACT/ACT is standard. Implement as a pluggable parameter.

### `pricing/bond_pricer.py`

**Purpose:** Price a bond given its cash flows and a spot curve.

```python
class BondPricer:
    def price(self, cashflows: CashFlowSchedule, spot_curve: SpotCurve) -> float:
        """
        Price = sum(CF_i * DF(t_i)) for all future cash flows
        where DF(t) = 1 / (1 + z(t))^t is the discount factor at tenor t
        z(t) = spot rate at tenor t (interpolated from the spot curve)
        """

    def price_with_spread(self, cashflows: CashFlowSchedule, spot_curve: SpotCurve,
                          spread_bps: float) -> float:
        """
        Same as price() but adds a spread to each spot rate before discounting
        Effective rate at tenor t = z(t) + spread
        This is how corporate bonds are priced — government curve + credit spread
        """

    def accrued_interest(self, bond: BondConfig, as_of_date: date) -> float:
        """
        Accrued interest = coupon_per_period * (days_since_last_coupon / days_in_coupon_period)
        """

    def clean_price(self, ...) -> float:
        """Clean = Dirty - Accrued"""

    def dirty_price(self, ...) -> float:
        """Dirty = sum of discounted cash flows (full price)"""
```

### `pricing/ytm_solver.py`

**Purpose:** Solve for YTM given a bond's market price.

```
Function: solve_ytm(cashflows: CashFlowSchedule, market_price: float) -> float
    """
    Find y such that:
    market_price = sum(CF_i / (1+y)^t_i)

    Use scipy.optimize.brentq with bracket [0.0001, 0.50]
    Or Newton-Raphson with analytical derivative:
        dPrice/dy = -sum(t_i * CF_i / (1+y)^(t_i+1))
    """
```

---

## LAYER 4 — Greeks Engine

### `greeks/duration.py`

```python
class DurationCalculator:
    def macaulay_duration(self, cashflows: CashFlowSchedule, ytm: float) -> float:
        """
        MacD = (1/Price) * sum(t_i * CF_i / (1+y)^t_i)
        Weighted average time of cash flows
        """

    def modified_duration(self, cashflows: CashFlowSchedule, ytm: float) -> float:
        """
        ModD = MacD / (1 + y/freq)
        where freq = coupon frequency (2 for semi-annual)
        This is the actual price sensitivity measure
        """

    def effective_duration(self, pricer: BondPricer, cashflows: CashFlowSchedule,
                           spot_curve: SpotCurve, bump_bps: float = 1.0) -> float:
        """
        Numerical duration using full repricing:
        EffD = (P_down - P_up) / (2 * P_0 * dy)

        1. Price bond at current curve: P_0
        2. Bump entire curve UP by bump_bps: P_up
        3. Bump entire curve DOWN by bump_bps: P_down
        4. EffD = (P_down - P_up) / (2 * P_0 * bump_bps/10000)

        More accurate than analytical — works for any bond type
        """
```

### `greeks/dv01.py`

```python
class DV01Calculator:
    def dv01(self, modified_duration: float, price: float, face_value: float,
             notional: float) -> float:
        """
        DV01 = ModD * Price/100 * Notional / 10000
        Result is in currency units (rupees)
        Meaning: how much portfolio value changes per 1bp rate move
        """

    def portfolio_dv01(self, bonds: list[BondPosition]) -> float:
        """
        Sum of individual DV01s
        """
```

### `greeks/kr01.py`

**Purpose:** Compute Key Rate DV01 at each tenor bucket — this is the key differentiator.

```python
class KR01Calculator:
    KEY_RATE_TENORS = [0.5, 1, 2, 3, 5, 7, 10, 20, 30]  # standard buckets

    def compute_kr01(self, pricer: BondPricer, cashflows: CashFlowSchedule,
                     spot_curve: SpotCurve, notional: float) -> dict[float, float]:
        """
        For each key rate tenor t_k:
            1. Create a bumped curve: bump ONLY the rate at t_k by +1bp
               Use triangular bump: full 1bp at t_k, linearly fading to 0
               at adjacent key rate tenors
            2. Reprice bond on bumped curve: P_bumped
            3. KR01 at t_k = (P_original - P_bumped) * notional / face_value

        Returns: {0.5: kr01_6m, 1: kr01_1y, 2: kr01_2y, ..., 30: kr01_30y}
        Sum of all KR01s ≈ total DV01 (small differences due to interpolation)
        """

    def triangular_bump(self, spot_curve: SpotCurve, bump_tenor: float,
                        bump_bps: float = 1.0) -> SpotCurve:
        """
        Bump the rate at bump_tenor by bump_bps.
        Linearly interpolate the bump to 0 at adjacent key rate tenors.
        This is the standard approach — avoids artificial kinks in the curve.

        Example: bumping the 5Y point
        - At 5Y: +1bp
        - At 3Y: 0bp (adjacent lower key rate)
        - At 7Y: 0bp (adjacent upper key rate)
        - At 4Y: +0.5bp (linear interpolation)
        - At 6Y: +0.5bp (linear interpolation)
        - All other tenors: 0bp
        """

    def portfolio_kr01(self, bonds: list[BondPosition], ...) -> dict[float, float]:
        """Sum individual KR01s at each tenor bucket"""
```

### `greeks/convexity.py`

```python
class ConvexityCalculator:
    def convexity(self, pricer: BondPricer, cashflows: CashFlowSchedule,
                  spot_curve: SpotCurve, bump_bps: float = 1.0) -> float:
        """
        Numerical convexity:
        Conv = (P_up + P_down - 2*P_0) / (P_0 * dy^2)

        where dy = bump_bps / 10000
        P_up = price when curve bumped up
        P_down = price when curve bumped down
        P_0 = current price
        """

    def price_change_with_convexity(self, mod_duration: float, convexity: float,
                                     dy_bps: float) -> float:
        """
        dP/P = -ModD * dy + 0.5 * Conv * dy^2
        where dy = dy_bps / 10000
        Returns percentage price change
        """
```

### `greeks/spread_duration.py`

```python
class SpreadDurationCalculator:
    def spread_duration(self, pricer: BondPricer, cashflows: CashFlowSchedule,
                        spot_curve: SpotCurve, current_spread_bps: float,
                        bump_bps: float = 1.0) -> float:
        """
        How much does bond price change for a 1bp change in credit spread?

        1. Price at current spread: P_0 = price_with_spread(curve, spread)
        2. Price at spread + 1bp: P_up = price_with_spread(curve, spread + 1bp)
        3. Price at spread - 1bp: P_down = price_with_spread(curve, spread - 1bp)
        4. SpreadDur = (P_down - P_up) / (2 * P_0 * 0.0001)
        """

    def spread_dv01(self, spread_duration: float, price: float, notional: float) -> float:
        """
        Spread DV01 = SpreadDur * Price/100 * Notional / 10000
        Rupee sensitivity to a 1bp spread move
        """
```

---

## LAYER 5 — Risk Engine

### `risk/monte_carlo.py`

**Purpose:** PCA-based Monte Carlo simulation engine — the heart of FixedSense.

```python
class MonteCarloEngine:
    def __init__(self, pca_model: PCAModel, spot_curve: SpotCurve,
                 n_scenarios: int = 10000, horizon_days: int = 1):
        self.pca = pca_model
        self.curve = spot_curve
        self.n_scenarios = n_scenarios
        self.horizon = horizon_days

    def simulate(self) -> MonteCarloResult:
        """
        Full simulation pipeline:

        1. Generate random factor scores:
           For each scenario i in 1..n_scenarios:
               z = np.random.standard_normal(3)   # 3 independent normals
               # Scale by factor volatility and horizon
               factor_scores[i] = z * np.sqrt(eigenvalues * horizon_days)

        2. Reconstruct curve changes:
           For each scenario i:
               curve_change[i] = factor_loadings @ factor_scores[i]
               # shape: (N_tenors,) — the simulated change at each tenor

        3. Build shocked curves:
           For each scenario i:
               shocked_curve[i] = current_curve + curve_change[i]

        4. Reprice portfolio on each shocked curve:
           For each scenario i:
               portfolio_value[i] = sum(price(bond, shocked_curve[i]) * notional for bond in portfolio)

        5. Compute P&L:
           pnl[i] = portfolio_value[i] - current_portfolio_value

        Returns: MonteCarloResult with pnl_distribution, shocked_curves, factor_scores
        """

    def simulate_with_spread_shocks(self, spread_vol_bps: float) -> MonteCarloResult:
        """
        Extended version: simulate BOTH rate and spread shocks

        Same as above, but additionally:
        - For each corporate bond, add a random spread shock:
          spread_change = np.random.normal(0, spread_vol_bps / 10000)
        - Reprice corporate bonds at: shocked_curve + (current_spread + spread_change)
        - Government bonds: just shocked_curve

        This separates rate risk from spread risk in the simulation
        """

@dataclass
class MonteCarloResult:
    pnl_distribution: np.ndarray      # shape (n_scenarios,) — P&L for each scenario
    current_portfolio_value: float
    shocked_portfolio_values: np.ndarray
    factor_scores: np.ndarray          # shape (n_scenarios, 3)
    curve_changes: np.ndarray          # shape (n_scenarios, N_tenors)
```

### `risk/var_calculator.py`

```python
class VaRCalculator:
    def parametric_var(self, portfolio_value: float, portfolio_vol: float,
                       confidence: float = 0.95, horizon_days: int = 1) -> float:
        """
        VaR = portfolio_value * z_score * vol * sqrt(horizon)
        z_score at 95% = 1.645
        z_score at 99% = 2.326

        Simple but assumes normality — use as a quick sanity check
        """

    def historical_var(self, historical_returns: np.ndarray,
                       portfolio_value: float,
                       confidence: float = 0.95) -> float:
        """
        Sort historical returns ascending
        VaR = -percentile(returns, (1 - confidence) * 100) * portfolio_value
        No distribution assumptions
        """

    def monte_carlo_var(self, mc_result: MonteCarloResult,
                        confidence: float = 0.95) -> float:
        """
        Sort simulated P&L ascending
        VaR = -np.percentile(mc_result.pnl_distribution, (1 - confidence) * 100)
        Most accurate — uses full PCA-based simulation
        """

    def compute_all(self, ...) -> VaRReport:
        """Compute all three methods and return comparison"""
```

### `risk/cvar_calculator.py`

```python
class CVaRCalculator:
    def monte_carlo_cvar(self, mc_result: MonteCarloResult,
                         confidence: float = 0.95) -> float:
        """
        CVaR = average of all losses beyond VaR threshold

        sorted_pnl = np.sort(mc_result.pnl_distribution)
        cutoff_index = int(len(sorted_pnl) * (1 - confidence))
        cvar = -np.mean(sorted_pnl[:cutoff_index])

        This is also called Expected Shortfall (ES)
        """

    def historical_cvar(self, historical_returns: np.ndarray,
                        portfolio_value: float,
                        confidence: float = 0.95) -> float:
        """Same logic but using historical returns instead of simulated"""

    def tail_ratio(self, var: float, cvar: float) -> float:
        """
        CVaR / VaR ratio
        Close to 1.0 = thin tail (normal-ish distribution)
        Above 2.0 = fat tail (dangerous — extreme losses are much worse than VaR suggests)
        """
```

### `risk/risk_decomposition.py`

```python
class RiskDecomposition:
    def marginal_var(self, mc_engine: MonteCarloEngine,
                     bond_positions: list[BondPosition],
                     confidence: float = 0.95) -> dict[str, float]:
        """
        For each bond i:
            1. Compute full portfolio VaR: VaR_full
            2. Remove bond i from portfolio
            3. Compute reduced portfolio VaR: VaR_without_i
            4. Marginal VaR of bond i = VaR_full - VaR_without_i

        NOTE: Sum of marginal VaRs != total VaR (because of diversification)
        Use component VaR for additive decomposition
        """

    def incremental_var(self, mc_engine: MonteCarloEngine,
                        current_portfolio: list[BondPosition],
                        new_bond: BondPosition,
                        confidence: float = 0.95) -> float:
        """
        Trade simulator:
        1. Compute current VaR
        2. Add new_bond to portfolio
        3. Compute new VaR
        4. Incremental VaR = new VaR - current VaR

        Positive = adding risk. Negative = hedging existing risk.
        """

    def component_var_by_factor(self, mc_result: MonteCarloResult,
                                pca_model: PCAModel,
                                confidence: float = 0.95) -> dict[str, float]:
        """
        Decompose VaR by risk factor:

        For each scenario, decompose P&L into:
        - Rate P&L: from level factor (PC1)
        - Slope P&L: from slope factor (PC2)
        - Curve P&L: from curvature factor (PC3)
        - Spread P&L: from spread shocks (if simulated)
        - Residual: total P&L minus above components

        Then compute VaR of each component's P&L distribution

        Returns: {"rate": var_rate, "slope": var_slope, "curve": var_curve,
                  "spread": var_spread, "residual": var_residual}
        """
```

### `risk/backtester.py`

```python
class VaRBacktester:
    def backtest(self, predicted_var: pd.Series, actual_pnl: pd.Series,
                 confidence: float = 0.95) -> BacktestResult:
        """
        Compare predicted daily VaR against actual realized P&L

        For each day t:
            exception = 1 if actual_loss[t] > predicted_var[t] else 0

        Count total exceptions over window (usually 250 days)

        Expected exceptions = (1 - confidence) * window_days
            At 95% over 250 days: expected = 12.5

        Basel traffic light:
            Green: 0-4 exceptions (model is good)
            Yellow: 5-9 exceptions (model needs review)
            Red: 10+ exceptions (model is rejected)

        Also compute:
            - Kupiec POF test (proportion of failures test)
            - Christoffersen test (independence of exceptions)
        """

@dataclass
class BacktestResult:
    total_days: int
    exceptions: int
    exception_rate: float
    expected_exceptions: float
    traffic_light: str          # "green", "yellow", "red"
    kupiec_p_value: float
    exception_dates: list[date]
```

---

## LAYER 6 — P&L Attribution

### `pnl/attribution.py`

```python
class PnLAttribution:
    def compute_daily_attribution(self,
                                   portfolio: Portfolio,
                                   yesterday_curve: SpotCurve,
                                   today_curve: SpotCurve,
                                   yesterday_spreads: dict[str, float],
                                   today_spreads: dict[str, float]) -> AttributionResult:
        """
        Break down daily P&L into 4 components:

        1. CARRY P&L:
           For each bond:
               daily_carry = (coupon_rate * face_value * notional) / 365
           Total carry = sum of all daily carries
           Always positive (you earn coupon every day you hold)

        2. RATE P&L:
           For each bond:
               rate_change_bps = (today_govt_yield_at_bond_tenor - yesterday_govt_yield_at_bond_tenor) * 10000
               rate_pnl = -dv01_of_bond * rate_change_bps
           Total rate P&L = sum over all bonds
           Negative when rates rise (you lose money)

        3. SPREAD P&L (corporate bonds only):
           For each corporate bond:
               spread_change_bps = (today_spread - yesterday_spread) * 10000
               spread_pnl = -spread_dv01_of_bond * spread_change_bps
           Government bonds contribute 0 to spread P&L
           Negative when spreads widen (you lose money)

        4. RESIDUAL:
           residual = actual_total_pnl - carry - rate_pnl - spread_pnl
           Captures: convexity effect, theta/pull-to-par, model error, liquidity

        Returns: AttributionResult with all 4 components + total + per-bond breakdown
        """

    def compute_actual_total_pnl(self, portfolio: Portfolio,
                                  yesterday_curve: SpotCurve,
                                  today_curve: SpotCurve) -> float:
        """
        Full reprice: price every bond on today's curve - yesterday's value
        This is the true P&L (not approximate)
        """

@dataclass
class AttributionResult:
    date: date
    total_pnl: float
    carry_pnl: float
    rate_pnl: float
    spread_pnl: float
    residual_pnl: float
    per_bond: dict[str, BondAttributionResult]    # breakdown per bond
    explain_pct: float    # (carry + rate + spread) / total — how much is explained
```

---

## LAYER 7 — Stress Testing

### `stress/historical_replay.py`

```python
class HistoricalReplay:
    def replay(self, portfolio: Portfolio,
               current_curve: SpotCurve,
               crisis_yield_changes: pd.DataFrame,
               crisis_spread_changes: dict[str, pd.Series]) -> StressResult:
        """
        Apply historical crisis data to current portfolio.

        crisis_yield_changes: DataFrame with columns = tenor points, rows = crisis dates
            Each cell = daily yield change in that crisis at that tenor

        Algorithm:
        1. Start from current curve
        2. For each crisis day t:
            a. Apply the historical daily curve change to the curve
            b. Apply historical spread changes to corporate bonds
            c. Reprice entire portfolio
            d. Record daily P&L
        3. Report: cumulative P&L path, max drawdown, worst single day

        This answers: "If the 2008 crisis happened starting TODAY,
                       what would happen to THIS portfolio?"
        """

@dataclass
class StressResult:
    scenario_name: str
    scenario_id: str
    cumulative_pnl_path: list[float]    # daily cumulative P&L
    total_impact_pct: float             # total % loss
    total_impact_abs: float             # total absolute loss
    max_drawdown_pct: float
    worst_day_pnl: float
    worst_day_date: date
    daily_pnl: list[float]
```

### `stress/hypothetical.py`

```python
class HypotheticalScenario:
    def parallel_shift(self, portfolio: Portfolio, spot_curve: SpotCurve,
                       shift_bps: float) -> StressResult:
        """Shift entire curve by shift_bps. Simple but useful as baseline."""

    def steepener(self, portfolio: Portfolio, spot_curve: SpotCurve,
                  short_end_bps: float, long_end_bps: float) -> StressResult:
        """
        Different shifts for short vs long end.
        short_end = tenors <= 3Y
        long_end = tenors >= 10Y
        Intermediate tenors: linear interpolation
        """

    def spread_shock(self, portfolio: Portfolio, spot_curve: SpotCurve,
                     spread_changes_by_rating: dict[str, float]) -> StressResult:
        """
        Apply different spread shocks by credit rating.
        E.g., AAA: +100bps, AA: +200bps, BBB: +500bps
        Government bonds unaffected.
        """

    def combined_shock(self, portfolio: Portfolio, spot_curve: SpotCurve,
                       rate_shift_bps: float,
                       spread_changes_by_rating: dict[str, float]) -> StressResult:
        """
        Apply BOTH rate and spread shocks simultaneously.
        This is the most realistic — crises hit everything at once.
        """
```

### `stress/scenario_runner.py`

```python
class ScenarioRunner:
    def run_all_scenarios(self, portfolio: Portfolio,
                          spot_curve: SpotCurve,
                          config: ScenariosConfig) -> list[StressResult]:
        """
        Run all configured scenarios (historical + hypothetical).
        Returns list of StressResult sorted by impact (worst first).
        """

    def generate_report(self, results: list[StressResult]) -> pd.DataFrame:
        """
        Summary table:
        | Scenario | Total Impact (%) | Total Impact (Rs) | Max Drawdown | Worst Day |
        Sorted by severity
        """
```

---

## LAYER 8 — Regulatory Module

### `regulatory/expected_shortfall.py`

```python
class FRTBExpectedShortfall:
    def compute_es(self, mc_result: MonteCarloResult,
                   confidence: float = 0.975) -> float:
        """
        FRTB-compliant Expected Shortfall at 97.5% confidence.
        ES = average of losses beyond the 97.5th percentile
        FRTB requires 97.5% (not 95%) — this captures more of the tail
        """

    def compute_es_by_risk_class(self, mc_result: MonteCarloResult,
                                  risk_class_pnl: dict[str, np.ndarray]) -> dict[str, float]:
        """
        FRTB requires ES broken down by risk class:
        - GIRR (General Interest Rate Risk) — from rate factor P&L
        - CSR (Credit Spread Risk) — from spread factor P&L
        Each gets its own ES number
        """
```

### `regulatory/stressed_var.py`

```python
class StressedES:
    def compute_stressed_es(self, portfolio: Portfolio,
                             spot_curve: SpotCurve,
                             pca_model_stressed: PCAModel,
                             confidence: float = 0.975) -> float:
        """
        Same as normal ES, but using a PCA model fitted on STRESSED period data.

        1. Identify the worst 12-month period in the lookback window
           (e.g., Sep 2008 - Sep 2009)
        2. Fit a separate PCA model on that period's yield curve changes
        3. Run Monte Carlo using the stressed PCA model's volatilities
        4. Compute ES on the stressed simulation

        The stressed PCA will have LARGER eigenvalues (higher volatility),
        producing a larger ES number — this is the regulatory buffer for tail risk.
        """
```

---

## LAYER 9 — Streaming & Alerts

### `streaming/kafka_producer.py`

```python
class RiskEventProducer:
    def __init__(self, broker: str, topic: str):
        """Initialize confluent_kafka.Producer"""

    def publish_risk_snapshot(self, snapshot: RiskSnapshot) -> None:
        """
        Publish current risk metrics to Kafka topic.
        Message format (JSON):
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "portfolio_id": "fixedsense_main",
            "nav": 100_00_00_000,
            "total_dv01": 650000,
            "var_95": 1250000,
            "cvar_95": 1890000,
            "var_utilization_pct": 0.72,
            "kr01_profile": {"2Y": 50000, "5Y": 180000, "10Y": 350000, "30Y": 70000},
            "daily_pnl": -85000,
            "pnl_attribution": {"carry": 42000, "rate": -95000, "spread": -32000}
        }
        """

    def publish_alert(self, alert: Alert) -> None:
        """
        Publish to alerts topic when thresholds breached.
        {
            "alert_type": "VAR_BREACH",
            "severity": "HIGH",
            "message": "VaR utilization at 92% — above 80% threshold",
            "current_value": 0.92,
            "threshold": 0.80,
            "timestamp": "..."
        }
        """
```

### `streaming/alert_engine.py`

```python
class AlertEngine:
    def check_var_breach(self, current_var: float, var_limit: float) -> Alert | None:
        """Alert if VaR > threshold % of limit"""

    def check_concentration(self, kr01_profile: dict) -> Alert | None:
        """Alert if any single tenor has > 60% of total DV01"""

    def check_pnl_deviation(self, actual_pnl: float, predicted_pnl: float) -> Alert | None:
        """Alert if |actual - predicted| > 3x typical residual"""

    def check_spread_widening(self, spread_changes: dict) -> Alert | None:
        """Alert if any bond's spread widens > 20bps in one day"""

    def run_all_checks(self, risk_snapshot: RiskSnapshot) -> list[Alert]:
        """Run all alert checks and return triggered alerts"""
```

### `streaming/notifier.py`

```python
class Notifier:
    def send_email(self, alert: Alert, recipients: list[str]) -> None:
        """Send alert via SMTP (use smtplib)"""

    def send_slack(self, alert: Alert, webhook_url: str) -> None:
        """Send alert to Slack webhook (use requests.post)"""
```

---

## LAYER 10 — Streamlit Dashboard

### `dashboard/app.py`

Main Streamlit app entry point. Multi-page layout.

```python
"""
Structure:
    Sidebar: Portfolio selector, date picker, refresh button
    Pages:
        1. Overview — Portfolio summary
        2. Greeks — Duration, KR01, convexity charts
        3. Risk — VaR/CVaR, Monte Carlo distribution, decomposition
        4. P&L — Attribution waterfall, time series
        5. Stress — Scenario results table + charts
        6. Trade Simulator — What-if analysis
        7. Regulatory — FRTB metrics, backtesting results
"""
```

### `dashboard/pages/overview.py`

**Metrics row (top):**
- NAV (total portfolio value)
- Daily P&L (with green/red color)
- DV01 (in Rs)
- VaR 95% (in Rs)
- CVaR 95% (in Rs)
- VaR utilization % (with progress bar)

**Charts:**
- Current yield curve (spot + par + forward overlaid)
- Portfolio composition (pie chart by bond / by rating)
- NAV time series (last 30 days)

### `dashboard/pages/greeks.py`

**Charts:**
- KR01 bar chart (x=tenor, y=KR01 in Rs) — THE signature chart
- Duration contribution by bond (stacked bar)
- Convexity table per bond
- DV01 time series (has it been growing?)

### `dashboard/pages/risk.py`

**Charts:**
- Monte Carlo P&L distribution histogram with VaR/CVaR lines marked
- VaR decomposition by factor (horizontal bar: rate / spread / curve / residual)
- Marginal VaR by bond (which bond is adding the most risk?)
- VaR time series (last 30 days — is risk growing?)
- VaR backtest chart (actual P&L vs predicted VaR boundary)

### `dashboard/pages/pnl.py`

**Charts:**
- P&L attribution waterfall (carry → rate → spread → residual → total)
- P&L attribution time series (stacked area chart over last 30 days)
- P&L by bond (which bond contributed most today?)
- Cumulative P&L chart (total return since inception)

### `dashboard/pages/stress.py`

**Charts:**
- Scenario impact table (sorted by severity)
- Scenario comparison bar chart (horizontal bars showing % impact)
- For selected scenario: cumulative P&L path during the crisis
- Scenario heatmap: impact by scenario × bond

### `dashboard/pages/trade_simulator.py`

**Interactive:**
- Input: select a bond, enter notional amount (buy/sell)
- Show BEFORE vs AFTER comparison:
  - DV01 change
  - KR01 profile change (overlay chart)
  - VaR change
  - CVaR change
  - Marginal VaR of the new position
- Decision support: "Adding Rs 10Cr of GOI 30Y increases VaR by Rs 1.8L (+14%)"

### `dashboard/pages/regulatory.py`

**Display:**
- Expected Shortfall (97.5%) — current value
- Stressed Expected Shortfall — current value
- Capital requirement estimate (ES × multiplier)
- VaR backtest: traffic light status (green/yellow/red)
- Exception chart: calendar heatmap of VaR breach days

---

## Running the Project

### Local Development (no AWS needed)

```bash
# Clone and setup
git clone <repo>
cd fixedsense
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env: add FRED_API_KEY (get free at https://fred.stlouisfed.org/docs/api/api_key.html)

# Run data pipeline (fetches data, stores locally)
python -m data.processing.pipeline --mode local

# Run risk computation
python -m risk.monte_carlo --date today

# Launch dashboard
streamlit run dashboard/app.py
```

### Docker Setup

```yaml
# docker-compose.yml
services:
  fixedsense:
    build: .
    ports:
      - "8501:8501"   # Streamlit
    environment:
      - FRED_API_KEY=${FRED_API_KEY}
      - STORAGE_MODE=local
    volumes:
      - ./data/local_lake:/app/data/local_lake

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    ports:
      - "2181:2181"
```

### Production (AWS)

```bash
# Terraform for S3 bucket
# Airflow DAG for nightly pipeline
# ECS/Fargate for dashboard
# MSK for Kafka
# SNS for email alerts
```

---

## Testing Strategy

### Unit Tests (every module)

```
test_bootstrapper.py:
    - Test with known par yields → verify spot rates match expected values
    - Test that 1Y spot rate equals 1Y par rate
    - Test that longer spot rates are bootstrapped correctly

test_bond_pricer.py:
    - Price a par bond → should return face value
    - Price a zero-coupon bond → should equal FV / (1+z)^T
    - Verify clean price + accrued = dirty price

test_greeks.py:
    - Duration of zero-coupon bond should equal maturity
    - DV01 sanity: positive, proportional to duration × notional
    - KR01 sum should approximately equal total DV01
    - Convexity should always be positive for vanilla bonds

test_monte_carlo.py:
    - With zero volatility → all scenarios should produce same price
    - VaR at 50% confidence → should be close to 0 (median)
    - CVaR should always be >= VaR
    - More simulations → VaR should converge (run 5x and check variance)

test_pnl_attribution.py:
    - carry + rate + spread + residual should sum to total P&L
    - Carry should always be positive
    - On a day with no rate change → rate P&L should be ~0
```

### Integration Tests

```
test_full_pipeline.py:
    - Run pipeline with sample data → verify S3 writes
    - Bootstrap curve → price bonds → compute Greeks → run MC → compute VaR
    - Full end-to-end: should produce valid dashboard data
```

---

## Key Implementation Notes for Claude Code

1. **Start with Layer 0 and Layer 2.** Config + yield curve engine. Everything depends on these.

2. **Use local storage first.** Don't set up AWS until the logic works. The `LocalStorage` class should have the exact same interface as `S3Client`.

3. **Use sample data for development.** Before connecting to FRED API, create a `sample_data/` folder with pre-downloaded yield curve CSVs. This lets you iterate without API calls.

4. **NumPy vectorization everywhere.** Never use Python loops for pricing or Greek calculations. A loop over 10,000 Monte Carlo scenarios with 6 bonds should take < 2 seconds.

5. **Type hints on every function.** Use `dataclass` for all data objects. This is a quantitative system — precision matters.

6. **PCA model must be refit periodically** (weekly or monthly). Store the fitted model as a pickle in S3. The daily pipeline uses the cached model; a weekly job refits it.

7. **For the dashboard**, use `st.cache_data` for expensive computations (Monte Carlo, PCA). Use `st.session_state` for portfolio selections and user inputs.

8. **Kafka is optional for v1.** Build the dashboard first, then add Kafka streaming. The alert logic should work without Kafka (just log to console in dev mode).

9. **The waterfall chart** in P&L attribution should use Plotly's `go.Waterfall` — it has native support for this chart type.

10. **Every chart needs a title, axis labels, and annotations.** The dashboard is a portfolio manager's tool — it should be self-explanatory without documentation.

---

## What This Looks Like on Your CV

```
FixedSense — Production-Grade Fixed Income Risk System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Built end-to-end bond portfolio risk system with PCA-based yield curve
  factor model, computing KR01 Greeks at 9 tenor points, Monte Carlo VaR/CVaR
  (10,000 scenarios), and FRTB-compliant Expected Shortfall

• Implemented daily P&L attribution engine decomposing returns into carry,
  rate, spread, and residual components with automated alerting via Kafka

• Designed S3 data lake with Parquet storage (raw/processed/analytics zones),
  automated FRED API ingestion, and Streamlit dashboard with real-time
  risk monitoring and interactive trade simulation

Tech: Python, NumPy, SciPy, Pandas, PyArrow, Kafka, Streamlit, Plotly, S3, Docker
```

---

## Build Sequence Summary

| Phase | What to Build | Time Estimate |
|-------|--------------|---------------|
| 1 | Config + data models + sample data | 1 day |
| 2 | Yield curve engine (bootstrap + PCA) | 2-3 days |
| 3 | Bond pricer + cash flows | 1 day |
| 4 | Greeks engine (duration, DV01, KR01, convexity) | 2 days |
| 5 | Monte Carlo + VaR/CVaR | 2 days |
| 6 | Risk decomposition | 1-2 days |
| 7 | P&L attribution | 1-2 days |
| 8 | Stress testing | 1-2 days |
| 9 | Streamlit dashboard | 3-4 days |
| 10 | Kafka + alerts | 1-2 days |
| 11 | Regulatory module | 1 day |
| 12 | Testing + polish | 2-3 days |
| **Total** | | **~3-4 weeks** |
