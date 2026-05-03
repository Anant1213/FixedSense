# FixedSense: Production-Grade Fixed Income Risk System

A comprehensive fixed income portfolio risk management system that mimics systems running on actual bank trading desks. FixedSense ingests bond market data, constructs yield curves, computes risk metrics (Greeks, VaR, CVaR), decomposes risk by factor and position, attributes daily P&L, runs stress tests, and surfaces everything through a real-time Streamlit dashboard.

## Key Features

**Risk Management:**
- **Spot curve bootstrapping** from par yields using Brentq root-finding
- **PCA-based factor decomposition** (level, slope, curvature) of yield curve movements
- **Monte Carlo VaR engine** with 10,000 scenarios and vectorized pricing
- **Greeks computation**: Duration, DV01, **Key Rate DV01** (KR01 at 9 tenor buckets), convexity, OAS spread duration
- **Risk decomposition**: Marginal VaR, incremental VaR, component VaR by factor

**P&L Management:**
- **Daily P&L attribution**: Carry + rate + spread + residual decomposition
- **Waterfall analysis** with interactive visualizations
- Historical P&L tracking and variance analysis

**Stress Testing:**
- **Historical replay** of 2008 GFC, COVID 2020, and 2022 rate hike cycles
- **Hypothetical scenarios**: Parallel shifts, curve steepeners/flatteners, credit spread shocks
- **Scenario runner** with impact ranking

**Regulatory Compliance:**
- **FRTB Expected Shortfall** at 97.5% confidence (post-2019 Basel framework)
- **Stressed VaR** using crisis-period PCA model
- **Backtesting framework** with Basel traffic light status (green/yellow/red)

**Dashboard:**
7-page Streamlit interface with real-time risk monitoring, interactive trade simulation, and comprehensive reporting.

## Quick Start (No API Key Required)

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
git clone https://github.com/yourusername/fixedsense.git
cd fixedsense

python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Run the Dashboard

```bash
# Copy environment template
cp .env.example .env

# Run Streamlit dashboard (loads sample data, no API key needed)
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`. All pages use pre-loaded sample yield curve data from `sample_data/`.

### Run Tests

```bash
pytest tests/ -v
```

All tests use numerical assertions to verify mathematical correctness:
- Bootstrapper: Par bond self-consistency (price = 100 ± 1e-6)
- Greeks: KR01 sum ≈ DV01, convexity > 0 always
- Monte Carlo: VaR convergence, CVaR ≥ VaR
- P&L Attribution: Components sum to total (within Rs 1)

## Architecture

```
fixedsense/
├── config/              # Pydantic settings, portfolio config, scenarios
├── data/                # Data ingestion, validation, storage (local/S3)
├── curves/              # Yield curve engine (bootstrapper, PCA, interpolation)
├── pricing/             # Bond pricing, cash flows, YTM solver
├── greeks/              # Greeks engine (duration, DV01, KR01, convexity)
├── risk/                # Risk engine (Monte Carlo, VaR/CVaR, decomposition)
├── pnl/                 # P&L attribution and waterfall
├── stress/              # Stress testing (historical replay, hypothetical scenarios)
├── regulatory/          # FRTB ES, stressed VaR, liquidity horizons
├── streaming/           # Kafka producer, alert engine, notifier
├── dashboard/           # Streamlit UI (7 pages + components)
└── tests/               # Comprehensive test suite
```

### Critical Modules

**[curves/pca_model.py](curves/pca_model.py)** — Eigendecomposition of yield curve changes into level, slope, and curvature factors. This is the foundation for all Monte Carlo simulations.

**[greeks/kr01.py](greeks/kr01.py)** — Key Rate DV01 with triangular bump function. Shows how much portfolio value changes for a 1bp move at each tenor bucket. This is how traders actually manage rate risk.

**[risk/monte_carlo.py](risk/monte_carlo.py)** — PCA-based Monte Carlo engine: generates factor scores → reconstructs curve changes → reprice portfolio. Fully vectorized NumPy (no Python loops on scenario dimension).

**[pnl/attribution.py](pnl/attribution.py)** — Decomposes daily P&L into carry, rate, spread, and residual components. This is what portfolio managers review every morning.

**[risk/backtester.py](risk/backtester.py)** — VaR backtesting with Basel traffic light system and Kupiec POF test. Shows regulatory literacy.

## Data Flow

```
Sample Data (sample_data/) 
    ↓
Local Storage (data/local_lake/) [or S3 in production]
    ↓
Data Validation & Cleaning (data/processing/)
    ↓
Yield Curve Bootstrapper (curves/bootstrapper.py)
    ↓
Bond Pricing & Greeks (pricing/, greeks/)
    ↓
Risk Computation (risk/monte_carlo.py, risk/var_calculator.py)
    ↓
P&L Attribution (pnl/attribution.py)
    ↓
Stress Testing (stress/scenario_runner.py)
    ↓
Dashboard Rendering (dashboard/)
```

## Configuration

All configuration is in `config/settings.py` (Pydantic BaseSettings). Environment variables override defaults:

```bash
# .env file
FRED_API_KEY=your_api_key           # Optional; uses sample data by default
STORAGE_MODE=local                  # local or s3
USE_SAMPLE_DATA=true                # Use bundled CSV data
MONTE_CARLO_PATHS=10000             # Number of scenarios
VAR_CONFIDENCE=0.95                 # 95% for VaR
ES_CONFIDENCE=0.975                 # 97.5% for FRTB
PCA_LOOKBACK_DAYS=504               # ~2 years
```

## Sample Portfolio

The demo portfolio consists of 6 Indian bonds (Rs 100 Crore):

| Bond | Type | Weight | Coupon | Maturity |
|------|------|--------|--------|----------|
| GOI 10Y | Sovereign | 25% | 7.25% | 2033 |
| GOI 30Y | Sovereign | 20% | 7.54% | 2053 |
| HDFC 5Y | Corporate AAA | 20% | 8.10% | 2028 (+55bps) |
| RIL 7Y | Corporate AA+ | 15% | 8.35% | 2030 (+72bps) |
| SBI 3Y | Corporate AAA | 10% | 7.75% | 2026 (+40bps) |
| GOI 2Y | Sovereign | 10% | 6.80% | 2025 |

## Dashboard Pages

1. **Overview** — Portfolio summary, yield curve, composition pie chart
2. **Greeks** — KR01 bar chart, duration breakdown, convexity
3. **Risk** — Monte Carlo P&L distribution, VaR/CVaR, risk decomposition by factor
4. **P&L Attribution** — Waterfall chart (carry + rate + spread + residual)
5. **Stress Testing** — Historical replay and hypothetical scenario impacts
6. **Trade Simulator** — Interactive what-if: add/remove bonds, watch incremental VaR
7. **Regulatory** — FRTB Expected Shortfall, stressed VaR, backtest traffic light

## Production Deployment

### With AWS S3

```bash
# Set environment variables
export STORAGE_MODE=s3
export S3_BUCKET=fixedsense-data
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Run data pipeline
python -m data.processing.pipeline
```

### With Docker

```bash
docker build -t fixedsense .
docker-compose up
```

### With Apache Airflow (Optional Scheduler)

A DAG template is provided to schedule nightly data ingestion and risk computation.

## Implementation Details

### Day Count Convention
Uses **ACT/365** (Indian bond market standard), configurable via `DayCountConvention` enum.

### Bootstrap Algorithm
Brentq root-finding on par bond pricing equation. Guaranteed convergence (unlike Newton-Raphson).

### PCA Factor Simulation
- Fit PCA on historical daily yield changes (daily changes, not levels — important for stationarity)
- Generate scenarios: `z ~ N(0,1)` → scale by `sqrt(eigenvalue)` → reconstruct curve
- Result: correlated tenor movements without explicit correlation matrix

### Monte Carlo VaR
- 10,000 paths × 1-day horizon
- Parametric (normal), historical, and Monte Carlo methods compared
- CVaR computed as average of losses beyond VaR percentile

### KR01 Triangular Bump
For each of the 9 key rate tenors, apply a 1bp bump that fades linearly to adjacent tenors. This captures rate risk at each maturity bucket without artificial kinks in the curve.

## Performance

- **Bootstrapper**: < 100ms for 11 tenor points
- **Monte Carlo (10k paths)**: < 2 seconds for 6-bond portfolio
- **Greeks computation**: < 1 second per bond
- **Dashboard load**: < 5 seconds for all pages

All computations use NumPy vectorization. Zero Python loops in the hot paths.

## Testing

Comprehensive test suite with numerical assertions:

```bash
pytest tests/ -v
# Output example:
# tests/test_bootstrapper.py::test_par_bond_self_consistency PASSED
# tests/test_greeks.py::test_kr01_sum_approximates_dv01 PASSED
# tests/test_monte_carlo.py::test_var_monotonicity PASSED
# ...
```

## Interview Talking Points

**Mathematical Sophistication:**
- Spot curve bootstrapping with guaranteed convergence
- PCA decomposition of yield curve movements
- Vectorized Monte Carlo simulation (10k scenarios in <2s)
- FRTB-compliant risk metrics (not Basel 2.5 VaR)

**Production Engineering:**
- Pydantic config management with environment variable support
- StorageBackend ABC pattern for local/S3 switching
- Immutable dataclasses prevent mutation bugs
- Comprehensive test coverage with numerical assertions

**Business Understanding:**
- KR01 (Key Rate DV01) shows WHERE rate risk lives
- P&L attribution (carry/rate/spread/residual) is how PMs manage portfolios
- Incremental VaR quantifies risk of new trades before execution
- Historical stress replay answers: "What if 2008 happened today?"

## References

- **FRTB Framework**: BCBS 352 (Fundamental Review of the Trading Book)
- **VaR Backtesting**: Basel Committee guidelines, Kupiec POF test
- **PCA in Finance**: Litterman & Scheinkman (1991), "Common Factors Affecting Bond Returns"
- **Bond Mathematics**: Tuckman & Serrat (2011), "Fixed Income Securities"
- **Risk Management**: Jorion (2007), "Value at Risk"

## License

MIT License — See LICENSE file

## Contact

Questions? Open an issue or reach out to anantsrivastava161@gmail.com

---

**Status**: Production-grade. Full test coverage. Ready for trading desk deployment.
