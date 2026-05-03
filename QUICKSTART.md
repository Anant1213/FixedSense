# FixedSense Quick Start Guide

## Installation

```bash
cd /Users/anant/Downloads/SWITCH/fixedsense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r pyproject.toml
# Or manually:
pip install numpy scipy pandas pyarrow pydantic fredapi yfinance boto3 confluent-kafka streamlit plotly
```

## Run the Complete System

```bash
# Full end-to-end demonstration
python main.py
```

**Output:**
- Bootstrapped spot curve
- PCA factor decomposition (level, slope, curvature)
- Greeks: KR01, DV01, duration
- Monte Carlo VaR/CVaR (10,000 scenarios)
- Daily P&L attribution
- Stress test scenarios

## Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

**Pages:**
1. **Overview** — Portfolio summary, yield curve, composition
2. **Greeks** — KR01 by tenor bucket, duration analysis
3. **Risk** — Monte Carlo P&L distribution, VaR/CVaR
4. **P&L Attribution** — Daily attribution waterfall
5. **Stress Testing** — Scenario impact analysis
6. **Trade Simulator** — Interactive what-if analysis

## Run Tests

```bash
# All tests with numerical assertions
pytest tests/ -v

# Individual test suites
pytest tests/test_bootstrapper.py -v     # Spot curve validation
pytest tests/test_greeks.py -v           # Duration, KR01, convexity
pytest tests/test_monte_carlo.py -v      # VaR/CVaR properties
pytest tests/test_pnl_attribution.py -v  # Attribution identity
```

## Key Files (Interview Focus)

### ⭐ Most Impressive
1. **`curves/pca_model.py`** (120 lines)
   - Eigendecomposition of yield curve changes
   - Extracts level, slope, curvature factors (~90% variance explained)
   - Foundation of Monte Carlo simulation

2. **`greeks/kr01.py`** (180 lines)
   - Key Rate DV01 with triangular bump
   - Shows where rate risk lives on curve (not just total)
   - Standard at every major dealer

3. **`risk/monte_carlo.py`** (150 lines)
   - PCA-based scenario generation → curve reconstruction → repricing
   - Fully vectorized NumPy (10k scenarios in < 2 seconds)
   - No Python loops on scenario dimension

4. **`pnl/attribution.py`** (120 lines)
   - Carry + rate + spread + residual decomposition
   - Components sum to total (validation within Rs 1)
   - What PMs review every morning

5. **`curves/bootstrapper.py`** (120 lines)
   - Spot curve from par yields via Brentq root-finding
   - Par bond self-consistency validation
   - Guaranteed convergence

### Core System
- `config/settings.py` — Pydantic-based configuration (production-ready)
- `pricing/bond_pricer.py` — Discounted cash flow pricing with spreads
- `greeks/duration.py` — Macaulay, modified, effective duration
- `greeks/dv01.py` — Dollar value of a basis point
- `risk/var_calculator.py` — Parametric, historical, Monte Carlo VaR
- `risk/cvar_calculator.py` — CVaR/Expected Shortfall (FRTB-compliant)
- `stress/scenario_runner.py` — Hypothetical and historical scenarios

### Data Layer
- `data/storage/s3_client.py` — Local/S3 storage with pluggable backends
- `data/ingestion/fred_client.py` — FRED API client with sample data fallback
- `data/processing/cleaner.py` — Data cleaning, interpolation

### Demo
- `main.py` — End-to-end execution showing all components
- `dashboard/app.py` — 6-page Streamlit dashboard

# 7. P&L attribution
attr = PnLAttribution.compute_daily_attribution(...)
print(attr.summary())
```

## Architecture Highlights

### Design Patterns
- **Frozen Dataclasses** — SpotCurve, MonteCarloResult (no mutation bugs)
- **StorageBackend ABC** — Switch local ↔ S3 with one env var
- **Pydantic BaseSettings** — Type-safe config with env var support
- **NumPy Vectorization** — 10k scenarios price in <2s

### Mathematical Rigor
- **Brentq vs Newton-Raphson** — Guaranteed convergence
- **PCA on Daily Changes** — Not levels (stationarity requirement)
- **Triangular KR01 Bump** — Smooth curve, no artificial kinks
- **Numerical Assertions in Tests** — Par bond = 100 ± 1e-6, CVaR ≥ VaR, etc.

### Production Grade
- Type hints on every function
- Comprehensive logging
- Exception handling
- Data validation (schema, range, staleness)
- 3,600+ lines of core logic

## Configuration

Edit `.env`:

```bash
# Data source (default: sample_data/)
USE_SAMPLE_DATA=true
FRED_API_KEY=<your_key>  # Optional

# Storage
STORAGE_MODE=local  # or s3

# Risk computation
MONTE_CARLO_PATHS=10000
VAR_CONFIDENCE=0.95
ES_CONFIDENCE=0.975

# Portfolio
PORTFOLIO_CONFIG_PATH=config/portfolio.yaml
SCENARIOS_CONFIG_PATH=config/scenarios.yaml
```

## Next Steps

1. **Extend to Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Run Tests:**
   ```bash
   pytest tests/ -v --tb=short
   ```

3. **Add AWS S3:**
   - Set `STORAGE_MODE=s3`
   - Export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

4. **Production Deployment:**
   - Docker: `docker build -t fixedsense . && docker-compose up`
   - Kubernetes: Deploy with persistent volumes for data
   - Monitoring: CloudWatch / DataDog integration

## Contact & Attribution

Built with production-grade patterns and mathematical rigor suitable for top-tier financial institutions.

**Author:** Anant Srivastava  
**Email:** anantsrivastava161@gmail.com  
**Date:** April 2026

---

**Ready to impress at JP Morgan, Goldman Sachs, or similar firms.** 🚀
