# 🔗 Forex Correlation Matrix + Pair Rotation Tracker

A quant tool that computes rolling correlations across 8 major forex pairs, detects statistical divergence events when correlated pairs decouple, and ranks pairs by momentum rotation — all in a single dark-themed dashboard.


## 💡 Core Concept

Forex pairs don't move independently — they share underlying USD flows, risk sentiment, and commodity linkages. This tool exploits two edges:

### 1. Correlation Divergence (Mean-Reversion Edge)
```
EURUSD and GBPUSD normally correlate at ~0.87.
When their normalised spread Z-Score exceeds ±2σ:
  → One pair is temporarily mispriced vs the other
  → Spread tends to mean-revert → statistical edge
```

### 2. Pair Rotation (Momentum Edge)
```
Track which pairs are leading (strong momentum) vs
lagging (weak momentum) over a rolling N-day window.
Rotate exposure toward leaders, fade laggards.
```

---

## ✅ Features

- **Rolling Correlation Heatmap** — 8×8 colour-coded matrix, updates on configurable window
- **Spread Z-Score Engine** — detects divergence events with threshold alerts
- **Pair Rotation Bar Chart** — ranks all 8 pairs by N-day return momentum
- **Rolling Corr Timeline** — tracks 3 key pair relationships over time
- Realistic correlated price simulation using **Cholesky decomposition**
- Console summary with full divergence event log + rotation leaderboard

---



## 📁 Project Structure

```
forex-correlation-tracker/
├── forex_correlation.py    # All-in-one tracker script
├── output.png              # Sample dashboard output
└── README.md
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAIRS` | 8 majors | List of forex pairs to track |
| `CORR_WINDOW` | 30 days | Rolling window for correlation |
| `DIV_ZSCORE_THRESH` | 2.0 | Z-score threshold for divergence alert |
| `SPREAD_WINDOW` | 20 days | Lookback for spread mean/std |
| `ROTATION_WINDOW` | 10 days | Momentum window for rotation ranking |
| `N_DAYS` | 180 days | Total simulation period |

---

## 📊 Dashboard Panels

| Panel | Description |
|---|---|
| **Correlation Heatmap** | Full 8×8 matrix with colour coding (red = negative, green = positive) |
| **Rotation Bar Chart** | N-day % return per pair ranked #1–#8 |
| **Divergence Z-Score** | EURUSD/GBPUSD spread with ±2σ threshold bands and event markers |
| **Rolling Corr × 3** | Live rolling correlation lines for EUR/GBP, AUD/NZD, CAD/CHF |

---

## 📈 Sample Output (180 Days)

```
Rolling 30D Correlations:
  EURUSD/GBPUSD   +0.731  ██████████
  AUDUSD/NZDUSD   +0.792  ███████████
  USDCAD/USDCHF   +0.630  █████████
  EURUSD/USDCHF   -0.885  █████████████

Divergence Events: 7 detected
  Largest: -2.38σ  (Long Spread signal)

Rotation Leader:  EURGBP  ▲ +0.280%
Rotation Laggard: NZDUSD  ▼ -2.380%
```

---

## 🔌 Use with Real Data

```python
import yfinance as yf

tickers = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",    "USDCHF": "CHF=X",
    "USDJPY": "JPY=X",    "EURGBP": "EURGBP=X"
}
prices = {k: yf.download(v, period="1y", interval="1d")["Close"]
          for k, v in tickers.items()}
prices_df = pd.DataFrame(prices).dropna()
```

---

## 📚 Quant Concepts Used

| Concept | Description |
|---|---|
| Cholesky Decomposition | Generate realistically correlated synthetic returns |
| Rolling Pearson Correlation | Time-varying pair relationships |
| Z-Score Spread | Standardised divergence measure |
| Cross-Sectional Momentum | Pair rotation ranking by recent returns |

---

## 🛠 Requirements

- Python 3.8+
- pandas · numpy · matplotlib · scipy

MAINLY BUILT FOR TRADING IN NEW YORK  ZONE WITH MARKING ASIAN / LONDON HIGHS AND LOWS

