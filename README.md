# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

> **Open Project 2026 — Society of Business**  
> Multi-agent AI framework for real-time EV charging tariff optimization using large-scale session data.

---

## Problem Statement

Static fixed-rate tariff models (₹15/kWh) are blind to real-world operational dynamics — causing peak-hour congestion, charger underutilization during off-peak windows, and deteriorating user experience. This project builds a self-improving agentic AI pricing engine that autonomously predicts demand, recommends dynamic tariffs in real time, and continuously learns from outcomes.

---

## Datasets

| Dataset | Source | Coverage | Format |
|---------|--------|----------|--------|
| ACN-Data (Adaptive Charging Network) | [ev.caltech.edu](https://ev.caltech.edu/dataset.html) | 14,991 sessions, Caltech site, Apr–Dec 2018 | JSON → CSV |
| ST-EVCDP (UrbanEV) | [ST-EVCDP GitHub](https://github.com/IntelligentSystemsLab/ST-EVCDP) | 247 zones, 2,134,080 records, 5-min intervals, Jun–Jul 2022 | CSV |

> **Note:** Raw data files are not included in this repository due to size. Download from the links above and place in the `/data` folder.

---

## Project Architecture

```
Raw Data (ACN + ST-EVCDP)
        ↓
Stage 1–3: Data Ingestion, Audit & Preprocessing
        ↓
Stage 4–6: Feature Engineering (78 features → 25 final)
        ↓
Stage 7: Multicollinearity Resolution (VIF + Correlation)
        ↓
┌───────────────────────────────────────────┐
│           THREE-AGENT PIPELINE            │
│                                           │
│  Agent 1: Demand Prediction (XGBoost)     │
│  Agent 2: Tariff Pricing                  │
│  Agent 3: Monitoring & Learning           │
└───────────────────────────────────────────┘
        ↓
Stage 11: Results, Evaluation & Export
```

---

## Three-Agent Framework

### Agent 1 — Demand Prediction Agent
- **Model:** XGBoost (`tree_method=hist`, 400 rounds, 25 features)
- **Task:** Predict utilisation rate 5 minutes ahead for each of 247 zones
- **Train:** Days 1–24 → 1,707,017 rows | **Test:** Days 25–30 → 426,569 rows
- **Split:** Chronological — zero data leakage

| Metric | Value |
|--------|-------|
| RMSE | 0.01437 |
| MAE | 0.00591 |
| R² | 0.99339 |
| Congestion Precision | 0.97 |
| Congestion Recall | 0.95 |

### Agent 2 — Tariff Pricing Agent
- **Base tariff:** ₹15.0/kWh (fixed problem statement baseline)
- **Surge:** util > 0.65 → up to ₹27.0/kWh (+80%)
- **Discount:** util < 0.15 → down to ₹14.2/kWh (−5%)
- **Elasticity assumed:** −0.5 (10% price rise → 5% demand drop)

| Metric | Value |
|--------|-------|
| Revenue Gain % | +0.54% |
| Avg Dynamic Tariff | ₹14.72/kWh |
| CUR Improvement | +0.47% |
| Off-Peak Uplift | 21.8% |
| Surge Sessions | 4.4% |
| Discount Sessions | 22.4% |

### Agent 3 — Monitoring & Learning Agent
- **Episodes:** 6 real (test days 25–30) + 44 simulated = 50 total
- **Drift detection:** PSI = 0.0092 → STABLE (threshold: 0.25)
- **Retraining triggered:** 1 time (Episode 1 only)

| KPI | Episode 1 | Episode 50 |
|-----|-----------|------------|
| Pricing Efficiency | 15.396 ¥/kWh | 15.433 ¥/kWh |
| Customer Response Rate | 1.9% | 38.5% |
| Off-Peak Uplift | 7.5% | 31.7% |
| Wait Time Reduction | 35.0% | ~26% (converged) |
| RMSE | 0.01837 | 0.01148 |

---

## Feature Engineering Summary

**25 final ML features across 6 categories:**

| Category | Features | Count |
|----------|----------|-------|
| Temporal | hour_sin, hour_cos, day_sin, day_cos, is_weekend, is_morning_rush, is_evening_rush, is_midnight_window, interval_of_day | 9 |
| Station | count, fast_charger_ratio, CBD, dynamic_pricing | 4 |
| Price | price_yuan, price_dev | 2 |
| Lag | util_lag_1, vol_lag_1 | 2 |
| Rolling | util_ewm_1h, util_roll_std_1h, util_delta_1, util_dev_from_1h | 4 |
| Spatial | neighbor_avg_util, neighbor_max_util, spillover_pressure, zone_avg_util | 4 |

**3 features dropped for multicollinearity:**
- `neighbor_wt_util` — r > 0.92 with `neighbor_avg_util`
- `util_roll_mean_6h` — r > 0.85 with `util_roll_mean_1h` and `util_ewm_1h`
- `minutes_since_midnight` — r = 1.0 with `interval_of_day`

---

## Repository Structure

```
ev-dynamic-pricing/
│
├── README.md
├── ev_pricing_enhanced.ipynb        ← Main notebook (all 16 stages)
│
├── data/
│   └── README.md                    ← Download instructions
│
├── outputs/
│   ├── stev_clean_export.csv        ← Preprocessed ST-EVCDP (2.13M rows)
│   ├── acn_clean_export.csv         ← Preprocessed ACN sessions (14,991 rows)
│   ├── final_feature_list.csv       ← 25 selected ML features
│   ├── feature_vif_scores.csv       ← VIF scores for all candidate features
│   └── high_correlation_pairs.csv   ← 37 high-correlation pairs (|r| > 0.85)
│
├── figures/                         ← All stage visualisations
│
└── models/
    └── demand_prediction_agent.json ← Saved XGBoost model
```

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ev-dynamic-pricing.git
cd ev-dynamic-pricing
```

2. **Install dependencies**
```bash
pip install pandas numpy polars xgboost scikit-learn matplotlib seaborn scipy
```

3. **Download data**
- ACN-Data: [ev.caltech.edu/dataset.html](https://ev.caltech.edu/dataset.html)
- ST-EVCDP: [github.com/IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP)
- Place all raw files in `/data` folder

4. **Run the notebook**
```bash
jupyter notebook ev_pricing_enhanced.ipynb
```
> Run all cells sequentially (Stages 1–11). Total runtime: ~15–20 minutes on standard hardware.

---

## Key Assumptions & Limitations

| Assumption | Stage | Risk |
|-----------|-------|------|
| Demand elasticity = −0.5 | Tariff Agent | May over/underestimate real user price sensitivity |
| Simulated episodes use Gaussian noise | Monitoring Agent | Real deployment drift may be non-Gaussian |
| kWhRequested imputed for 85.1% of ACN sessions | Preprocessing | Imputation bias may affect idle ratio estimates |
| Spatial adjacency threshold fixed at 5km | Feature Engineering | Different threshold changes neighbour graph structure |
| No causal claims on revenue/CUR improvements | All agents | All results are model projections pending field validation |

---

## Results Summary

```
╔══════════════════════════════════════════════════════════╗
║           EV DYNAMIC PRICING — PROJECT RESULTS           ║
╠══════════════════════════════════════════════════════════╣
║  DEMAND PREDICTION AGENT                                 ║
║    RMSE: 0.01437  |  MAE: 0.00591  |  R²: 0.99339       ║
╠══════════════════════════════════════════════════════════╣
║  TARIFF PRICING AGENT                                    ║
║    Revenue Gain: +0.54%  |  CUR: +0.47%                 ║
║    Off-Peak Uplift: 21.8%  |  Avg Tariff: ₹14.72/kWh    ║
╠══════════════════════════════════════════════════════════╣
║  MONITORING & LEARNING AGENT                             ║
║    Episodes: 6 real + 44 simulated = 50 total            ║
║    PSI: 0.0092 → STABLE  |  Retraining: 1 time           ║
║    Customer Response: 1.9% → 38.5% over 50 episodes      ║
╚══════════════════════════════════════════════════════════╝
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Polars](https://img.shields.io/badge/Polars-0.20-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

- **Data:** Polars, Pandas, NumPy
- **ML:** XGBoost, Scikit-learn
- **Visualisation:** Matplotlib, Seaborn, SciPy
- **Environment:** Jupyter Notebook / Google Colab

---

## Author

**Mahi** — IIT Roorkee  
Open Project 2026 | Society of Business
