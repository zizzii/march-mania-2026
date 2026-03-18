# March Mania 2026

NCAA tournament win probability prediction for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition (men's + women's).

**Metric**: Log loss on predicted win probabilities.

## Results

| Gender | Model | CV Log Loss |
|--------|-------|-------------|
| Men | 85% LR + 15% XGB blend | 0.5621 |
| Women | Pure LR | 0.4160 |

Baseline (naive Elo only): M=0.6079, W=0.5471.

## Setup

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost

# Place Kaggle competition CSVs in data/
# (download from the competition page)

# Generate features only (sanity check)
python features.py

# Train models and produce submission files (~5-10 min)
python train.py
```

Outputs: `submission_stage1.csv` (seasons 2022–2025) and `submission_stage2.csv` (season 2026).

## Architecture

### `features.py` — Feature Engineering

Builds per-team, per-season features from raw CSVs. Entry point: `build_team_features(gender='M'|'W')`.

| Feature | Source | Notes |
|---------|--------|-------|
| Elo | Regular season compact results | MOV-adjusted, K=20, 75% carryover |
| WinRate, AvgMargin | Regular season compact results | Season aggregates |
| Recent Form | Regular season compact results | Last 15 games, exp-decay half-life=5 |
| SOS | Regular season compact results | Avg opponent end-of-season Elo |
| OffRtg / DefRtg / NetRtg | Detailed results | Season-normalized z-scores |
| NeutralNetRtg_z | Detailed results | Net rating in non-home games only |
| Four Factors | Detailed results | eFG%, TOVPct, FTR, ORBPct |
| Massey consensus | MMasseyOrdinals.csv | Percentile across all ranking systems + KPK/POM/NET/MOR |
| SeedNum | NCAATourneySeeds.csv | Real 2026 seeds loaded (bracket released 2026-03-17) |
| Conf tourney wins | ConferenceTourneyGames.csv | NaN for 2026 (not played yet) |

### `train.py` — Model & Submission

Matchup features are **differences**: `D_feat = T1_feat − T2_feat` (T1 < T2 by TeamID).

**Key insight**: tournament training data is tiny (~67 games/year). Logistic Regression beats XGBoost by ~0.03 log-loss. The 8-feature LR with polynomial interactions is the core model.

**LR core features** (8):
`D_Elo`, `D_MasseyMean`, `D_MasseyMin`, `D_SeedNum`, `D_WinRate`, `D_AvgMargin`, `D_SOS`, `D_NetRtg_z`

These expand to 36 features via `PolynomialFeatures(degree=2, interaction_only=True)`.

**CV strategy**: time-series leave-one-season-out over the last 5 seasons.

## After Selection Sunday (2026-03-17 — done)

Real 2026 seeds have been added to `MNCAATourneySeeds.csv` and `WNCAATourneySeeds.csv`. Re-running `python train.py` picks them up automatically — no code changes needed. SeedNum is the strongest individual feature and substantially improves Stage 2 predictions.
