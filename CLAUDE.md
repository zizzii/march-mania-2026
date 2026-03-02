# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate features only (sanity-check feature shapes)
python features.py

# Train models and produce both submission files (~5-10 min)
python train.py
```

Outputs: `submission_stage1.csv` (Seasons 2022-2025, ~519K rows) and `submission_stage2.csv` (Season 2026, ~132K rows).

## Architecture

Two-file structure:

- **`features.py`** — Feature engineering only. Reads raw CSVs from `data/`, returns `(Season, TeamID)` indexed DataFrames. Entry point: `build_team_features(gender='M'|'W')`.
- **`train.py`** — Imports `features.py`, builds matchup-level diffs, runs time-series CV, fits final models, writes submission CSVs. Entry point: `main()`.

### Feature pipeline (features.py)

`build_team_features()` merges these per-team per-season tables:

| Source | Function | Notes |
|---|---|---|
| `{M,W}RegularSeasonCompactResults.csv` | `compute_elo()` | MOV-adjusted Elo, K=20, 75% carryover |
| same | `compute_winrate_compact()` | WinRate, AvgMargin |
| same | `compute_recent_form()` | Last 15 games, exp-decay weights (half-life=5) |
| same | `compute_sos()` | Avg opponent end-of-season Elo |
| `{M,W}RegularSeasonDetailedResults.csv` | `compute_efficiency()` | OffRtg/DefRtg/NetRtg, Four Factors, z-scores, NeutralNetRtg_z |
| `{M,W}ConferenceTourneyGames.csv` | `compute_conf_tourney()` | NaN for 2026 (not played yet) |
| `MMasseyOrdinals.csv` | `compute_massey_features()` | Consensus percentile (MasseyMean/Min/Std) + KPK/POM/NET/MOR |
| `{M,W}NCAATourneySeeds.csv` | `parse_seeds()` + `impute_seeds()` | Real seeds when available; imputed for 2026 from MasseyMean (M) or Elo (W) |

### Model pipeline (train.py)

Matchup features are computed as **differences**: `D_{feat} = T1_{feat} - T2_{feat}` where T1 < T2 by TeamID.

**Final models:**
- **Men**: 85% LR + 15% XGB blend. LR uses 8 core features expanded via `PolynomialFeatures(degree=2, interaction_only=True)` → 36 features, C=1.0.
- **Women**: Pure LR, same poly expansion, C=0.2.

**CV strategy**: Time-series leave-one-season-out over last 5 seasons. CV decides whether to use blend or pure LR.

**Key constants in train.py:**
- `LR_CORE_FEATURES` — the 8 features used by LR
- `ALL_DIFF_FEATURES` — the ~37 features available to XGB
- `LR_C = {'M': 1.0, 'W': 0.2}` — regularization per gender
- `XGB_BLEND = 0.15` — XGB weight in ensemble
- `CLIP_LO, CLIP_HI = 0.02, 0.98` — probability clipping

### Critical modeling constraint

Tournament training data is tiny (~67 games/year, ~300-400 samples per CV fold). **More features hurt.** The 8-feature LR beats XGB by ~0.03 log-loss. Do not add features to the LR feature set without CV validation.

## Re-run checklist after Selection Sunday (mid-March 2026)

1. Update `{M,W}NCAATourneySeeds.csv` with real 2026 seeds
2. Re-run `python train.py` — real seeds replace imputed ones and substantially improve Stage 2 predictions
3. SeedNum is the strongest individual feature when available

## Data notes

- M team IDs < 2000; W team IDs ≥ 3000
- Massey file covers both genders; filter by TeamID range (`< 3000` for M, `>= 3000` for W)
- Poll systems (AP, USA, DES) are excluded from Massey consensus — they only rank ~25 teams
- Detailed results: M available 2003+, W available 2010+; code falls back to compact gracefully
- Submission ID format: `Season_Team1ID_Team2ID` with Team1ID < Team2ID
