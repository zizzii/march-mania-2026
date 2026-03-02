"""
Train an ensemble model for March Mania 2026 and generate submission files.

Key insight: tournament datasets are small (~67 games/year). Logistic Regression
on 8 carefully chosen features dramatically outperforms XGBoost (0.573 vs 0.601
for men). XGBoost overfits on <2000 tournament training samples.

Final model: Logistic Regression (primary) + XGBoost (20% blend for nonlinearity)
Features: D_Elo, D_MasseyMean, D_MasseyMin, D_SeedNum, D_WinRate, D_AvgMargin,
          D_SOS, D_NetRtg_z
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

from features import build_team_features, compute_seed_matchup_rates, parse_seeds


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# All possible difference features (used for XGB fallback / exploration)
ALL_DIFF_FEATURES = [
    'Elo', 'WinRate', 'AvgMargin', 'RecentWinRate', 'RecentMargin',
    'SOS', 'ConfTourneyWins',
    'OffRtg', 'DefRtg', 'NetRtg', 'OffRtg_z', 'DefRtg_z', 'NetRtg_z', 'Tempo',
    'FGPct', 'FG3Pct', 'FTPct', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk',
    'MasseyMean', 'MasseyStd', 'MasseyMin',
    'Rank_KPK', 'Rank_MOR', 'Rank_NET', 'Rank_POM', 'SeedNum',
    # New: Four Factors + neutral-site efficiency
    'eFG', 'TOVPct', 'FTR', 'ORBPct', 'NeutralNetRtg_z',
]

# Core features for LR — fewest features that give best CV (validated)
LR_CORE_FEATURES = [
    'Elo', 'MasseyMean', 'MasseyMin', 'SeedNum',
    'WinRate', 'AvgMargin', 'SOS', 'NetRtg_z',
]

CLIP_LO, CLIP_HI = 0.02, 0.98

# Per-gender tuned LR regularization (C selected via 5-year CV with poly features)
LR_C = {'M': 1.0, 'W': 0.2}
# XGB blend weight (0 = pure LR; small fraction adds nonlinear correction)
XGB_BLEND = 0.15


def compute_diff_features(df: pd.DataFrame,
                          feature_list: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add D_* difference columns for all features available in df."""
    available = []
    for feat in feature_list:
        c1, c2 = f'T1_{feat}', f'T2_{feat}'
        if c1 in df.columns and c2 in df.columns:
            df[f'D_{feat}'] = df[c1] - df[c2]
            available.append(f'D_{feat}')
    return df, available


def build_matchup_df(tourney_results: pd.DataFrame,
                     team_feats: pd.DataFrame,
                     seed_matchup_rates: pd.DataFrame = None,
                     gender: str = 'M') -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build matchup DataFrame with all difference features.
    Returns (merged_df, lr_feat_cols, all_feat_cols).
    """
    games = tourney_results.copy()
    games['T1'] = games[['WTeamID', 'LTeamID']].min(axis=1)
    games['T2'] = games[['WTeamID', 'LTeamID']].max(axis=1)
    games['Label'] = (games['WTeamID'] == games['T1']).astype(int)

    t1f = team_feats.add_prefix('T1_').rename(columns={'T1_Season': 'Season', 'T1_TeamID': 'T1'})
    t2f = team_feats.add_prefix('T2_').rename(columns={'T2_Season': 'Season', 'T2_TeamID': 'T2'})
    merged = games.merge(t1f, on=['Season', 'T1'], how='left')
    merged = merged.merge(t2f, on=['Season', 'T2'], how='left')
    merged, all_feats = compute_diff_features(merged, ALL_DIFF_FEATURES)

    # Seed matchup historical win rate (matchup-level, not a diff feature)
    matchup_extra = []
    if (seed_matchup_rates is not None
            and 'T1_SeedNum' in merged.columns
            and 'T2_SeedNum' in merged.columns):
        smr = seed_matchup_rates.set_index(['SeedNum1', 'SeedNum2'])['SeedMatchupRate']
        s1 = merged['T1_SeedNum'].fillna(17).clip(1, 17).astype(int)
        s2 = merged['T2_SeedNum'].fillna(17).clip(1, 17).astype(int)
        fallback = smr.mean()

        def lookup_rate(t1_seed, t2_seed):
            if t1_seed <= t2_seed:
                return smr.get((t1_seed, t2_seed), fallback)
            else:
                return 1.0 - smr.get((t2_seed, t1_seed), 1.0 - fallback)

        merged['SeedMatchupRate'] = [lookup_rate(a, b) for a, b in zip(s1, s2)]
        matchup_extra = ['SeedMatchupRate']

    lr_feats = [f'D_{f}' for f in LR_CORE_FEATURES if f'D_{f}' in all_feats]
    all_feats = all_feats + matchup_extra

    # Only require Elo (available for all years)
    merged = merged.dropna(subset=['D_Elo'])
    return merged, lr_feats, all_feats


def make_lr_pipeline(C=1.0):
    """LR pipeline: pairwise interactions → z-score → logistic regression."""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])


def make_xgb(n_estimators=400):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
    )


def ensemble_predict(lr_pipeline, xgb_model, X_lr, X_xgb, xgb_weight=XGB_BLEND):
    """Predict using LR pipeline (poly+scale+lr) + optional XGB blend."""
    p_lr = np.clip(lr_pipeline.predict_proba(X_lr)[:, 1], CLIP_LO, CLIP_HI)
    if xgb_model is not None and xgb_weight > 0:
        p_xgb = np.clip(xgb_model.predict_proba(X_xgb)[:, 1], CLIP_LO, CLIP_HI)
        return np.clip((1 - xgb_weight) * p_lr + xgb_weight * p_xgb, CLIP_LO, CLIP_HI)
    return p_lr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(gender: str = 'M', eval_years: int = 5):
    """
    Time-series CV evaluation then final retraining on all data.
    Returns (lr_model, scaler, xgb_model, lr_feat_cols, all_feat_cols, cv_ll, team_feats).
    """
    print(f"\n{'='*55}")
    print(f"Training {gender} model")
    print(f"{'='*55}")

    tourney = pd.read_csv(f'data/{gender}NCAATourneyCompactResults.csv')
    seeds_raw = pd.read_csv(f'data/{gender}NCAATourneySeeds.csv')
    team_feats = build_team_features(gender)

    seed_matchup_rates = compute_seed_matchup_rates(tourney, seeds_raw)
    matchups, lr_feats, all_feats = build_matchup_df(tourney, team_feats,
                                                      seed_matchup_rates, gender)
    C = LR_C[gender]

    print(f"Training samples: {len(matchups)}")
    print(f"LR features ({len(lr_feats)}): {lr_feats}")
    print(f"XGB features: {len(all_feats)}")
    print(f"LR regularization C={C}, XGB blend weight={XGB_BLEND}")

    all_seasons = sorted(matchups.Season.unique())
    test_seasons = all_seasons[-eval_years:]
    cv_scores_lr, cv_scores_blend = [], []

    for test_season in test_seasons:
        train_df = matchups[matchups.Season < test_season]
        test_df = matchups[matchups.Season == test_season]

        if len(test_df) == 0:
            continue

        X_tr_lr = train_df[lr_feats].fillna(0)
        X_te_lr = test_df[lr_feats].fillna(0)
        X_tr_xgb = train_df[all_feats]
        X_te_xgb = test_df[all_feats]
        y_tr = train_df['Label']
        y_te = test_df['Label']

        lr_pipe = make_lr_pipeline(C)
        lr_pipe.fit(X_tr_lr, y_tr)
        p_lr = np.clip(lr_pipe.predict_proba(X_te_lr)[:, 1], CLIP_LO, CLIP_HI)

        xgb_m = make_xgb()
        xgb_m.fit(X_tr_xgb, y_tr)
        p_blend = np.clip(ensemble_predict(lr_pipe, xgb_m, X_te_lr, X_te_xgb), CLIP_LO, CLIP_HI)

        ll_lr = log_loss(y_te, p_lr)
        ll_blend = log_loss(y_te, p_blend)
        cv_scores_lr.append(ll_lr)
        cv_scores_blend.append(ll_blend)
        print(f"  {test_season}: LR={ll_lr:.4f}  Blend={ll_blend:.4f}  (n={len(y_te)})")

    mean_lr = np.mean(cv_scores_lr)
    mean_blend = np.mean(cv_scores_blend)
    best_cv = min(mean_lr, mean_blend)
    print(f"\nMean CV — LR: {mean_lr:.4f}  Blend: {mean_blend:.4f}  → Using: {'LR' if mean_lr <= mean_blend else 'Blend'}")
    use_blend = mean_blend < mean_lr

    # Final model on ALL data
    X_all_lr = matchups[lr_feats].fillna(0)
    X_all_xgb = matchups[all_feats]
    y_all = matchups['Label']

    final_lr_pipe = make_lr_pipeline(C)
    final_lr_pipe.fit(X_all_lr, y_all)

    final_xgb = None
    if use_blend:
        final_xgb = make_xgb(500)
        final_xgb.fit(X_all_xgb, y_all)

    # Show LR base-feature coefficients (first 8 from the 36-feature poly expansion)
    lr_step = final_lr_pipe.named_steps['lr']
    poly_step = final_lr_pipe.named_steps['poly']
    feat_names = poly_step.get_feature_names_out(lr_feats)
    print(f"\nTop LR features by |coefficient|:")
    fi = sorted(zip(feat_names, lr_step.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for fname, coef in fi[:12]:
        print(f"  {fname}: {coef:+.4f}")

    return final_lr_pipe, final_xgb, lr_feats, all_feats, best_cv, team_feats, seed_matchup_rates


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_for_season(sub_template_path: str, season: int,
                       m_lr_pipe, m_xgb, m_lr_feats, m_all_feats, m_team_feats, m_smr,
                       w_lr_pipe, w_xgb, w_lr_feats, w_all_feats, w_team_feats, w_smr) -> pd.DataFrame:
    """Generate predictions for all matchups in a given season."""
    sub = pd.read_csv(sub_template_path)
    ids = sub.ID.str.split('_', expand=True)
    sub['Season'] = ids[0].astype(int)
    sub['T1'] = ids[1].astype(int)
    sub['T2'] = ids[2].astype(int)
    sub_s = sub[sub.Season == season]

    sub_m = sub_s[sub_s.T1 < 2000].copy()
    sub_w = sub_s[sub_s.T1 >= 3000].copy()

    def pred_g(sub_g, team_feats, lr_pipe, xgb_m, lr_feats, all_feats, smr):
        if len(sub_g) == 0:
            return pd.DataFrame(columns=['ID', 'Pred'])
        f = team_feats[team_feats.Season == season].copy()
        t1f = f.add_prefix('T1_').rename(columns={'T1_Season': 'Season', 'T1_TeamID': 'T1'})
        t2f = f.add_prefix('T2_').rename(columns={'T2_Season': 'Season', 'T2_TeamID': 'T2'})
        mg = sub_g.merge(t1f, on=['Season', 'T1'], how='left')
        mg = mg.merge(t2f, on=['Season', 'T2'], how='left')
        for feat in ALL_DIFF_FEATURES:
            c1, c2 = f'T1_{feat}', f'T2_{feat}'
            if c1 in mg.columns and c2 in mg.columns:
                mg[f'D_{feat}'] = mg[c1] - mg[c2]

        # Seed matchup rate (matchup-level feature)
        if (smr is not None and 'SeedMatchupRate' in all_feats
                and 'T1_SeedNum' in mg.columns and 'T2_SeedNum' in mg.columns):
            rate_map = smr.set_index(['SeedNum1', 'SeedNum2'])['SeedMatchupRate']
            fallback = smr['SeedMatchupRate'].mean()
            s1 = mg['T1_SeedNum'].fillna(17).clip(1, 17).astype(int)
            s2 = mg['T2_SeedNum'].fillna(17).clip(1, 17).astype(int)

            def lookup_rate(t1s, t2s):
                if t1s <= t2s:
                    return rate_map.get((t1s, t2s), fallback)
                return 1.0 - rate_map.get((t2s, t1s), 1.0 - fallback)

            mg['SeedMatchupRate'] = [lookup_rate(a, b) for a, b in zip(s1, s2)]

        X_lr = mg[lr_feats].fillna(0)
        X_xgb = mg[[f for f in all_feats if f in mg.columns]]
        p = ensemble_predict(lr_pipe, xgb_m, X_lr, X_xgb)
        return pd.DataFrame({'ID': mg['ID'].values, 'Pred': p})

    r_m = pred_g(sub_m, m_team_feats, m_lr_pipe, m_xgb, m_lr_feats, m_all_feats, m_smr)
    r_w = pred_g(sub_w, w_team_feats, w_lr_pipe, w_xgb, w_lr_feats, w_all_feats, w_smr)
    return pd.concat([r_m, r_w], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    m_lr_pipe, m_xgb, m_lr_feats, m_all_feats, m_ll, m_team_feats, m_smr = train_and_evaluate('M', 5)
    w_lr_pipe, w_xgb, w_lr_feats, w_all_feats, w_ll, w_team_feats, w_smr = train_and_evaluate('W', 5)

    print(f"\n{'='*55}")
    print(f"Final CV scores:  M={m_ll:.4f}  W={w_ll:.4f}")
    print(f"{'='*55}")

    # --- Stage 1 ---
    print("\nGenerating Stage 1 submission...")
    sub1 = pd.read_csv('data/SampleSubmissionStage1.csv')
    seasons_s1 = sorted(sub1.ID.str.split('_').str[0].astype(int).unique())
    print(f"Stage1 seasons: {seasons_s1}")

    all_preds = []
    for season in seasons_s1:
        result = predict_for_season(
            'data/SampleSubmissionStage1.csv', season,
            m_lr_pipe, m_xgb, m_lr_feats, m_all_feats, m_team_feats, m_smr,
            w_lr_pipe, w_xgb, w_lr_feats, w_all_feats, w_team_feats, w_smr,
        )
        all_preds.append(result)
        print(f"  Season {season}: {len(result)} rows")

    sub_s1 = pd.concat(all_preds).sort_values('ID').reset_index(drop=True)
    sub_s1.to_csv('submission_stage1.csv', index=False)
    print(f"Stage1 saved: {len(sub_s1)} rows → submission_stage1.csv")

    # --- Stage 2 ---
    print("\nGenerating Stage 2 submission (2026)...")
    result_s2 = predict_for_season(
        'data/SampleSubmissionStage2.csv', 2026,
        m_lr_pipe, m_xgb, m_lr_feats, m_all_feats, m_team_feats, m_smr,
        w_lr_pipe, w_xgb, w_lr_feats, w_all_feats, w_team_feats, w_smr,
    )
    result_s2 = result_s2.sort_values('ID').reset_index(drop=True)
    result_s2.to_csv('submission_stage2.csv', index=False)
    print(f"Stage2 saved: {len(result_s2)} rows → submission_stage2.csv")

    print("\nSanity checks:")
    print(f"  Pred range:  [{result_s2.Pred.min():.4f}, {result_s2.Pred.max():.4f}]")
    print(f"  Pred mean:   {result_s2.Pred.mean():.4f}  (expected ~0.5)")
    print(f"  NaN count:   {result_s2.Pred.isna().sum()}")
    print(result_s2.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
