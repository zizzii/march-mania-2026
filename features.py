"""
Feature engineering for March Mania 2026.
Computes per-team per-season features:
  - Elo ratings with margin-of-victory adjustment
  - Offensive / defensive efficiency (from detailed results) + season-normalized z-scores
  - Win rate, average margin
  - Recent form (last 15 regular season games)
  - Strength of schedule (average opponent end-of-season Elo)
  - Conference tournament wins
  - Massey consensus ranking (mean/std across all systems) + key individual systems
  - Seed number (for seasons where seeds are known)
"""

import pandas as pd
import numpy as np

# Poll systems that only rank ~25 teams (top-25 voter polls).
# Unranked teams get the worst percentile, which corrupts MasseyMean for ~340 teams.
POLL_SYSTEMS = frozenset({'AP', 'USA', 'DES'})


# ---------------------------------------------------------------------------
# ELO (margin-of-victory adjusted)
# ---------------------------------------------------------------------------

def compute_elo(results_df: pd.DataFrame,
                k: float = 20.0,
                initial: float = 1500.0,
                carryover: float = 0.75) -> pd.DataFrame:
    """
    Compute end-of-regular-season Elo for each team/season.
    Uses margin-of-victory multiplier (Oliver/Hollinger formula).

    Parameters
    ----------
    results_df : DataFrame with columns Season, DayNum, WTeamID, WScore, LTeamID, LScore
    k          : base K-factor
    initial    : starting Elo for new teams
    carryover  : fraction of prior-season Elo deviation carried into new season

    Returns
    -------
    DataFrame with columns (Season, TeamID, Elo)
    """
    results = results_df.sort_values(['Season', 'DayNum']).copy()
    seasons = sorted(results.Season.unique())

    elo: dict[int, float] = {}
    records = []

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for season in seasons:
        # Apply carryover at start of each season
        for tid in list(elo.keys()):
            elo[tid] = initial + carryover * (elo[tid] - initial)

        for _, row in results[results.Season == season].iterrows():
            w, l = int(row.WTeamID), int(row.LTeamID)
            ew = elo.get(w, initial)
            el = elo.get(l, initial)

            elo_diff = abs(ew - el)
            exp_w = expected(ew, el)

            # Margin-of-victory multiplier
            margin = abs(row.WScore - row.LScore)
            mov_mult = np.log(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
            k_adj = k * mov_mult

            elo[w] = ew + k_adj * (1 - exp_w)
            elo[l] = el + k_adj * (0 - (1 - exp_w))

        for tid, e in elo.items():
            records.append({'Season': season, 'TeamID': tid, 'Elo': e})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# EFFICIENCY STATS
# ---------------------------------------------------------------------------

def compute_efficiency(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team per-season offensive/defensive efficiency, Four Factors,
    and neutral-site net rating.

    Possessions ≈ FGA - OReb + TO + 0.44*FTA  (Oliver formula)
    OffRtg = 100 * Points / Possessions
    DefRtg = 100 * OpponentPoints / OpponentPossessions
    NetRtg = OffRtg - DefRtg

    Four Factors (Dean Oliver):
      eFG%   = (FGM + 0.5*FGM3) / FGA       — shooting quality
      TOV%   = TO / (FGA + 0.44*FTA + TO)   — ball security
      ORB%   = OR / (OR + OppDR)            — second-chance rate
      FTR    = FTA / FGA                    — free throw rate

    NeutralNetRtg_z: net rating in non-home games only (purer tournament proxy).
    """
    df = detailed_df.copy()
    has_wloc = 'WLoc' in df.columns

    def poss(fga, or_, to, fta):
        return fga - or_ + to + 0.44 * fta

    df['WPoss'] = poss(df.WFGA, df.WOR, df.WTO, df.WFTA)
    df['LPoss'] = poss(df.LFGA, df.LOR, df.LTO, df.LFTA)

    # Winners — include opponent DR (LDR) for ORB%
    w = df[['Season', 'WTeamID', 'WScore', 'LScore', 'WPoss', 'LPoss',
            'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
            'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'LDR']].copy()
    w.columns = ['Season', 'TeamID', 'Pts', 'OppPts', 'Poss', 'OppPoss',
                 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'OppDR']
    w['Win'] = 1

    # Losers — include opponent DR (WDR) for ORB%
    l = df[['Season', 'LTeamID', 'LScore', 'WScore', 'LPoss', 'WPoss',
            'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'WDR']].copy()
    l.columns = ['Season', 'TeamID', 'Pts', 'OppPts', 'Poss', 'OppPoss',
                 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'OppDR']
    l['Win'] = 0

    # Home-court flag (WLoc='H' → winner at home; WLoc='A' → loser at home)
    if has_wloc:
        w['IsHome'] = (df['WLoc'] == 'H').values
        l['IsHome'] = (df['WLoc'] == 'A').values

    games = pd.concat([w, l], ignore_index=True)
    games['Poss'] = games['Poss'].clip(lower=1)
    games['OppPoss'] = games['OppPoss'].clip(lower=1)
    games['OffRtg'] = 100 * games['Pts'] / games['Poss']
    games['DefRtg'] = 100 * games['OppPts'] / games['OppPoss']
    games['FGPct'] = games['FGM'] / games['FGA'].clip(lower=1)
    games['FG3Pct'] = games['FGM3'] / games['FGA3'].clip(lower=1)
    games['FTPct'] = games['FTM'] / games['FTA'].clip(lower=1)
    games['Margin'] = games['Pts'] - games['OppPts']

    # Four Factors
    games['eFG'] = (games['FGM'] + 0.5 * games['FGM3']) / games['FGA'].clip(lower=1)
    games['TOVPct'] = games['TO'] / (games['FGA'] + 0.44 * games['FTA'] + games['TO']).clip(lower=1)
    games['FTR'] = games['FTA'] / games['FGA'].clip(lower=1)
    games['ORBPct'] = games['OR'] / (games['OR'] + games['OppDR']).clip(lower=1)

    agg = games.groupby(['Season', 'TeamID']).agg(
        Games=('Win', 'count'),
        WinRate=('Win', 'mean'),
        OffRtg=('OffRtg', 'mean'),
        DefRtg=('DefRtg', 'mean'),
        Tempo=('Poss', 'mean'),
        FGPct=('FGPct', 'mean'),
        FG3Pct=('FG3Pct', 'mean'),
        FTPct=('FTPct', 'mean'),
        OR=('OR', 'mean'),
        DR=('DR', 'mean'),
        Ast=('Ast', 'mean'),
        TO=('TO', 'mean'),
        Stl=('Stl', 'mean'),
        Blk=('Blk', 'mean'),
        AvgMargin=('Margin', 'mean'),
        eFG=('eFG', 'mean'),
        TOVPct=('TOVPct', 'mean'),
        FTR=('FTR', 'mean'),
        ORBPct=('ORBPct', 'mean'),
    ).reset_index()
    agg['NetRtg'] = agg['OffRtg'] - agg['DefRtg']

    # Season-normalized z-scores to remove scoring era effects
    for col in ['OffRtg', 'DefRtg', 'NetRtg']:
        season_mean = agg.groupby('Season')[col].transform('mean')
        season_std = agg.groupby('Season')[col].transform('std').clip(lower=0.1)
        agg[f'{col}_z'] = (agg[col] - season_mean) / season_std

    # Neutral-site (non-home) net rating — purer tournament proxy
    if has_wloc:
        neutral = games[~games['IsHome']].copy()
        if len(neutral) > 0:
            nagg = neutral.groupby(['Season', 'TeamID']).agg(
                NeutralOffRtg=('OffRtg', 'mean'),
                NeutralDefRtg=('DefRtg', 'mean'),
            ).reset_index()
            nagg['NeutralNetRtg'] = nagg['NeutralOffRtg'] - nagg['NeutralDefRtg']
            n_mean = nagg.groupby('Season')['NeutralNetRtg'].transform('mean')
            n_std = nagg.groupby('Season')['NeutralNetRtg'].transform('std').clip(lower=0.1)
            nagg['NeutralNetRtg_z'] = (nagg['NeutralNetRtg'] - n_mean) / n_std
            agg = agg.merge(nagg[['Season', 'TeamID', 'NeutralNetRtg_z']],
                            on=['Season', 'TeamID'], how='left')

    return agg


def compute_winrate_compact(compact_df: pd.DataFrame) -> pd.DataFrame:
    """Win rate and average margin from compact results (fallback for seasons without detailed data)."""
    df = compact_df.copy()
    df['Margin'] = df.WScore - df.LScore
    w = df[['Season', 'WTeamID', 'Margin']].rename(columns={'WTeamID': 'TeamID'})
    w['Win'] = 1
    l = df[['Season', 'LTeamID', 'Margin']].rename(columns={'LTeamID': 'TeamID'})
    l['Win'] = 0
    l['Margin'] = -l['Margin']
    games = pd.concat([w, l], ignore_index=True)
    agg = games.groupby(['Season', 'TeamID']).agg(
        WinRate=('Win', 'mean'),
        AvgMargin=('Margin', 'mean'),
        Games=('Win', 'count'),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# RECENT FORM (last N games)
# ---------------------------------------------------------------------------

def compute_recent_form(compact_df: pd.DataFrame, n_games: int = 15) -> pd.DataFrame:
    """
    Win rate and margin in the last n_games of regular season, with exponential
    decay weights (half-life ~5 games) so most recent games matter more.
    """
    df = compact_df.copy()
    df['Margin'] = df.WScore - df.LScore

    w = df[['Season', 'DayNum', 'WTeamID', 'Margin']].copy()
    w.columns = ['Season', 'DayNum', 'TeamID', 'Margin']
    w['Win'] = 1.0

    l = df[['Season', 'DayNum', 'LTeamID', 'Margin']].copy()
    l.columns = ['Season', 'DayNum', 'TeamID', 'Margin']
    l['Win'] = 0.0
    l['Margin'] = -l['Margin']

    games = pd.concat([w, l], ignore_index=True)
    games = games.sort_values(['Season', 'TeamID', 'DayNum'])

    # Keep last n_games per team per season (sorted oldest → newest)
    recent = games.groupby(['Season', 'TeamID']).tail(n_games).copy()

    def ewm_agg(group):
        n = len(group)
        # Oldest game gets exp(-0.14*(n-1)), most recent gets exp(0)=1
        weights = np.exp(-0.14 * np.arange(n - 1, -1, -1))
        weights /= weights.sum()
        return pd.Series({
            'RecentWinRate': float(np.dot(group['Win'].values, weights)),
            'RecentMargin': float(np.dot(group['Margin'].values, weights)),
        })

    result = recent.groupby(['Season', 'TeamID']).apply(ewm_agg).reset_index()
    return result


# ---------------------------------------------------------------------------
# MASSEY RANKINGS (consensus across all systems + key individual systems)
# ---------------------------------------------------------------------------

def compute_massey_features(massey_df: pd.DataFrame,
                            key_systems: tuple = ('KPK', 'POM', 'NET', 'MOR'),
                            max_day: int = 133,
                            exclude_systems: frozenset = POLL_SYSTEMS) -> pd.DataFrame:
    """
    Compute ranking features:
    1. Individual ranks for key_systems (lower = better)
    2. Consensus features across ALL available systems:
       - MasseyMean: mean percentile rank (0=best, 1=worst) across all systems
       - MasseyStd: std of percentile ranks (low = high inter-system agreement)
       - MasseyMin: best single-system percentile (how good is team at its best ranking)

    Parameters
    ----------
    massey_df : MMasseyOrdinals DataFrame (already filtered to M or W teams)
    key_systems : individual systems to keep as separate features
    max_day : only consider rankings through this day (pre-tournament cutoff)
    """
    df = massey_df[massey_df.RankingDayNum <= max_day].copy()
    # Remove voting polls — they only rank ~25 teams, giving all others the worst percentile
    if exclude_systems:
        df = df[~df.SystemName.isin(exclude_systems)]

    # For each system, take the latest ranking available before max_day
    idx = df.groupby(['Season', 'TeamID', 'SystemName'])['RankingDayNum'].idxmax()
    latest = df.loc[idx].copy()

    # --- Consensus features across ALL systems ---
    # Compute within-season percentile rank for each system
    # (rank / n_teams in that system → 0=best, 1=worst)
    latest['Pct'] = (latest.groupby(['Season', 'SystemName'])['OrdinalRank']
                           .rank(pct=True))

    consensus = (latest.groupby(['Season', 'TeamID'])['Pct']
                       .agg(MasseyMean='mean', MasseyStd='std', MasseyMin='min')
                       .reset_index())
    consensus['MasseyStd'] = consensus['MasseyStd'].fillna(0)

    # --- Individual key systems ---
    key_df = latest[latest.SystemName.isin(key_systems)]
    if len(key_df) > 0:
        pivot = key_df.pivot_table(index=['Season', 'TeamID'],
                                   columns='SystemName',
                                   values='OrdinalRank').reset_index()
        pivot.columns.name = None
        rename = {s: f'Rank_{s}' for s in key_systems if s in pivot.columns}
        pivot = pivot.rename(columns=rename)
        result = consensus.merge(pivot, on=['Season', 'TeamID'], how='left')
    else:
        result = consensus

    return result


# ---------------------------------------------------------------------------
# STRENGTH OF SCHEDULE
# ---------------------------------------------------------------------------

def compute_sos(compact_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strength of schedule: average end-of-season Elo of opponents faced.
    Uses the final end-of-season Elo (already computed) as opponent quality proxy.
    """
    df = compact_df.copy()

    w = df[['Season', 'WTeamID', 'LTeamID']].rename(
        columns={'WTeamID': 'TeamID', 'LTeamID': 'OppID'})
    l = df[['Season', 'LTeamID', 'WTeamID']].rename(
        columns={'LTeamID': 'TeamID', 'WTeamID': 'OppID'})
    games = pd.concat([w, l], ignore_index=True)

    # Join end-of-season Elo for opponents
    opp_elo = elo_df[['Season', 'TeamID', 'Elo']].rename(
        columns={'TeamID': 'OppID', 'Elo': 'OppElo'})
    games = games.merge(opp_elo, on=['Season', 'OppID'], how='left')

    sos = (games.groupby(['Season', 'TeamID'])['OppElo']
                .mean()
                .reset_index()
                .rename(columns={'OppElo': 'SOS'}))
    return sos


# ---------------------------------------------------------------------------
# CONFERENCE TOURNAMENT PERFORMANCE
# ---------------------------------------------------------------------------

def compute_conf_tourney(conf_tourney_df: pd.DataFrame) -> pd.DataFrame:
    """
    Conference tournament wins per team per season.
    Winning your conference tournament signals peak form and earns an automatic bid.
    """
    df = conf_tourney_df.copy()
    wins = (df.groupby(['Season', 'WTeamID'])
              .size()
              .reset_index(name='ConfTourneyWins')
              .rename(columns={'WTeamID': 'TeamID'}))
    return wins


# ---------------------------------------------------------------------------
# SEEDS
# ---------------------------------------------------------------------------

def parse_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric seed (1-16) from seed string like 'W01', 'X11a'."""
    seeds = seeds_df.copy()
    seeds['SeedNum'] = seeds.Seed.str.extract(r'(\d+)').astype(float)
    return seeds[['Season', 'TeamID', 'SeedNum']]


# ---------------------------------------------------------------------------
# SEED IMPUTATION (for seasons where real seeds are unknown, e.g. 2026)
# ---------------------------------------------------------------------------

def impute_seeds(base_df: pd.DataFrame, gender: str,
                 massey_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For rows where SeedNum is NaN, estimate seed from within-season ranking.

    Men: use MasseyMean percentile rank (available from MMasseyOrdinals).
    Women: use Elo rank (no women-specific Massey for all systems historically).

    Assignment rule: sort teams by quality (ascending MasseyMean or descending Elo),
    assign position 1-4 → seed 1, 5-8 → seed 2, ..., 57-64 → seed 16, 65+ → 17.
    17 signals "likely not in tournament" and makes the seed feature work for all matchups.
    """
    df = base_df.copy()
    nan_mask = df['SeedNum'].isna()
    if not nan_mask.any():
        return df

    nan_seasons = df.loc[nan_mask, 'Season'].unique()

    for season in nan_seasons:
        season_mask = (df['Season'] == season) & nan_mask

        if gender == 'M' and 'MasseyMean' in df.columns:
            # Lower MasseyMean percentile = better team
            quality = df.loc[df['Season'] == season, 'MasseyMean']
            rank_within_season = quality.rank(method='first', ascending=True)
        else:
            # Higher Elo = better team; invert for rank
            quality = df.loc[df['Season'] == season, 'Elo']
            rank_within_season = quality.rank(method='first', ascending=False)

        # Seed 1-16 for top 64 teams (4 per seed); 17 for rest (not expected in tournament)
        imputed = np.where(
            rank_within_season <= 64,
            np.ceil(rank_within_season / 4).clip(upper=16),
            17.0
        )
        df.loc[df['Season'] == season, '_ImputedSeed'] = imputed
        df.loc[season_mask, 'SeedNum'] = df.loc[season_mask, '_ImputedSeed']

    if '_ImputedSeed' in df.columns:
        df = df.drop(columns=['_ImputedSeed'])
    return df


# ---------------------------------------------------------------------------
# SEED MATCHUP HISTORICAL WIN RATES
# ---------------------------------------------------------------------------

def compute_seed_matchup_rates(tourney_df: pd.DataFrame,
                                seeds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lookup table of historical P(lower-seed-num team wins) for each
    seed pairing (e.g., seed 1 vs 16 → 0.976, seed 8 vs 9 → 0.486).

    Returns DataFrame with (SeedNum1, SeedNum2, SeedMatchupRate, SeedMatchupN)
    where SeedNum1 <= SeedNum2 (lower number = better seed).
    """
    seeds = parse_seeds(seeds_df)[['Season', 'TeamID', 'SeedNum']]

    games = tourney_df.copy()
    games['T1'] = games[['WTeamID', 'LTeamID']].min(axis=1)
    games['T2'] = games[['WTeamID', 'LTeamID']].max(axis=1)
    games['T1_Won'] = (games['WTeamID'] == games['T1']).astype(int)

    games = games.merge(
        seeds.rename(columns={'TeamID': 'T1', 'SeedNum': 'SeedNum1'}),
        on=['Season', 'T1'], how='left',
    )
    games = games.merge(
        seeds.rename(columns={'TeamID': 'T2', 'SeedNum': 'SeedNum2'}),
        on=['Season', 'T2'], how='left',
    )

    # Drop games without known seeds (play-in or missing data)
    games = games.dropna(subset=['SeedNum1', 'SeedNum2'])

    # Normalize: SeedNum1 <= SeedNum2 (better seed in position 1)
    # Flip direction of T1_Won when needed
    needs_swap = games['SeedNum1'] > games['SeedNum2']
    s1 = games['SeedNum1'].copy()
    games.loc[needs_swap, 'SeedNum1'] = games.loc[needs_swap, 'SeedNum2']
    games.loc[needs_swap, 'SeedNum2'] = s1[needs_swap]
    games.loc[needs_swap, 'T1_Won'] = 1 - games.loc[needs_swap, 'T1_Won']

    games['SeedNum1'] = games['SeedNum1'].astype(int)
    games['SeedNum2'] = games['SeedNum2'].astype(int)

    rates = (games.groupby(['SeedNum1', 'SeedNum2'])['T1_Won']
                  .agg(SeedMatchupRate='mean', SeedMatchupN='count')
                  .reset_index())
    return rates


# ---------------------------------------------------------------------------
# MASTER FEATURE TABLE
# ---------------------------------------------------------------------------

def build_team_features(gender: str = 'M') -> pd.DataFrame:
    """
    Build a master feature table indexed by (Season, TeamID) for the given gender.

    gender: 'M' or 'W'
    """
    prefix = gender

    compact = pd.read_csv(f'data/{prefix}RegularSeasonCompactResults.csv')
    seeds_raw = pd.read_csv(f'data/{prefix}NCAATourneySeeds.csv')

    # Elo (MOV-adjusted)
    elo_df = compute_elo(compact)

    # Win rate from compact (covers all seasons)
    wr_compact = compute_winrate_compact(compact)

    # Recent form
    recent_df = compute_recent_form(compact)

    # Strength of schedule
    sos_df = compute_sos(compact, elo_df)

    # Efficiency from detailed (2003+ for M, 2010+ for W)
    det_path = f'data/{prefix}RegularSeasonDetailedResults.csv'
    try:
        detailed = pd.read_csv(det_path)
        eff = compute_efficiency(detailed)
    except Exception:
        eff = pd.DataFrame()

    # Conference tournament performance (not available for 2026 yet — will be NaN)
    conf_path = f'data/{prefix}ConferenceTourneyGames.csv'
    try:
        conf_df = pd.read_csv(conf_path)
        conf_wins = compute_conf_tourney(conf_df)
    except Exception:
        conf_wins = pd.DataFrame(columns=['Season', 'TeamID', 'ConfTourneyWins'])

    # Massey rankings (only men's teams in MMasseyOrdinals)
    massey = pd.read_csv('data/MMasseyOrdinals.csv')
    if gender == 'W':
        massey_filtered = massey[massey.TeamID >= 3000]
    else:
        massey_filtered = massey[massey.TeamID < 3000]
    rank_df = compute_massey_features(massey_filtered)

    # Seeds
    seed_df = parse_seeds(seeds_raw)

    # Merge everything
    base = elo_df.merge(
        wr_compact[['Season', 'TeamID', 'WinRate', 'AvgMargin', 'Games']],
        on=['Season', 'TeamID'], how='left'
    )
    base = base.merge(recent_df, on=['Season', 'TeamID'], how='left')
    base = base.merge(sos_df, on=['Season', 'TeamID'], how='left')
    if not conf_wins.empty:
        base = base.merge(conf_wins, on=['Season', 'TeamID'], how='left')

    if not eff.empty:
        eff_cols = ['Season', 'TeamID', 'OffRtg', 'DefRtg', 'NetRtg', 'Tempo',
                    'FGPct', 'FG3Pct', 'FTPct', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk',
                    'OffRtg_z', 'DefRtg_z', 'NetRtg_z',
                    'eFG', 'TOVPct', 'FTR', 'ORBPct', 'NeutralNetRtg_z']
        base = base.merge(eff[[c for c in eff_cols if c in eff.columns]],
                          on=['Season', 'TeamID'], how='left')

    base = base.merge(rank_df, on=['Season', 'TeamID'], how='left')
    base = base.merge(seed_df, on=['Season', 'TeamID'], how='left')

    # For seasons without seeds (e.g. 2026), impute from ranking
    base = impute_seeds(base, gender)

    return base


if __name__ == '__main__':
    print("Building men's features...")
    m_feats = build_team_features('M')
    print(f"  Shape: {m_feats.shape}")
    print(f"  Seasons: {m_feats.Season.min()} - {m_feats.Season.max()}")
    print(f"  Cols: {list(m_feats.columns)}")
    print(m_feats[m_feats.Season == 2026].tail(3))

    print("\nBuilding women's features...")
    w_feats = build_team_features('W')
    print(f"  Shape: {w_feats.shape}")
    print(f"  Cols: {list(w_feats.columns)}")
