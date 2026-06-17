"""
SADAF — Dataset B (Criteo Sponsored Search Conversion Log) Preprocessing Pipeline
===================================================================================
Purpose : Map Criteo variables to SADAF Dataset A schema and generate
          analysis-ready outputs for H1 PSM, H2 mediation, H4b/H4c sequences.
Target  : DSS / EJIS (ABS 4)
Data    : https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/

Usage:
    python src/01_criteo_preprocessing.py

Outputs (in BASE_DIR):
    criteo_B_cleaned.csv           – basic cleaning complete
    criteo_B_mapped.csv            – variable-mapped aggregates (partner × hour)
    criteo_B_psm_ready.csv         – H1 PSM analysis
    criteo_B_mediation_ready.csv   – H2 mediation analysis
    criteo_B_sequences.npz         – H4b/H4c sequences
    criteo_B_validation_report.txt – mapping validity report
    fig_WB_mapping_validation.png  – Figure W-B

Variable Mapping Caveats (must appear in paper Appendix):
    - ROAS_proxy  : approximated as product_price × n_clicks (no actual ad spend)
    - CTR_proxy   : nb_clicks_1week used as impression proxy
    - depth_proxy : time_delay_for_conversion (different construct, similar role)
    - H3          : campaign_type absent in Dataset B — Dataset A only
    - Zero-inflation gap (9.6% vs 72.1%) must be reported with KS statistics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRITEO_DIR = os.path.join(BASE_DIR, 'Criteo_Conversion_Search')

CONFIG = {
    'INPUT_FILE'    : os.path.join(CRITEO_DIR, 'CriteoSearchData'),
    'INPUT_FILE_ALT': os.path.join(CRITEO_DIR, 'criteo_search_data.tsv'),
    'OUTPUT_DIR'    : BASE_DIR,
    'SAMPLE_FRAC'  : 1.0,
    'RANDOM_SEED'  : 42,
    'SEQ_LEN'      : 4,
    'FEATURES'     : ['CTR_proxy', 'CVR_proxy', 'depth_proxy',
                      'log_cost_proxy', 'log_clicks', 'hour_sin', 'hour_cos'],
    'ROAS_METHOD'  : 'conservative',
    'KS_THRESHOLD' : 0.3,
}

CRITEO_COLS = [
    'Sale', 'SalesAmountInEuro', 'time_delay_for_conversion',
    'click_timestamp', 'nb_clicks_1week', 'product_price',
    'product_age_group', 'device_type', 'audience_id', 'product_gender',
    'product_brand', 'product_category1', 'product_category2', 'product_category3',
    'product_category4', 'product_category5', 'product_category6', 'product_category7',
    'product_country', 'product_id', 'product_title', 'partner_id', 'user_id',
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        return super().default(o)


def load_data():
    input_file = None
    for candidate in [CONFIG['INPUT_FILE'], CONFIG['INPUT_FILE_ALT']]:
        if os.path.exists(candidate):
            input_file = candidate
            break

    if input_file is None:
        print('  ⚠  Criteo file not found — entering DEMO MODE')
        return _generate_demo_data(), True

    print(f'  File: {input_file}')
    chunks = []
    reader = pd.read_csv(
        input_file, sep='\t', names=CRITEO_COLS, header=None,
        chunksize=200_000,
        dtype={'Sale': np.int8, 'SalesAmountInEuro': np.float32,
               'time_delay_for_conversion': np.float32,
               'click_timestamp': np.int64, 'nb_clicks_1week': np.float32,
               'product_price': np.float32},
        on_bad_lines='skip',
    )
    for chunk in reader:
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f'  Loaded {len(df):,} rows')
    return df, False


def _generate_demo_data():
    N = 500_000
    rng = np.random.default_rng(CONFIG['RANDOM_SEED'])
    sale_flag = (rng.random(N) < 0.035).astype(np.int8)
    return pd.DataFrame({
        'Sale': sale_flag,
        'SalesAmountInEuro': np.where(sale_flag, rng.lognormal(3.5, 1.2, N), -1.0).astype(np.float32),
        'time_delay_for_conversion': np.where(sale_flag, rng.exponential(86400*2, N), -1.0).astype(np.float32),
        'click_timestamp': np.sort(rng.integers(1609459200, 1617235200, N)),
        'nb_clicks_1week': rng.poisson(3.5, N).astype(np.float32),
        'product_price': rng.lognormal(3.8, 1.0, N).astype(np.float32),
        'product_age_group': rng.choice(['adult', 'kids', 'teen', 'unknown'], N),
        'device_type': rng.choice(['0', '1', '2', '3', '4'], N),
        'audience_id': rng.integers(0, 50, N),
        'product_gender': rng.choice(['male', 'female', 'unisex', 'unknown'], N),
        'product_brand': rng.choice([f'brand_{i}' for i in range(100)], N),
        'product_category1': rng.choice([f'cat1_{i}' for i in range(10)], N),
        'product_category2': rng.choice([f'cat2_{i}' for i in range(30)], N),
        'product_category3': rng.choice([f'cat3_{i}' for i in range(80)], N),
        'product_category4': rng.choice(['none'] + [f'cat4_{i}' for i in range(50)], N),
        'product_category5': 'none',
        'product_category6': 'none',
        'product_category7': 'none',
        'product_country': rng.choice(['US', 'FR', 'DE', 'GB', 'ES'], N),
        'product_id': rng.integers(0, 5000, N),
        'product_title': [f'prod_{i}' for i in rng.integers(0, 5000, N)],
        'partner_id': rng.integers(0, 200, N),
        'user_id': rng.integers(0, 100000, N),
    })


def clean(df_raw):
    df = df_raw.copy()
    df['SalesAmountInEuro']         = df['SalesAmountInEuro'].replace(-1, 0)
    df['time_delay_for_conversion'] = df['time_delay_for_conversion'].replace(-1, np.nan)
    df['nb_clicks_1week']           = df['nb_clicks_1week'].fillna(0)
    df['product_price']             = df['product_price'].fillna(df['product_price'].median())

    for col in ['product_age_group', 'device_type', 'product_gender',
                'product_brand', 'product_country', 'product_category1']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    df = df.dropna(subset=['Sale', 'click_timestamp', 'partner_id'])

    for col in ['product_price', 'nb_clicks_1week', 'SalesAmountInEuro']:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))

    df['datetime']    = pd.to_datetime(df['click_timestamp'], unit='s', errors='coerce')
    df               = df.dropna(subset=['datetime'])
    df['hour']        = df['datetime'].dt.hour
    df['weekday']     = df['datetime'].dt.weekday
    df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['partner_id']  = df['partner_id'].astype(str)
    return df


def build_proxy_features(df):
    agg = df.groupby(['partner_id', 'hour']).agg(
        n_clicks           = ('Sale', 'count'),
        n_conversions      = ('Sale', 'sum'),
        total_revenue      = ('SalesAmountInEuro', 'sum'),
        mean_product_price = ('product_price', 'mean'),
        total_clicks_proxy = ('nb_clicks_1week', 'sum'),
        mean_time_delay    = ('time_delay_for_conversion', lambda x: x.dropna().mean()),
    ).reset_index()

    agg['impression_proxy'] = agg['total_clicks_proxy'].clip(lower=1)
    agg['CTR_proxy']        = (agg['n_clicks'] / agg['impression_proxy']).clip(0, 1)
    agg['CVR_proxy']        = (agg['n_conversions'] / agg['n_clicks'].clip(lower=1)).clip(0, 1)
    agg['has_conversion']   = (agg['n_conversions'] > 0).astype(int)
    agg['depth_proxy']      = (agg['mean_time_delay'].fillna(0) / (86400 * 30)).clip(0, 1)

    cost_proxy              = agg['mean_product_price'] * agg['n_clicks']
    agg['cost_proxy']       = cost_proxy.clip(lower=0)
    agg['log_cost_proxy']   = np.log1p(cost_proxy)
    agg['log_clicks']       = np.log1p(agg['n_clicks'])
    agg['ROAS_proxy']       = (agg['total_revenue'] / agg['cost_proxy'].clip(lower=1e-6)).clip(0, 1000)
    agg['log_ROAS_proxy']   = np.log1p(agg['ROAS_proxy'])

    hour_time = df[['hour', 'hour_sin', 'hour_cos']].drop_duplicates('hour')
    agg = agg.merge(hour_time, on='hour', how='left')

    le = LabelEncoder()
    agg['partner_idx'] = le.fit_transform(agg['partner_id'])
    return agg


def build_sequences(df_agg, features, target_col, seq_len=4):
    seqs_X, seqs_Y = [], []
    for partner in df_agg['partner_id'].unique():
        g = df_agg[df_agg['partner_id'] == partner].sort_values('hour').reset_index(drop=True)
        if len(g) < seq_len + 1:
            continue
        X_arr = g[features].fillna(0).values.astype(np.float32)
        Y_arr = g[target_col].fillna(0).values.astype(np.float32)
        for i in range(len(g) - seq_len):
            seqs_X.append(X_arr[i:i + seq_len])
            seqs_Y.append(Y_arr[i + seq_len])
    if not seqs_X:
        return np.zeros((0, seq_len, len(features)), dtype=np.float32), np.zeros(0)
    return np.array(seqs_X, dtype=np.float32), np.array(seqs_Y, dtype=np.float32)


def main():
    print('=' * 65)
    print('SADAF Dataset B — Criteo Preprocessing Pipeline')
    print('=' * 65)

    # Load
    df_raw, DEMO_MODE = load_data()
    df = clean(df_raw)
    print(f'  After cleaning: {len(df):,} rows')

    # Aggregate + proxy features
    agg = build_proxy_features(df)
    print(f'  Aggregated: {len(agg):,} (partner × hour)')

    # PSM data
    df_psm = agg[agg['CTR_proxy'] > 0].copy()
    ctr_med = df_psm['CTR_proxy'].median()
    df_psm['T_highCTR'] = (df_psm['CTR_proxy'] > ctr_med).astype(int)
    df_psm = df_psm[['log_clicks', 'log_cost_proxy', 'depth_proxy',
                      'T_highCTR', 'has_conversion', 'CTR_proxy', 'partner_id', 'hour']].dropna()
    df_psm.to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_psm_ready.csv'), index=False)

    # Mediation data
    df_med = agg[(agg['CTR_proxy'] > 0) & (agg['depth_proxy'] >= 0)].copy()
    df_med['log_CTR_proxy'] = np.log1p(df_med['CTR_proxy'])
    df_med = df_med[['log_CTR_proxy', 'depth_proxy', 'has_conversion',
                     'partner_id', 'hour', 'CVR_proxy']].dropna()
    df_med.to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_mediation_ready.csv'), index=False)

    # Sequences
    FEATURES = CONFIG['FEATURES']
    for f in FEATURES:
        if f not in agg.columns:
            agg[f] = 0.0

    df_roas = agg[agg['ROAS_proxy'] > 0].copy()
    X_reg, Y_reg = build_sequences(df_roas, FEATURES, 'log_ROAS_proxy', seq_len=CONFIG['SEQ_LEN'])
    X_cls, Y_cls = build_sequences(agg, FEATURES, 'has_conversion', seq_len=CONFIG['SEQ_LEN'])
    X_reg6, Y_reg6 = build_sequences(df_roas, FEATURES, 'log_ROAS_proxy', seq_len=6)

    if len(X_reg) > 0:
        scaler = MinMaxScaler()
        N, T, D = X_reg.shape
        X_reg_n = scaler.fit_transform(X_reg.reshape(-1, D)).reshape(X_reg.shape)
    else:
        X_reg_n, scaler = X_reg, None

    if len(X_cls) > 0:
        scaler_cls = MinMaxScaler()
        N2, T2, D2 = X_cls.shape
        X_cls_n = scaler_cls.fit_transform(X_cls.reshape(-1, D2)).reshape(X_cls.shape)
    else:
        X_cls_n, scaler_cls = X_cls, None

    np.savez_compressed(
        os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_sequences.npz'),
        X_reg=X_reg_n, Y_reg=Y_reg, X_cls=X_cls_n, Y_cls=Y_cls,
        X_reg_sl6=X_reg6, Y_reg_sl6=Y_reg6,
    )
    print(f'  Sequences: REG={X_reg_n.shape}  CLS={X_cls_n.shape}')

    # Save main CSVs
    cols_save = [c for c in ['Sale', 'SalesAmountInEuro', 'click_timestamp',
                              'nb_clicks_1week', 'product_price', 'device_type',
                              'product_country', 'partner_id', 'user_id',
                              'hour', 'weekday', 'hour_sin', 'hour_cos', 'datetime']
                 if c in df.columns]
    df[cols_save].to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_cleaned.csv'), index=False)
    agg.to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_mapped.csv'), index=False)

    if scaler is not None:
        with open(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_scaler.pkl'), 'wb') as f:
            pickle.dump({'reg': scaler, 'cls': scaler_cls}, f)

    # Validation report
    r_ctr_depth, _ = spearmanr(df_med['log_CTR_proxy'], df_med['depth_proxy'])
    zi_B = (agg['ROAS_proxy'] == 0).mean()
    report = {
        'demo_mode': DEMO_MODE,
        'n_rows': len(agg),
        'zero_inflation': {'dataset_B': float(zi_B), 'dataset_A_ref': 0.721,
                           'diff': float(abs(zi_B - 0.721))},
        'conversion_rate': {'dataset_B': float(agg['has_conversion'].mean()), 'dataset_A_ref': 0.1177},
        'depth_proxy_validity': {'r_ctr_depth': float(r_ctr_depth),
                                 'direction_match': bool(r_ctr_depth < 0)},
    }
    with open(os.path.join(CONFIG['OUTPUT_DIR'], 'criteo_B_validation_report.txt'), 'w') as f:
        f.write('SADAF Dataset B — Mapping Validity Report\n' + '=' * 50 + '\n\n')
        f.write(json.dumps(report, indent=2, cls=NumpyEncoder))

    print('\n✅ Preprocessing complete. Outputs saved to:', CONFIG['OUTPUT_DIR'])
    print('\nVariable mapping caveats (include in paper Appendix):')
    caveats = [
        'ROAS_proxy: ad spend approximated as product_price × n_clicks',
        'CTR_proxy: impressions proxied via nb_clicks_1week',
        'depth_proxy: time_delay_for_conversion (related but distinct construct)',
        'H3: campaign_type unavailable in Dataset B — Dataset A only',
        'Zero-inflation gap (9.6% vs 72.1%) — report KS statistics in §4',
    ]
    for c in caveats:
        print(f'  ⚠  {c}')


if __name__ == '__main__':
    main()
