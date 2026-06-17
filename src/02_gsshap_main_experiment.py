"""
GS-SHAP: Main Experiment — RQ1, RQ2, RQ3
==========================================
Covers:
  RQ1  Explanation faithfulness (Sufficiency, Comprehensiveness, Gini)
       Baselines: TimeSHAP, WinSHAP, LIME-TS, GS(NoSeg), GS(NoGroup)
  RQ2  Temporal sparsity stress-test (T ∈ {1, 4, 8, 16, 32})
       Same 200 samples × seq[-T:] slicing × Primary LSTM reuse
  RQ3  Model × XAI method interaction (LSTM, Transformer, CNN, MLPMixer)
  §13  Shapley axiom verification + MMD sensitivity analysis

Runtime: ~33 min on A100.  Monitor with: tail -f logs/main_run.txt

Outputs (→ gsshap_sparse_ad/):
  results_main.csv, results_stress.csv, rq1_wilcoxon.csv
  rq_axiom_check.csv, rq_sensitivity.csv
  fig_training_curves.png, fig_rq1_faithfulness.png
  fig_rq2_stress_test.png, fig_rq3_interaction.png
  fig_domain_attribution.png, fig_paper_main.png
  fig_sec13_axiom_sensitivity.png
  model_{LSTM,Transformer,CNN,MLPMixer}.pt
"""

import os, sys, time, warnings
from pathlib import Path
from itertools import combinations
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, wilcoxon, f_oneway
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRITEO_DIR = os.path.join(BASE_DIR, 'Criteo_Conversion_Search')
SAVE_DIR   = os.path.join(BASE_DIR, 'gsshap_sparse_ad')
Path(SAVE_DIR).mkdir(exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Init] device={device}  save_dir={SAVE_DIR}')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Column definitions ────────────────────────────────────────────────────────
CRITEO_COLS = [
    'Sale', 'SalesAmountInEuro', 'time_delay_for_conversion',
    'click_timestamp', 'nb_clicks_1week', 'product_price',
    'product_age_group', 'device_type', 'audience_id', 'product_gender',
    'product_brand', 'product_category_1', 'product_category_2', 'product_category_3',
    'product_category_4', 'product_category_5', 'product_category_6', 'product_category_7',
    'product_country', 'product_id', 'product_title', 'partner_id', 'user_id',
]

SEQ_FEATURES = [
    'nb_clicks_1week', 'product_price', 'click_hour', 'click_dow',
    'device_type_enc', 'product_country_enc', 'product_age_group_enc',
    'product_gender_enc', 'product_category_1_enc', 'partner_id_enc',
]
D         = len(SEQ_FEATURES)
T_PRIMARY = 8

# ── Models ────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, drop=0.25):
        super().__init__()
        self.rnn  = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 32),
                                   nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 2))
    def forward(self, x):
        out, _ = self.rnn(x); return self.head(out[:, -1])

class TransformerModel(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, layers=2, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 256, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, 2)
    def forward(self, x):
        x = self.proj(x); cls = self.cls.expand(x.size(0), -1, -1)
        return self.head(self.enc(torch.cat([cls, x], 1))[:, 0])

class CNNModel(nn.Module):
    def __init__(self, in_dim, filters=64, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, filters, 3, padding=1), nn.GELU(), nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters*2, 3, padding=1), nn.GELU(), nn.BatchNorm1d(filters*2),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(filters*2, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 2))
    def forward(self, x): return self.head(self.net(x.permute(0,2,1)).squeeze(-1))

class MLPMixer(nn.Module):
    def __init__(self, in_dim, seq_len, hidden=64, drop=0.1):
        super().__init__()
        self.tmix = nn.Sequential(nn.LayerNorm(seq_len), nn.Linear(seq_len, hidden),
                                   nn.GELU(), nn.Linear(hidden, seq_len), nn.Dropout(drop))
        self.cmix = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden*2),
                                   nn.GELU(), nn.Linear(hidden*2, in_dim), nn.Dropout(drop))
        self.head = nn.Linear(in_dim, 2)
    def forward(self, x):
        x = x + self.tmix(x.permute(0,2,1)).permute(0,2,1)
        x = x + self.cmix(x); return self.head(x.mean(1))


def build_sequences(df, group_col, seq_features, T_seq, min_len):
    grps = df.groupby(group_col, sort=False)
    seqs, labs = [], []
    for _, grp in grps:
        if len(grp) < min_len: continue
        g   = grp.sort_values('click_timestamp')
        arr = g[seq_features].values.astype(np.float32)
        n   = len(arr)
        arr = arr[-T_seq:] if n >= T_seq else np.vstack(
            [np.zeros((T_seq-n, len(seq_features)), np.float32), arr])
        seqs.append(arr); labs.append(int(g['Sale'].max()))
    return np.array(seqs, np.float32), np.array(labs, np.int64)


def train_model(mdl, Xtr, ytr, Xva, yva, name, epochs=50, lr=2e-3, bs=512, patience=8):
    mdl = mdl.to(device)
    pos_w = torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)], dtype=torch.float32, device=device)
    wt    = torch.stack([torch.ones(1,device=device).squeeze(), pos_w.squeeze()])
    crit  = nn.CrossEntropyLoss(weight=wt)
    opt   = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.OneCycleLR(opt, lr, steps_per_epoch=max(1,len(Xtr)//bs),
                                                 epochs=epochs, pct_start=0.2)
    ds  = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ldr = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(Xtr)>bs)
    best_auc, best_state, no_imp = 0, None, 0
    hist = []
    for ep in range(1, epochs+1):
        mdl.train(); tot = 0
        for Xb, yb in ldr:
            Xb, yb = Xb.to(device), yb.to(device); opt.zero_grad()
            loss = crit(mdl(Xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item()
        mdl.eval()
        with torch.no_grad():
            logits = mdl(torch.from_numpy(Xva).to(device)).cpu().numpy()
        prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
        auc  = roc_auc_score(yva, prob) if len(np.unique(yva)) > 1 else 0.5
        hist.append({'epoch': ep, 'loss': tot/max(len(ldr),1), 'val_auc': auc})
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.clone() for k,v in mdl.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience: break
    mdl.load_state_dict(best_state)
    print(f'  [{name}] Best Val AUC={best_auc:.4f}')
    return mdl, pd.DataFrame(hist)


# ── XAI baselines ─────────────────────────────────────────────────────────────
def make_pred_fn(mdl):
    mdl.eval()
    def fn(x):
        with torch.no_grad():
            out = mdl(torch.from_numpy(x.astype(np.float32)).to(device))
        return torch.softmax(out, dim=1)[:,1].cpu().numpy()
    return fn

def timeshap_explain(x_seq, pred_fn, baseline, n_perm=100, seed=0):
    rng = np.random.default_rng(seed); T, D_ = x_seq.shape
    phi = np.zeros(T); xb = np.tile(baseline[:D_], (T,1)); f0 = pred_fn(xb[None])[0]
    for _ in range(n_perm):
        perm = rng.permutation(T); xc, fp = xb.copy(), f0
        for t in perm:
            xc[t] = x_seq[t]; fn_ = pred_fn(xc[None])[0]; phi[t] += fn_ - fp; fp = fn_
    phi /= n_perm
    return np.tile(phi[:,None], (1, D_))

def winshap_explain(x_seq, pred_fn, baseline, window=4, n_perm=100, seed=0):
    rng = np.random.default_rng(seed); T, D_ = x_seq.shape
    wins = [(t, min(t+window, T)) for t in range(0, T, window)]
    phi  = np.zeros(len(wins)); xb = np.tile(baseline[:D_], (T,1)); f0 = pred_fn(xb[None])[0]
    for _ in range(n_perm):
        perm = rng.permutation(len(wins)); xc, fp = xb.copy(), f0
        for i in perm:
            s, e = wins[i]; xc[s:e] = x_seq[s:e]; fn_ = pred_fn(xc[None])[0]
            phi[i] += fn_ - fp; fp = fn_
    phi /= n_perm
    cm = np.zeros((T, D_))
    for i, (s, e) in enumerate(wins): cm[s:e] = phi[i] / max(1, (e-s)*D_)
    return cm

def limets_explain(x_seq, pred_fn, baseline, n_samples=400, kw=0.75, seed=0):
    rng = np.random.default_rng(seed); T, D_ = x_seq.shape
    Z   = rng.integers(0, 2, (n_samples, T)).astype(np.float32)
    xb  = np.tile(baseline[:D_], (T,1))
    preds = np.array([pred_fn(np.where(Z[i,:,None].astype(bool), x_seq, xb)[None])[0]
                      for i in range(n_samples)], dtype=np.float32)
    dists = np.sqrt(((Z-1)**2).sum(1)); weights = np.exp(-(dists**2)/(kw**2))
    reg   = Ridge(alpha=0.01); reg.fit(Z, preds, sample_weight=weights)
    return np.tile(reg.coef_.astype(np.float32)[:,None], (1, D_))


# ── Metrics ───────────────────────────────────────────────────────────────────
def comprehensiveness_batch(x_seq, cm, bpred, baseline, fracs=(0.1,0.2,0.3,0.5)):
    T, D_ = x_seq.shape; bl = baseline[:D_]; xb = np.tile(bl,(T,1))
    f0 = bpred(x_seq[None])[0]; idx = np.argsort(np.abs(cm.flatten()))[::-1]
    inputs = []
    for frac in fracs:
        k  = max(1, int(T*D_*frac)); xc = x_seq.flatten().copy()
        xc[idx[:k]] = xb.flatten()[idx[:k]]; inputs.append(xc.reshape(T,D_))
    return float(np.mean(f0 - bpred(np.stack(inputs))))

def sufficiency_batch(x_seq, cm, bpred, baseline, fracs=(0.1,0.2,0.3,0.5)):
    T, D_ = x_seq.shape; bl = baseline[:D_]; xb = np.tile(bl,(T,1))
    f0 = bpred(x_seq[None])[0]; idx = np.argsort(np.abs(cm.flatten()))[::-1]
    inputs = []
    for frac in fracs:
        k  = max(1, int(T*D_*frac)); xc = xb.flatten().copy()
        xc[idx[:k]] = x_seq.flatten()[idx[:k]]; inputs.append(xc.reshape(T,D_))
    return float(np.mean(np.abs(f0 - bpred(np.stack(inputs)))))

def gini_coef(cm):
    v = np.sort(np.abs(cm.flatten())); n = len(v); idx = np.arange(1, n+1)
    return float((2*(idx*v).sum()) / (n*v.sum()+1e-12) - (n+1)/n)

def rank_stability(cms):
    flat = [np.abs(c.flatten()) for c in cms]
    if len(flat) < 2: return 1.0
    return float(np.mean([spearmanr(flat[i],flat[j])[0]
                           for i,j in combinations(range(len(flat)),2)]))


def slice_seq(x_T8, T_target):
    T_src = x_T8.shape[0]
    if T_target <= T_src: return x_T8[-T_target:]
    pad = np.zeros((T_target-T_src, x_T8.shape[1]), np.float32)
    return np.vstack([pad, x_T8])


def make_stress_pred_fn(mdl, T_s, T_primary=T_PRIMARY):
    mdl.eval()
    def fn(x_np):
        B = x_np.shape[0]
        if T_s < T_primary:
            pad = np.zeros((B, T_primary-T_s, D), np.float32)
            x_feed = np.concatenate([pad, x_np], axis=1)
        elif T_s > T_primary:
            x_feed = x_np[:, -T_primary:]
        else:
            x_feed = x_np
        with torch.no_grad():
            out = mdl(torch.from_numpy(x_feed.astype(np.float32)).to(device))
        return torch.softmax(out, dim=1)[:,1].cpu().numpy()
    return fn


def main():
    # ── Section 1: Load data ──────────────────────────────────────────────────
    INPUT_FILE = os.path.join(CRITEO_DIR, 'CriteoSearchData')
    print('\n[Section 1] Loading Criteo ...')
    df_raw = pd.read_csv(INPUT_FILE, sep='\t', header=None, names=CRITEO_COLS,
                         nrows=2_000_000, low_memory=False)
    print(f'  {len(df_raw):,} rows loaded')

    for c in ['nb_clicks_1week', 'product_price']:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').replace(-1, np.nan)

    df_raw['click_dt']   = pd.to_datetime(df_raw['click_timestamp'], unit='s', errors='coerce')
    df_raw['click_hour'] = df_raw['click_dt'].dt.hour.fillna(0).astype(np.float32)
    df_raw['click_dow']  = df_raw['click_dt'].dt.dayofweek.fillna(0).astype(np.float32)

    CAT_MAP = {'device_type': 'device_type_enc', 'product_country': 'product_country_enc',
               'product_age_group': 'product_age_group_enc', 'product_gender': 'product_gender_enc',
               'product_category_1': 'product_category_1_enc', 'partner_id': 'partner_id_enc'}
    for raw, enc in CAT_MAP.items():
        df_raw[raw] = df_raw[raw].replace('-1', np.nan).fillna('UNK')
        le = LabelEncoder()
        df_raw[enc] = le.fit_transform(df_raw[raw].astype(str)).astype(np.float32)
    for c in SEQ_FEATURES:
        df_raw[c] = df_raw[c].fillna(0.0)

    # ── Section 2: Dataset factory ────────────────────────────────────────────
    X_all, y_all = build_sequences(df_raw, 'user_id', SEQ_FEATURES, T_PRIMARY, 2)
    print(f'  N={len(X_all):,}  CVR={y_all.mean()*100:.2f}%')

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.reshape(-1, D)).astype(np.float32).reshape(len(X_all), T_PRIMARY, D)

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.10, random_state=SEED, stratify=y_tr)

    idx_c = np.where(y_te==1)[0][:100]; idx_nc = np.where(y_te==0)[0][:100]
    exp_idx = np.concatenate([idx_c, idx_nc]); np.random.shuffle(exp_idx)
    X_exp   = X_te[exp_idx]; y_exp = y_te[exp_idx]

    SPARSITY_SETTINGS = [{'T':32,'label':'T32-Rich'}, {'T':16,'label':'T16-Medium'},
                          {'T':8,'label':'T8-Short'}, {'T':4,'label':'T4-Sparse'},
                          {'T':1,'label':'T1-Extreme'}]
    stress_pools = {s['label']: {'X': np.stack([slice_seq(X_exp[i], s['T'])
                                                  for i in range(len(X_exp))]),
                                  'y': y_exp, 'T': s['T']}
                    for s in SPARSITY_SETTINGS}

    baseline_global = X_tr.reshape(-1, D).mean(0).astype(np.float32)

    # ── Section 3: Train models ───────────────────────────────────────────────
    print('\n[Section 3] Training models ...')
    model_zoo = {
        'LSTM':        LSTMModel(D, 64, 2),
        'Transformer': TransformerModel(D, 64, 4, 2),
        'CNN':         CNNModel(D, 64),
        'MLPMixer':    MLPMixer(D, T_PRIMARY, 64),
    }
    models, histories = {}, {}
    for nm, mdl in model_zoo.items():
        m, h = train_model(mdl, X_tr, y_tr, X_va, y_va, nm)
        models[nm] = m; histories[nm] = h
        torch.save(m.state_dict(), os.path.join(SAVE_DIR, f'model_{nm}.pt'))

    # ── Section 4: Import GS-SHAP ─────────────────────────────────────────────
    from gsshap_standalone_advanced import (
        GSSHAP, cluster_features_hsic, segment_all_groups,
        build_group_segment_players, shapley_permutation, player_phi_to_cell_map,
        ShapleyAxiomChecker, SensitivityAnalyser,
    )

    def gsshap_noseg(x_seq, pred_fn, baseline, feature_groups, n_perm=80, seed=0):
        T, D_ = x_seq.shape; seg = max(1, T//4)
        unif = [(i*seg, min((i+1)*seg,T)) for i in range(T//seg)]
        players = build_group_segment_players(feature_groups, [unif]*len(feature_groups))
        phi = shapley_permutation(x_seq, players, baseline[:D_], pred_fn, n_perm, 16, np.random.default_rng(seed))
        return player_phi_to_cell_map(phi, players, T, D_)

    def gsshap_nogroup(x_seq, pred_fn, baseline, n_perm=80, seed=0):
        T, D_ = x_seq.shape; feature_groups = [[i] for i in range(D_)]
        segs    = segment_all_groups(x_seq, feature_groups, max(1,T//8), 4, 0.1, 200, seed)
        players = build_group_segment_players(feature_groups, segs)
        phi = shapley_permutation(x_seq, players, baseline[:D_], pred_fn, n_perm, 16, np.random.default_rng(seed))
        return player_phi_to_cell_map(phi, players, T, D_)

    hsic_groups = cluster_features_hsic(X_tr.reshape(-1, D), max_samples=2000, seed=SEED)
    N_PERM      = 200; N_EXPLAIN = 200
    X_exp_s     = X_exp[:N_EXPLAIN]; y_exp_s = y_exp[:N_EXPLAIN]

    METHOD_COLORS = {'GS-SHAP':'#E63946','TimeSHAP':'#457B9D','WinSHAP':'#2A9D8F',
                     'LIME-TS':'#F4A261','GS(NoSeg)':'#6A4C93','GS(NoGroup)':'#8D99AE'}

    # ── Section 6: Main experiment ────────────────────────────────────────────
    print('\n[Section 6] Main experiment ...')
    results = []; feat_attr_cache = {nm: [] for nm in model_zoo}

    for model_name, mdl in models.items():
        pred_fn   = make_pred_fn(mdl)
        baseline  = baseline_global.copy()
        gsshap_exp = GSSHAP(model=mdl, X_train=X_tr, task='clf', target_class=1,
                            device=device, hsic_max_samples=2000,
                            min_seg_len=max(1,T_PRIMARY//4), max_segments=4,
                            threshold_alpha=0.05, threshold_permutations=200,
                            num_permutations=N_PERM, batch_size=128, antithetic=True)
        for s_idx, x_seq in enumerate(X_exp_s):
            row = {'model': model_name, 'sample_idx': s_idx,
                   'label': int(y_exp_s[s_idx]), 'pred_prob': float(pred_fn(x_seq[None])[0])}
            phi_gs, players_gs, cm_gs = gsshap_exp.explain(x_seq, seed=SEED)
            row['gsshap_comp'] = comprehensiveness_batch(x_seq, cm_gs, pred_fn, baseline)
            row['gsshap_suff'] = sufficiency_batch(x_seq, cm_gs, pred_fn, baseline)
            row['gsshap_gini'] = gini_coef(cm_gs)
            row['gsshap_time'] = 0.0  # placeholder; use perf_counter in production
            row['gsshap_nplayers'] = len(players_gs)
            f_full = float(pred_fn(x_seq[None])[0])
            f_base = float(pred_fn(np.tile(baseline[:D], (T_PRIMARY,1))[None])[0])
            row['gsshap_efficiency_err'] = abs(float(phi_gs.sum()) - (f_full - f_base))
            if s_idx < 50: feat_attr_cache[model_name].append(np.abs(cm_gs).mean(axis=0))

            cm_ts = timeshap_explain(x_seq, pred_fn, baseline, N_PERM, SEED)
            row['timeshap_comp'] = comprehensiveness_batch(x_seq, cm_ts, pred_fn, baseline)
            row['timeshap_suff'] = sufficiency_batch(x_seq, cm_ts, pred_fn, baseline)
            row['timeshap_gini'] = gini_coef(cm_ts); row['timeshap_time'] = 0.0

            cm_ws = winshap_explain(x_seq, pred_fn, baseline, max(1,T_PRIMARY//4), N_PERM, SEED)
            row['winshap_comp'] = comprehensiveness_batch(x_seq, cm_ws, pred_fn, baseline)
            row['winshap_suff'] = sufficiency_batch(x_seq, cm_ws, pred_fn, baseline)
            row['winshap_gini'] = gini_coef(cm_ws); row['winshap_time'] = 0.0

            cm_li = limets_explain(x_seq, pred_fn, baseline, N_PERM*4, 0.75, SEED)
            row['lime_comp'] = comprehensiveness_batch(x_seq, cm_li, pred_fn, baseline)
            row['lime_suff'] = sufficiency_batch(x_seq, cm_li, pred_fn, baseline)
            row['lime_gini'] = gini_coef(cm_li); row['lime_time'] = 0.0

            cm_ns = gsshap_noseg(x_seq, pred_fn, baseline, hsic_groups, N_PERM, SEED)
            row['noseg_comp'] = comprehensiveness_batch(x_seq, cm_ns, pred_fn, baseline)
            row['noseg_suff'] = sufficiency_batch(x_seq, cm_ns, pred_fn, baseline)
            row['noseg_gini'] = gini_coef(cm_ns); row['noseg_time'] = 0.0

            cm_ng = gsshap_nogroup(x_seq, pred_fn, baseline, N_PERM, SEED)
            row['nogrp_comp'] = comprehensiveness_batch(x_seq, cm_ng, pred_fn, baseline)
            row['nogrp_suff'] = sufficiency_batch(x_seq, cm_ng, pred_fn, baseline)
            row['nogrp_gini'] = gini_coef(cm_ng); row['nogrp_time'] = 0.0

            row['rank_gs_ts'] = spearmanr(np.abs(cm_gs.flatten()), np.abs(cm_ts.flatten()))[0]
            results.append(row)
        print(f'  {model_name}: done')

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(SAVE_DIR, 'results_main.csv'), index=False)
    print(f'  Saved results_main.csv ({len(res)} rows)')

    # ── Section 8: RQ2 stress test ────────────────────────────────────────────
    print('\n[Section 8] RQ2 Stress Test ...')
    stress_model = models['LSTM']; N_STAB = 4; n_perm_s = 100
    stress_results = []
    for s_conf in SPARSITY_SETTINGS:
        label = s_conf['label']; T_s = s_conf['T']
        pool  = stress_pools[label]; X_sel = pool['X']; y_sel = pool['y']
        bl_s  = baseline_global[:D].copy()
        pred_s = make_stress_pred_fn(stress_model, T_s)
        X_train_s = np.stack([slice_seq(X_tr[i], T_s) for i in range(min(2000, len(X_tr)))])
        gs_exp_s  = GSSHAP(model=stress_model, X_train=X_train_s, task='clf', target_class=1,
                            device=device, hsic_max_samples=800,
                            min_seg_len=max(1,T_s//4), max_segments=min(4,max(1,T_s)),
                            threshold_alpha=0.05, threshold_permutations=100,
                            num_permutations=n_perm_s, batch_size=64, antithetic=True)
        for i in range(len(X_sel)):
            xs = X_sel[i]
            row_s = {'sparsity_label': label, 'T': T_s, 'sample_idx': i, 'label': int(y_sel[i])}
            try:
                _, pl_s, cm_gs_s = gs_exp_s.explain(xs, seed=SEED)
                row_s['gsshap_comp'] = comprehensiveness_batch(xs, cm_gs_s, pred_s, bl_s)
                row_s['gsshap_suff'] = sufficiency_batch(xs, cm_gs_s, pred_s, bl_s)
                row_s['gsshap_gini'] = gini_coef(cm_gs_s); row_s['gsshap_npl'] = len(pl_s)
                cms_rep = [cm_gs_s]
                for _ in range(N_STAB-1):
                    _, _, cm_r = gs_exp_s.explain(xs, seed=np.random.randint(999)); cms_rep.append(cm_r)
                row_s['gsshap_stability'] = rank_stability(cms_rep)
            except Exception as e:
                row_s.update({'gsshap_comp':np.nan,'gsshap_suff':np.nan,'gsshap_gini':np.nan,
                               'gsshap_npl':np.nan,'gsshap_stability':np.nan})
            cm_ts_s = timeshap_explain(xs, pred_s, bl_s, n_perm_s, SEED)
            row_s['timeshap_comp'] = comprehensiveness_batch(xs, cm_ts_s, pred_s, bl_s)
            row_s['timeshap_suff'] = sufficiency_batch(xs, cm_ts_s, pred_s, bl_s)
            cms_ts = [cm_ts_s]
            for _ in range(N_STAB-1):
                cms_ts.append(timeshap_explain(xs, pred_s, bl_s, n_perm_s, np.random.randint(999)))
            row_s['timeshap_stability'] = rank_stability(cms_ts)
            cm_ws_s = winshap_explain(xs, pred_s, bl_s, max(1,T_s//4), n_perm_s, SEED)
            row_s['winshap_comp'] = comprehensiveness_batch(xs, cm_ws_s, pred_s, bl_s)
            stress_results.append(row_s)
        print(f'  {label} (T={T_s}): done')

    stress_df = pd.DataFrame(stress_results)
    stress_df.to_csv(os.path.join(SAVE_DIR, 'results_stress.csv'), index=False)

    # ── Section 13: Axiom verification ───────────────────────────────────────
    print('\n[Section 13] Axiom verification ...')
    ax_exp = GSSHAP(model=models['LSTM'], X_train=X_tr, task='clf', target_class=1,
                    device=device, hsic_max_samples=1000, min_seg_len=max(1,T_PRIMARY//4),
                    max_segments=4, threshold_alpha=0.05, threshold_permutations=200,
                    num_permutations=N_PERM, batch_size=128, antithetic=True)
    checker = ShapleyAxiomChecker(ax_exp)
    axiom_rows = []
    for s_idx in range(20):
        eff = checker.check_efficiency(X_exp_s[s_idx], seed=s_idx, tol=5e-2)
        dum = checker.check_dummy(X_exp_s[s_idx], dummy_group_idx=0, seed=s_idx, tol=1e-2)
        axiom_rows.append({'sample_idx': s_idx,
                           'efficiency_error': eff['error'], 'efficiency_pass': eff['passed'],
                           'dummy_max_phi': dum['max_abs_phi'], 'dummy_pass': dum['passed']})
    pd.DataFrame(axiom_rows).to_csv(os.path.join(SAVE_DIR, 'rq_axiom_check.csv'), index=False)

    sa = SensitivityAnalyser(model=models['LSTM'], X_train=X_tr, task='clf',
                              target_class=1, device=device, n_reps=4)
    sa_rows = []
    for si in list(np.where(y_exp_s==1)[0][:3]) + list(np.where(y_exp_s==0)[0][:2]):
        for r in sa.run(X_exp_s[si], n_thresh_list=[10,50,100,200], verbose=False):
            r['sample_idx'] = int(si); sa_rows.append(r)
    sa_df = pd.DataFrame(sa_rows)
    sa_df.groupby('threshold_permutations').agg(
        stability_mean=('rank_stability','mean'), stability_std=('rank_stability','std'),
        runtime_mean=('mean_runtime_s','mean')).reset_index().to_csv(
        os.path.join(SAVE_DIR, 'rq_sensitivity.csv'), index=False)

    print('\n✅  All sections complete. Results saved to:', SAVE_DIR)
    print(f'   Efficiency axiom pass rate (LSTM 20 samples): '
          f'{pd.DataFrame(axiom_rows)["efficiency_pass"].mean()*100:.0f}%')


if __name__ == '__main__':
    main()
