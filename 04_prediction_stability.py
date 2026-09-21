"""
Step 4: stability and calibration of the final model's predictions (added
during revision).

Final model = StandardScaler -> PCA(95% variance) -> LogisticRegression(C=0.001, L2)
on BiomedBERT embeddings (mean pooling, frozen weights) of the serialized
records, trained on 100% of the 183 labelled samples, decision threshold 0.39.

(1) Reproduces the point estimates of Tables 5 and 6.
(2) Bootstrap resampling of the 183 training rows (B = 1000): per-plant
    probability distribution and frequency of the induction call at t = 0.39.
(3) Threshold sensitivity: predicted class over t in [0.30, 0.50] and at the
    five per-shuffle CV thresholds (0.30, 0.38, 0.39, 0.43, 0.44).
(4) Calibration of out-of-fold probabilities (10 x stratified 5-fold CV):
    Brier score, log-loss, AUC, expected calibration error, calibration
    slope/intercept, reliability table.
(5) Tables 5 and 6 augmented with reference labels, citations and experimental
    conditions taken from the source workbook.
"""
import os, re, gc, warnings, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                             average_precision_score)

RANDOM_STATE = 42
PCA_VARIANCE = 0.95
OPT_THRESHOLD = 0.39
BEST_C = 0.001
N_BOOT = 1000
CV_SEED_THRESHOLDS = {42: 0.38, 123: 0.39, 456: 0.44, 789: 0.30, 2024: 0.43}  # results/model_selection_cv_details.csv
EMB_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
DATA_FILE = os.path.join('data', 'data_med_plants.xlsx')

feature_columns = ['SPECIES', 'REGION', 'USED PART', 'METABOLITE CONTENT',
                   'TESTED MICROORGANISME', 'EFFECTIVE CONCENTRATION']


def serialize_row(row):
    return (f"Species: {row['SPECIES']}. Region: {row['REGION']}. Part: {row['USED PART']}. "
            f"Metabolites: {row['METABOLITE CONTENT']}. Microorganism: {row['TESTED MICROORGANISME']}. "
            f"Concentration: {row['EFFECTIVE CONCENTRATION']}.")


def clean_species_name(name):
    s = re.sub(r'\([^)]*\)', '', str(name).strip())
    s = re.sub(r'\d+', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def clean_concentration(val):
    return re.sub(r'\s*\(?\s*[Cc][Ll][Aa][Ss][Ss][Ee]\s*\d+\s*\)?', '', str(val).strip()).strip()


def clean_df(df):
    df.columns = df.columns.str.strip()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
    df['SPECIES'] = df['SPECIES'].apply(clean_species_name)
    df['EFFECTIVE CONCENTRATION'] = df['EFFECTIVE CONCENTRATION'].apply(clean_concentration)
    return df


# ── data (identical to predict_final.py) ─────────────────────────────────
df = pd.read_excel(DATA_FILE, sheet_name='Dataset(litterature)', header=1)
df.columns = df.columns.str.strip()
for col in feature_columns:
    df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
df = df.dropna(subset=['TARGET'])
y = df['TARGET'].astype(int).values
train_texts = df.apply(serialize_row, axis=1).tolist()

df_val_raw = pd.read_excel(DATA_FILE, sheet_name='validation', header=1).dropna(how='all')
raw_cols = list(df_val_raw.columns)
df_val_raw = df_val_raw.rename(columns={raw_cols[7]: 'REF_LABEL', raw_cols[8]: 'REFERENCE',
                                        raw_cols[6]: 'REPORTED_EFFECT'})
df_val = clean_df(df_val_raw)
df_val = df_val[(df_val['SPECIES'].str.len() > 2) & (df_val['SPECIES'] != 'Unknown')].copy()
df_val['REF_LABEL'] = df_val['REF_LABEL'].astype(int)
val_texts = df_val.apply(serialize_row, axis=1).tolist()

df_mor_raw = pd.read_excel(DATA_FILE, sheet_name='Prediction', header=1).dropna(how='all')
df_mor_raw = df_mor_raw.rename(columns={'references': 'REFERENCE'})
df_mor = clean_df(df_mor_raw)
df_mor = df_mor[(df_mor['SPECIES'].str.len() > 2) & (df_mor['SPECIES'] != 'Unknown')].copy()
mor_texts = df_mor.apply(serialize_row, axis=1).tolist()
print(f"train N={len(train_texts)} (pos={y.sum()}), validation n={len(df_val)}, Moroccan n={len(df_mor)}")

# ── embeddings (cached) ──────────────────────────────────────────────────
os.makedirs('cache', exist_ok=True); os.makedirs('results', exist_ok=True)
cache = {'train': 'cache/_X_final_train_BiomedBERT.npy',
         'val': 'cache/_X_final_val_BiomedBERT.npy',
         'mor': 'cache/_X_final_moroccan_BiomedBERT.npy'}
if all(os.path.exists(p) for p in cache.values()):
    X_tr, X_val, X_mor = (np.load(cache[k]) for k in ('train', 'val', 'mor'))
    print("embeddings loaded from cache")
else:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMB_MODEL, device='cpu')
    print(f"max_seq_length={enc.max_seq_length}, pooling={enc[1].get_config_dict()}")
    X_tr = enc.encode(train_texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    X_val = enc.encode(val_texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    X_mor = enc.encode(mor_texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    np.save(cache['train'], X_tr); np.save(cache['val'], X_val); np.save(cache['mor'], X_mor)
    del enc; gc.collect()
X_new = np.vstack([X_val, X_mor])
names = list(df_val['SPECIES']) + list(df_mor['SPECIES'])
group = ['validation'] * len(df_val) + ['Moroccan'] * len(df_mor)


def fit_predict(Xa, ya, Xb, seed=RANDOM_STATE):
    sc = StandardScaler().fit(Xa)
    pca = PCA(n_components=PCA_VARIANCE, random_state=seed).fit(sc.transform(Xa))
    clf = LogisticRegression(C=BEST_C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=seed)
    clf.fit(pca.transform(sc.transform(Xa)), ya)
    return clf.predict_proba(pca.transform(sc.transform(Xb)))[:, 1], pca.n_components_


# ── (1) point estimates ──────────────────────────────────────────────────
p_point, n_pc = fit_predict(X_tr, y, X_new)
print(f"\n(1) Point estimates (100% data, {n_pc} PCs, t={OPT_THRESHOLD})")
published = {'Kalanchoe pinnata': 0.765, 'Phyllanthus amarus': 0.687, 'Eremanthus crotonoides': 0.437,
             'Empetrum nigrum L.': 0.376, 'Myrtus communis': 0.588, 'Salvia clandestina': 0.538,
             'Origanum elongatum': 0.458, 'Capparis spinosa': 0.414, 'Origanum compactum': 0.409,
             'Warionia saharae': 0.406, 'Ephedra altissima': 0.394, 'Euphorbia echinus': 0.387,
             'Ephedra fragilis': 0.352, 'Lavandula dentata': 0.345, 'Cannabis sativa': 0.215}
max_dev = 0
for n, p in zip(names, p_point):
    pub = published.get(n, np.nan)
    max_dev = max(max_dev, abs(p - pub))
    print(f"  {n:24s} P={p:.3f}  published={pub:.3f}")
print(f"  max |deviation| from published tables = {max_dev:.4f}")

# ── (2) bootstrap ────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_STATE)
P = np.zeros((N_BOOT, len(names)))
b = 0
while b < N_BOOT:
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    P[b], _ = fit_predict(X_tr[idx], y[idx], X_new, seed=int(rng.integers(0, 10**6)))
    b += 1
    if b % 200 == 0:
        print(f"  bootstrap {b}/{N_BOOT}")
np.save('cache/bootstrap_probabilities.npy', P)

boot = pd.DataFrame({
    'Group': group, 'Species': names,
    'P_point': np.round(p_point, 3),
    'Pred_point_t0.39': (p_point >= OPT_THRESHOLD).astype(int),
    'P_boot_mean': np.round(P.mean(0), 3),
    'P_boot_sd': np.round(P.std(0), 3),
    'P_boot_2.5%': np.round(np.percentile(P, 2.5, axis=0), 3),
    'P_boot_97.5%': np.round(np.percentile(P, 97.5, axis=0), 3),
    'Freq_induction_t0.39': np.round((P >= OPT_THRESHOLD).mean(0), 3),
})
# rank stability among the 11 Moroccan plants
mor_mask = np.array(group) == 'Moroccan'
ranks = (-P[:, mor_mask]).argsort(1).argsort(1) + 1
rank_mean = np.full(len(names), np.nan); rank_lo = rank_mean.copy(); rank_hi = rank_mean.copy()
rank_mean[mor_mask] = ranks.mean(0)
rank_lo[mor_mask] = np.percentile(ranks, 2.5, axis=0)
rank_hi[mor_mask] = np.percentile(ranks, 97.5, axis=0)
boot['Rank_boot_mean'] = np.round(rank_mean, 1)
boot['Rank_boot_2.5%'] = rank_lo
boot['Rank_boot_97.5%'] = rank_hi
boot.to_csv('results/prediction_stability_bootstrap.csv', index=False)
print(f"\n(2) Bootstrap (B={N_BOOT}) stability")
print(boot.to_string(index=False))

# ── (3) threshold sensitivity ────────────────────────────────────────────
ts = [round(t, 2) for t in np.arange(0.30, 0.505, 0.01)]
sens = pd.DataFrame({'Group': group, 'Species': names, 'P_point': np.round(p_point, 3)})
for t in ts:
    sens[f't={t:.2f}'] = (p_point >= t).astype(int)
sens.to_csv('results/threshold_sensitivity.csv', index=False)
print("\n(3) Threshold sensitivity (Moroccan set): n predicted 'induction' per threshold")
mor_p = p_point[mor_mask]
for t in ts:
    flips = int(((mor_p >= t) != (mor_p >= OPT_THRESHOLD)).sum())
    print(f"  t={t:.2f}: {int((mor_p >= t).sum()):2d}/11 induction, {flips} flips vs t=0.39")
print("  per-shuffle CV thresholds:")
for s, t in CV_SEED_THRESHOLDS.items():
    flips = [n for n, p in zip(np.array(names)[mor_mask], mor_p) if (p >= t) != (p >= OPT_THRESHOLD)]
    print(f"    seed {s}: t={t:.2f} -> {int((mor_p >= t).sum())}/11 induction; flips: {flips}")
band = sorted(set([0.34, 0.44] + list(CV_SEED_THRESHOLDS.values())))
stable = [n for n, p in zip(names, p_point) if (p >= max(band)) or (p < min(band))]
print(f"  plants whose call is unchanged for every t in [{min(band):.2f}, {max(band):.2f}]: {stable}")

# ── (4) calibration on out-of-fold probabilities ─────────────────────────
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
oof = np.zeros((10, len(y)))
for k, (tr, te) in enumerate(rskf.split(X_tr, y)):
    r = k // 5
    oof[r, te], _ = fit_predict(X_tr[tr], y[tr], X_tr[te])
rows = []
for r in range(10):
    p = oof[r]
    lg = np.log(p / (1 - p))
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y)
    # ECE with 5 quantile bins
    qb = np.quantile(p, np.linspace(0, 1, 6)); qb[-1] += 1e-9
    bins = np.clip(np.digitize(p, qb[1:-1]), 0, 4)
    ece = sum(abs(p[bins == i].mean() - y[bins == i].mean()) * (bins == i).mean() for i in range(5))
    rows.append({'repeat': r, 'AUC': roc_auc_score(y, p), 'AP': average_precision_score(y, p),
                 'Brier': brier_score_loss(y, p), 'LogLoss': log_loss(y, p),
                 'ECE_5qbins': ece, 'Cal_slope': cal.coef_[0, 0], 'Cal_intercept': cal.intercept_[0],
                 'P_min': p.min(), 'P_max': p.max(), 'P_mean': p.mean(),
                 'F1_t0.39': __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y, (p >= OPT_THRESHOLD).astype(int)),
                 'F1_t0.50': __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y, (p >= 0.5).astype(int))})
cal_df = pd.DataFrame(rows)
prior = y.mean()
summary = cal_df.drop(columns='repeat').agg(['mean', 'std']).T
summary['Brier_class_prior'] = prior * (1 - prior)
summary.to_csv('results/calibration_summary.csv')
print("\n(4) Calibration of out-of-fold probabilities (10 x stratified 5-fold, N=183)")
print(summary.round(3).to_string())
print(f"  Brier of constant class-prior predictor = {prior*(1-prior):.3f}")
p_pool = oof.mean(0)
qb = np.quantile(p_pool, np.linspace(0, 1, 6)); qb[-1] += 1e-9
bins = np.clip(np.digitize(p_pool, qb[1:-1]), 0, 4)
rel = pd.DataFrame({'bin': range(1, 6),
                    'P_range': [f"{p_pool[bins==i].min():.3f}-{p_pool[bins==i].max():.3f}" for i in range(5)],
                    'n': [(bins == i).sum() for i in range(5)],
                    'mean_predicted': [p_pool[bins == i].mean() for i in range(5)],
                    'observed_induction_rate': [y[bins == i].mean() for i in range(5)]})
rel.to_csv('results/calibration_reliability.csv', index=False)
print("  Reliability table (OOF probability averaged over the 10 repeats, quantile bins):")
print(rel.round(3).to_string(index=False))
pd.DataFrame({'SPECIES': df['SPECIES'].values, 'y': y, 'P_oof_mean': np.round(p_pool, 4),
              'P_oof_sd_across_repeats': np.round(oof.std(0), 4)}).to_csv('results/calibration_out_of_fold.csv', index=False)

# ── (5) augmented Tables 5 / 6 ───────────────────────────────────────────
t5 = df_val[['SPECIES', 'REGION', 'USED PART', 'METABOLITE CONTENT', 'TESTED MICROORGANISME',
             'EFFECTIVE CONCENTRATION', 'REPORTED_EFFECT', 'REF_LABEL', 'REFERENCE']].copy()
t5['P_class1'] = np.round(p_point[:len(df_val)], 3)
t5['Prediction_t0.39'] = (p_point[:len(df_val)] >= OPT_THRESHOLD).astype(int)
t5['Agreement'] = np.where(t5['Prediction_t0.39'] == t5['REF_LABEL'], 'yes', 'NO')
t5['Freq_induction_boot'] = boot.loc[boot.Group == 'validation', 'Freq_induction_t0.39'].values
t5.to_csv('results/table5_validation_with_reference_labels.csv', index=False)
print("\n(5) Table 5 with reference labels")
print(t5[['SPECIES', 'REPORTED_EFFECT', 'REF_LABEL', 'P_class1', 'Prediction_t0.39', 'Agreement', 'REFERENCE']].to_string(index=False))
print(f"  agreement: {(t5['Agreement']=='yes').sum()}/{len(t5)}")

t6 = df_mor[['SPECIES', 'REGION', 'USED PART', 'METABOLITE CONTENT', 'TESTED MICROORGANISME',
             'EFFECTIVE CONCENTRATION', 'REFERENCE']].copy()
t6['P_class1'] = np.round(p_point[len(df_val):], 3)
t6['Prediction_t0.39'] = (p_point[len(df_val):] >= OPT_THRESHOLD).astype(int)
mb = boot[boot.Group == 'Moroccan']
t6['P_boot_95CI'] = [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(mb['P_boot_2.5%'], mb['P_boot_97.5%'])]
t6['Freq_induction_boot'] = mb['Freq_induction_t0.39'].values
t6 = t6.sort_values('P_class1', ascending=False)
t6.to_csv('results/table6_moroccan_with_conditions.csv', index=False)
print("\n(5) Table 6 with experimental conditions and bootstrap CI")
print(t6[['SPECIES', 'TESTED MICROORGANISME', 'EFFECTIVE CONCENTRATION', 'REFERENCE', 'P_class1',
          'Prediction_t0.39', 'P_boot_95CI', 'Freq_induction_boot']].to_string(index=False))

# ── figure ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axes[0]
order = np.argsort(-p_point)
data = [P[:, i] for i in order]
labels = [f"{names[i]}{' (val.)' if group[i]=='validation' else ''}" for i in order]
bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
for patch, i in zip(bp['boxes'], order):
    patch.set_facecolor('#f4a582' if group[i] == 'Moroccan' else '#92c5de'); patch.set_alpha(0.8)
ax.scatter(p_point[order], np.arange(1, len(order) + 1), marker='D', color='k', s=18, zorder=5, label='point estimate (100% data)')
ax.axvline(OPT_THRESHOLD, color='r', ls='--', lw=1.2, label='t = 0.39')
ax.axvspan(0.30, 0.44, color='r', alpha=0.07, label='per-shuffle CV thresholds (0.30-0.44)')
ax.set_yticks(np.arange(1, len(order) + 1)); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel('P(class = 1, induction)'); ax.set_title('A. Bootstrap distribution of predicted probabilities (B = 1000)', fontsize=10, loc='left')
ax.legend(fontsize=7, loc='lower right')
ax = axes[1]
ax.plot([0, 1], [0, 1], 'k:', lw=1, label='perfect calibration')
ax.plot(rel['mean_predicted'], rel['observed_induction_rate'], 'o-', color='#2166ac', label='out-of-fold (10 x 5-fold CV)')
for _, r_ in rel.iterrows():
    ax.annotate(f"n={int(r_['n'])}", (r_['mean_predicted'], r_['observed_induction_rate']), textcoords='offset points', xytext=(5, -10), fontsize=7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('mean predicted probability'); ax.set_ylabel('observed fraction of induction (class 1)')
ax.set_title(f"B. Reliability diagram (Brier = {cal_df['Brier'].mean():.3f}, slope = {cal_df['Cal_slope'].mean():.2f})", fontsize=10, loc='left')
ax.legend(fontsize=7, loc='upper left')
plt.tight_layout(); plt.savefig('results/figure_stability_calibration.png', dpi=200); plt.savefig('results/figure_stability_calibration.pdf')
print("\nSaved to results/: prediction_stability_bootstrap.csv, threshold_sensitivity.csv, calibration_summary.csv,")
print("  calibration_reliability.csv, calibration_out_of_fold.csv, table5_validation_with_reference_labels.csv,")
print("  table6_moroccan_with_conditions.csv, figure_stability_calibration.png/.pdf")
