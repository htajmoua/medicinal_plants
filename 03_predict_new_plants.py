"""
Step 3: final predictions (Tables 5 and 6).
The selected model (logistic regression, C = 0.001, L2, on BiomedBERT
embeddings) is retrained on 100% of the labelled data and applied to the
4 validation plants and the 11 Moroccan plants with the decision threshold
0.39 selected in step 2.
"""
import pandas as pd
import numpy as np
import os, warnings, re, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score

RANDOM_STATE = 42
PCA_VARIANCE = 0.95
OPT_THRESHOLD = 0.39
BEST_C = 0.001

feature_columns = [
    'SPECIES', 'REGION', 'USED PART', 'METABOLITE CONTENT',
    'TESTED MICROORGANISME', 'EFFECTIVE CONCENTRATION'
]

def serialize_row(row):
    parts = []
    if 'SPECIES' in row.index:
        parts.append(f"Species: {row['SPECIES']}")
    if 'REGION' in row.index:
        parts.append(f"Region: {row['REGION']}")
    if 'USED PART' in row.index:
        parts.append(f"Part: {row['USED PART']}")
    if 'METABOLITE CONTENT' in row.index:
        parts.append(f"Metabolites: {row['METABOLITE CONTENT']}")
    if 'TESTED MICROORGANISME' in row.index:
        parts.append(f"Microorganism: {row['TESTED MICROORGANISME']}")
    if 'EFFECTIVE CONCENTRATION' in row.index:
        parts.append(f"Concentration: {row['EFFECTIVE CONCENTRATION']}")
    return ". ".join(parts) + "."

def clean_species_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\d+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_concentration(val: str) -> str:
    s = str(val).strip()
    s = re.sub(r'\s*\(?\s*[Cc][Ll][Aa][Ss][Ss][Ee]\s*\d+\s*\)?', '', s)
    return s.strip()

def clean_df(df):
    df.columns = df.columns.str.strip()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
    if 'SPECIES' in df.columns:
        df['SPECIES'] = df['SPECIES'].apply(clean_species_name)
    if 'EFFECTIVE CONCENTRATION' in df.columns:
        df['EFFECTIVE CONCENTRATION'] = df['EFFECTIVE CONCENTRATION'].apply(clean_concentration)
    return df

# ── 1. Load & encode ALL training data ───────────────────────
print("=" * 60)
print("FINAL MODEL: LogReg + BiomedBERT (100% data, t=0.39)")
print("=" * 60)

os.makedirs('results', exist_ok=True)
data_file = os.path.join('data', 'data_med_plants.xlsx')
df = pd.read_excel(data_file, sheet_name='Dataset(litterature)', header=1)
df.columns = df.columns.str.strip()
for col in feature_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
df = df.dropna(subset=['TARGET'])
df['TARGET'] = df['TARGET'].astype(int)

all_texts = df.apply(serialize_row, axis=1).tolist()
all_labels = df['TARGET'].values
print(f"Training samples: {len(all_texts)}")
print(f"Distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

# ── 2. Load validation ───────────────────────────────────────
df_val = pd.read_excel(data_file, sheet_name='validation', header=1)
df_val = df_val.dropna(how='all')
df_val = clean_df(df_val)
real_mask_val = df_val['SPECIES'].str.len() > 2
df_val = df_val[real_mask_val].copy()
val_texts = df_val.apply(serialize_row, axis=1).tolist()

has_val_target = 'TARGET' in df_val.columns and df_val['TARGET'].notna().any()
if has_val_target:
    df_val['TARGET'] = df_val['TARGET'].astype(int)
print(f"Validation points: {len(df_val)}")

# ── 3. Load Moroccan plants ──────────────────────────────────
df_pred = pd.read_excel(data_file, sheet_name='Prediction', header=1)
df_pred = df_pred.dropna(how='all')
df_pred = clean_df(df_pred)
real_mask = df_pred['SPECIES'].str.len() > 2
df_moroccan = df_pred[real_mask & (df_pred['SPECIES'] != 'Unknown')].copy()
pred_texts = df_moroccan.apply(serialize_row, axis=1).tolist()
print(f"Moroccan plants: {len(df_moroccan)}")

# ── 4. Encode ─────────────────────────────────────────────────
print("\nEncoding with BiomedBERT...")
encoder = SentenceTransformer(
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', device='cpu'
)
X_all = encoder.encode(all_texts, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
X_val = encoder.encode(val_texts, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
X_pred = encoder.encode(pred_texts, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
del encoder
gc.collect()

# ── 5. Train pipeline on 100% ────────────────────────────────
print("\nTraining LogReg (C=0.001) on all data...")
scaler = StandardScaler()
X_all_s = scaler.fit_transform(X_all)
pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
X_all_p = pca.fit_transform(X_all_s)
print(f"  PCA: {X_all.shape[1]} -> {X_all_p.shape[1]} components")

clf = LogisticRegression(C=BEST_C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)
clf.fit(X_all_p, all_labels)
print("  Done.")

# ── 6. Predict validation ────────────────────────────────────
X_val_p = pca.transform(scaler.transform(X_val))
val_probs = clf.predict_proba(X_val_p)[:, 1]
val_preds_opt = (val_probs >= OPT_THRESHOLD).astype(int)

df_val['Prob_Class1'] = val_probs
df_val['Pred_OptT'] = val_preds_opt

print(f"\n{'='*60}")
print(f"  VALIDATION SET (t={OPT_THRESHOLD})")
print(f"{'='*60}")
val_cols = ['SPECIES', 'REGION', 'USED PART']
if has_val_target:
    val_cols.append('TARGET')
val_cols += ['Pred_OptT', 'Prob_Class1']
print(df_val[val_cols].to_string(index=False))

if has_val_target:
    print(f"\n  Accuracy (t={OPT_THRESHOLD}): {accuracy_score(df_val['TARGET'], val_preds_opt):.4f}")

# ── 7. Predict Moroccan plants ────────────────────────────────
X_pred_p = pca.transform(scaler.transform(X_pred))
pred_probs = clf.predict_proba(X_pred_p)[:, 1]
pred_preds_opt = (pred_probs >= OPT_THRESHOLD).astype(int)

df_moroccan['Prob_Class1'] = pred_probs
df_moroccan['Pred_OptT'] = pred_preds_opt

print(f"\n{'='*60}")
print(f"  MOROCCAN PLANTS — sorted by probability (t={OPT_THRESHOLD})")
print(f"{'='*60}")
ranking = df_moroccan[['SPECIES', 'Pred_OptT', 'Prob_Class1']].copy()
ranking = ranking.sort_values('Prob_Class1', ascending=False)
print(ranking.to_string(index=False))

n_pos_opt = pred_preds_opt.sum()
n_neg_opt = len(pred_preds_opt) - n_pos_opt
print(f"\n  Distribution (t={OPT_THRESHOLD}): {n_pos_opt} positive, {n_neg_opt} negative")

# Save
df_val[val_cols].to_csv('results/predictions_validation_set.csv', index=False)
ranking.to_csv('results/predictions_moroccan_plants.csv', index=False)
print("\nSaved: results/predictions_validation_set.csv")
print("Saved: results/predictions_moroccan_plants.csv")
