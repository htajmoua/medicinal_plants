"""
Step 7: nearest-neighbour analysis (Table S3 of the Supplementary Materials).
For each of the 15 unlabelled plants (4 validation plants and 11 Moroccan
plants), the three closest training records in the raw BiomedBERT embedding
space (cosine similarity of the mean-pooled 768-d vectors, before
standardisation and PCA), with their labels and descriptors, together with the
score P(class = 1) of the final model.
Uses the embeddings cached by 01_encode_embeddings.py (training records) and
04_prediction_stability.py (unlabelled plants); the unlabelled plants are
re-encoded if that cache is absent.
Outputs: results/nearest_neighbors_table_S3.csv, results/nearest_neighbors.txt
"""
import os, re, warnings, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
PCA_VARIANCE = 0.95
OPT_THRESHOLD = 0.39
BEST_C = 0.001
EMB_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
DATA_FILE = os.path.join('data', 'data_med_plants.xlsx')
os.makedirs('results', exist_ok=True); os.makedirs('cache', exist_ok=True)

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
    """Same cleaning as in 04_prediction_stability.py for the unlabelled sets."""
    df.columns = df.columns.str.strip()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
    df['SPECIES'] = df['SPECIES'].apply(clean_species_name)
    df['EFFECTIVE CONCENTRATION'] = df['EFFECTIVE CONCENTRATION'].apply(clean_concentration)
    return df[(df['SPECIES'].str.len() > 2) & (df['SPECIES'] != 'Unknown')].copy()


# ── training records (same reading as 01 and 04: no species cleaning) ────
df_train = pd.read_excel(DATA_FILE, sheet_name='Dataset(litterature)', header=1)
df_train.columns = df_train.columns.str.strip()
df_raw = df_train.copy()   # descriptors as written in the workbook, for display in Table S3
for col in feature_columns:
    df_train[col] = df_train[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
df_train = df_train.dropna(subset=['TARGET'])
df_raw = df_raw.loc[df_train.index].fillna('ND')
y = df_train['TARGET'].astype(int).values

# ── unlabelled plants: validation sheet then Prediction sheet ────────────
df_val = clean_df(pd.read_excel(DATA_FILE, sheet_name='validation', header=1).dropna(how='all'))
df_mor = clean_df(pd.read_excel(DATA_FILE, sheet_name='Prediction', header=1).dropna(how='all'))
names = list(df_val['SPECIES']) + list(df_mor['SPECIES'])
group = ['validation'] * len(df_val) + ['Moroccan'] * len(df_mor)
print(f"train N={len(df_train)}, validation n={len(df_val)}, Moroccan n={len(df_mor)}")

# ── embeddings ───────────────────────────────────────────────────────────
X_train = np.load('cache/_X_all_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext.npy')
cache_val, cache_mor = 'cache/_X_final_val_BiomedBERT.npy', 'cache/_X_final_moroccan_BiomedBERT.npy'
if os.path.exists(cache_val) and os.path.exists(cache_mor):
    X_val, X_mor = np.load(cache_val), np.load(cache_mor)
    print("unlabelled-plant embeddings loaded from the cache written by 04_prediction_stability.py")
else:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMB_MODEL, device='cpu')
    X_val = enc.encode(df_val.apply(serialize_row, axis=1).tolist(), batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    X_mor = enc.encode(df_mor.apply(serialize_row, axis=1).tolist(), batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    np.save(cache_val, X_val); np.save(cache_mor, X_mor)
    del enc; gc.collect()
X_new = np.vstack([X_val, X_mor])
assert len(names) == len(X_new)

# ── scores of the final model (as in 03 and 04) ──────────────────────────
scaler = StandardScaler().fit(X_train)
pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE).fit(scaler.transform(X_train))
clf = LogisticRegression(C=BEST_C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)
clf.fit(pca.transform(scaler.transform(X_train)), y)
probs = clf.predict_proba(pca.transform(scaler.transform(X_new)))[:, 1]

# ── three nearest labelled records in the raw embedding space ────────────
sims = cosine_similarity(X_new, X_train)
rows, lines = [], []
for i, (name, g) in enumerate(zip(names, group)):
    top3 = np.argsort(-sims[i])[:3]
    lines.append('=' * 90)
    lines.append(f"  {name} ({g})  |  P={probs[i]:.3f}  |  {'induction' if probs[i] >= OPT_THRESHOLD else 'inhibition'} at t={OPT_THRESHOLD}")
    lines.append('-' * 90)
    for rank, j in enumerate(top3, start=1):
        rec = df_raw.iloc[j]   # raw workbook descriptors (the model input replaces 'ND' by 'Unknown')
        rows.append({
            'Set': g, 'Plant': name, 'P': round(float(probs[i]), 3), 'Rank': rank,
            'Neighbour': str(rec['SPECIES']).strip(),
            'Neighbour_label': 'induction' if y[j] == 1 else 'inhibition',
            'Region': str(rec['REGION']).strip(), 'Used_part': str(rec['USED PART']).strip(),
            'Metabolites': str(rec['METABOLITE CONTENT']).strip()[:60],
            'Microorganism': str(rec['TESTED MICROORGANISME']).strip()[:45],
            'Cosine': round(float(sims[i, j]), 3),
        })
        r = rows[-1]
        lines.append(f"  {rank}. {r['Neighbour']:35s} [{r['Neighbour_label']}] cosine={r['Cosine']:.3f}")
        lines.append(f"     region: {r['Region']}  |  part: {r['Used_part']}")
        lines.append(f"     metabolites: {r['Metabolites']}")
        lines.append(f"     microorganism: {r['Microorganism']}")
    lines.append('')

table = pd.DataFrame(rows)
table.to_csv('results/nearest_neighbors_table_S3.csv', index=False)
with open('results/nearest_neighbors.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f"cosine similarity over all {len(names)} x {len(X_train)} pairs: {sims.min():.3f} to {sims.max():.3f}")
print(table.to_string(index=False))
print("\nSaved: results/nearest_neighbors_table_S3.csv, results/nearest_neighbors.txt")
