"""
Step 7: nearest-neighbour analysis. For each Moroccan plant, the three closest
training records in BiomedBERT embedding space (cosine similarity), with their
labels and descriptors. Uses the embeddings cached by 01_encode_embeddings.py.
Output: results/nearest_neighbors.txt
"""
import pandas as pd, numpy as np, os, re, warnings, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ALL_FEATURES = ['SPECIES','REGION','USED PART','METABOLITE CONTENT',
                'TESTED MICROORGANISME','EFFECTIVE CONCENTRATION']
LABELS = {'SPECIES':'Species','REGION':'Region','USED PART':'Part',
          'METABOLITE CONTENT':'Metabolites','TESTED MICROORGANISME':'Microorganism',
          'EFFECTIVE CONCENTRATION':'Concentration'}

def serialize_row(row):
    return '. '.join(f'{LABELS[c]}: {row[c]}' for c in ALL_FEATURES if c in row.index) + '.'

def csn(n):
    s = re.sub(r'\([^)]*\)','',str(n).strip())
    s = re.sub(r'\d+','',s)
    return re.sub(r'\s+',' ',s).strip()

def cc(v):
    return re.sub(r'\s*\(?\s*[Cc][Ll][Aa][Ss][Ss][Ee]\s*\d+\s*\)?','',str(v).strip()).strip()

# ── Load train data ──
os.makedirs('results', exist_ok=True)
data_file = os.path.join('data', 'data_med_plants.xlsx')
df_train = pd.read_excel(data_file, sheet_name='Dataset(litterature)', header=1)
df_train.columns = df_train.columns.str.strip()
for c in ALL_FEATURES:
    if c in df_train.columns:
        df_train[c] = df_train[c].fillna('Unknown').replace('ND','Unknown').replace('','Unknown').astype(str)
if 'SPECIES' in df_train.columns:
    df_train['SPECIES'] = df_train['SPECIES'].apply(csn)
if 'EFFECTIVE CONCENTRATION' in df_train.columns:
    df_train['EFFECTIVE CONCENTRATION'] = df_train['EFFECTIVE CONCENTRATION'].apply(cc)
df_train = df_train.dropna(subset=['TARGET'])
df_train['TARGET'] = df_train['TARGET'].astype(int)
y_all = df_train['TARGET'].values

# ── Load Moroccan plants ──
df_pred = pd.read_excel(data_file, sheet_name='Prediction', header=1)
df_pred = df_pred.dropna(how='all')
df_pred.columns = df_pred.columns.str.strip()
for c in ALL_FEATURES:
    if c not in df_pred.columns:
        df_pred[c] = 'Unknown'
    else:
        df_pred[c] = df_pred[c].fillna('Unknown').replace('ND','Unknown').replace('','Unknown').astype(str)
if 'SPECIES' in df_pred.columns:
    df_pred['SPECIES'] = df_pred['SPECIES'].apply(csn)
if 'EFFECTIVE CONCENTRATION' in df_pred.columns:
    df_pred['EFFECTIVE CONCENTRATION'] = df_pred['EFFECTIVE CONCENTRATION'].apply(cc)
df_moroccan = df_pred[df_pred['SPECIES'].str.len() > 2].copy()

print(f"Train: {len(df_train)} | Moroccan: {len(df_moroccan)}")

# ── Load / compute embeddings ──
X_train_emb = np.load('cache/_X_all_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext.npy')

emb_pred_file = 'cache/_X_moroccan_BiomedBERT.npy'
if os.path.exists(emb_pred_file):
    print("Loading cached Moroccan embeddings...")
    X_pred_emb = np.load(emb_pred_file)
else:
    print("Loading BiomedBERT model (this takes a few minutes on CPU)...", flush=True)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', device='cpu')
    print("Model loaded, encoding...", flush=True)
    pred_texts = df_moroccan.apply(serialize_row, axis=1).tolist()
    X_pred_emb = encoder.encode(pred_texts, show_progress_bar=True, batch_size=8,
                                convert_to_numpy=True)
    np.save(emb_pred_file, X_pred_emb)
    print(f"Saved embeddings to {emb_pred_file}")
    del encoder; gc.collect()

# ── Predictions ──
scaler = StandardScaler()
X_s = scaler.fit_transform(X_train_emb)
pca = PCA(n_components=0.95, random_state=42)
X_p = pca.fit_transform(X_s)
X_pp = pca.transform(scaler.transform(X_pred_emb))
clf = LogisticRegression(C=0.001, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
clf.fit(X_p, y_all)
probs = clf.predict_proba(X_pp)[:, 1]

# ── Cosine similarities ──
sims = cosine_similarity(X_pred_emb, X_train_emb)
order = np.argsort(-probs)

# ── Write results to file ──
out = []
for i in order:
    sp = str(df_moroccan.iloc[i]['SPECIES'])
    prob = probs[i]
    pred = 'POSITIVE' if prob >= 0.39 else 'NEGATIVE'
    top3 = np.argsort(-sims[i])[:3]
    out.append(f"{'='*90}")
    out.append(f"  {sp}  |  P={prob:.3f}  |  {pred}")
    out.append(f"{'─'*90}")
    for rank, j in enumerate(top3):
        s = sims[i][j]
        t_sp = str(df_train.iloc[j]['SPECIES'])
        t_tgt = int(df_train.iloc[j]['TARGET'])
        tl = 'IMMUNO+' if t_tgt == 1 else 'IMMUNO-'
        t_reg = str(df_train.iloc[j]['REGION'])
        t_part = str(df_train.iloc[j]['USED PART'])
        t_met = str(df_train.iloc[j]['METABOLITE CONTENT'])[:80]
        t_mic = str(df_train.iloc[j]['TESTED MICROORGANISME'])[:50]
        out.append(f"  {rank+1}. {t_sp:35s} [{tl}] sim={s:.3f}")
        out.append(f"     region: {t_reg}  |  part: {t_part}")
        out.append(f"     metabolites: {t_met}")
        out.append(f"     microorganism: {t_mic}")
    out.append("")

text = '\n'.join(out)
with open('results/nearest_neighbors.txt', 'w') as f:
    f.write(text)

print(text)
print("\nSaved: results/nearest_neighbors.txt")
