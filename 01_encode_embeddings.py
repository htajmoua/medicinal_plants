"""
Step 1: serialize the 183 labelled records and encode them with the two text
encoders (BiomedBERT and all-mpnet-base-v2). Embeddings and labels are cached in
cache/ and reused by the following scripts.
"""
import pandas as pd
import numpy as np
import os, warnings, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer

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

data_file = os.path.join('data', 'data_med_plants.xlsx')
df = pd.read_excel(data_file, sheet_name='Dataset(litterature)', header=1)
df.columns = df.columns.str.strip()
for col in feature_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').replace('ND', 'Unknown').replace('', 'Unknown').astype(str)
df = df.dropna(subset=['TARGET'])
df['TARGET'] = df['TARGET'].astype(int)

texts = df.apply(serialize_row, axis=1).tolist()
y = df['TARGET'].values
os.makedirs('cache', exist_ok=True)
np.save(os.path.join('cache', '_y_all.npy'), y)
print(f"Total samples: {len(texts)}, dist: {dict(zip(*np.unique(y, return_counts=True)))}")

for model_name in [
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'all-mpnet-base-v2',
]:
    short = model_name.split('/')[-1]
    print(f"\nEncoding ALL data with {short}...")
    encoder = SentenceTransformer(model_name, device='cpu')
    X_all = encoder.encode(texts, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
    np.save(os.path.join('cache', f'_X_all_{short}.npy'), X_all)
    print(f"  Saved: shape {X_all.shape}")
    del encoder, X_all
    gc.collect()

print("\n✓ All embeddings saved.")
