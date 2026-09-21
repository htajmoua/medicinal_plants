"""
Step 5: Figure 1. Cross-validated F1 scores (optimal threshold) of the five
classifiers with BiomedBERT and all-mpnet-base-v2 embeddings, mean and standard
deviation over the five fold shuffles. Reads results/model_selection_summary.csv
written by 02_model_selection.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
df = pd.read_csv('results/model_selection_summary.csv')
label = {'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'BiomedBERT (domain-specific)',
         'all-mpnet-base-v2': 'all-mpnet-base-v2 (general-purpose)'}
color = {'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': '#d6604d',
         'all-mpnet-base-v2': '#4393c3'}
bio = df[df['Embedding'].str.startswith('BiomedNLP')].sort_values('CV_F1_mean', ascending=False)
classifiers = bio['Classifier'].tolist()
x = np.arange(len(classifiers))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, emb in enumerate(label):
    sub = df[df['Embedding'] == emb].set_index('Classifier').loc[classifiers]
    ax.bar(x + (k - 0.5) * width, sub['CV_F1_mean'], width, yerr=sub['CV_F1_std'], capsize=3,
           color=color[emb], edgecolor='k', linewidth=0.5, label=label[emb])
    for xi, (m, s) in enumerate(zip(sub['CV_F1_mean'], sub['CV_F1_std'])):
        ax.text(xi + (k - 0.5) * width, m + s + 0.006, f'{m:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(classifiers, fontsize=9)
ax.set_ylabel('Cross-validated F1 score (optimal threshold)')
ax.set_ylim(0.6, 0.85)
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/figure_model_selection.png', dpi=200)
plt.savefig('results/figure_model_selection.pdf')
print('Saved: results/figure_model_selection.png/.pdf')
