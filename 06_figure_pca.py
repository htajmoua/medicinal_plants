"""
Step 6: PCA graph of the BiomedBERT embedding space (added during revision).
Panel A: PC1 x PC2 of the standardized 768-d embeddings (PCA fitted on the 183
labelled samples) with the 4 validation plants and the 11 Moroccan plants
projected onto the same axes. Panel B: cumulative explained variance (95% cut).
Panel C: predicted probability ranking of the 15 new plants with the 0.39
threshold and bootstrap 95% intervals.
Requires cache/_y_all.npy (step 1) and the embeddings and bootstrap results
written by 04_prediction_stability.py.
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

X_tr = np.load('cache/_X_final_train_BiomedBERT.npy'); X_val = np.load('cache/_X_final_val_BiomedBERT.npy'); X_mor = np.load('cache/_X_final_moroccan_BiomedBERT.npy')
y = np.load('cache/_y_all.npy')
stab = pd.read_csv('results/prediction_stability_bootstrap.csv')
names = stab['Species'].tolist(); group = stab['Group'].tolist(); p = stab['P_point'].values
assert len(names) == len(X_val) + len(X_mor)

sc = StandardScaler().fit(X_tr)
pca_full = PCA(random_state=42).fit(sc.transform(X_tr))
cum = np.cumsum(pca_full.explained_variance_ratio_); n95 = int(np.searchsorted(cum, 0.95) + 1)
Z_tr = pca_full.transform(sc.transform(X_tr)); Z_new = pca_full.transform(sc.transform(np.vstack([X_val, X_mor])))
ev = pca_full.explained_variance_ratio_
print(f"PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, PC1+PC2={ev[:2].sum():.1%}; components for 95% variance = {n95}")

fig = plt.figure(figsize=(14, 5.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 1.0])
ax = fig.add_subplot(gs[0])
ax.scatter(Z_tr[y == 0, 0], Z_tr[y == 0, 1], s=22, c='#4393c3', alpha=0.55, label='training, inhibition (class 0, n=101)')
ax.scatter(Z_tr[y == 1, 0], Z_tr[y == 1, 1], s=22, c='#d6604d', alpha=0.55, label='training, induction (class 1, n=82)')
for i, (n, g) in enumerate(zip(names, group)):
    if g == 'validation':
        ax.scatter(Z_new[i, 0], Z_new[i, 1], marker='s', s=48, c='#1b7837', edgecolor='k', zorder=5)
    else:
        bold = n in ('Capparis spinosa', 'Ephedra fragilis')
        ax.scatter(Z_new[i, 0], Z_new[i, 1], marker='*', s=150 if bold else 95, c='#ffd92f' if not bold else '#e7298a', edgecolor='k', zorder=6)
# label placement: crowded points get leader lines to a label column on the right
xmax = max(Z_tr[:, 0].max(), Z_new[:, 0].max())
order_y = np.argsort(-Z_new[:, 1])
placed = []
for i in order_y:
    n = names[i]; x0, y0 = Z_new[i, 0], Z_new[i, 1]
    crowded = any(np.hypot(x0 - Z_new[j, 0], y0 - Z_new[j, 1]) < 3.5 for j in range(len(names)) if j != i)
    bold = n in ('Capparis spinosa', 'Ephedra fragilis')
    if crowded:
        yl = y0
        while any(abs(yl - p_) < 1.6 for p_ in placed):
            yl -= 1.6
        placed.append(yl)
        ax.annotate(n, (x0, y0), xytext=(xmax + 3, yl), textcoords='data', fontsize=6.5, va='center',
                    fontweight='bold' if bold else 'normal',
                    arrowprops=dict(arrowstyle='-', color='0.4', lw=0.6, shrinkA=0, shrinkB=2))
    else:
        ax.annotate(n, (x0, y0), textcoords='offset points', xytext=(4, 3), fontsize=6.5,
                    fontweight='bold' if bold else 'normal')
ax.set_xlim(Z_tr[:, 0].min() - 2, xmax + 22)
ax.scatter([], [], marker='s', s=48, c='#1b7837', edgecolor='k', label='validation set (n=4)')
ax.scatter([], [], marker='*', s=95, c='#ffd92f', edgecolor='k', label='Moroccan prediction set (n=11)')
ax.scatter([], [], marker='*', s=150, c='#e7298a', edgecolor='k', label='experimentally tested (C. spinosa, E. fragilis)')
ax.set_xlabel(f'PC1 ({ev[0]:.1%} of variance)'); ax.set_ylabel(f'PC2 ({ev[1]:.1%} of variance)')
ax.set_title('A. PCA of BiomedBERT embeddings (fitted on the 183 labelled samples)', fontsize=9.5, loc='left')
ax.legend(fontsize=6.5, loc='best')
ax = fig.add_subplot(gs[1])
ax.plot(np.arange(1, len(cum) + 1), cum, color='#2166ac'); ax.axhline(0.95, color='r', ls='--', lw=1); ax.axvline(n95, color='r', ls=':', lw=1)
ax.annotate(f'{n95} components\n= 95% variance', (n95, 0.95), textcoords='offset points', xytext=(8, -30), fontsize=7.5)
ax.set_xlabel('number of principal components'); ax.set_ylabel('cumulative explained variance'); ax.set_ylim(0, 1.02)
ax.set_title('B. Explained variance', fontsize=9.5, loc='left')
ax = fig.add_subplot(gs[2])
order = np.argsort(p)
cols = ['#1b7837' if group[i] == 'validation' else ('#e7298a' if names[i] in ('Capparis spinosa', 'Ephedra fragilis') else '#fdb863') for i in order]
ax.barh(np.arange(len(order)), p[order], color=cols, edgecolor='k', lw=0.4)
ax.errorbar(p[order], np.arange(len(order)), xerr=[p[order] - stab['P_boot_2.5%'].values[order], stab['P_boot_97.5%'].values[order] - p[order]], fmt='none', ecolor='k', elinewidth=0.8, capsize=2)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([f"{names[i]}{' (val.)' if group[i]=='validation' else ''}" for i in order], fontsize=7)
ax.axvline(0.39, color='r', ls='--', lw=1.2); ax.text(0.395, len(order) - 0.6, 't = 0.39', color='r', fontsize=7)
ax.set_xlabel('P(class = 1) with bootstrap 95% CI'); ax.set_xlim(0, 1)
ax.set_title('C. Model ranking of the 15 new plants', fontsize=9.5, loc='left')
plt.tight_layout(); plt.savefig('results/figure_pca_embedding_space.png', dpi=200); plt.savefig('results/figure_pca_embedding_space.pdf')
print('Saved: results/figure_pca_embedding_space.png/.pdf')
