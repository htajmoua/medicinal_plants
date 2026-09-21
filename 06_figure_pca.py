"""
Step 6: principal component map of the BiomedBERT embedding space (Figure 2 of
the manuscript, added during revision).
Panel A: PC1 x PC2 of the standardized 768-d embeddings (PCA fitted on the 183
labelled records) with the 4 validation plants and the 11 Moroccan plants
projected onto the same axes. Panel B: cumulative explained variance (95% cut).
Panel C: model scores of the 15 unlabelled plants with the 0.39 threshold and
bootstrap 95% intervals.
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

plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
fig = plt.figure(figsize=(7.2, 6.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[0.8, 1.2], hspace=0.45, wspace=0.62,
                      left=0.06, right=0.99, top=0.95, bottom=0.075)
gsA = gs[:, 0].subgridspec(2, 1, height_ratios=[1, 0.16], hspace=0.22)
# ---- A: PC1 x PC2 ----
ax = fig.add_subplot(gsA[0])
ax.scatter(Z_tr[y == 0, 0], Z_tr[y == 0, 1], s=14, c='#4393c3', alpha=0.55, lw=0, label='training, inhibition (class 0, n=101)')
ax.scatter(Z_tr[y == 1, 0], Z_tr[y == 1, 1], s=14, c='#d6604d', alpha=0.55, lw=0, label='training, induction (class 1, n=82)')
tested = ('Capparis spinosa', 'Ephedra fragilis')
# numbering follows Tables 5 and 6: validation plants then Moroccan plants, each by decreasing score
order_num = [i for i in sorted(range(len(names)), key=lambda i: (group[i] != 'validation', -p[i]))]
num = {names[i]: k + 1 for k, i in enumerate(order_num)}
offsets = [(8, 8), (8, -13), (-15, 8), (-15, -13)]       # spread the numbers of overlapping markers
done = []
for i, (n, g) in enumerate(zip(names, group)):
    x0, y0 = Z_new[i, 0], Z_new[i, 1]
    bold = n in tested
    if g == 'validation':
        ax.scatter(x0, y0, marker='s', s=30, c='#1b7837', edgecolor='k', lw=0.5, zorder=5)
    else:
        ax.scatter(x0, y0, marker='*', s=110 if bold else 70, c='#e7298a' if bold else '#ffd92f', edgecolor='k', lw=0.5, zorder=6)
    near = [j for j in range(len(names)) if j != i and np.hypot(x0 - Z_new[j, 0], y0 - Z_new[j, 1]) < 3.5]
    k = sum(1 for j in done if j in near)
    done.append(i)
    off = offsets[k % 4] if near else (4, 3)
    ax.annotate(str(num[n]), (x0, y0), textcoords='offset points', xytext=off, fontsize=6.5, zorder=7,
                fontweight='bold' if bold else 'normal', ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color='0.3', lw=0.4, shrinkA=0, shrinkB=3) if near else None)
ax.set_xlim(Z_tr[:, 0].min() - 2, max(Z_tr[:, 0].max(), Z_new[:, 0].max()) + 3)
ax.set_ylim(Z_tr[:, 1].min() - 2, Z_tr[:, 1].max() + 9)
ax.scatter([], [], marker='s', s=30, c='#1b7837', edgecolor='k', lw=0.5, label='validation set (n=4)')
ax.scatter([], [], marker='*', s=70, c='#ffd92f', edgecolor='k', lw=0.5, label='Moroccan prediction set (n=11)')
ax.scatter([], [], marker='*', s=110, c='#e7298a', edgecolor='k', lw=0.5, label='experimentally tested (C. spinosa, E. fragilis)')
ax.set_xlabel(f'PC1 ({ev[0]:.1%} of variance)'); ax.set_ylabel(f'PC2 ({ev[1]:.1%} of variance)')
ax.set_title('(A) PCA of BiomedBERT embeddings, fitted on the 183 labelled records', loc='left')
ax.legend(loc='upper left', frameon=True, framealpha=0.9, handlelength=1.2, borderpad=0.4)
# key to the numbered markers, below the axes, two columns
axk = fig.add_subplot(gsA[1]); axk.axis('off')
lines = [f"{num[names[i]]:>2}  {names[i]}{' (val.)' if group[i] == 'validation' else ''}" for i in order_num]
axk.text(0.0, 1.0, '\n'.join(lines[:8]), transform=axk.transAxes, fontsize=5.8, va='top', ha='left', linespacing=1.3)
axk.text(0.52, 1.0, '\n'.join(lines[8:]), transform=axk.transAxes, fontsize=5.8, va='top', ha='left', linespacing=1.3)
# ---- B: explained variance ----
ax = fig.add_subplot(gs[0, 1])
ax.plot(np.arange(1, len(cum) + 1), cum, color='#2166ac', lw=1.2); ax.axhline(0.95, color='r', ls='--', lw=0.8); ax.axvline(n95, color='r', ls=':', lw=0.8)
ax.text(n95 - 6, 0.70, f'{n95} components\n= 95% variance', ha='right', va='top', fontsize=6.5)
ax.set_xlabel('number of principal components'); ax.set_ylabel('cumulative explained variance'); ax.set_ylim(0, 1.02)
ax.set_title('(B) Explained variance', loc='left')
# ---- C: ranking ----
ax = fig.add_subplot(gs[1, 1])
order = np.argsort(p)
cols = ['#1b7837' if group[i] == 'validation' else ('#e7298a' if names[i] in ('Capparis spinosa', 'Ephedra fragilis') else '#fdb863') for i in order]
ax.barh(np.arange(len(order)), p[order], color=cols, edgecolor='k', lw=0.3, height=0.7)
ax.errorbar(p[order], np.arange(len(order)), xerr=[p[order] - stab['P_boot_2.5%'].values[order], stab['P_boot_97.5%'].values[order] - p[order]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.5)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([f"{names[i]}{' (val.)' if group[i]=='validation' else ''}" for i in order], fontsize=6)
ax.axvline(0.39, color='r', ls='--', lw=0.9); ax.set_ylim(-1.4, len(order) - 0.4); ax.text(0.405, -1.05, 't = 0.39', color='r', fontsize=6, va='center')
ax.set_xlabel('P(class = 1) with bootstrap 95% CI'); ax.set_xlim(0, 1)
ax.set_title('(C) Model scores of the 15 unlabelled plants', loc='left')
plt.savefig('results/figure_pca_embedding_space.png', dpi=300); plt.savefig('results/figure_pca_embedding_space.pdf')
print('Saved: results/figure_pca_embedding_space.png/.pdf')
