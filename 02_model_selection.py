"""
Step 2: model selection and final test evaluation.
One fixed stratified 80/20 split (seed 42). On the training set, 3-fold
stratified cross-validation is repeated over 5 fold shuffles; for every
classifier and hyperparameter setting the decision threshold maximizing the F1
score is searched on the validation fold. The best model (highest mean
optimal-threshold CV F1) is retrained on the full training set and evaluated
once on the held-out test set. Uses the embeddings cached by 01_encode_embeddings.py.
Outputs: results/model_selection_summary.csv (Figure 1),
results/model_selection_cv_details.csv, results/test_set_results.csv (Table 4).
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

PCA_VARIANCE = 0.95
CV_FOLDS = 3
TEST_SIZE = 0.2
SPLIT_SEED = 42  # single fixed split
CV_SEEDS = [42, 123, 456, 789, 2024]  # 5 fold shuffles


def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


os.makedirs('results', exist_ok=True)
y_all = np.load(os.path.join('cache', '_y_all.npy'))

classifiers = {
    'Logistic Regression': {
        'cls': lambda: LogisticRegression(solver='lbfgs', max_iter=2000),
        'params_list': [
            {'C': c, 'penalty': 'l2'}
            for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        ] + [
            {'C': c, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 2000}
            for c in [0.001, 0.01, 0.1, 1, 10, 100]
        ]
    },
    'Random Forest': {
        'cls': lambda: RandomForestClassifier(),
        'params_list': [
            {'n_estimators': n, 'max_depth': d, 'min_samples_split': s, 'min_samples_leaf': l}
            for n in [50, 100, 200]
            for d in [3, 5, 7, None]
            for s in [2, 5, 10]
            for l in [1, 2, 4]
        ]
    },
    'SVM-RBF': {
        'cls': lambda: SVC(probability=True, kernel='rbf'),
        'params_list': [
            {'C': c, 'gamma': g}
            for c in [0.1, 1, 10, 100]
            for g in ['scale', 'auto', 0.001, 0.01, 0.1]
        ]
    },
    'SVM-Linear': {
        'cls': lambda: SVC(probability=True, kernel='linear'),
        'params_list': [
            {'C': c}
            for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        ]
    },
    'XGBoost': {
        'cls': lambda: XGBClassifier(eval_metric='logloss'),
        'params_list': [
            {'n_estimators': n, 'max_depth': d, 'learning_rate': lr,
             'subsample': ss, 'colsample_bytree': cb}
            for n in [50, 100, 200]
            for d in [3, 5, 7]
            for lr in [0.01, 0.1, 0.3]
            for ss in [0.8, 1.0]
            for cb in [0.8, 1.0]
        ]
    },
}

embeddings = [
    'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'all-mpnet-base-v2',
]

# ── Fixed train/test split ───────────────────────────────────
print(f"Fixed train/test split (seed={SPLIT_SEED}, test_size={TEST_SIZE})")
split_indices = train_test_split(
    np.arange(len(y_all)), test_size=TEST_SIZE,
    random_state=SPLIT_SEED, stratify=y_all
)
train_idx, test_idx = split_indices
y_train = y_all[train_idx]
y_test = y_all[test_idx]
print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
print(f"  Train class balance: {y_train.mean():.2%} positive")
print(f"  Test  class balance: {y_test.mean():.2%} positive")

# ── Model selection (CV only on train set, 5 fold shuffles) ──
selection_results = []  # per (embedding, classifier, cv_seed)

for emb_name in embeddings:
    print(f"\n{'#'*60}")
    print(f"  Embedding: {emb_name}")
    print(f"{'#'*60}")

    X_all = np.load(os.path.join('cache', f'_X_all_{emb_name}.npy'))
    X_train_emb = X_all[train_idx]
    X_test_emb = X_all[test_idx]

    for cls_name, cls_cfg in classifiers.items():
        cv_f1_per_seed = []
        cv_t_per_seed = []
        best_params_per_seed = []

        for cv_seed in CV_SEEDS:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=cv_seed)
            best_cv_f1 = -1
            best_params = None
            best_mean_t = 0.5

            for params in cls_cfg['params_list']:
                fold_f1s = []
                fold_ts = []
                ok = True

                for tr_i, va_i in cv.split(X_train_emb, y_train):
                    X_tr = X_train_emb[tr_i]
                    X_va = X_train_emb[va_i]
                    y_tr = y_train[tr_i]
                    y_va = y_train[va_i]

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    pca = PCA(n_components=PCA_VARIANCE, random_state=42)
                    X_tr_p = pca.fit_transform(X_tr_s)
                    X_va_p = pca.transform(X_va_s)

                    try:
                        clf = cls_cfg['cls']()
                        safe_params = {k: v for k, v in params.items()
                                       if k in clf.get_params()}
                        clf.set_params(**safe_params)
                        clf.fit(X_tr_p, y_tr)
                        y_prob = clf.predict_proba(X_va_p)[:, 1]
                        t, f = find_optimal_threshold(y_va, y_prob)
                        fold_f1s.append(f)
                        fold_ts.append(t)
                    except Exception:
                        ok = False
                        break

                if not ok:
                    continue
                mf = np.mean(fold_f1s)
                if mf > best_cv_f1:
                    best_cv_f1 = mf
                    best_params = params
                    best_mean_t = np.mean(fold_ts)

            if best_params is not None:
                cv_f1_per_seed.append(best_cv_f1)
                cv_t_per_seed.append(best_mean_t)
                best_params_per_seed.append(best_params)

                selection_results.append({
                    'Embedding': emb_name,
                    'Classifier': cls_name,
                    'CV_Seed': cv_seed,
                    'CV_F1_OptT': best_cv_f1,
                    'Opt_Threshold': round(best_mean_t, 2),
                    'Best_Params': str(best_params),
                })

        if cv_f1_per_seed:
            mean_cv = np.mean(cv_f1_per_seed)
            std_cv = np.std(cv_f1_per_seed)
            print(f"    {cls_name:22s} CV_F1={mean_cv:.3f}±{std_cv:.3f} "
                  f"mean_t={np.mean(cv_t_per_seed):.2f}")

# ── Aggregate CV results for model selection ─────────────────
print("\n" + "=" * 70)
print("MODEL SELECTION (aggregated over 5 CV fold shuffles)")
print("=" * 70)

df_sel = pd.DataFrame(selection_results)
agg_sel = df_sel.groupby(['Embedding', 'Classifier']).agg(
    CV_F1_mean=('CV_F1_OptT', 'mean'),
    CV_F1_std=('CV_F1_OptT', 'std'),
    Opt_Threshold_mean=('Opt_Threshold', 'mean'),
).reset_index().sort_values('CV_F1_mean', ascending=False)

for _, row in agg_sel.iterrows():
    print(f"  {row['Embedding'][:12]:12s} | {row['Classifier']:22s} | "
          f"CV_F1={row['CV_F1_mean']:.3f}±{row['CV_F1_std']:.3f} | "
          f"t={row['Opt_Threshold_mean']:.2f}")

# ── Select best model ────────────────────────────────────────
best = agg_sel.iloc[0]
best_emb = best['Embedding']
best_cls = best['Classifier']
best_t = best['Opt_Threshold_mean']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_cls} + {best_emb}")
print(f"  CV F1 (opt-t): {best['CV_F1_mean']:.4f} ± {best['CV_F1_std']:.4f}")
print(f"  Optimal threshold: {best_t:.2f}")
print(f"{'='*70}")

# ── Find most frequent best params for the winning model ─────
best_rows = df_sel[(df_sel['Embedding'] == best_emb) &
                   (df_sel['Classifier'] == best_cls)]
most_common_params = best_rows['Best_Params'].mode().iloc[0]
print(f"  Most common hyperparams: {most_common_params}")

# ── Final evaluation on test set (ONCE) ──────────────────────
print(f"\n{'='*70}")
print("FINAL TEST SET EVALUATION (single evaluation, never seen during selection)")
print(f"{'='*70}")

X_all = np.load(os.path.join('cache', f'_X_all_{best_emb}.npy'))
X_train_emb = X_all[train_idx]
X_test_emb = X_all[test_idx]

# Parse params
import ast
final_params = ast.literal_eval(most_common_params)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train_emb)
X_te_s = scaler.transform(X_test_emb)
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_tr_p = pca.fit_transform(X_tr_s)
X_te_p = pca.transform(X_te_s)

cls_cfg = classifiers[best_cls]
clf = cls_cfg['cls']()
safe_params = {k: v for k, v in final_params.items() if k in clf.get_params()}
clf.set_params(**safe_params)
clf.fit(X_tr_p, y_train)

y_prob_test = clf.predict_proba(X_te_p)[:, 1]
y_pred_opt = (y_prob_test >= best_t).astype(int)

test_acc = accuracy_score(y_test, y_pred_opt)
test_f1 = f1_score(y_test, y_pred_opt, zero_division=0)
test_prec = precision_score(y_test, y_pred_opt, zero_division=0)
test_rec = recall_score(y_test, y_pred_opt, zero_division=0)
test_ap = average_precision_score(y_test, y_prob_test)
test_auc = roc_auc_score(y_test, y_prob_test)

print(f"  Accuracy:   {test_acc:.4f}")
print(f"  F1-score:   {test_f1:.4f}")
print(f"  Precision:  {test_prec:.4f}")
print(f"  Recall:     {test_rec:.4f}")
print(f"  AP:         {test_ap:.4f}")
print(f"  AUC:        {test_auc:.4f}")
print(f"  Threshold:  {best_t:.2f}")

# ── Save ─────────────────────────────────────────────────────
df_sel.to_csv('results/model_selection_cv_details.csv', index=False)
agg_sel.to_csv('results/model_selection_summary.csv', index=False)

test_results = pd.DataFrame([{
    'Embedding': best_emb,
    'Classifier': best_cls,
    'Params': most_common_params,
    'Threshold': round(best_t, 2),
    'Test_Accuracy': test_acc,
    'Test_F1': test_f1,
    'Test_Precision': test_prec,
    'Test_Recall': test_rec,
    'Test_AP': test_ap,
    'Test_AUC': test_auc,
}])
test_results.to_csv('results/test_set_results.csv', index=False)

print("\nSaved: results/model_selection_cv_details.csv")
print("Saved: results/model_selection_summary.csv")
print("Saved: results/test_set_results.csv")
