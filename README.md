# Machine learning prediction of the immunomodulatory activity of medicinal plants

Code accompanying the manuscript:

Assouab A., Tajmouati H., Abdou-Allah F., Akarid K. *Unveiling the Immunomodulatory Potential of Medicinal Plants: A Machine Learning Framework Coupled with Experimental Validation.* Submitted to *Life* (MDPI), 2026.

The task is to predict, from literature-derived descriptors of a plant extract (species, geographical region, plant part, metabolite content, tested microorganism, effective concentration), whether the extract was reported to induce (class 1) or inhibit (class 0) the expression of iNOS and Th1 cytokines. Each record is serialized into one sentence, encoded with a frozen biomedical language model (BiomedBERT), reduced by PCA and classified with a regularized logistic regression. The selected model is then applied to 4 external validation plants and to 11 Moroccan medicinal plants, two of which (*Capparis spinosa* and *Ephedra fragilis*) were tested in vitro.

## Data

The dataset (183 labelled records, 4 validation plants and 11 Moroccan plants, with the literature reference of every record) is deposited on Zenodo:

Assouab A., Akarid K. (2026). *Dataset of Medicinal Plants with Immunomodulatory Properties: Phytotherapeutic Attributes and Binary Classification Labels.* Zenodo. https://doi.org/10.5281/zenodo.19451645

Download the workbook into `data/` before running the scripts:

```bash
mkdir -p data
curl -L -o data/data_med_plants.xlsx "https://zenodo.org/records/19451646/files/data_med_plants.xlsx?download=1"
```

The scripts read three sheets of the workbook: `Dataset(litterature)` (training data), `validation` and `Prediction`.

## Installation

Python 3.11 is required. The versions pinned in `requirements.txt` are the ones used for the manuscript.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The BiomedBERT checkpoint (about 440 MB) is downloaded from the Hugging Face Hub the first time it is used. All computations run on CPU.

## Pipeline

Run the scripts from the repository root, in numerical order. Intermediate embeddings are cached in `cache/`, all outputs are written to `results/`.

| Script | What it does | Outputs in `results/` | Manuscript |
|---|---|---|---|
| `01_encode_embeddings.py` | Serializes the 183 records and encodes them with BiomedBERT and all-mpnet-base-v2 | (`cache/`) | Sections 2.2, 2.3 |
| `02_model_selection.py` | Fixed stratified 80/20 split, 3-fold cross-validation repeated over 5 fold shuffles with optimal-threshold F1, grid search for 5 classifiers, single evaluation on the held-out test set | `model_selection_summary.csv`, `model_selection_cv_details.csv`, `test_set_results.csv` | Figure 1, Table 4 |
| `03_predict_new_plants.py` | Retrains the selected model on all 183 records and scores the validation and Moroccan plants at threshold 0.39 | `predictions_validation_set.csv`, `predictions_moroccan_plants.csv` | Tables 5, 6 |
| `04_prediction_stability.py` | Bootstrap stability of the scores, threshold sensitivity, calibration of out-of-fold probabilities, Tables 5 and 6 with reference labels and experimental conditions | `prediction_stability_bootstrap.csv`, `threshold_sensitivity.csv`, `calibration_*.csv`, `table5_*.csv`, `table6_*.csv`, `figure_stability_calibration.png` | Tables 5-6, Tables S1-S2, Figure S1 |
| `05_figure_model_selection.py` | Bar chart of the cross-validated F1 scores | `figure_model_selection.png` | Figure 1 |
| `06_figure_pca.py` | PCA of the embedding space with the new plants projected, explained variance, score ranking | `figure_pca_embedding_space.png` | Figure 2 |
| `07_nearest_neighbors.py` | Three closest training records of each of the 15 unlabelled plants (4 validation, 11 Moroccan) in embedding space, with cosine similarities | `nearest_neighbors_table_S3.csv`, `nearest_neighbors.txt` | Table S3 |

Script 02 is the longest step because it evaluates the full hyperparameter grids of the five classifiers for both encoders. The other scripts run in a few minutes on a laptop.

## Model details

- Text encoder: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (Hugging Face Hub, revision `e1354b7a`), loaded with sentence-transformers, mean pooling over the token embeddings, 768 dimensions. The weights are frozen: the encoder is used as a fixed feature extractor and is never fine-tuned.
- Tokenizer: WordPiece, uncased, maximum sequence length 512. Serialized records are 26 to 173 tokens long, so no truncation occurs.
- General-purpose baseline encoder: `sentence-transformers/all-mpnet-base-v2`.
- Downstream pipeline: StandardScaler, PCA keeping 95% of the variance (91 components on the full training set), classifier. Scaler and PCA are fitted inside each cross-validation fold.
- Selected model: logistic regression, L2 penalty, C = 0.001, decision threshold 0.39 (mean of the optimal thresholds over the five fold shuffles).
- Random seeds: train/test split 42; fold shuffles 42, 123, 456, 789 and 2024; PCA and classifiers 42; bootstrap 42.

## Results

`results/` contains the outputs obtained with the pinned environment. The values reported in the manuscript can be checked against `test_set_results.csv` (Table 4), `predictions_validation_set.csv` (Table 5), `predictions_moroccan_plants.csv` (Table 6) and `model_selection_summary.csv` (Figure 1). The files produced by scripts 04 and 06 were added during the revision of the manuscript.

Reproducibility note: the logistic regression results, the selected model, its threshold and the test set metrics are deterministic and were reproduced exactly with a fresh run of the whole pipeline. In `02_model_selection.py` the random forest and the two SVMs (Platt scaling of the probabilities) are not seeded, so their cross-validated F1 scores in `model_selection_summary.csv` can move by up to about 0.01 between runs without changing the ranking of the encoders or the selected model.

## License

The code is released under the MIT License (see `LICENSE`). The dataset is distributed by its authors on Zenodo under the Creative Commons Attribution 4.0 license.
