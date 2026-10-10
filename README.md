# Imputation, Decision Trees & Ensemble Learning

A hands-on practice project with **scikit-learn** covering missing-value imputation for high-dimensional data, decision-tree tuning and interpretation, and a comparison of tree ensembles (bagging, pasting, random forest and AdaBoost).

## What's covered

| # | Topic | Dataset | Key idea |
|---|-------|---------|----------|
| 1 | **Imputation + sparse logistic regression** | Gene expression (172 × ~21,000) | Median imputation without leakage, L1 for feature selection when p ≫ n |
| 2 | **Decision trees** | Iris (150 × 4) | Effect of `max_depth`, reading a fitted tree to explain a prediction |
| 3 | **Ensemble learning** | Iris, 2 features (150 × 2) | Bagging vs pasting vs random forest vs AdaBoost across tree sizes |

👉 Notebook: [`notebooks/imputation_trees_ensembles.ipynb`](notebooks/imputation_trees_ensembles.ipynb)

## Results at a glance

**L1 logistic regression with median imputation (accuracy, test set)**

| C | Accuracy | Genes selected |
|---|---|---|
| 0.05 | 0.907 | 10 / 21,050 |
| **10** | **1.000** | 220 / 21,050 |

**Decision tree depth (accuracy, test set)**

| max_depth | Accuracy |
|---|---|
| 1 | 0.667 |
| 2 | 0.933 |
| **3** | **0.967** |
| 4 | 0.933 |

**Ensembles on two Iris features (accuracy, test set)**

| min_samples_leaf | Bagging | Pasting | Random Forest | AdaBoost |
|---|---|---|---|---|
| 2 (deep trees) | **0.733** | **0.667** | **0.800** | **0.733** |
| 5 | 0.633 | 0.667 | 0.800 | 0.667 |
| 10 | 0.633 | 0.567 | 0.700 | 0.733 |
| 20 (shallow trees) | 0.633 | 0.633 | 0.700 | 0.733 |

## Key learnings

- **Impute instead of dropping when samples are scarce.** Median imputation keeps all 172 samples.
- **Imputers are fitted like scalers:** learn the statistics on the training split only, then apply them to the test split.
- **The L1 strength (`C`) trades sparsity for accuracy:** 10 genes give 91% accuracy, and 220 genes give 100%.
- **Tree depth is a bias–variance dial.** Depth 1 underfits, and depth 4 starts to overfit.
- **Decision trees are explainable.** You can follow any prediction by hand through three threshold checks.
- **Averaging ensembles (bagging, pasting, random forest) like deep trees,** because averaging cancels out the variance of the individual trees.
- **Random forests decorrelate trees** with random feature subsets, and they gave the best ensemble result here.
- **AdaBoost reduces bias in sequence,** so it does not depend on deep trees in the same way.

## Tech stack

Python · NumPy · pandas · scikit-learn · Matplotlib · Jupyter

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/imputation_trees_ensembles.ipynb
```

See [`data/README.md`](data/README.md) for dataset details. Iris is loaded automatically by scikit-learn.
