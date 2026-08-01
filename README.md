# Breast Cancer Wisconsin (Diagnostic) - Supervised Learning Classification

A supervised machine learning project that classifies breast tumors as **malignant** or **benign**
based on cell nuclei measurements taken from digitized images of fine needle aspirate (FNA) biopsies.

## Objective

Implement and compare multiple supervised learning algorithms on a real-world medical dataset,
covering the complete ML workflow: data preprocessing, exploratory data analysis (EDA), model
building, evaluation, and documentation.

## Dataset

- **Source:** UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Dataset,
  accessed via `sklearn.datasets.load_breast_cancer()`.
- **Samples:** 569 (after preprocessing)
- **Features:** 30 numeric features describing cell nuclei (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension - each
  reported as mean, standard error, and "worst"/largest value).
- **Target:** Binary - `0 = Malignant`, `1 = Benign`
- **Problem Type:** Binary Classification

## Repository Structure

```
Breast_Cancer_Classification/
├── Dataset/
│   ├── breast_cancer_data.csv       # Raw dataset used for modeling
│   ├── summary_statistics.csv       # Descriptive statistics
│   └── model_comparison.csv         # Final model evaluation results
├── Notebook/
│   └── Breast_Cancer_Classification.ipynb   # Full, executed analysis notebook
├── Images/
│   ├── target_countplot.png
│   ├── correlation_heatmap.png
│   ├── histograms.png
│   ├── pairplot.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── model_comparison_bar.png
│   └── feature_importance.png
├── README.md
├── requirements.txt
└── Report.pdf
```

## Methodology

1. **Dataset Understanding** - shape, data types, target distribution, problem type.
2. **Preprocessing** - missing value / duplicate checks, IQR-based outlier scan, feature
   scaling with `StandardScaler`, 80/20 stratified train-test split.
3. **EDA** - summary statistics, correlation heatmap, histograms, count plot, pairplot, and
   five written observations.
4. **Model Building** - six algorithms trained: Logistic Regression, Decision Tree, Random
   Forest, KNN, Naive Bayes, and SVM (RBF kernel).
5. **Evaluation** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC for
   every model, plus a side-by-side comparison.
6. **Conclusion** - best model selection, justification, challenges faced, and future
   improvement ideas.

## Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9825   | 0.9861    | 0.9861 | 0.9861   | 0.9954  |
| SVM                  | 0.9825   | 0.9861    | 0.9861 | 0.9861   | 0.9950  |
| KNN                  | 0.9737   | 0.9600    | 1.0000 | 0.9796   | 0.9884  |
| Random Forest        | 0.9561   | 0.9589    | 0.9722 | 0.9655   | 0.9932  |
| Naive Bayes          | 0.9298   | 0.9444    | 0.9444 | 0.9444   | 0.9868  |
| Decision Tree        | 0.9123   | 0.9559    | 0.9028 | 0.9286   | 0.9157  |

**Best Model: Logistic Regression**, with ~98.2% test accuracy and a ROC-AUC of ~0.995.

## How to Run

```bash
git clone <this-repo-url>
cd Breast_Cancer_Classification
pip install -r requirements.txt
jupyter notebook Notebook/Breast_Cancer_Classification.ipynb
```

## Author

Individual assignment submission - Supervised Learning coursework.
