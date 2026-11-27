# ShopSmart – E-commerce Purchase Prediction

A supervised machine learning project that predicts whether an online shopping session ends in a **purchase**, using a pruned Decision Tree built with Python and scikit-learn.

## Problem Statement

Most visitors to an online store leave without buying. If a store can predict which sessions are likely to convert, it can focus offers and support on those visitors. This project builds a **binary classifier** that predicts the `Revenue` target (purchase or no purchase) from session behaviour.

## Dataset

- **File:** `shop_smart_ecommerce.csv`
- **Size:** 12,330 sessions and 18 columns
- **Target:** `Revenue` (whether the session ended in a purchase)
- **Class imbalance:** only about 16% of sessions end in a purchase, so **F1 score** was used instead of plain accuracy
- **Benchmark to beat:** F1 = 0.55

## Approach

1. **Data cleaning:** removed 125 duplicate rows (12,205 sessions left)
2. **EDA:** explored session behaviour and how it relates to purchases
3. **Train-test split:** stratified 80/20 split, so both sets keep the same purchase ratio (2,441 test sessions)
4. **Preprocessing:** built a scikit-learn `Pipeline` with a `ColumnTransformer`, so the same preprocessing is applied to training and test data
5. **Baseline model:** pruned Decision Tree (`max_depth=6`, `min_samples_leaf=30`, `class_weight="balanced"`)
6. **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation, scored on F1, searching:
   - `max_depth`: 3, 4, 5, 6, 8, 10
   - `min_samples_split`: 10, 30, 50, 100
   - `min_samples_leaf`: 5, 10, 20, 30, 50
   - `class_weight`: None, balanced

## Results

| Model | Train F1 | Test F1 |
|---|---|---|
| Baseline pruned Decision Tree | 0.653 | 0.625 |
| **Tuned Decision Tree (GridSearchCV)** | 0.673 | **0.649** |
| Benchmark | – | 0.55 |

**Best parameters:** `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=50`, `class_weight=None` (best cross-validation F1: 0.656).

The tuned model beats the 0.55 benchmark by about 0.10 F1. The small gap between train and test F1 (0.673 vs 0.649) shows the pruned tree is not heavily overfitting.

**Baseline tree on the test set:**

| Class | Precision | Recall | F1 | Sessions |
|---|---|---|---|---|
| No purchase (0) | 0.96 | 0.85 | 0.90 | 2,059 |
| Purchase (1) | 0.50 | 0.83 | 0.63 | 382 |

Overall accuracy was 0.84, and the baseline caught 318 of the 382 real purchases.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Project Structure

```
├── ShopSmart_code.ipynb        # Notebook: cleaning, EDA, pipeline, tuning, evaluation
├── shop_smart_ecommerce.csv    # Dataset
└── README.md
```

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/HeyPiyush19/shopsmart-purchase-prediction.git
   cd shopsmart-purchase-prediction
   ```
2. Install the libraries:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Open the notebook:
   ```
   jupyter notebook ShopSmart_code.ipynb
   ```
4. Run all cells.

## Author

**Piyush Chaurasiya**
BCA Final Year Student | Aspiring Data Analyst & AI/ML Engineer
GitHub: [HeyPiyush19](https://github.com/HeyPiyush19)
