# House Price Prediction

A regression project that predicts the sale price of a house from basic property details (lot area, building type, age, basement size, etc.) using scikit-learn.

## Results

Evaluated on a held-out 20% test set (292 houses):

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.669 | 31,814 | 50,365 | 18.3% |
| Linear Regression (log target) | 0.707 | 28,862 | 47,384 | 15.6% |
| **Random Forest** | **0.841** | **22,330** | **34,969** | **13.6%** |

5-fold cross-validated R² (mean): Linear Regression **0.60**, Random Forest **0.77**.
One fold scores noticeably lower for both models, so results vary with the split and the single-split score is slightly optimistic. The cross-validated number is the more honest one.

Top features by importance: total basement area, house age, lot area, building sub-class, years since remodel.

## Dataset

`HousePricepre.csv` has 2,919 rows and 13 columns, but only **1,460 rows have a `SalePrice`**. The rest are unlabeled and are dropped, because filling the target with a mean would create fake labels and make the evaluation meaningless.

| Column | Meaning |
|---|---|
| `MSSubClass` | Building class code (treated as a category) |
| `MSZoning` | Zoning classification |
| `LotArea` | Lot size (sq ft) |
| `LotConfig` | Lot configuration |
| `BldgType` | Type of dwelling |
| `OverallCond` | Overall condition rating (1-10) |
| `YearBuilt`, `YearRemodAdd` | Build / remodel year |
| `Exterior1st` | Exterior covering |
| `BsmtFinSF2`, `TotalBsmtSF` | Basement areas (sq ft) |
| `SalePrice` | **Target**: sale price |

## Approach

1. Load data and inspect missing values
2. Keep only rows with a known `SalePrice`; drop `Id`
3. Explore the target (right-skewed, so a log version is also tried) and feature correlations
4. Feature engineering: `MSSubClass` as category, `HouseAge`, `YearsSinceRemodel`, `IsRemodeled`, one-hot encoding
5. 80/20 train-test split (`random_state=42`)
6. Train Linear Regression, Linear Regression on log(price), and Random Forest
7. Evaluate with R², MAE, RMSE, MAPE; cross-validate; analyse the largest errors and feature importance

## Limitations

- Only 11 raw features. There is no living area, neighbourhood or overall quality rating, which are the strongest price drivers in the full dataset, so accuracy has a ceiling.
- Data comes from a single city and period (sales up to 2010).
- No hyperparameter tuning yet.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

Keep `HousePricepre.csv` in the same folder as the notebook.

## Possible improvements

- Hyperparameter tuning with `GridSearchCV`
- Gradient boosting (XGBoost / LightGBM)
- Add more features (living area, neighbourhood, quality)

## Tech stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
