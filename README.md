# NYC Airbnb Price Prediction

A machine learning project that predicts Airbnb listing prices in New York City using the [New York City Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset from Kaggle.

---

## Project Overview

The goal of this project is to build and compare multiple regression models to accurately predict the nightly price of an Airbnb listing based on features such as location, room type, and review activity.

---

## Dataset

- **Source:** Kaggle – `dgomonov/new-york-city-airbnb-open-data`
- **File:** `AB_NYC_2019.csv`
- **Size:** ~49,000 listings across 5 NYC boroughs

---

## Project Structure

```
NYC_Airbnb_Price_Prediction.ipynb   # Main notebook
README.md
```

---

## Workflow

### 1. Data Cleaning
- Removed columns with heavy missing values (`last_review`, `reviews_per_month`)
- Dropped remaining rows with nulls
- Removed price outliers (kept bottom 99% → prices ≤ $795)
- Filtered `minimum_nights` outliers (99th percentile cap)

### 2. Data Preprocessing
- Dropped non-predictive columns
- One-hot encoded categorical features (`neighbourhood_group`, `room_type`)
- Addressed multicollinearity by dropping one highly correlated neighbourhood group column
- Applied `StandardScaler` on training data

### 3. Exploratory Data Analysis (EDA)
- Price distribution (original vs. log-transformed)
- Geographic price heatmap (latitude/longitude)
- Price by room type (boxplot)
- Correlation heatmap

### 4. Data Splitting
- 80/20 train-test split
- Features standardized using `StandardScaler`

### 5. Models Trained

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline model; multiple multicollinearity scenarios compared |
| **Random Forest** | Tuned with `RandomizedSearchCV`; cross-validated |
| **XGBoost** | Tuned with `RandomizedSearchCV`; learning curve analysis |
| **MLP (PyTorch)** | Deep learning approach; tested with up to 300 epochs and batch normalization |

### 6. Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## Setup & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib seaborn kaggle
```

### Kaggle API Token

To download the dataset automatically via the notebook, you need a Kaggle API token:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. In the notebook, paste your token where indicated:
   ```python
   os.environ['KAGGLE_API_TOKEN'] = "YOUR_KAGGLE_TOKEN_HERE"
   ```

---

## Results

Final evaluation on the test set:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $53.00 | $83.76 | 0.3650 |
| XGBoost (tuned) | $44.89 | $74.83 | 0.4931 |
| MLP V2 (Batch Norm) | $45.54 | $76.40 | 0.4717 |
| **Random Forest (tuned)** | **$44.13** | **$74.12** | **0.5027** |

✅ Best model: **Random Forest (tuned)** with R² = 0.5027

---

## Authors

Developed as a collaborative group project.
