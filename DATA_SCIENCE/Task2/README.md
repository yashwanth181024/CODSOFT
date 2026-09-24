# Task-2 🎬 Movie Rating Prediction System

> **CodSoft Data Science Internship | June Batch C5 ID:BY26RY204383**  
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning pipeline that predicts the **IMDb rating of a movie** based on its metadata (year, duration, genre, director, and actors) using regression models.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `IMDb_Movies_India.csv` |
| Features | Year, Duration, Genre, Director, Actor 1, Actor 2, Actor 3 |
| Target | IMDb Rating (1.0 – 10.0) |
| Encoding | Target Encoding applied safely to categorical features |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Linear Regression | Baseline linear regression model |
| Random Forest Regressor | Ensemble of decision trees |
| Gradient Boosting Regressor | Boosted trees for improved accuracy |

---

## 🔧 Data Processing

| Step | Detail |
|---|---|
| Cleaning | Extract numeric values from `Duration` and `Year` |
| Missing Values | Median imputation for numeric, "Unknown" for categorical |
| Target Encoding | Applied to categorical features (Genre, Director, Actors) |
| Scaling | StandardScaler applied to all features |
| Train-Test Split | 80/20 split with random seed for reproducibility |

---

## 📊 Output Files

| File | Description |
|---|---|
| `movie_rating_distribution.png` | Distribution of IMDb ratings |
| `actual_vs_predicted.png` | Scatter plot of actual vs predicted ratings |
| `regression_model_comparison.png` | R² score comparison of models |
| `movie_rating_pipeline.pkl` | Saved ML pipeline (model + scaler + encoders) |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python movie_rating_predictor.py
```

> Place `IMDb_Movies_India.csv` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`MovieRatingPredictor`)
- Safe target encoding (no data leakage)
- Automatic best model selection based on R² score
- Clean visualizations for reporting
- Real-time prediction for custom movie inputs
- Pipeline saved with `joblib` for reuse

---

## 📈 Key Results

- Best model selected automatically based on validation R² score  
- Visual insights into rating distribution and prediction accuracy  
- Supports custom movie metadata input for dynamic rating prediction  

---

## 🛠️ Libraries

```
pandas | numpy | matplotlib | seaborn | scikit-learn | joblib
```

---

## 👤 Author

**Yashwanth G S**  
GitHub : https://github.com/Yashwanth18102004
Internship : CodSoft DS Internship | June Batch C5 ID:BY26RY204383
