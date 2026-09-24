# Task-4 📈 Sales Prediction System

> **CodSoft Data Science Internship | June Batch C5 ID:BY26RY204383**  
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning pipeline that predicts **product sales** based on advertising budgets across **TV, Radio, and Newspaper** channels using regression models.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `advertising.csv` |
| Features | TV, Radio, Newspaper |
| Target | Sales |
| Missing Values | None (cleaned automatically) |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Linear Regression | Baseline regression model |
| Random Forest Regressor | Ensemble of decision trees |
| Gradient Boosting Regressor | Boosted trees for improved accuracy |

---

## 🔧 Data Processing

| Step | Detail |
|---|---|
| Cleaning | Drops index column (`Unnamed: 0`) if present |
| Scaling | StandardScaler applied to all features |
| Train-Test Split | 80/20 split with random seed for reproducibility |

---

## 📊 Output Files

| File | Description |
|---|---|
| `sales_correlation_matrix.png` | Correlation heatmap of advertising channels vs sales |
| `actual_vs_predicted_sales.png` | Scatter plot of actual vs predicted sales |
| `regression_model_comparison.png` | R² score comparison of models |
| `sales_prediction_pipeline.pkl` | Saved ML pipeline (model + scaler) |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python sales_predictor.py
```

> Place `advertising.csv` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`SalesPredictor`)
- Automatic best model selection based on R² score
- Clean visualizations for reporting
- Real-time prediction for custom advertising budgets
- Pipeline saved with `joblib` for reuse

---

## 📈 Key Results

- Best model selected automatically based on validation R² score  
- Correlation heatmap reveals impact of TV, Radio, and Newspaper on sales  
- Supports custom marketing mix input for dynamic sales prediction  

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
