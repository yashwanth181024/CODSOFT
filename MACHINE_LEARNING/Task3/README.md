# Task 3 — Customer Churn Prediction 📉

> **CodSoft Machine Learning Internship |  September Batch C21 ID:BY26RY234729 **
> **Author : Yashwanth G S**
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Develop a machine learning model to predict whether a customer will **churn (leave)** or **stay** in a subscription-based service using historical customer data including usage behavior and demographics.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `Churn_Modelling.csv` |
| Total Records | 10,000 customers |
| Features | 14 columns |
| Target Column | `Exited` (0 = Stayed, 1 = Churned) |
| Source | Bank Customer Churn Dataset |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear baseline model |
| Random Forest | Ensemble of 200 decision trees |
| Gradient Boosting | Sequential boosting — Best Model ✅ |

---

## 📈 Results

| Algorithm | ROC-AUC |
|---|---|
| Logistic Regression | — |
| Random Forest | — |
| Gradient Boosting | 0.8635 ✅ Best |

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `lifetime_value` | tenure × MonthlyCharges |
| `avg_monthly_spend` | Average spend per month |
| `charge_ratio` | TotalCharges / MonthlyCharges |
| `new_customer` | Flag for tenure ≤ 6 months |
| `loyal_customer` | Flag for tenure ≥ 48 months |

---

## 📊 Output Files

| File | Description |
|---|---|
| `plot1_churn_distribution.png` | Churn count and rate chart |
| `plot2_feature_distributions.png` | Top 6 features vs churn |
| `plot3_model_comparison.png` | All metrics across 3 models |
| `plot4_confusion_matrix.png` | Confusion matrix + business interpretation |
| `plot5_roc_and_scores.png` | ROC curve + churn score distribution |
| `plot6_feature_importance.png` | Top 15 churn predictors |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_churn_prediction.py
```

> Place `Churn_Modelling.csv` in the same folder as the script.

---

## 💡 Business Insights

| Insight | Finding |
|---|---|
| Highest churn risk | New customers (tenure ≤ 6 months) |
| Lowest churn risk | Loyal customers (tenure ≥ 48 months) |
| Best algorithm | Gradient Boosting (ROC-AUC = 0.8635) |
| Class imbalance | Handled using class_weight balanced |

---

## 🛠️ Libraries

```
pandas | numpy | matplotlib | seaborn | scikit-learn
```

---

## 👤 Author

**Yashwanth G S**
GitHub : https://github.com/Yashwanth18102004
Internship : CodSoft ML Internship |  September Batch C21 ID:BY26RY234729 
