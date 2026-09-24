# Task 2 — Credit Card Fraud Detection 💳

> **CodSoft Machine Learning Internship |  September Batch C21 ID:BY26RY234729 **
> **Author : Yashwanth G S**
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning model to detect fraudulent credit card transactions and classify them as **Fraudulent** or **Legitimate** using three algorithms.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Files | `fraudTrain.csv` + `fraudTest.csv` |
| Total Records | 1,852,394 transactions |
| Features | 23 columns |
| Target Column | `is_fraud` (0 = Legitimate, 1 = Fraud) |
| Fraud Rate | 0.52% |
| Imbalance Ratio | 191 : 1 |

---

## ⚙️ Algorithms Used

| Algorithm | ROC-AUC |
|---|---|
| Logistic Regression | 0.9238 |
| Decision Tree | 0.9699 ✅ Best |
| Random Forest | 0.9653 |

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `card_merchant_distance` | Distance between cardholder and merchant location |
| `amt_deviation` | Z-score of transaction amount |
| `high_amt_flag` | Flag for unusually large transactions |
| `night_flag` | Flag for night-time transactions (10 PM – 5 AM) |
| `small_city` | Flag for low-population area transactions |
| `suspicion_score` | Composite fraud risk score |

---

## 📊 Output Files

| File | Description |
|---|---|
| `plot1_class_balance.png` | Class distribution chart |
| `plot2_algorithm_comparison.png` | All metrics compared across 3 models |
| `plot3_confusion_matrix.png` | Confusion matrix + business interpretation |
| `plot4_roc_and_scores.png` | ROC curve + fraud score distribution |
| `plot5_feature_importance.png` | Top 15 most important features |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python credit_card_fraud_detection.py
```

> Place `fraudTrain.csv` and `fraudTest.csv` in the same folder.

---

## 📈 Key Results

- Best Model : **Decision Tree** (ROC-AUC = 0.9699)
- Fraud Detection Rate : **99%** (Recall)
- Class imbalance handled using `class_weight='balanced'`
- 6 custom features engineered from raw transaction data

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
