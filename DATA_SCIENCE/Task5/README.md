# Task-5 💳 Credit Card Fraud Detection System

> **CodSoft Data Science Internship | June Batch C5 ID:BY26RY204383**  
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning pipeline that detects **fraudulent credit card transactions** using classification algorithms, while handling severe class imbalance.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `creditcard.csv` |
| Features | V1–V28 (PCA components), normAmount |
| Target | Class (0 = Genuine, 1 = Fraud) |
| Imbalance | Highly imbalanced dataset (fraud cases are rare) |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model |
| Random Forest Classifier | Ensemble of decision trees |

---

## 🔧 Data Processing

| Step | Detail |
|---|---|
| Normalization | StandardScaler applied to transaction `Amount` |
| Feature Engineering | Added `normAmount`, dropped `Time` and `Amount` |
| Train-Test Split | Stratified split (80/20) |
| Class Imbalance Handling | Random undersampling applied to training set only |

---

## 📊 Output Files

| File | Description |
|---|---|
| `fraud_class_distribution.png` | Fraud vs genuine transaction distribution (log scale) |
| `fraud_confusion_matrix.png` | Confusion matrix of best model |
| `fraud_model_comparison.png` | F1-score comparison bar chart |
| `fraud_detection_pipeline.pkl` | Saved ML pipeline (model + scaler) |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python fraud_detector.py
```

> Place `creditcard.csv` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`FraudDetector`)
- Handles extreme class imbalance with undersampling
- Automatic best model selection based on **F1-score**
- Clean visualizations for reporting
- Pipeline saved with `joblib` for reuse

---

## 📈 Key Results

- Best model selected automatically based on validation F1-score  
- Fraud detection performance evaluated with precision, recall, and F1-score  
- Confusion matrix heatmap highlights fraud detection accuracy  

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
