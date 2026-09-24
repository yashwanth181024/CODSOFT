# Task-1 🚢 Titanic Survival Prediction 🧑‍🚢

> **CodSoft Data Science Internship | June Batch C5 ID:BY26RY204383**
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning pipeline that predicts whether a **Titanic passenger survived or not** based on their demographic and travel details.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `Titanic-Dataset.csv` |
| Features | Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone |
| Target | Survived (0 = Deceased, 1 = Survived) |
| Missing Values | Handled with median/mode imputation |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model |
| Decision Tree | Tree-based classifier with max depth |
| Random Forest | Ensemble of decision trees |

---

## 🔧 Data Processing

| Step | Detail |
|---|---|
| Missing Values | Median for Age/Fare, Mode for Embarked |
| Feature Engineering | FamilySize, IsAlone |
| Encoding | LabelEncoder for categorical features |
| Scaling | StandardScaler for numerical features |
| Train-Test Split | Stratified split (80/20) |

---

## 📊 Output Files

| File | Description |
|---|---|
| `titanic_data_insights.png` | Survival rate by gender & class |
| `model_comparison.png` | Accuracy comparison bar chart |
| `titanic_confusion_matrix.png` | Confusion matrix of best model |
| `titanic_pipeline.pkl` | Saved ML pipeline (model + scaler + encoders) |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python titanic_classifier.py
```

> Place `Titanic-Dataset.csv` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`TitanicSurvivalClassifier`)
- Automatic best model selection
- Clean visualizations for reporting
- Real-time prediction for new passenger data
- Pipeline saved with `joblib` for reuse

---

## 📈 Key Results

- Best model selected automatically based on validation accuracy
- Visual insights into survival by gender and passenger class
- Confusion matrix heatmap for performance evaluation

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
