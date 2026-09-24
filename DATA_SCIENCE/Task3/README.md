# Task-3 🌸 Iris Flower Classification System

> **CodSoft Data Science Internship | June Batch C5 ID:BY26RY204383**  
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning pipeline that predicts the **species of an Iris flower** based on its sepal and petal measurements using classification algorithms.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `IRIS.csv` |
| Features | sepal_length, sepal_width, petal_length, petal_width |
| Target | species (Iris-setosa, Iris-versicolor, Iris-virginica) |
| Missing Values | None |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |

---

## 🔧 Data Processing

| Step | Detail |
|---|---|
| Target Encoding | LabelEncoder for species (Setosa, Versicolor, Virginica) |
| Scaling | StandardScaler applied to all features |
| Train-Test Split | Stratified split (80/20) |

---

## 📊 Output Files

| File | Description |
|---|---|
| `iris_data_insights.png` | Pairplot visualization of features by species |
| `model_comparison.png` | Accuracy comparison bar chart |
| `iris_confusion_matrix.png` | Confusion matrix of best model |
| `iris_pipeline.pkl` | Saved ML pipeline (model + scaler + encoder) |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python iris_classifier.py
```

> Place `IRIS.csv` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`IrisClassifier`)
- Automatic best model selection based on accuracy
- Clean visualizations for reporting
- Real-time prediction for new flower measurements
- Pipeline saved with `joblib` for reuse

---

## 📈 Key Results

- Best model selected automatically based on validation accuracy  
- Visual insights into feature relationships across species  
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
