# Task 1 — Movie Genre Classification 🎬

> **CodSoft Machine Learning Internship | September Batch C21 ID:BY26RY234729**
> **Author : Yashwanth G S**
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build a machine learning model that predicts the **genre of a movie** based on its **plot summary** using Natural Language Processing (NLP) techniques.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `train_data.txt` |
| Format | `ID ::: TITLE ::: GENRE ::: PLOT` |
| Total Genres | 27 genres |
| Genres Used | Top 10 most frequent |

---

## ⚙️ Algorithms Used

| Algorithm | Description |
|---|---|
| Naive Bayes | Probabilistic text classifier |
| Logistic Regression | Linear classification model |
| SVM (LinearSVC) | Support Vector Machine |

---

## 🔧 Text Processing

| Step | Detail |
|---|---|
| Vectorization | TF-IDF (unigrams + bigrams) |
| Max Features | 30,000 |
| Text Cleaning | Lowercase + remove punctuation |
| Stop Words | English stop words removed |

---

## 📊 Output Files

| File | Description |
|---|---|
| `genre_visual.png` | Top 10 genre distribution chart |
| `model_results.png` | Accuracy comparison bar chart |
| `confusion.png` | Confusion matrix of best model |
| `genre_classifier.pkl` | Saved best model file |

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
python movie_genre_classification.py
```

> Place `train_data.txt` in the same folder as the script.

---

## 💡 Features

- Class-based OOP structure (`MovieGenreClassifier`)
- Automatic best model selection
- Interactive prediction — enter your own movie plot
- Model saved with `joblib` for reuse

---

## 📈 Key Results

- Best Model selected automatically based on accuracy
- Handles 10 most frequent movie genres
- Interactive prediction loop at end of script

---

## 🛠️ Libraries

```
pandas | scikit-learn | matplotlib | seaborn | joblib
```

---

## 👤 Author

**Yashwanth G S**
GitHub : https://github.com/Yashwanth18102004
Internship : CodSoft ML Internship | September Batch C21 ID:BY26RY234729
