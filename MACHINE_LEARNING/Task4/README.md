# Task 4 — Spam SMS Detection 📱

> **CodSoft Machine Learning Internship |  September Batch C21 ID:BY26RY234729 **
> **Author : Yashwanth G S**
> **GitHub : https://github.com/Yashwanth18102004**

---

## 📌 Objective

Build an AI model that classifies SMS messages as **Spam** or **Legitimate** using Natural Language Processing techniques including TF-IDF vectorization and multiple classifiers.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `spam.csv` |
| Total Messages | 5,572 SMS messages |
| Target Column | `label` (ham = Legitimate, spam = Spam) |
| Spam Rate | ~13% |

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
| Max Features | 10,000 |
| URL handling | Replaced with `url` token |
| Number handling | Replaced with `num` token |
| Text cleaning | Lowercase + remove punctuation |

---

## 📊 Output Files

| File | Description |
|---|---|
| `plot1_distribution.png` | Spam vs Legitimate count + pie chart |
| `plot2_message_length.png` | Message length distribution by class |
| `plot3_model_comparison.png` | All metrics across 3 models |
| `plot4_confusion_matrix.png` | Confusion matrix + detection summary |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python spam_sms_detection.py
```

> Place `spam.csv` in the same folder as the script.

---

## 💡 Sample Predictions

| Message | Prediction |
|---|---|
| Congratulations! You won a FREE iPhone. Click here! | 🚨 SPAM |
| Hey, are we still meeting for lunch tomorrow? | ✅ LEGITIMATE |
| URGENT: Your account will be suspended. Call now! | 🚨 SPAM |
| Can you please send me the homework notes? | ✅ LEGITIMATE |

---

## 📈 Key Results

- Auto-detects label and text columns from dataset
- Handles URLs and numbers with custom token replacement
- Best model selected automatically based on F1-Score
- Interactive prediction for any custom SMS message

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
