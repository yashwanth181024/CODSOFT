import pandas as pd
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


class MovieGenreClassifier:

    def __init__(self, file_path):
        self.file_path = file_path
        self.vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.models = {
            "Logistic": LogisticRegression(max_iter=1000),
            "NaiveBayes": MultinomialNB(),
            "SVM": LinearSVC()
        }
        self.best_model = None

    def load_data(self):
        dataset = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ::: ")
                if len(parts) == 4:
                    dataset.append(parts)

        df = pd.DataFrame(dataset, columns=["id", "title", "genre", "plot"])
        return df

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def prepare(self, df):
        df["processed"] = df["plot"].apply(self.clean_text)

        top = df["genre"].value_counts().head(10).index
        df = df[df["genre"].isin(top)]

        X = self.vectorizer.fit_transform(df["processed"])
        y = df["genre"]

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def visualize(self, df):
        counts = df["genre"].value_counts().head(10)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index)
        plt.title("Top 10 Genres Distribution")
        plt.savefig("genre_visual.png")
        plt.close()

    def train(self, X_train, y_train, X_test, y_test):
        results = {}
        best_score = 0

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"\n🔹 {name} Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds))

            results[name] = acc

            if acc > best_score:
                best_score = acc
                self.best_model = model
                best_preds = preds
                best_true = y_test

        self.plot_results(results)
        self.plot_confusion(best_true, best_preds)

        print(f"\nFinal Selected Model: {type(self.best_model).__name__}")
        print(f"Accuracy: {best_score:.4f}")

    def plot_results(self, results):
        plt.bar(results.keys(), results.values())
        plt.ylim(0, 1)
        plt.title("Model Performance Comparison")
        plt.savefig("model_results.png")
        plt.close()

    def plot_confusion(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, cmap="coolwarm")
        plt.title("Confusion Matrix")
        plt.savefig("confusion.png")
        plt.close()

    def save_model(self):
        joblib.dump((self.best_model, self.vectorizer), "genre_classifier.pkl")
        print("Model saved successfully")

    def predict(self, text):
        cleaned = self.clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        return self.best_model.predict(vec)[0]


if __name__ == "__main__":

    print("\nMovie Genre Classification System Started\n")

    FILE = "train_data.txt"

    if not os.path.exists(FILE):
        print("Dataset not found. Please check file path.")
        exit()

    system = MovieGenreClassifier(FILE)

    data = system.load_data()
    system.visualize(data)

    X_train, X_test, y_train, y_test = system.prepare(data)

    system.train(X_train, y_train, X_test, y_test)

    system.save_model()

    print("\nTry your own movie plots below!")

    while True:
        user_input = input("\nEnter plot (type 'exit' to stop): ")
        if user_input.lower() == "exit":
            print("Exiting system...")
            break

        result = system.predict(user_input)
        print(f"Predicted Genre: {result.upper()}")