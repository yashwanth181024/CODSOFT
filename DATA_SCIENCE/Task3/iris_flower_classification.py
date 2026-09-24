import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for clean notebook execution
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class IrisClassifier:

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""

    def load_and_preprocess_data(self):
        """Loads dataset, encodes targets, and splits data into train/test sets."""
        df = pd.read_csv(self.file_path)
        df = df.copy()

        # Target classification column is 'species'
        X = df.drop(columns=['species'])
        y = df['species']

        # Encode target labels (Iris-setosa, Iris-versicolor, Iris-virginica -> 0, 1, 2)
        y_encoded = self.label_encoder.fit_transform(y)

        # Stratified train-test split to ensure equal representation of species
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features uniformly
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def visualize_pairplot(self):
        """Generates a pairplot showing feature distributions across species."""
        df = pd.read_csv(self.file_path)
        sns.set_theme(style="ticks")
        
        # Plotting relationships between features
        pair_plot = sns.pairplot(df, hue="species", palette="Set2", diag_kind="kde")
        pair_plot.fig.suptitle("Iris Feature Relationships & Distributions", y=1.02)
        
        plt.savefig("iris_data_insights.png", bbox_inches='tight')
        plt.close()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains models and displays clear evaluation reports."""
        best_acc = 0
        results = {}

        print(f"{'Model':<20} | {'Validation Accuracy':<18}")
        print("-" * 42)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc * 100

            print(f"{name:<20} | {acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_model_name = name

        self.plot_metrics_comparison(results)
        
        # Calculate and plot confusion matrix for the winning model
        best_preds = self.best_model.predict(X_test)
        self.plot_confusion_matrix(y_test, best_preds)

        print(f"\n🏆 Final Selected Model: {self.best_model_name}")
        print(f"Top Validation Accuracy: {best_acc*100:.2f}%")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plots a confusion matrix heatmap for the best performing model."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=self.label_encoder.classes_, 
            yticklabels=self.label_encoder.classes_
        )
        plt.xlabel('Predicted Species')
        plt.ylabel('Actual Species')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig("iris_confusion_matrix.png")
        plt.close()

    def plot_metrics_comparison(self, results):
        """Plots a comparison bar chart of model performances."""
        plt.figure(figsize=(7, 4))
        colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen']
        plt.bar(results.keys(), results.values(), color=colors, width=0.4)
        plt.ylabel("Accuracy Score (%)")
        plt.title("Classification Model Performance Comparison")
        plt.ylim(80, 105)
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

    def save_pipeline(self):
        """Saves the scaler, encoder, and model for reproducibility."""
        payload = {
            "model": self.best_model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder
        }
        joblib.dump(payload, "iris_pipeline.pkl")
        print("Pipeline assets ('iris_pipeline.pkl') saved successfully.")

    def predict_species(self, sample_measurements, feature_names):
        """Predicts the species name for an incoming test dictionary."""
        input_df = pd.DataFrame([sample_measurements])[feature_names]
        scaled_input = self.scaler.transform(input_df)
        encoded_pred = self.best_model.predict(scaled_input)[0]
        return self.label_encoder.inverse_transform([encoded_pred])[0]


if __name__ == "__main__":
    print("🌸 Iris Flower Classification System Started 🌸\n")

    # FIXED: Absolute structural pathway mapped for your workspace setup
    FILE_NAME = r"C:\Users\yashw\Yashu\Internship\DATA_SCIENCE\Task3\IRIS.csv"

    if not os.path.exists(FILE_NAME):
        print(f"❌ Error: File missing at designated path: {FILE_NAME}")
        print("Please ensure your 'IRIS.csv' file is unzipped and placed in the Task3 folder.")
        exit()

    # Instantiate class workflow
    classifier = IrisClassifier(FILE_NAME)

    # 1. Exploratory Data Visualization
    classifier.visualize_pairplot()
    print("📈 Data visualization charts exported successfully.")

    # 2. Data Splits & Scale Conversions
    X_train, X_test, y_train, y_test, feature_names = classifier.load_and_preprocess_data()

    # 3. Training & Validation Execution
    classifier.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Save Artifacts
    classifier.save_pipeline()

    print("\n--- Pipeline Manual Verification Case ---")
    sample_flower = {
        'sepal_length': 6.1,
        'sepal_width': 2.8,
        'petal_length': 4.7,
        'petal_width': 1.2
    }
    
    predicted_species = classifier.predict_species(sample_flower, feature_names)
    print(f"Sample Flower Status Outcome: {predicted_species.upper()}")