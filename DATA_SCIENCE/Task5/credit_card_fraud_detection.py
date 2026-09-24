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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class FraudDetector:

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        }
        self.best_model = None
        self.best_model_name = ""

    def load_and_preprocess_data(self):
        """Loads transaction data, normalizes features, and handles class imbalance via undersampling."""
        df = pd.read_csv(self.file_path)
        df = df.copy()

        # FIXED: Corrected the reshape syntax error from -index to -1
        if 'Amount' in df.columns:
            df['normAmount'] = self.scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
            df.drop(columns=['Time', 'Amount'], inplace=True, errors='ignore')

        # Features (V1 to V28, normAmount) and Target (Class: 0 = Genuine, 1 = Fraud)
        X = df.drop(columns=['Class'])
        y = df['Class']

        # CRITICAL STEP: Split original data BEFORE sampling to avoid validation data leakage
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Original Training Imbalance: Fraud={sum(y_train_raw==1)}, Genuine={sum(y_train_raw==0)}")

        # Implement Random Undersampling on Training Data Only
        fraud_indices = np.array(y_train_raw[y_train_raw == 1].index)
        genuine_indices = y_train_raw[y_train_raw == 0].index
        
        # Sample an equal amount of genuine transactions
        random_genuine_indices = np.random.choice(genuine_indices, len(fraud_indices), replace=False)
        
        balanced_indices = np.concatenate([fraud_indices, random_genuine_indices])
        
        X_train_balanced = X_train_raw.loc[balanced_indices]
        y_train_balanced = y_train_raw.loc[balanced_indices]

        print(f"⚖️  Balanced Training Set: Fraud={sum(y_train_balanced==1)}, Genuine={sum(y_train_balanced==0)}\n")

        return X_train_balanced, X_test, y_train_balanced, y_test, X.columns

    def plot_fraud_distribution(self):
        """Saves a plot visualizing the intense original class disparity."""
        df = pd.read_csv(self.file_path)
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=df, palette=['royalblue', 'crimson'])
        plt.title('Transaction Class Distribution (0: Genuine, 1: Fraud)')
        plt.yscale('log')  # Log scale used to make fraud visible
        plt.ylabel('Count (Log Scale)')
        plt.tight_layout()
        plt.savefig("fraud_class_distribution.png")
        plt.close()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains architectures and highlights imbalanced metrics (Precision, Recall, F1)."""
        best_f1 = 0
        results = {}

        print(f"{'Model':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 58)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"{name:<20} | {prec:<10.3f} | {rec:<10.3f} | {f1:<10.3f}")
            results[name] = f1

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

        # Export performance visuals
        self.plot_confusion_matrix(y_test, self.best_model.predict(X_test))
        self.plot_metrics_comparison(results)
        
        print(f"\n🏆 Final Selected Model: {self.best_model_name}")
        print(f"Top Validation F1-Score: {best_f1:.3f}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Generates a confusion matrix tracking caught frauds vs missed alarms."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Genuine', 'Fraud'], 
                    yticklabels=['Genuine', 'Fraud'])
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig("fraud_confusion_matrix.png")
        plt.close()

    def plot_metrics_comparison(self, results):
        """Benchmarks model F1 values."""
        plt.figure(figsize=(6, 4))
        plt.bar(results.keys(), results.values(), color=['purple', 'teal'], width=0.3)
        plt.ylabel("F1-Score Metric")
        plt.title("Model F1-Score Performance Comparison")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig("fraud_model_comparison.png")
        plt.close()

    def save_pipeline(self):
        """Saves operational assets."""
        payload = {
            "model": self.best_model,
            "scaler": self.scaler
        }
        joblib.dump(payload, "fraud_detection_pipeline.pkl")
        print("Pipeline assets ('fraud_detection_pipeline.pkl') saved successfully.")


if __name__ == "__main__":
    print("💳 Credit Card Fraud Detection System Active 💳\n")

    FILE_NAME = r"C:\Users\yashw\Yashu\Internship\DATA_SCIENCE\Task5\creditcard.csv"

    if not os.path.exists(FILE_NAME):
        print(f"❌ Error: File missing at designated path: {FILE_NAME}")
        print("Please place 'creditcard.csv' in your Task5 directory.")
        exit()

    # Run system workflow
    detector = FraudDetector(FILE_NAME)
    detector.plot_fraud_distribution()
    print("📊 Data balance distribution visualizations exported.")

    X_train, X_test, y_train, y_test, feature_names = detector.load_and_preprocess_data()
    detector.train_and_evaluate(X_train, X_test, y_train, y_test)
    detector.save_pipeline()