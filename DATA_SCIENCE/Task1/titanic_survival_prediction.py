import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings to keep the output completely clean and professional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class TitanicSurvivalClassifier:

    def __init__(self, file_path):
        self.file_path = file_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        }
        self.best_model = None

    def load_data(self):
        """Loads the csv dataset into a pandas DataFrame."""
        df = pd.read_csv(self.file_path)
        return df

    def prepare_data(self, df):
        """Cleans data, handles missing values, engineers features, and encodes categories."""
        df = df.copy()

        # 1. Handle Missing Values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

        # 2. Feature Engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

        # 3. Drop features that won't help machine learning models generalize well
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # 4. Categorical Encoding (Sex, Embarked)
        categorical_cols = ['Sex', 'Embarked']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le  

        # 5. Separate Target and Features
        X = df.drop(columns=['Survived'])
        y = df['Survived']

        # 6. Scale numerical features
        num_cols = ['Age', 'Fare', 'FamilySize']
        X[num_cols] = self.scaler.fit_transform(X[num_cols])

        # 7. Stratified Train-Test Split
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def visualize_data(self, df):
        """Generates explanatory data visualizations for your project submission."""
        plt.figure(figsize=(12, 5))

        # Subplot 1: Survival rate by Gender
        plt.subplot(1, 2, 1)
        sns.barplot(x='Sex', y='Survived', data=df, palette='Set2', hue='Sex', legend=False)
        plt.title('Survival Rate by Gender')

        # Subplot 2: Survival rate by Passenger Class
        plt.subplot(1, 2, 2)
        sns.barplot(x='Pclass', y='Survived', data=df, palette='Set1', hue='Pclass', legend=False)
        plt.title('Survival Rate by Passenger Class')

        plt.tight_layout()
        plt.savefig("titanic_data_insights.png")
        plt.close()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Trains multiple models and flags the highest accuracy model."""
        results = {}
        best_score = 0
        best_preds = None

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"\n🔹 {name} Accuracy: {acc * 100:.2f}%")
            print(classification_report(y_test, preds, target_names=['Deceased', 'Survived']))

            results[name] = acc

            if acc > best_score:
                best_score = acc
                self.best_model = model
                best_preds = preds

        self.plot_performance(results)
        self.plot_confusion_matrix(y_test, best_preds)

        print(f"\n🏆 Final Selected Model: {type(self.best_model).__name__}")
        print(f"Top Validation Accuracy: {best_score * 100:.2f}%")

    def plot_performance(self, results):
        """Saves a bar plot comparing performance results."""
        plt.figure(figsize=(7, 4))
        plt.bar(results.keys(), [val * 100 for val in results.values()], color=['skyblue', 'salmon', 'lightgreen'])
        plt.ylabel("Accuracy (%)")
        plt.ylim(50, 100)
        plt.title("Model Accuracy Benchmark")
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Saves a labeled confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Predicted Deceased', 'Predicted Survived'],
                    yticklabels=['Actual Deceased', 'Actual Survived'])
        plt.title("Confusion Matrix (Best Model)")
        plt.tight_layout()
        plt.savefig("titanic_confusion_matrix.png")
        plt.close()

    def save_assets(self):
        """Saves the pipeline components so it can be deployed later."""
        payload = {
            "model": self.best_model,
            "scaler": self.scaler,
            "encoders": self.label_encoders
        }
        joblib.dump(payload, "titanic_pipeline.pkl")
        print("Pipeline assets ('titanic_pipeline.pkl') saved successfully.")

    def predict_survival(self, features_dict):
        """Allows real-time single-row pipeline prediction testing."""
        input_df = pd.DataFrame([features_dict])
        
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
                
        num_cols = ['Age', 'Fare', 'FamilySize']
        input_df[num_cols] = self.scaler.transform(input_df[num_cols])
        
        prediction = self.best_model.predict(input_df)[0]
        return "SURVIVED" if prediction == 1 else "DID NOT SURVIVE"


if __name__ == "__main__":
    print("\n🚢 Titanic Survival Prediction System Started 🚢\n")

    FILE_PATH = "Titanic-Dataset.csv" 

    if not os.path.exists(FILE_PATH):
        print(f"Error: '{FILE_PATH}' file not found in current directory.")
        exit()

    pipeline = TitanicSurvivalClassifier(FILE_PATH)

    # 1. Load & Plot Data
    raw_data = pipeline.load_data()
    pipeline.visualize_data(raw_data)
    print("📈 Data visualization charts exported successfully.")

    # 2. Process
    X_train, X_test, y_train, y_test = pipeline.prepare_data(raw_data)

    # 3. Model Benchmark Execution
    pipeline.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Save Artifacts
    pipeline.save_assets()

    # 5. Test Live Sample
    print("\n--- Pipeline Manual Verification Case ---")
    test_passenger = {
        'Pclass': 1,            
        'Sex': 'female',        
        'Age': 22.0,            
        'Fare': 71.28,          
        'Embarked': 'C',        
        'FamilySize': 2,        
        'IsAlone': 0            
    }
    
    status = pipeline.predict_survival(test_passenger)
    print(f"Sample Passenger Status Outcome: {status}")