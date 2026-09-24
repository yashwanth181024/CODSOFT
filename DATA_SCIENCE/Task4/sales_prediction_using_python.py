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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class SalesPredictor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.models = {
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""

    def load_and_preprocess_data(self):
        """Loads the advertising dataset, checks for missing values, and splits into train/test sets."""
        df = pd.read_csv(self.file_path)
        df = df.copy()

        # Drop common index columns if they exist (like 'Unnamed: 0')
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        # Features: TV, Radio, Newspaper | Target: Sales
        X = df[['TV', 'Radio', 'Newspaper']]
        y = df['Sales']

        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features uniformly to assist optimization stability
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def generate_insights(self):
        """Generates and saves a correlation heatmap to analyze advertising spend impact."""
        df = pd.read_csv(self.file_path)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        plt.figure(figsize=(6, 5))
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)
        plt.title('Advertising Channels vs Sales Correlation Matrix')
        plt.tight_layout()
        plt.savefig("sales_correlation_matrix.png")
        plt.close()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains regression architectures and tables validation deltas."""
        best_r2 = -float('inf')
        results = {}
        best_preds = None

        print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'R² Score':<8}")
        print("-" * 52)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            print(f"{name:<20} | {mae:<8.2f} | {rmse:<8.3f} | {r2:<8.3f}")
            results[name] = r2

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name
                best_preds = preds

        # Generate evaluation plots
        self.plot_predictions(y_test, best_preds)
        self.plot_metrics_comparison(results)
        
        print(f"\n🏆 Final Selected Model: {self.best_model_name}")
        print(f"Top Validation R² Score: {best_r2:.3f}")

    def plot_predictions(self, y_true, y_pred):
        """Generates actual vs predicted sales trend plot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color='crimson', edgecolors='black')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.xlabel('Actual Sales units')
        plt.ylabel('Predicted Sales units')
        plt.title(f'Actual vs Predicted Sales - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig("actual_vs_predicted_sales.png")
        plt.close()

    def plot_metrics_comparison(self, results):
        """Benchmarks model R² values."""
        plt.figure(figsize=(7, 4))
        plt.bar(results.keys(), results.values(), color=['tan', 'dodgerblue', 'darkorange'], width=0.4)
        plt.ylabel("R² Score (Goodness of Fit)")
        plt.title("Regression Performance Comparison (R²)")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig("regression_model_comparison.png")
        plt.close()

    def save_pipeline(self):
        """Saves pipeline components to disk for live application testing."""
        payload = {
            "model": self.best_model,
            "scaler": self.scaler
        }
        joblib.dump(payload, "sales_prediction_pipeline.pkl")
        print("Pipeline assets ('sales_prediction_pipeline.pkl') saved successfully.")

    def predict_sales(self, budget_dict, feature_order):
        """Infers predicted sales value given dynamic budget allocations."""
        input_df = pd.DataFrame([budget_dict])[feature_order]
        scaled_input = self.scaler.transform(input_df)
        prediction = self.best_model.predict(scaled_input)[0]
        return max(0.0, round(prediction, 2))


if __name__ == "__main__":
    print("📈 Sales Prediction System Initialization Started 📈\n")

    # Absolute systematic tracking path mapping
    FILE_NAME = r"C:\Users\yashw\Yashu\Internship\DATA_SCIENCE\Task4\advertising.csv"

    if not os.path.exists(FILE_NAME):
        print(f"❌ Error: File missing at designated path: {FILE_NAME}")
        print("Please check that your dataset is placed in the Task4 folder and named 'advertising.csv'.")
        exit()

    # Initialize workflow object
    predictor = SalesPredictor(FILE_NAME)

    # 1. Exploratory Data Discovery Heatmap
    predictor.generate_insights()
    print("📊 Data correlation graphs exported successfully.")

    # 2. Data Splits & Structural Preprocessing
    X_train, X_test, y_train, y_test, feature_names = predictor.load_and_preprocess_data()

    # 3. Training & Validation Performance Evaluation Matrix
    predictor.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Save Artifacts for Deployment
    predictor.save_pipeline()

    print("\n--- Pipeline Manual Verification Case ---")
    # Sample allocation scenario: High TV budget, moderate Radio, low Newspaper
    sample_marketing_mix = {
        'TV': 250.0,
        'Radio': 35.0,
        'Newspaper': 15.0
    }
    
    predicted_revenue = predictor.predict_sales(sample_marketing_mix, feature_names)
    print(f"Estimated Sales Output: {predicted_revenue} units")