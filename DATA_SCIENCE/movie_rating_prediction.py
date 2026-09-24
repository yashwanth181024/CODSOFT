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

class MovieRatingPredictor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.encoding_maps = {} 
        self.global_mean = 0
        self.models = {
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        }
        self.best_model = None

    def load_and_clean_data(self):
        """Loads the IMDb dataset and resolves string corruptions and missing values."""
        # Using latin-1 encoding to prevent crashes on non-ASCII characters in Indian names
        df = pd.read_csv(self.file_path, encoding='latin-1')
        df = df.copy()

        # Drop rows missing our target value (Rating)
        df.dropna(subset=['Rating'], inplace=True)

        # 1. Clean 'Duration' column (e.g., '109 min' -> 109)
        if 'Duration' in df.columns:
            df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)
            df['Duration'] = df['Duration'].fillna(df['Duration'].median())

        # 2. Clean 'Year' column (e.g., '(2019)' -> 2019)
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d+)').astype(float)
            df['Year'] = df['Year'].fillna(df['Year'].median())

        # 3. Handle categorical missing texts safely
        text_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()

        return df

    def prepare_data(self, df):
        """Applies Target Encoding to text safely without Data Leakage."""
        df = df.copy()
        categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
        
        X = df[['Year', 'Duration'] + categorical_features].copy()
        y = df['Rating']

        # Split data *before* target encoding to guarantee zero data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.global_mean = y_train.mean()

        # Apply Target Encoding calculated from the training set only
        for col in categorical_features:
            target_mean = y_train.groupby(X_train[col]).mean()
            self.encoding_maps[col] = target_mean
            
            X_train[col] = X_train[col].map(target_mean).fillna(self.global_mean)
            X_test[col] = X_test[col].map(target_mean).fillna(self.global_mean)

        # Scale all processed features uniformly
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

    def visualize_data(self, df):
        """Generates distribution plots for the target variable."""
        plt.figure(figsize=(10, 4))
        sns.histplot(df['Rating'], kde=True, color='purple', bins=20)
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('IMDb Rating')
        plt.tight_layout()
        plt.savefig("movie_rating_distribution.png")
        plt.close()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Evaluates regression models via mean metrics."""
        results = {}
        best_r2 = -float('inf')
        best_preds = None

        print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'R² Score':<8}")
        print("-" * 52)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            print(f"{name:<20} | {mae:<8.3f} | {rmse:<8.3f} | {r2:<8.3f}")
            results[name] = r2

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                best_preds = preds

        self.plot_predictions(y_test, best_preds)
        self.plot_metrics_comparison(results)
        
        print(f"\n🏆 Final Selected Model: {type(self.best_model).__name__}")
        print(f"Top Validation R² Score: {best_r2:.3f}")

    def plot_predictions(self, y_true, y_pred):
        """Generates actual vs predicted values plot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.3, color='teal')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Movie Ratings')
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png")
        plt.close()

    def plot_metrics_comparison(self, results):
        """Benchmarks model R² values."""
        plt.figure(figsize=(7, 4))
        plt.bar(results.keys(), results.values(), color=['tan', 'orange', 'crimson'])
        plt.ylabel("R² Score")
        plt.title("Regression Model Performance Benchmark (R²)")
        plt.tight_layout()
        plt.savefig("regression_model_comparison.png")
        plt.close()

    def save_pipeline(self):
        """Saves pipeline assets to disk."""
        payload = {
            "model": self.best_model,
            "scaler": self.scaler,
            "encoding_maps": self.encoding_maps,
            "global_mean": self.global_mean
        }
        joblib.dump(payload, "movie_rating_pipeline.pkl")
        print("Pipeline assets ('movie_rating_pipeline.pkl') saved successfully.")

    def predict_rating(self, input_dict, feature_order):
        """Predicts a single custom movie input dynamically."""
        input_df = pd.DataFrame([input_dict])
        
        for col, map_data in self.encoding_maps.items():
            val = str(input_df.loc[0, col]).strip()
            input_df[col] = map_data.get(val, self.global_mean)
            
        input_df = input_df[feature_order]
        scaled_input = self.scaler.transform(input_df)
        prediction = self.best_model.predict(scaled_input)[0]
        return min(max(round(prediction, 1), 1.0), 10.0)


if __name__ == "__main__":
    print("🎬 Movie Rating Prediction System Started 🎬\n")

    # FIXED: String now accurately uses underscores to match 'IMDb_Movies_India.csv'
    FILE_NAME = "IMDb_Movies_India.csv"

    if not os.path.exists(FILE_NAME):
        print(f"Error: '{FILE_NAME}' file not found in current directory.")
        exit()

    predictor = MovieRatingPredictor(FILE_NAME)

    # Clean, Preprocess and Train Pipeline
    clean_df = predictor.load_and_clean_data()
    predictor.visualize_data(clean_df)
    print("📈 Data visualization charts exported successfully.")

    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(clean_df)
    predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    predictor.save_pipeline()

    print("\n--- Pipeline Manual Verification Case ---")
    sample_movie = {
        'Year': 2026.0,
        'Duration': 140.0,
        'Genre': 'Action, Drama',
        'Director': 'S.S. Rajamouli',
        'Actor 1': 'Prabhas',
        'Actor 2': 'Rana Daggubati',
        'Actor 3': 'Anushka Shetty'
    }
    
    predicted_score = predictor.predict_rating(sample_movie, feature_names)
    print(f"Predicted IMDb Score for custom case: {predicted_score}/10")