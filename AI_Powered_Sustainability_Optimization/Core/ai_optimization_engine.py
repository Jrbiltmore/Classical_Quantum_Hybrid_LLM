
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class AIOptimizationEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        self.best_params_ = {}

    def load_data(self):
        logging.info("Loading data from the CSV file.")
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        logging.info("Preprocessing data: Handling missing values and scaling.")
        data.fillna(method='ffill', inplace=True)
        self.features = data.drop(columns=['target'])
        self.target = data['target']
        self.features = self.scaler.fit_transform(self.features)

    def hyperparameter_tuning(self):
        logging.info("Starting hyperparameter tuning using GridSearchCV.")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(self.features, self.target)
        self.best_params_ = grid_search.best_params_
        logging.info(f"Best hyperparameters: {self.best_params_}")

    def build_model(self):
        logging.info("Building the RandomForestRegressor model with best hyperparameters.")
        self.model = RandomForestRegressor(**self.best_params_, random_state=42)

    def train_model(self):
        logging.info("Training the model.")
        self.model.fit(self.features, self.target)

    def evaluate_model(self):
        logging.info("Evaluating the model performance.")
        predictions = self.model.predict(self.features)
        mse = mean_squared_error(self.target, predictions)
        r2 = r2_score(self.target, predictions)
        logging.info(f"Model performance - MSE: {mse}, R2: {r2}")
        return mse, r2

    def predict(self, new_data):
        logging.info("Making predictions on new data.")
        processed_data = self.scaler.transform(new_data)
        predictions = self.model.predict(processed_data)
        return predictions

    def save_model(self, model_path):
        import joblib
        logging.info("Saving the model to the specified path.")
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        import joblib
        logging.info("Loading the model from the specified path.")
        self.model = joblib.load(model_path)

if __name__ == "__main__":
    engine = AIOptimizationEngine(data_path="sustainability_data.csv")
    data = engine.load_data()
    engine.preprocess_data(data)
    engine.hyperparameter_tuning()
    engine.build_model()
    engine.train_model()
    mse, r2 = engine.evaluate_model()
    logging.info(f"Final Model - MSE: {mse}, R2: {r2}")
