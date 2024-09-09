
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO)

class WasteGenerationPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()
        self.features = None
        self.target = None

    def load_data(self):
        logging.info("Loading waste generation data from CSV file.")
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        logging.info("Preprocessing data for waste generation prediction.")
        data.fillna(method='ffill', inplace=True)
        self.features = data.drop(columns=['waste_generated'])
        self.target = data['waste_generated']

    def train_model(self):
        logging.info("Training the waste generation prediction model.")
        self.model.fit(self.features, self.target)

    def predict_waste(self, new_data):
        logging.info("Predicting waste generation for new data.")
        predictions = self.model.predict(new_data)
        return predictions

    def evaluate_model(self):
        from sklearn.metrics import mean_squared_error, r2_score
        logging.info("Evaluating model performance.")
        predictions = self.model.predict(self.features)
        mse = mean_squared_error(self.target, predictions)
        r2 = r2_score(self.target, predictions)
        logging.info(f"Model performance - MSE: {mse}, R2: {r2}")
        return mse, r2

if __name__ == "__main__":
    predictor = WasteGenerationPredictor("waste_data.csv")
    data = predictor.load_data()
    predictor.preprocess_data(data)
    predictor.train_model()
    mse, r2 = predictor.evaluate_model()
    logging.info(f"Final Model - MSE: {mse}, R2: {r2}")
