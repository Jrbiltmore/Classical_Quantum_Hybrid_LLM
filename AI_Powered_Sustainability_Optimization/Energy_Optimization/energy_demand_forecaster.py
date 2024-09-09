# energy_demand_forecaster.py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class EnergyDemandForecaster:
    def __init__(self, historical_data):
        self.historical_data = pd.read_csv(historical_data)
        self.model = SVR(kernel='rbf')

    def preprocess_data(self):
        """
        Preprocess historical energy demand data
        """
        features = self.historical_data.drop(columns=['energy_demand'])
        target = self.historical_data['energy_demand']
        return features, target

    def train_model(self, features, target):
        """
        Train the model on historical energy demand data
        """
        self.model.fit(features, target)

    def predict_demand(self, future_data):
        """
        Predict future energy demand based on new input data
        """
        predicted_demand = self.model.predict(future_data)
        return predicted_demand

    def evaluate_model(self, test_data, true_demand):
        """
        Evaluate the model's performance using Mean Squared Error (MSE)
        """
        predictions = self.model.predict(test_data)
        mse = mean_squared_error(true_demand, predictions)
        print(f"Model Evaluation - MSE: {mse}")
        return mse

if __name__ == "__main__":
    forecaster = EnergyDemandForecaster("historical_energy_data.csv")
    features, target = forecaster.preprocess_data()
    forecaster.train_model(features, target)

    # Example prediction with future data
    future_data = pd.read_csv("future_energy_data.csv")
    predicted_demand = forecaster.predict_demand(future_data)
    print(f"Predicted Future Energy Demand: {predicted_demand}")
