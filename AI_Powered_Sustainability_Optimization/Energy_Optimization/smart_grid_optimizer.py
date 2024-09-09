# smart_grid_optimizer.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

class SmartGridOptimizer:
    def __init__(self, grid_data):
        self.grid_data = pd.read_csv(grid_data)
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    def preprocess_data(self):
        """
        Preprocess smart grid data for modeling and optimization
        """
        features = self.grid_data.drop(columns=['load_demand'])
        target = self.grid_data['load_demand']
        return features, target

    def train_model(self, features, target):
        """
        Train the neural network model on smart grid load data
        """
        self.model.fit(features, target)

    def optimize_smart_grid(self, new_grid_data):
        """
        Predict and optimize smart grid load based on new data
        """
        optimized_load = self.model.predict(new_grid_data)
        return optimized_load

    def balance_load_demand(self):
        """
        Balance the load demand in the smart grid using AI predictions
        """
        features, target = self.preprocess_data()
        self.train_model(features, target)
        future_grid_data = pd.read_csv("future_grid_data.csv")
        optimized_load = self.optimize_smart_grid(future_grid_data)
        return optimized_load

    def visualize_grid_performance(self, optimized_load):
        """
        Visualize the performance of the smart grid after optimization
        """
        import matplotlib.pyplot as plt
        plt.plot(self.grid_data['load_demand'], label='Actual Load')
        plt.plot(optimized_load, label='Optimized Load')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    optimizer = SmartGridOptimizer("smart_grid_data.csv")
    optimized_load = optimizer.balance_load_demand()
    optimizer.visualize_grid_performance(optimized_load)
