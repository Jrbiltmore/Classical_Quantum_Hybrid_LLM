# power_usage_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PowerUsageAnalyzer:
    def __init__(self, usage_data):
        self.usage_data = pd.read_csv(usage_data)

    def analyze_usage_patterns(self):
        """
        Analyze power usage patterns to identify inefficiencies and optimize usage
        """
        peak_usage = self.usage_data[self.usage_data['usage'] == self.usage_data['usage'].max()]
        off_peak_usage = self.usage_data[self.usage_data['usage'] == self.usage_data['usage'].min()]
        average_usage = self.usage_data['usage'].mean()

        print(f"Peak Usage: {peak_usage['usage'].values[0]} kWh")
        print(f"Off-Peak Usage: {off_peak_usage['usage'].values[0]} kWh")
        print(f"Average Usage: {average_usage} kWh")
        return peak_usage, off_peak_usage, average_usage

    def visualize_usage(self):
        """
        Visualize power usage trends over time
        """
        plt.plot(self.usage_data['timestamp'], self.usage_data['usage'])
        plt.title("Power Usage Over Time")
        plt.xlabel("Time")
        plt.ylabel("Usage (kWh)")
        plt.show()

    def suggest_optimizations(self):
        """
        Suggest optimizations based on usage patterns
        """
        peak_usage, off_peak_usage, average_usage = self.analyze_usage_patterns()
        if peak_usage['usage'].values[0] > 1.5 * average_usage:
            print("Suggestion: Implement demand response strategies during peak usage times.")
        if off_peak_usage['usage'].values[0] < 0.5 * average_usage:
            print("Suggestion: Consider shifting non-essential operations to off-peak times for cost savings.")

if __name__ == "__main__":
    analyzer = PowerUsageAnalyzer("power_usage_data.csv")
    analyzer.analyze_usage_patterns()
    analyzer.visualize_usage()
    analyzer.suggest_optimizations()
