# real_time_monitoring.py
import time
import pandas as pd
import numpy as np

class RealTimeMonitoring:
    def __init__(self, data_source):
        self.data_source = data_source
        self.current_data = pd.read_csv(data_source)
        self.energy_usage = 0
        self.waste_generation = 0
        self.carbon_emissions = 0

    def update_data(self):
        """
        Simulate real-time data updates from sensors
        """
        self.current_data['timestamp'] = pd.to_datetime(self.current_data['timestamp'])
        self.energy_usage = self.current_data['energy_usage'].sum()
        self.waste_generation = self.current_data['waste'].sum()
        self.carbon_emissions = self.current_data['emissions'].sum()

    def monitor_energy_usage(self):
        """
        Monitor real-time energy usage and log changes
        """
        previous_energy_usage = self.energy_usage
        self.update_data()
        if self.energy_usage != previous_energy_usage:
            print(f"Energy usage updated: {self.energy_usage} kWh")

    def monitor_waste_generation(self):
        """
        Monitor real-time waste generation
        """
        previous_waste = self.waste_generation
        self.update_data()
        if self.waste_generation != previous_waste:
            print(f"Waste generation updated: {self.waste_generation} tons")

    def monitor_carbon_emissions(self):
        """
        Monitor real-time carbon emissions
        """
        previous_emissions = self.carbon_emissions
        self.update_data()
        if self.carbon_emissions != previous_emissions:
            print(f"Carbon emissions updated: {self.carbon_emissions} CO2e")

    def log_real_time_data(self):
        """
        Log real-time data for further analysis and reporting
        """
        log_data = {
            'timestamp': pd.Timestamp.now(),
            'energy_usage': self.energy_usage,
            'waste_generation': self.waste_generation,
            'carbon_emissions': self.carbon_emissions
        }
        log_df = pd.DataFrame([log_data])
        log_df.to_csv('real_time_log.csv', mode='a', header=False, index=False)
        print("Real-time data logged successfully.")

    def start_monitoring(self, interval=60):
        """
        Continuously monitor sustainability metrics at regular intervals
        """
        try:
            while True:
                self.monitor_energy_usage()
                self.monitor_waste_generation()
                self.monitor_carbon_emissions()
                self.log_real_time_data()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Real-time monitoring stopped.")

if __name__ == "__main__":
    monitor = RealTimeMonitoring("real_time_sustainability_data.csv")
    monitor.start_monitoring()
