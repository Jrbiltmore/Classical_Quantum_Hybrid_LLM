# carbon_footprint_calculator.py
import pandas as pd
import numpy as np

class CarbonFootprintCalculator:
    def __init__(self, emissions_data):
        self.emissions_data = pd.read_csv(emissions_data)
        self.total_carbon_footprint = 0

    def calculate_carbon_footprint(self):
        """
        Calculate the total carbon footprint based on emission data (CO2e - carbon dioxide equivalent)
        """
        self.total_carbon_footprint = self.emissions_data['emissions'].sum()
        return self.total_carbon_footprint

    def calculate_scope_1_emissions(self):
        """
        Calculate direct emissions from owned or controlled sources (Scope 1)
        """
        scope_1_emissions = self.emissions_data[self.emissions_data['scope'] == 1]['emissions'].sum()
        return scope_1_emissions

    def calculate_scope_2_emissions(self):
        """
        Calculate indirect emissions from the generation of purchased electricity, steam, heating, and cooling (Scope 2)
        """
        scope_2_emissions = self.emissions_data[self.emissions_data['scope'] == 2]['emissions'].sum()
        return scope_2_emissions

    def calculate_scope_3_emissions(self):
        """
        Calculate all other indirect emissions that occur in a company’s value chain (Scope 3)
        """
        scope_3_emissions = self.emissions_data[self.emissions_data['scope'] == 3]['emissions'].sum()
        return scope_3_emissions

    def offset_emissions(self, offset_projects):
        """
        Subtract the impact of carbon offset projects from the total carbon footprint
        """
        total_offsets = offset_projects['offset_amount'].sum()
        self.total_carbon_footprint -= total_offsets
        return self.total_carbon_footprint

    def forecast_future_emissions(self, growth_rate=0.03):
        """
        Forecast future emissions based on the current data and an estimated annual growth rate
        """
        future_emissions = self.total_carbon_footprint * (1 + growth_rate)
        return future_emissions

if __name__ == "__main__":
    emissions_data = "emissions_data.csv"
    calculator = CarbonFootprintCalculator(emissions_data)
    total_emissions = calculator.calculate_carbon_footprint()
    print(f"Total Carbon Footprint: {total_emissions} CO2e")
    scope_1 = calculator.calculate_scope_1_emissions()
    scope_2 = calculator.calculate_scope_2_emissions()
    scope_3 = calculator.calculate_scope_3_emissions()
    print(f"Scope 1 Emissions: {scope_1} CO2e")
    print(f"Scope 2 Emissions: {scope_2} CO2e")
    print(f"Scope 3 Emissions: {scope_3} CO2e")
