
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data
