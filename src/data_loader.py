import pandas as pd
from sklearn.preprocessing import StandardScaler

class CargarYProcesarDatos:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
    
    def preprocesar_unsupervised(self):
        df = pd.read_csv(self.filepath)
        X = df.select_dtypes(include=['float64', 'int64'])
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled