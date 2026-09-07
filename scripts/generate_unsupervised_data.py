import os
import numpy as np
import pandas as pd

def generar_datos_clientes(n_samples=300, random_state=42):
    np.random.seed(random_state)
    
    # Segmento 1: Compradores jóvenes de alto gasto
    cluster1 = np.hstack([
        np.random.normal(23, 3, (100, 1)),
        np.random.normal(25000, 4000, (100, 1)),
        np.random.normal(80, 8, (100, 1)),
        np.random.normal(15, 3, (100, 1))
    ])
    
    # Segmento 2: Clientes de alto ingreso y alto gasto (VIP)
    cluster2 = np.hstack([
        np.random.normal(42, 6, (100, 1)),
        np.random.normal(85000, 8000, (100, 1)),
        np.random.normal(85, 6, (100, 1)),
        np.random.normal(22, 4, (100, 1))
    ])
    
    # Segmento 3: Clientes conservadores de gasto bajo
    cluster3 = np.hstack([
        np.random.normal(55, 7, (100, 1)),
        np.random.normal(50000, 6000, (100, 1)),
        np.random.normal(25, 7, (100, 1)),
        np.random.normal(5, 2, (100, 1))
    ])
    
    data = np.vstack([cluster1, cluster2, cluster3])
    columns = ['edad', 'ingreso_anual', 'puntaje_gasto', 'frecuencia_compra']
    df = pd.DataFrame(data, columns=columns)
    
    df['edad'] = df['edad'].astype(int)
    df['ingreso_anual'] = df['ingreso_anual'].round(2)
    df['puntaje_gasto'] = df['puntaje_gasto'].clip(1, 100).astype(int)
    df['frecuencia_compra'] = df['frecuencia_compra'].clip(1).astype(int)
    
    output_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'customer_behavior.csv')
    df.to_csv(file_path, index=False)
    
    print(f"[+] Dataset generado exitosamente en: {file_path}")

if __name__ == '__main__':
    generar_datos_clientes()