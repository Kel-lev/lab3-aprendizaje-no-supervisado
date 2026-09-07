from src import CargarYProcesarDatos, ReductorPCA, AgrupadorKMeans

def ejecutar_pipeline():
    print("=" * 60)
    print(" PIPELINE MODULAR DE APRENDIZAJE NO SUPERVISADO ")
    print("=" * 60)
    
    # 1. Carga y Normalización
    print("\n[0] Cargando y normalizando datos...")
    loader = CargarYProcesarDatos('data/customer_behavior.csv')
    X_scaled = loader.preprocesar_unsupervised()
    
    # 2. Reducción de Dimensionalidad con PCA
    print("\n[1] Aplicando PCA (Reducción a 2D)...")
    pca_module = ReductorPCA(n_components=2)
    X_pca, varianza = pca_module.transformar(X_scaled)
    print(f" -> Varianza Explicada: {varianza}")
    print(f" -> Varianza Total Conservada: {sum(varianza) * 100:.2f}%")
    
    # 3. Método del Codo MEJORADO
    print("\n[2] Ejecutando Método del Codo...")
    kmeans_module = AgrupadorKMeans(n_clusters=3)
    kmeans_module.metodo_codo(X_pca)
    
    # 4. Visualización de todos los clusters (K=2, 3, 5)
    print("\n[3] Visualizando comparativa de clusters...")
    kmeans_module.visualizar_todos_los_clusters(X_pca)
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETADO EXITOSAMENTE ")
    print("="*60)

if __name__ == '__main__':
    ejecutar_pipeline()