from sklearn.decomposition import PCA

class ReductorPCA:
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)
    
    def transformar(self, X_scaled):
        X_pca = self.pca.fit_transform(X_scaled)
        varianza_explicada = self.pca.explained_variance_ratio_
        return X_pca, varianza_explicada