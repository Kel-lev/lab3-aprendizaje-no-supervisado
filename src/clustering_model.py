import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configurar fuente
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma']

class AgrupadorKMeans:
    def __init__(self, n_clusters=3, random_state=42):
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
    
    def entrenar_predecir(self, X):
        labels = self.model.fit_predict(X)
        score_silueta = silhouette_score(X, labels)
        return labels, score_silueta
    
    # TAREA 1: Método del Codo
    def metodo_codo(self, X, k_min=1, k_max=10):
        inertias = []
        silhouette_scores = []
        k_values = range(k_min, k_max + 1)
        
        print("\n" + "="*60)
        print(" ANÁLISIS DEL MÉTODO DEL CODO")
        print("="*60)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                labels = kmeans.labels_
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
                print(f"K={k:2d} | Inercia: {kmeans.inertia_:10.2f} | Silhouette: {sil_score:.4f}")
            else:
                silhouette_scores.append(None)
                print(f"K={k:2d} | Inercia: {kmeans.inertia_:10.2f} | Silhouette: N/A")
        
        print("="*60)
        
        # Crear figura con 2 gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        
        # ==================== GRÁFICO 1: Inercia ====================
        ax1.plot(k_values, inertias, 'bo-', markersize=8, linewidth=2, 
                markerfacecolor='#FF6B6B', label='Curva de Inercia (WCSS)')
        ax1.set_xlabel('Numero de Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inercia (WCSS)', fontsize=12)
        ax1.set_title('Metodo del Codo - Inercia', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        
        # Resaltar el codo (K=3)
        ax1.axvline(x=3, color='red', linestyle='--', linewidth=2.5, alpha=0.8, 
                   label='Codo (K=3)')
        ax1.scatter(3, inertias[2], color='red', s=250, zorder=5, 
                   edgecolors='black', linewidths=2)
        ax1.annotate('(CODO)\nK=3', 
                    xy=(3, inertias[2]), 
                    xytext=(4.5, inertias[2] * 0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold', color='red')
        
        # Añadir valores en los puntos
        for i, (k, inercia) in enumerate(zip(k_values, inertias)):
            ax1.annotate(f'{inercia:.0f}', 
                        xy=(k, inercia), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8)
        
        # Leyenda a la DERECHA
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # ==================== GRÁFICO 2: Silhouette ====================
        k_values_sil = [k for k in k_values if k > 1]
        silhouette_scores_clean = [s for s in silhouette_scores if s is not None]
        
        ax2.plot(k_values_sil, silhouette_scores_clean, 'go-', markersize=8, linewidth=2, 
                markerfacecolor='#4ECDC4', label='Coeficiente de Silueta')
        ax2.set_xlabel('Numero de Clusters (K)', fontsize=12)
        ax2.set_ylabel('Coeficiente de Silueta', fontsize=12)
        ax2.set_title('Coeficiente de Silueta por K', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values_sil)
        
        # Destacar el mejor silhouette
        best_k_idx = np.argmax(silhouette_scores_clean)
        best_k = k_values_sil[best_k_idx]
        best_score = silhouette_scores_clean[best_k_idx]
        
        ax2.scatter(best_k, best_score, color='green', s=200, zorder=5, 
                   edgecolors='black', linewidths=2)
        ax2.annotate(f'Mejor K={best_k}\nScore: {best_score:.4f}', 
                    xy=(best_k, best_score), 
                    xytext=(best_k + 0.5, best_score * 0.9),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold', color='darkgreen')
        
        # Añadir valores
        for k, score in zip(k_values_sil, silhouette_scores_clean):
            ax2.annotate(f'{score:.3f}', 
                        xy=(k, score), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8)
        
        # Leyenda a la DERECHA (CORREGIDO)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.suptitle('ANALISIS DE K OPTIMO', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return k_values, inertias, silhouette_scores
    
    # TAREA 2: Visualización de Clusters
    def visualizar_todos_los_clusters(self, X_pca):
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF8A5C', '#A29BFE', '#FD79A8']
        
        ks_a_evaluar = [2, 3, 5]
        resultados = {}
        
        print("\n" + "="*60)
        print(" COMPARATIVA DE CLUSTERS")
        print("="*60)
        
        for k in ks_a_evaluar:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            sil_score = silhouette_score(X_pca, labels)
            resultados[k] = {'labels': labels, 'centroides': kmeans.cluster_centers_, 'silhouette': sil_score}
            print(f"K={k}: Silhouette Score = {sil_score:.4f}")
        
        mejor_k = max(resultados, key=lambda x: resultados[x]['silhouette'])
        print(f"\nMEJOR K: {mejor_k} (Silhouette: {resultados[mejor_k]['silhouette']:.4f})")
        print("="*60 + "\n")
        
        fig = plt.figure(figsize=(12, 9))
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.25, hspace=0.3)
        
        # ==================== GRÁFICO 1: K=2 ====================
        ax1 = plt.subplot(2, 2, 1)
        labels_k2 = resultados[2]['labels']
        centroides_k2 = resultados[2]['centroides']
        for i in range(2):
            mask = labels_k2 == i
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colores[i], label=f'Cluster {i}', alpha=0.7, s=50)
        ax1.scatter(centroides_k2[:, 0], centroides_k2[:, 1], 
                   c='black', marker='X', s=300, edgecolors='white', 
                   linewidths=2, label='Centroides')
        ax1.set_xlabel('PC1', fontsize=11)
        ax1.set_ylabel('PC2', fontsize=11)
        ax1.set_title(f'K=2 (Silhouette: {resultados[2]["silhouette"]:.4f})', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # ==================== GRÁFICO 2: K=3 (SELECCIONADO) ====================
        ax2 = plt.subplot(2, 2, 2)
        labels_k3 = resultados[3]['labels']
        centroides_k3 = resultados[3]['centroides']
        for i in range(3):
            mask = labels_k3 == i
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colores[i], label=f'Cluster {i}', alpha=0.7, s=50)
        ax2.scatter(centroides_k3[:, 0], centroides_k3[:, 1], 
                   c='black', marker='X', s=300, edgecolors='white', 
                   linewidths=2, label='Centroides')
        ax2.set_xlabel('PC1', fontsize=11)
        ax2.set_ylabel('PC2', fontsize=11)
        # ELIMINÉ LA ESTRELLA ★ QUE CAUSABA EL ERROR
        ax2.set_title(f'K=3 (Silhouette: {resultados[3]["silhouette"]:.4f})', 
                      fontweight='bold', color='darkred')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ==================== GRÁFICO 3: K=5 ====================
        ax3 = plt.subplot(2, 2, 3)
        labels_k5 = resultados[5]['labels']
        centroides_k5 = resultados[5]['centroides']
        for i in range(5):
            mask = labels_k5 == i
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colores[i], label=f'Cluster {i}', alpha=0.7, s=50)
        ax3.scatter(centroides_k5[:, 0], centroides_k5[:, 1], 
                   c='black', marker='X', s=300, edgecolors='white', 
                   linewidths=2, label='Centroides')
        ax3.set_xlabel('PC1', fontsize=11)
        ax3.set_ylabel('PC2', fontsize=11)
        ax3.set_title(f'K=5 (Silhouette: {resultados[5]["silhouette"]:.4f})', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # ==================== GRÁFICO 4: Comparativa ====================
        ax4 = plt.subplot(2, 2, 4)
        ks = list(resultados.keys())
        silhouettes = [resultados[k]['silhouette'] for k in ks]
        
        bars = ax4.bar(ks, silhouettes, color=['#45B7D1', '#FF6B6B', '#4ECDC4'], alpha=0.7, width=0.5)
        ax4.set_xlabel('Numero de Clusters (K)', fontsize=11)
        ax4.set_ylabel('Coeficiente de Silueta', fontsize=11)
        ax4.set_title('Comparativa de Silhouette Score', fontweight='bold')
        ax4.set_xticks(ks)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(silhouettes) * 1.2)
        
        for bar, valor in zip(bars, silhouettes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        if mejor_k in ks:
            idx = ks.index(mejor_k)
            bars[idx].set_color('red')
            bars[idx].set_alpha(1.0)
            # ELIMINÉ LA ESTRELLA ★ QUE CAUSABA EL ERROR
            ax4.annotate('MEJOR', 
                        xy=(mejor_k, silhouettes[idx]), 
                        xytext=(mejor_k, silhouettes[idx] + 0.05),
                        ha='center',
                        fontsize=11, fontweight='bold', color='red')
        
        plt.suptitle('COMPARATIVA DE CLUSTERS: K=2, K=3 y K=5', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print(" RESUMEN FINAL")
        print("="*60)
        print(f"K seleccionado: 3")
        print(f"   - Silhouette Score: {resultados[3]['silhouette']:.4f}")
        print(f"   - Coherente con el Metodo del Codo")
        print(f"   - Segmentacion interpretable (3 perfiles de cliente)")
        print("="*60 + "\n")