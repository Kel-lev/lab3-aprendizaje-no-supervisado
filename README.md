# Laboratorio 03 - Aprendizaje No Supervisado: Clustering y PCA

## Datos de la Asignatura
- **Asignatura:** Inteligencia Artificial 
- **Tema:** Clustering y Reducción de Dimensionalidad
- **Integrantes:** Ccorihuaman Delgado, Hamlet Nayeli - Leva Ayte, Kelma Ivonne

## Instrucciones de Ejecución

### 1. Instalar dependencias

pip install -r requirements.txt

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar el dataset sintético
python scripts/generate_unsupervised_data.py

# 3. Ejecutar el pipeline completo
python main.py


## Comandos de git utilizados
# 1. Inicializar repositorio local
git init

# 2. Agregar archivos al área de preparación (Staging)
git add .

# 3. Crear el primer commit estructurado
git commit -m "feat: implementacion pipeline no supervisado PCA y KMeans (Lab03)"

# 4. Cambiar la rama principal a main
git branch -M main

# 5. Enlazar con el repositorio remoto de GitHub
git remote add origin https://github.com/Kel-lev/lab3-aprendizaje-no-supervisado.git

# 6. Subir el código al repositorio remoto
git push -u origin main

## Salida en consola: 
============================================================
 PIPELINE MODULAR DE APRENDIZAJE NO SUPERVISADO 
============================================================

[0] Cargando y normalizando datos...
Datos cargados y normalizados

[1] Aplicando PCA (Reducción a 2D)...
 -> Varianza Explicada: [0.5824 0.3537]
 -> Varianza Total Conservada: 94.19%

[2] Ejecutando Método del Codo...
============================================================
 ANÁLISIS DEL MÉTODO DEL CODO
============================================================
K= 1 | Inercia:    1130.27 | Silhouette: N/A
K= 2 | Inercia:     458.99 | Silhouette: 0.6211
K= 3 | Inercia:      56.19 | Silhouette: 0.8212
K= 4 | Inercia:      46.60 | Silhouette: 0.6671
K= 5 | Inercia:      39.31 | Silhouette: 0.5173
K= 6 | Inercia:      32.98 | Silhouette: 0.3766
K= 7 | Inercia:      27.19 | Silhouette: 0.3902
K= 8 | Inercia:      24.22 | Silhouette: 0.3877
K= 9 | Inercia:      21.24 | Silhouette: 0.3858
K=10 | Inercia:      19.11 | Silhouette: 0.3962
============================================================

[3] Comparativa de Clusters...
============================================================
 COMPARATIVA DE CLUSTERS
============================================================
K=2: Silhouette Score = 0.6211
K=3: Silhouette Score = 0.8212
K=5: Silhouette Score = 0.5173

MEJOR K: 3 (Silhouette: 0.8212)
============================================================

============================================================
 RESUMEN FINAL
============================================================
K seleccionado: 3
   - Silhouette Score: 0.8212
   - Coherente con el Metodo del Codo
   - Segmentacion interpretable (3 perfiles de cliente)
============================================================

============================================================
 PIPELINE COMPLETADO EXITOSAMENTE 
============================================================

## Imágenes de los Resultados

### Método del Codo

![Método del Codo](/Figura_1.png)

### Clustering visualizado mediante PCA

![Clusters PCA](/Figura_2.png)

## Respuestas Técnicas

**1.	¿Qué problemas ocurren en el algoritmo K-Means si omitimos el escalado con StandardScaler en variables con magnitudes desproporcionadas?**
Si omitimos el escalado con StandardScaler, las variables que tienen valores más grandes, como el ingreso anual (~$85,000), tendrán mayor influencia en el cálculo de las distancias que variables con valores más pequeños, como la edad (~40 años).
Esto puede provocar que:
K-Means se enfoque principalmente en las variables de mayor magnitud.
Los clusters se formen principalmente según el ingreso, dejando de lado variables importantes como la frecuencia de compra o el puntaje de gasto.
Por ello, es importante escalar los datos para que todas las variables tengan una influencia más equilibrada.
**2.	¿Qué representan geométricamente las dos dimensiones generadas por el PCA respecto a la matriz original de 4 variables?**
Las dos dimensiones del PCA representan las dos direcciones que mejor resumen la información de los datos originales:
PC1: Es la dirección que explica la mayor cantidad de variación de los datos.
PC2: Es una dirección diferente y perpendicular a PC1 que explica la segunda mayor cantidad de variación.
En conjunto, PC1 y PC2 permiten representar los datos originales de 4 dimensiones en un plano 2D, manteniendo la mayor cantidad de información posible.

## Conclusiones

En este laboratorio se implementó un pipeline de aprendizaje no supervisado utilizando técnicas de reducción de dimensionalidad y clustering, específicamente PCA y K-Means, con el objetivo de identificar posibles segmentos dentro de un conjunto de datos de clientes.

En primer lugar, se realizó la normalización de los datos para evitar que las variables con escalas diferentes tuvieran una influencia desproporcionada en el proceso de clustering. Posteriormente, se aplicó PCA para reducir la dimensionalidad a dos componentes principales. Estas componentes conservaron el 94.19% de la varianza total, lo que indica que la representación bidimensional mantiene gran parte de la información presente en los datos originales y permite visualizar los resultados de manera más sencilla.

Para determinar el número adecuado de clusters se aplicó el Método del Codo, evaluando valores de K entre 1 y 10. Los resultados mostraron una disminución muy significativa de la inercia hasta K=3, mientras que a partir de este valor las mejoras fueron considerablemente menores. Esto permitió identificar K=3 como un posible punto de equilibrio entre la reducción de la inercia y la cantidad de grupos utilizados.

Posteriormente, se utilizó el Silhouette Score para evaluar la calidad de la agrupación. Se obtuvieron valores de 0.6211 para K=2, 0.8212 para K=3 y 0.5173 para K=5. El valor de 0.8212 obtenido con K=3 representa la mejor calidad de separación entre los grupos evaluados, indicando que los clusters presentan una buena cohesión interna y una adecuada separación entre sí.

En consecuencia, tanto el Método del Codo como el Silhouette Score respaldan la selección de K=3 como el número adecuado de clusters para este conjunto de datos. Esto permite identificar tres perfiles de clientes diferenciados, proporcionando una segmentación interpretable que podría ser utilizada como base para analizar las características de cada grupo y desarrollar estrategias específicas para cada segmento.

En términos generales, el laboratorio permitió aplicar de manera práctica un pipeline completo de aprendizaje no supervisado: preparación y normalización de datos, reducción de dimensionalidad mediante PCA, determinación del número de clusters, aplicación de K-Means y evaluación de la calidad de la agrupación. Los resultados obtenidos muestran que la combinación de estas técnicas permite descubrir estructuras y patrones con características similares en los datos sin necesidad de contar previamente con etiquetas o categorías conocidas.