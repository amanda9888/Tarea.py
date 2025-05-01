Timport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos sintéticos
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# Método del Codo
sse = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Visualización del Método del Codo
plt.figure(figsize=(10, 6))
plt.plot(k_values, sse, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Suma de Errores Cuadrados (SSE)')
plt.xticks(k_values)
plt.grid(True)
plt.show()
