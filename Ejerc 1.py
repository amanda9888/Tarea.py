checkout main
git checkout -b ejercicio1
git push origin ejercicio1
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Datos de entrenamiento
X = np.array([[2,0], [4,4], [1,1], [2,4], [2,2], [2,3], [3,4], [3,3]])
y = np.array([0,1,0,1,0,1,0,1])

# Crear clasificador k=1 con distancia Manhattan
clf = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
clf.fit(X, y)

# Predecir el nuevo punto
new_point = np.array([[2.5, 2.5]])
print(clf.predict(new_point))  # Output: [0]
