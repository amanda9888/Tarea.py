git checkout -b branch_1
import numpy as np

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Inicialización de los pesos y el sesgo
weights = np.random.rand(2)
bias = np.random.rand(1)

git add perceptron.py
git commit -m "Agregar datos de entrada y salida"

git checkout -b branch_2

import numpy as np

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Inicialización de los pesos y el sesgo
weights = np.random.rand(2)
bias = np.random.rand(1)

# Función de activación (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Entrenamiento del perceptrón
learning_rate = 0.1
epochs = 10

for _ in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

git add perceptron.py
git commit -m "Agregar lógica de entrenamiento"


git checkout -b branch_3

import numpy as np

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Inicialización de los pesos y el sesgo
weights = np.random.rand(2)
bias = np.random.rand(1)

# Función de activación (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Entrenamiento del perceptrón
learning_rate = 0.1
epochs = 10

for _ in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Predicción
def predict(input_data):
    linear_output = np.dot(input_data, weights) + bias
    return step_function(linear_output)

# Ejemplo de uso
print(predict([1, 1]))  # Debería imprimir 1

git add perceptron.py
git commit -m "Agregar función de predicción y ejemplo de  predicción y ejemplo de uso "

# Subir branch_1
git checkout branch_1
git push origin branch_1

# Subir branch_2
git checkout branch_2
git push origin branch_2

# Subir branch_3
git checkout branch_3
git push origin branch_3
