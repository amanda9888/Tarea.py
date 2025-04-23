# Carga y Exploración de Datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el dataset
df = pd.read_csv('winequality-red.csv')

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Distribución de la variable objetivo (calidad)
plt.figure(figsize=(8, 4))
sns.countplot(x='quality', data=df, palette='Reds')
plt.title('Distribución de la Calidad del Vino')
plt.xlabel('Calidad')
plt.ylabel('Cantidad de Vinos')
plt.show()

# Preprocesamiento de Datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Árbol de Decisión
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_scaled, y_train)

y_pred_tree = tree_model.predict(X_test_scaled)
print("Precisión del Árbol de Decisión:", accuracy_score(y_test, y_pred_tree))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_tree))

plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=[str(i) for i in sorted(y.unique())],
          filled=True, fontsize=10)
plt.title("Árbol de Decisión")
plt.show()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print("Precisión del Random Forest:", accuracy_score(y_test, y_pred_rf))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_rf))

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_scaled)
print("Precisión del Mejor Random Forest:", accuracy_score(y_test, y_pred_best_rf))

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette='Reds_r')
plt.title("Importancia de las Características - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()
plt.show()
