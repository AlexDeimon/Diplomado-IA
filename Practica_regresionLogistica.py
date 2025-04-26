# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Paso 1: Generación de Datos
# =============================================================================

# Generar datos sintéticos para clasificación binaria
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.03,
    random_state=42
)

# Visualizar los datos
plt.figure(figsize=(10,6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Clase 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Clase 1')
plt.title('Distribución de los Datos Originales')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# =============================================================================
# Paso 2: Preprocesamiento de Datos
# =============================================================================

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Añadir término de intercepción (bias)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# =============================================================================
# Paso 3: Implementación de Regresión Logística
# =============================================================================

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        # Inicializar pesos
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            # Propagación
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)

            # Cálculo de costo
            cost = self._compute_cost(h, y)
            self.cost_history.append(cost)

            # Actualización de pesos (descenso de gradiente)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# =============================================================================
# Paso 4: Entrenamiento del Modelo
# =============================================================================

# Instanciar y entrenar modelo
model = LogisticRegression(learning_rate=0.1, n_iterations=3000)
model.fit(X_train, y_train)

# Visualizar curva de aprendizaje
plt.plot(model.cost_history)
plt.title('Curva de Aprendizaje')
plt.xlabel('Iteración')
plt.ylabel('Costo (Log Loss)')
plt.show()

# =============================================================================
# Paso 5: Evaluación del Modelo
# =============================================================================

from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report,
                             roc_auc_score)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Métricas
print(f'Exactitud: {accuracy_score(y_test, y_pred):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}')
print('\nMatriz de Confusión:')
print(confusion_matrix(y_test, y_pred))
print('\nReporte de Clasificación:')
print(classification_report(y_test, y_pred))

# =============================================================================
# Paso 6: Visualización de Frontera de Decisión
# =============================================================================

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[np.ones((xx.ravel().shape[0], 1)),
                           xx.ravel(),
                           yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:,1], X[:,2], c=y, s=20, edgecolor='k')
    plt.title('Frontera de Decisión')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plot_decision_boundary(X_train, y_train, model)

# =============================================================================
# # =============================================================================
# # Realiza lo siguiente
# # =============================================================================
# =============================================================================

# =============================================================================
# # =============================================================================
# Modifica la tasa de aprendizaje y observa su efecto en la convergencia
# Desactiva la normalización de los datos (`StandardScaler`) y evalúa cómo afecta al rendimiento del modelo. ¿Por qué es importante normalizar los datos para la regresión logística?
# Cambia el número de iteraciones (`n_iterations`) y la tasa de aprendizaje (`learning_rate`). ¿Qué ocurre si el modelo tiene demasiadas iteraciones o una tasa de aprendizaje muy alta/baja? Grafica la curva de aprendizaje en diferentes configuraciones.
# Modifica el umbral de decisión en la función `predict` (por ejemplo, cambia `threshold=0.5` a otros valores como 0.3 o 0.7). ¿Cómo afecta esto a la matriz de confusión y las métricas como el AUC-ROC?
# Genera un nuevo conjunto de datos con `make_classification` donde las clases no sean linealmente separables. Observa cómo se comporta la frontera de decisión.
# Implementa una condición de parada temprana que detenga el entrenamiento si el cambio en el costo entre iteraciones consecutivas es menor a un umbral ("umbral_costo"). ¿Cómo afecta esto al tiempo de entrenamiento y a la convergencia?
# Implementa regularización L2 para prevenir sobreajuste. Compara los resultados con y sin regularización.
# Implementa la estandarización MinMaxScaler. Compara el rendimiento con y sin estandarización de características de ambas técnicas (StandarScaler - MinMaxScaler)
# # =============================================================================
# =============================================================================
