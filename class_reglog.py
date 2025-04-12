# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:09:38 2025

@author: drago
"""


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