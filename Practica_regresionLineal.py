# -*- coding: utf-8 -*-

# Importamos las librerías necesarias
import pandas as pd
from sklearn.datasets import fetch_california_housing


# Paso 1: Cargar el conjunto de datos
# Para este ejercicio, usaremos un conjunto de datos de precios de casas de California

cali = fetch_california_housing()
data = pd.DataFrame(cali.data, columns=cali.feature_names)


# =============================================================================
# # =============================================================================
# # Realiza lo siguiente
# # =============================================================================
# =============================================================================


# =============================================================================
# Visualiza la distribución de la variable objetivo (MedHouseVal).

# Crea una matriz de correlación para identificar relaciones entre las características.

# Visualiza la relación entre una característica de tu elección (por ejemplo, HouseAge) y la variable objetivo (MedHouseVal).

# Elimina datos no útiles (por ejemplo, valores nulos).

# Elimina datos atípicos utilizando un umbral de 3 desviaciones estándar.

# Entrena un modelo de regresión lineal utilizando el conjunto de entrenamiento seleccionando una característica y el regresos de sklearn.

# Calcula el error cuadrático medio (MSE) y el coeficiente de determinación (R²) para evaluar el rendimiento del modelo.

# Visualiza los resultados de la predicción en comparación con los datos reales.

# Muestra el intercepto y los coeficientes de la regresión lineal.

# Presenta una tabla con los coeficientes de cada característica.

# Entrena y evalúa otro modelo de regresión lineal utilizando diferentes características (por ejemplo, MedInc, HouseAge, AveRooms).

# Compara los resultados con el modelo anterior.

# ¿Cómo se comparan los errores cuadráticos medios (MSE) de los dos modelos?

# ¿Cuál de los dos modelos tiene un mejor rendimiento y por qué?

# ¿Qué puedes concluir sobre la importancia de seleccionar las características adecuadas para el modelo de regresión lineal?

# =============================================================================
























