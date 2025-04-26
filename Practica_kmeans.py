# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Paso 1: Generar datos de ejemplo
# Creamos un conjunto de datos de dos dimensiones con tres grupos distintos
def generar_datos():
    np.random.seed(42)
    grupo_1 = np.random.randn(100, 2) + np.array([0, -3])
    grupo_2 = np.random.randn(100, 2) + np.array([3, 3])
    grupo_3 = np.random.randn(100, 2) + np.array([-3, 3])
    datos = np.vstack([grupo_1, grupo_2, grupo_3])
    return datos

# Paso 2: Inicializar los centroides
# Seleccionamos aleatoriamente los centroides iniciales de los datos
def inicializar_centroides(datos, k):
    np.random.seed(42)
    indices = np.random.choice(datos.shape[0], k, replace=False)
    centroides = datos[indices]
    return centroides

# Paso 3: Asignar cada punto al centroide más cercano
# Calculamos la distancia de cada punto a los centroides y asignamos el punto al centroide más cercano
def asignar_centroides(datos, centroides):
    distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)
    etiquetas = np.argmin(distancias, axis=1)
    return etiquetas

# Paso 4: Actualizar los centroides
# Recalculamos los centroides como la media de los puntos asignados a cada centroide
def actualizar_centroides(datos, etiquetas, k):
    nuevos_centroides = np.array([datos[etiquetas == i].mean(axis=0) for i in range(k)])
    return nuevos_centroides

# Paso 5: Ejecutar el algoritmo K-means con visualización de iteraciones
def kmeans_con_visualizacion(datos, k, max_iter=100):
    # Inicializamos los centroides
    centroides = inicializar_centroides(datos, k)

    for i in range(max_iter):
        # Asignamos los puntos a los centroides más cercanos
        etiquetas = asignar_centroides(datos, centroides)

        # Visualizamos la iteración actual
        visualizar_iteracion(datos, etiquetas, centroides, i)

        # Calculamos los nuevos centroides
        nuevos_centroides = actualizar_centroides(datos, etiquetas, k)

        # Verificamos si los centroides han cambiado
        if np.all(centroides == nuevos_centroides):
            break
        centroides = nuevos_centroides

    return etiquetas, centroides

# Paso 6: Visualizar los resultados de cada iteración
def visualizar_iteracion(datos, etiquetas, centroides, iteracion):
    plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=200)
    plt.title(f'Iteración {iteracion}')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.pause(0.5)
    # plt.clf()

# Paso 7: Visualizar los resultados finales
def visualizar_resultados(datos, etiquetas, centroides):
    plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=200)
    plt.title('Resultado del algoritmo K-means')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.show()

# Paso 8: Calcular métricas de validación
def calcular_metricas(datos, etiquetas):
    silueta = silhouette_score(datos, etiquetas)
    calinski_harabasz = calinski_harabasz_score(datos, etiquetas)
    return silueta, calinski_harabasz

# Ejecución del algoritmo K-means
if __name__ == "__main__":
    # Generamos los datos de ejemplo
    datos = generar_datos()

    # Definimos el número de clusters
    k = 3

    # Ejecutamos el algoritmo K-means con visualización de iteraciones
    etiquetas, centroides = kmeans_con_visualizacion(datos, k)

    # Calculamos las métricas de validación
    silueta, calinski_harabasz = calcular_metricas(datos, etiquetas)
    print(f"Coeficiente de Silueta: {silueta:.2f}")
    print(f"Índice Calinski-Harabasz: {calinski_harabasz:.2f}")

    # Visualizamos los resultados finales
    visualizar_resultados(datos, etiquetas, centroides)


# =============================================================================
# # =============================================================================
# # Realiza lo siguiente
# # =============================================================================
# =============================================================================

# =============================================================================
# Modifica la función `generar_datos` para crear datasets con más de tres grupos (`k=4` o `k=5`). ¿Cómo afecta esto al número de iteraciones necesarias para que el algoritmo converja?  Visualiza los nuevos datos generados y describe las diferencias.
# Introduce ruido a los datos generados añadiendo puntos aleatorios fuera de los clusters. ¿Cómo afecta el ruido a la convergencia del algoritmo? ¿Qué ocurre con las métricas de validación?
# Cambia la estrategia de inicialización de los centroides. Inicializa los centroides aleatoriamente en un rango fijo (np.random.uniform). Compara los resultados con la inicialización original
# Ejecuta el algoritmo varias veces con diferentes inicializaciones de centroides. ¿Obtienes siempre los mismos resultados? ¿Por qué? ¿Cómo puedes mitigar el impacto de una mala inicialización?
#  Investiga sobre el `silhouette_score` y `calinski_harabasz_score`.
# Implementa el Método del Codo para identificar visualmente el número óptimo de clusters. Grafica la "inercia" (suma de las distancias al cuadrado entre los puntos y sus centroides asignados) para diferentes valores de `k`.
# Implementa una condición de parada que dependa del cambio en la posición de los centroides. Por ejemplo, detén el algoritmo si el desplazamiento promedio de los centroides entre iteraciones es menor a un umbral. Compara el número de iteraciones necesarias para converger con y sin esta condición.
# Utiliza un conjunto de datos real, como el dataset `Iris`  y aplica el algoritmo K-means. ¿Qué pasos adicionales de preprocesamiento necesitas realizar? ¿Cómo interpretas los clusters obtenidos en un contexto real?
# =============================================================================
