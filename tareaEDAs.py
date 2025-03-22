# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:52:26 2025

@author: Diego Alejandro Sandoval Fernandez
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el dataset
data = pd.read_csv('winemag-data-130k-v2.csv')

# Visualización rapida de los datos
data.head()

# =============================================================================
# 1. Utilizando el dataset “winemag”, genere un histograma que muestra las 
# frecuencias de los puntos (points) dados a los vinos.
# =============================================================================
sns.histplot(data = data['points'], bins=20)
plt.xlabel('Puntos de los Vino')
plt.ylabel('Frecuencia')
plt.show()

# =============================================================================
# Observaciones: La mayoria de los vinos tienen puntuaciones en un rango 
# medio, hay muy pocos vinos con puntos en los extremos (80 - 82.5 y 95 - 100) 
# por lo que es inusual que haya calificaciones muy bajas o muy altas
# =============================================================================
# =============================================================================
# 2. Ahora genere un gráfico tipo distplot para evaluar la distribución del
# precio de los vinos. 
# =============================================================================
sns.displot(data = data['price'], bins=20)
plt.xlabel('Precios de los Vino')
plt.ylabel('Frecuencia')
plt.show()

# =============================================================================
# Observacion: hay demasiados vinos con precios cercanos a 0, y a simple vista 
# se ve que hay muy pocos vinos con precios altos, mayores a... 200
# =============================================================================
# Confirmando...
sns.displot(data[data['price'] < 500]['price'], bins=20)
plt.xlabel('Precios de los Vino')
plt.ylabel('Frecuencia')
plt.show()
sns.displot(data[data['price'] > 500]['price'], bins=20)
plt.xlabel('Precios de los Vino')
plt.ylabel('Frecuencia')
plt.show()
# =============================================================================
# Se evidencia una gran cantidad de vinos, entre 1000 y 7000 con precios entre 
# 0 y 100, y una menor cantidad con precios mayores a 500
# =============================================================================
# Con dicho análisis, tome una decisión para eliminar (drop) datos atípicos.
# =============================================================================
# Chat GPT me dio dos opciones, rango intercuartil (IQR) y desviación estándar
# y para datos sesgados recomendo mejor IQR y me dio el siguiente código:
# Calcular cuartiles y rango intercuartil (IQR)
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites para valores normales
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar el DataFrame solo con los valores dentro del rango permitido
data_filtrado = data[(data['price'] >= limite_inferior) & (data['price'] <= limite_superior)]

# Mostrar la cantidad de datos antes y después
print(f"Datos antes de filtrar: {len(data)}")
print(f"Datos después de filtrar: {len(data_filtrado)}")
print(f"Datos Excluidos: {len(data) - len(data_filtrado)}")

# =============================================================================
# Observaciones: de 129.971 datos, se filtaron 113.734 y se excluyeron 16.237
# Por lo que 12.5% de los datos fueron atipicos y el 87.5% se consideran dentro
# del rango correcto
# =============================================================================
# =============================================================================
# 3. genere una tabla (crosstab) que le permita agrupar (groupby) y evaluar que
# países tienen los vinos mas caros y mejor valorados. 
# =============================================================================
data_agrupado = data_filtrado.groupby('country')[['price', 'points']].mean()

data_agrupado.plot.bar(figsize=(12,6))  

plt.show()

# para la tabla le pedi ayuda a chat GPT, me indicó primero categorizar los 
# precios y puntuaciones en rangos.

# Categoria de precios
data_filtrado['rango_precio'] = pd.cut(data_filtrado['price'], bins=[0, 20, 50, 500, 1000, 5000], 
                                       labels=['Muy barato', 'Barato', 'Medio', 'Caro', 'Muy caro'])

# Categoria de puntos
data_filtrado['rango_puntos'] = pd.cut(data_filtrado['points'], bins=[0, 85, 90, 95, 100], 
                                       labels=['Normal', 'Bueno', 'Muy bueno', 'Excelente'])

# Tabla cruzada
tabla_crosstab = pd.crosstab(index=data_filtrado['country'], 
                             columns=[data_filtrado['rango_precio'], data_filtrado['rango_puntos']])

# Mostrar la tabla
print(tabla_crosstab)

# =============================================================================
# Posteriormente genere una grafica de barras con esta información y muestre 
# los datos de forma ascendente, los primeros 20.
# =============================================================================
f,ax = plt.subplots(1,2)

data_agrupado[['price']].sort_values('price', ascending=False).head(20).plot.bar(ax=ax[0], legend=False)
ax[0].set_title("Top 20 países con los vinos más caros")
ax[0].set_xlabel("Países")

data_agrupado[['points']].sort_values('points', ascending=False).head(20).plot.bar(ax=ax[1], legend=False)
ax[1].set_title("Top 20 países con los vinos mejor calificados")
ax[1].set_xlabel("Países")

plt.show()

# separando en 2 graficas para que se vean mejor
data_agrupado[['price']].sort_values('price', ascending=False).head(20).plot.bar(legend=False)
plt.title("Top 20 países con los vinos más caros")
plt.xlabel("Países")
plt.show()

data_agrupado[['points']].sort_values('points', ascending=False).head(20).plot.bar(legend=False)
plt.title("Top 20 países con los vinos mejor calificados")
plt.xlabel("Países")
plt.show()

# =============================================================================
# Observaciones: 
# 1. El pais con el vino mas caro con mucha diferencia es inglaterra, tambien 
# el mejor calificado
# 2. No hay una diferencia tan marcada en las puntuaciones como en los precios.
# 3. Inglaterra, Estados unidos, canada, Italia, Austria, Alemania, Israel, 
# Hungria, nueva zelanda, Francia y Suiza estan en ambas graficas por lo que generan
# vinos costosos y bien calificados; Lebanon, Austria, Croacia, Uruguay, mexico,
# Serbia, Republica checa y brazil estan entre los más caros pero no entre los
# mejores calificados; India, China, Luxemburgo, Morocco, Portugal, turkia y
# bulgaria estan entre los mejores calificados a un precio más barato y Slovenia
# ocupa el mismo lugar tanto en puntaje como en precio
# =============================================================================
# =============================================================================
# 4. Genere una gráfica de caja (boxplot) donde se observe la relación entre variedad y
# calidad (points)
# =============================================================================
sns.boxplot(data=data_filtrado, x='variety', y='points')

plt.xlabel("Variedad de vino")
plt.ylabel("Calidad de vino")

plt.show()

# Asi tal cual un grafico de caja de todas las variedades es muy complicado de
# entender ya que tiene demasiadas variedades de vino en el eje X
# Con ayuda de chat GPT generó este codigo para una mejor visualización:
# Obtener las 10 variedades más comunes
top_varieties = data_filtrado['variety'].value_counts().index[:10]

# Filtrar el dataset para incluir solo esas variedades
df_top = data_filtrado[data_filtrado['variety'].isin(top_varieties)]

# Ajustar el tamaño de la figura
plt.figure(figsize=(12, 6))

# Crear el boxplot ordenando por la mediana
sns.boxplot(data=df_top, x='variety', y='points', order=df_top.groupby('variety')['points'].median().sort_values().index)

# Rotar etiquetas del eje X
plt.xticks(rotation=45)

# Agregar título y etiquetas
plt.title("Distribución de puntuaciones por variedad de vino (Top 10 variedades)")
plt.xlabel("Variedad de vino")
plt.ylabel("Puntuación (points)")

# Mostrar gráfico
plt.show()

# =============================================================================
# Observaciones: 
# 1. La mayoria de las variedades de vino tienen puntuaciones entre 85 y 90 puntos
# 2. La mayoria de valores atípicos esta por encima de 92.5 y por debajo de 82.5
# 3. La mediana esta aproximadamente en 87-90 puntos para la mayoría de las variedades
# siendo las mas altas las de Pinot Noir, Riesling y Syrah; y las mas bajas
# metroit, Rose y Sauvignon Blanc
# =============================================================================
# =============================================================================
# 5. Cree una nueva etiqueta (atributo) al dataframe llamada “score points”, 
# el cual es la razón del puntaje obtenido del vino (cada índice en el dataframe) 
# entre el promedio de puntos obtenido por todos los vinos de ese país
# =============================================================================
puntos_x_pais = data_filtrado.groupby('country')['points'].transform('mean')
data_filtrado['score points'] = data_filtrado['points'] / puntos_x_pais

data_filtrado[['title','country', 'points', 'score points']].head()

# =============================================================================
# 6. La etiqueta price tiene casi 9,000 datos faltantes, defina una estrategia 
# para generar artificialmente dichos datos faltantes.
# =============================================================================
print(data['price'].isnull().sum())
print(data[data['price'].isnull()]['country'].value_counts())
mediana_precios_x_pais = data.groupby('country')['price'].median()
data.loc[data['price'].isnull(), 'price'] = data['country'].map(mediana_precios_x_pais)
print(data['price'].isnull().sum())
print(data[data['price'].isnull()]['country'].value_counts())

# =============================================================================
# La estrategia que se me ocurrio fue asignar a los datos faltantes la mediana 
# por cada pais, sin embargo no tome en cuenta el puntaje y ademas aun quedan 5
# datoa nulos en precio. Chat GPT sugirió utilizar regresión simple con los
# puntos para generar los datos faltantes

from sklearn.linear_model import LinearRegression

remaining_nulls = data['price'].isna().sum()
if remaining_nulls > 0:
    model = LinearRegression()
    df_no_nulls = data.dropna(subset=['price'])
    model.fit(df_no_nulls[['points']], df_no_nulls['price'])
    data.loc[data['price'].isna(), 'price'] = model.predict(data.loc[data['price'].isna(), ['points']])

print(data['price'].isnull().sum())

# =============================================================================
# 7. Desarrolle un breve ciclo para contar cuántas veces aparece cada una de 
# estas dos palabras en la columna de “description”. Imprima en consola el 
# número de veces que aparece estas palabras.
# =============================================================================
count = 0
for desc in data['description'].dropna():  
    if 'tropical' in desc.lower() or 'fruity' in desc.lower():  
        count += 1  

print(f'Hay {count} vinos que son "tropical" o "fruity"')

count = 0
for desc in data['description'].dropna():  
    if 'tropical' in desc.lower():  
        count += 1  

print(f'Hay {count} vinos que son "tropical"')

count = 0
for desc in data['description'].dropna():  
    if 'fruity' in desc.lower():  
        count += 1  

print(f'Hay {count} vinos que son "fruity"')

# =============================================================================
# 8. Transforme la variable categórica “country” en continua. 
# =============================================================================
frecuencia_x_pais = data_filtrado['country'].value_counts().reset_index()
print(frecuencia_x_pais)
ranking_paises = {pais: rank + 1 for rank, (pais, _) in enumerate(frecuencia_x_pais.itertuples(index=False))}
data_filtrado['Numero de Pais'] = data_filtrado['country'].map(ranking_paises)

# =============================================================================
# 9. Cree una nueva etiqueta llamada “stars” con el número de estrellas 
# correspondiente a cada reseña en el conjunto de datos. 
# Una puntuación de 95 o más cuenta como 3 estrellas, una puntuación de al menos
# 85 pero menos de 95 es 2 estrellas. Cualquier otra puntuación es 1 estrella
# =============================================================================
data_filtrado['stars'] = pd.cut(data_filtrado['points'], bins=[0, 85, 95, 100], 
                                       labels=[1, 2, 3])
print(data_filtrado['stars'].value_counts())

# =============================================================================
# Elimine las etiquetas que considere no aportan información relevante y/o útil.
# Justifique el por qué. Justifique su estrategia usando comentarios en el script.
# =============================================================================
data_final = data_filtrado.drop(['Unnamed: 0','country','designation','points',
                                 'province','region_1','region_2','taster_name',
                                 'taster_twitter_handle','variety','winery',
                                 'rango_puntos'],axis=1)

# =============================================================================
# Se eliminaron:
# Unnamed: 0: columna equivalente al indice
# country: reemplazado por Numero de pais
# designation: pienso que la designación no aporta información relevante
# points: reemplazado por stars 
# province, region_1 y region_2: información de la ubicación del vino, considero
#  irrelevante teniendo Numero de pais
# taster_name y taster_twitter_handle: datos del catador, considero irrelevante
# variety y winery: la variedad de la uva, el nombre de la bodega junto con la 
#  designación se encuentran en los titulos
# rango_puntos: columna creada para el punto 3
# =============================================================================