# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:06:09 2025

@author: diego
"""

import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

# =============================================================================
# Parte 1: EDAS. Analisis Exploratorio de Datos
# =============================================================================

# Leer el dataset
data = pd.read_csv('Titanic - train.csv')

# Visualización rapida de los datos
data.head()

# datos nulos
data.isnull().sum()

# Visualización proporción de sobrevivientes
f,ax = pl.subplots(1,2)
data['Survived'].value_counts().plot.pie(autopct = '%1.1f%%', ax = ax[0])
ax[0].set_title('Survived')

sb.countplot(x = 'Survived', data=data,ax=ax[1])
ax[1].set_title('Survived')
pl.show()

# =============================================================================
# Caracteristicas en ciencia de datos
# =============================================================================
# 1. Categoricas
# 2. Ordinales
# 3. Continuas

# =============================================================================
# CATEGORICAS
# =============================================================================

data.groupby(['Sex', 'Survived'])['Survived'].count()

f,ax = pl.subplots(1,2, figsize = (20,10))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')
sb.countplot(x = 'Sex', hue = 'Survived', data = data, ax = ax[1])
pl.show()

f,ax = pl.subplots(1,2, figsize = (20,10))
data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Pclass')
sb.countplot(x = 'Pclass', hue = 'Survived', data = data, ax = ax[1])
pl.show()

# =============================================================================
# ORDINALES
# =============================================================================
sb.catplot(data = data, x = 'Pclass', y = 'Survived', hue = 'Sex', kind = "point")
pl.show()

# Analizar la relación entre la tarifa pagada (Fare) y la probabilidad de supervivencia
f,ax = pl.subplots(1, 2, figsize=(20, 10))

sb.boxplot(data=data, x='Survived', y='Fare', ax=ax[0])
ax[0].set_title('Survived vs Fare (Boxplot)')
ax[0].set_ylabel('Fare')
ax[0].set_xlabel('Survived')

sb.histplot(data=data, x='Fare', hue='Survived', kde=True, bins=30, ax=ax[1], multiple='stack')
ax[1].set_title('Fare Distribution by Survived')
ax[1].set_xlabel('Fare')

pl.show()

# =============================================================================
# CONTINUAS
# =============================================================================
data['Age'].max()
data['Age'].min()
data['Age'].mean()

f,ax = pl.subplots(1,2)
sb.violinplot(data = data, x = 'Pclass', y = 'Age', hue = 'Survived', split = True, ax = ax[0])
sb.violinplot(data = data, x = 'Sex', y = 'Age', hue = 'Survived', split = True, ax = ax[1])
pl.show()

#distribucion de supervivencia por edad
f,ax = pl.subplots(1, 2, figsize=(20, 10))

sb.histplot(data=data, x='Age', hue='Survived', kde=True, bins=30, ax=ax[0], multiple='stack')
ax[0].set_title('Age Distribution by Survived')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Count')

sb.boxplot(data=data, x='Survived', y='Age', ax=ax[1])
ax[1].set_title('Survived vs Age (Boxplot)')
ax[1].set_xlabel('Survived')
ax[1].set_ylabel('Age')

pl.show()

# Análisis de la relación entre la cantidad de familiares a bordo (SibSp + Parch) y la supervivencia
data['FamilySize'] = data['SibSp'] + data['Parch']

f,ax = pl.subplots(1, 2, figsize=(20, 10))

sb.boxplot(data=data, x='Survived', y='FamilySize', ax=ax[0])
ax[0].set_title('Survived vs Family Size (Boxplot)')
ax[0].set_xlabel('Survived')
ax[0].set_ylabel('Family Size')

sb.countplot(data=data, x='FamilySize', hue='Survived', ax=ax[1])
ax[1].set_title('Family Size Distribution by Survived')
ax[1].set_xlabel('Family Size')
ax[1].set_ylabel('Count')

pl.show()