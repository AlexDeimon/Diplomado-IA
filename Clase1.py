# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:06:09 2025

@author: diego
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
f,ax = plt.subplots(1,2)
data['Survived'].value_counts().plot.pie(autopct = '%1.1f%%', ax = ax[0])
ax[0].set_title('Survived')

sns.countplot(x = 'Survived',data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
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

# ------------------Sex
f,ax = plt.subplots(1,2, figsize = (20,10))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot(x = 'Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Sex: Survived vs Died')
plt.show()

# ------------------Class
f,ax = plt.subplots(1,2, figsize = (20,10))
data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Pclass')
sns.countplot(x = 'Pclass', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Died')
plt.show()


# =============================================================================
# ORDINALES
# =============================================================================
sns.catplot(data=data, x = 'Pclass', y = 'Survived', hue = 'Sex', kind ="point")
plt.show()

# Analizar la relación entre la tarifa pagada (Fare) y la probabilidad de supervivencia
f, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.boxplot(data=data, x='Survived', y='Fare', ax=ax[0])
ax[0].set_title('Survived vs Fare (Boxplot)')
ax[0].set_ylabel('Fare')
ax[0].set_xlabel('Survived')

sns.histplot(data=data, x='Fare', hue='Survived', kde=True, bins=30, ax=ax[1], multiple='stack')
ax[1].set_title('Fare Distribution by Survived')
ax[1].set_xlabel('Fare')

plt.show()

# =============================================================================
# CONTINUAS
# =============================================================================
data['Age'].max()
data['Age'].min()
data['Age'].mean()

f,ax = plt.subplots(1,2)
sns.violinplot(data=data, x = 'Pclass', y= 'Age',hue = 'Survived',split = True, ax=ax[0])
ax[0].set_title('Pclass vs Age vs Survived')

sns.violinplot(data=data, x = 'Sex', y= 'Age',hue = 'Survived',split = True, ax=ax[1])
ax[1].set_title('Sex vs Age vs Survived')

plt.show()

#distribucion de supervivencia por edad
f, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.histplot(data=data, x='Age', hue='Survived', kde=True, bins=30, ax=ax[0], multiple='stack')
ax[0].set_title('Age Distribution by Survived')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Count')

sns.boxplot(data=data, x='Survived', y='Age', ax=ax[1])
ax[1].set_title('Survived vs Age (Boxplot)')
ax[1].set_xlabel('Survived')
ax[1].set_ylabel('Age')

plt.show()


# Análisis de la relación entre la cantidad de familiares a bordo (SibSp + Parch) y la supervivencia
data['FamilySize'] = data['SibSp'] + data['Parch']

f, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.boxplot(data=data, x='Survived', y='FamilySize', ax=ax[0])
ax[0].set_title('Survived vs Family Size (Boxplot)')
ax[0].set_xlabel('Survived')
ax[0].set_ylabel('Family Size')

sns.countplot(data=data, x='FamilySize', hue='Survived', ax=ax[1])
ax[1].set_title('Family Size Distribution by Survived')
ax[1].set_xlabel('Family Size')
ax[1].set_ylabel('Count')

plt.show()


# =============================================================================
# CLASE 2
# =============================================================================
data['Inicial'] = 0

#extraemos los prefijos del data frame
for i in data:
    data['Inicial'] = data.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(data.Inicial, data.Sex)

# a remplazar los prefijos con otro valor

data['Inicial'].replace(['Capt','Col', 'Countess','Don','Dr','Jonkheer','Lady','Major','Mlle','Mme','Ms','Rev','Sir'],['Mr','Mr','Mrs','Mr','Mr','Mr','Miss','Mr','Miss','Miss','Miss','Mr','Mr'],inplace = True)

#calcular los promedio por grupo
data.groupby('Inicial')['Age'].mean()

#rellenar los datos faltantes (NaN)
data.loc[(data.Age.isnull()) & (data.Inicial == 'Master'), 'Age'] = 4
data.loc[(data.Age.isnull()) & (data.Inicial == 'Miss'), 'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Inicial == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Inicial == 'Mrs'), 'Age'] = 35

data.isnull().any()

f,ax = plt.subplots(1,2)
data[data['Survived'] == 0].Age.plot.hist(ax=ax[0],bins=20, edgecolor='black')
x0 = list(range(0,80,5))
ax[0].set_xticks(x0)
data[data['Survived'] == 1].Age.plot.hist(ax=ax[1],bins=20, edgecolor='black')
ax[1].set_xticks(x0)
plt.show()

# =============================================================================
# #Procesemos la etiqueta de embarque
# =============================================================================

data.isnull().sum()

puerto_sexo = pd.crosstab([data.Embarked, data.Pclass], [data.Sex,data.Survived], margins=True)
sns.heatmap(puerto_sexo, cmap='summer', annot=True, fmt='d')

#evaluamos la probalidad de supervivencia en funcion de el puerto de embarque
sns.catplot(data=data, x='Embarked', y = 'Survived', kind='point')
plt.show()

#discretzar el numero de pasajeros por puerto y la clase y el sexo

f,ax = plt.subplots(2,2)
sns.countplot(data=data, x = 'Embarked', ax = ax[0,0])
ax[0,0].set_title('Numero de pasajero embarcados')

sns.countplot(data=data, x = 'Embarked', hue = 'Sex', ax = ax[0,1])
ax[0,1].set_title('Hombre/mujer vs puerto')

sns.countplot(data=data, x = 'Embarked', hue = 'Survived', ax = ax[1,0])
ax[1,0].set_title('Supervicencia vs puerto')

sns.countplot(data=data, x = 'Embarked', hue = 'Pclass', ax = ax[1,1])
ax[1,1].set_title('Clase vs puerto')

plt.show()


#asignos valor a los faltantes de la etiqueta Embarked

data['Embarked'].fillna('S', inplace=True)
data.Embarked.isnull().any()

# =============================================================================
# #procesemos la etiqueta Fare (variable continua)
# =============================================================================

data['Fare'].max()
data['Fare'].min()
data['Fare'].mean()

f,ax = plt.subplots(1,3)
sns.distplot(data[data['Pclass'] == 1 ].Fare, ax=ax[0])
ax[0].set_title('Precios en primera clase')
sns.distplot(data[data['Pclass'] == 2 ].Fare, ax=ax[1])
ax[1].set_title('Precios en segunda clase')
sns.distplot(data[data['Pclass'] == 3 ].Fare, ax=ax[2])
ax[2].set_title('Precios en tercera clase')

plt.show()
#por analisis se concluye que el precio no es una variable crítica


# =============================================================================
# #procesemos la etiqueta Siblings y Parch (variable continua)
# =============================================================================

familia = pd.crosstab([data.SibSp], data.Survived)
sns.heatmap(familia, cmap = 'summer', annot = True,fmt='d')

pd.crosstab(data.SibSp,data.Pclass)

#el numero de parientes influye (??) en la tasa de supervivencia


# =============================================================================
# =============================================================================
# # Parte 2: Limpieza de datos
# # =============================================================================
#
# =============================================================================


# procesamiento de la etiqueta de edad

data['Grupo_edad'] = 0
data.loc[(data['Age'] >= 0) & (data['Age'] <= 5), 'Grupo_edad'] = 1
data.loc[(data['Age'] > 5) & (data['Age'] <= 10), 'Grupo_edad'] = 2
data.loc[(data['Age'] > 10) & (data['Age'] <= 15), 'Grupo_edad'] = 3
data.loc[(data['Age'] > 15) & (data['Age'] <= 20), 'Grupo_edad'] = 4
data.loc[(data['Age'] > 20) & (data['Age'] <= 25), 'Grupo_edad'] = 5
data.loc[(data['Age'] > 25) & (data['Age'] <= 30), 'Grupo_edad'] = 6
data.loc[(data['Age'] > 30) & (data['Age'] <= 35), 'Grupo_edad'] = 7
data.loc[(data['Age'] > 35) & (data['Age'] <= 40), 'Grupo_edad'] = 8
data.loc[(data['Age'] > 40) & (data['Age'] <= 45), 'Grupo_edad'] = 9
data.loc[(data['Age'] > 45) & (data['Age'] <= 50), 'Grupo_edad'] = 10
data.loc[(data['Age'] > 50) & (data['Age'] <= 55), 'Grupo_edad'] = 11
data.loc[(data['Age'] > 55) & (data['Age'] <= 60), 'Grupo_edad'] = 12
data.loc[(data['Age'] > 60) & (data['Age'] <= 65), 'Grupo_edad'] = 13
data.loc[(data['Age'] > 65) & (data['Age'] <= 70), 'Grupo_edad'] = 14
data.loc[(data['Age'] > 70) & (data['Age'] <= 75), 'Grupo_edad'] = 15
data.loc[(data['Age'] > 75) & (data['Age'] <= 80), 'Grupo_edad'] = 16

data['Grupo_edad'].mean()

grupo_edad = data['Grupo_edad'].value_counts().to_frame()
sns.heatmap(grupo_edad, cmap='summer', annot = True, fmt = 'd')

sns.catplot(data=data, x='Grupo_edad',y ='Survived',col ='Pclass',kind='point')

# procesamiento de la etiqueta de familiares (SibSp)

data['Familia'] = 0
data['Familia'] = data['SibSp'] + data['Parch']

sns.catplot(data=data, x='Familia',y='Survived', kind='point')

#procesamiento de tarifa (meramente didáctico, convertremos variabel continua a ordinal)

data['Grupo_tarifa']= pd.qcut(data['Fare'],4)

rango_tarifa = data.groupby(['Grupo_tarifa'])['Survived'].mean().to_frame()
sns.heatmap(rango_tarifa, cmap='summer', annot=True, fmt='.3f')

sns.catplot(data=data, x='Grupo_tarifa',y='Survived', kind='point')

data['Categoria_tarifa'] = 0
data.loc[(data['Fare'] <= 7.91), 'Categoria_tarifa'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Categoria_tarifa'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31.0), 'Categoria_tarifa'] = 2
data.loc[(data['Fare'] > 31.0) & (data['Fare'] <= 512.329), 'Categoria_tarifa'] = 3

sns.catplot(data=data, x='Categoria_tarifa',y='Survived',hue='Sex', kind='point')

#procesamiento de la etiqueta Sex (convertir varibales string a numericas)

data['Sex'].replace(['male', 'female'], [0,1],inplace=True)
data['Embarked'].replace(['S', 'C','Q'], [0,1,2],inplace=True)

# procamiento de etiquetas que no se utilizaran (eliminarlas)

data.drop(['PassengerId', 'Name','Age','SibSp','Parch','Ticket','Cabin','Grupo_tarifa','Inicial', 'Fare'], axis=1,inplace=True)

sns.heatmap(data.corr(),annot=True, cmap='winter', fmt='.3f')