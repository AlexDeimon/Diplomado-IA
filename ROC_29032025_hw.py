# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:12:59 2025

@author: drago
"""

# =============================================================================
# Curva ROC con datos desbalanceados
# =============================================================================


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, ConfusionMatrixDisplay, roc_auc_score
from sklearn.datasets import make_classification

# =============================================================================
#
# #generar dos clases desbalanceadas
# =============================================================================

x,y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.8, 0.2], random_state=42)

# dividir nuestros datos para train y para test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# entrenamos
modelo = LogisticRegression(max_iter=500)
modelo.fit(x_train,y_train)

#realizar predicciones
y_pred = modelo.predict(x_test)
y_pred_prob = modelo.predict_proba(x_test)[:,1]

#creamos matriz de confusion
mc = confusion_matrix(y_test,y_pred)

mcd = ConfusionMatrixDisplay(mc, display_labels= ['Clase 0', 'Clase 1'])
mcd.plot()

# calcular la tasa de fpr, y tpr
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('Curva ROC (datos desbalanceados)')
plt.legend(loc = 'lower right')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
# =============================================================================
#
# #generar dos clases balanceadas
# =============================================================================


x,y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.5, 0.5], random_state=42)

# dividir nuestros datos para train y para test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# entrenamos
modelo = LogisticRegression(max_iter=500)
modelo.fit(x_train,y_train)

#realizar predicciones
y_pred = modelo.predict(x_test)
y_pred_prob = modelo.predict_proba(x_test)[:,1]

#creamos matriz de confusion
mc = confusion_matrix(y_test,y_pred)

mcd = ConfusionMatrixDisplay(mc)
mcd.plot()

# calcular la tasa de fpr, y tpr
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('Curva ROC (datos balanceados)')
plt.legend(loc = 'lower right')
plt.show()


# =============================================================================
# # =============================================================================
# # Objetivo: Ajustar el umbral de clasificación en el conjunto de datos balanceados/desbalanceado y observar los cambios en la matriz de confusión y la curva ROC.

                            # RESPONDE LAS SIGUIENTES PREGUNTAS

# # ¿Cómo cambia la matriz de confusión al ajustar el umbral de clasificación?
# # ¿Cómo afecta el ajuste del umbral a la curva ROC y al AUC-ROC?
# # =============================================================================
# =============================================================================



# =============================================================================
# Manejo de clases desbalanceadas con técnicas de sobremuestreo y submuestreo
# =============================================================================

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Aplicar SMOTE para sobremuestrear la clase minoritaria
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x_train, y_train)

# Entrenar el modelo con datos sobremuestreados
modelo_smote = LogisticRegression(max_iter=10000)
modelo_smote.fit(x_smote, y_smote)

# =============================================================================
# # =============================================================================
# # Objetivo: Completa el código de arriba para predecir nuevo valores con sobremuestreo
# # Calcula la matriz de confusión
# # Calcula la curva ROC
# # Calcula el área bajo la curva
# # Grafica la curva ROC
# # =============================================================================
# NOTA: Usar clases desbalanceadas
# =============================================================================

# Aplicar submuestreo para reducir la clase mayoritaria
rus = RandomUnderSampler(random_state=42)
x_rus, y_rus = rus.fit_resample(x_train, y_train)

# Entrenar el modelo con datos submuestreados
modelo_rus = LogisticRegression(max_iter=10000)
modelo_rus.fit(x_rus, y_rus)

# =============================================================================
# # =============================================================================
# # Objetivo: Completa el códigode arriba para predecir nuevo valores con submuestreo
# # Calcula la matriz de confusión
# # Calcula la curva ROC
# # Calcula el área bajo la curva
# # Grafica la curva ROC
# # =============================================================================
# NOTA: Usar clases desbalanceadas
# =============================================================================



# =============================================================================
                            # RESPONDE LAS SIGUIENTES PREGUNTAS
# =============================================================================
# =============================================================================
# # ¿Cómo cambia la matriz de confusión al aplicar técnicas de sobremuestreo y submuestreo?
# # ¿Qué técnica (SMOTE o submuestreo) proporciona un mejor rendimiento en términos de AUC-ROC?
# # ¿Qué observaciones puedes hacer sobre el impacto del balanceo de clases en el rendimiento del modelo?
# =============================================================================








