"""
Pedro Mauri Mt - A01029143
Algoritmo de aprendizaje Adaline puertas lógicas (Perceptrón mejorado)
"""

import numpy as np

w = [1.5, 0.5, 1.5]

# AND:
train_data_AND = np.array([
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1]
])

# OR:
train_data_OR = np.array([
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
])

# XOR:
train_data_XOR = np.array([
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
])

# XNOR:
train_data_XNOR = np.array([
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1]
])

alpha = 0.01
epocas = 1000

print(f"Pesos iniciales: {w}")

error_anterior = float("inf")

# Ciclo de entrenamiento
for epoch in range(epocas):

    error = 0

    for k in range(len(train_data_AND)):
        net = w[0] * train_data_AND[k][0] + w[1] * train_data_AND[k][1] + w[2] * train_data_AND[k][2]
        d = train_data_AND[k][3]
        
        e = d - net
    
        error += e ** 2

        for i in range(len(w)):
            w[i] += alpha * e * train_data_AND[k][i]

    error_actual = error / (2 * len(train_data_AND))
    
    print(f"\n--Época {epoch + 1}, Error: {error_actual}--")

    if abs(error_anterior - error_actual) < 1e-8:
        break

    error_anterior = error_actual

print(f"Pesos finales: {w}")

# Evaluación con pesos finales
for k in range(len(train_data_AND)):
    net = w[0] * train_data_AND[k][0] + w[1] * train_data_AND[k][1] + w[2] * train_data_AND[k][2]
    
    print(f"{train_data_AND[k][1]}, {train_data_AND[k][2]} -> {1 if net >= 0.5 else 0} (esperado: {train_data_AND[k][3]})")
