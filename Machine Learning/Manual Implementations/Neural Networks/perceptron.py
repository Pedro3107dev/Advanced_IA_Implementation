"""
Pedro Mauri Mt - A01029143
Algoritmo de aprendizaje Perceptron puertas lógicas
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

for epoch in range(epocas):

    error = 0

    for k in range(len(train_data_AND)):
        net = w[0] * train_data_AND[k][0] + w[1] * train_data_AND[k][1] + w[2] * train_data_AND[k][2]
        d = train_data_AND[k][3]

        if net >= 0:
            y_prognosticada = 1
        else:
            y_prognosticada = 0
        
        e = d - y_prognosticada
    
        error += abs(e)

        for i in range(len(w)):
            w[i] += alpha * e * train_data_AND[k][i]
    
    print(f"\n--Época {epoch + 1}, Error: {error}--")

    if error == 0:
        break

print(f"Pesos finales: {w}")
