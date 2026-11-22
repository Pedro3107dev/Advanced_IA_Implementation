"""
Pedro Mauri Mtz - A01029143
Backpropagation (BPN) algoritmo de aprendizaje para puertas lógicas
"""

import numpy as np

# AND:
train_data_AND = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# OR:
train_data_OR = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# XOR
train_data_XOR = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# XNOR:
train_data_XNOR = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

epochs = 20000
alpha = 0.1
np.random.seed(42)

w13 = np.random.rand()
w14 = np.random.rand()
w23 = np.random.rand()
w24 = np.random.rand()

b3 = np.random.rand()
b4 = np.random.rand()

b5 = np.random.rand()

w35 = np.random.rand()
w45 = np.random.rand()

# Data a utilizar
data = train_data_AND

# Entrenamiento
for epoch in range(epochs):

    error_epoch = 0

    for k in range(len(data)):
        x1 = data[k][0]
        x2 = data[k][1]
        y = data[k][2]

        """
        Calcular 'y'
        """

        net3 = x1 * w13 + x2 * w23 + b3
        o3 = 1 / (1 + np.exp(-net3))

        net4 = x1 * w14 + x2 * w24 + b4
        o4 = 1 / (1 + np.exp(-net4))

        net5 = o3 * w35 + o4 * w45 + b5
        o5 = 1 / (1 + np.exp(-net5))

        if o5 >= 0.5:
            o5_rounded = 1
        else:
            o5_rounded = 0

        """
        Calcular error y ajustar Ws y Bs
        """
        # w35 -= alpha * (-(y - o5) * o5 * (1 - o5) * o3)
        # w45 -= alpha * (-(y - o5) * o5 * (1 - o5) * o4)
        # b5 -= alpha * (-(y - o5) * o5 * (1 - o5))

        # w13 -= alpha * (-(y - o5) * o5 * (1 - o5) * w35 * o3 * (1 - o3) * x1)
        # w23 -= alpha * (-(y - o5) * o5 * (1 - o5) * w35 * o3 * (1 - o3) * x2)
        # b3 -= alpha * (-(y - o5) * o5 * (1 - o5) * w35 * o3 * (1 - o3))

        # w14 -= alpha * (-(y - o5) * o5 * (1 - o5) * w45 * o4 * (1 - o4) * x1)
        # w24 -= alpha * (-(y - o5) * o5 * (1 - o5) * w45 * o4 * (1 - o4) * x2)
        # b4 -= alpha * (-(y - o5) * o5 * (1 - o5) * w45 * o4 * (1 - o4))

        d5 = (o5 - y) * o5 * (1 - o5)
        d3 = d5 * w35 * o3 * (1 - o3)
        d4 = d5 * w45 * o4 * (1 - o4)

        w35 -= alpha * d5 * o3
        w45 -= alpha * d5 * o4
        b5 -= alpha * d5

        w13 -= alpha * d3 * x1
        w23 -= alpha * d3 * x2
        b3 -= alpha * d3

        w14 -= alpha * d4 * x1
        w24 -= alpha * d4 * x2
        b4 -= alpha * d4

        e = 1/2 * (y - o5) ** 2
        error_epoch += e

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Error: {round(error_epoch, 4)}")

# Testeo
for k in range(len(data)):
    x1, x2, y = data[k]
    
    o3 = 1 / (1 + np.exp(-(x1 * w13 + x2 * w23 + b3)))
    o4 = 1 / (1 + np.exp(-(x1 * w14 + x2 * w24 + b4)))
    o5 = 1 / (1 + np.exp(-(o3 * w35 + o4 * w45 + b5)))
    
    print(f"{x1} {x2} -> {o5:.4f} ({1 if o5 >= 0.5 else 0}) | esperado: {y}")
