"""
Pedro Mauri Mtz - A01029143
Red LAM (Memoria Asociativa Lineal)
"""
import numpy as np

"""
Data:
    'a' (entrada) y respectiva 'b' (salida)
"""
a1 = [0, 1, 0, 1]
b1 = [1, 0, 1]

a2 = [0, 0, 0, 0]
b2 = [0, 0, 0]

a3 = [1, 1, 1, 1]
b3 = [1, 1, 1]

"""
Para calcular la matriz de pesos:
    Wij = Σ (2aij - 1)(2bij - 1)
"""
Wij = []

for i in range(len(a1)):
    temp = []
    for j in range(len(b1)):
        temp.append((2*a1[i] - 1)*(2*b1[j] - 1) + (2*a2[i] - 1)*(2*b2[j] - 1) + (2*a3[i] - 1)*(2*b3[j] - 1))
    Wij.append(temp)

print("\n-- Cálculos --\n")
print("Matriz de pesos:")
print(np.array(Wij))


"""
Para calcular el bias:
    bias = -1/2 * sum(Wij)
"""
bias = [0, 0, 0]

for i in range(len(b1)):
    bias[i] = -1/2 * sum(Wij[i])

print("Bias:")
print(np.array(bias))


"""
Para realizar pruebas:
    f(x * Wij + bias)
"""
def activacion(x):
    return 1 if x >= 0 else 0

a = [1, 1, 0, 0]
b = []

b = np.dot(a, Wij) + bias

for i in range(len(b)):
    temp = activacion(b[i])
    b[i] = temp

print("\n-- Pruebas --\n")
print(f"Entrada: {a}")
print(f"Salida calculada: {np.array(b)}")
