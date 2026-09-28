"""
Pedro Mauri Mtz - A01029143
Red BAM (Memoria Asociativa Bidireccional)
"""
import numpy as np

"""
Data:
    'x' (entrada) y respectiva 'y' (salida)
"""
x1 = [-1, 1, -1]
y1 = [1, -1, 1]

x2 = [-1, -1, -1]
y2 = [-1, -1, -1]

x3 = [1, 1, 1]
y3 = [1, 1, 1]


"""
Para calcular la matriz de pesos:
    Wij = Σ (xi * yj)
"""
Wij = []

for i in range(len(x1)):
    temp = []
    for j in range(len(y1)):
        temp.append((x1[i] * y1[j]) + (x2[i] * y2[j]) + (x3[i] * y3[j]))
    Wij.append(temp)

print("\n-- Cálculos --\n")
print("Matriz de pesos:")
print(np.array(Wij))


"""
Para realizar pruebas:
    y = f(x * Wij)
"""
def activacion(x):
    return 1 if x >= 0 else -1

x = [1, -1, 1]
y = []

y = np.dot(x, Wij)

for i in range(len(y)):
    temp = activacion(y[i])
    y[i] = temp

print("\n-- Pruebas --\n")
print(f"Entrada: {x}")
print(f"Salida calculada: {np.array(y)}")
