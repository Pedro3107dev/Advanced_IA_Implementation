"""
Pedro Mauri Mtz - A01029143
Red de hopfield
"""
# import numpy as np

"""
Valores iniciales:
C0 = Patrón 1
C1 = Patrón 2
c2 = Patrón 3
n = Número de neuronas (Tamaño de los patrones)
Wij = Matriz de pesos
"""

# c0 = [1, 1, 1, 1, -1, -1]
# c1 = [1, 1, -1, -1, 1, 1]
# c2 = [-1, -1, 1, 1, 1, 1]

# from numbers import c0, c1, c2, c3, c4, c5, c6, c7, c8, c9
c0 = [1, 0, 1, 0]
c1 = [0, 1, 0, 1]

n = len(c0)
Wij = []

"""
Calcular Matriz de pesos:
"""
for i in range(n):
    temp = []
    for j in range(n):
        if i == j:
            temp.append(0)
        else:
            temp.append(1/n * ((c0[i] * c0[j]) + (c1[i] * c1[j])))
            # Wij.append(1/n * ((c0[i] * c0[j]) + (c1[i] * c1[j]) + (c2[i] * c2[j])))
    Wij.append(temp)
"""
Printear matriz de pesos
"""
# Wij = np.array(Wij).reshape(n, n)
print("\nMatriz de pesos:")
for i in range(n):
    print(Wij[i])

"""
Función de activación:
    Si x >= 0: return 1
    Si x < 0: return -1
"""
def funcion_activacion(input):
    for i in range(len(input)):
        if input[i] >= 0:
            input[i] = 1
        else:
            input[i] = -1
    return input


"""
Tras realizar la matriz de pesos, veamos que patrones reconoce...

Función de Detección de patrón:

Convertir input a arreglo de Numpy
Aplica la función de activación a la entrada
Printear Input

Producto punto entre la matriz de pesos y el vector de entrada

Aplica la función de activación a la salida
Printear Output

Comparar patrón dado con patrón calculado por el producto punto
"""
def deteccion_patron(input, Wij, iteraciones, iteracion): 
    """
    Cálculo de salida
    """

    #input = np.array(input)
    input = funcion_activacion(input)
    print("\nInput:")
    print(input)

    #output = Wij.dot(input)

    output = funcion_activacion(output)
    print("\nOutput:")
    print(output)

    """
    Comparación de patrones
    """
    iteracion += 1
    print(f"--- Iteración {iteracion} ---")
    
    correct_value_counter = 0
    for i in range(len(output)):
        print(f"\nEntrada: {input[i]}, Salida: {output[i]}")
        if (input[i] == output[i]):
            print("Los valores coinciden!!")
            correct_value_counter += 1
        else:
            print("Los valores no coinciden...")

    if correct_value_counter == len(output):
        print("\nLa red a convergido!!")
    elif iteracion < iteraciones:
        print("El patrón no ha sido reconocido...\nIntentando otra vez...")
        deteccion_patron(input, Wij, iteraciones, iteracion)
    else:
        print(f"Se han alcanzado el número máximo de iteraciones... ({iteraciones})")

# iteracion = 0
# iteraciones = 1
# input = [1, 1, 1, 1, -1, -1]
"""
inputs examen:
t1 = [1, 0, 1, 0]
t2 = [1, 0, 0, 0]
t3 = [0, 1, 0, 1]
t4 = [0, 0, 0, 1]
t5 = [1, 1, 0, 0]
t6 = [1, 0, 0, 1]
t7 = [0, 1, 1, 0]
t8 = [0, 0, 1, 1]
"""

# deteccion_patron(input, Wij, iteraciones, iteracion)





# """
# Andrés Jaramillo - A01029079
# Pedro Mauri Mtz - A01029143
# Red de hopfield
# """
# import numpy as np

# # from test_data_numbers import c0, c1, c2, c3, c4, c5, c6, c7, c8, c9

# """
# Conversión de 0s a -1s y 1s a 1s
# """
# def zeroToNegOneMatrix(cn):
#     for i in range(len(cn)):
#         for j in range(len(cn[i])):
#             if cn[i][j] == 0:
#                 cn[i][j] = -1
#     return cn

# def zeroToNegOneVector(c):
#     for i in range(len(c)):
#         if c[i] == 0:
#             c[i] = -1
#     return c

# """
# Para calcular la matriz de pesos:
#     Wij = 1/n * Σ (cμ[i] * cμ[j])
# """
# def W(c1, c2):
#     return c1 * c2

# def calcWij(cn):
#     Wij = []

#     for i in range(len(cn[1])):
#         Wij.append([])
#         for j in range(len(cn[1])):
#             sum = []
#             num = 0
#             for m in range(len(cn)):
#                 sum.append(W(cn[m][i], cn[m][j]))
#             for s in sum:
#                 num += s
#             if i == j:
#                 Wij[i].append(0.0)
#             else:
#                 Wij[i].append(1/len(cn) * num)
    
#     print("=" * 21)
#     print("|  Matriz de pesos  |")
#     print("=" * 21)

#     for i in range(len(Wij)):
#         print(f"| {Wij[i]} |")

#     return Wij

# """
# Para realizar pruebas:
#     y = f(Wij * x)
# """
# def activacion(x):
#     return 1 if x >= 0 else -1

# def prueba(c, Wij, iteraciones, iteracion):
#     print(f"Iteración {iteracion}/{iteraciones}:")
#     d = []
#     d = np.dot(c, Wij)

#     for i in range(len(d)):
#         temp = activacion(d[i])
#         d[i] = temp

#     print("=" * ((2 * len(c) + 4) + 18))
#     print(f"|  Entrada  | {c} |")
#     print("=" * ((2 * len(c) + 4) + 18))
#     print("=" * ((2 * len(d) + 4) + 20))
#     print(f"|  Salida  | {d} |")
#     print("=" * ((2 * len(d) + 4) + 20))

#     """
#     Comparación de patrones
#     """
#     correct_value_counter = 0
#     for i in range(len(d)):
#         # print(f"\nEntrada: {c[i]}, Salida: {d[i]}")
#         if (c[i] == d[i]):
#             # print("Los valores coinciden!!")
#             correct_value_counter += 1
#         else:
#             # print("Los valores no coinciden...")
#             correct_value_counter += 0

#     """
#     Recurión para intentar converger al patrón correcto
#     """
#     if correct_value_counter == len(d):
#         print("La red a convergido!!")
#     elif iteracion < iteraciones:
#         iteracion += 1
#         print("El patrón no ha sido reconocido...\nIntentando otra vez...")
#         prueba(d, Wij, iteraciones, iteracion)
#     else:
#         print(f"Se han alcanzado el número máximo de iteraciones... ({iteraciones})")
    
#     return d

# """
# Sección main:
# """
# def mainHopfield(cn, c, iteraciones):
#     # Convertir 0s a -1s en cn y c
#     zeroToNegOneMatrix(cn)
#     zeroToNegOneVector(c)

#     # Calcular matriz de pesos
#     print("\n--- Entrenamiento ---")
#     Wij = calcWij(cn)

#     # Pruebas:
#     print("\n--- Pruebas ---")
#     iteracion = 1
#     prueba(c, Wij, iteraciones, iteracion)




# # Datos para entrenar:
# c0 = [1, 0, 1, 0]
# c1 = [0, 1, 0, 1]

# # Vector de vectores de datos de entrenamiento
# cn = [c0, c1]

# """
# test inputs de examen:
# t1 = [1, -1, 1, -1]
# t2 = [1, -1, -1, -1]
# t3 = [-1, 1, -1, 1]
# t4 = [-1, -1, -1, 1]
# t5 = [1, 1, -1, -1]
# t6 = [1, -1, -1, 1]
# t7 = [-1, 1, 1, -1]
# t8 = [-1, -1, 1, 1]
# """

# c = [1, -1, 1, -1]
# iteraciones = 5

# mainHopfield(cn, c, iteraciones)
