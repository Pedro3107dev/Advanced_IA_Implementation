"""
Pedro Mauri Mtz - A01029143
Ejercicio de Perceptrón Simple (Main script)
"""

import numpy as np

from formulas.y_cal import y_cal
from formulas.error import error
from formulas.new_w import new_weights

"""
Estructura de la Data:
x0, x1, x2, x3, x4, y
"""

"""
Datos de entrenamieinto
"""
train_data = np.array([
    [1, 0.8, 0.2, 0.9, 0.7, 0.5, 1],
    [1, -0.5, 0.7, 0.3, 0.2, 1.1, 0],
    [1, 0.9, 0.1, 0.8, 0.9, 0.6, 1],
    [1, 0.0, 0.5, 0.4, 0.3, 0.8, 0],
    [1, 0.7, 0.3, 0.7, 0.8, 0.4, 1],
    [1, -0.1, 0.8, 0.1, 0.1, 1.5, 0],
    [1, 0.6, 0.1, 0.6, 0.7, 0.5, 1],
    [1, 0.2, 0.4, 0.2, 0.4, 0.9, 0],
    [1, 1.0, 0.0, 1.0, 1.0, 0.3, 1],
    [1, -0.2, 0.6, 0.0, 0.0, 1.2, 0],
    [1, 0.5, 0.2, 0.7, 0.6, 0.3, 1],
    [1, 0.1, 0.3, 0.4, 0.5, 0.7, 0]
])

train_x0 = train_data[:, 0]
train_x1 = train_data[:, 1]
train_x2 = train_data[:, 2]
train_x3 = train_data[:, 3]
train_x4 = train_data[:, 4]
train_x5 = train_data[:, 5]
train_y = train_data[:, 6]

"""
Datos de pruebas
"""
test_data = np.array([
    [1, 0.9, 0.0, 0.9, 0.9, 0.6, 1],
    [1, -0.3, 0.5, 0.1, 0.2, 1.0, 0],
    [1, 0.8, 0.1, 0.8, 0.8, 0.4, 1]
])

test_x0 = test_data[:, 0]
test_x1 = test_data[:, 1]
test_x2 = test_data[:, 2]
test_x3 = test_data[:, 3]
test_x4 = test_data[:, 4]
test_x5 = test_data[:, 5]
test_y = test_data[:, 6]

"""
Pesos iniciales (random)
"""
w0 = np.random.uniform(-1, 1)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
w3 = np.random.uniform(-1, 1)
w4 = np.random.uniform(-1, 1)
w5 = np.random.uniform(-1, 1)

"""
Guardar pesos iniciales en las wi
"""
w0i = w0
w1i = w1
w2i = w2
w3i = w3
w4i = w4
w5i = w5

"""
Valor del paso para el cambio en los pesos
"""
step = 0.01

"""
Maximo numero de cambios posibles de los pesos para el entrenamiento
"""
max_errors = 100

"""
Ciclo for para entrenamiento
"""
for iteration in range(max_errors):

    """
    Contador para cuantas veces se tuvieron que cambiar los pesos
    """
    errors_in_iteration = 0

    """
    Indicador de cuantas iteraciones se han echo en los datos
    """
    print(f"\n--Iteración {iteration + 1}--")

    for i in range(len(train_data)):

        print(f"\nDato {i+1}/{len(train_data)}")
        y = y_cal(train_x0[i], train_x1[i], train_x2[i], train_x3[i], train_x4[i], train_x5[i], w0, w1, w2, w3, w4, w5)

        e = error(train_y[i], y)
        print(f"Error: {e}")

        if e != 0:
            print(f"Error encontrado, actualizando pesos...")
            errors_in_iteration += 1
            w0 = new_weights(w0, train_x0[i], e, step)
            w1 = new_weights(w1, train_x1[i], e, step)
            w2 = new_weights(w2, train_x2[i], e, step)
            w3 = new_weights(w3, train_x3[i], e, step)
            w4 = new_weights(w4, train_x4[i], e, step)
            w5 = new_weights(w5, train_x5[i], e, step)
            break

    if errors_in_iteration == 0:
        """
        Se acabó el entrenamiento en menos de 100 iteraciones, ahora se muetran los pesos iniciales y finales
        """
        print(f"\n\n-Entrenamiento completado en {iteration + 1} iteraciones!!-")
        print(f"\nPesos iniciales:\n{w0i, w1i, w2i, w3i, w4i, w5i}")
        print(f"\nPesos finales:\n({w0:.4f}, {w1:.4f}, {w2:.4f}, {w3:.4f}, {w4:.4f}, {w5:.4f})")


        """
        Ciclo for para realizar las pruebas
        """
        print("\n\n--Resultados de pruebas--")

        for j in range(len(test_data)):
            """
            Contador de cuantos errores se encontraron en la etapa de pruebas
            """
            errors_in_test = 0

            print(f"\nDato {j+1}/{len(test_data)}")
            y = y_cal(test_x0[j], test_x1[j], test_x2[j], test_x3[j], test_x4[j], test_x5[j], w0, w1, w2, w3, w4, w5)
            
            e = error(test_y[j], y)
            print(f"Error: {e}")

            if e != 0:
                print(f"Error encontrado")
                errors_in_test += 1
            print(f"Errores en pruebas: {errors_in_test}")
        
        """
        Acaba codigo (interrumpe el ciclo for de max_iteraciones)
        """
        break

    elif errors_in_iteration != 0 and iteration == max_errors - 1:
        print(f"\n-Entrenamiento no completado en {max_errors} iteraciones!!-")
        print(f"\nPesos iniciales:\n{w0i, w1i, w2i, w3i, w4i, w5i}")
        print(f"\nPesos finales:\n({w0:.4f}, {w1:.4f}, {w2:.4f}, {w3:.4f}, {w4:.4f}, {w5:.4f})")
