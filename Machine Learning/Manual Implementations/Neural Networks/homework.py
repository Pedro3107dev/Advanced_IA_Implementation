"""
Pedro Mauri Mtz - A01029143
Homework (Proyecto 3) - Redes Neuronales BPN
"""
import os
import numpy as np

np.random.seed(42)

"""
Función que abre y lee el archivo de datos y guarda los valores en los respectivos vectores x e y
"""
def load_data():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Datasets", "digitos.txt")
    data = np.loadtxt(ruta, delimiter = None)
    x = data[:, :400]
    y = data[:, 400]

    return x, y

"""
Función para convertir las etiquetas en vectores one-hot
"""
def convert_labels_to_vectors(y, num_labels):
    y_adjusted = np.where(y == 10, 0, y)
    m = len(y_adjusted)
    y_one_hot = np.zeros((m, num_labels))
    y_one_hot[np.arange(m), y_adjusted.astype(int)] = 1

    return y_one_hot

"""
Función para inicializar los pesos y bias
"""
def randInicializacionPesos(L_in, L_out):
    epsilon = 0.12
    W = np.random.rand(L_out, L_in + 1) * 2 * epsilon - epsilon

    return W

"""
Función de activación sigmoidal y su gradiente
"""
def sigmoidalGradiente(z):
    g = 1 / (1 + np.exp(-z))

    return g * (1 - g)

"""
Función con ciclo de entrenamiento (Ajustes de pesos y bias segun su gradiente)
"""
def entrenaRN(input_layer_size, hidden_layer_size, num_labels, x, y):
    # Inicialización de pesos y bias
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)
    
    # Parámetros de entrenamiento
    m = x.shape[0] # Número de muestras de entrenamiento
    epochs = 20000
    alpha = 0.1

    for epoch in range(epochs):
        # Capa oculta
        Z1 = W1 @ x.T
        A1 = 1 / (1 + np.exp(-Z1))

        # Agregar bias a la capa oculta
        A1_with_bias = np.vstack((np.ones((1, A1.shape[1])), A1))

        # Capa de salida
        Z2 = W2 @ A1_with_bias
        A2 = 1 / (1 + np.exp(-Z2))

        # Cálculo del costo
        cost = -(1 / m) * np.sum(y.T * np.log(A2) + (1 - y.T) * np.log(1 - A2))

        # Calcular el error en la capa de salida
        delta_2 = A2 - y.T

        # Calcular el error en la capa oculta
        delta_1 = (W2[:, 1:].T @ delta_2) * sigmoidalGradiente(Z1)

        # Calcular gradientes
        grad_W2 = (1 / m) * (delta_2 @ A1_with_bias.T)
        grad_W1 = (1 / m) * (delta_1 @ x)
        
        # Actualizar pesos y bias
        W2 -= alpha * grad_W2
        W1 -= alpha * grad_W1

        # Imprimir costo:
        if epoch % 1000 == 0:
            print(f"Epoch {epoch + 1}, Cost: {cost:.4f}")

    return W1, W2

"""
Función para predecir las etiquetas usando la red neuronal ya entrenada
"""
def prediceRNYaEntrenada(x, W1, W2):
    Z1 = W1 @ x.T
    A1 = 1 / (1 + np.exp(-Z1))
    A1_with_bias = np.vstack((np.ones((1, A1.shape[1])), A1))

    Z2 = W2 @ A1_with_bias
    A2 = 1 / (1 + np.exp(-Z2))

    return np.argmax(A2.T, axis = 1)

# Cargar datos:
x, y = load_data()
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
y_one_hot = convert_labels_to_vectors(y, num_labels)

# Agregar una columna de 1s para el bias de la capa de entrada
x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))

# Partición entrenamiento / prueba (80% / 20%)
indices = np.random.permutation(x_with_bias.shape[0])
corte = int(0.8 * len(indices))
train_idx, test_idx = indices[:corte], indices[corte:]

# Entrenar red neuronal:
W1, W2 = entrenaRN(x.shape[1], hidden_layer_size, num_labels, x_with_bias[train_idx], y_one_hot[train_idx])

# Calcular presisión:
y_adjusted = np.where(y == 10, 0, y)
pred_train = prediceRNYaEntrenada(x_with_bias[train_idx], W1, W2)
print(f"\nAccuracy (entrenamiento): {np.mean(pred_train == y_adjusted[train_idx]) * 100:.2f}%")

# Testear red neuronal:
pred_test = prediceRNYaEntrenada(x_with_bias[test_idx], W1, W2)
print(f"Accuracy (prueba): {np.mean(pred_test == y_adjusted[test_idx]) * 100:.2f}%")
