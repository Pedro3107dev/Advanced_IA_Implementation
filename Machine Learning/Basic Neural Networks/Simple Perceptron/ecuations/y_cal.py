"""
Pedro Mauri Mtz - A01029143
Fórmula de salida 'y' teniendo los pesos y datos (y = suma(wi * xi) + w0 * x0)
Y fórmula de activación (Si y >= 0 => y = 1 , Si y < 0 => y = 0)
"""

def y_cal(x0, x1, x2, x3, x4, x5, w0, w1, w2, w3, w4, w5):
    y = (x0 * w0) + (x1 * w1) + (x2 * w2) + (x3 * w3) + (x4 * w4) + (x5 * w5)
    return 1 if y >= 0 else 0
