"""
Pedro Mauri Mtz - A01029143
Formula para calcular nuevos pesos para el perceptrón simple cuando haya un error
"""

"""
wi = wi + (step * e * xi)
"""
def new_weights(w, data, e, step):
    return w + step * e * data
