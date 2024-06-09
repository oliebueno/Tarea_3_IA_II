import pandas as pd


# Funcion para leer los datos

def reader(path_data):
    data = pd.read_csv(path_data, header=None)  # Lectura de los datos
    data.columns = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width', 'type']  # Agregar columnas
    return data


# Función para convertir las etiquetas en binario

def change_label_bin(type):
    if type == 'Iris-setosa':
        return 1
    else:
        return 0
