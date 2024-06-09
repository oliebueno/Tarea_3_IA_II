import sys
import numpy as np
import Utils.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Programa principal

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Uso: python main.py <nombre_archivo1>")
    else:
        # Lectura de los datos
        data = utils.reader(sys.argv[1])

        # Se crean las etiquetas bianrias, 1 para Iris-setosa y 0 para las demás
        data['binary_label'] = data['type'].apply(utils.change_label_bin)

        # Se separan las caracteristicas de las etiquetas
        x = data.drop(['type', 'binary_label'], axis=1)
        y_binary = data['binary_label']
        y_ternary = data['type']

        # Se separan los conjuntos de tados para cada modelo

        x_train_bin, x_test_bin, y_train_bin, y_test_bin = train_test_split(
            x, y_binary, test_size=0.2, random_state=42)

        x_train_ter, x_test_ter, y_train_ter, y_test_ter = train_test_split(
            x, y_ternary, test_size=0.2, random_state=42)

        # Se crea el modelo de regresión logística para el clasificador binario

        binary_model = LogisticRegression()
        binary_model.fit(x_train_bin, y_train_bin)

        # Se prueba el modelo

        binary_pred = binary_model.predict(x_test_bin)

        # Se calculan lás métricas

        bin_accuracy = accuracy_score(y_test_bin, binary_pred)
        bin_recall = recall_score(y_test_bin, binary_pred)
        bin_matrix = confusion_matrix(y_test_bin, binary_pred)

        # Se crea el modelo de regresión logística para el clasificador ternario

        ternary_model = LogisticRegression(solver='lbfgs')
        ternary_model.fit(x_train_ter, y_train_ter)

        # Se prueba el modelo

        ternary_pred = ternary_model.predict(x_test_ter)

        # Se crea la matriz de confusión

        ter_matrix = confusion_matrix(y_test_ter, ternary_pred)

        # Se desplegan los resultados

        print("Métricas para el clasificador binario:")
        print(f"Precisión: {bin_accuracy}")
        print(f"Recall: {bin_recall}")
        print(f"Matriz de Confusión:\n{bin_matrix}")

        print("\nMétricas para el clasificador ternario:")
        print(f"Matriz de Confusión:\n{ter_matrix}")
