import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib  # Agregar este import

# Constantes y configuración
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Mejorar la generación de datos con más muestras
data = pd.DataFrame({
    "temperatura": [30, 25, 40, 35, 20, 28, 33, 26, 38, 22, 31, 36, 27, 34, 29],
    "humedad": [20, 50, 10, 15, 60, 30, 25, 55, 12, 45, 35, 18, 52, 28, 40],
    "viento": [10, 5, 20, 15, 5, 12, 18, 6, 22, 8, 14, 16, 7, 19, 11],
    "incendio": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
})

# Función para visualizar datos
def visualizar_datos(data):
    plt.figure(figsize=(10, 6))
    sns.pairplot(data, hue="incendio")
    plt.show()

# Función para preparar datos
def preparar_datos(data, test_size, random_seed):
    X = data[["temperatura", "humedad", "viento"]]
    y = data["incendio"]
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)

# Función para entrenar y evaluar modelo
def entrenar_modelo(X_train, y_train, X_test, y_test):
    # Configuración de hiperparámetros más completa
    parametros = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        parametros,
        cv=5,  # Aumentado de 3 a 5
        scoring='accuracy',
        n_jobs=-1  # Usar todos los núcleos disponibles
    )
    
    grid_search.fit(X_train, y_train)
    mejor_modelo = grid_search.best_estimator_
    
    # Evaluación del modelo
    score_train = mejor_modelo.score(X_train, y_train)
    score_test = mejor_modelo.score(X_test, y_test)
    
    return mejor_modelo, score_train, score_test, grid_search.best_params_

# Ejecución principal
if __name__ == "__main__":
    # Preparar datos
    X_train, X_test, y_train, y_test = preparar_datos(data, TEST_SIZE, RANDOM_SEED)
    
    # Entrenar y evaluar
    modelo, score_train, score_test, mejores_params = entrenar_modelo(X_train, y_train, X_test, y_test)
    
    # Guardar el modelo entrenado
    joblib.dump(modelo, "modelo_incendios.pkl")
    print("\nModelo guardado como 'modelo_incendios.pkl'")
    
    # Mostrar resultados
    print(f"Precisión en entrenamiento: {score_train * 100:.2f}%")
    print(f"Precisión en prueba: {score_test * 100:.2f}%")
    print("\nMejores parámetros encontrados:")
    for param, valor in mejores_params.items():
        print(f"{param}: {valor}")


