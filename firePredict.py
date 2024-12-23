import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Ruta del modelo y los datos
MODEL_PATH = "modelo_incendios.pkl"
DATA_PATH = "datos_historicos.csv"

# Cargar datos históricos
def cargar_datos(ruta):
    return pd.read_csv(ruta)

# Preprocesar datos
def preprocesar_datos(data):
    # Seleccionar características y etiquetas
    X = data[["temperatura", "humedad", "viento", "precipitacion", "mes", "dia_del_año", "hora", "latitud", "longitud"]]
    y = data["incendio"]
    return X, y

# Entrenar modelo
def entrenar_modelo(X, y):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X, y)
    return modelo

# Guardar modelo
def guardar_modelo(modelo, ruta):
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en {ruta}")

# Predecir probabilidad de incendio
def predecir_probabilidad(modelo, datos):
    probabilidad = modelo.predict_proba([datos])[0][1]  # Probabilidad de clase 1 (incendio)
    return probabilidad * 100

# Ejecución principal
if __name__ == "__main__":
    # Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    datos_historicos = cargar_datos(DATA_PATH)
    X, y = preprocesar_datos(datos_historicos)
    
    # Dividir datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    print("Entrenando modelo...")
    modelo = entrenar_modelo(X_train, y_train)
    
    # Evaluar modelo
    score = modelo.score(X_test, y_test)
    print(f"Precisión del modelo: {score * 100:.2f}%")
    
    # Guardar modelo entrenado
    guardar_modelo(modelo, MODEL_PATH)
    """  
    # Predecir probabilidad para nuevos datos
    print("Prediciendo probabilidad de incendio...")
    nuevos_datos = [35, 20, 15, 0, 7, 210, 14, -34.6, -58.4]  # Ejemplo: Temp, Humedad, Viento, Precipitación, Mes, Día, Hora, Lat, Lon
    probabilidad = predecir_probabilidad(modelo, nuevos_datos)
    print(f"Probabilidad de incendio: {probabilidad:.2f}%") """
