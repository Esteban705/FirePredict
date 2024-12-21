import pandas as pd
import joblib

# Cargar el modelo entrenado (asegúrate de haberlo guardado previamente como "modelo_incendios.pkl")
try:
    modelo = joblib.load("modelo_incendios.pkl")
except FileNotFoundError:
    print("El modelo 'modelo_incendios.pkl' no se encontró. Por favor, asegúrate de haber entrenado y guardado el modelo.")
    exit()

def interfaz_interactiva():
    print("Bienvenido al sistema de predicción de incendios forestales 🌲🔥\n")
    print("Por favor, ingresa los datos climáticos para realizar una predicción.")

    try:
        # Solicitar datos al usuario
        temperatura = float(input("Ingresa la temperatura (°C): "))
        humedad = float(input("Ingresa la humedad (%): "))
        viento = float(input("Ingresa la velocidad del viento (km/h): "))

        # Crear DataFrame con los datos ingresados
        datos_usuario = pd.DataFrame({
            "temperatura": [temperatura],
            "humedad": [humedad],
            "viento": [viento]
        })

        # Realizar predicción
        prediccion = modelo.predict(datos_usuario)[0]
        resultado = "Incendio" if prediccion == 1 else "No incendio"
        print(f"\nResultado de la predicción: {resultado}")
    
    except ValueError:
        print("Error: Por favor, ingresa valores numéricos válidos.")
    
    # Opción para reiniciar o salir
    opcion = input("\n¿Deseas realizar otra predicción? (sí/no): ").strip().lower()
    if opcion == "sí":
        interfaz_interactiva()
    else:
        print("¡Gracias por usar el sistema de predicción de incendios! 🌟")

# Iniciar la interfaz
if __name__ == "__main__":
    interfaz_interactiva()
