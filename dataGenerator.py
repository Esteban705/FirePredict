import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

# Configurar el cliente de Open-Meteo con caché y reintentos
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Definir las coordenadas de la ubicación (Catamarca, Argentina)
latitude = -28.4696
longitude = -65.7852

# Definir el rango de fechas (últimos 7 días)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Parámetros para la solicitud a la API
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "hourly": [
        "temperature_2m",
        "relativehumidity_2m",
        "precipitation",
        "windspeed_10m",
        "windgusts_10m",
        "shortwave_radiation"
    ],
    "timezone": "auto"
}

# Hacer la solicitud a la API
responses = openmeteo.weather_api(url, params=params)

# Procesar la primera ubicación
response = responses[0]
print(f"Coordenadas: {response.Latitude()}°N, {response.Longitude()}°E")
print(f"Zona horaria: {response.Timezone()} ({response.TimezoneAbbreviation()})")

# Procesar datos horarios
hourly = response.Hourly()
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relativehumidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "precipitation": hourly.Variables(2).ValuesAsNumpy(),
    "windspeed_10m": hourly.Variables(3).ValuesAsNumpy(),
    "windgusts_10m": hourly.Variables(4).ValuesAsNumpy(),
    "shortwave_radiation": hourly.Variables(5).ValuesAsNumpy()
}

# Crear un DataFrame con los datos horarios
hourly_dataframe = pd.DataFrame(data=hourly_data)
print("\nDatos horarios:")
print(hourly_dataframe.head())

# Calcular promedios diarios
daily_dataframe = hourly_dataframe.resample('D', on='date').agg({
    "temperature_2m": "mean",
    "relativehumidity_2m": "mean",
    "precipitation": "sum",
    "windspeed_10m": "mean",
    "windgusts_10m": "max",
    "shortwave_radiation": "mean"
}).reset_index()

# Clasificar el clima según las condiciones
def classify_weather(row):
    if row['precipitation'] > 5:
        return "tormentoso"
    elif row['relativehumidity_2m'] < 30:
        return "seco"
    elif row['relativehumidity_2m'] > 70:
        return "húmedo"
    else:
        return "normal"

daily_dataframe['clima'] = daily_dataframe.apply(classify_weather, axis=1)

# Mostrar los datos diarios
print("\nDatos diarios:")
print(daily_dataframe)

# Guardar los datos en un archivo CSV (opcional)
daily_dataframe.to_csv('datos_meteorologicos_diarios.csv', index=False)