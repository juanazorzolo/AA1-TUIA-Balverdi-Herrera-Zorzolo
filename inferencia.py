import pandas as pd
import numpy as np
from joblib import load
import streamlit as st
from tensorflow.keras.models import load_model
import io

# Cargar los archivos .pkl
imputer_numeric = load('imputer_numeric.pkl')
scaler = load('scaler.pkl')
model = load('model_sequential_optimized.pkl')

def procesar_data(data):
    # Convertir 'RainToday' a valores binarios
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    
    # Imputar categoricas
    data['RainToday'] = data['RainToday'].fillna(data['RainToday'].mode().iloc[0])

    columna_rain_today = data['RainToday']

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')

    # Eliminar columnas sin valor significativo predictivo
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Date', 'Location', 'RainTomorrow','RainToday']
    data = data.drop(columns=categorical_cols)

    # Eliminar filas nan
    data = data.dropna(thresh=data.shape[1] - 15)

    # Imputar numericas
    numeric_columns = data.select_dtypes(include=['float64']).columns
    data[numeric_columns] = imputer_numeric.transform(data[numeric_columns])

    # Escalado
    float_columns = data.select_dtypes(include=['float64']).columns
    columns_to_scale = [col for col in float_columns if col not in ['RainTomorrow', 'RainToday']]
    data[columns_to_scale] = scaler.transform(data[columns_to_scale])

    data['RainToday'] = columna_rain_today
    return data

# Definir la interfaz
st.title("Predicción de lluvia - WeatherAUS")
st.write("Sube un archivo CSV con los datos climáticos para predecir si lloverá mañana.")

# Cargar el archivo de entrada
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"], key="file_uploader_1")

if uploaded_file:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Preprocesar los datos
    data_procesado = procesar_data(data)

    # Crear una columna de predicciones inicialmente vacía
    data['RainTomorrow_Prediction'] = np.nan

    # Tamaño de lote
    batch_size = 1024  # Ajusta el tamaño según la memoria disponible

    # Procesar por lotes
    for start_idx in range(0, len(data_procesado), batch_size):
        end_idx = start_idx + batch_size
        batch = data_procesado.iloc[start_idx:end_idx]

        try:
            # Realizar predicciones para el lote
            predictions = model.predict(batch)

            # Convertir predicciones a binario
            binary_predictions = (predictions > 0.5).astype(int).flatten()

            # Asignar las predicciones al DataFrame original
            data.loc[batch.index, 'RainTomorrow_Prediction'] = binary_predictions
        except Exception as e:
            st.warning(f"Error procesando las filas {start_idx}-{end_idx}: {e}")
            # Dejar las predicciones de estas filas como NaN

    # Mostrar los resultados de las predicciones
    st.write("Resultados de la predicción:")
    st.dataframe(data)

    # Generar el archivo CSV con las predicciones
    csv_result = data.to_csv(index=False)

    # Crear un enlace de descarga para el archivo CSV
    st.download_button(
        label="Descargar archivo CSV con predicciones",
        data=csv_result,
        file_name="predicciones_climaticas.csv",
        mime="text/csv"
    )