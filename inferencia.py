import pandas as pd
import numpy as np
from joblib import load
import streamlit as st
from tensorflow.keras.models import load_model
import io

# Cargar los archivos .pkl
imputer_numeric = load('imputer_numeric.pkl')
imputer_category = load('imputer_category.pkl')
scaler = load('scaler.pkl')
model = load('model_sequential_optimized.pkl')

def procesar_data(data):
    # Filtrar 10 ciudades aleatorias ESTO SI O NO?
    ciudades_aleatorias = data['Location'].sample(n=10, random_state=42)
    data = data[data['Location'].isin(ciudades_aleatorias)]

    # Convertir 'RainToday' y 'RainTomorrow' a valores binarios
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

    # Eliminar columnas sin valor significativo predictivo
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    data = data.drop(columns=categorical_cols)

    # Eliminar filas nan
    data = data.dropna(thresh=data.shape[1] - 15)

    # Identificar columnas numéricas y categóricas
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    
    # valores nulos
    data_imputado = data.copy()
    data_imputado[numeric_columns] = imputer_numeric.transform(data[numeric_columns])
    data_imputado[categorical_columns] = imputer_category.transform(data[categorical_columns])

    # escalado
    float_columns = data.select_dtypes(include=['float64']).columns
    columns_to_scale = [col for col in float_columns if col not in ['RainTomorrow', 'RainToday']]

    data_escalado = data_imputado.copy()
    data_escalado[columns_to_scale] = scaler.transform(data_imputado[columns_to_scale])

    return data_escalado

# Definir la interfaz
st.title("Predicción de lluvia - WeatherAUS")
st.write("Sube un archivo CSV con los datos climáticos para predecir si lloverá mañana.")

# Cargar el archivo de entrada
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"], key="file_uploader_1")

if uploaded_file:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Preprocesar los datos con la función que ya tienes
    data_procesado = procesar_data(data)

    # Realizar predicciones con el modelo
    predictions = model.predict(data_procesado)

    # Convertir las predicciones a valores binarios (0 o 1)
    predictions = (predictions > 0.5).astype(int)

    # Agregar las predicciones al DataFrame original
    data['RainTomorrow_Prediction'] = predictions

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























import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import io

# Cargar los archivos .pkl
imputer_numeric = load('imputer_numeric.pkl')
imputer_category = load('imputer_category.pkl')
scaler = load('scaler.pkl')
model_sequential_optimized = load('model_sequential_optimized.pkl')

def procesar_data(data):
    # Filtrar 10 ciudades aleatorias (opcional, se puede eliminar si no es necesario)
    ciudades_aleatorias = data['Location'].sample(n=10, random_state=42)
    data = data[data['Location'].isin(ciudades_aleatorias)]

    # Convertir 'RainToday' y 'RainTomorrow' a valores binarios
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

    # Eliminar columnas sin valor significativo predictivo
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    data = data.drop(columns=categorical_cols)

    # Eliminar filas nan
    data = data.dropna(thresh=data.shape[1] - 15)

    # Identificar columnas numéricas y categóricas
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    
    # valores nulos
    data_imputado = data.copy()
    data_imputado[numeric_columns] = imputer_numeric.transform(data[numeric_columns])
    data_imputado[categorical_columns] = imputer_category.transform(data[categorical_columns])

    # escalado
    float_columns = data.select_dtypes(include=['float64']).columns
    columns_to_scale = [col for col in float_columns if col not in ['RainTomorrow', 'RainToday']]

    data_escalado = data_imputado.copy()
    data_escalado[columns_to_scale] = scaler.transform(data_imputado[columns_to_scale])

    return data_escalado

# Definir la interfaz de Streamlit
st.title("Predicción de lluvia - WeatherAUS")
st.write("Sube un archivo CSV con los datos climáticos para predecir si lloverá mañana.")

# Cargar el archivo de entrada
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Preprocesar los datos con la función que ya tienes
    data_preprocesada = procesar_data(data)

    # Realizar predicciones con el modelo
    predictions = model_sequential_optimized.predict(data_preprocesada)

    # Convertir las predicciones a valores binarios (0 o 1)
    predictions = (predictions > 0.5).astype(int)

    # Agregar las predicciones al DataFrame original
    data['RainTomorrow_Prediction'] = predictions

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
