# Predicción de Lluvias en Australia
## Descripción del Proyecto:

En este trabajo, se tiene como objetivo predecir la probabilidad de lluvia al día siguiente en diversas ciudades de Australia, utilizando la variable 'RainTomorrow' del conjunto de datos weatherAUS.csv. Para lograr esta predicción, se emplearon un conjunto de características climáticas, como la dirección y velocidad del viento, la temperatura y la humedad, entre otras.

## Contenido del Repositorio:
- **`TP_clasificacion_AA1.ipynb`**: Notebook principal que contiene todo el proceso de análisis, preprocesamiento, desarrollo y evaluación de los modelos.
- **`weatherAUS.csv`**: Dataset utilizado para la predicción, con información climática histórica de Australia.
- **`dockerfile`**: Configuración para crear una imagen Docker del proyecto.
- **`requirements.txt`**: Lista de dependencias necesarias.
- **`inferencia.py`**: Script para realizar predicciones utilizando el modelo entrenado.
- **`model_sequential_optimized.pkl`**: Modelo optimizado de red neuronal almacenado.
- **`scaler.pkl`**: Eescalador para preprocesar los datos de entrada.
- **`imputer_numeric.pkl`**: Imputador para manejar valores faltantes en variables numéricas.
- **`explainer_rnn.pkl`**: Objeto de explicabilidad SHAP para el modelo de red neuronal.
- **`shap_values.pkl`**: Valores de SHAP para análisis de explicabilidad.
- **`logs.log`**: Archivo de logs para seguimiento del funcionamiento del modelo y procesos relacionados.

## Indicaciones:

### 1. Clonar el Repositorio
```
git clone https://github.com/juanazorzolo/AA1-TUIA-Herrera-Zorzolo

cd AA1-TUIA-Herrera-Zorzolo
```

### EJECUTAR LOCALMENTE
### 1. Instalar Dependencias
```
pip install -r requirements.txt
```

### 2. Ejecutar Streamlit 
```
streamlit run inferencia.py
```

### 3. Subir el Archivo de Predicciones
En la interfaz de Streamlit, subir el archivo para realizar las predicciones.


### EJECUTAR CON DOCKER
### 1. Construir Imagen Docker
```
docker build -t predecir_lluvia .
```

### 2. Ejecutar el Contenedor
```
docker run -it --name contenedor predecir_lluvia
```

### 3. Subir el Archivo de Predicciones
En la interfaz de Streamlit, subir el archivo para realizar las predicciones.

## Requisitos del Sistema
- Python 3.10 o superior
- Docker instalado
- pip para manejar dependencias

