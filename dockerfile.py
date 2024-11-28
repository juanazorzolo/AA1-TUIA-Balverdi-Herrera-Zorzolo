# Usa una imagen base de Python 3.10
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /usr/src/app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt ./

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Expone el puerto 8501 utilizado por Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación de Streamlit
CMD ["streamlit", "run", "inferencia.py", "--server.port=8501", "--server.address=0.0.0.0"]

# EN LA TERMINAL
# Contruir imagen de docker
#$ docker build -t my-python-app .

#$ docker run -it --rm --name my-running-app my-python-app
#$ docker run -it --rm --name my-running-script -v "$PWD":/usr/src/myapp -w /usr/src/myapp python:2 python your-daemon-or-script.py