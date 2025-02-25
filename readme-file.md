# Proyecto de Analítica Computacional para la Toma de Decisiones

Este repositorio contiene un prototipo funcional del proyecto de analítica de datos enfocado en el costo de arrendamiento de apartamentos. El proyecto aborda desde la definición de preguntas de negocio, la exploración y preparación de datos, la construcción de modelos predictivos, hasta el desarrollo y despliegue del tablero interactivo en Dash.

## 1. Estructura del Proyecto

La estructura principal del proyecto es la siguiente:

```
ANALITICA_COMPUTACIONAL_PROYECTO
├── .vscode/                    # Configuraciones de VSCode (ajustes de entorno, extensiones, etc.)
├── data/                       # Carpeta con los datos limpios, crudos y predicciones
├── Despliegue/                 # Scripts, configuraciones y/o archivos para el despliegue del tablero
│   ├── Tablero1.py
├── Soportes adicionales/       # Recursos complementarios (imágenes, reportes, etc.)
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── 5.png
│   └── maquina_desplegada.txt
├── .gitignore                  # Archivo para ignorar archivos o carpetas en control de versiones
├── diccionario_datos.txt       # Descripción de las variables incluidas en el dataset
├── exploracion_datos.ipynb     # Notebook de análisis exploratorio
├── modelos.ipynb               # Notebook con los modelos predictivos
├── preparacion_datos.ipynb     # Notebook de preparación y limpieza de datos
└── readme-file.md                   # Este archivo de documentación general
```

### Archivos y carpetas principales:

- **`.vscode/`**: Configuraciones específicas de Visual Studio Code.
- **`data/`**: Almacena los datos crudos y/o transformados necesarios.
- **`Despliegue/`**: Contiene script para el despliegue del tablero.
- **`Soportes adicionales/`**: Recursos complementarios solicitados por la entrega.
- **`.gitignore`**: Lista de archivos y carpetas excluidas del control de versiones.
- **`diccionario_datos.txt`**: Explica cada variable del dataset.
- **`exploracion_datos.ipynb`**: Notebook de análisis exploratorio de datos.
- **`modelos.ipynb`**: Notebook con construcción y evaluación de modelos predictivos.
- **`preparacion_datos.ipynb`**: Notebook para limpieza y preparación de datos.

## 2. Descripción General

El proyecto está orientado a una inmobiliaria interesada en conocer los factores determinantes del precio de alquiler de apartamentos y obtener un modelo predictivo para toma de decisiones. Se integran las siguientes tareas:

### Preguntas de negocio
- Definición de preguntas concretas sobre el mercado de alquileres. (PDF)

### Exploración de datos
- Análisis estadístico y visual de las variables.

### Preparación de datos
- Limpieza, tratamiento de valores faltantes, normalización y transformación de datos.

### Modelado predictivo
- Desarrollo de modelos de regresión para predecir precios de alquiler.

### Tablero interactivo
- Implementación de un tablero en Dash con visualizaciones dinámicas y capacidad predictiva.
- Despliegue en un entorno accesible (instancia EC2).

## 3. Roles y Responsabilidades

Cada miembro del equipo asume dos roles de la siguiente lista:

- Análisis de Negocio (Tomas Acosta)
- Ingeniería de Datos (Tomas Acosta)
- Análisis de Datos (Juan Sebastian Rojas)
- Ciencia de Datos (Diego Alejandro Castro)
- Tablero de Datos (David Felipe Pineda)
- Despliegue y Mantenimiento (Diego Alejandro Castro)

La calificación y el aporte de cada miembro se basan en las tareas asociadas a sus roles y al proyecto.

## 4. Requerimientos y Ejecución

### Clonar el repositorio
```bash
git clone https://github.com/Tomas1307/analitica_computacional_proyecto.git
cd ANALITICA_COMPUTACIONAL_PROYECTO
```

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar Notebooks
- Se recomienda utilizar un entorno virtual (conda o venv).
- Abrir los notebooks con Jupyter Lab o Jupyter Notebook y ejecutar las celdas en orden.

### Desplegar Tablero
- Revisar la carpeta `Despliegue/` y/o los archivos de soporte para instrucciones específicas.
- El proceso del tablero se puede visualizar en la dirección local de http://127.0.0.1:8050 o al momento de encender la isntancia en la direccion ipv4 publica de EC2 y el puerto 8050.
