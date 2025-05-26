# Análisis Predictivo del Rendimiento Académico - Pruebas Saber 11

## Descripción del Proyecto

Este proyecto desarrolla un producto de analítica de datos sobre los resultados de las pruebas Saber 11 en Colombia, diseñado específicamente para el **ICFES** como usuario final. El ICFES es la entidad responsable de diseñar, aplicar y analizar los resultados de las pruebas de Estado en Colombia, y este análisis le permitirá identificar patrones importantes para diseñar estrategias efectivas de política educativa, mejorar la equidad y calidad de las pruebas, y referir estudiantes con alto desempeño para posibles becas.

## Usuario Final

**ICFES (Instituto Colombiano para la Evaluación de la Educación)**

La información obtenida de este proyecto le permitirá a la entidad:
- Identificar patrones demográficos y socioeconómicos críticos
- Diseñar estrategias efectivas de política educativa
- Mejorar la equidad y calidad de las pruebas
- Referir estudiantes con desempeño alto para posibles becas

## Preguntas de Negocio

### Pregunta de Negocio 1
**¿Cuáles son los patrones demográficos y socioeconómicos asociados con bajo desempeño en la prueba Saber 11 (2010), que permitan orientar estrategias focalizadas de intervención educativa?**

### Pregunta de Negocio 2
**¿Cuáles características demográficas, socioeconómicas y académicas son las que más afectan los resultados de matemáticas en la prueba Saber 11 en los estudiantes que posiblemente son becados por un desempeño por encima del promedio (2010)?**

## Metodología

El sistema implementa dos modelos de Machine Learning con redes neuronales convolucionales:

1. **Regresión**: Predicción del puntaje en matemáticas de estudiantes
2. **Clasificación**: Predicción del nivel socioeconómico (derivado del puntaje promedio: arriba del promedio = 1, debajo = 0)

### Proceso de Experimentación
- **17 modelos diferentes** entrenados para cada problema (regresión y clasificación)
- **Grid Search** extensivo con MLflow para optimización de hiperparámetros
- **Validación cruzada K-Fold** aplicada especialmente al modelo de regresión
- Seguimiento completo de experimentos en MLflow

## Equipo de Desarrollo

| Rol | Responsable |
|-----|-------------|
| **Ciencia de Datos y Despliegue** | Tomas Acosta |
| **Ingeniería y Exploración de Datos** | David Felipe Pineda |
| **Tablero de Datos** | Diego Alejandro Castro |
| **Análisis de Negocio** | Juan Sebastian Rojas |

## Tecnologías Utilizadas

### Frameworks y Librerías
```
pytorch >= 1.9.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
captum >= 0.4.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
mlflow >= 1.20.0
```

### Arquitectura del Modelo
- **Embeddings**: Para variables categóricas
- **Capas Convolucionales**: Extracción de patrones
- **Capas Densas**: Predicción final
- **Regularización**: Dropout para prevenir overfitting

## Estructura del Proyecto

```
government_icfes_dataset/
├── __pycache__/                  # Cache de Python
├── best_models/                  # Modelos entrenados guardados
│   ├── classification/
│   └── regression/
├── checkpoints/                  # Checkpoints de entrenamiento
├── checkpoints_kfold/           # Checkpoints para validación cruzada
├── data/                        # Datasets del proyecto
│   ├── datos_variables_seleccionadas.csv
│   └── predicciones_completas.csv
├── mlartifacts/                 # Artefactos de MLflow
├── mlruns/                      # Experimentos de MLflow
├── analisis_modelos.ipynb       # Interpretación del modelo y predicciones
├── convolutional.py             # Core - Arquitectura CNN
├── exploration.ipynb            # Preparación y exploración de datos
├── gridsearch_mlflow.py         # Grid search con MLflow
├── label_encoder_unk.py         # Codificador personalizado
├── ml_flow_execution.ipynb      # Ejecución completa con K-Fold y MLflow
├── plots_ds.py                  # Todas las visualizaciones del modelo
├── README.md                    # Documentación del proyecto
├── requirements.txt             # Dependencias
├── testing.ipynb               # Testing inicial del entrenamiento
└── variables_seleccion.ipynb   # Selección y justificación de variables
```

## Componentes Principales

### **exploration.ipynb**
- Análisis exploratorio de datos del ICFES
- Visualizaciones de distribuciones y correlaciones
- Identificación de patrones en el rendimiento académico

### **variables_seleccion.ipynb**
- Justificación técnica de variables seleccionadas
- Análisis de relevancia para ambas tareas
- Optimización del conjunto de características

### **convolutional.py**
- Implementación de arquitecturas CNN personalizadas
- Clases `ConvRegressor` y `ConvClassifier`
- Manejo de embeddings para variables categóricas
- Funciones de entrenamiento y evaluación

### **analisis_modelos.ipynb**
- Interpretabilidad con **Integrated Gradients** y **Feature Ablation**
- Análisis del impacto de cada variable en las predicciones
- Generación de predicciones para el conjunto de prueba

### **gridsearch_mlflow.py**
- Búsqueda automática de hiperparámetros
- Integración con MLflow para seguimiento
- Optimización de arquitectura CNN

### **ml_flow_execution.ipynb**
- Pipeline completo de entrenamiento
- Validación cruzada K-Fold
- Registro de experimentos en MLflow
- Guardado de mejores modelos

### **plots_ds.py**
- Generación de todas las visualizaciones del proyecto
- Dashboard interactivo con métricas de rendimiento
- Gráficas de interpretabilidad y comparación de modelos

## Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/government-icfes-analysis.git
cd government-icfes-analysis
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar MLflow
```bash
# Iniciar servidor MLflow para seguimiento de experimentos
mlflow ui
```

### 5. Verificar Instalación
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import captum; print('Captum:', captum.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
```

## Flujo de Trabajo del Proyecto

### 1. **Exploración y Preparación de Datos**
```bash
# Ejecutar análisis exploratorio
jupyter notebook exploration.ipynb
```

### 2. **Selección de Variables**
```bash
# Revisar justificación de variables seleccionadas
jupyter notebook variables_seleccion.ipynb
```

### 3. **Entrenamiento con Grid Search**
```bash
# Ejecutar búsqueda de hiperparámetros con MLflow
python gridsearch_mlflow.py
```

### 4. **Entrenamiento Completo con K-Fold**
```bash
# Ejecutar entrenamiento completo con validación cruzada
jupyter notebook ml_flow_execution.ipynb
```

### 5. **Análisis de Interpretabilidad y Predicciones**
```bash
# Generar interpretaciones y predicciones finales
jupyter notebook analisis_modelos.ipynb
```

### 6. **Generación de Visualizaciones**
```python
# Generar todas las gráficas de rendimiento
from plots_ds import create_presentation_dashboard
import pandas as pd

# Ejecutar dashboard completo
create_presentation_dashboard(
    df_test_full=pd.read_csv("./government_icfes_dataset/data/predicciones_completas.csv")
)
```

## Resultados del Modelo

### Mejores Hiperparámetros Encontrados

#### Modelo de Regresión (Puntaje Matemáticas)
- **batch_size**: 16
- **conv_filters**: [32, 64]
- **dense_units**: 64
- **embedding_dim**: 4
- **epochs**: 80
- **patience**: 10

**Métricas:**
- **MAE**: 6.417
- **MSE**: 68.343
- **R²**: 0.333
- **RMSE**: 8.267

#### Modelo de Clasificación (Nivel Socioeconómico)
- **batch_size**: 16
- **conv_filters**: [32, 64]
- **dense_units**: 64
- **embedding_dim**: 8
- **epochs**: 10
- **patience**: 3

**Métricas:**
- **Accuracy**: 0.692
- **F1-Score**: 0.681
- **Precision**: 0.695
- **Recall**: 0.668
- **ROC-AUC**: 0.760

### Variables Más Importantes

#### Para Puntaje en Matemáticas:
1. **cole_mcpio_ubicacion** (3.33) - Municipio del colegio
2. **estu_tipodocumento** (3.04) - Tipo de documento del estudiante
3. **desemp_ingles** (1.97) - Desempeño en inglés
4. **estu_mcpio_reside** (1.46) - Municipio de residencia
5. **cole_naturaleza** (0.97) - Naturaleza del colegio

#### Para Nivel Socioeconómico:
1. **estu_tipodocumento** (0.106) - Tipo de documento
2. **desemp_ingles** (0.079) - Desempeño en inglés
3. **fami_tienelavadora** (0.027) - Posesión de lavadora
4. **cole_mcpio_ubicacion** (0.025) - Municipio del colegio
5. **cole_area_ubicacion** (0.022) - Área de ubicación

## Metodología

### Arquitectura CNN
- **Embeddings**: Transformación de variables categóricas a vectores densos
- **Convolución 1D**: Extracción de patrones secuenciales
- **Pooling**: Reducción dimensional y regularización
- **Capas Densas**: Predicción final

### Interpretabilidad
- **Integrated Gradients**: Para variables numéricas
- **Feature Ablation**: Para variables categóricas
- **Análisis Comparativo**: Entre ambos modelos

### Validación
- **K-Fold Cross Validation**: Aplicado especialmente al modelo de regresión
- **Grid Search**: 17 experimentos por cada problema
- **MLflow**: Seguimiento completo de experimentos

## Consideraciones Importantes

### Limitaciones
- El modelo está entrenado específicamente en datos del ICFES colombiano del 2010
- Las interpretaciones son válidas para el contexto educativo nacional
- Se requiere calibración para otros contextos geográficos o temporales

### Recomendaciones de Uso para el ICFES
- Validar predicciones con expertos en educación
- Considerar factores socioeconómicos adicionales en políticas públicas
- Actualizar modelos periódicamente con nuevos datos
- Usar insights para diseño de estrategias de intervención focalizadas

## Contribuciones

### Cómo Contribuir
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Áreas de Mejora
- [ ] Incorporar más variables socioeconómicas
- [ ] Implementar modelos ensemble
- [ ] Añadir análisis de fairness
- [ ] Optimizar arquitectura con AutoML
- [ ] Crear API para predicciones en tiempo real

## Contacto

**Equipo de Desarrollo:**
- **Tomas Acosta** - Ciencia de Datos y Despliegue
- **David Felipe Pineda** - Ingeniería y Exploración de Datos
- **Diego Alejandro Castro** - Tablero de Datos
- **Juan Sebastian Rojas** - Análisis de Negocio

## Reconocimientos

- **ICFES**: Por proporcionar los datos del dataset de las pruebas Saber 11
- **PyTorch Team**: Por el framework de deep learning
- **Captum Team**: Por las herramientas de interpretabilidad
- **MLflow Team**: Por la plataforma de MLOps

---

### Notas Adicionales

Para ejecutar el análisis completo de visualizaciones, simplemente ejecuta:

```python
from plots_ds import create_presentation_dashboard
import pandas as pd

create_presentation_dashboard(
    df_test_full=pd.read_csv("./government_icfes_dataset/data/predicciones_completas.csv")
)
```

**El proyecto está diseñado como un prototipo funcional para el ICFES, proporcionando insights accionables para la toma de decisiones en política educativa.**