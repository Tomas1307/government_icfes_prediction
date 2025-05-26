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

## Estructura del Proyecto

```
government_icfes_dataset/
├── anexos/                           # Documentación adicional por roles
│   ├── ciencia_datos/               # Documentación técnica de modelos
│   ├── despliegue/                  # Guías de implementación
│   ├── exploracion_datos/           # Análisis exploratorio detallado
│   ├── ingenieria_datos/            # Pipelines y ETL
│   └── tablero_dash/                # Documentación del dashboard
├── best_models/                      # Modelos entrenados guardados
│   ├── classification/              # Mejores modelos de clasificación
│   └── regression/                  # Mejores modelos de regresión
├── data/                            # Datasets del proyecto
│   ├── datos_variables_seleccionadas.csv
│   └── predicciones_completas.csv
├── despliegue/                      # Aplicación Dash y deployment
│   ├── __pycache__/
│   ├── __init__.py                  # Inicialización del 
│   ├── app.py                       # Aplicación principal Dash
│   ├── datosylayout.py              # Layout y componentes UI
│   ├── prediccion.py                # Lógica de predicciones
│   └── tab_estu_mcpio_reside.py     # Componentes específicos
├── encoders/                        # Codificadores personalizados
├── mlartifacts/                     # Artefactos de MLflow
├── mlruns/                          # Experimentos de MLflow
├── results_ml/                      # Resultados de machine learning
├── analisis_modelos.ipynb           # Interpretación y análisis final
├── convolutional.py                 # Arquitectura CNN core
├── Dockerfile                       # Containerización para despliegue
├── exploration.ipynb                # EDA y preparación de datos
├── gridsearch_mlflow.py             # Optimización de hiperparámetros
├── label_encoder_unk.py             # Codificador con manejo de unknowns
├── limpieza.ipynb                   # Proceso de limpieza de datos
├── ml_flow_execution.ipynb          # Pipeline completo MLflow + K-Fold
├── plots_ds.py                      # Visualizaciones del proyecto
├── README.md                        # Documentación principal
├── requirements.txt                 # Dependencias del proyecto
├── testing.ipynb                    # Testing y validación inicial
└── variables_seleccion.ipynb        # Selección y justificación de variables
```

## Equipo de Desarrollo y Roles

| Rol | Responsable | Archivos Principales |
|-----|-------------|---------------------|
| **Ciencia de Datos** | Tomas Acosta | `convolutional.py`, `gridsearch_mlflow.py`, `ml_flow_execution.ipynb`, `analisis_modelos.ipynb`,`variables_seleccion.ipynb`, `plots_ds.py` , `label_encoder_unk.py`, `encoders/`  |
| **Ingeniería de Datos** | David Felipe Pineda | `limpieza.ipynb`|
| **Análisis de Datos** | David Felipe Pineda | `exploration.ipynb`|
| **Análisis de Negocio** | Juan Sebastian Rojas | Definición de preguntas, `anexos/` |
| **Tablero de Datos** | Diego Alejandro Castro | `despliegue/app.py`, `despliegue/datosylayout.py` |
| **Despliegue y Mantenimiento** | Tomas Acosta | `Dockerfile`, `despliegue/`, configuración AWS |

---

## 📊 Análisis de Datos

**Responsable:** David Felipe Pineda

### Componentes Principales

#### **exploration.ipynb**
- Análisis exploratorio exhaustivo de datos del ICFES
- Visualizaciones de distribuciones y correlaciones
- Identificación de patrones en el rendimiento académico
- Análisis de variables categóricas y numéricas
- Detección de outliers y datos anómalos

### Hallazgos Principales del EDA
- **Distribución geográfica**: Concentración de mejores resultados en centros urbanos
- **Factor socioeconómico**: Correlación significativa con el rendimiento académico
- **Tipo de institución**: Diferencias marcadas entre colegios públicos y privados
- **Variables predictoras clave**: Municipio, tipo de documento, desempeño en inglés

---

## 🔧 Ingeniería de Datos

**Responsable:** David Felipe Pineda

### Componentes Principales

#### **limpieza.ipynb**
- Proceso completo de limpieza y transformación de datos
- Manejo de valores faltantes con estrategias específicas por variable
- Normalización y estandarización de datos numéricos
- Codificación de variables categóricas
- Creación de variables derivadas


### Pipeline de Datos
1. **Extracción**: Datos originales desde portal de Datos Abiertos Colombia
2. **Transformación**: Limpieza, codificación y normalización
3. **Validación**: Verificación de calidad y consistencia
4. **Carga**: Preparación para modelamiento y análisis



---

## 🤖 Ciencia de Datos

**Responsable:** Tomas Acosta

### Componentes Principales

#### **convolutional.py**
- Implementación de arquitecturas CNN personalizadas
- Clases `ConvRegressor` y `ConvClassifier` 
- Manejo de embeddings para variables categóricas
- Funciones de entrenamiento y evaluación
- Arquitectura híbrida: embeddings + convolución + capas densas

#### **gridsearch_mlflow.py**
- Búsqueda automática de hiperparámetros
- Integración completa con MLflow para seguimiento
- Optimización de arquitectura CNN
- 17 configuraciones diferentes por problema

#### **ml_flow_execution.ipynb**
- Pipeline completo de entrenamiento con validación cruzada K-Fold
- Registro sistemático de experimentos en MLflow
- Guardado automático de mejores modelos
- Evaluación comparativa de configuraciones

#### **label_encoder_unk.py**
- Codificador personalizado para manejar categorías desconocidas
- Preservación de información para datos no vistos durante entrenamiento
- Mapeo consistente entre entrenamiento y predicción

#### **encoders/**
- Serialización de todos los encoders utilizados
- Mantenimiento de consistencia en transformaciones
- Reutilización en el pipeline de predicción

#### **analisis_modelos.ipynb**
- Interpretabilidad con **Integrated Gradients** y **Feature Ablation**
- Análisis del impacto de cada variable en las predicciones
- Generación de predicciones para conjunto de prueba
- Explicabilidad de decisiones del modelo

#### **variables_seleccion.ipynb**
- Justificación técnica de variables seleccionadas
- Análisis de relevancia para ambas tareas (regresión y clasificación)
- Optimización del conjunto de características
- Análisis de correlación entre variables
- Reducción dimensional cuando necesario

#### **plots_ds.py**
- Generación de todas las visualizaciones del proyecto
- Dashboard interactivo con métricas de rendimiento
- Gráficas de interpretabilidad y comparación de modelos
- Visualizaciones específicas para el usuario final (ICFES)

### Arquitectura del Modelo

#### Diseño CNN Híbrido
```python
# Componentes principales:
- Embeddings: Variables categóricas → vectores densos
- Conv1D: Extracción de patrones secuenciales  
- MaxPooling: Reducción dimensional
- Dense Layers: Predicción final
- Dropout: Regularización anti-overfitting
```

#### Metodología de Entrenamiento
- **Grid Search**: 17 experimentos por cada problema
- **K-Fold Cross Validation**: 5 folds para robustez estadística
- **Early Stopping**: Prevención de overfitting
- **MLflow Tracking**: Seguimiento completo de experimentos

### Resultados del Modelo

#### Modelo de Regresión (Puntaje Matemáticas)
- **MAE**: 6.417 - **MSE**: 68.343 - **R²**: 0.333 - **RMSE**: 8.267
- **Hiperparámetros óptimos**: batch_size=16, conv_filters=[32,64], dense_units=64

#### Modelo de Clasificación (Nivel Socioeconómico)  
- **Accuracy**: 0.692 - **F1-Score**: 0.681 - **ROC-AUC**: 0.760
- **Hiperparámetros óptimos**: batch_size=16, embedding_dim=8, epochs=10

### Variables de Mayor Importancia
1. **cole_mcpio_ubicacion** - Municipio del colegio
2. **estu_tipodocumento** - Tipo de documento del estudiante  
3. **desemp_ingles** - Desempeño en inglés
4. **estu_mcpio_reside** - Municipio de residencia
5. **cole_naturaleza** - Naturaleza del colegio (público/privado)

---

## 📋 Análisis de Negocio

**Responsable:** Juan Sebastian Rojas

### Definición del Usuario Final
**ICFES (Instituto Colombiano para la Evaluación de la Educación)**

#### Necesidades Identificadas
1. **Identificación de patrones críticos** en rendimiento académico
2. **Diseño de políticas educativas** basadas en evidencia
3. **Mejora de equidad** en el sistema educativo nacional
4. **Identificación de talento** para programas de becas

#### Preguntas de Negocio Estratégicas

**Pregunta 1 - Intervención Educativa:**
¿Cuáles son los patrones demográficos y socioeconómicos asociados con bajo desempeño que permitan orientar estrategias focalizadas de intervención?

**Pregunta 2 - Identificación de Talento:**
¿Cuáles características son las que más afectan los resultados en matemáticas en estudiantes con potencial para becas?

### Insights de Negocio

#### Factores Críticos Identificados
- **Ubicación geográfica**: Municipios rurales requieren mayor atención
- **Nivel socioeconómico**: Fuerte predictor del rendimiento académico
- **Tipo de institución**: Brechas significativas público vs privado
- **Competencias transversales**: Inglés como predictor de desempeño general

#### Recomendaciones Estratégicas para el ICFES
1. **Focalización geográfica**: Programas específicos para municipios de bajo rendimiento
2. **Apoyo socioeconómico**: Estrategias diferenciadas por estrato
3. **Fortalecimiento institucional**: Mejoras dirigidas a colegios públicos
4. **Desarrollo integral**: Fortalecimiento de competencias en inglés

---

## 📱 Tablero de Datos

**Responsable:** Diego Alejandro Castro

### Componentes del Dashboard

#### **despliegue/app.py**
- Aplicación principal desarrollada en Dash
- Integración de modelos pre-entrenados
- Interface interactiva para predicciones en tiempo real
- Navegación intuitiva para usuarios del ICFES

#### **despliegue/datosylayout.py**
- Diseño responsive y user-friendly
- Componentes visuales optimizados para análisis
- Layout adaptativo para diferentes dispositivos
- Elementos de UI específicos para el contexto educativo

#### **despliegue/prediccion.py**
- Lógica de predicción integrada
- Carga de modelos serializados
- Procesamiento de inputs del usuario
- Generación de resultados explicables

### Funcionalidades del Tablero

#### 🎯 Predicciones Interactivas
- **Predicción de puntaje en matemáticas** basada en características del estudiante
- **Clasificación de nivel socioeconómico** para identificación de patrones
- **Inputs dinámicos** con validación en tiempo real
- **Resultados explicables** mostrando factores más influyentes

#### 📊 Visualizaciones Integradas
1. **Dashboard de rendimiento** por regiones y demografía
2. **Análisis comparativo** entre diferentes perfiles de estudiantes  
3. **Tendencias históricas** y patrones identificados

#### 🎨 Diseño UX/UI
- **Interface intuitiva** diseñada específicamente para analistas del ICFES
- **Responsive design** adaptable a diferentes dispositivos
- **Navegación clara** entre secciones de análisis y predicción
- **Feedback visual** inmediato para acciones del usuario

---

## 🚀 Despliegue y Mantenimiento

**Responsable:** Tomas Acosta

### Arquitectura de Despliegue

#### **Dockerfile**
```dockerfile
# Containerización completa de la aplicación
- Base image: Python 3.11 slim
- Dependencias: requirements.txt
- Modelo pre-entrenado incluido
- Puerto expuesto: 8050
- Comando de inicio automático
```

#### Proceso de Despliegue
```bash
# 1. Construcción del contenedor
sudo docker build -t government_icfes:latest .

# 2. Ejecución en producción  
docker run -p 8050:8050 government_icfes

# 3. Verificacion
curl http://localhost:8050
```

### Infraestructura AWS

#### Servicios Utilizados
1. **EC2**: Instancia para hosting de la aplicación
2. **Docker**: Containerización para portabilidad
3. **Security Groups**: Configuración de puertos y acceso
4. **Elastic IP**: IP estática para acceso consistente

#### Configuración de Producción
- **Instancia**: t2.medium (2 vCPU, 4GB RAM)
- **SO**: Amazon Ubuntu
- **Puertos abiertos**: 8050 (aplicación), 22 (SSH)
- **Monitoreo**: CloudWatch básico

---

## 🛠️ Tecnologías Utilizadas

### Frameworks y Librerías Core
```python
pytorch >= 1.9.0          # Deep learning framework
pandas >= 1.3.0           # Manipulación de datos
numpy >= 1.21.0           # Computación numérica
scikit-learn >= 1.0.0     # ML utilities
dash >= 2.0.0             # Web dashboard framework
plotly >= 5.0.0           # Visualizaciones interactivas
```

### MLOps y Experimentación
```python
mlflow >= 1.20.0          # Seguimiento de experimentos
captum >= 0.4.0           # Interpretabilidad de modelos
optuna >= 2.10.0          # Optimización de hiperparámetros
```

### Deployment y Containerización
```bash
docker >= 20.10           # Containerización
flask >= 2.0.0            # API backend  
gunicorn >= 20.1.0        # WSGI server
nginx                     # Reverse proxy (opcional)
```

---

## 📖 Guía de Instalación y Uso

### 1. Clonar y Configurar
```bash
git clone https://github.com/Tomas1307/government_icfes_prediction.git
cd government-icfes-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecutar Análisis Completo
```bash
# EDA y preparación de datos
jupyter notebook exploration.ipynb

# Selección de variables
jupyter notebook variables_seleccion.ipynb

# Entrenamiento con optimización
python gridsearch_mlflow.py

# Pipeline completo con K-Fold
jupyter notebook ml_flow_execution.ipynb

# Análisis de interpretabilidad
jupyter notebook analisis_modelos.ipynb
```

### 3. Lanzar Dashboard Local
```bash
cd despliegue/
python app.py
# Acceder a: http://localhost:8050
```

### 4. Despliegue en Producción
```bash
# Construcción del contenedor
docker build -t saber11-app .

# Ejecución en producción
docker run -d -p 8050:8050 --name saber11-prod saber11-app

# Verificación
curl http://3.224.198.176:8050
```

---

## 📈 Resultados y Evaluación

### Métricas de Rendimiento

#### Modelo de Regresión (Matemáticas)
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **MAE** | 6.417 | Error promedio de ~6.4 puntos |
| **RMSE** | 8.267 | Desviación típica del error |
| **R²** | 0.333 | Explica 33.3% de la varianza |

#### Modelo de Clasificación (Socioeconómico)
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 0.692 | 69.2% de clasificaciones correctas |
| **F1-Score** | 0.681 | Buen balance precisión-recall |  
| **ROC-AUC** | 0.760 | Excelente capacidad discriminativa |

### Impact para el ICFES

#### Insights Accionables
1. **Focalización geográfica**: Identificación de municipios prioritarios
2. **Perfiles de riesgo**: Estudiantes con alta probabilidad de bajo rendimiento
3. **Factores modificables**: Variables sobre las que se puede intervenir
4. **Identificación de talento**: Criterios para programas de becas

#### ROI Estimado
- **Reducción de costos**: Focalización más eficiente de recursos
- **Mejora de outcomes**: Intervenciones basadas en evidencia
- **Escalabilidad**: Aplicable a futuras cohortes de estudiantes

---

## 🔄 Próximos Pasos y Mejoras

### Roadmap de Desarrollo
- [ ] **Modelos ensemble** para mayor robustez
- [ ] **Actualización con datos recientes** (2020-2024)
- [ ] **API REST** para integración con sistemas ICFES
- [ ] **Dashboard móvil** para acceso en campo
- [ ] **Análisis de fairness** y sesgo algorítmico

### Escalabilidad
- [ ] **Procesamiento en batch** para grandes volúmenes
- [ ] **Auto-reentrenamiento** con nuevos datos
- [ ] **Monitoreo de drift** en distribuciones
- [ ] **A/B testing** para validación de mejoras

---


**Equipo de Desarrollo:**
- **Tomas Acosta** - Ciencia de Datos y Despliegue
- **David Felipe Pineda** - Ingeniería y Exploración de Datos  
- **Diego Alejandro Castro** - Tablero de Datos
- **Juan Sebastian Rojas** - Análisis de Negocio

**Repositorio:** [GitHub - Government ICFES Analysis](https://github.com/Tomas1307/government_icfes_prediction.git)

**Documentación técnica adicional disponible en:** `anexos/` por cada rol específico.

**El proyecto está diseñado como un prototipo funcional para el ICFES, proporcionando insights accionables para la toma de decisiones en política educativa basada en evidencia.**