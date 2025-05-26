import pandas as pd
import unicodedata
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc # Importar Dash Bootstrap Components

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).upper().strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto

# -----------------------------
# CARGA Y LIMPIEZA DE DATOS
# -----------------------------
try:
    df = pd.read_csv("./data/datos_variables_seleccionadas.csv")
    df_mapa = pd.read_csv("./data/Municipios_col.csv", encoding="latin1")

    # Renombrar columnas por si tienen caracteres especiales
    df_mapa.columns = df_mapa.columns.str.encode('latin1').str.decode('utf-8')

    # Limpiar y normalizar nombres de municipios
    df['estu_mcpio_reside'] = df['estu_mcpio_reside'].apply(limpiar_texto)
    df_mapa['Nombre Municipio'] = df_mapa['Nombre Municipio'].apply(limpiar_texto)

    # Preparar DataFrame de coordenadas
    df_coord = df_mapa[['Nombre Municipio', 'Latitud', 'longitud']].copy()
    # Convertir a float de manera robusta, reemplazando comas por puntos si es necesario
    df_coord.loc[:, 'Latitud'] = df_coord['Latitud'].astype(str).str.replace(",", ".", regex=False).astype(float)
    df_coord.loc[:, 'longitud'] = df_coord['longitud'].astype(str).str.replace(",", ".", regex=False).astype(float)

    # Crear diccionario municipio -> (lat, lon)
    municipio_coords = {
        row['Nombre Municipio']: (row['Latitud'], row['longitud']) for _, row in df_coord.iterrows()
        if pd.notna(row['Latitud']) and pd.notna(row['longitud'])
    }

    # Obtener lista de municipios únicos del dataframe principal
    municipios_icfes_data = set(df['estu_mcpio_reside'].dropna().unique())

    # Filtrar municipios para asegurar que tengan coordenadas válidas
    municipios_validos = sorted([m for m in municipios_icfes_data if m in municipio_coords])
    municipios_unicos = municipios_validos if municipios_validos else [] # Asegura que no sea None si está vacío

except Exception as e:
    print(f"Error al cargar o procesar datos en tab_estu_mcpio_reside.py: {e}")
    df = pd.DataFrame() # DataFrame vacío para evitar errores
    municipios_unicos = []
    municipio_coords = {}


# -----------------------------
# VARIABLES Y LAYOUT
# -----------------------------
# Estas son las variables para el dropdown/tabs del gráfico principal
variables_analisis = [
    {'label': 'Puntaje de Inglés', 'value': 'punt_ingles'},
    {'label': 'Puntaje de Matemáticas', 'value': 'punt_matematicas'},
    {'label': 'Edad del Estudiante', 'value': 'edad'},
    {'label': 'Estrato Socioeconómico', 'value': 'fami_estratovivienda'},
    {'label': 'Nivel Económico (eco)', 'value': 'eco'}, # Asumiendo que 'eco' existe o se creará
    {'label': 'Número de Cuartos en el Hogar', 'value': 'fami_cuartoshogar_int'} # Asumiendo que 'fami_cuartoshogar_int' existe
]


# -----------------------------
# FUNCIONES DE GRÁFICA ROBUSTAS
# -----------------------------
def generar_grafico(municipio, variable):
    df_filtrado = df[df['estu_mcpio_reside'] == municipio]
    if df_filtrado.empty or variable not in df_filtrado.columns:
        return html.Div(f"⚠️ No hay datos disponibles para el municipio seleccionado o la variable '{variable}' no existe.", className="alert alert-warning text-center")

    fig = px.histogram(df_filtrado, x=variable, nbins=20, title=f'Distribución de {variable.replace("_", " ").title()} en {municipio}',
                       labels={variable: variable.replace("_", " ").title()}, template="plotly_white")
    fig.update_layout(transition_duration=500, bargap=0.2)
    return dcc.Graph(figure=fig)

def generar_grafico_estrato(municipio):
    df_filtrado = df[df['estu_mcpio_reside'] == municipio]
    if df_filtrado.empty or 'fami_estratovivienda' not in df_filtrado.columns:
        return html.Div("⚠️ No hay datos de estrato disponibles para este municipio.", className="alert alert-warning text-center")
    
    estratos = df_filtrado['fami_estratovivienda'].dropna()
    if estratos.empty:
        return html.Div("⚠️ Sin datos de estrato para este municipio.", className="alert alert-warning text-center")

    estratos_counts = estratos.value_counts().reset_index()
    estratos_counts.columns = ['Estrato', 'Cantidad']
    
    fig = px.pie(estratos_counts, names='Estrato', values='Cantidad',
                 title=f'Distribución de Estrato Socioeconómico en {municipio}', template="plotly_white")
    fig.update_layout(transition_duration=500)
    return dcc.Graph(figure=fig)

def generar_mapa_municipio(municipio):
    coords = municipio_coords.get(municipio)
    if not coords or pd.isna(coords[0]) or pd.isna(coords[1]):
        return html.Div("⚠️ No se encontraron coordenadas válidas para este municipio.", className="alert alert-warning text-center")

    fig = go.Figure(go.Scattermapbox(
        lat=[coords[0]],
        lon=[coords[1]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=14, color='red'), # Marcador rojo
        text=[municipio],
    ))
    fig.update_layout(
        mapbox=dict(
            style="open-street-map", # Estilo de mapa más moderno
            zoom=8, # Zoom ligeramente mayor para ver el municipio
            center={"lat": coords[0], "lon": coords[1]}
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}, # Margen superior para el título
        title=f"Ubicación geográfica de {municipio}",
        title_x=0.5 # Centrar el título
    )
    fig.update_layout(transition_duration=500)
    return dcc.Graph(figure=fig)


def generar_correlacion(municipio):
    df_filtrado = df[df['estu_mcpio_reside'] == municipio]
    if df_filtrado.empty:
        return html.Div("⚠️ No hay datos disponibles para este municipio.", className="alert alert-warning text-center")

    # Asegúrate de que las columnas existan y sean numéricas
    cols_corr = ['punt_matematicas', 'punt_ingles', 'edad', 'fami_cuartoshogar_int']
    df_corr = df_filtrado[cols_corr].apply(pd.to_numeric, errors='coerce').dropna()

    if df_corr.empty or len(df_corr.columns) < 2:
        return html.Div("⚠️ No hay suficientes datos numéricos para calcular la correlación para este municipio.", className="alert alert-warning text-center")
    
    corr_matrix = df_corr.corr()
    
    fig = px.imshow(corr_matrix,
                    text_auto=True, # Mostrar valores de correlación
                    aspect="auto",
                    color_continuous_scale=px.colors.sequential.Viridis, # Escala de color más atractiva
                    title=f'Matriz de Correlación de Variables Numéricas en {municipio}',
                    labels=dict(color="Coeficiente de Correlación")) # Etiqueta de la barra de color
    fig.update_layout(transition_duration=500)
    return dcc.Graph(figure=fig)

# -----------------------------
# LAYOUT DEL TAB
# -----------------------------
def layout():
    return dbc.Container([
        html.H2("🌍 Análisis Descriptivo por Municipio", className="my-4 text-center text-primary"),

        dbc.Card([
            dbc.CardHeader(html.H4("Opciones de Filtrado y Visualización", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona un Municipio:", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id='municipio-dropdown',
                            options=[{'label': m, 'value': m} for m in municipios_unicos],
                            value=municipios_unicos[0] if municipios_unicos else None, # Valor predeterminado
                            placeholder="Selecciona un municipio...",
                            clearable=False,
                            className="mb-3"
                        )
                    ], md=6), # Ocupa la mitad del ancho en pantallas medianas y grandes
                    dbc.Col([
                        html.Label("Variable para el Gráfico Principal:", className="fw-bold mb-1"),
                        # Este dcc.Tabs se comporta como un Dropdown para la variable del gráfico principal
                        dcc.Tabs(
                            id='tabs-variables',
                            value=variables_analisis[0]['value'], # Valor predeterminado
                            children=[
                                dcc.Tab(label=var['label'], value=var['value'], 
                                        className="custom-tab", selected_className="custom-tab--selected")
                                for var in variables_analisis
                            ],
                            className="custom-tabs-container mb-3" # Clase para el contenedor de tabs
                        )
                    ], md=6) # Ocupa la mitad del ancho
                ])
            ])
        ], className="shadow-sm mb-4"), # Tarjeta con sombra y margen inferior

        # Fila para el Gráfico Principal (histograma de variable seleccionada)
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H4("Distribución de Variable Seleccionada", className="mb-0")),
                    dbc.CardBody(
                        html.Div(id='grafico-container', className="p-2") # Contenedor para el gráfico principal
                    )
                ], className="shadow-sm h-100"),
                lg=12, # Ocupa todo el ancho en pantallas grandes
                className="mb-4" # Margen inferior
            ),
        ]),

        # Fila para Gráfico de Estrato y Mapa (lado a lado en pantallas grandes)
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H4("Distribución por Estrato Socioeconómico", className="mb-0")),
                    dbc.CardBody(
                        html.Div(id='grafico-estrato', className="p-2") # Contenedor para el gráfico de estrato
                    )
                ], className="shadow-sm h-100"),
                lg=6, # Ocupa la mitad en pantallas grandes
                md=12, # Ocupa todo el ancho en pantallas medianas y pequeñas
                className="mb-4" # Margen inferior
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H4("Ubicación Geográfica del Municipio", className="mb-0")),
                    dbc.CardBody(
                        html.Div(id='grafico-mapa', className="p-2") # Contenedor para el mapa
                    )
                ], className="shadow-sm h-100"),
                lg=6, # Ocupa la mitad en pantallas grandes
                md=12, # Ocupa todo el ancho en pantallas medianas y pequeñas
                className="mb-4" # Margen inferior
            )
        ]),

        # Fila para la Matriz de Correlación
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H4("Matriz de Correlación de Variables Numéricas", className="mb-0")),
                    dbc.CardBody(
                        html.Div(id='grafico-correlacion', className="p-2") # Contenedor para la matriz de correlación
                    )
                ], className="shadow-sm h-100"),
                lg=12, # Ocupa todo el ancho
                className="mb-4" # Margen inferior
            )
        ])

    ], fluid=True) # El contenedor principal es fluido
# Crear diccionario municipio -> (lat, lon) solo para coordenadas válidas
municipio_coords = {
    row['Nombre Municipio']: (row['Latitud'], row['longitud']) 
    for _, row in df_coord.iterrows()
    if pd.notna(row['Latitud']) and pd.notna(row['longitud'])
}

# Obtener lista de municipios únicos del dataframe principal (ICFES)
municipios_en_icfes = set(df['estu_mcpio_reside'].dropna().unique())

# Identificar municipios sin coordenadas
municipios_sin_coords = sorted(municipios_en_icfes - set(municipio_coords.keys()))

if municipios_sin_coords:
    print(f"[INFO] Municipios sin coordenadas ({len(municipios_sin_coords)}):")
    for m in municipios_sin_coords:
        print(f"  - {m}")
else:
    print("[INFO] Todos los municipios del dataset ICFES tienen coordenadas.")

# Puedes guardar esta lista a un CSV si es muy larga para revisar:
if municipios_sin_coords:
    pd.DataFrame({'Municipio_Faltante': municipios_sin_coords}).to_csv(
        './data/municipios_a_buscar_coordenadas.csv', index=False, encoding='utf-8'
    )
    print("\nLista de municipios sin coordenadas guardada en 'municipios_a_buscar_coordenadas.csv'")
