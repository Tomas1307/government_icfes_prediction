import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# Importar funciones de los otros archivos
from Despliegue.tab_estu_mcpio_reside import (
    layout as layout_mcpio_reside,
    generar_grafico,
    generar_grafico_estrato,
    generar_mapa_municipio,
    generar_correlacion
)
from Despliegue.prediccion import ejecutar_predicciones

# --- Datos y Layout del Primer Tab (Predicción) - Anteriormente en datosylayout.py ---
cols = [
    "periodo", "estu_tipodocumento", "cole_area_ubicacion", "cole_caracter",
    "cole_cod_dane_establecimiento", "cole_cod_depto_ubicacion", "cole_codigo_icfes",
    "cole_depto_ubicacion", "cole_jornada", "cole_mcpio_ubicacion", "cole_naturaleza",
    "estu_depto_presentacion", "estu_depto_reside", "estu_genero", "estu_mcpio_presentacion",
    "estu_mcpio_reside", "fami_cuartoshogar", "fami_educacionmadre", "fami_educacionpadre",
    "fami_estratovivienda", "fami_tieneautomovil", "fami_tienelavadora", "desemp_ingles",
    "punt_ingles", "fami_cuartoshogar_int", "edad", "fami_nivel_tecnologia"
]

default_values = {
    "periodo": 20102,
    "estu_tipodocumento": "TI",
    "cole_area_ubicacion": "urbano",
    "cole_caracter": "TÉCNICO",
    "cole_cod_dane_establecimiento": 186219000070,
    "cole_cod_depto_ubicacion": 86,
    "cole_codigo_icfes": 29025,
    "cole_depto_ubicacion": "OTROS",
    "cole_jornada": "MAÑANA",
    "cole_mcpio_ubicacion": "Colón",
    "cole_naturaleza": "OFICIAL",
    "estu_depto_presentacion": "OTROS",
    "estu_depto_reside": "OTROS",
    "estu_genero": "F",
    "estu_mcpio_presentacion": "SIBUNDOY",
    "estu_mcpio_reside": "COLÓN",
    "fami_cuartoshogar": 5,
    "fami_educacionmadre": "SECUNDARIA (BACHILLERATO) INCOMPLETA",
    "fami_educacionpadre": "PRIMARIA COMPLETA",
    "fami_estratovivienda": 1,
    "fami_tieneautomovil": 0,
    "fami_tienelavadora": 1,
    "desemp_ingles": "A1",
    "punt_ingles": 49.66,
    "fami_cuartoshogar_int": 5,
    "edad": 31.0,
    "fami_nivel_tecnologia": 1
}

# Crear la estructura de los campos de entrada dinámicamente en un diseño de cuadrícula
input_form_elements = []
# Puedes ajustar el número de columnas (e.g., 3 para más compacto, 2 para más espacio)
num_cols_per_row = 3
for i in range(0, len(cols), num_cols_per_row):
    row_elements = []
    for col_name in cols[i : i + num_cols_per_row]:
        row_elements.append(
            dbc.Col(
                html.Div([
                    html.Label(col_name.replace('_', ' ').title(), className="mb-1 fw-bold"), # Mejorar etiqueta
                    dcc.Input(
                        id={'type': 'input-col', 'index': col_name},
                        type='text',
                        value=str(default_values.get(col_name, '')), # Usar .get para evitar KeyError si falta una clave
                        className="form-control" # Clase de Bootstrap para inputs
                    )
                ], className="mb-3"), # Margen inferior para cada input div
                md=int(12 / num_cols_per_row), # Ancho de columna responsivo
            )
        )
    input_form_elements.append(dbc.Row(row_elements))


# Layout del primer tab con estilo mejorado
prediccion_tab_layout = dbc.Container([
    html.H2("📊 Predicción del ICFES Saber 11", className="my-4 text-center text-primary"),

    dbc.Card([
        dbc.CardHeader(html.H4("Ingresa los datos del estudiante", className="mb-0")),
        dbc.CardBody([
            *input_form_elements, # Desempaqueta las filas de inputs
            html.Div(
                dbc.Button(
                    'Realizar Predicción',
                    id='btn-predict',
                    n_clicks=0,
                    className='btn btn-primary btn-lg w-100 mt-4' # Botón grande y ancho completo
                ),
                className="d-grid gap-2" # Para que el botón ocupe todo el ancho
            )
        ])
    ], className="shadow-sm mb-4"), # Sombra suave y margen inferior para la tarjeta

    dbc.Card([
        dbc.CardBody(
            html.Div(id='output-message', className="text-center fw-bold fs-5")
        )
    ], className="shadow-sm mb-4 border-0 bg-light"), # Tarjeta simple para mensajes

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Resultados de Regresión (Puntaje de Matemáticas)", className="mb-0")),
                dbc.CardBody(
                    dash_table.DataTable(
                        id='regression-table',
                        columns=[
                            {"name": "Métricas de Regresión", "id": "Métricas de Regresión"},
                            {"name": "Valor", "id": "Valor"}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '1.0em'},
                        style_header={
                            'backgroundColor': 'var(--bs-primary)', # Color de encabezado de tabla
                            'color': 'white',
                            'fontWeight': 'bold',
                            'fontSize': '1.1em'
                        },
                        # Estilo condicional para las filas
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                    )
                )
            ], className="shadow-sm h-100"), # Tarjeta con sombra, altura al 100%
            md=6 # Ocupa la mitad del ancho en pantallas medianas y grandes
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Resultados de Clasificación (Nivel Económico)", className="mb-0")),
                dbc.CardBody(
                    dash_table.DataTable(
                        id='classification-table',
                        columns=[
                            {"name": "Métricas de Clasificación", "id": "Métricas de Clasificación"},
                            {"name": "Valor", "id": "Valor"}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '1.0em'},
                        style_header={
                            'backgroundColor': 'var(--bs-primary)', # Color de encabezado de tabla
                            'color': 'white',
                            'fontWeight': 'bold',
                            'fontSize': '1.1em'
                        },
                        # Estilo condicional para las filas
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                    )
                )
            ], className="shadow-sm h-100"), # Tarjeta con sombra, altura al 100%
            md=6 # Ocupa la mitad del ancho en pantallas medianas y grandes
        )
    ], className="mb-4") # Margen inferior para la fila de resultados
], fluid=True) # El contenedor es fluido para ocupar todo el ancho disponible


# Inicializar la app con estilo
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Predicción Saber 11"

# Layout principal con tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Predicción", children=prediccion_tab_layout), # Usamos la variable con el layout mejorado
        dcc.Tab(label="Análisis por Municipio", children=layout_mcpio_reside()),
    ])
])

# Callback para predicción
@app.callback(
    Output('regression-table', 'data'),
    Output('regression-table', 'columns'),
    Output('classification-table', 'data'),
    Output('classification-table', 'columns'),
    Output('output-message', 'children'),
    Input('btn-predict', 'n_clicks'),
    State({'type': 'input-col', 'index': dash.ALL}, 'value')
)
def callback_predict(n_clicks, input_values):
    return ejecutar_predicciones(n_clicks, input_values)

# Callback para gráfico principal
@app.callback(
    Output('grafico-container', 'children'),
    Input('municipio-dropdown', 'value'),
    Input('tabs-variables', 'value')
)
def actualizar_grafico(municipio, variable):
    return generar_grafico(municipio, variable)

# ✅ Callback con manejo de errores y mensajes de depuración
@app.callback(
    Output('grafico-estrato', 'children'),
    Output('grafico-mapa', 'children'),
    Output('grafico-correlacion', 'children'),
    Input('municipio-dropdown', 'value')
)
def actualizar_graficos_adicionales(municipio):
    try:
        print(f"[INFO] Municipio seleccionado: {municipio}")
        graf_estrato = generar_grafico_estrato(municipio)
        graf_mapa = generar_mapa_municipio(municipio)
        graf_corr = generar_correlacion(municipio)
        return graf_estrato, graf_mapa, graf_corr
    except Exception as e:
        print(f"[ERROR] Callback falló: {e}")
        mensaje_error = html.Div(f"⚠️ Error al generar los gráficos para {municipio}")
        return mensaje_error, html.Div(), html.Div()

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)