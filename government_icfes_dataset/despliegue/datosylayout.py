from dash import html, dcc, dash_table
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


def get_layout():
    return html.Div([
        html.H2("📊 Predicción Saber 11 con entradas manuales"),

        html.Div([
            html.Div([
                html.Label(col),
                dcc.Input(
                    id={'type': 'input-col', 'index': col},
                    type='text',
                    value=str(default_values[col]),
                    style={'width': '100%'}
                )
            ], style={'margin-bottom': '10px', 'width': '300px'})
            for col in cols
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),

        html.Button("Predecir", id='btn-predict', n_clicks=0, style={'margin-top': '20px'}),

        html.H3("📈 Predicción Regresión (punt_matematicas)"),
        dash_table.DataTable(id='regression-table', style_table={'overflowX': 'auto'}),

        html.H3("📊 Predicción Clasificación (eco)"),
        dash_table.DataTable(id='classification-table', style_table={'overflowX': 'auto'}),

        html.Div(id='output-message', style={'margin-top': '20px', 'color': 'red'})
    ])
