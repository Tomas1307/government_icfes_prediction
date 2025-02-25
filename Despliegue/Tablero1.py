import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Cargar datos
file_path = "data/modelo_con_predicciones.csv"  
df = pd.read_csv(file_path)

# Crear una nueva columna 'state' a partir de las columnas one-hot de los estados
state_columns = [col for col in df.columns if col.startswith("state_")]
df["state"] = df[state_columns].idxmax(axis=1).str.replace("state_", "")

# Crear categorías de precio
bins = [df['y_actual'].min(), df['y_actual'].quantile(0.33), df['y_actual'].quantile(0.66), df['y_actual'].max()]
labels = ['Bajo', 'Medio', 'Alto']
df['price_category'] = pd.cut(df['y_actual'], bins=bins, labels=labels)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Variables de amenidades disponibles
amenities_variables = ['Cable or Satellite', 'Clubhouse', 'Fireplace', 'Garbage Disposal', 
                       'Internet Access', 'Parking', 'Pool', 'Storage']

# Layout del dashboard
app.layout = dbc.Container([
    html.H1("Dashboard de Análisis de Alquileres"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Seleccionar Ciudad:"),
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': city.replace("cityname_", ""), 'value': city} for city in df.filter(like='cityname_').columns],
                multi=True,
                placeholder="Seleccione ciudades"
            )
        ], width=4),
        dbc.Col([
            html.Label("Seleccionar Estado:"),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': state.replace("state_", ""), 'value': state} for state in state_columns],
                multi=True,
                placeholder="Seleccione estados"
            )
        ], width=4),
        dbc.Col([
            html.Label("Seleccionar Variable para Boxplot:"),
            dcc.Dropdown(
                id='boxplot-variable-dropdown',
                options=[{'label': col, 'value': col} for col in amenities_variables],
                value='Pool',
                clearable=False
            )
        ], width=4),
    ]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='scatter-price-sqft')], width=6),
        dbc.Col([dcc.Graph(id='boxplot-price-city')], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Seleccionar Amenidades para Análisis:"),
            dcc.Dropdown(
                id='amenities-dropdown',
                options=[{'label': col, 'value': col} for col in amenities_variables],
                multi=True,
                value=['Pool', 'Parking'],
                placeholder="Seleccione las amenidades"
            )
        ], width=6)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([dcc.Graph(id='amenities-impact')], width=6),
        dbc.Col([dcc.Graph(id='price-segmentation')], width=6)
    ]),
    
    html.Hr(),

    dbc.Row([
        dbc.Col([dcc.Graph(id='price-heatmap')], width=12)
    ])
])

# Callbacks
@app.callback(
    [Output('scatter-price-sqft', 'figure'),
     Output('boxplot-price-city', 'figure'),
     Output('amenities-impact', 'figure'),
     Output('price-segmentation', 'figure'),
     Output('price-heatmap', 'figure')],
    [Input('city-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('boxplot-variable-dropdown', 'value'),
     Input('amenities-dropdown', 'value')]
)
def update_graphs(selected_cities, selected_states, selected_boxplot_variable, selected_amenities):
    scatter_fig = px.scatter(df, x='square_feet', y='y_actual', color='price_category',
                             title="Precio vs. Metros Cuadrados", labels={'y_actual': 'Precio'})
    
    boxplot_fig = px.box(df, x=selected_boxplot_variable, y='y_actual',
                          title=f"Distribución de Precios por {selected_boxplot_variable}", points=False)  
    
    amenities_impact = df[selected_amenities + ['y_actual']].melt(id_vars='y_actual')
    amenities_fig = px.box(amenities_impact, x='variable', y='y_actual', title="Impacto de Amenidades en Precio",
                           boxmode='overlay', points=False)
    
    segmentation_fig = px.histogram(df, x='price_category', title="Segmentación de Precios")
    
    state_avg_price = df.groupby('state')['y_actual'].mean().reset_index()
    heatmap_fig = px.choropleth(
        state_avg_price, 
        locations='state', 
        locationmode='USA-states',
        color='y_actual',
        color_continuous_scale="blues", 
        scope="usa",
        title="Precio promedio de alquiler por estado",
        labels={'y_actual': 'Precio Promedio'}
    )
    
    for state in state_avg_price['state']:
        state_data = state_avg_price[state_avg_price['state'] == state]
        heatmap_fig.add_trace(
            go.Scattergeo(
                locations=state_data['state'],
                locationmode='USA-states',
                text=state,
                mode='text',
                showlegend=False
            )
        )
    
    heatmap_fig.update_layout(height=700)  
    
    return scatter_fig, boxplot_fig, amenities_fig, segmentation_fig, heatmap_fig

if __name__ == '__main__':
    app.run_server(debug=True)


