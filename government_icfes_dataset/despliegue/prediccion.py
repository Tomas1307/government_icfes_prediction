import pandas as pd
import torch
import numpy as np # Necesario para np.exp en la predicción de clasificación
import joblib
import os
import dash # Necesario para html.Div en ejecutar_predicciones si no está importado globalmente
from dash import html # Para html.Div

# Importar tus clases de modelo y encoder, asegúrate de que estén en las rutas correctas
from convolutional import ConvRegressorModel, ConvClassifierModel
from label_encoder_unk import LabelEncoderWithUnknown

# --- La función predict_new_sample que proporcionaste (tal cual) ---
def predict_new_sample(sample_data, encoders_file):
    """
    Hace predicción en una nueva muestra
    """

    # Asegúrate de que estas rutas de modelo sean correctas en tu entorno
    reg_model_path = "./best_models/regression/model_regression_20250522_180347.pt"
    clf_model_path = "./best_models/classification/model_classification_20250522_100219.pt"

    # Cargar el archivo de encoders/scalers
    # NOTA: Este archivo (all_encoders_*.pkl) DEBE contener un diccionario
    # con las claves 'reg_encoders', 'reg_scaler', etc., como espera esta función.
    # Si tu analisis_modelos.ipynb guarda los encoders/scalers por separado,
    # esta parte fallará.
    try:
        encoders_data = joblib.load(encoders_file)
    except FileNotFoundError:
        print(f"❌ Error: El archivo de encoders '{encoders_file}' no fue encontrado.")
        return {"error": f"Archivo de encoders no encontrado: {encoders_file}"}
    except Exception as e:
        print(f"❌ Error al cargar el archivo de encoders: {e}")
        return {"error": f"Error al cargar encoders: {e}"}

    # Columnas esperadas en la cadena de entrada.
    # NOTA: Esta lista incluye 'punt_matematicas' y 'eco'. Tu cadena de entrada
    # 'sample_data' DEBE contener valores para estas, aunque luego se dropeen.
    columns = ['periodo','estu_tipodocumento','cole_area_ubicacion','cole_caracter',
                'cole_cod_dane_establecimiento','cole_cod_depto_ubicacion','cole_codigo_icfes',
                'cole_depto_ubicacion','cole_jornada','cole_mcpio_ubicacion','cole_naturaleza',
                'estu_depto_presentacion','estu_depto_reside','estu_genero','estu_mcpio_presentacion',
                'estu_mcpio_reside','fami_cuartoshogar','fami_educacionmadre','fami_educacionpadre',
                'fami_estratovivienda','fami_tieneautomovil','fami_tienelavadora','desemp_ingles',
                'punt_ingles','punt_matematicas','eco','fami_cuartoshogar_int','edad','fami_nivel_tecnologia']

    try:
        values = sample_data.strip().split(',')
        if len(values) != len(columns):
            return {"error": f"Número incorrecto de valores. Se esperaban {len(columns)} pero se recibieron {len(values)}."}
        df = pd.DataFrame([values], columns=columns)
    except Exception as e:
        return {"error": f"Error al parsear la cadena de entrada o crear DataFrame: {e}. Formato esperado: 'v1,v2,...,vn'."}

    # Asegúrate de que las columnas numéricas existan antes de la conversión
    numeric_cols = ['periodo','cole_cod_dane_establecimiento','cole_cod_depto_ubicacion',
                    'cole_codigo_icfes','fami_cuartoshogar','fami_estratovivienda','fami_tieneautomovil',
                    'fami_tienelavadora','punt_ingles','punt_matematicas','eco','fami_cuartoshogar_int',
                    'edad','fami_nivel_tecnologia']

    for col in numeric_cols:
        if col in df.columns: # Verificar si la columna existe en el DataFrame
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Advertencia: La columna numérica '{col}' no se encontró en el DataFrame de entrada.")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === REGRESIÓN ===
    # Asegúrate de que 'punt_matematicas' y 'eco' estén en df antes de dropearlas
    # y de que los nombres de las columnas en encoders_data['reg_cat_columns'] y
    # encoders_data['reg_num_columns'] coincidan con el df_reg resultante.
    df_reg = df.copy()
    if 'eco' in df_reg.columns:
        df_reg = df_reg.drop(columns=['eco'])
    if 'punt_matematicas' in df_reg.columns:
        df_reg = df_reg.drop(columns=['punt_matematicas'])


    # Iterar sobre las columnas categóricas definidas en encoders_data
    for col in encoders_data['reg_cat_columns']:
        if col in df_reg.columns:
            # Asegurarse de que el encoder exista para esta columna
            if col in encoders_data['reg_encoders']:
                df_reg[col] = encoders_data['reg_encoders'][col].transform(df_reg[col].astype(str))
            else:
                print(f"Advertencia: Encoder para '{col}' no encontrado en reg_encoders. Asignando valor desconocido.")
                # Fallback: Asignar el token desconocido si el encoder no está presente
                # Esto requeriría que LabelEncoderWithUnknown esté bien implementado y tenga el token.
                df_reg[col] = -1 # O el índice del token desconocido si se conoce


    # Escalar columnas numéricas
    # Asegurarse de que las columnas numéricas existan en df_reg antes de escalarlas
    try:
        df_reg[encoders_data['reg_num_columns']] = encoders_data['reg_scaler'].transform(df_reg[encoders_data['reg_num_columns']])
    except Exception as e:
        return {"error": f"Error durante el escalado de características numéricas de regresión: {e}. Verifique los datos de entrada."}


    reg_cat = torch.tensor(df_reg[encoders_data['reg_cat_columns']].values, dtype=torch.long).to(device)
    reg_num = torch.tensor(df_reg[encoders_data['reg_num_columns']].values, dtype=torch.float32).to(device)

    # CAMBIO: usar dimensiones guardadas en lugar de calcularlas
    reg_cat_dims = encoders_data['reg_cat_dims']
    reg_num_features = len(encoders_data['reg_num_columns'])

    reg_model = ConvRegressorModel(
        cat_dims=reg_cat_dims,
        num_features=reg_num_features,
        embedding_dim=4,
        conv_filters=[32, 64],
        dense_units=64
    ).to(device)

    try:
        reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
        reg_model.eval()
    except Exception as e:
        return {"error": f"Error al cargar o inicializar el modelo de regresión: {e}"}

    with torch.no_grad():
        reg_prediction = reg_model(reg_cat, reg_num).cpu().numpy()[0][0]

    # === CLASIFICACIÓN ===
    # Asegúrate de que 'punt_matematicas' y 'eco' estén en df antes de dropearlas
    df_clf = df.copy()
    if 'punt_matematicas' in df_clf.columns:
        df_clf = df_clf.drop(columns=['punt_matematicas'])
    if 'eco' in df_clf.columns:
        df_clf = df_clf.drop(columns=['eco'])

    for col in encoders_data['clf_cat_columns']:
        if col in df_clf.columns:
            if col in encoders_data['clf_encoders']:
                df_clf[col] = encoders_data['clf_encoders'][col].transform(df_clf[col].astype(str))
            else:
                print(f"Advertencia: Encoder para '{col}' no encontrado en clf_encoders. Asignando valor desconocido.")
                df_clf[col] = -1 # O el índice del token desconocido si se conoce


    try:
        df_clf[encoders_data['clf_num_columns']] = encoders_data['clf_scaler'].transform(df_clf[encoders_data['clf_num_columns']])
    except Exception as e:
        return {"error": f"Error durante el escalado de características numéricas de clasificación: {e}. Verifique los datos de entrada."}

    clf_cat = torch.tensor(df_clf[encoders_data['clf_cat_columns']].values, dtype=torch.long).to(device)
    clf_num = torch.tensor(df_clf[encoders_data['clf_num_columns']].values, dtype=torch.float32).to(device)

    clf_cat_dims = encoders_data['clf_cat_dims']
    clf_num_features = len(encoders_data['clf_num_columns'])

    clf_model = ConvClassifierModel(
        cat_dims=clf_cat_dims,
        num_features=clf_num_features,
        embedding_dim=8,
        conv_filters=[32, 64],
        dense_units=64
    ).to(device)

    try:
        clf_model.load_state_dict(torch.load(clf_model_path, map_location=device))
        clf_model.eval()
    except Exception as e:
        return {"error": f"Error al cargar o inicializar el modelo de clasificación: {e}"}


    with torch.no_grad():
        clf_prediction_raw = clf_model(clf_cat, clf_num).cpu().numpy()[0][0]
        # Aplicar Sigmoid para obtener la probabilidad en clasificación binaria
        clf_prediction_proba = 1 / (1 + np.exp(-clf_prediction_raw))
        clf_prediction_label = int(clf_prediction_proba >= 0.5)

    return {
        'punt_matematicas_pred': float(reg_prediction),
        'eco_pred_proba': float(clf_prediction_proba),
        'eco_pred_label': clf_prediction_label
    }

# --- Función 'ejecutar_predicciones' para el callback de Dash ---
# Esta función es la que tablero.py importará y llamará
def ejecutar_predicciones(n_clicks, input_values):
    if n_clicks is None:
        return [], [], [], [], "" # Estado inicial para Dash

    try:


        # Para que esto funcione, los input_values de Dash (que vienen de datosylayout.py::cols con 27 items)
        # deben ser *extendidos* con 2 valores para punt_matematicas y eco para coincidir con `columns` (29 items).
        # Un ejemplo:
        temp_input_values = list(input_values)
        temp_input_values.insert(24, "0") # Placeholder para punt_matematicas
        temp_input_values.insert(25, "0") # Placeholder para eco
        
        input_data_string = ",".join([str(val) for val in temp_input_values])

        # Llamar a la función predict_new_sample
        # El archivo de encoders puede ser el que guardaste si es el formato all_encoders_*.pkl
        # Si no, deberás ajustar esta ruta o la forma en que predict_new_sample carga los encoders.
        encoders_path = "./encoders/all_encoders_20250525_164501.pkl" # Revisa este nombre de archivo
        predictions = predict_new_sample(input_data_string, encoders_path)

        if "error" in predictions:
            return [], [], [], [], html.Div(f"Error en la predicción: {predictions['error']}", style={'color': 'red'})

        # Formatear las predicciones para las Dash DataTables
        reg_data = [{'Métricas de Regresión': 'Puntaje de Matemáticas Predicho', 'Valor': f"{predictions['punt_matematicas_pred']:.2f}"}]
        reg_columns = [{"name": "Métricas de Regresión", "id": "Métricas de Regresión"}, {"name": "Valor", "id": "Valor"}]

        clf_data = [
            {'Métricas de Clasificación': 'Probabilidad (Nivel Económico 1)', 'Valor': f"{predictions['eco_pred_proba']:.4f}"},
            {'Métricas de Clasificación': 'Clase Predicha (Nivel Económico)', 'Valor': str(predictions['eco_pred_label']) + " (0=Bajo, 1=Alto/Medio)"}
        ]
        clf_columns = [{"name": "Métricas de Clasificación", "id": "Métricas de Clasificación"}, {"name": "Valor", "id": "Valor"}]

        return reg_data, reg_columns, clf_data, clf_columns, html.Div("✅ Predicciones realizadas exitosamente.", style={'color': 'green'})

    except Exception as e:
        # Capturar cualquier error inesperado para mostrarlo en la interfaz
        print(f"Error inesperado en ejecutar_predicciones: {e}")
        return [], [], [], [], html.Div(f"Un error inesperado ocurrió: {e}", style={'color': 'red'})

# --- Ejemplo de uso (solo para probar este archivo prediccion.py directamente) ---
if __name__ == "__main__":
    # Esta cadena de ejemplo DEBE tener 29 valores (27 características + 2 placeholders para punt_matematicas y eco)
    # según la lista `columns` dentro de `predict_new_sample`.
    # Puntaje_matematicas (48.17) y eco (0) están incluidos como placeholders en este ejemplo.
    sample_data_test = "20102,TI,urbano,TÉCNICO,186219000070,86,29025,OTROS,MAÑANA,Colón,OFICIAL,OTROS,OTROS,F,SIBUNDOY,COLÓN,5,SECUNDARIA (BACHILLERATO) INCOMPLETA,PRIMARIA COMPLETA,1,0,1,A1,49.66,48.17,0,5,31.0,1"
    
    # Asegúrate de que este archivo de encoders exista y tenga el formato esperado
    encoders_path_for_test = "./encoders/all_encoders_20250525_164501.pkl" # Revisa este nombre

    print(f"Realizando predicción de prueba para la entrada:\n{sample_data_test}\n")
    predictions_test = predict_new_sample(sample_data_test, encoders_path_for_test)

    if "error" in predictions_test:
        print(f"❌ Error en la predicción de prueba: {predictions_test['error']}")
    else:
        print("✅ Predicciones de prueba exitosas:")
        print(f"  Puntaje de Matemáticas Predicho: {predictions_test['punt_matematicas_pred']:.2f}")
        print(f"  Probabilidad de Nivel Económico (eco=1): {predictions_test['eco_pred_proba']:.4f}")
        print(f"  Nivel Económico Predicho (0=Bajo, 1=Alto/Medio): {predictions_test['eco_pred_label']}")