import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error, r2_score

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """
    Crea una matriz de confusión elegante con métricas
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=2, cbar_kws={'shrink': 0.8},
                ax=ax, xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                   ha='center', va='center', fontsize=11, 
                   color='darkred', weight='bold')
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=14, fontweight='bold')
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, title="Curva ROC"):
    """
    Crea una curva ROC profesional
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Clasificador Aleatorio (AUC = 0.5)')
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Punto Óptimo (umbral = {optimal_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if roc_auc >= 0.9:
        interpretation = "Excelente"
    elif roc_auc >= 0.8:
        interpretation = "Bueno"
    elif roc_auc >= 0.7:
        interpretation = "Aceptable"
    else:
        interpretation = "Pobre"
    
    plt.figtext(0.15, 0.15, f'Interpretación: {interpretation}', 
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_predictions_vs_actual(y_true, y_pred, title="Predicciones vs Valores Reales"):
    """
    Scatter plot de predicciones vs valores reales
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, s=60, 
                c='steelblue', edgecolors='darkblue', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=3, label='Predicción Perfecta (y = x)', alpha=0.8)
    
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "g-", lw=2, alpha=0.8, 
             label=f'Línea de Regresión (y = {z[0]:.2f}x + {z[1]:.2f})')
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    plt.xlabel('Puntaje Matemáticas Real', fontsize=14, fontweight='bold')
    plt.ylabel('Puntaje Matemáticas Predicho', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.1f}%'
    plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_dict, title, top_n=15, orientation='horizontal'):
    """
    Gráfica de importancia de características
    """
    sorted_importance = dict(sorted(importance_dict.items(), 
                                  key=lambda x: abs(x[1]), reverse=True))
    
    top_features = dict(list(sorted_importance.items())[:top_n])
    
    variables = list(top_features.keys())
    values = list(top_features.values())
    
    colors = ["#3CAC8E" if v > 0 else "#4BDD61" for v in values]
    
    if orientation == 'horizontal':
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(variables))
        bars = plt.barh(y_pos, values, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            plt.text(val + (max(values)*0.01 if val > 0 else min(values)*0.01), 
                    i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right', 
                    fontsize=10, weight='bold')
        
        plt.yticks(y_pos, variables, fontsize=11)
        plt.xlabel('Importancia', fontsize=14, fontweight='bold')
        
    else:  
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x_pos = np.arange(len(variables))
        bars = plt.bar(x_pos, values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            plt.text(i, val + (max(values)*0.01 if val > 0 else min(values)*0.01), 
                    f'{val:.3f}', 
                    ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=10, weight='bold', rotation=45)
        
        plt.xticks(x_pos, variables, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Importancia', fontsize=14, fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x' if orientation == 'horizontal' else 'y', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1) if orientation == 'horizontal' else plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#66DBB8", label='Impacto Positivo'),
                      Patch(facecolor="#0CB86B", label='Impacto Negativo')]
    plt.legend(handles=legend_elements, loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig

def plot_importance_comparison(importance_reg, importance_clf, title="Comparación de Importancias"):
    """
    Compara las importancias entre modelos de regresión y clasificación
    """
    common_vars = set(importance_reg.keys()) & set(importance_clf.keys())
    
    if not common_vars:
        print("No hay variables comunes entre los modelos")
        return
    
    reg_values = [importance_reg[var] for var in common_vars]
    clf_values = [importance_clf[var] for var in common_vars]
    var_names = list(common_vars)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = plt.scatter(reg_values, clf_values, s=100, alpha=0.7, 
                         c=range(len(var_names)), cmap='viridis',
                         edgecolors='black', linewidth=1)
    
    for i, var in enumerate(var_names):
        plt.annotate(var, (reg_values[i], clf_values[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(0.05, 0.95, 'Importante en\nambos modelos', 
             transform=ax.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.xlabel('Importancia en Regresión (Puntaje Matemáticas)', fontsize=12, fontweight='bold')
    plt.ylabel('Importancia en Clasificación (Nivel Socioeconómico)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_residual_analysis(y_true, y_pred, title="Análisis de Residuos"):
    """
    Análisis completo de residuos para regresión
    """
    residuals = y_true - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.hist(residuals, bins=30, alpha=0.7, color='skyblue', 
             edgecolor='black', density=True)
    ax1.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {residuals.mean():.2f}')
    ax1.set_xlabel('Residuos')
    ax1.set_ylabel('Densidad')
    ax1.set_title('Distribución de Residuos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normalidad de Residuos)')
    ax2.grid(True, alpha=0.3)
    
    ax3.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Valores Predichos')
    ax3.set_ylabel('Residuos')
    ax3.set_title('Residuos vs Predicciones')
    ax3.grid(True, alpha=0.3)
    
    ax4.scatter(y_true, residuals, alpha=0.6, s=30)
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Valores Reales')
    ax4.set_ylabel('Residuos')
    ax4.set_title('Residuos vs Valores Reales')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_presentation_dashboard(df_test_full):
    """
    Crear todas las graficas
    """
    y_true_reg = df_test_full['punt_matematicas']
    y_pred_reg = df_test_full['punt_matematicas_pred']
    y_true_clf = df_test_full['eco']
    y_pred_proba_clf = df_test_full['eco_pred_proba']
    y_pred_label_clf = df_test_full['eco_pred_label']
    
    importance_reg = {
        'cole_mcpio_ubicacion': 3.3308,
        'estu_tipodocumento': 3.0369,
        'desemp_ingles': 1.9709,
        'estu_mcpio_reside': 1.4625,
        'cole_naturaleza': 0.9739,
        'cole_cod_depto_ubicacion': 0.8829,
        'cole_codigo_icfes': 0.8491,
        'fami_educacionmadre': -0.7295,
        'fami_educacionpadre': -0.7123,
        'fami_cuartoshogar': -0.6007,
        'estu_mcpio_presentacion': 0.5738,
        'cole_jornada': -0.4479,
        'edad': 0.4375,
        'fami_nivel_tecnologia': 0.2796,
        'fami_tienelavadora': 0.1326,
        'fami_estratovivienda': -1.1398,
        'fami_tieneautomovil': -0.8658
    }
    
    importance_clf = {
        'estu_tipodocumento': 0.1063,
        'desemp_ingles': 0.0790,
        'cole_jornada': -0.0279,
        'cole_mcpio_ubicacion': 0.0252,
        'cole_area_ubicacion': 0.0224,
        'cole_caracter': 0.0176,
        'fami_educacionpadre': -0.0168,
        'fami_cuartoshogar': -0.0152,
        'cole_naturaleza': 0.0128,
        'estu_mcpio_presentacion': 0.0073,
        'estu_depto_reside': 0.0055,
        'estu_mcpio_reside': -0.0042,
        'estu_depto_presentacion': 0.0021,
        'fami_educacionmadre': -0.0016,
        'cole_depto_ubicacion': 0.0011,
        'fami_tienelavadora': 0.0268,
        'fami_cuartoshogar_int': 0.0149,
        'cole_cod_depto_ubicacion': 0.0142,
        'fami_estratovivienda': -0.0215
    }
    
    
    fig1 = plot_confusion_matrix(y_true_clf, y_pred_label_clf, 
                                "Matriz de Confusión - Clasificación Nivel Socioeconómico")
    
    fig2 = plot_roc_curve(y_true_clf, y_pred_proba_clf,
                         "Curva ROC - Clasificación Nivel Socioeconómico")
    
    fig3 = plot_predictions_vs_actual(y_true_reg, y_pred_reg,
                                     "Predicciones vs Valores Reales - Puntaje Matemáticas")
    
    fig4 = plot_feature_importance(importance_reg, 
                                  "Importancia de Variables - Regresión (Puntaje Matemáticas)",
                                  top_n=15)
    
    fig5 = plot_feature_importance(importance_clf,
                                  "Importancia de Variables - Clasificación (Nivel Socioeconómico)",
                                  top_n=15)
    
    fig6 = plot_importance_comparison(importance_reg, importance_clf,
                                     "Comparación de Importancias entre Modelos")
    
    fig7 = plot_residual_analysis(y_true_reg, y_pred_reg,
                                 "Análisis Completo de Residuos")
    
    figures = [
        (fig1, 'confusion_matrix.png'),
        (fig2, 'roc_curve.png'),
        (fig3, 'scatter_predictions.png'),
        (fig4, 'importance_regression.png'),
        (fig5, 'importance_classification.png'),
        (fig6, 'importance_comparison.png'),
        (fig7, 'residual_analysis.png')
    ]
    
    for fig, filename in figures:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    print("Todas las gráficas guardadas exitosamente!")
    print("Archivos creados:")
    for _, filename in figures:
        print(f"   • {filename}")
    
    #plt.show()


create_presentation_dashboard(df_test_full=pd.read_csv("./government_icfes_dataset/data/predicciones_completas.csv"))
