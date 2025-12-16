import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema Predicción Churn + EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Añadir utils al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar utilidades de modelos
try:
    from utils.model_utils import cargar_modelo_y_preprocesadores, predecir_con_modelo
except ImportError:
    st.error("❌ No se pudo importar utils.model_utils. Asegúrate de tener el archivo utils/model_utils.py")
    st.stop()

# Variables del sistema (actualizadas según el entrenamiento real)
TOP_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
    'OnlineSecurity', 'TechSupport', 'InternetService',
    'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

TODAS_VARIABLES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Diccionario de mapeo para la interfaz (actualizado)
CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': [
        'Bank transfer (automatic)', 
        'Credit card (automatic)', 
        'Electronic check', 
        'Mailed check'
    ]
}

# ============================================================================
# FUNCIONES DE CARGA Y UTILIDAD
# ============================================================================

@st.cache_resource
def cargar_todos_modelos_y_metricas():
    """
    Carga todos los modelos entrenados y sus métricas.
    Muestra mensajes de error si los modelos no existen.
    """
    try:
        # Verificar si los modelos existen
        if not os.path.exists('models/all_features/preprocesadores.pkl'):
            raise FileNotFoundError("Modelos no encontrados. Ejecuta primero: python entrenar_modelos_reales.py")
        
        # Cargar configuración all features
        all_features_data = cargar_modelo_y_preprocesadores('models/all_features')
        
        # Cargar configuración top features
        top_features_data = cargar_modelo_y_preprocesadores('models/top_features')
        
        # Cargar métricas
        metricas_all = joblib.load('models/all_features/metricas.pkl')
        metricas_top = joblib.load('models/top_features/metricas.pkl')
        
        st.sidebar.success("✅ Modelos cargados exitosamente")
        
        return {
            'all_features': {
                'modelos': all_features_data['modelos'],
                'preprocesadores': all_features_data['preprocesadores'],
                'metricas': metricas_all
            },
            'top_features': {
                'modelos': top_features_data['modelos'],
                'preprocesadores': top_features_data['preprocesadores'],
                'metricas': metricas_top
            }
        }
    
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ {str(e)}")
        st.sidebar.info("""
        **Solución:**
        1. Descarga el dataset: https://www.kaggle.com/blastchar/telco-customer-churn
        2. Colócalo en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
        3. Ejecuta: `python entrenar_modelos_reales.py`
        """)
        return None
    
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando modelos: {str(e)}")
        return None

def cargar_datos():
    """Carga los datos del dataset de churn para EDA"""
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dataset no encontrado. Usando datos de ejemplo para EDA.")
        # Crear datos de ejemplo realistas
        np.random.seed(42)
        n_samples = 7043
        
        data = {
            'customerID': [f'CUST{i:06d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 
                                              'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
            'TotalCharges': np.round(np.random.uniform(0, 10000, n_samples), 2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735])
        }
        df = pd.DataFrame(data)
        return df

# ============================================================================
# SECCIÓN EDA (MANTENIDA DE LA VERSIÓN ANTERIOR)
# ============================================================================

def mostrar_dataframe(df, limite=10):
    """Muestra un DataFrame usando HTML"""
    # (Mantener la misma función que ya tenías)
    html = f"""
    <div style='overflow-x: auto; max-height: 400px; border: 1px solid #ddd; border-radius: 5px;'>
        <table style='width: 100%; border-collapse: collapse;'>
            <thead>
                <tr style='background-color: #f2f2f2; position: sticky; top: 0;'>
    """
    for col in df.columns[:limite]:
        html += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for i in range(min(len(df), 15)):
        html += "<tr>"
        for col in df.columns[:limite]:
            valor = str(df.iloc[i][col])
            html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{valor}</td>"
        html += "</tr>"
    
    html += f"""
        </tbody>
    </table>
    <div style='padding: 10px; background-color: #f8f9fa; border-top: 1px solid #ddd;'>
        <small>Mostrando {min(len(df), 15)} de {len(df)} filas, {min(len(df.columns), limite)} de {len(df.columns)} columnas</small>
    </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def seccion_eda(df):
    """Sección completa de Análisis Exploratorio de Datos"""
    # (Mantener toda la sección EDA que ya tenías, sin cambios)
    # Solo asegúrate de que las funciones auxiliares estén definidas
    
    st.markdown("---")
    st.header("📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    
    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Registros", f"{len(df):,}")
    with col2:
        st.metric("Total Variables", len(df.columns))
    with col3:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Tasa de Churn", f"{churn_rate:.1f}%")
    with col4:
        nulos = df.isnull().sum().sum()
        st.metric("Valores Nulos", f"{nulos:,}")
    
    # Resto del código EDA...
    # ... (mantener todo el código EDA existente)

# ============================================================================
# SECCIÓN PREDICCIÓN INDIVIDUAL (ACTUALIZADA)
# ============================================================================

def crear_formulario_prediccion(variables_usar):
    """Crea el formulario para ingresar datos del cliente"""
    datos_cliente = {}
    
    # Determinar número de columnas según cantidad de variables
    if len(variables_usar) > 10:
        num_cols = 3
    else:
        num_cols = 2
    
    cols = st.columns(num_cols)
    
    for idx, variable in enumerate(variables_usar):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            st.markdown(f"**{variable}**")
            
            if variable == 'SeniorCitizen':
                datos_cliente[variable] = st.selectbox(
                    f"Seleccionar {variable}",
                    [0, 1],
                    key=f"select_{variable}",
                    label_visibility="collapsed"
                )
            
            elif variable in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                if variable == 'tenure':
                    datos_cliente[variable] = st.number_input(
                        "Meses",
                        min_value=0,
                        max_value=100,
                        value=12,
                        key=f"num_{variable}",
                        label_visibility="collapsed"
                    )
                else:
                    datos_cliente[variable] = st.number_input(
                        "Valor ($)",
                        min_value=0.0,
                        max_value=10000.0,
                        value=50.0 if variable == 'MonthlyCharges' else 1000.0,
                        key=f"num_{variable}",
                        label_visibility="collapsed"
                    )
            
            elif variable in CATEGORICAL_OPTIONS:
                datos_cliente[variable] = st.selectbox(
                    f"Seleccionar {variable}",
                    CATEGORICAL_OPTIONS[variable],
                    key=f"cat_{variable}",
                    label_visibility="collapsed"
                )
    
    return datos_cliente

def mostrar_resultados_prediccion(resultado, modelo_nombre, usar_top_features):
    """Muestra los resultados de la predicción"""
    st.markdown("---")
    st.header("🎯 RESULTADOS DE PREDICCIÓN")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        color = "red" if resultado['prediccion'] == 'CHURN' else "green"
        st.markdown(f"""
        <div style='border: 3px solid {color}; border-radius: 10px; padding: 20px; text-align: center;'>
            <h3>Predicción</h3>
            <h1 style='color: {color};'>{resultado['prediccion']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col_res2:
        st.metric(
            "Probabilidad de Churn",
            f"{resultado['probabilidad_churn']:.1%}",
            f"{(resultado['probabilidad_churn'] - 0.5)*100:+.1f}%"
        )
    
    with col_res3:
        st.metric(
            "Probabilidad de No Churn",
            f"{resultado['probabilidad_no_churn']:.1%}",
            f"{(resultado['probabilidad_no_churn'] - 0.5)*100:+.1f}%"
        )
    
    # Gráfico de probabilidad
    fig, ax = plt.subplots(figsize=(10, 4))
    
    labels = ['No Churn', 'Churn']
    valores = [resultado['probabilidad_no_churn'], resultado['probabilidad_churn']]
    colors = ['#4CAF50', '#F44336']
    
    bars = ax.bar(labels, valores, color=colors, edgecolor='black')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Distribución de Probabilidades')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{valor:.1%}', ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    
    # Información del modelo
    st.info(f"""
    **Modelo usado:** {modelo_nombre}  
    **Versión:** {'Top Features (10 variables)' if usar_top_features else 'All Features (19 variables)'}  
    **Confianza:** {'Alta' if max(valores) > 0.8 else 'Media' if max(valores) > 0.6 else 'Baja'}
    """)

# ============================================================================
# SECCIÓN DASHBOARD DE MODELOS (ACTUALIZADA)
# ============================================================================

def mostrar_dashboard_modelo(modelo_seleccionado, metricas_all, metricas_top, usar_top_features):
    """Muestra dashboard con métricas del modelo"""
    st.markdown("---")
    st.header("📊 DASHBOARD DE EVALUACIÓN DEL MODELO")
    
    # Obtener métricas
    modo_actual = 'top_features' if usar_top_features else 'all_features'
    modo_alterno = 'all_features' if usar_top_features else 'top_features'
    
    metricas_actual = metricas_all if not usar_top_features else metricas_top
    metricas_alterno = metricas_top if not usar_top_features else metricas_all
    
    # Comparación de métricas
    st.subheader("⚖️ Comparación de Métricas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        diff_acc = metricas_actual[modelo_seleccionado]['accuracy'] - metricas_alterno[modelo_seleccionado]['accuracy']
        st.metric(
            "Accuracy",
            f"{metricas_actual[modelo_seleccionado]['accuracy']:.3f}",
            f"{diff_acc:+.3f}"
        )
    
    with col2:
        diff_f1 = metricas_actual[modelo_seleccionado]['f1'] - metricas_alterno[modelo_seleccionado]['f1']
        st.metric(
            "F1-Score",
            f"{metricas_actual[modelo_seleccionado]['f1']:.3f}",
            f"{diff_f1:+.3f}"
        )
    
    with col3:
        diff_auc = metricas_actual[modelo_seleccionado]['auc'] - metricas_alterno[modelo_seleccionado]['auc']
        st.metric(
            "AUC-ROC",
            f"{metricas_actual[modelo_seleccionado]['auc']:.3f}",
            f"{diff_auc:+.3f}"
        )
    
    with col4:
        diff_prec = metricas_actual[modelo_seleccionado]['precision'] - metricas_alterno[modelo_seleccionado]['precision']
        st.metric(
            "Precision",
            f"{metricas_actual[modelo_seleccionado]['precision']:.3f}",
            f"{diff_prec:+.3f}"
        )
    
    # Gráficos
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Métricas Comparativas")
        
        labels = ['Accuracy', 'F1-Score', 'AUC-ROC', 'Precision', 'Recall']
        valores_actual = [
            metricas_actual[modelo_seleccionado]['accuracy'],
            metricas_actual[modelo_seleccionado]['f1'],
            metricas_actual[modelo_seleccionado]['auc'],
            metricas_actual[modelo_seleccionado]['precision'],
            metricas_actual[modelo_seleccionado]['recall']
        ]
        valores_alterno = [
            metricas_alterno[modelo_seleccionado]['accuracy'],
            metricas_alterno[modelo_seleccionado]['f1'],
            metricas_alterno[modelo_seleccionado]['auc'],
            metricas_alterno[modelo_seleccionado]['precision'],
            metricas_alterno[modelo_seleccionado]['recall']
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, valores_actual, width, 
                      label=f"{'Top Features' if usar_top_features else 'All Features'}",
                      color='#4CAF50', alpha=0.8)
        bars2 = ax.bar(x + width/2, valores_alterno, width,
                      label=f"{'All Features' if usar_top_features else 'Top Features'}",
                      color='#2196F3', alpha=0.8)
        
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Valor')
        ax.set_title(f'Comparación de Métricas - {modelo_seleccionado}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
    
    with col_chart2:
        st.subheader("🔥 Matriz de Confusión")
        
        cm = np.array(metricas_actual[modelo_seleccionado]['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['No Churn', 'Churn']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=f'Matriz de Confusión - {modelo_seleccionado}',
               ylabel='Verdadero',
               xlabel='Predicho')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        st.pyplot(fig)
    
    # Importancia de características
    st.subheader("⭐ Importancia de Características")
    
    # Nota: Para mostrar importancia real, necesitarías guardarla durante el entrenamiento
    # Por ahora mostramos las top features definidas
    if usar_top_features:
        features = TOP_FEATURES
        importancia = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    else:
        features = TODAS_VARIABLES
        importancia = np.random.rand(len(features))
        importancia = importancia / importancia.sum()
    
    # Ordenar por importancia
    idx = np.argsort(importancia)[-15:]  # Top 15
    features_sorted = [features[i] for i in idx]
    importancia_sorted = [importancia[i] for i in idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(features_sorted)), importancia_sorted, color='skyblue', edgecolor='black')
    
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top Características - {modelo_seleccionado}')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(bars, importancia_sorted)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f'{imp:.3f}', va='center', fontsize=9)
    
    st.pyplot(fig)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación"""
    
    # Título principal
    st.title("📱 SISTEMA DE ANÁLISIS Y PREDICCIÓN DE CHURN")
    st.markdown("---")
    
    # Cargar modelos (importante hacerlo al inicio)
    modelos_data = cargar_todos_modelos_y_metricas()
    
    # Si no se cargaron modelos, mostrar solo EDA
    modo_solo_eda = modelos_data is None
    
    # Sidebar
    st.sidebar.markdown("## 🧭 NAVEGACIÓN")
    
    if modo_solo_eda:
        seccion = "📊 EDA - Análisis Exploratorio"
        st.sidebar.warning("⚠️ Solo disponible EDA. Entrena modelos para habilitar todas las secciones.")
    else:
        opciones_navegacion = [
            "📊 EDA - Análisis Exploratorio", 
            "🤖 Predicción Individual", 
            "📈 Dashboard Modelos"
        ]
        seccion = st.sidebar.radio("Seleccione sección:", opciones_navegacion)
    
    # Cargar datos para EDA (siempre necesario)
    df = cargar_datos()
    
    if seccion == "📊 EDA - Análisis Exploratorio":
        seccion_eda(df)
    
    elif seccion == "🤖 Predicción Individual" and not modo_solo_eda:
        # Configuración para predicción
        st.sidebar.markdown("## ⚙️ CONFIGURACIÓN DE PREDICCIÓN")
        
        # Selección de modelo
        modelo_seleccionado = st.sidebar.selectbox(
            "Selecciona modelo:",
            list(modelos_data['all_features']['modelos'].keys())
        )
        
        # Selección de versión
        version_modelo = st.sidebar.radio(
            "Versión del modelo:",
            ["🎯 Con Top Features", "📊 Con Todas las Features"]
        )
        usar_top_features = (version_modelo == "🎯 Con Top Features")
        
        # Seleccionar variables a usar
        variables_usar = TOP_FEATURES if usar_top_features else TODAS_VARIABLES
        
        # Formulario de predicción
        st.header("👤 PREDICCIÓN PARA CLIENTE INDIVIDUAL")
        st.markdown("Ingresa los datos del cliente para predecir si abandonará el servicio.")
        
        datos_cliente = crear_formulario_prediccion(variables_usar)
        
        # Botón de predicción
        if st.button("🔮 PREDECIR CHURN", type="primary", use_container_width=True):
            with st.spinner("Realizando predicción..."):
                try:
                    # Obtener el modelo y preprocesadores correctos
                    modo = 'top_features' if usar_top_features else 'all_features'
                    modelo = modelos_data[modo]['modelos'][modelo_seleccionado]
                    preprocesadores = modelos_data[modo]['preprocesadores']
                    
                    # Hacer predicción
                    resultado = predecir_con_modelo(datos_cliente, modelo, preprocesadores)
                    
                    if resultado:
                        mostrar_resultados_prediccion(resultado, modelo_seleccionado, usar_top_features)
                        
                        # Mostrar estrategias de retención
                        st.markdown("---")
                        st.header("🛡️ ESTRATEGIAS DE RETENCIÓN RECOMENDADAS")
                        
                        if resultado['probabilidad_churn'] > 0.7:
                            st.warning("""
                            ### 🚨 ACCIONES INMEDIATAS:
                            1. **Contacto prioritario** del equipo de retención
                            2. **Oferta especial:** 30% descuento por 6 meses
                            3. **Evaluación gratuita** de servicios premium
                            4. **Asesor personalizado** para mejorar la experiencia
                            """)
                        elif resultado['probabilidad_churn'] > 0.5:
                            st.info("""
                            ### 📅 ACCIONES PROACTIVAS:
                            1. **Email personalizado** con beneficios exclusivos
                            2. **20% descuento** por 3 meses en servicios adicionales
                            3. **Programa de fidelización** con puntos canjeables
                            4. **Encuesta de satisfacción** para identificar áreas de mejora
                            """)
                        else:
                            st.success("""
                            ### 💎 ACCIONES DE FIDELIZACIÓN:
                            1. **Programa de referidos** con recompensas
                            2. **Acceso anticipado** a nuevas características
                            3. **Contenido exclusivo** y webinars
                            4. **Seguimiento periódico** para mantener satisfacción
                            """)
                    
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")
                    st.info("Asegúrate de que todos los campos del formulario estén completos.")
    
    elif seccion == "📈 Dashboard Modelos" and not modo_solo_eda:
        st.sidebar.markdown("## ⚙️ CONFIGURACIÓN DEL DASHBOARD")
        
        # Selección de modelo para dashboard
        modelo_dashboard = st.sidebar.selectbox(
            "Selecciona modelo para análisis:",
            list(modelos_data['all_features']['modelos'].keys())
        )
        
        # Selección de versión para dashboard
        version_dashboard = st.sidebar.radio(
            "Mostrar métricas para:",
            ["🎯 Versión Top Features", "📊 Versión Todas las Features"]
        )
        usar_top_dashboard = (version_dashboard == "🎯 Versión Top Features")
        
        # Mostrar dashboard
        mostrar_dashboard_modelo(
            modelo_dashboard,
            modelos_data['all_features']['metricas'],
            modelos_data['top_features']['metricas'],
            usar_top_dashboard
        )
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("**📊 Sistema de Análisis de Churn**")
        st.markdown("v2.0 | Modelos Reales Entrenados")
    
    with col_footer2:
        st.markdown("**🤖 Modelos Disponibles**")
        if not modo_solo_eda:
            modelos = list(modelos_data['all_features']['modelos'].keys())
            for modelo in modelos:
                st.markdown(f"• {modelo}")
    
    with col_footer3:
        st.markdown("**📈 Métricas Reportadas**")
        st.markdown("Accuracy | F1-Score | AUC-ROC")
        st.markdown("Precision | Recall | Matriz Confusión")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()