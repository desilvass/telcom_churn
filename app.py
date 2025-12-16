import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN PARA STREAMLIT CLOUD
# ============================================================================

# Configuración de la página
st.set_page_config(
    page_title="Sistema Predicción Churn + EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Intentar importar utils.model_utils con manejo de errores
try:
    from utils.model_utils import cargar_modelos, hacer_prediccion, cargar_metricas
    IMPORT_UTILS_EXITOSO = True
except ImportError:
    IMPORT_UTILS_EXITOSO = False
    st.sidebar.warning("⚠️ No se pudo importar utils.model_utils")
    st.sidebar.info("""
    **Para usar predicciones:**
    1. Ejecuta localmente: `python entrenar_modelos_reales.py`
    2. Sube la carpeta `models/` a GitHub
    3. O usa el modo solo-EDA por ahora
    """)

# ============================================================================
# VARIABLES Y CONSTANTES
# ============================================================================

TOP_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
    'OnlineSecurity', 'TechSupport', 'InternetService',
    'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

ALL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

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
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 
                     'Electronic check', 'Mailed check']
}

# ============================================================================
# FUNCIONES DE CARGA (MODIFICADAS PARA STREAMLIT CLOUD)
# ============================================================================

def cargar_modelos_streamlit(modo='all_features'):
    """Carga modelos para Streamlit Cloud con manejo de errores"""
    try:
        if not IMPORT_UTILS_EXITOSO:
            return None
        
        modelos_data = cargar_modelos(f'models/{modo}')
        return modelos_data
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando modelos: {str(e)[:100]}...")
        return None

def cargar_datos():
    """Carga datos para EDA"""
    try:
        # Intentar cargar dataset real
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn")
        return df
    except:
        try:
            # Intentar desde data/ folder
            df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            return df
        except:
            # Crear datos demo para EDA
            st.info("📊 Usando datos de demostración para EDA")
            return crear_datos_demo()

def crear_datos_demo():
    """Crea datos de demostración para EDA"""
    np.random.seed(42)
    n_samples = 1000
    
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
    return pd.DataFrame(data)

# ============================================================================
# FUNCIONES EDA (MANTENIDAS)
# ============================================================================

def mostrar_dataframe(df, limite=10):
    """Muestra DataFrame con HTML"""
    html = f"""
    <div style='overflow-x: auto; max-height: 400px; border: 1px solid #ddd; border-radius: 5px;'>
        <table style='width: 100%; border-collapse: collapse;'>
            <thead><tr style='background-color: #f2f2f2; position: sticky; top: 0;'>"""
    
    for col in df.columns[:limite]:
        html += f"<th style='padding: 8px; border: 1px solid #ddd;'>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for i in range(min(len(df), 15)):
        html += "<tr>"
        for col in df.columns[:limite]:
            valor = str(df.iloc[i][col])
            html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{valor}</td>"
        html += "</tr>"
    
    html += f"""</tbody></table>
    <div style='padding: 10px; background-color: #f8f9fa; border-top: 1px solid #ddd;'>
        <small>Mostrando {min(len(df), 15)} de {len(df)} filas</small>
    </div></div>"""
    
    st.markdown(html, unsafe_allow_html=True)

def seccion_eda(df):
    """Sección EDA completa"""
    st.markdown("---")
    st.header("📊 ANÁLISIS EXPLORATORIO DE DATOS")
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Registros", f"{len(df):,}")
    with col2: st.metric("Total Variables", len(df.columns))
    with col3: st.metric("Tasa de Churn", f"{(df['Churn'] == 'Yes').mean()*100:.1f}%")
    with col4: st.metric("Valores Nulos", f"{df.isnull().sum().sum():,}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Vista General", "📈 Distribuciones", "🎯 Insights"])
    
    with tab1:
        st.subheader("Muestra del Dataset")
        mostrar_dataframe(df.head(10))
        
        st.subheader("Información del Dataset")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text_area("Detalles", buffer.getvalue(), height=200)
    
    with tab2:
        st.subheader("Distribución de Variables")
        variable = st.selectbox("Selecciona variable:", [c for c in df.columns if c != 'customerID'])
        
        if variable:
            col_left, col_right = st.columns(2)
            with col_left:
                fig, ax = plt.subplots(figsize=(10, 6))
                if df[variable].dtype in ['int64', 'float64']:
                    df[variable].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f'Distribución de {variable}')
                else:
                    counts = df[variable].value_counts().head(10)
                    ax.bar(range(len(counts)), counts.values, color='lightcoral', edgecolor='black')
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha='right')
                    ax.set_title(f'Distribución de {variable}')
                st.pyplot(fig)
            
            with col_right:
                st.markdown("**📋 Información:**")
                if df[variable].dtype in ['int64', 'float64']:
                    stats = df[variable].describe()
                    for stat, val in stats.items():
                        st.markdown(f"• **{stat}:** {val:.2f}")
                else:
                    st.markdown(f"• **Tipo:** Categórica")
                    st.markdown(f"• **Valores únicos:** {df[variable].nunique()}")
                    st.markdown(f"• **Valor más frecuente:** {df[variable].mode()[0]}")
    
    with tab3:
        st.subheader("🎯 Insights de Churn")
        
        # Churn por contrato
        if 'Contract' in df.columns:
            churn_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            st.success(f"📝 **Contrato más riesgoso:** Mes a mes ({churn_contract.get('Month-to-month', 0):.1f}% churn)")
        
        # Churn por antigüedad
        if 'tenure' in df.columns:
            tenure_churn = df.groupby('Churn')['tenure'].mean()
            st.info(f"⏳ **Antigüedad promedio:** Churn: {tenure_churn.get('Yes', 0):.1f} meses vs No Churn: {tenure_churn.get('No', 0):.1f} meses")
        
        st.subheader("💡 Recomendaciones")
        st.markdown("""
        1. 🎯 **Enfoque en contratos mes a mes** - Implementar programas de fidelización
        2. 🛡️ **Promover OnlineSecurity y TechSupport** - Trials gratuitos
        3. 💳 **Incentivar pagos automáticos** - Descuentos por migración
        4. 📊 **Segmentación proactiva** - Atención especial a clientes nuevos
        """)

# ============================================================================
# SECCIÓN PREDICCIÓN (CON MANEJO DE ERRORES)
# ============================================================================

def seccion_prediccion():
    """Sección de predicción con manejo de modelos faltantes"""
    
    if not IMPORT_UTILS_EXITOSO:
        st.error("""
        ## ⚠️ Funcionalidad de Predicción No Disponible
        
        **Razón:** No se encontró el módulo `utils.model_utils`
        
        **Solución:**
        1. Ejecuta localmente: `python entrenar_modelos_reales.py`
        2. Sube la carpeta `models/` generada a GitHub
        3. Actualiza la app en Streamlit Cloud
        
        **Mientras tanto, puedes usar:**
        - 📊 **EDA Completo** para análisis exploratorio
        - 📈 **Dashboard** con métricas de ejemplo
        """)
        return
    
    st.sidebar.markdown("## ⚙️ CONFIGURACIÓN")
    
    # Intentar cargar modelos
    try:
        modelos_all = cargar_modelos_streamlit('all_features')
        modelos_top = cargar_modelos_streamlit('top_features')
        
        if not modelos_all or not modelos_top:
            st.warning("⚠️ Modelos no encontrados. Usando modo demo para predicción.")
            mostrar_prediccion_demo()
            return
    except:
        st.warning("⚠️ Error cargando modelos. Usando modo demo.")
        mostrar_prediccion_demo()
        return
    
    # Si los modelos se cargaron correctamente
    modelo_seleccionado = st.sidebar.selectbox(
        "Modelo:",
        list(modelos_all['modelos'].keys())
    )
    
    version = st.sidebar.radio(
        "Versión:",
        ["🎯 Top Features", "📊 Todas las Features"]
    )
    
    usar_top = (version == "🎯 Top Features")
    variables = TOP_FEATURES if usar_top else ALL_FEATURES
    modelo_data = modelos_top if usar_top else modelos_all
    
    # Formulario
    st.header("🤖 PREDICCIÓN INDIVIDUAL")
    datos_cliente = crear_formulario(variables)
    
    if st.button("🔮 PREdecIR CHURN", type="primary"):
        modelo = modelo_data['modelos'][modelo_seleccionado]
        preprocesadores = modelo_data['preprocesadores']
        
        try:
            resultado = hacer_prediccion(datos_cliente, modelo, preprocesadores)
            mostrar_resultado(resultado, modelo_seleccionado)
        except Exception as e:
            st.error(f"❌ Error en predicción: {str(e)}")
            st.info("Usando predicción demo como respaldo...")
            mostrar_prediccion_demo()

def crear_formulario(variables):
    """Crea formulario de entrada"""
    datos = {}
    cols = st.columns(2)
    
    for idx, var in enumerate(variables):
        with cols[idx % 2]:
            if var in CATEGORICAL_OPTIONS:
                datos[var] = st.selectbox(var, CATEGORICAL_OPTIONS[var])
            elif var == 'SeniorCitizen':
                datos[var] = st.selectbox(var, [0, 1])
            elif var == 'tenure':
                datos[var] = st.number_input(var, 0, 100, 12)
            elif var in ['MonthlyCharges', 'TotalCharges']:
                datos[var] = st.number_input(f"{var} ($)", 0.0, 10000.0, 50.0 if var == 'MonthlyCharges' else 1000.0)
    
    return datos

def mostrar_resultado(resultado, modelo_nombre):
    """Muestra resultados de predicción"""
    st.markdown("---")
    st.header("🎯 RESULTADOS")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "red" if resultado['prediccion'] == 'CHURN' else "green"
        st.markdown(f"""
        <div style='border: 3px solid {color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>Predicción</h3>
            <h1 style='color: {color};'>{resultado['prediccion']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Prob. Churn", f"{resultado['probabilidad_churn']:.1%}")
    
    with col3:
        st.metric("Prob. No Churn", f"{resultado['probabilidad_no_churn']:.1%}")
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['No Churn', 'Churn']
    valores = [resultado['probabilidad_no_churn'], resultado['probabilidad_churn']]
    colors = ['#4CAF50', '#F44336']
    ax.bar(labels, valores, color=colors, edgecolor='black')
    ax.set_ylabel('Probabilidad')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

def mostrar_prediccion_demo():
    """Muestra predicción demo cuando no hay modelos"""
    st.info("""
    ## 🔮 PREDICCIÓN DEMO
    
    **Esta es una simulación** porque los modelos no están disponibles en Streamlit Cloud.
    
    **Para predicciones reales:**
    1. Clona el repositorio localmente
    2. Ejecuta: `python entrenar_modelos_reales.py`
    3. Los modelos se generarán en la carpeta `models/`
    4. Sube toda la carpeta `models/` a GitHub
    5. Actualiza la app en Streamlit Cloud
    """)
    
    # Simulación interactiva
    st.subheader("Simulación de Predicción")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        contrato = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
        antiguedad = st.slider("Antigüedad (meses)", 0, 72, 12)
    
    with col_sim2:
        seguridad = st.selectbox("Online Security", ["Yes", "No"])
        pago = st.selectbox("Método Pago", ["Electronic check", "Bank transfer", "Credit card"])
    
    # Cálculo simple de riesgo
    riesgo = 0
    if contrato == "Month-to-month": riesgo += 40
    if antiguedad < 6: riesgo += 30
    if seguridad == "No": riesgo += 20
    if pago == "Electronic check": riesgo += 15
    
    riesgo = min(95, riesgo)
    prob_churn = riesgo / 100
    
    # Mostrar resultado simulado
    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        prediccion = "CHURN" if prob_churn > 0.5 else "NO CHURN"
        color = "red" if prediccion == "CHURN" else "green"
        st.markdown(f"**Predicción:** :{color}[{prediccion}]")
    with col_res2:
        st.metric("Probabilidad Churn", f"{prob_churn:.1%}")

# ============================================================================
# SECCIÓN DASHBOARD
# ============================================================================

def seccion_dashboard():
    """Sección dashboard de modelos"""
    st.header("📈 DASHBOARD DE MODELOS")
    
    if not IMPORT_UTILS_EXITOSO:
        st.warning("""
        ## 📊 DASHBOARD DEMO
        
        Mostrando métricas de ejemplo. Para métricas reales:
        1. Entrena los modelos localmente
        2. Sube la carpeta `models/` a GitHub
        """)
        mostrar_dashboard_demo()
        return
    
    try:
        metricas_all = cargar_metricas('models/all_features')
        metricas_top = cargar_metricas('models/top_features')
        mostrar_dashboard_real(metricas_all, metricas_top)
    except:
        st.warning("⚠️ No se pudieron cargar métricas reales. Mostrando demo.")
        mostrar_dashboard_demo()

def mostrar_dashboard_demo():
    """Dashboard demo"""
    st.subheader("⚖️ Métricas de Ejemplo")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Accuracy", "0.85", "+0.02")
    with col2: st.metric("F1-Score", "0.78", "+0.03")
    with col3: st.metric("AUC-ROC", "0.91", "+0.01")
    
    # Gráfico demo
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos = ['Random Forest', 'XGBoost', 'Regresión Logística']
    accuracy = [0.85, 0.87, 0.82]
    f1 = [0.78, 0.80, 0.75]
    
    x = np.arange(len(modelos))
    width = 0.35
    
    ax.bar(x - width/2, accuracy, width, label='Accuracy', color='skyblue')
    ax.bar(x + width/2, f1, width, label='F1-Score', color='lightcoral')
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Modelos (Demo)')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)

def mostrar_dashboard_real(metricas_all, metricas_top):
    """Dashboard con métricas reales"""
    modelo = st.selectbox("Selecciona modelo:", list(metricas_all.keys()))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", f"{metricas_all[modelo]['accuracy']:.3f}")
    with col2: st.metric("F1-Score", f"{metricas_all[modelo]['f1']:.3f}")
    with col3: st.metric("AUC-ROC", f"{metricas_all[modelo]['auc']:.3f}")
    with col4: st.metric("Precision", f"{metricas_all[modelo]['precision']:.3f}")
    
    # Comparación
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall']
    valores_all = [
        metricas_all[modelo]['accuracy'],
        metricas_all[modelo]['f1'],
        metricas_all[modelo]['auc'],
        metricas_all[modelo]['precision'],
        metricas_all[modelo].get('recall', 0.75)
    ]
    valores_top = [
        metricas_top[modelo]['accuracy'],
        metricas_top[modelo]['f1'],
        metricas_top[modelo]['auc'],
        metricas_top[modelo]['precision'],
        metricas_top[modelo].get('recall', 0.75)
    ]
    
    x = np.arange(len(labels))
    ax.bar(x - 0.2, valores_all, 0.4, label='All Features', color='skyblue')
    ax.bar(x + 0.2, valores_top, 0.4, label='Top Features', color='lightcoral')
    
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valor')
    ax.set_title(f'Comparación - {modelo}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    st.title("📱 SISTEMA DE ANÁLISIS Y PREDICCIÓN DE CHURN")
    st.markdown("---")
    
    # Cargar datos para EDA
    df = cargar_datos()
    
    # Navegación
    st.sidebar.markdown("## 🧭 NAVEGACIÓN")
    
    # Opciones de navegación basadas en disponibilidad
    if IMPORT_UTILS_EXITOSO:
        opciones = ["📊 EDA", "🤖 Predicción", "📈 Dashboard"]
    else:
        opciones = ["📊 EDA", "📈 Dashboard"]
        st.sidebar.warning("⚠️ Predicción deshabilitada")
    
    seccion = st.sidebar.radio("Selecciona:", opciones)
    
    # Mostrar sección seleccionada
    if seccion == "📊 EDA":
        seccion_eda(df)
    elif seccion == "🤖 Predicción":
        seccion_prediccion()
    elif seccion == "📈 Dashboard":
        seccion_dashboard()
    
    # Footer
    st.markdown("---")
    st.caption("Sistema de Análisis de Churn v2.0 | " + 
              ("✅ Modelos cargados" if IMPORT_UTILS_EXITOSO else "⚠️ Modo demo activado"))

# ============================================================================
# EJECUCIÓN
# ============================================================================


if __name__ == "__main__":
    main()