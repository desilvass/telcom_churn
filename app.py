import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN BÁSICA - SIMPLIFICADA
# ============================================================================

st.set_page_config(
    page_title="Sistema Predicción Churn",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# VARIABLES DEL SISTEMA
# ============================================================================

VARIABLES_RELEVANTES = [
    'Contract', 'tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService',
    'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

TODAS_VARIABLES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# ============================================================================
# FUNCIONES SIMPLIFICADAS SIN HTML COMPLEJO
# ============================================================================

def mostrar_dataframe_simple(df, limite=10):
    """Muestra DataFrame de forma simple y estable"""
    # Mostrar solo las primeras filas y columnas
    df_display = df.iloc[:15, :limite].copy()
    
    # Usar st.dataframe que es más estable
    st.dataframe(df_display, use_container_width=True, height=300)
    
    # Información simple
    st.caption(f"📊 Mostrando {min(len(df), 15)} de {len(df)} filas | {min(len(df.columns), limite)} de {len(df.columns)} columnas")

def mostrar_estadisticas_simple(df):
    """Muestra estadísticas de forma simple"""
    # Crear estadísticas
    stats = df.describe().T
    stats['missing'] = df.isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    
    # Formatear para mejor visualización
    display_stats = pd.DataFrame({
        'Variable': stats.index,
        'Count': stats['count'].apply(lambda x: f'{x:,.0f}'),
        'Mean': stats['mean'].apply(lambda x: f'{x:.2f}'),
        'Std': stats['std'].apply(lambda x: f'{x:.2f}'),
        'Min': stats['min'].apply(lambda x: f'{x:.2f}'),
        '25%': stats['25%'].apply(lambda x: f'{x:.2f}'),
        '50%': stats['50%'].apply(lambda x: f'{x:.2f}'),
        '75%': stats['75%'].apply(lambda x: f'{x:.2f}'),
        'Max': stats['max'].apply(lambda x: f'{x:.2f}'),
        'Missing': stats['missing'].apply(lambda x: f'{x:,.0f}'),
        'Missing %': stats['missing_pct'].apply(lambda x: f'{x:.1f}%')
    })
    
    # Mostrar en tabla simple
    st.dataframe(display_stats, use_container_width=True, height=400)

# ============================================================================
# FUNCIÓN PARA CARGAR DATOS - MEJORADA
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga los datos del dataset de churn con múltiples intentos"""
    nombres_posibles = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/Telco-Customer-Churn.csv"
    ]
    
    for nombre in nombres_posibles:
        try:
            df = pd.read_csv(nombre)
            st.sidebar.success(f"✅ Dataset cargado: {nombre}")
            return df
        except:
            continue
    
    # Si no encuentra ningún archivo, crear datos demo
    st.sidebar.warning("⚠️ Dataset no encontrado. Usando datos de demostración.")
    return crear_datos_demo()

def crear_datos_demo():
    """Crea datos de demostración realistas"""
    np.random.seed(42)
    n_samples = 2000  # Reducido para mejor rendimiento
    
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
# SECCIÓN EDA - SIMPLIFICADA
# ============================================================================

def seccion_eda(df):
    """Sección completa de Análisis Exploratorio simplificada"""
    
    st.markdown("---")
    st.header("📊 ANÁLISIS EXPLORATORIO DE DATOS")
    
    # Métricas resumen - SIMPLIFICADO
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
    
    # Tabs simplificadas
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Vista General", "📈 Distribuciones", "🎯 Análisis Churn", "🔍 Insights"])
    
    with tab1:
        st.subheader("Muestra del Dataset")
        mostrar_dataframe_simple(df.head(10))
        
        st.subheader("Resumen Estadístico")
        mostrar_estadisticas_simple(df)
        
        # Variables categóricas vs numéricas - SIMPLIFICADO
        st.subheader("Tipos de Variables")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.write("**Variables Numéricas:**")
            for col in numeric_cols[:10]:  # Mostrar solo 10
                st.write(f"• {col}")
            if len(numeric_cols) > 10:
                st.write(f"... y {len(numeric_cols)-10} más")
        
        with col_info2:
            st.write("**Variables Categóricas:**")
            for col in categorical_cols[:10]:  # Mostrar solo 10
                if col != 'customerID':
                    st.write(f"• {col}")
            if len(categorical_cols) > 10:
                st.write(f"... y {len(categorical_cols)-10} más")
    
    with tab2:
        st.subheader("Distribución de Variables")
        
        # Selector de variable
        todas_columnas = [col for col in df.columns if col != 'customerID']
        variable_seleccionada = st.selectbox("Selecciona una variable para analizar:", todas_columnas)
        
        if variable_seleccionada:
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                # Determinar tipo de variable
                if df[variable_seleccionada].dtype in ['int64', 'float64']:
                    # Variable numérica
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[variable_seleccionada].hist(bins=30, ax=ax, edgecolor='black', color='skyblue')
                    ax.set_title(f'Distribución de {variable_seleccionada}')
                    ax.set_xlabel(variable_seleccionada)
                    ax.set_ylabel('Frecuencia')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Estadísticas simplificadas
                    stats = df[variable_seleccionada].describe()
                    st.write("**Estadísticas:**")
                    for stat, value in stats.items():
                        st.write(f"• **{stat}:** {value:.2f}")
                
                else:
                    # Variable categórica
                    fig, ax = plt.subplots(figsize=(10, 6))
                    counts = df[variable_seleccionada].value_counts().head(10)
                    
                    if len(counts) > 10:
                        st.info(f"Mostrando las 10 categorías más frecuentes")
                    
                    bars = ax.bar(range(len(counts)), counts.values, color='lightcoral', edgecolor='black')
                    ax.set_title(f'Distribución de {variable_seleccionada}')
                    ax.set_xlabel(variable_seleccionada)
                    ax.set_ylabel('Frecuencia')
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Añadir etiquetas simplificadas
                    for bar, count in zip(bars, counts.values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count:,}', ha='center', va='bottom', fontsize=9)
                    
                    st.pyplot(fig)
            
            with col_dist2:
                # Información detallada simplificada
                st.write("**Información de la Variable:**")
                
                if df[variable_seleccionada].dtype in ['int64', 'float64']:
                    st.write(f"• **Tipo:** Numérica")
                    st.write(f"• **Valores únicos:** {df[variable_seleccionada].nunique():,}")
                    st.write(f"• **Rango:** {df[variable_seleccionada].min():.2f} - {df[variable_seleccionada].max():.2f}")
                else:
                    st.write(f"• **Tipo:** Categórica")
                    st.write(f"• **Valores únicos:** {df[variable_seleccionada].nunique()}")
                    if not df[variable_seleccionada].mode().empty:
                        st.write(f"• **Valor más frecuente:** {df[variable_seleccionada].mode()[0]}")
                
                nulos_count = df[variable_seleccionada].isnull().sum()
                st.write(f"• **Valores nulos:** {nulos_count:,}")
    
    with tab3:
        st.subheader("Análisis de Churn")
        
        # Distribución global SIMPLIFICADA
        col_churn1, col_churn2 = st.columns(2)
        
        with col_churn1:
            fig, ax = plt.subplots(figsize=(8, 6))
            churn_counts = df['Churn'].value_counts()
            colors = ['#4CAF50', '#F44336']
            
            # Gráfico de torta simple
            wedges, texts, autotexts = ax.pie(churn_counts.values, labels=churn_counts.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Distribución Global de Churn')
            st.pyplot(fig)
        
        with col_churn2:
            # Selector para análisis cruzado
            vars_analisis = [col for col in df.columns if col not in ['customerID', 'Churn']]
            var_cruzada = st.selectbox("Analizar relación con:", vars_analisis, key="cruzada")
            
            if var_cruzada:
                # Crear tabla cruzada simplificada
                if df[var_cruzada].dtype in ['int64', 'float64']:
                    # Para numéricas: boxplot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    data_yes = df[df['Churn'] == 'Yes'][var_cruzada]
                    data_no = df[df['Churn'] == 'No'][var_cruzada]
                    
                    positions = [1, 2]
                    box_data = [data_yes, data_no]
                    
                    bp = ax.boxplot(box_data, positions=positions, widths=0.6, 
                                   patch_artist=True, showmeans=True)
                    
                    # Colores
                    colors_box = ['#F44336', '#4CAF50']
                    for patch, color in zip(bp['boxes'], colors_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_xticklabels(['Churn = Yes', 'Churn = No'])
                    ax.set_ylabel(var_cruzada)
                    ax.set_title(f'Distribución de {var_cruzada} por Churn')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                else:
                    # Para categóricas: gráfico de barras simplificado
                    cross_tab = pd.crosstab(df[var_cruzada], df['Churn'], normalize='index') * 100
                    
                    # Limitar a top 10 categorías
                    if len(cross_tab) > 10:
                        st.info(f"Mostrando las 10 categorías más frecuentes")
                        cross_tab = cross_tab.head(10)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    cross_tab.plot(kind='bar', stacked=True, ax=ax, 
                                  color=['#4CAF50', '#F44336'], edgecolor='black')
                    
                    ax.set_title(f'Tasa de Churn por {var_cruzada}')
                    ax.set_xlabel(var_cruzada)
                    ax.set_ylabel('Porcentaje (%)')
                    ax.legend(title='Churn', loc='upper right')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    
                    st.pyplot(fig)
        
        # Análisis de correlaciones SIMPLIFICADO
        st.subheader("Análisis de Correlaciones")
        
        # Preparar datos para correlación
        df_corr = df.copy()
        
        # Convertir solo variables importantes
        cat_to_num = {
            'gender': {'Male': 0, 'Female': 1},
            'Partner': {'No': 0, 'Yes': 1},
            'Dependents': {'No': 0, 'Yes': 1},
            'PaperlessBilling': {'No': 0, 'Yes': 1},
            'Churn': {'No': 0, 'Yes': 1}
        }
        
        for col, mapping in cat_to_num.items():
            if col in df_corr.columns:
                df_corr[col] = df_corr[col].map(mapping)
        
        # Seleccionar solo columnas numéricas
        numeric_for_corr = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        numeric_for_corr = [col for col in numeric_for_corr if col in df_corr.columns]
        
        if len(numeric_for_corr) > 0:
            # Calcular matriz de correlación
            corr_matrix = df_corr[numeric_for_corr].corr()
            
            # Heatmap de correlación simplificado
            fig, ax = plt.subplots(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Matriz de Correlación entre Variables', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
        else:
            st.warning("No hay suficientes variables numéricas para calcular correlaciones.")
    
    with tab4:
        st.subheader("Insights y Recomendaciones")
        
        # Calcular insights simplificados
        st.write("### 📈 Principales Hallazgos:")
        
        # Insight 1: Tasa de churn
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.info(f"**Tasa de churn global:** {churn_rate:.1f}% de los clientes abandonan el servicio")
        
        # Insight 2: Variables importantes
        if 'Contract' in df.columns:
            contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            max_contract = contract_churn.idxmax()
            max_rate = contract_churn.max()
            st.warning(f"**Contrato más riesgoso:** Clientes con contrato '{max_contract}' tienen {max_rate:.1f}% de tasa de churn")
        
        # Insight 3: Antigüedad
        if 'tenure' in df.columns:
            tenure_churn = df.groupby('Churn')['tenure'].mean()
            st.info(f"**Antigüedad promedio:** Clientes que abandonan tienen {tenure_churn.get('Yes', 0):.1f} meses vs {tenure_churn.get('No', 0):.1f} meses de los que se quedan")
        
        # Recomendaciones simplificadas
        st.write("### 💡 Recomendaciones para Acción:")
        recomendaciones = [
            "🎯 **Enfoque en contratos mes a mes:** Implementar programas de fidelización",
            "🛡️ **Promover servicios de valor:** Trials gratuitos de OnlineSecurity y TechSupport",
            "💳 **Incentivar pagos automáticos:** Descuentos por migración a transferencia bancaria",
            "📊 **Segmentación proactiva:** Atención especial a clientes con menos de 12 meses",
            "📱 **Mejora en experiencia:** Simplificar facturación electrónica"
        ]
        
        for rec in recomendaciones:
            st.write(f"• {rec}")

# ============================================================================
# SECCIÓN PREDICCIÓN - SIMPLIFICADA
# ============================================================================

def predecir_churn_simple(datos_cliente):
    """Predicciones simplificadas y estables"""
    
    # Calcular riesgo basado en reglas simples
    riesgo = 0
    
    # Regla 1: Contrato
    if datos_cliente.get('Contract') == 'Month-to-month':
        riesgo += 40
    elif datos_cliente.get('Contract') == 'One year':
        riesgo += 20
    else:
        riesgo += 10
    
    # Regla 2: Antigüedad
    tenure = datos_cliente.get('tenure', 0)
    if tenure < 6:
        riesgo += 30
    elif tenure < 12:
        riesgo += 20
    elif tenure < 24:
        riesgo += 10
    
    # Regla 3: Facturación electrónica
    if datos_cliente.get('PaperlessBilling') == 'Yes':
        riesgo += 10
    
    # Regla 4: Método de pago
    if 'Electronic' in str(datos_cliente.get('PaymentMethod', '')):
        riesgo += 15
    
    # Convertir a probabilidad
    riesgo = min(95, max(5, riesgo))
    probabilidad = riesgo / 100
    
    return {
        'prediccion': 'CHURN' if probabilidad > 0.5 else 'NO CHURN',
        'probabilidad': probabilidad,
        'riesgo': riesgo,
        'nivel_riesgo': 'ALTO' if riesgo > 60 else 'MEDIO' if riesgo > 40 else 'BAJO'
    }

def mostrar_formulario_prediccion():
    """Muestra formulario simplificado para predicción"""
    st.header("👤 Predicción para Cliente Individual")
    
    # Selección de variables
    modo = st.sidebar.radio("Modo de análisis:", ["Variables Relevantes", "Todas las Variables"], key="modo_pred")
    usar_todas = (modo == "Todas las Variables")
    variables_usar = TODAS_VARIABLES if usar_todas else VARIABLES_RELEVANTES
    
    # Formulario en columnas
    col1, col2 = st.columns(2)
    datos_cliente = {}
    
    with col1:
        if 'Contract' in variables_usar:
            datos_cliente['Contract'] = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
        
        if 'tenure' in variables_usar:
            datos_cliente['tenure'] = st.number_input("Antigüedad (meses)", 0, 100, 12)
        
        if 'MonthlyCharges' in variables_usar:
            datos_cliente['MonthlyCharges'] = st.number_input("Cargos mensuales ($)", 0.0, 200.0, 50.0)
        
        if 'PaymentMethod' in variables_usar:
            datos_cliente['PaymentMethod'] = st.selectbox("Método pago", 
                                                        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    
    with col2:
        if 'PaperlessBilling' in variables_usar:
            datos_cliente['PaperlessBilling'] = st.selectbox("Facturación electrónica", ["Yes", "No"])
        
        if 'InternetService' in variables_usar:
            datos_cliente['InternetService'] = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        
        if 'OnlineSecurity' in variables_usar:
            datos_cliente['OnlineSecurity'] = st.selectbox("Seguridad online", ["Yes", "No"])
        
        if 'TechSupport' in variables_usar:
            datos_cliente['TechSupport'] = st.selectbox("Soporte técnico", ["Yes", "No"])
    
    # Variables adicionales si se usan todas
    if usar_todas:
        st.subheader("Información Adicional")
        col3, col4 = st.columns(2)
        
        with col3:
            if 'gender' in variables_usar:
                datos_cliente['gender'] = st.selectbox("Género", ["Male", "Female"])
            if 'SeniorCitizen' in variables_usar:
                datos_cliente['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
            if 'Partner' in variables_usar:
                datos_cliente['Partner'] = st.selectbox("Partner", ["Yes", "No"])
        
        with col4:
            if 'Dependents' in variables_usar:
                datos_cliente['Dependents'] = st.selectbox("Dependents", ["Yes", "No"])
            if 'TotalCharges' in variables_usar:
                datos_cliente['TotalCharges'] = st.number_input("Cargos totales ($)", 0.0, 10000.0, 1000.0)
    
    return datos_cliente

def mostrar_resultados_prediccion(resultado):
    """Muestra resultados de predicción simplificados"""
    st.markdown("---")
    st.header("🎯 Resultados de Predicción")
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_text = resultado['prediccion']
        pred_color = "red" if pred_text == 'CHURN' else "green"
        st.markdown(f"**Predicción:** :{pred_color}[{pred_text}]")
    
    with col2:
        st.metric("Probabilidad", f"{resultado['probabilidad']:.1%}")
    
    with col3:
        st.metric("Nivel de Riesgo", resultado['nivel_riesgo'])
    
    # Gráfico simple
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['NO CHURN', 'CHURN']
    valores = [1 - resultado['probabilidad'], resultado['probabilidad']]
    colores = ['#4CAF50', '#F44336']
    
    bars = ax.bar(labels, valores, color=colores, edgecolor='black')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Distribución de Probabilidades')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{valor:.1%}', ha='center', va='bottom', fontsize=12)
    
    st.pyplot(fig)
    
    # Estrategias según riesgo
    st.markdown("---")
    st.header("🛡️ Estrategias Recomendadas")
    
    riesgo = resultado['riesgo']
    if riesgo > 60:
        st.warning("**🚨 Acciones Inmediatas:**")
        st.write("1. Contacto telefónico urgente del equipo de retención")
        st.write("2. Oferta especial: 25% descuento por 6 meses")
        st.write("3. Upgrade gratuito de servicios por 3 meses")
    elif riesgo > 40:
        st.info("**📅 Acciones Proactivas:**")
        st.write("1. Email personalizado con oferta especial")
        st.write("2. 15% descuento por 3 meses")
        st.write("3. Programa de fidelización con beneficios")
    else:
        st.success("**💎 Estrategias de Fidelización:**")
        st.write("1. Kit de bienvenida extendido")
        st.write("2. Acceso exclusivo a nuevas funciones")
        st.write("3. Programa de referidos mejorado")
    
    # ROI simplificado
    st.markdown("---")
    st.header("💰 Análisis de Retorno")
    
    presupuesto = st.sidebar.slider("Presupuesto retención ($)", 0, 500, 100, key="presupuesto")
    valor_cliente = 1500
    
    col_roi1, col_roi2, col_roi3 = st.columns(3)
    with col_roi1:
        st.metric("Inversión", f"${presupuesto}")
    with col_roi2:
        st.metric("Valor Cliente", f"${valor_cliente}")
    with col_roi3:
        if presupuesto > 0:
            roi = ((valor_cliente - presupuesto) / presupuesto) * 100
            st.metric("ROI Estimado", f"{roi:.0f}%")
        else:
            st.metric("ROI Estimado", "N/A")
    
    if presupuesto > 0:
        if valor_cliente > presupuesto * 2:
            st.success("**Conclusión:** Inversión altamente rentable")
        elif valor_cliente > presupuesto:
            st.info("**Conclusión:** Inversión moderadamente rentable")
        else:
            st.warning("**Conclusión:** Evaluar estrategia de retención")

# ============================================================================
# FUNCIÓN PRINCIPAL - SIMPLIFICADA
# ============================================================================

def main():
    """Función principal simplificada"""
    
    # Título principal
    st.title("📱 Sistema de Análisis y Predicción de Churn")
    
    # Cargar datos una sola vez
    df = cargar_datos()
    
    # Navegación simplificada
    st.sidebar.markdown("## 🧭 Navegación")
    seccion = st.sidebar.radio(
        "Seleccione sección:",
        ["📊 EDA - Análisis Exploratorio", "🤖 Predicción Individual"]
    )
    
    if seccion == "📊 EDA - Análisis Exploratorio":
        seccion_eda(df)
    
    else:
        # Configuración para predicción
        st.sidebar.markdown("## ⚙️ Configuración")
        
        # Modelos disponibles
        modelos = st.sidebar.multiselect(
            "Selecciona modelos:",
            ["Random Forest", "XGBoost", "Regresión Logística"],
            default=["Random Forest"],
            max_selections=3
        )
        
        # Prioridad
        prioridad = st.sidebar.selectbox("Prioridad", ["Baja", "Media", "Alta"], key="prioridad")
        
        # Mostrar formulario y predicción
        datos_cliente = mostrar_formulario_prediccion()
        
        # Botón de predicción
        if st.button("🔮 Predecir Churn", type="primary", use_container_width=True):
            if len(modelos) > 0:
                resultado = predecir_churn_simple(datos_cliente)
                mostrar_resultados_prediccion(resultado)
            else:
                st.error("⚠️ Por favor selecciona al menos un modelo")
    
    # Footer simplificado
    st.markdown("---")
    st.caption("Sistema de Análisis de Churn v2.0 | Streamlit Cloud")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()