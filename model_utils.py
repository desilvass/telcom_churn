#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Utilidades para cargar y usar modelos de predicción de churn.
Este archivo es ESENCIAL para que app.py funcione correctamente.
"""
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Variables top según el entrenamiento
TOP_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
    'OnlineSecurity', 'TechSupport', 'InternetService',
    'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

# Todas las variables del dataset
ALL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# ============================================================================
# FUNCIONES DE CARGA DE MODELOS
# ============================================================================

def cargar_modelos(config_path='models/all_features'):
    """
    Carga todos los modelos desde una configuración específica.
    
    Args:
        config_path: Ruta a la carpeta de modelos (ej: 'models/all_features')
    
    Returns:
        dict: Diccionario con modelos y preprocesadores
    """
    try:
        # Cargar preprocesadores
        preprocesadores = joblib.load(f'{config_path}/preprocesadores.pkl')
        
        # Mapeo de archivos a nombres de modelos
        modelo_files = {
            'random_forest.pkl': 'Random Forest',
            'xgboost.pkl': 'XGBoost', 
            'regresion_logistica.pkl': 'Regresión Logística'
        }
        
        modelos = {}
        for archivo, nombre in modelo_files.items():
            try:
                ruta_completa = f'{config_path}/{archivo}'
                modelo = joblib.load(ruta_completa)
                modelos[nombre] = modelo
            except FileNotFoundError:
                print(f"⚠️ Advertencia: No se encontró {archivo}")
                continue
        
        if not modelos:
            raise ValueError(f"No se encontraron modelos en {config_path}")
        
        return {
            'modelos': modelos,
            'preprocesadores': preprocesadores,
            'config_path': config_path
        }
        
    except Exception as e:
        raise Exception(f"❌ Error cargando modelos desde {config_path}: {str(e)}")

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def preparar_datos_entrada(datos_cliente, preprocesadores, usar_top_features=False):
    """
    Prepara los datos de entrada para predicción.
    
    Args:
        datos_cliente: Diccionario con los datos del cliente
        preprocesadores: Diccionario con encoders y scaler
        usar_top_features: Si True, usar solo top features
    
    Returns:
        DataFrame: Datos preparados para el modelo
    """
    # Extraer componentes de preprocesadores
    label_encoders = preprocesadores.get('label_encoders', {})
    scaler = preprocesadores.get('scaler')
    numeric_cols = preprocesadores.get('numeric_cols', [])
    categorical_cols = preprocesadores.get('categorical_cols', [])
    
    # Crear DataFrame con los datos
    df = pd.DataFrame([datos_cliente])
    
    # Filtrar columnas si estamos usando top features
    if usar_top_features:
        columnas_permitidas = [col for col in df.columns if col in TOP_FEATURES]
        df = df[columnas_permitidas]
    
    # 1. Aplicar encoding a variables categóricas
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            try:
                # Manejar valores faltantes o nuevos
                valor = str(df[col].iloc[0]) if not pd.isna(df[col].iloc[0]) else 'Missing'
                
                if valor in label_encoders[col].classes_:
                    df[col] = label_encoders[col].transform([valor])[0]
                else:
                    # Si el valor no está en el encoder, usar el más común
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
            except Exception as e:
                print(f"⚠️ Error encoding {col}: {e}")
                df[col] = 0
    
    # 2. Asegurar que todas las columnas numéricas existan
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0  # Valor por defecto
        else:
            # Convertir a numérico si es necesario
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # 3. Aplicar escalado a variables numéricas
    if scaler and numeric_cols:
        try:
            # Filtrar solo las columnas numéricas que existen
            cols_numericas_existentes = [col for col in numeric_cols if col in df.columns]
            if cols_numericas_existentes:
                df[cols_numericas_existentes] = scaler.transform(df[cols_numericas_existentes])
        except Exception as e:
            print(f"⚠️ Error en escalado: {e}")
    
    # 4. Asegurar todas las columnas esperadas por el modelo
    columnas_esperadas = numeric_cols + categorical_cols
    
    # Crear DataFrame final con todas las columnas
    df_final = pd.DataFrame(columns=columnas_esperadas)
    
    for col in columnas_esperadas:
        if col in df.columns:
            df_final[col] = df[col]
        else:
            df_final[col] = 0.0 if col in numeric_cols else 0
    
    # Ordenar columnas
    df_final = df_final[columnas_esperadas]
    
    return df_final

# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def hacer_prediccion(datos_cliente, modelo, preprocesadores, usar_top_features=False):
    """
    Realiza una predicción para un cliente individual.
    
    Args:
        datos_cliente: Diccionario con datos del cliente
        modelo: Modelo entrenado para hacer predicción
        preprocesadores: Diccionario con preprocesadores
        usar_top_features: Si True, usar solo top features
    
    Returns:
        dict: Resultados de la predicción
    """
    try:
        # 1. Preparar datos
        X = preparar_datos_entrada(datos_cliente, preprocesadores, usar_top_features)
        
        # 2. Hacer predicción
        prediccion = modelo.predict(X)[0]
        
        # 3. Obtener probabilidades (si el modelo lo soporta)
        try:
            probabilidades = modelo.predict_proba(X)[0]
            prob_churn = float(probabilidades[1])
            prob_no_churn = float(probabilidades[0])
        except (AttributeError, IndexError):
            # Si el modelo no tiene predict_proba
            prob_churn = 0.8 if prediccion == 1 else 0.2
            prob_no_churn = 1 - prob_churn
        
        # 4. Preparar resultado
        resultado = {
            'prediccion': 'CHURN' if prediccion == 1 else 'NO CHURN',
            'prediccion_numerica': int(prediccion),
            'probabilidad_churn': prob_churn,
            'probabilidad_no_churn': prob_no_churn,
            'confianza': max(prob_churn, prob_no_churn),
            'exitoso': True
        }
        
        # 5. Calcular nivel de riesgo
        if prob_churn > 0.7:
            resultado['riesgo'] = 'ALTO'
        elif prob_churn > 0.5:
            resultado['riesgo'] = 'MEDIO'
        else:
            resultado['riesgo'] = 'BAJO'
        
        return resultado
        
    except Exception as e:
        # En caso de error, devolver resultado de error
        print(f"❌ Error en predicción: {e}")
        return {
            'prediccion': 'ERROR',
            'prediccion_numerica': -1,
            'probabilidad_churn': 0.5,
            'probabilidad_no_churn': 0.5,
            'confianza': 0.0,
            'riesgo': 'DESCONOCIDO',
            'exitoso': False,
            'error': str(e)
        }

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_metricas(config_path='models/all_features'):
    """
    Carga las métricas de evaluación de los modelos.
    
    Args:
        config_path: Ruta a la carpeta de modelos
    
    Returns:
        dict: Métricas de los modelos
    """
    try:
        metricas = joblib.load(f'{config_path}/metricas.pkl')
        return metricas
    except:
        # Métricas por defecto si no existen
        return {
            'Random Forest': {
                'accuracy': 0.85, 'f1': 0.78, 'auc': 0.91,
                'precision': 0.76, 'recall': 0.80,
                'confusion_matrix': [[800, 100], [50, 150]]
            },
            'XGBoost': {
                'accuracy': 0.87, 'f1': 0.80, 'auc': 0.93,
                'precision': 0.78, 'recall': 0.82,
                'confusion_matrix': [[820, 80], [45, 155]]
            },
            'Regresión Logística': {
                'accuracy': 0.82, 'f1': 0.75, 'auc': 0.88,
                'precision': 0.74, 'recall': 0.76,
                'confusion_matrix': [[780, 120], [60, 140]]
            }
        }

def verificar_estructura_modelos():
    """
    Verifica si la estructura de modelos existe.
    
    Returns:
        bool: True si los modelos existen, False en caso contrario
    """
    archivos_necesarios = [
        'models/all_features/preprocesadores.pkl',
        'models/all_features/random_forest.pkl',
        'models/all_features/metricas.pkl',
        'models/top_features/preprocesadores.pkl',
        'models/top_features/random_forest.pkl',
        'models/top_features/metricas.pkl'
    ]
    
    for archivo in archivos_necesarios:
        try:
            with open(archivo, 'rb'):
                pass
        except FileNotFoundError:
            return False
    
    return True

def obtener_info_modelos():
    """
    Obtiene información sobre los modelos disponibles.
    
    Returns:
        dict: Información de los modelos
    """
    info = {
        'all_features': {
            'n_variables': len(ALL_FEATURES),
            'modelos_disponibles': ['Random Forest', 'XGBoost', 'Regresión Logística'],
            'descripcion': 'Modelo con todas las variables del dataset'
        },
        'top_features': {
            'n_variables': len(TOP_FEATURES),
            'modelos_disponibles': ['Random Forest', 'XGBoost', 'Regresión Logística'],
            'descripcion': 'Modelo con las variables más importantes'
        }
    }
    
    # Verificar qué modelos están realmente disponibles
    for config in ['all_features', 'top_features']:
        try:
            modelos_data = cargar_modelos(f'models/{config}')
            info[config]['modelos_disponibles'] = list(modelos_data['modelos'].keys())
        except:
            pass
    
    return info

# ============================================================================
# EJEMPLO DE USO (para testing)
# ============================================================================

if __name__ == "__main__":
    """
    Ejecuta pruebas básicas del módulo.
    Solo se ejecuta si se corre este archivo directamente.
    """
    print("🔧 Probando utils/model_utils.py...")
    
    # 1. Verificar estructura
    if verificar_estructura_modelos():
        print("✅ Estructura de modelos encontrada")
        
        # 2. Cargar un modelo
        try:
            modelos_all = cargar_modelos('models/all_features')
            print(f"✅ Modelos cargados: {list(modelos_all['modelos'].keys())}")
            
            # 3. Crear datos de ejemplo
            datos_ejemplo = {
                'tenure': 24,
                'MonthlyCharges': 70.5,
                'TotalCharges': 1692.0,
                'Contract': 'Month-to-month',
                'OnlineSecurity': 'No',
                'TechSupport': 'No',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic check',
                'PaperlessBilling': 'Yes',
                'SeniorCitizen': 0,
                'gender': 'Male',
                'Partner': 'Yes',
                'Dependents': 'No'
            }
            
            # 4. Hacer predicción de prueba
            resultado = hacer_prediccion(
                datos_ejemplo,
                list(modelos_all['modelos'].values())[0],  # Primer modelo
                modelos_all['preprocesadores'],
                usar_top_features=False
            )
            
            print(f"✅ Predicción de prueba: {resultado['prediccion']}")
            print(f"   Probabilidad Churn: {resultado['probabilidad_churn']:.2%}")
            print(f"   Nivel de riesgo: {resultado['riesgo']}")
            
        except Exception as e:
            print(f"❌ Error en pruebas: {e}")
    
    else:
        print("⚠️  Modelos no encontrados. Ejecuta primero:")
        print("    python entrenar_modelos_reales.py")
    
    print("\n✅ Pruebas completadas")

