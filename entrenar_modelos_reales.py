#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script para entrenar modelos reales con el dataset de churn de telecomunicaciones.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score, confusion_matrix,
                           classification_report)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os

# Crear directorios si no existen
os.makedirs('models/all_features', exist_ok=True)
os.makedirs('models/top_features', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Variables importantes (basadas en análisis de importancia)
TOP_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
    'OnlineSecurity', 'TechSupport', 'InternetService',
    'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

def cargar_y_preprocesar_datos(ruta='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Carga y preprocesa el dataset de churn.
    """
    print("📂 Cargando datos...")
    
    # Cargar datos
    df = pd.read_csv(ruta)
    
    # Convertir TotalCharges a numérico (manejar valores vacíos)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Eliminar columnas no útiles para modelado
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Variable objetivo
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Manejar valores nulos
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Separar características y target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    print(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"   Distribución de clases: {y.value_counts().to_dict()}")
    print(f"   Tasa de churn: {(y.mean()*100):.1f}%")
    
    return X, y, df.columns.tolist()

def preprocesar_caracteristicas(X, usar_top_features=False):
    """
    Preprocesa las características: encoding y escalado.
    
    Args:
        X: DataFrame con características
        usar_top_features: Si True, usar solo las características más importantes
    
    Returns:
        X_processed: DataFrame procesado
        label_encoders: Diccionario con los encoders
        scaler: Scaler ajustado
    """
    print("🔧 Preprocesando características...")
    
    # Seleccionar características
    if usar_top_features:
        features_seleccionadas = [f for f in TOP_FEATURES if f in X.columns]
        X = X[features_seleccionadas]
        print(f"   Usando {len(features_seleccionadas)} características top")
    else:
        print(f"   Usando todas las {X.shape[1]} características")
    
    # Separar variables numéricas y categóricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"   Numéricas: {len(numeric_cols)}, Categóricas: {len(categorical_cols)}")
    
    # Copiar para no modificar el original
    X_processed = X.copy()
    
    # Encoding de variables categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
    
    # Escalado de variables numéricas
    scaler = StandardScaler()
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    
    return X_processed, label_encoders, scaler, numeric_cols, categorical_cols

def entrenar_modelos(X_train, y_train, X_test, y_test, nombre_config):
    """
    Entrena múltiples modelos y evalúa su rendimiento.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        nombre_config: Nombre de la configuración (para guardar)
    
    Returns:
        modelos: Diccionario con modelos entrenados
        metricas: Diccionario con métricas de evaluación
    """
    print(f"\n🎯 Entrenando modelos con configuración: {nombre_config}")
    
    modelos = {}
    metricas = {}
    
    # 1. Random Forest
    print("   🌲 Entrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    modelos['Random Forest'] = rf
    
    # 2. XGBoost
    print("   ⚡ Entrenando XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Manejar desbalance
    )
    xgb_model.fit(X_train, y_train)
    modelos['XGBoost'] = xgb_model
    
    # 3. Regresión Logística
    print("   📈 Entrenando Regresión Logística...")
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        solver='liblinear',
        class_weight='balanced'
    )
    lr.fit(X_train, y_train)
    modelos['Regresión Logística'] = lr
    
    # Evaluar modelos
    print("\n📊 Evaluando modelos...")
    
    for nombre, modelo in modelos.items():
        # Predicciones
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metricas_modelo = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Validación cruzada para F1
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1')
        metricas_modelo['cv_f1_mean'] = cv_scores.mean()
        metricas_modelo['cv_f1_std'] = cv_scores.std()
        
        metricas[nombre] = metricas_modelo
        
        # Mostrar métricas
        print(f"\n   {nombre}:")
        print(f"   Accuracy:  {metricas_modelo['accuracy']:.4f}")
        print(f"   F1-Score:  {metricas_modelo['f1']:.4f}")
        print(f"   AUC-ROC:   {metricas_modelo['auc']:.4f}")
        print(f"   Precision: {metricas_modelo['precision']:.4f}")
        print(f"   Recall:    {metricas_modelo['recall']:.4f}")
        print(f"   CV F1:     {metricas_modelo['cv_f1_mean']:.4f} (±{metricas_modelo['cv_f1_std']:.4f})")
    
    return modelos, metricas

def guardar_modelos_y_metricas(modelos, metricas, label_encoders, scaler, 
                              numeric_cols, categorical_cols, config):
    """
    Guarda modelos, encoders, scaler y métricas.
    """
    print(f"\n💾 Guardando modelos y métricas para {config}...")
    
    # Guardar cada modelo
    for nombre, modelo in modelos.items():
        nombre_archivo = f"models/{config}/{nombre.lower().replace(' ', '_')}.pkl"
        joblib.dump(modelo, nombre_archivo)
        print(f"   ✅ {nombre} guardado en {nombre_archivo}")
    
    # Guardar preprocesadores
    preprocesadores = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'config': config
    }
    joblib.dump(preprocesadores, f"models/{config}/preprocesadores.pkl")
    
    # Guardar métricas
    joblib.dump(metricas, f"models/{config}/metricas.pkl")
    
    # Guardar reporte en texto
    with open(f'reports/metricas_{config}.txt', 'w') as f:
        f.write(f"=== MÉTRICAS DE MODELOS - {config.upper()} ===\n\n")
        for nombre, metrica in metricas.items():
            f.write(f"{nombre}:\n")
            f.write(f"  Accuracy:  {metrica['accuracy']:.4f}\n")
            f.write(f"  F1-Score:  {metrica['f1']:.4f}\n")
            f.write(f"  AUC-ROC:   {metrica['auc']:.4f}\n")
            f.write(f"  Precision: {metrica['precision']:.4f}\n")
            f.write(f"  Recall:    {metrica['recall']:.4f}\n")
            f.write(f"  CV F1:     {metrica['cv_f1_mean']:.4f} (±{metrica['cv_f1_std']:.4f})\n")
            f.write("-" * 50 + "\n")
    
    print(f"   📄 Reporte guardado en reports/metricas_{config}.txt")

def visualizar_resultados(metricas_all, metricas_top, X_train_all, modelos_all):
    """
    Crea visualizaciones comparativas de los modelos.
    """
    print("\n📈 Generando visualizaciones...")
    
    # 1. Comparación de métricas entre configuraciones
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparación de Métricas: All Features vs Top Features', fontsize=16)
    
    metricas_a_comparar = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    colores = ['skyblue', 'lightcoral']
    
    for idx, metrica in enumerate(metricas_a_comparar):
        ax = axes[idx // 3, idx % 3]
        
        modelos_nombres = list(metricas_all.keys())
        valores_all = [metricas_all[m][metrica] for m in modelos_nombres]
        valores_top = [metricas_top[m][metrica] for m in modelos_nombres]
        
        x = np.arange(len(modelos_nombres))
        width = 0.35
        
        bars_all = ax.bar(x - width/2, valores_all, width, label='All Features', color=colores[0])
        bars_top = ax.bar(x + width/2, valores_top, width, label='Top Features', color=colores[1])
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel(metrica.capitalize())
        ax.set_title(f'{metrica.capitalize()} Score')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_nombres, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/comparacion_metricas.png', dpi=300, bbox_inches='tight')
    print("   ✅ Gráfico de comparación guardado en plots/comparacion_metricas.png")
    
    # 2. Importancia de características para Random Forest (all features)
    if 'Random Forest' in modelos_all:
        rf_model = modelos_all['Random Forest']
        importances = rf_model.feature_importances_
        
        # Obtener nombres de características (necesitas X_train_all.columns)
        if hasattr(X_train_all, 'columns'):
            feature_names = X_train_all.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Ordenar por importancia
        indices = np.argsort(importances)[-15:]  # Top 15
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importancia')
        plt.title('Top 15 Características Más Importantes (Random Forest)')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('plots/importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
        print("   ✅ Gráfico de importancia guardado en plots/importancia_caracteristicas.png")
    
    # 3. Matrices de confusión para todos los modelos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Matrices de Confusión - All Features', fontsize=16)
    
    for idx, (nombre, modelo) in enumerate(modelos_all.items()):
        ax = axes[idx]
        cm = np.array(metricas_all[nombre]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(nombre)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Verdadero')
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_yticklabels(['No Churn', 'Churn'])
    
    plt.tight_layout()
    plt.savefig('plots/matrices_confusion.png', dpi=300, bbox_inches='tight')
    print("   ✅ Matrices de confusión guardadas en plots/matrices_confusion.png")
    
    plt.close('all')

def crear_archivo_config():
    """
    Crea archivo de configuración para la app.
    """
    config = {
        'top_features': TOP_FEATURES,
        'modelos_disponibles': ['Random Forest', 'XGBoost', 'Regresión Logística'],
        'version_modelos': ['all_features', 'top_features'],
        'ultima_actualizacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(config, 'models/config.pkl')
    print("\n✅ Archivo de configuración creado: models/config.pkl")

def main():
    """
    Función principal para entrenar todos los modelos.
    """
    print("=" * 60)
    print("🚀 ENTRENAMIENTO DE MODELOS DE CHURN - TELECOMUNICACIONES")
    print("=" * 60)
    
    # 1. Cargar y preprocesar datos
    X, y, todas_columnas = cargar_y_preprocesar_datos()
    
    # 2. Dividir datos (misma división para ambas configuraciones)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   Prueba: {X_test.shape[0]} muestras")
    
    # 3. Entrenar con TODAS las características
    X_train_all, label_encoders_all, scaler_all, num_all, cat_all = preprocesar_caracteristicas(
        X_train, usar_top_features=False
    )
    X_test_all, _, _, _, _ = preprocesar_caracteristicas(
        X_test, usar_top_features=False
    )
    
    modelos_all, metricas_all = entrenar_modelos(
        X_train_all, y_train, X_test_all, y_test, 'all_features'
    )
    
    guardar_modelos_y_metricas(
        modelos_all, metricas_all, label_encoders_all, scaler_all,
        num_all, cat_all, 'all_features'
    )
    
    # 4. Entrenar con TOP características
    X_train_top, label_encoders_top, scaler_top, num_top, cat_top = preprocesar_caracteristicas(
        X_train, usar_top_features=True
    )
    X_test_top, _, _, _, _ = preprocesar_caracteristicas(
        X_test, usar_top_features=True
    )
    
    modelos_top, metricas_top = entrenar_modelos(
        X_train_top, y_train, X_test_top, y_test, 'top_features'
    )
    
    guardar_modelos_y_metricas(
        modelos_top, metricas_top, label_encoders_top, scaler_top,
        num_top, cat_top, 'top_features'
    )
    
    # 5. Visualizar resultados
    visualizar_resultados(metricas_all, metricas_top, X_train_all, modelos_all)
    
    # 6. Crear archivo de configuración
    crear_archivo_config()
    
    print("\n" + "=" * 60)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 60)
    print("\n📁 Archivos generados:")
    print("   ├── models/all_features/        # Modelos con todas las características")
    print("   ├── models/top_features/        # Modelos con características top")
    print("   ├── plots/                      # Gráficos de análisis")
    print("   └── reports/                    # Reportes de métricas")
    print("\n🔧 Para usar en la app Streamlit:")
    print("   1. Asegúrate de tener la estructura de directorios correcta")
    print("   2. Ejecuta la app con: streamlit run app.py")
    print("   3. Los modelos se cargarán automáticamente")

if __name__ == "__main__":
    main()

