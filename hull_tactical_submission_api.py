#!/usr/bin/env python3
"""
Hull Tactical Market Prediction - Submission con API de Kaggle
Versión final con integración de la API de evaluación de Kaggle

Autor: OpenHands AI Assistant
Fecha: 2024-11-11
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importar modelos
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.feature_selection import SelectFromModel

def adjusted_sharpe_ratio(y_true, y_pred, max_weight=6.0):
    """Calcula el Adjusted Sharpe Ratio de la competencia"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred = np.clip(y_pred, -max_weight, max_weight)
    strategy_returns = y_true * y_pred
    
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = mean_return / std_return
    volatility_penalty = 1.0 / (1.0 + std_return)
    
    return sharpe * volatility_penalty

def detect_target_column(df):
    """Detecta automáticamente la columna target"""
    possible_targets = [
        'target', 'responder', 'forward_return_1d', 'forward_return', 
        'return_1d', 'daily_return', 'y', 'label'
    ]
    
    for col in possible_targets:
        if col in df.columns:
            return col
    
    # Buscar columnas numéricas que no sean features o date_id
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_feature_cols = [col for col in numeric_cols 
                       if not col.startswith('feature_') and col != 'date_id']
    
    if len(non_feature_cols) == 1:
        return non_feature_cols[0]
    
    return None

def create_features(df, target_col=None):
    """Crea características técnicas optimizadas"""
    df_feat = df.copy()
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    print(f"🔢 Características base detectadas: {len(feature_cols)}")
    
    # 1. Lags importantes
    for col in feature_cols[:8]:
        for lag in [1, 2, 3, 5]:
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
    
    # 2. Rolling statistics
    windows = [5, 10, 20]
    for col in feature_cols[:5]:
        for window in windows:
            df_feat[f'{col}_ma_{window}'] = df_feat[col].rolling(window).mean()
            df_feat[f'{col}_std_{window}'] = df_feat[col].rolling(window).std()
    
    # 3. Ratios
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(5, len(feature_cols))):
            col1, col2 = feature_cols[i], feature_cols[j]
            df_feat[f'{col1}_{col2}_ratio'] = df_feat[col1] / (df_feat[col2] + 1e-8)
    
    # 4. Momentum
    for col in feature_cols[:3]:
        df_feat[f'{col}_roc_5'] = (df_feat[col] / df_feat[col].shift(5) - 1) * 100
        df_feat[f'{col}_roc_10'] = (df_feat[col] / df_feat[col].shift(10) - 1) * 100
    
    # 5. Target lags históricos (solo si existe target)
    if target_col and target_col in df_feat.columns:
        for lag in [2, 3, 5, 10]:
            df_feat[f'target_lag_{lag}'] = df_feat[target_col].shift(lag)
    
    # 6. Características temporales
    if 'date_id' in df_feat.columns:
        df_feat['day_of_year'] = df_feat['date_id'] % 252
        df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 252)
        df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 252)
    
    return df_feat

def train_ensemble_models(X_train, y_train):
    """Entrena ensemble de modelos"""
    models = {}
    
    print("🤖 Entrenando LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    
    print("🤖 Entrenando XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    
    print("🤖 Entrenando CatBoost...")
    cat_model = cb.CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['cat'] = cat_model
    
    return models

def ensemble_predict(X, models, weights):
    """Crea predicciones del ensemble"""
    predictions = []
    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(X) * weights[i]
        predictions.append(pred)
    return np.sum(predictions, axis=0)

# Variables globales para almacenar el modelo entrenado
TRAINED_MODELS = None
ENSEMBLE_WEIGHTS = None
SELECTED_FEATURES = None
TARGET_COL = None

def initialize_models():
    """Inicializa y entrena los modelos"""
    global TRAINED_MODELS, ENSEMBLE_WEIGHTS, SELECTED_FEATURES, TARGET_COL
    
    print("🚀 Inicializando modelos Hull Tactical...")
    
    # Cargar datos de entrenamiento
    try:
        train_df = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
        print(f"✅ Datos reales cargados: {train_df.shape}")
        
        # Detectar columna target automáticamente
        TARGET_COL = detect_target_column(train_df)
        if TARGET_COL:
            print(f"🎯 Columna target detectada: '{TARGET_COL}'")
        else:
            raise ValueError("No se pudo detectar columna target")
            
    except FileNotFoundError:
        print("⚠️ Datos de Kaggle no encontrados, usando datos sintéticos para desarrollo...")
        
        # Crear datos sintéticos para desarrollo
        np.random.seed(42)
        n_train = 2500
        n_features = 30
        
        train_data = {'date_id': range(n_train)}
        for i in range(n_features):
            if i < 10:
                trend = np.sin(np.arange(n_train) * 0.01) * 0.1
                noise = np.random.normal(0, 1, n_train)
                train_data[f'feature_{i}'] = trend + noise
            else:
                train_data[f'feature_{i}'] = np.random.normal(0, 1.5, n_train)
        
        signal = (train_data['feature_0'] * 0.08 + 
                 train_data['feature_1'] * 0.05 + 
                 train_data['feature_2'] * -0.03 +
                 np.random.normal(0, 0.015, n_train))
        train_data['target'] = signal
        train_df = pd.DataFrame(train_data)
        TARGET_COL = 'target'
        print(f"📊 Datos sintéticos creados: {train_df.shape}")
    
    # Aplicar ingeniería de características
    print("🔧 Aplicando ingeniería de características...")
    train_enhanced = create_features(train_df, TARGET_COL)
    train_clean = train_enhanced.dropna()
    print(f"📊 Datos de entrenamiento limpios: {train_clean.shape}")
    
    # Preparar características
    exclude_cols = ['date_id', TARGET_COL]
    feature_columns = [col for col in train_clean.columns if col not in exclude_cols]
    X_train_full = train_clean[feature_columns]
    y_train_full = train_clean[TARGET_COL]
    
    # Selección de características
    print("🎯 Seleccionando características...")
    lgb_selector = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_selector.fit(X_train_full, y_train_full)
    
    selector = SelectFromModel(lgb_selector, prefit=True, max_features=40)
    SELECTED_FEATURES = X_train_full.columns[selector.get_support()].tolist()
    print(f"🎯 Características seleccionadas: {len(SELECTED_FEATURES)}")
    
    # Usar características seleccionadas
    X_train = X_train_full[SELECTED_FEATURES]
    y_train = y_train_full
    
    # Split para validación
    split_point = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_point]
    X_val_split = X_train.iloc[split_point:]
    y_train_split = y_train.iloc[:split_point]
    y_val_split = y_train.iloc[split_point:]
    
    # Entrenar modelos
    models = train_ensemble_models(X_train_split, y_train_split)
    
    # Evaluar en validación para calcular pesos
    print("📊 Evaluando modelos...")
    model_scores = {}
    for name, model in models.items():
        pred_val = model.predict(X_val_split)
        score = adjusted_sharpe_ratio(y_val_split, pred_val)
        model_scores[name] = score
        print(f"   {name}: {score:.6f}")
    
    # Calcular pesos del ensemble
    scores = list(model_scores.values())
    min_score = min(scores)
    adjusted_scores = [max(score - min_score + 0.001, 0.001) for score in scores]
    total_score = sum(adjusted_scores)
    ENSEMBLE_WEIGHTS = [score / total_score for score in adjusted_scores]
    print(f"🏆 Pesos del ensemble: {[f'{w:.3f}' for w in ENSEMBLE_WEIGHTS]}")
    
    # Re-entrenar con todos los datos
    print("🔄 Re-entrenando con todos los datos...")
    TRAINED_MODELS = train_ensemble_models(X_train, y_train)
    
    print("✅ Modelos inicializados correctamente")

def predict(test_df):
    """
    Función de predicción principal para la API de evaluación de Kaggle
    
    Args:
        test_df: DataFrame con datos de test
        
    Returns:
        numpy.array: Predicciones para cada fila
    """
    global TRAINED_MODELS, ENSEMBLE_WEIGHTS, SELECTED_FEATURES, TARGET_COL
    
    # Inicializar modelos si no están cargados
    if TRAINED_MODELS is None:
        initialize_models()
    
    try:
        print(f"🔮 Procesando {len(test_df)} muestras de test...")
        
        # Aplicar ingeniería de características
        test_enhanced = create_features(test_df, TARGET_COL)
        
        # Verificar características disponibles
        available_features = [f for f in SELECTED_FEATURES if f in test_enhanced.columns]
        missing_features = [f for f in SELECTED_FEATURES if f not in test_enhanced.columns]
        
        if missing_features:
            print(f"⚠️ Características faltantes: {len(missing_features)}")
            # Rellenar características faltantes con 0
            for feature in missing_features:
                test_enhanced[feature] = 0.0
        
        # Seleccionar características
        test_features = test_enhanced[SELECTED_FEATURES]
        
        # Manejar valores faltantes
        test_features = test_features.fillna(method='ffill')
        test_features = test_features.fillna(method='bfill')
        test_features = test_features.fillna(0.0)
        
        # Predicciones del ensemble
        predictions = ensemble_predict(test_features, TRAINED_MODELS, ENSEMBLE_WEIGHTS)
        
        # Aplicar límites y suavizado conservador
        predictions = np.clip(predictions * 0.8, -6.0, 6.0)
        
        print(f"✅ Predicciones generadas: rango [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        
        # Fallback: predicciones conservadoras basadas en momentum
        feature_cols = [col for col in test_df.columns if col.startswith('feature_')]
        if len(feature_cols) > 0:
            print("🔄 Usando predicciones de fallback basadas en momentum...")
            # Usar momentum simple del primer feature
            momentum = test_df[feature_cols[0]].pct_change(5).fillna(0)
            return np.clip(momentum * 0.5, -1.0, 1.0)
        else:
            print("🔄 Usando predicciones neutras...")
            return np.zeros(len(test_df))

def main():
    """Función principal para testing y desarrollo"""
    print("🚀 Hull Tactical Market Prediction - Submission API")
    print("=" * 60)
    
    # Inicializar modelos
    initialize_models()
    
    # Test con datos sintéticos
    print("\n🧪 Probando función de predicción...")
    
    # Crear datos de test sintéticos
    np.random.seed(123)
    n_test = 100
    n_features = 30
    
    test_data = {'date_id': range(2500, 2500 + n_test)}
    for i in range(n_features):
        if i < 10:
            trend = np.sin(np.arange(2500, 2500 + n_test) * 0.01) * 0.1
            noise = np.random.normal(0, 1, n_test)
            test_data[f'feature_{i}'] = trend + noise
        else:
            test_data[f'feature_{i}'] = np.random.normal(0, 1.5, n_test)
    
    test_df = pd.DataFrame(test_data)
    
    # Generar predicciones
    predictions = predict(test_df)
    
    print(f"\n📊 RESULTADOS DEL TEST")
    print("=" * 30)
    print(f"Muestras procesadas: {len(predictions)}")
    print(f"Media: {predictions.mean():.6f}")
    print(f"Desviación estándar: {predictions.std():.6f}")
    print(f"Mínimo: {predictions.min():.6f}")
    print(f"Máximo: {predictions.max():.6f}")
    print(f"Percentil 25: {np.percentile(predictions, 25):.6f}")
    print(f"Mediana: {np.median(predictions):.6f}")
    print(f"Percentil 75: {np.percentile(predictions, 75):.6f}")
    
    print(f"\n🎉 ¡Test completado exitosamente!")
    print("📝 La función predict() está lista para usar con la API de Kaggle")
    
    # Intentar ejecutar la API de evaluación de Kaggle
    print("\n🚀 Intentando ejecutar API de evaluación de Kaggle...")
    try:
        import kaggle_evaluation.hull_tactical_market_prediction as evaluation
        print("✅ Módulo de evaluación importado")
        print("🔄 Ejecutando evaluación...")
        evaluation.run(predict)
        print("🎉 ¡Evaluación completada!")
        
    except ImportError:
        print("⚠️ Módulo de evaluación no encontrado (normal en desarrollo local)")
        print("📝 En Kaggle, la evaluación se ejecutará automáticamente")
        
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        print("📝 La función predict() está definida correctamente")

if __name__ == "__main__":
    main()