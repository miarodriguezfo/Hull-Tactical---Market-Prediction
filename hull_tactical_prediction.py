#!/usr/bin/env python3
"""
Hull Tactical Market Prediction - Submission Script
Solución AutoML optimizada para la competencia de Kaggle

Autor: OpenHands AI Assistant
Fecha: 2024-11-11
Objetivo: Predecir retornos del S&P 500 optimizando Adjusted Sharpe Ratio
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importar modelos
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel

# Variables globales para el modelo
MODELS = {}
SCALER = None
SELECTED_FEATURES = []
FEATURE_COLS = []

def adjusted_sharpe_ratio(y_true, y_pred, max_weight=6.0):
    """
    Calcula el Adjusted Sharpe Ratio de la competencia Hull Tactical
    
    Args:
        y_true: Retornos reales
        y_pred: Predicciones (posiciones/pesos)
        max_weight: Peso máximo permitido
    
    Returns:
        float: Adjusted Sharpe Ratio
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Aplicar límites
    y_pred = np.clip(y_pred, -max_weight, max_weight)
    
    # Retornos de la estrategia
    strategy_returns = y_true * y_pred
    
    # Métricas
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    if std_return == 0:
        return 0.0
    
    # Sharpe ratio con penalización por volatilidad
    sharpe = mean_return / std_return
    volatility_penalty = 1.0 / (1.0 + std_return)
    
    return sharpe * volatility_penalty

def create_features(df):
    """
    Crea características técnicas optimizadas para trading
    
    Args:
        df: DataFrame con datos de entrada
    
    Returns:
        DataFrame con características adicionales
    """
    df_feat = df.copy()
    
    # Identificar columnas de características
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # 1. Lags importantes (solo los más relevantes para evitar overfitting)
    for col in feature_cols[:8]:  # Top 8 features
        for lag in [1, 2, 3, 5]:
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
    
    # 2. Rolling statistics
    windows = [5, 10, 20]
    for col in feature_cols[:5]:  # Top 5 features
        for window in windows:
            df_feat[f'{col}_ma_{window}'] = df_feat[col].rolling(window).mean()
            df_feat[f'{col}_std_{window}'] = df_feat[col].rolling(window).std()
            df_feat[f'{col}_max_{window}'] = df_feat[col].rolling(window).max()
            df_feat[f'{col}_min_{window}'] = df_feat[col].rolling(window).min()
    
    # 3. Ratios entre características principales
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(5, len(feature_cols))):
            col1, col2 = feature_cols[i], feature_cols[j]
            df_feat[f'{col1}_{col2}_ratio'] = df_feat[col1] / (df_feat[col2] + 1e-8)
            df_feat[f'{col1}_{col2}_diff'] = df_feat[col1] - df_feat[col2]
    
    # 4. Momentum indicators
    for col in feature_cols[:3]:
        # Rate of Change
        df_feat[f'{col}_roc_5'] = (df_feat[col] / df_feat[col].shift(5) - 1) * 100
        df_feat[f'{col}_roc_10'] = (df_feat[col] / df_feat[col].shift(10) - 1) * 100
        
        # RSI aproximado
        delta = df_feat[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df_feat[f'{col}_rsi'] = 100 - (100 / (1 + rs))
    
    # 5. Volatilidad realizada
    for col in feature_cols[:3]:
        returns = df_feat[col].pct_change()
        for window in [10, 20]:
            df_feat[f'{col}_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
    
    # 6. Target lags (históricos para evitar data leakage)
    if 'forward_return_1d' in df_feat.columns:
        target_col = 'forward_return_1d'
        for lag in [2, 3, 5, 10]:  # Empezar desde lag 2
            df_feat[f'target_lag_{lag}'] = df_feat[target_col].shift(lag)
    
    # 7. Características temporales
    df_feat['day_of_year'] = df_feat['date_id'] % 252  # Días de trading por año
    df_feat['week_of_year'] = df_feat['date_id'] % 52
    df_feat['month_of_year'] = df_feat['date_id'] % 12
    
    # Características cíclicas
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 252)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 252)
    df_feat['week_sin'] = np.sin(2 * np.pi * df_feat['week_of_year'] / 52)
    df_feat['week_cos'] = np.cos(2 * np.pi * df_feat['week_of_year'] / 52)
    
    return df_feat

def train_models(X_train, y_train):
    """
    Entrena los modelos del ensemble
    
    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        dict: Diccionario con modelos entrenados
    """
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
        verbose=-1,
        n_jobs=-1
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
        verbosity=0,
        n_jobs=-1
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
        verbose=False,
        thread_count=-1
    )
    cat_model.fit(X_train, y_train)
    models['cat'] = cat_model
    
    return models

def create_ensemble_predictions(models, X, weights=None):
    """
    Crea predicciones del ensemble
    
    Args:
        models: Diccionario de modelos
        X: Características
        weights: Pesos para cada modelo
    
    Returns:
        np.array: Predicciones del ensemble
    """
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    predictions = []
    model_names = list(models.keys())
    
    for i, (name, model) in enumerate(models.items()):
        try:
            pred = model.predict(X) * weights[i]
            predictions.append(pred)
        except Exception as e:
            print(f"Error con modelo {name}: {e}")
            # Fallback: predicciones cero
            predictions.append(np.zeros(len(X)) * weights[i])
    
    return np.sum(predictions, axis=0)

def initialize_model():
    """
    Inicializa y entrena el modelo con datos sintéticos para desarrollo
    Esta función se ejecuta una sola vez al importar el módulo
    """
    global MODELS, SCALER, SELECTED_FEATURES, FEATURE_COLS
    
    print("🚀 Inicializando modelo Hull Tactical...")
    
    try:
        # Intentar cargar datos reales
        train_df = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
        print(f"✅ Datos reales cargados: {train_df.shape}")
    except FileNotFoundError:
        # Crear datos sintéticos para desarrollo/testing
        print("⚠️ Creando datos sintéticos para desarrollo...")
        np.random.seed(42)
        n_samples = 2000
        n_features = 25
        
        data = {'date_id': range(n_samples)}
        
        # Features simulados con patrones realistas
        for i in range(n_features):
            if i < 8:
                # Features principales con tendencia
                trend = np.sin(np.arange(n_samples) * 0.01) * 0.1
                noise = np.random.normal(0, 1, n_samples)
                data[f'feature_{i}'] = trend + noise
            elif i < 15:
                # Features de volatilidad
                data[f'feature_{i}'] = np.random.exponential(1, n_samples)
            else:
                # Features adicionales
                data[f'feature_{i}'] = np.random.normal(0, 1.5, n_samples)
        
        # Target con señal predictiva
        signal = (data['feature_0'] * 0.08 + 
                 data['feature_1'] * 0.05 + 
                 data['feature_2'] * -0.03 +
                 np.random.normal(0, 0.015, n_samples))
        data['forward_return_1d'] = signal
        
        train_df = pd.DataFrame(data)
        print(f"📊 Datos sintéticos creados: {train_df.shape}")
    
    # Identificar columnas de características
    FEATURE_COLS = [col for col in train_df.columns if col.startswith('feature_')]
    
    # Aplicar ingeniería de características
    print("🔧 Aplicando ingeniería de características...")
    train_enhanced = create_features(train_df)
    
    # Eliminar NaN y preparar datos
    train_clean = train_enhanced.dropna()
    print(f"📊 Datos limpios: {train_clean.shape}")
    
    # Preparar características y target
    feature_columns = [col for col in train_clean.columns 
                      if col not in ['date_id', 'forward_return_1d']]
    X = train_clean[feature_columns]
    y = train_clean['forward_return_1d']
    
    # Selección de características con LightGBM
    print("🎯 Seleccionando características...")
    lgb_selector = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_selector.fit(X, y)
    
    # Seleccionar top 40 características
    selector = SelectFromModel(lgb_selector, prefit=True, max_features=40)
    X_selected = selector.transform(X)
    SELECTED_FEATURES = X.columns[selector.get_support()].tolist()
    
    print(f"🎯 Características seleccionadas: {len(SELECTED_FEATURES)}")
    
    # Usar solo características seleccionadas
    X_final = X[SELECTED_FEATURES]
    
    # Split temporal para entrenamiento
    split_point = int(len(X_final) * 0.85)  # Usar más datos para entrenamiento
    X_train = X_final.iloc[:split_point]
    y_train = y.iloc[:split_point]
    
    print(f"📊 Datos de entrenamiento: {X_train.shape}")
    
    # Entrenar modelos
    MODELS = train_models(X_train, y_train)
    
    # Evaluar modelos en datos de validación
    if split_point < len(X_final):
        X_val = X_final.iloc[split_point:]
        y_val = y.iloc[split_point:]
        
        model_scores = {}
        for name, model in MODELS.items():
            pred_val = model.predict(X_val)
            score = adjusted_sharpe_ratio(y_val, pred_val)
            model_scores[name] = score
            print(f"📊 {name} validation score: {score:.6f}")
        
        # Calcular pesos basados en performance
        scores = list(model_scores.values())
        min_score = min(scores)
        adjusted_scores = [max(score - min_score + 0.001, 0.001) for score in scores]
        total_score = sum(adjusted_scores)
        weights = [score / total_score for score in adjusted_scores]
        
        print(f"🏆 Pesos del ensemble: {[f'{w:.3f}' for w in weights]}")
        
        # Guardar pesos como variable global
        global ENSEMBLE_WEIGHTS
        ENSEMBLE_WEIGHTS = weights
    else:
        # Pesos iguales si no hay datos de validación
        ENSEMBLE_WEIGHTS = [1/len(MODELS)] * len(MODELS)
    
    print("✅ Modelo inicializado correctamente")

def predict(test_df):
    """
    Función de predicción principal para la API de Kaggle
    
    Args:
        test_df: DataFrame con datos de test
    
    Returns:
        np.array: Predicciones (posiciones/pesos)
    """
    try:
        # Verificar que el modelo esté inicializado
        if not MODELS or not SELECTED_FEATURES:
            print("⚠️ Modelo no inicializado, inicializando...")
            initialize_model()
        
        print(f"🔮 Prediciendo para {len(test_df)} muestras...")
        
        # Aplicar ingeniería de características
        test_enhanced = create_features(test_df)
        
        # Seleccionar características
        available_features = [f for f in SELECTED_FEATURES if f in test_enhanced.columns]
        missing_features = [f for f in SELECTED_FEATURES if f not in test_enhanced.columns]
        
        if missing_features:
            print(f"⚠️ Características faltantes: {len(missing_features)}")
            # Crear características faltantes con valores por defecto
            for feature in missing_features:
                test_enhanced[feature] = 0.0
        
        test_features = test_enhanced[SELECTED_FEATURES]
        
        # Manejar valores faltantes
        # Primero, forward fill para series temporales
        test_features = test_features.fillna(method='ffill')
        # Luego, backward fill para los primeros valores
        test_features = test_features.fillna(method='bfill')
        # Finalmente, rellenar con cero cualquier valor restante
        test_features = test_features.fillna(0.0)
        
        # Verificar que no hay NaN
        if test_features.isnull().any().any():
            print("⚠️ Aún hay valores NaN, rellenando con cero...")
            test_features = test_features.fillna(0.0)
        
        # Predicciones del ensemble
        predictions = create_ensemble_predictions(MODELS, test_features, ENSEMBLE_WEIGHTS)
        
        # Aplicar límites de posición
        predictions = np.clip(predictions, -6.0, 6.0)
        
        # Aplicar suavizado para reducir volatilidad
        predictions = predictions * 0.8  # Factor de conservadurismo
        
        print(f"✅ Predicciones generadas: rango [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        print("🔄 Usando predicciones de fallback...")
        
        # Fallback: predicciones conservadoras basadas en momentum simple
        try:
            if len(FEATURE_COLS) > 0 and FEATURE_COLS[0] in test_df.columns:
                # Usar momentum del primer feature como señal básica
                momentum = test_df[FEATURE_COLS[0]].pct_change(5).fillna(0)
                predictions = np.clip(momentum * 0.5, -1.0, 1.0)  # Muy conservador
            else:
                predictions = np.zeros(len(test_df))  # Predicciones neutras
        except:
            predictions = np.zeros(len(test_df))
        
        return predictions

# Inicializar modelo al importar
if __name__ == "__main__" or 'kaggle' in str(globals()):
    initialize_model()

# Para ejecución en Kaggle
if 'kaggle' in str(globals()):
    try:
        import kaggle_evaluation.hull_tactical_market_prediction as evaluation
        print("🚀 Ejecutando evaluación de Kaggle...")
        evaluation.run(predict)
    except ImportError:
        print("⚠️ Módulo de evaluación de Kaggle no encontrado")
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")

print("🎯 Hull Tactical Market Prediction - Modelo listo para predicciones")