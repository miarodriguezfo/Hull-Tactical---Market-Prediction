#!/usr/bin/env python3
"""
🎯 Sistema de Optimización Avanzado - Hull Tactical Market Prediction
Optimización bayesiana, ensemble dinámico y gestión de riesgo

Autor: OpenHands AI Assistant
Fecha: 2024-11-12
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

# Ensemble methods
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

@dataclass
class OptimizationConfig:
    """Configuración para optimización avanzada"""
    
    # Optuna settings
    n_trials: int = 500
    timeout: int = 7200  # 2 horas
    n_jobs: int = -1
    
    # Ensemble settings
    ensemble_size: int = 15
    stacking_cv: int = 5
    blending_holdout: float = 0.2
    
    # Risk management
    max_position: float = 6.0
    min_position: float = -6.0
    volatility_target: float = 0.15
    max_drawdown: float = 0.08
    
    # Validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Multi-objective
    objectives: List[str] = None
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['hull_score', 'sharpe_ratio', 'max_drawdown']

class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble avanzado con pesos dinámicos"""
    
    def __init__(self, models: List[Tuple[str, object]], 
                 weighting_method: str = 'performance',
                 meta_learner: Optional[object] = None):
        self.models = models
        self.weighting_method = weighting_method
        self.meta_learner = meta_learner
        self.weights_ = None
        self.performance_scores_ = None
        
    def fit(self, X, y):
        """Entrenar ensemble con pesos dinámicos"""
        
        # Entrenar modelos base
        self.fitted_models_ = []
        predictions = []
        
        # Time series split para validación interna
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in self.models:
            try:
                # Entrenar en todo el dataset
                model.fit(X, y)
                self.fitted_models_.append((name, model))
                
                # Validación cruzada para calcular pesos
                cv_preds = []
                cv_true = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train_cv = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                    y_train_cv = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                    X_val_cv = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                    y_val_cv = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                    
                    # Crear copia del modelo para CV
                    model_cv = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    model_cv.fit(X_train_cv, y_train_cv)
                    pred_cv = model_cv.predict(X_val_cv)
                    
                    cv_preds.extend(pred_cv)
                    cv_true.extend(y_val_cv)
                
                predictions.append(cv_preds)
                
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could be trained")
        
        # Calcular pesos basados en performance
        self.weights_ = self._calculate_weights(np.array(predictions).T, np.array(cv_true))
        
        # Entrenar meta-learner si se especifica
        if self.meta_learner is not None:
            meta_features = np.array(predictions).T
            self.meta_learner.fit(meta_features, cv_true)
        
        return self
    
    def _calculate_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calcular pesos basados en performance"""
        
        if self.weighting_method == 'equal':
            return np.ones(predictions.shape[1]) / predictions.shape[1]
        
        elif self.weighting_method == 'performance':
            # Pesos basados en Hull metric
            scores = []
            for i in range(predictions.shape[1]):
                score = self._hull_metric(y_true, predictions[:, i])
                scores.append(max(score, 0.001))  # Evitar pesos negativos
            
            scores = np.array(scores)
            weights = scores / scores.sum()
            return weights
        
        elif self.weighting_method == 'inverse_error':
            # Pesos basados en error inverso
            errors = []
            for i in range(predictions.shape[1]):
                error = mean_squared_error(y_true, predictions[:, i])
                errors.append(error + 1e-8)  # Evitar división por cero
            
            errors = np.array(errors)
            weights = (1 / errors) / (1 / errors).sum()
            return weights
        
        else:
            return np.ones(predictions.shape[1]) / predictions.shape[1]
    
    def _hull_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Métrica Hull simplificada para cálculo de pesos"""
        try:
            y_pred_clipped = np.clip(y_pred, -6.0, 6.0)
            strategy_returns = y_true * y_pred_clipped
            
            if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                return 0.0
            
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            return float(sharpe)
        except:
            return 0.0
    
    def predict(self, X):
        """Hacer predicciones con ensemble"""
        
        if not hasattr(self, 'fitted_models_'):
            raise ValueError("Ensemble not fitted")
        
        predictions = []
        
        for name, model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Prediction failed for {name}: {e}")
                continue
        
        if not predictions:
            return np.zeros(len(X))
        
        predictions = np.array(predictions).T
        
        # Usar meta-learner si está disponible
        if self.meta_learner is not None and hasattr(self.meta_learner, 'predict'):
            try:
                return self.meta_learner.predict(predictions)
            except:
                pass
        
        # Promedio ponderado
        if self.weights_ is not None and len(self.weights_) == predictions.shape[1]:
            return np.average(predictions, axis=1, weights=self.weights_)
        else:
            return np.mean(predictions, axis=1)

class MultiObjectiveOptimizer:
    """Optimizador multi-objetivo con Optuna"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_params = {}
        self.best_scores = {}
        
    def hull_metric_detailed(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Métrica Hull detallada"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions
        y_pred = np.clip(y_pred, self.config.min_position, self.config.max_position)
        
        # Strategy returns
        risk_free_rate = 0.02/252
        strategy_returns = risk_free_rate * (1 - y_pred) + y_pred * y_true
        
        # Basic metrics
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return {
                'hull_score': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0
            }
        
        # Strategy excess returns
        strategy_excess = strategy_returns - risk_free_rate
        strategy_cumulative = (1 + strategy_excess).prod()
        strategy_mean_excess = strategy_cumulative ** (1 / len(strategy_excess)) - 1
        strategy_std = strategy_returns.std()
        
        # Market stats
        market_excess = y_true - risk_free_rate
        market_cumulative = (1 + market_excess).prod()
        market_mean_excess = market_cumulative ** (1 / len(market_excess)) - 1
        market_std = y_true.std()
        
        # Sharpe ratio
        trading_days = 252
        sharpe = strategy_mean_excess / strategy_std * np.sqrt(trading_days)
        
        # Volatility penalty
        strategy_vol = strategy_std * np.sqrt(trading_days)
        market_vol = market_std * np.sqrt(trading_days) if market_std > 0 else 0.01
        
        excess_vol = max(0, strategy_vol / market_vol - 1.2)
        vol_penalty = 1 + excess_vol
        
        # Return penalty
        return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days)
        return_penalty = 1 + (return_gap**2) / 100
        
        # Adjusted Sharpe
        adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
        
        # Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'hull_score': min(float(adjusted_sharpe), 1_000_000),
            'sharpe_ratio': float(sharpe),
            'volatility': float(strategy_vol),
            'max_drawdown': float(max_drawdown),
            'total_return': float(strategy_cumulative - 1)
        }
    
    def create_lgbm_objective(self, X: pd.DataFrame, y: pd.Series):
        """Objetivo para LightGBM"""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            return self._evaluate_model(lgb.LGBMRegressor(**params), X, y)
        
        return objective
    
    def create_xgb_objective(self, X: pd.DataFrame, y: pd.Series):
        """Objetivo para XGBoost"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            return self._evaluate_model(xgb.XGBRegressor(**params), X, y)
        
        return objective
    
    def create_catboost_objective(self, X: pd.DataFrame, y: pd.Series):
        """Objetivo para CatBoost"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': False
            }
            
            return self._evaluate_model(cb.CatBoostRegressor(**params), X, y)
        
        return objective
    
    def create_ensemble_objective(self, base_models: List[Tuple[str, object]], 
                                X: pd.DataFrame, y: pd.Series):
        """Objetivo para ensemble"""
        
        def objective(trial):
            # Seleccionar modelos para el ensemble
            n_models = trial.suggest_int('n_models', 3, min(len(base_models), self.config.ensemble_size))
            selected_indices = trial.suggest_categorical('selected_models', 
                                                       list(range(len(base_models))))
            
            # Método de weighting
            weighting_method = trial.suggest_categorical('weighting_method', 
                                                       ['equal', 'performance', 'inverse_error'])
            
            # Meta-learner
            use_meta_learner = trial.suggest_categorical('use_meta_learner', [True, False])
            
            selected_models = [base_models[i] for i in range(min(n_models, len(base_models)))]
            
            meta_learner = None
            if use_meta_learner:
                meta_type = trial.suggest_categorical('meta_learner_type', 
                                                    ['ridge', 'elastic', 'lgbm'])
                if meta_type == 'ridge':
                    alpha = trial.suggest_float('meta_alpha', 1e-8, 10.0, log=True)
                    meta_learner = Ridge(alpha=alpha, random_state=42)
                elif meta_type == 'elastic':
                    alpha = trial.suggest_float('meta_alpha', 1e-8, 10.0, log=True)
                    l1_ratio = trial.suggest_float('meta_l1_ratio', 0.1, 0.9)
                    meta_learner = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                else:
                    meta_learner = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            
            ensemble = AdvancedEnsemble(
                models=selected_models,
                weighting_method=weighting_method,
                meta_learner=meta_learner
            )
            
            return self._evaluate_model(ensemble, X, y)
        
        return objective
    
    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluar modelo con validación cruzada temporal"""
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            try:
                # Rellenar NaN
                X_train_filled = X_train.fillna(0)
                X_val_filled = X_val.fillna(0)
                
                # Entrenar y predecir
                model.fit(X_train_filled, y_train)
                y_pred = model.predict(X_val_filled)
                
                # Calcular métricas
                metrics = self.hull_metric_detailed(y_val.values, y_pred)
                scores.append(metrics['hull_score'])
                
            except Exception as e:
                # Si falla, retornar score muy bajo
                return -1000.0
        
        return np.mean(scores) if scores else -1000.0
    
    def optimize_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimizar un tipo de modelo específico"""
        
        print(f"🎯 Optimizing {model_type}...")
        
        # Crear objetivo según el tipo de modelo
        if model_type == 'lgbm':
            objective = self.create_lgbm_objective(X, y)
        elif model_type == 'xgb':
            objective = self.create_xgb_objective(X, y)
        elif model_type == 'catboost':
            objective = self.create_catboost_objective(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Crear estudio
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Optimizar
        study.optimize(
            objective,
            n_trials=self.config.n_trials // 3,  # Dividir trials entre modelos
            timeout=self.config.timeout // 3,
            n_jobs=1,  # Evitar conflictos con paralelización interna
            show_progress_bar=True
        )
        
        self.best_params[model_type] = study.best_params
        self.best_scores[model_type] = study.best_value
        
        print(f"  ✅ Best {model_type} score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_ensemble(self, base_models: List[Tuple[str, object]], 
                         X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimizar ensemble"""
        
        print("🤖 Optimizing Ensemble...")
        
        objective = self.create_ensemble_objective(base_models, X, y)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials // 3,
            timeout=self.config.timeout // 3,
            n_jobs=1,
            show_progress_bar=True
        )
        
        self.best_params['ensemble'] = study.best_params
        self.best_scores['ensemble'] = study.best_value
        
        print(f"  ✅ Best ensemble score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def comprehensive_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimización exhaustiva de todos los modelos"""
        
        print("🚀 Starting Comprehensive Optimization...")
        print("=" * 60)
        
        results = {}
        
        # 1. Optimizar modelos individuales
        model_types = ['lgbm', 'xgb', 'catboost']
        
        for model_type in model_types:
            try:
                result = self.optimize_model(model_type, X, y)
                results[model_type] = result
            except Exception as e:
                print(f"❌ {model_type} optimization failed: {e}")
                continue
        
        # 2. Crear modelos optimizados
        optimized_models = []
        
        for model_type in model_types:
            if model_type in self.best_params:
                try:
                    if model_type == 'lgbm':
                        model = lgb.LGBMRegressor(**self.best_params[model_type])
                    elif model_type == 'xgb':
                        model = xgb.XGBRegressor(**self.best_params[model_type])
                    elif model_type == 'catboost':
                        model = cb.CatBoostRegressor(**self.best_params[model_type])
                    
                    optimized_models.append((model_type, model))
                    
                except Exception as e:
                    print(f"❌ Failed to create optimized {model_type}: {e}")
                    continue
        
        # 3. Optimizar ensemble
        if len(optimized_models) >= 2:
            try:
                ensemble_result = self.optimize_ensemble(optimized_models, X, y)
                results['ensemble'] = ensemble_result
            except Exception as e:
                print(f"❌ Ensemble optimization failed: {e}")
        
        # 4. Resultados finales
        print("=" * 60)
        print("🏆 OPTIMIZATION RESULTS")
        print("=" * 60)
        
        best_score = 0
        best_model_type = None
        
        for model_type, score in self.best_scores.items():
            print(f"📊 {model_type.upper()}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model_type = model_type
        
        print(f"🥇 BEST MODEL: {best_model_type} (Score: {best_score:.4f})")
        
        results['summary'] = {
            'best_model_type': best_model_type,
            'best_score': best_score,
            'all_scores': self.best_scores.copy(),
            'all_params': self.best_params.copy()
        }
        
        return results

def create_optimized_model(model_type: str, params: Dict) -> object:
    """Crear modelo optimizado con parámetros"""
    
    if model_type == 'lgbm':
        return lgb.LGBMRegressor(**params)
    elif model_type == 'xgb':
        return xgb.XGBRegressor(**params)
    elif model_type == 'catboost':
        return cb.CatBoostRegressor(**params)
    elif model_type == 'ensemble':
        # Para ensemble, necesitamos crear los modelos base primero
        return None  # Se maneja por separado
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_advanced_optimization():
    """Ejecutar optimización avanzada"""
    
    print("🎯 HULL TACTICAL - ADVANCED OPTIMIZATION")
    print("=" * 60)
    print("🚀 Multi-objective Bayesian Optimization")
    print("🤖 Advanced Ensemble Methods")
    print("🛡️ Risk Management Integration")
    print("=" * 60)
    
    # Configuración
    config = OptimizationConfig(
        n_trials=500,
        timeout=7200,  # 2 horas
        ensemble_size=15,
        cv_folds=5
    )
    
    # Crear optimizador
    optimizer = MultiObjectiveOptimizer(config)
    
    print("✅ Advanced optimization framework initialized")
    print(f"🎯 Trials: {config.n_trials}")
    print(f"⏱️ Timeout: {config.timeout//60} minutes")
    print(f"🤖 Ensemble size: {config.ensemble_size}")
    print(f"🔄 CV folds: {config.cv_folds}")
    
    return optimizer, config

if __name__ == "__main__":
    optimizer, config = run_advanced_optimization()
    print("\n🚀 Advanced optimization framework ready!")
    print("📝 Use optimizer.comprehensive_optimization(X, y) to optimize your models")