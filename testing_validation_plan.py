#!/usr/bin/env python3
"""
🧪 Plan de Pruebas Exhaustivo - Hull Tactical Market Prediction
Objetivo: Validar que el modelo pueda alcanzar scores de 10-11+ para el primer puesto

Autor: OpenHands AI Assistant
Fecha: 2024-11-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Importar modelos y métricas
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

@dataclass
class TestingConfig:
    """Configuración para el plan de pruebas"""
    
    # Validación cruzada
    cv_folds: int = 10
    cv_method: str = 'time_series'  # 'time_series', 'walk_forward', 'purged'
    
    # Backtesting
    backtest_periods: int = 5
    min_train_size: int = 500
    step_size: int = 50
    
    # Métricas objetivo
    target_hull_score: float = 10.0  # Score objetivo para primer puesto
    target_sharpe: float = 2.0
    max_volatility: float = 0.20
    max_drawdown: float = 0.10
    
    # Optimización
    optuna_trials: int = 200
    optuna_timeout: int = 3600  # 1 hora
    
    # Robustez
    noise_levels: List[float] = None
    bootstrap_samples: int = 100
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]

class ComprehensiveValidator:
    """Validador exhaustivo para modelos de trading"""
    
    def __init__(self, config: TestingConfig):
        self.config = config
        self.results = {}
        self.best_params = {}
        
    def hull_metric_detailed(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           risk_free_rate: float = 0.02/252) -> Dict[str, float]:
        """
        Implementación detallada de la métrica Hull con componentes individuales
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions
        y_pred = np.clip(y_pred, -6.0, 6.0)
        
        # Strategy returns
        strategy_returns = risk_free_rate * (1 - y_pred) + y_pred * y_true
        
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
        
        # Trading days
        trading_days = 252
        
        if strategy_std == 0 or market_std == 0:
            return {
                'hull_score': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'vol_penalty': 1.0,
                'return_penalty': 1.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
        
        # Sharpe ratio
        sharpe = strategy_mean_excess / strategy_std * np.sqrt(trading_days)
        
        # Volatility penalty
        strategy_vol = strategy_std * np.sqrt(trading_days)
        market_vol = market_std * np.sqrt(trading_days)
        
        excess_vol = max(0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0
        vol_penalty = 1 + excess_vol
        
        # Return penalty
        return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days)
        return_penalty = 1 + (return_gap**2) / 100
        
        # Adjusted Sharpe
        adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
        
        # Additional metrics
        total_return = strategy_cumulative - 1
        
        # Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'hull_score': min(float(adjusted_sharpe), 1_000_000),
            'sharpe_ratio': float(sharpe),
            'volatility': float(strategy_vol),
            'vol_penalty': float(vol_penalty),
            'return_penalty': float(return_penalty),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'strategy_mean_return': float(strategy_mean_excess * trading_days),
            'market_mean_return': float(market_mean_excess * trading_days)
        }
    
    def time_series_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Validación cruzada con series temporales"""
        print(f"🔄 Running Time Series Cross Validation ({self.config.cv_folds} folds)...")
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        fold_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold + 1}/{self.config.cv_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Entrenar modelo
            try:
                model.fit(X_train_fold.fillna(0), y_train_fold)
                y_pred_fold = model.predict(X_val_fold.fillna(0))
                
                # Calcular métricas
                metrics = self.hull_metric_detailed(y_val_fold.values, y_pred_fold)
                fold_scores.append(metrics['hull_score'])
                fold_metrics.append(metrics)
                
                print(f"    Hull Score: {metrics['hull_score']:.4f}")
                
            except Exception as e:
                print(f"    Fold {fold + 1} failed: {e}")
                continue
        
        if fold_scores:
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                avg_metrics[f'cv_{key}_mean'] = np.mean([m[key] for m in fold_metrics])
                avg_metrics[f'cv_{key}_std'] = np.std([m[key] for m in fold_metrics])
            
            print(f"  ✅ CV Hull Score: {avg_metrics['cv_hull_score_mean']:.4f} ± {avg_metrics['cv_hull_score_std']:.4f}")
            return avg_metrics
        else:
            return {'cv_hull_score_mean': 0.0, 'cv_hull_score_std': 0.0}
    
    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Validación walk-forward"""
        print(f"🚶 Running Walk-Forward Validation...")
        
        scores = []
        all_metrics = []
        
        for i in range(self.config.min_train_size, len(X) - self.config.step_size, self.config.step_size):
            print(f"  Period {len(scores) + 1}: Train[0:{i}], Test[{i}:{i + self.config.step_size}]")
            
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i + self.config.step_size]
            y_test = y.iloc[i:i + self.config.step_size]
            
            try:
                model.fit(X_train.fillna(0), y_train)
                y_pred = model.predict(X_test.fillna(0))
                
                metrics = self.hull_metric_detailed(y_test.values, y_pred)
                scores.append(metrics['hull_score'])
                all_metrics.append(metrics)
                
                print(f"    Hull Score: {metrics['hull_score']:.4f}")
                
            except Exception as e:
                print(f"    Period failed: {e}")
                continue
        
        if scores:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[f'wf_{key}_mean'] = np.mean([m[key] for m in all_metrics])
                avg_metrics[f'wf_{key}_std'] = np.std([m[key] for m in all_metrics])
            
            print(f"  ✅ WF Hull Score: {avg_metrics['wf_hull_score_mean']:.4f} ± {avg_metrics['wf_hull_score_std']:.4f}")
            return avg_metrics
        else:
            return {'wf_hull_score_mean': 0.0, 'wf_hull_score_std': 0.0}
    
    def robustness_testing(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Pruebas de robustez con ruido"""
        print(f"🛡️ Running Robustness Testing...")
        
        robustness_results = {}
        
        for noise_level in self.config.noise_levels:
            print(f"  Testing with {noise_level*100:.1f}% noise...")
            
            scores = []
            
            for trial in range(10):  # 10 trials per noise level
                # Añadir ruido a las características
                X_noisy = X.copy()
                for col in X_noisy.columns:
                    if X_noisy[col].dtype in ['float64', 'int64']:
                        noise = np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
                        X_noisy[col] += noise
                
                # Split temporal
                split_idx = int(len(X_noisy) * 0.8)
                X_train = X_noisy.iloc[:split_idx]
                y_train = y.iloc[:split_idx]
                X_test = X_noisy.iloc[split_idx:]
                y_test = y.iloc[split_idx:]
                
                try:
                    model.fit(X_train.fillna(0), y_train)
                    y_pred = model.predict(X_test.fillna(0))
                    
                    metrics = self.hull_metric_detailed(y_test.values, y_pred)
                    scores.append(metrics['hull_score'])
                    
                except Exception as e:
                    continue
            
            if scores:
                robustness_results[f'noise_{noise_level:.2f}_mean'] = np.mean(scores)
                robustness_results[f'noise_{noise_level:.2f}_std'] = np.std(scores)
                print(f"    Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return robustness_results
    
    def bootstrap_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Validación bootstrap"""
        print(f"🎲 Running Bootstrap Validation ({self.config.bootstrap_samples} samples)...")
        
        scores = []
        
        for i in range(self.config.bootstrap_samples):
            if i % 20 == 0:
                print(f"  Bootstrap sample {i + 1}/{self.config.bootstrap_samples}")
            
            # Bootstrap sample
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[sample_idx].reset_index(drop=True)
            y_bootstrap = y.iloc[sample_idx].reset_index(drop=True)
            
            # Split temporal en el bootstrap
            split_idx = int(len(X_bootstrap) * 0.8)
            X_train = X_bootstrap.iloc[:split_idx]
            y_train = y_bootstrap.iloc[:split_idx]
            X_test = X_bootstrap.iloc[split_idx:]
            y_test = y_bootstrap.iloc[split_idx:]
            
            try:
                model.fit(X_train.fillna(0), y_train)
                y_pred = model.predict(X_test.fillna(0))
                
                metrics = self.hull_metric_detailed(y_test.values, y_pred)
                scores.append(metrics['hull_score'])
                
            except Exception as e:
                continue
        
        if scores:
            bootstrap_results = {
                'bootstrap_mean': np.mean(scores),
                'bootstrap_std': np.std(scores),
                'bootstrap_q25': np.percentile(scores, 25),
                'bootstrap_q75': np.percentile(scores, 75),
                'bootstrap_min': np.min(scores),
                'bootstrap_max': np.max(scores)
            }
            
            print(f"  ✅ Bootstrap Score: {bootstrap_results['bootstrap_mean']:.4f} ± {bootstrap_results['bootstrap_std']:.4f}")
            print(f"  📊 Range: [{bootstrap_results['bootstrap_min']:.4f}, {bootstrap_results['bootstrap_max']:.4f}]")
            
            return bootstrap_results
        else:
            return {'bootstrap_mean': 0.0, 'bootstrap_std': 0.0}
    
    def hyperparameter_optimization(self, model_class, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimización de hiperparámetros con Optuna"""
        print(f"🎯 Running Hyperparameter Optimization...")
        
        def objective(trial):
            # Definir espacio de búsqueda según el tipo de modelo
            if 'LGBMRegressor' in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            elif 'XGBRegressor' in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
            else:
                # Parámetros genéricos
                params = {
                    'random_state': 42
                }
            
            # Crear modelo con parámetros
            model = model_class(**params)
            
            # Validación cruzada
            tscv = TimeSeriesSplit(n_splits=3)  # Menos folds para velocidad
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                try:
                    model.fit(X_train_fold.fillna(0), y_train_fold)
                    y_pred_fold = model.predict(X_val_fold.fillna(0))
                    
                    metrics = self.hull_metric_detailed(y_val_fold.values, y_pred_fold)
                    scores.append(metrics['hull_score'])
                    
                except Exception as e:
                    return 0.0
            
            return np.mean(scores) if scores else 0.0
        
        # Crear estudio Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimizar
        study.optimize(
            objective, 
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True
        )
        
        print(f"  ✅ Best Score: {study.best_value:.4f}")
        print(f"  🎯 Best Params: {study.best_params}")
        
        self.best_params[str(model_class)] = study.best_params
        
        return {
            'optuna_best_score': study.best_value,
            'optuna_best_params': study.best_params,
            'optuna_n_trials': len(study.trials)
        }
    
    def comprehensive_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Validación exhaustiva completa"""
        print("🧪 Starting Comprehensive Validation...")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Time Series Cross Validation
        cv_results = self.time_series_cross_validation(model, X, y)
        all_results.update(cv_results)
        
        # 2. Walk-Forward Validation
        wf_results = self.walk_forward_validation(model, X, y)
        all_results.update(wf_results)
        
        # 3. Robustness Testing
        robustness_results = self.robustness_testing(model, X, y)
        all_results.update(robustness_results)
        
        # 4. Bootstrap Validation
        bootstrap_results = self.bootstrap_validation(model, X, y)
        all_results.update(bootstrap_results)
        
        # 5. Hyperparameter Optimization
        if hasattr(model, '__class__'):
            optuna_results = self.hyperparameter_optimization(model.__class__, X, y)
            all_results.update(optuna_results)
        
        # Calcular score final compuesto
        cv_score = all_results.get('cv_hull_score_mean', 0)
        wf_score = all_results.get('wf_hull_score_mean', 0)
        bootstrap_score = all_results.get('bootstrap_mean', 0)
        
        # Score compuesto (promedio ponderado)
        composite_score = (cv_score * 0.4 + wf_score * 0.4 + bootstrap_score * 0.2)
        all_results['composite_score'] = composite_score
        
        # Evaluación de objetivos
        target_achieved = composite_score >= self.config.target_hull_score
        all_results['target_achieved'] = target_achieved
        
        print("=" * 60)
        print("🏆 COMPREHENSIVE VALIDATION RESULTS")
        print("=" * 60)
        print(f"📊 Cross Validation Score: {cv_score:.4f}")
        print(f"🚶 Walk-Forward Score: {wf_score:.4f}")
        print(f"🎲 Bootstrap Score: {bootstrap_score:.4f}")
        print(f"🎯 Composite Score: {composite_score:.4f}")
        print(f"🏁 Target ({self.config.target_hull_score:.1f}): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        if target_achieved:
            print("🎉 MODEL READY FOR FIRST PLACE COMPETITION!")
        else:
            print("⚠️  MODEL NEEDS IMPROVEMENT FOR FIRST PLACE")
            print(f"   Gap to target: {self.config.target_hull_score - composite_score:.4f}")
        
        print("=" * 60)
        
        self.results = all_results
        return all_results
    
    def generate_validation_report(self, results: Dict[str, float], 
                                 model_name: str = "Model") -> str:
        """Generar reporte detallado de validación"""
        
        report = f"""
# 🧪 VALIDATION REPORT - {model_name}
## Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 OBJECTIVE ASSESSMENT
- **Target Hull Score**: {self.config.target_hull_score:.1f}
- **Achieved Score**: {results.get('composite_score', 0):.4f}
- **Status**: {'✅ TARGET ACHIEVED' if results.get('target_achieved', False) else '❌ TARGET NOT ACHIEVED'}

## 📊 VALIDATION METRICS

### Cross Validation (Time Series)
- **Mean Score**: {results.get('cv_hull_score_mean', 0):.4f} ± {results.get('cv_hull_score_std', 0):.4f}
- **Sharpe Ratio**: {results.get('cv_sharpe_ratio_mean', 0):.4f}
- **Volatility**: {results.get('cv_volatility_mean', 0):.4f}
- **Max Drawdown**: {results.get('cv_max_drawdown_mean', 0):.4f}

### Walk-Forward Validation
- **Mean Score**: {results.get('wf_hull_score_mean', 0):.4f} ± {results.get('wf_hull_score_std', 0):.4f}
- **Consistency**: {1 - results.get('wf_hull_score_std', 1) / max(results.get('wf_hull_score_mean', 1), 0.001):.2%}

### Bootstrap Validation
- **Mean Score**: {results.get('bootstrap_mean', 0):.4f} ± {results.get('bootstrap_std', 0):.4f}
- **95% Confidence**: [{results.get('bootstrap_q25', 0):.4f}, {results.get('bootstrap_q75', 0):.4f}]
- **Range**: [{results.get('bootstrap_min', 0):.4f}, {results.get('bootstrap_max', 0):.4f}]

### Robustness Testing
"""
        
        # Añadir resultados de robustez
        for noise_level in self.config.noise_levels:
            mean_key = f'noise_{noise_level:.2f}_mean'
            std_key = f'noise_{noise_level:.2f}_std'
            if mean_key in results:
                report += f"- **{noise_level*100:.1f}% Noise**: {results[mean_key]:.4f} ± {results[std_key]:.4f}\n"
        
        report += f"""
### Hyperparameter Optimization
- **Best Score**: {results.get('optuna_best_score', 0):.4f}
- **Trials**: {results.get('optuna_n_trials', 0)}

## 🏆 COMPETITION READINESS

### First Place Criteria
- Hull Score ≥ {self.config.target_hull_score}: {'✅' if results.get('composite_score', 0) >= self.config.target_hull_score else '❌'}
- Sharpe Ratio ≥ {self.config.target_sharpe}: {'✅' if results.get('cv_sharpe_ratio_mean', 0) >= self.config.target_sharpe else '❌'}
- Volatility ≤ {self.config.max_volatility*100:.0f}%: {'✅' if results.get('cv_volatility_mean', 1) <= self.config.max_volatility else '❌'}
- Max Drawdown ≤ {self.config.max_drawdown*100:.0f}%: {'✅' if abs(results.get('cv_max_drawdown_mean', -1)) <= self.config.max_drawdown else '❌'}

### Recommendations
"""
        
        # Generar recomendaciones
        composite_score = results.get('composite_score', 0)
        if composite_score >= self.config.target_hull_score:
            report += "- ✅ Model is ready for submission\n"
            report += "- 🚀 Expected to compete for first place\n"
            report += "- 📈 Consider ensemble methods for additional improvement\n"
        else:
            gap = self.config.target_hull_score - composite_score
            report += f"- ⚠️ Score gap: {gap:.4f} points\n"
            report += "- 🔧 Recommend hyperparameter tuning\n"
            report += "- 📊 Consider additional feature engineering\n"
            report += "- 🤖 Try ensemble methods\n"
        
        return report

def run_comprehensive_testing():
    """Ejecutar plan de pruebas completo"""
    
    print("🧪 HULL TACTICAL - COMPREHENSIVE TESTING PLAN")
    print("=" * 60)
    print("🎯 Objective: Validate model for FIRST PLACE (Score 10-11+)")
    print("=" * 60)
    
    # Configuración
    config = TestingConfig(
        target_hull_score=10.0,  # Objetivo para primer puesto
        cv_folds=10,
        optuna_trials=200,
        bootstrap_samples=100
    )
    
    # Crear validador
    validator = ComprehensiveValidator(config)
    
    print("✅ Testing framework initialized")
    print(f"🎯 Target Hull Score: {config.target_hull_score}")
    print(f"🔄 CV Folds: {config.cv_folds}")
    print(f"🎲 Bootstrap Samples: {config.bootstrap_samples}")
    print(f"🎯 Optuna Trials: {config.optuna_trials}")
    
    return validator, config

if __name__ == "__main__":
    validator, config = run_comprehensive_testing()
    print("\n🚀 Testing framework ready!")
    print("📝 Use validator.comprehensive_validation(model, X, y) to test your model")