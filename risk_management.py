#!/usr/bin/env python3
"""
🛡️ Sistema de Gestión de Riesgo - Hull Tactical Market Prediction
Optimización de constraints y volatility targeting

Autor: OpenHands AI Assistant
Fecha: 2024-11-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

@dataclass
class RiskConfig:
    """Configuración de gestión de riesgo"""
    
    # Position constraints
    min_position: float = -6.0
    max_position: float = 6.0
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% anual
    max_volatility: float = 0.20     # 20% máximo
    
    # Drawdown limits
    max_drawdown: float = 0.08       # 8% máximo
    
    # Risk metrics
    var_confidence: float = 0.05     # 5% VaR
    lookback_window: int = 252       # 1 año
    
    # Optimization
    risk_aversion: float = 2.0       # Coeficiente de aversión al riesgo
    transaction_cost: float = 0.001  # 0.1% costo de transacción

class AdvancedRiskManager:
    """Gestor de riesgo avanzado para optimizar Hull metric"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.covariance_estimator = LedoitWolf()
        
    def calculate_portfolio_risk(self, positions: np.ndarray, 
                                returns: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de riesgo del portfolio"""
        
        if len(positions) != len(returns):
            raise ValueError("Positions and returns must have same length")
        
        # Portfolio returns
        portfolio_returns = positions * returns
        
        # Basic risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # VaR calculation
        var_5 = np.percentile(portfolio_returns, self.config.var_confidence * 100)
        
        # Expected Shortfall (CVaR)
        cvar_5 = np.mean(portfolio_returns[portfolio_returns <= var_5])
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sharpe Ratio
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
        
        return {
            'volatility': volatility,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_return': np.prod(1 + portfolio_returns) - 1
        }
    
    def volatility_targeting(self, raw_positions: np.ndarray, 
                           returns_history: np.ndarray) -> np.ndarray:
        """Aplicar volatility targeting a las posiciones"""
        
        if len(returns_history) < 20:
            return np.clip(raw_positions, self.config.min_position, self.config.max_position)
        
        # Estimar volatilidad histórica
        historical_vol = np.std(returns_history[-self.config.lookback_window:]) * np.sqrt(252)
        
        if historical_vol == 0:
            return np.clip(raw_positions, self.config.min_position, self.config.max_position)
        
        # Factor de escalamiento para alcanzar volatilidad objetivo
        vol_scalar = self.config.target_volatility / historical_vol
        
        # Aplicar límites al scalar
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Entre 50% y 200%
        
        # Escalar posiciones
        scaled_positions = raw_positions * vol_scalar
        
        # Aplicar constraints finales
        final_positions = np.clip(scaled_positions, self.config.min_position, self.config.max_position)
        
        return final_positions
    
    def dynamic_position_sizing(self, predictions: np.ndarray, 
                               returns_history: np.ndarray,
                               confidence_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Dimensionamiento dinámico de posiciones basado en confianza y riesgo"""
        
        # Si no hay scores de confianza, usar valores uniformes
        if confidence_scores is None:
            confidence_scores = np.ones(len(predictions))
        
        # Normalizar scores de confianza
        confidence_scores = np.clip(confidence_scores, 0.1, 1.0)
        
        # Calcular volatilidad rolling
        if len(returns_history) >= 20:
            rolling_vol = pd.Series(returns_history).rolling(window=20).std().iloc[-len(predictions):].values
            rolling_vol = np.where(np.isnan(rolling_vol), np.nanmean(rolling_vol), rolling_vol)
        else:
            rolling_vol = np.full(len(predictions), np.std(returns_history))
        
        # Factor de reducción basado en volatilidad
        vol_factor = np.clip(0.15 / (rolling_vol * np.sqrt(252) + 1e-8), 0.3, 1.5)
        
        # Combinar factores
        sizing_factor = confidence_scores * vol_factor
        
        # Aplicar a predicciones
        sized_positions = predictions * sizing_factor
        
        # Aplicar volatility targeting
        final_positions = self.volatility_targeting(sized_positions, returns_history)
        
        return final_positions
    
    def optimize_positions_for_hull_metric(self, raw_predictions: np.ndarray,
                                         returns_history: np.ndarray,
                                         expected_returns: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimizar posiciones específicamente para maximizar Hull metric"""
        
        if expected_returns is None:
            expected_returns = raw_predictions * 0.001  # Asumir pequeños returns esperados
        
        def hull_objective(positions):
            """Función objetivo para optimización"""
            
            # Simular returns del portfolio
            portfolio_returns = positions * expected_returns
            
            # Calcular métricas Hull
            if len(portfolio_returns) == 0 or np.std(portfolio_returns) == 0:
                return -1000  # Penalizar fuertemente
            
            # Strategy stats
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            # Sharpe ratio
            sharpe = mean_return / std_return * np.sqrt(252)
            
            # Penalties
            volatility = std_return * np.sqrt(252)
            vol_penalty = 1 + max(0, volatility / 0.15 - 1.2) * 0.5  # Más suave
            
            # Return penalty (simplificado)
            return_penalty = 1 + max(0, -mean_return * 252) / 100
            
            # Hull score aproximado
            hull_score = sharpe / (vol_penalty * return_penalty)
            
            # Penalizar posiciones extremas
            position_penalty = np.sum(np.abs(positions) ** 2) * 0.001
            
            return -(hull_score - position_penalty)  # Minimizar negativo = maximizar
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.config.max_position - np.max(x)},
            {'type': 'ineq', 'fun': lambda x: np.min(x) - self.config.min_position}
        ]
        
        # Bounds
        bounds = [(self.config.min_position, self.config.max_position) for _ in range(len(raw_predictions))]
        
        # Optimización
        try:
            result = minimize(
                hull_objective,
                raw_predictions,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                return result.x
            else:
                print(f"⚠️ Optimization failed: {result.message}")
                return np.clip(raw_predictions, self.config.min_position, self.config.max_position)
                
        except Exception as e:
            print(f"⚠️ Optimization error: {e}")
            return np.clip(raw_predictions, self.config.min_position, self.config.max_position)
    
    def apply_risk_overlay(self, raw_predictions: np.ndarray,
                          returns_history: np.ndarray,
                          method: str = 'comprehensive') -> np.ndarray:
        """Aplicar overlay de gestión de riesgo"""
        
        print(f"🛡️ Applying {method} risk management...")
        
        if method == 'simple':
            # Solo clipping básico
            return np.clip(raw_predictions, self.config.min_position, self.config.max_position)
        
        elif method == 'volatility_targeting':
            # Solo volatility targeting
            return self.volatility_targeting(raw_predictions, returns_history)
        
        elif method == 'dynamic_sizing':
            # Dimensionamiento dinámico
            return self.dynamic_position_sizing(raw_predictions, returns_history)
        
        elif method == 'hull_optimized':
            # Optimización específica para Hull metric
            return self.optimize_positions_for_hull_metric(raw_predictions, returns_history)
        
        elif method == 'comprehensive':
            # Aplicar todos los métodos en secuencia
            
            # 1. Dimensionamiento dinámico
            step1 = self.dynamic_position_sizing(raw_predictions, returns_history)
            
            # 2. Optimización Hull
            step2 = self.optimize_positions_for_hull_metric(step1, returns_history)
            
            # 3. Volatility targeting final
            final_positions = self.volatility_targeting(step2, returns_history)
            
            return final_positions
        
        else:
            raise ValueError(f"Unknown risk management method: {method}")
    
    def backtest_risk_strategy(self, predictions: np.ndarray,
                              actual_returns: np.ndarray,
                              method: str = 'comprehensive') -> Dict[str, float]:
        """Backtest de estrategia de gestión de riesgo"""
        
        print(f"📊 Backtesting {method} risk strategy...")
        
        # Aplicar gestión de riesgo
        managed_positions = self.apply_risk_overlay(predictions, actual_returns, method)
        
        # Calcular métricas
        risk_metrics = self.calculate_portfolio_risk(managed_positions, actual_returns)
        
        # Hull metric
        risk_free_rate = 0.02/252
        strategy_returns = risk_free_rate * (1 - managed_positions) + managed_positions * actual_returns
        
        # Hull calculation
        strategy_excess = strategy_returns - risk_free_rate
        strategy_cumulative = np.prod(1 + strategy_excess)
        strategy_mean_excess = strategy_cumulative ** (1 / len(strategy_excess)) - 1
        strategy_std = np.std(strategy_returns)
        
        market_excess = actual_returns - risk_free_rate
        market_cumulative = np.prod(1 + market_excess)
        market_mean_excess = market_cumulative ** (1 / len(market_excess)) - 1
        market_std = np.std(actual_returns)
        
        if strategy_std > 0 and market_std > 0:
            sharpe = strategy_mean_excess / strategy_std * np.sqrt(252)
            
            strategy_vol = strategy_std * np.sqrt(252)
            market_vol = market_std * np.sqrt(252)
            
            excess_vol = max(0, strategy_vol / market_vol - 1.2)
            vol_penalty = 1 + excess_vol
            
            return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * 252)
            return_penalty = 1 + (return_gap**2) / 100
            
            hull_score = sharpe / (vol_penalty * return_penalty)
        else:
            hull_score = 0.0
        
        risk_metrics['hull_score'] = min(hull_score, 1_000_000)
        risk_metrics['method'] = method
        
        return risk_metrics

def create_risk_optimized_predictions(raw_predictions: np.ndarray,
                                    returns_history: np.ndarray,
                                    config: Optional[RiskConfig] = None) -> np.ndarray:
    """Función principal para crear predicciones optimizadas por riesgo"""
    
    if config is None:
        config = RiskConfig()
    
    risk_manager = AdvancedRiskManager(config)
    
    # Aplicar gestión de riesgo comprehensiva
    optimized_predictions = risk_manager.apply_risk_overlay(
        raw_predictions, 
        returns_history, 
        method='comprehensive'
    )
    
    return optimized_predictions

def compare_risk_methods(predictions: np.ndarray,
                        returns_history: np.ndarray) -> pd.DataFrame:
    """Comparar diferentes métodos de gestión de riesgo"""
    
    config = RiskConfig()
    risk_manager = AdvancedRiskManager(config)
    
    methods = ['simple', 'volatility_targeting', 'dynamic_sizing', 'hull_optimized', 'comprehensive']
    results = []
    
    for method in methods:
        try:
            metrics = risk_manager.backtest_risk_strategy(predictions, returns_history, method)
            results.append(metrics)
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('hull_score', ascending=False)
        return df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    print("🛡️ HULL TACTICAL - ADVANCED RISK MANAGEMENT")
    print("=" * 60)
    print("🎯 Volatility Targeting")
    print("📊 Dynamic Position Sizing") 
    print("🔧 Hull Metric Optimization")
    print("📈 Comprehensive Risk Overlay")
    print("=" * 60)
    
    # Ejemplo de uso
    np.random.seed(42)
    n_samples = 1000
    
    # Datos sintéticos
    raw_preds = np.random.normal(0, 1, n_samples)
    returns_hist = np.random.normal(0, 0.02, n_samples)
    
    # Crear predicciones optimizadas
    optimized_preds = create_risk_optimized_predictions(raw_preds, returns_hist)
    
    print(f"\n📊 Risk Optimization Results:")
    print(f"  Original range: [{raw_preds.min():.3f}, {raw_preds.max():.3f}]")
    print(f"  Optimized range: [{optimized_preds.min():.3f}, {optimized_preds.max():.3f}]")
    print(f"  Volatility reduction: {(np.std(raw_preds) - np.std(optimized_preds))/np.std(raw_preds)*100:.1f}%")
    
    print("\n🚀 Risk management framework ready!")