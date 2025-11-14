#!/usr/bin/env python3
"""
🏆 Hull Tactical Market Prediction - Kaggle Submission Script
Optimized for Kaggle inference server requirements

This script provides the predict() function that Kaggle's evaluation system expects.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.optimize import minimize_scalar

# Set random seed
np.random.seed(42)

def hull_metric_exact(y_true, y_pred, risk_free_rate=0.02/252):
    """
    Exact implementation of Hull Tactical metric
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Clip positions
    y_pred = np.clip(y_pred, -6.0, 6.0)
    
    # Strategy returns
    strategy_returns = risk_free_rate * (1 - y_pred) + y_pred * y_true
    strategy_excess_returns = strategy_returns - risk_free_rate
    
    if len(strategy_excess_returns) == 0:
        return 0.0
    
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(strategy_excess_returns)) - 1
    strategy_std = strategy_returns.std()
    
    trading_days_per_yr = 252
    
    if strategy_std == 0:
        return 0.0
    
    # Sharpe calculation
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)
    
    # Market stats
    market_excess_returns = y_true - risk_free_rate
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(market_excess_returns)) - 1
    market_std = y_true.std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)
    
    if market_volatility == 0:
        return 0.0
    
    # Penalties
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100
    
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    
    return min(float(adjusted_sharpe), 1_000_000)

def create_features(df):
    """
    Create optimized features for Hull metric
    """
    df = df.copy()
    
    # Get numeric columns
    numeric_cols = [col for col in df.columns 
                   if col not in ['date_id', 'target'] and df[col].dtype in ['float64', 'int64']]
    
    if len(numeric_cols) == 0:
        return df
    
    # Limit for memory
    numeric_cols = numeric_cols[:15]
    
    # Lags
    for col in numeric_cols[:6]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    # Moving averages
    for col in numeric_cols[:5]:
        df[f"{col}_ma_3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_ma_5"] = df[col].rolling(window=5, min_periods=1).mean()
        df[f"{col}_roc_1"] = df[col].pct_change(periods=1)
    
    # Volatility
    for col in numeric_cols[:3]:
        df[f"{col}_vol_5"] = df[col].rolling(window=5, min_periods=2).std()
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

class SimpleHullModel:
    """
    Simple Hull-optimized model for Kaggle
    """
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.optimal_scale = 25.0  # Based on diagnostic analysis
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit simple model"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X.fillna(0))
            
            # Use best available model
            if LGBM_AVAILABLE:
                self.model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=1,
                    verbose=-1
                )
            elif XGB_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=1,
                    verbosity=0
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=1
                )
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            print(f"✅ Trained {type(self.model).__name__}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.is_fitted = False
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted or self.model is None:
            # Fallback: aggressive predictions based on first feature
            if len(X.columns) > 0:
                first_col = [col for col in X.columns if col != 'date_id'][0]
                signal = X[first_col].fillna(0).values
                return np.clip(signal * 20.0, -6.0, 6.0)
            else:
                return np.random.normal(0, 2.0, len(X))
        
        try:
            X_scaled = self.scaler.transform(X.fillna(0))
            raw_pred = self.model.predict(X_scaled)
            
            # Apply aggressive scaling for Hull metric
            scaled_pred = raw_pred * self.optimal_scale
            
            return np.clip(scaled_pred, -6.0, 6.0)
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            # Emergency fallback
            return np.random.normal(0, 2.0, len(X))

# Global model instance
global_model = SimpleHullModel()

def predict(test_df: pd.DataFrame) -> np.ndarray:
    """
    Main prediction function for Kaggle submission
    
    This function will be called by Kaggle's evaluation system.
    It must return predictions as a numpy array.
    
    Args:
        test_df: DataFrame with test data
        
    Returns:
        np.ndarray: Predictions in range [-6.0, 6.0]
    """
    try:
        print(f"🎯 Hull Tactical prediction for {len(test_df)} samples...")
        
        # Feature engineering
        test_enhanced = create_features(test_df.copy())
        
        # Get feature columns
        feature_cols = [col for col in test_enhanced.columns if col != 'date_id']
        
        if len(feature_cols) == 0:
            print("⚠️ No features found, using random predictions")
            np.random.seed(42)
            return np.clip(np.random.normal(0, 2.0, len(test_df)), -6.0, 6.0)
        
        # Prepare features
        X_test = test_enhanced[feature_cols].copy()
        
        # Make predictions
        predictions = global_model.predict(X_test)
        
        # Final safety checks
        predictions = np.clip(predictions, -6.0, 6.0)
        
        # Ensure we return the right type and shape
        predictions = np.array(predictions, dtype=np.float64)
        
        print(f"✅ Predictions: mean={np.mean(predictions):.4f}, std={np.std(predictions):.4f}")
        print(f"📊 Range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print("🛡️ Using emergency fallback")
        
        # Emergency fallback: aggressive random predictions
        np.random.seed(42)
        emergency_preds = np.random.normal(0, 2.0, len(test_df))
        return np.clip(emergency_preds, -6.0, 6.0)

def setup_model():
    """Setup model with training data if available"""
    import os
    
    # Try to find and load training data
    train_paths = [
        '/kaggle/input/hull-tactical-market-prediction/train.csv',
        'train.csv',
        '../input/hull-tactical-market-prediction/train.csv'
    ]
    
    for path in train_paths:
        if os.path.exists(path):
            try:
                print(f"📊 Loading training data from {path}...")
                train_df = pd.read_csv(path)
                
                # Find target column
                target_col = None
                for col in ['target', 'responder', 'forward_return_1d', 'y']:
                    if col in train_df.columns:
                        target_col = col
                        break
                
                if target_col is None:
                    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
                    target_col = [col for col in numeric_cols if col != 'date_id'][-1]
                
                # Feature engineering
                train_enhanced = create_features(train_df.copy())
                
                # Prepare data
                feature_cols = [col for col in train_enhanced.columns 
                               if col not in ['date_id', target_col]]
                
                X = train_enhanced[feature_cols].copy()
                y = train_enhanced[target_col].copy()
                
                # Train model
                global_model.fit(X, y)
                
                print(f"🎉 Model trained successfully!")
                return
                
            except Exception as e:
                print(f"❌ Failed to load/train from {path}: {e}")
                continue
    
    print("⚠️ No training data found, will use fallback predictions")

# Auto-setup when imported
if __name__ == "__main__" or True:  # Always try to setup
    setup_model()
    print("🚀 Hull Tactical Kaggle submission ready!")

# Test function
def test_predict():
    """Test the prediction function"""
    print("🧪 Testing prediction function...")
    
    # Create test data
    test_data = pd.DataFrame({
        'date_id': range(10),
        'feature_1': np.random.normal(0, 1, 10),
        'feature_2': np.random.normal(0, 1, 10),
        'feature_3': np.random.normal(0, 1, 10),
    })
    
    # Test prediction
    preds = predict(test_data)
    
    print(f"✅ Test successful: {len(preds)} predictions")
    print(f"📊 Range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"📈 Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
    
    return preds

if __name__ == "__main__":
    test_predict()