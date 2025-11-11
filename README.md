# Hull Tactical Market Prediction - AutoML Solution

Este repositorio contiene una solución completa usando AutoML para la competencia [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) de Kaggle.

## 🎯 Objetivo de la Competencia

Predecir los retornos diarios del S&P 500 usando un conjunto de características de mercado, optimizando para el **Adjusted Sharpe Ratio**.

## 📊 Métrica de Evaluación

La competencia utiliza un **Adjusted Sharpe Ratio** que:
- Calcula retornos de estrategia: `strategy_returns = y_true * y_pred`
- Penaliza alta volatilidad: `volatility_penalty = 1.0 / (1.0 + std_return)`
- Score final: `(mean_return / std_return) * volatility_penalty`
- Límites de posición: [-6.0, +6.0]

## 🚀 Solución Implementada

### Notebooks Principales

1. **`hull_tactical_automl_notebook.ipynb`** - Notebook completo con:
   - Análisis exploratorio de datos (EDA)
   - Ingeniería de características avanzada
   - Múltiples modelos AutoML (AutoGluon, FLAML, LightGBM, XGBoost, CatBoost)
   - Ensemble y optimización
   - Validación temporal

2. **`hull_tactical_submission.ipynb`** - Notebook optimizado para submission:
   - Código limpio y eficiente
   - Ensemble de los mejores modelos
   - Función de predicción lista para Kaggle

### Características Técnicas Implementadas

#### 🔧 Ingeniería de Características
- **Lags**: 1, 2, 3, 5, 10 períodos
- **Rolling Statistics**: Media móvil, desviación estándar, min/max
- **Ratios**: Entre diferentes características
- **Momentum**: Rate of Change (ROC), RSI aproximado
- **Volatilidad**: Volatilidad realizada en múltiples ventanas
- **Temporales**: Características cíclicas (día del año, sin/cos)
- **Target Lags**: Lags históricos del target (evitando data leakage)

#### 🤖 Modelos AutoML Utilizados
- **LightGBM**: Optimizado para series temporales
- **XGBoost**: Robusto y estable
- **CatBoost**: Manejo automático de características
- **AutoGluon**: AutoML completo (si disponible)
- **FLAML**: AutoML rápido y eficiente
- **Ensemble**: Combinación ponderada de los mejores modelos

#### ✅ Validación y Testing
- **TimeSeriesSplit**: Para evitar data leakage temporal
- **Split 80/20**: Validación final temporal
- **Cross-Validation**: 5 folds con validación temporal
- **Métricas**: Adjusted Sharpe Ratio, R², MSE, MAE

## 📈 Resultados Esperados

Basado en la investigación de notebooks públicos y la implementación:
- **Score objetivo**: > 0.35 (basado en mejores submissions públicas)
- **Validación cruzada**: Consistencia entre folds
- **Robustez**: Manejo de datos faltantes y casos edge

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install pandas numpy scikit-learn
pip install lightgbm xgboost catboost
pip install autogluon flaml optuna
pip install matplotlib seaborn plotly
pip install ta yfinance
```

### Ejecución Local
1. Clonar el repositorio
2. Instalar dependencias
3. Ejecutar `hull_tactical_automl_notebook.ipynb` para desarrollo completo
4. Usar `hull_tactical_submission.ipynb` para submission

### Ejecución en Kaggle
1. Subir `hull_tactical_submission.ipynb` como notebook de Kaggle
2. Asegurar que los datos de la competencia estén disponibles
3. Ejecutar el notebook completo
4. La función `predict()` será llamada automáticamente por la API

## 🏆 Estrategia de Ensemble

### Selección de Modelos
- Top 3 modelos por performance en validación
- Pesos basados en Adjusted Sharpe Ratio individual
- Combinación ponderada de predicciones

### Optimización
- Hiperparámetros optimizados para cada modelo
- Selección de características con importancia
- Validación temporal estricta

## 📊 Análisis de Características

### Top Características Identificadas
1. **Lags recientes** (1-3 períodos)
2. **Rolling means** (5, 10, 20 períodos)
3. **Ratios entre features** principales
4. **Momentum indicators** (ROC)
5. **Características temporales** cíclicas

### Selección Automática
- Random Forest feature importance
- F-test regression
- Mutual Information
- LightGBM feature importance

## 🔄 Pipeline de Predicción

```python
def predict(test_df):
    # 1. Ingeniería de características
    test_enhanced = create_features(test_df)
    
    # 2. Selección de características
    test_features = test_enhanced[selected_features]
    
    # 3. Manejo de valores faltantes
    test_features = handle_missing_values(test_features)
    
    # 4. Predicción del ensemble
    predictions = ensemble_predict(test_features)
    
    # 5. Aplicar límites
    predictions = np.clip(predictions, -6.0, 6.0)
    
    return predictions
```

## 📝 Próximas Mejoras

### Corto Plazo
- [ ] Optimización de hiperparámetros con Optuna
- [ ] Más características de análisis técnico
- [ ] Ensemble más sofisticado (stacking)

### Largo Plazo
- [ ] Modelos de deep learning (LSTM, Transformer)
- [ ] Análisis de régimen de mercado
- [ ] Features de sentiment y noticias
- [ ] Optimización multi-objetivo

## 📚 Referencias

- [Competencia Hull Tactical](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [Notebooks públicos de referencia](https://www.kaggle.com/competitions/hull-tactical-market-prediction/code)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [FLAML Documentation](https://microsoft.github.io/FLAML/)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

**Nota**: Esta solución está diseñada para la competencia Hull Tactical Market Prediction. Los resultados pueden variar según los datos y el entorno de ejecución.