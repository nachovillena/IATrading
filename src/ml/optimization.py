"""Hyperparameter optimization for ML models"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from ..core.types.trading_data import TradingData  

from .models import MLModelManager
from .features import FeatureEngineer
from ..core.exceptions import MLModelError
from ..utils.logger import Logger

class HyperparameterOptimizer:
    """Optimizes ML model or trading strategy hyperparameters using Optuna"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.model_manager = MLModelManager()
        self.feature_engineer = FeatureEngineer()
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'n_trials': 50,
            'cv_folds': 5,
            'random_state': 42,
            'direction': 'maximize'
        }
    
    def optimize_model(self, features: pd.DataFrame, targets: pd.Series, 
                      model_type: str = 'logistic') -> Dict[str, Any]:
        """Optimize hyperparameters for a model"""
        
        print(f"🔍 Optimizing {model_type} model hyperparameters")
        
        if features.empty or targets.empty:
            raise MLModelError("Empty features or targets provided")
        
        # Prepare data
        targets_encoded = targets + 1  # Convert -1,0,1 to 0,1,2
        
        # Define objective function
        def objective(trial):
            # Get hyperparameters based on model type
            if model_type == 'logistic':
                params = self._suggest_logistic_params(trial)
            elif model_type == 'random_forest':
                params = self._suggest_rf_params(trial)
            else:
                raise MLModelError(f"Unsupported model type: {model_type}")
            
            try:
                # Create model with suggested parameters
                model = self.model_manager.create_model(model_type, **params)
                
                # Perform cross-validation
                cv = StratifiedKFold(
                    n_splits=self.config['cv_folds'], 
                    shuffle=True, 
                    random_state=self.config['random_state']
                )
                
                scores = cross_val_score(
                    model.model, 
                    features, 
                    targets_encoded, 
                    cv=cv, 
                    scoring='accuracy'
                )
                
                return scores.mean()
                
            except Exception as e:
                print(f"    ❌ Trial failed: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction=self.config['direction'])
        study.optimize(objective, n_trials=self.config['n_trials'])
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"✅ Optimization completed - Best CV score: {best_score:.3f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _suggest_logistic_params(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Logistic Regression"""
        return {
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000, step=100),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
    
    def _suggest_rf_params(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
    
    def optimize_feature_selection(self, features: pd.DataFrame, targets: pd.Series,
                                 model_type: str = 'logistic') -> Dict[str, Any]:
        """Optimize feature selection along with hyperparameters"""
        
        print(f"🔍 Optimizing feature selection for {model_type}")
        
        # Get feature importance first
        feature_importance = self.feature_engineer.get_feature_importance(features, targets)
        all_features = list(feature_importance.index)
        
        targets_encoded = targets + 1
        
        def objective(trial):
            # Suggest number of features to select
            n_features = trial.suggest_int('n_features', 5, min(50, len(all_features)))
            
            # Select top features
            selected_features = all_features[:n_features]
            features_selected = features[selected_features]
            
            # Suggest model hyperparameters
            if model_type == 'logistic':
                model_params = self._suggest_logistic_params(trial)
            else:
                model_params = self._suggest_rf_params(trial)
            
            try:
                model = self.model_manager.create_model(model_type, **model_params)
                
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model.model, 
                    features_selected, 
                    targets_encoded, 
                    cv=cv, 
                    scoring='accuracy'
                )
                
                return scores.mean()
                
            except Exception:
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['n_trials'])
        
        # Extract best feature selection
        best_n_features = study.best_params['n_features']
        best_features = all_features[:best_n_features]
        
        # Remove feature selection params to get model params
        best_model_params = {k: v for k, v in study.best_params.items() 
                           if k != 'n_features'}
        
        return {
            'best_features': best_features,
            'best_model_params': best_model_params,
            'best_score': study.best_value,
            'n_features': best_n_features
        }

    def optimize(
        self,
        strategy: str,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        trials: int = 100,
        score_metric: str = "total_return",  # Puedes cambiar a "profit_factor", "win_rate", etc.
        min_trades: int = 10
    ) -> Dict[str, Any]:
        """
        Optimiza los hiperparámetros de cualquier estrategia de trading usando contexto multi-timeframe.
        - strategy: nombre de la estrategia
        - data: dict {timeframe: dataframe} con los históricos requeridos
        - trials: número de pruebas de Optuna
        - score_metric: métrica objetivo para optimizar
        - min_trades: mínimo de trades para considerar una solución válida
        """
        import optuna

        self.logger.info(f"🔍 Optimizando estrategia {strategy} con contexto: {list(data.keys())}")

        from src.strategies.manager import StrategyManager
        strategy_manager = StrategyManager()
        config = strategy_manager.load_strategy_config(strategy)
        param_space = config.get("param_space", {})
        default_params = config.get("default_params", {})

        trial_results = {}

        def objective(trial):
            params = {}
            for param, bounds in param_space.items():
                if (
                    isinstance(bounds, list)
                    and len(bounds) == 2
                    and all(isinstance(x, (int, float)) for x in bounds)
                    and not any(isinstance(x, bool) for x in bounds)
                ):
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        params[param] = trial.suggest_int(param, bounds[0], bounds[1])
                    else:
                        params[param] = trial.suggest_float(param, bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params[param] = trial.suggest_categorical(param, bounds)
                else:
                    params[param] = bounds

            # Ejemplo de filtro específico (puedes añadir más si lo necesitas)
            if 'ema_fast' in params and 'ema_slow' in params:
                if params['ema_fast'] >= params['ema_slow']:
                    return -1

            # Instancia la estrategia
            strategy_obj = strategy_manager.strategies.get(strategy)
            if strategy_obj is None:
                strategy_obj = strategy_manager._load_strategy_class(strategy)
            strategy_class = strategy_obj if isinstance(strategy_obj, type) else type(strategy_obj)

            config_for_trial = config.copy()
            config_for_trial.update(config.get("default_params", {}))
            config_for_trial.update(params)

            strategy_instance = strategy_class(config_for_trial)
            if not hasattr(strategy_instance, "main_timeframe"):
                strategy_instance.main_timeframe = config_for_trial.get("main_timeframe")

            main_tf = strategy_instance.main_timeframe
            df = data[main_tf]
            if df.index.has_duplicates:
                df = df[~df.index.duplicated(keep='first')]
            main_data = TradingData(
                symbol=symbol,
                timeframe=main_tf,
                data=df,
                provider="cache",
                timestamp=pd.Timestamp.now()
            )

            context_timeframes = [tf for tf in data.keys() if tf.strip().lower() != main_tf.strip().lower()]
            context_data = {}
            for tf in context_timeframes:
                df = data[tf]
                if df.index.has_duplicates:
                    df = df[~df.index.duplicated(keep='first')]
                context_data[tf] = TradingData(
                    symbol=symbol,
                    timeframe=tf,
                    data=df,
                    provider="cache",
                    timestamp=pd.Timestamp.now()
                )
            try:
                result = strategy_instance.backtest_simple(main_data, context_data=context_data)
                self.logger.info(f"    ✅ Trial succeeded: {result['num_trades']} trades, {result['total_return']:.2f} total return")
                score = result.get(score_metric, 0.0)
                trial_results[trial.number] = result
            except Exception as e:
                self.logger.error(f"    ❌ Trial failed: {e}")
                trial_results[trial.number] = None
                return 0.0

            if result.get('num_trades', 0) < min_trades:
                return -1

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials, n_jobs=-1)

        best_params = study.best_params
        best_score = study.best_value
        best_trial_number = study.best_trial.number
        best_result = trial_results.get(best_trial_number, {})

        # Devuelve todas las métricas del mejor trial
        self.logger.info(f"✅ Optimización completada - Mejor {score_metric}: {best_score:.3f}")
        self.logger.info(f"    Mejor params: {best_params}")
        for k, v in best_result.items():
            if k not in ("trade_profits", "equity_curve"):
                self.logger.info(f"    {k}: {v}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'best_metrics': best_result,
            'study': study
        }
