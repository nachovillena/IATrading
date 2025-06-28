"""Model training pipeline"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .models import MLModelManager
from .features import FeatureEngineer
from ..core.exceptions import MLModelError

class ModelTrainer:
    """Handles ML model training pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.model_manager = MLModelManager()
        self.feature_engineer = FeatureEngineer(config)
        
        # Directories
        self.model_dir = Path("modelos")
        self.scaler_dir = Path("scaler")
        self.model_dir.mkdir(exist_ok=True)
        self.scaler_dir.mkdir(exist_ok=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'test_size': 0.2,
            'random_state': 42,
            'model_type': 'logistic',
            'validation_split': 0.2
        }
    
    def train_strategy_model(self, data: pd.DataFrame, strategy_name: str, 
                           strategy_params: Dict[str, Any], 
                           symbol: str = 'EURUSD') -> Dict[str, Any]:
        """Train ML model for a specific strategy"""
        
        print(f"🤖 Training model for {strategy_name} strategy")
        
        try:
            # Import strategy manager here to avoid circular imports
            from ..trading.strategies import StrategyManager
            
            # Create strategy and generate signals
            strategy_manager = StrategyManager()
            strategy = strategy_manager.create_strategy(strategy_name, strategy_params)
            
            # Calculate indicators
            indicators = strategy.calculate_indicators(data)
            
            # Generate signals (these will be our targets)
            signals = strategy.generate_signals(data, indicators)
            
            # Engineer features
            features = self.feature_engineer.create_features(data, indicators)
            
            # Align features and targets
            common_index = features.index.intersection(signals.index)
            features_aligned = features.loc[common_index]
            targets_aligned = signals.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = ~(features_aligned.isnull().any(axis=1) | targets_aligned.isnull())
            features_clean = features_aligned[valid_mask]
            targets_clean = targets_aligned[valid_mask]
            
            if len(features_clean) == 0:
                raise MLModelError("No valid data after cleaning")
            
            print(f"📊 Training data: {len(features_clean)} samples, {len(features_clean.columns)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, 
                targets_clean,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=targets_clean if len(targets_clean.unique()) > 1 else None
            )
            
            # Scale features
            X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)
            X_test_scaled = self.feature_engineer.scale_features(X_test, fit=False)
            
            # Create and train model
            model = self.model_manager.create_model(self.config['model_type'])
            model.train(X_train_scaled, y_train)
            
            # Evaluate model
            train_predictions = model.predict(X_train_scaled)
            test_predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            training_results = self._evaluate_model(
                y_train, train_predictions, y_test, test_predictions
            )
            
            # Save model and scaler
            model_filename = f"modelo_{strategy_name}_{symbol}.pkl"
            scaler_filename = f"scaler_{strategy_name}_{symbol}.pkl"
            
            model_path = self.model_dir / model_filename
            model.save(str(model_path))
            
            # Save scaler
            import joblib
            scaler_path = self.scaler_dir / scaler_filename
            joblib.dump(self.feature_engineer.scalers, str(scaler_path))
            
            # Feature importance
            feature_importance = model.get_feature_importance()
            
            training_results.update({
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'feature_importance': feature_importance.to_dict(),
                'strategy_name': strategy_name,
                'symbol': symbol,
                'num_features': len(features_clean.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            print(f"✅ Model training completed - Test accuracy: {training_results['test_accuracy']:.3f}")
            
            return training_results
            
        except Exception as e:
            raise MLModelError(f"Model training failed: {e}")
    
    def _evaluate_model(self, y_train: pd.Series, train_pred: pd.Series,
                       y_test: pd.Series, test_pred: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        # Calculate accuracies
        train_accuracy = (y_train == train_pred).mean()
        test_accuracy = (y_test == test_pred).mean()
        
        # Get class distribution
        class_distribution = y_train.value_counts().to_dict()
        
        # Classification report for test set
        try:
            test_report = classification_report(
                y_test, test_pred, 
                labels=[-1, 0, 1],
                target_names=['SELL', 'HOLD', 'BUY'],
                output_dict=True,
                zero_division=0
            )
        except Exception:
            test_report = {}
        
        return {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'class_distribution': class_distribution,
            'classification_report': test_report
        }
    
    def load_trained_model(self, strategy_name: str, symbol: str = 'EURUSD'):
        """Load a previously trained model"""
        
        model_filename = f"modelo_{strategy_name}_{symbol}.pkl"
        scaler_filename = f"scaler_{strategy_name}_{symbol}.pkl"
        
        model_path = self.model_dir / model_filename
        scaler_path = self.scaler_dir / scaler_filename
        
        if not model_path.exists():
            raise MLModelError(f"Model not found: {model_path}")
        
        # Load model
        model = self.model_manager.create_model(self.config['model_type'])
        model.load(str(model_path))
        
        # Load scaler if available
        if scaler_path.exists():
            import joblib
            self.feature_engineer.scalers = joblib.load(str(scaler_path))
        
        return model
    
    def predict_with_model(self, data: pd.DataFrame, strategy_name: str, 
                          strategy_params: Dict[str, Any], 
                          symbol: str = 'EURUSD') -> pd.Series:
        """Make predictions using trained model"""
        
        try:
            # Load trained model
            model = self.load_trained_model(strategy_name, symbol)
            
            # Import strategy manager
            from ..trading.strategies import StrategyManager
            
            # Create strategy and generate indicators
            strategy_manager = StrategyManager()
            strategy = strategy_manager.create_strategy(strategy_name, strategy_params)
            indicators = strategy.calculate_indicators(data)
            
            # Engineer features
            features = self.feature_engineer.create_features(data, indicators)
            
            # Scale features
            features_scaled = self.feature_engineer.scale_features(features, fit=False)
            
            # Make predictions
            predictions = model.predict(features_scaled)
            
            return predictions
            
        except Exception as e:
            raise MLModelError(f"Prediction failed: {e}")
    
    def retrain_model(self, strategy_name: str, symbol: str = 'EURUSD') -> Dict[str, Any]:
        """Retrain model with latest data"""
        
        try:
            # Get latest data
            from ..data.pipeline import DataPipeline
            from ..services.config_service import ConfigService
            
            config_service = ConfigService()
            data_pipeline = DataPipeline(config_service.get_data_config())
            
            trading_data = data_pipeline.fetch_data(symbol, 'H1')
            
            if not trading_data or trading_data.data.empty:
                raise MLModelError(f"No data available for retraining")
            
            # Get optimized parameters
            from ..services.optimization_service import OptimizationService
            opt_service = OptimizationService(config_service)
            strategy_params = opt_service.get_best_parameters(symbol, strategy_name)
            
            if not strategy_params:
                raise MLModelError(f"No optimized parameters found for {strategy_name}")
            
            # Retrain model
            results = self.train_strategy_model(
                trading_data.data, 
                strategy_name, 
                strategy_params, 
                symbol
            )
            
            return results
            
        except Exception as e:
            raise MLModelError(f"Model retraining failed: {e}")
