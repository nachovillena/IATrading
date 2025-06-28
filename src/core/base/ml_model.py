"""Base ML Model class for trading predictions"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
from ...utils.logger import Logger


class BaseMLModel(ABC):
    """Base class for ML models used in trading predictions"""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize base ML model
        
        Args:
            model_config: Configuration dictionary for the model
        """
        self.config = model_config or {}
        class_name = self.__class__.__name__
        self.logger = Logger(
            name=class_name,
            to_console=True,
            to_file=f"{class_name}.log"
        )
        self.name = class_name
        self.model_name = self.__class__.__name__
        
        # Model state
        self.is_trained = False
        self.feature_names: Optional[list] = None
        self.target_columns: Optional[list] = None
        
        # Model metadata
        self.training_metadata: Dict[str, Any] = {}
        self.model_version = "1.0.0"
        
        # Performance metrics
        self.training_score: Optional[float] = None
        self.validation_score: Optional[float] = None
        
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, **kwargs) -> None:
        """Train the model with features and targets
        
        Args:
            features: Input features DataFrame
            targets: Target values Series (-1, 0, 1 for sell, hold, buy)
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions on new data
        
        Args:
            features: Input features DataFrame
            
        Returns:
            Predictions Series with values -1, 0, 1 (sell, hold, buy)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load a trained model from disk
        
        Args:
            path: Path to the saved model
        """
        pass
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities (optional implementation)
        
        Args:
            features: Input features DataFrame
            
        Returns:
            DataFrame with probabilities for each class
        """
        # Default implementation - can be overridden by subclasses
        predictions = self.predict(features)
        
        # Convert deterministic predictions to probabilities
        proba_data = []
        for pred in predictions:
            if pred == -1:  # Sell
                proba_data.append([0.8, 0.1, 0.1])
            elif pred == 0:  # Hold
                proba_data.append([0.1, 0.8, 0.1])
            else:  # Buy
                proba_data.append([0.1, 0.1, 0.8])
        
        return pd.DataFrame(
            proba_data,
            index=features.index,
            columns=['sell_prob', 'hold_prob', 'buy_prob']
        )
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance (optional implementation)
        
        Returns:
            Series with feature importance scores, or None if not available
        """
        self.logger.warning(f"{self.model_name} does not implement feature importance")
        return None
    
    def validate_features(self, features: pd.DataFrame) -> None:
        """Validate input features
        
        Args:
            features: Features to validate
            
        Raises:
            ValueError: If features are invalid
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            extra_features = set(features.columns) - set(self.feature_names)
            if extra_features:
                self.logger.warning(f"Extra features will be ignored: {extra_features}")
    
    def validate_targets(self, targets: pd.Series) -> None:
        """Validate target values
        
        Args:
            targets: Target values to validate
            
        Raises:
            ValueError: If targets are invalid
        """
        if targets.empty:
            raise ValueError("Targets Series is empty")
        
        valid_targets = {-1, 0, 1}
        invalid_targets = set(targets.unique()) - valid_targets
        if invalid_targets:
            raise ValueError(f"Invalid target values: {invalid_targets}. Must be -1, 0, or 1")
    
    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features before training/prediction
        
        Args:
            features: Raw features DataFrame
            
        Returns:
            Preprocessed features DataFrame
        """
        # Default implementation - can be overridden
        processed_features = features.copy()
        
        # Handle infinite values
        processed_features = processed_features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill, then backward fill
        processed_features = processed_features.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with 0
        processed_features = processed_features.fillna(0)
        
        return processed_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_score': self.training_score,
            'validation_score': self.validation_score,
            'training_metadata': self.training_metadata,
            'config': self.config
        }
    
    def set_feature_names(self, feature_names: list) -> None:
        """Set feature names for the model
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = list(feature_names)
        self.logger.info(f"Set {len(feature_names)} feature names")
    
    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            features: Test features
            targets: True targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(features)
        
        # Calculate basic metrics
        accuracy = (predictions == targets).mean()
        
        # Calculate per-class metrics
        metrics = {'accuracy': accuracy}
        
        for signal_value, signal_name in [(-1, 'sell'), (0, 'hold'), (1, 'buy')]:
            mask = targets == signal_value
            if mask.sum() > 0:
                precision = (predictions[mask] == signal_value).mean()
                metrics[f'{signal_name}_precision'] = precision
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of the model"""
        status = "Trained" if self.is_trained else "Not Trained"
        feature_count = len(self.feature_names) if self.feature_names else 0
        return f"{self.model_name} ({status}, {feature_count} features)"
    
    def __repr__(self) -> str:
        """Detailed representation of the model"""
        return f"{self.__class__.__name__}(config={self.config}, is_trained={self.is_trained})"