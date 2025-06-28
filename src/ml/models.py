"""ML Models for trading predictions"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import joblib
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ..core.base import BaseMLModel
from ..core.exceptions import MLModelError

class MLModelManager:
    """Manages ML models for trading"""
    
    def __init__(self):
        self.models = {}
        self.model_dir = Path("modelos")
        self.model_dir.mkdir(exist_ok=True)
        
        # Register default models
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default ML models"""
        self.models['logistic'] = LogisticRegressionModel
        self.models['random_forest'] = RandomForestModel
    
    def create_model(self, model_type: str, **kwargs) -> BaseMLModel:
        """Create ML model instance"""
        if model_type not in self.models:
            raise MLModelError(f"Unknown model type: {model_type}")
        
        return self.models[model_type](**kwargs)
    
    def get_available_models(self) -> list:
        """Get list of available model types"""
        return list(self.models.keys())

class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression model for trading signals"""
    
    def __init__(self, **kwargs):
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **kwargs
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the logistic regression model"""
        
        if features.empty or targets.empty:
            raise MLModelError("Empty features or targets provided")
        
        # Store feature names for later use
        self.feature_names = list(features.columns)
        
        # Convert targets to classification labels
        # -1, 0, 1 -> 0, 1, 2 for sklearn
        targets_encoded = targets + 1
        
        try:
            self.model.fit(features, targets_encoded)
            self.is_trained = True
            
            # Calculate training score
            train_score = self.model.score(features, targets_encoded)
            print(f"✅ Model trained with accuracy: {train_score:.3f}")
            
        except Exception as e:
            raise MLModelError(f"Training failed: {e}")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        
        if not self.is_trained:
            raise MLModelError("Model not trained")
        
        if features.empty:
            return pd.Series(dtype=int)
        
        # Ensure feature order matches training
        if self.feature_names:
            features = features[self.feature_names]
        
        try:
            # Get predictions and convert back to -1, 0, 1
            predictions = self.model.predict(features)
            predictions_decoded = predictions - 1
            
            return pd.Series(predictions_decoded, index=features.index)
            
        except Exception as e:
            raise MLModelError(f"Prediction failed: {e}")
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise MLModelError("Model not trained")
        
        if self.feature_names:
            features = features[self.feature_names]
        
        try:
            probabilities = self.model.predict_proba(features)
            return pd.DataFrame(
                probabilities, 
                index=features.index,
                columns=['sell_prob', 'hold_prob', 'buy_prob']
            )
        except Exception as e:
            raise MLModelError(f"Probability prediction failed: {e}")
    
    def save(self, path: str) -> None:
        """Save the model"""
        if not self.is_trained:
            raise MLModelError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': 'logistic_regression'
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load the model"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
        except Exception as e:
            raise MLModelError(f"Model loading failed: {e}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (coefficients for logistic regression)"""
        if not self.is_trained:
            raise MLModelError("Model not trained")
        
        if self.feature_names is None:
            raise MLModelError("Feature names not available")
        
        # For multiclass, take mean of absolute coefficients
        coefs = np.abs(self.model.coef_).mean(axis=0)
        
        return pd.Series(coefs, index=self.feature_names).sort_values(ascending=False)

class RandomForestModel(BaseMLModel):
    """Random Forest model for trading signals"""
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            **kwargs
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the random forest model"""
        
        if features.empty or targets.empty:
            raise MLModelError("Empty features or targets provided")
        
        self.feature_names = list(features.columns)
        
        # Convert targets to classification labels
        targets_encoded = targets + 1
        
        try:
            self.model.fit(features, targets_encoded)
            self.is_trained = True
            
            # Calculate training score
            train_score = self.model.score(features, targets_encoded)
            print(f"✅ Random Forest trained with accuracy: {train_score:.3f}")
            
        except Exception as e:
            raise MLModelError(f"Training failed: {e}")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        
        if not self.is_trained:
            raise MLModelError("Model not trained")
        
        if features.empty:
            return pd.Series(dtype=int)
        
        if self.feature_names:
            features = features[self.feature_names]
        
        try:
            predictions = self.model.predict(features)
            predictions_decoded = predictions - 1
            
            return pd.Series(predictions_decoded, index=features.index)
            
        except Exception as e:
            raise MLModelError(f"Prediction failed: {e}")
    
    def save(self, path: str) -> None:
        """Save the model"""
        if not self.is_trained:
            raise MLModelError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': 'random_forest'
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load the model"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
        except Exception as e:
            raise MLModelError(f"Model loading failed: {e}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            raise MLModelError("Model not trained")
        
        if self.feature_names is None:
            raise MLModelError("Feature names not available")
        
        importance = self.model.feature_importances_
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
