"""
Module for machine learning models used in the WINFUT trading robot.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

from config import ML_PARAMS

logger = logging.getLogger(__name__)

class ModelManager:
    """Class to train, evaluate, and manage machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.test_size = 1 - ML_PARAMS["train_test_split"]
        self.random_state = 42
        self.prediction_horizon = ML_PARAMS["prediction_horizon"]
        self.rf_params = ML_PARAMS["random_forest"]
        self.xgb_params = ML_PARAMS["xgboost"]
        
    def train_random_forest(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Trains a Random Forest model.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dict with model and performance metrics
        """
        # Use proper time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=self.rf_params["n_estimators"],
            max_depth=self.rf_params["max_depth"],
            min_samples_split=self.rf_params["min_samples_split"],
            min_samples_leaf=self.rf_params["min_samples_leaf"],
            random_state=self.random_state
        )
        
        # Time series cross-validation
        cv_scores = cross_val_score(rf_model, features, target, cv=tscv, scoring='accuracy')
        
        # Train on the full dataset
        rf_model.fit(features, target)
        
        # Get feature importances
        feature_importances = dict(zip(features.columns, rf_model.feature_importances_))
        
        # Sort feature importances
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_importances[:15])
        
        return {
            "model": rf_model,
            "name": "random_forest",
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "feature_importances": feature_importances,
            "top_features": top_features
        }
        
    def train_xgboost(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Trains an XGBoost model.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dict with model and performance metrics
        """
        # Use proper time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.xgb_params["n_estimators"],
            max_depth=self.xgb_params["max_depth"],
            learning_rate=self.xgb_params["learning_rate"],
            subsample=self.xgb_params["subsample"],
            colsample_bytree=self.xgb_params["colsample_bytree"],
            random_state=self.random_state
        )
        
        # Time series cross-validation
        cv_scores = cross_val_score(xgb_model, features, target, cv=tscv, scoring='accuracy')
        
        # Train on the full dataset
        xgb_model.fit(features, target)
        
        # Get feature importances
        feature_importances = dict(zip(features.columns, xgb_model.feature_importances_))
        
        # Sort feature importances
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_importances[:15])
        
        return {
            "model": xgb_model,
            "name": "xgboost",
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "feature_importances": feature_importances,
            "top_features": top_features
        }
    
    def train_models(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Trains all models.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dict of model results
        """
        try:
            logger.info("Training Random Forest model...")
            rf_results = self.train_random_forest(features, target)
            self.models["random_forest"] = rf_results
            
            logger.info("Training XGBoost model...")
            xgb_results = self.train_xgboost(features, target)
            self.models["xgboost"] = xgb_results
            
            logger.info("Model training completed successfully")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return {}
    
    def evaluate_models(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluates trained models on test data.
        
        Args:
            features: Test feature DataFrame
            target: Test target Series
            
        Returns:
            Dict of model evaluation metrics
        """
        evaluation = {}
        
        for model_name, model_data in self.models.items():
            model = model_data["model"]
            predictions = model.predict(features)
            
            metrics = {
                "accuracy": accuracy_score(target, predictions),
                "precision": precision_score(target, predictions),
                "recall": recall_score(target, predictions),
                "f1_score": f1_score(target, predictions)
            }
            
            evaluation[model_name] = metrics
            
            logger.info(f"Evaluation metrics for {model_name}:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            
        return evaluation
    
    def predict(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Makes predictions using all trained models.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dict of model predictions
        """
        predictions = {}
        probabilities = {}
        
        for model_name, model_data in self.models.items():
            model = model_data["model"]
            predictions[model_name] = model.predict(features)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                # Get probability of positive class (index 1)
                probabilities[model_name] = model.predict_proba(features)[:, 1]
            else:
                probabilities[model_name] = predictions[model_name].astype(float)
                
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def ensemble_predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes weighted ensemble predictions from all models.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.models:
            logger.error("No trained models available for prediction")
            return np.array([]), np.array([])
        
        all_predictions = self.predict(features)
        probabilities = all_predictions["probabilities"]
        
        # Get ensemble weights from config
        weights = ML_PARAMS["ensemble_weights"]
        
        # Initialize weighted predictions
        weighted_probs = np.zeros(len(features))
        total_weight = 0
        
        # Calculate weighted probabilities
        for model_name, weight in weights.items():
            if model_name in probabilities:
                weighted_probs += weight * probabilities[model_name]
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Convert to binary predictions
        ensemble_predictions = (weighted_probs >= 0.5).astype(int)
        
        return ensemble_predictions, weighted_probs
    
    def save_models(self, directory: str = "models") -> bool:
        """
        Saves trained models to disk.
        
        Args:
            directory: Directory to save models
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            for model_name, model_data in self.models.items():
                model = model_data["model"]
                model_path = os.path.join(directory, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved model {model_name} to {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, directory: str = "models") -> bool:
        """
        Loads trained models from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Boolean indicating success
        """
        try:
            for model_name in ["random_forest", "xgboost"]:
                model_path = os.path.join(directory, f"{model_name}.joblib")
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    self.models[model_name] = {"model": model, "name": model_name}
                    logger.info(f"Loaded model {model_name} from {model_path}")
                else:
                    logger.warning(f"Model file {model_path} not found")
            
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
