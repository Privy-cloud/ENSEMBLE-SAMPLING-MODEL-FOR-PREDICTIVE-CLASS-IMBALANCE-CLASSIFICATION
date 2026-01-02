"""
Model Training and Evaluation for Imbalanced Classification
Implements various classifiers and evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)
import joblib
import os


class ModelTrainer:
    """
    Train and evaluate machine learning models for imbalanced classification.
    """
    
    def __init__(self, random_state: int = 42, model_dir: str = 'models'):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
            model_dir: Directory to save trained models
        """
        self.random_state = random_state
        self.model_dir = model_dir
        self.models = {}
        self.results = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different classifier models."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(self.models.keys())}"
            )
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        print(f"{model_name} training completed.")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                # Train model
                model = self.train_model(model_name, X_train, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                results[model_name] = metrics
                
                # Save model
                self.save_model(model, model_name)
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def save_model(self, model: Any, model_name: str):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name for the saved model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model results.
        
        Returns:
            DataFrame with results summary
        """
        if not self.results:
            print("No results available. Train models first.")
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                row = {
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'MCC': metrics['mcc']
                }
                if 'roc_auc' in metrics:
                    row['ROC-AUC'] = metrics['roc_auc']
                summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        if not df_summary.empty:
            df_summary = df_summary.sort_values('F1-Score', ascending=False)
        
        return df_summary


if __name__ == "__main__":
    # Example usage
    from data_loader import BankMarketingDataLoader
    from ensemble_sampler import EnsembleSampler, prepare_train_test_split
    
    # Load data
    loader = BankMarketingDataLoader()
    df = loader.load_data(sample_size=5000)
    X, y = loader.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # Apply SMOTE sampling
    sampler = EnsembleSampler()
    X_train_resampled, y_train_resampled = sampler.apply_sampling(
        X_train, y_train, strategy='smote'
    )
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(
        X_train_resampled, y_train_resampled, X_test, y_test
    )
    
    # Display summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    summary = trainer.get_results_summary()
    print(summary.to_string(index=False))
