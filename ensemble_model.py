"""
Ensemble Sampling Model for Predictive Class Imbalance Classification
This module trains multiple ML models and creates an ensemble of the best 2.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class EnsembleSamplingModel:
    """
    A class to handle training multiple models and creating an ensemble
    of the best 2 models for imbalanced classification.
    """
    
    def __init__(self, sampling_strategy='smote', random_state=42):
        """
        Initialize the ensemble model.
        
        Parameters:
        -----------
        sampling_strategy : str, default='smote'
            Sampling strategy to handle class imbalance.
            Options: 'smote', 'adasyn', 'undersample', 'smoteenn', 'smotetomek', 'none'
        random_state : int, default=42
            Random state for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.best_models = None
        self.ensemble = None
        self.sampler = self._get_sampler()
        
    def _get_sampler(self):
        """Get the appropriate sampler based on strategy."""
        if self.sampling_strategy == 'smote':
            return SMOTE(random_state=self.random_state)
        elif self.sampling_strategy == 'adasyn':
            return ADASYN(random_state=self.random_state)
        elif self.sampling_strategy == 'undersample':
            return RandomUnderSampler(random_state=self.random_state)
        elif self.sampling_strategy == 'smoteenn':
            return SMOTEENN(random_state=self.random_state)
        elif self.sampling_strategy == 'smotetomek':
            return SMOTETomek(random_state=self.random_state)
        else:
            return None
    
    def _initialize_models(self):
        """Initialize multiple ML models for training."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=self.random_state
            )
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models and select the best 2 for ensemble.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels
        """
        self._initialize_models()
        
        # Apply sampling if specified
        if self.sampler is not None:
            print(f"Applying {self.sampling_strategy} sampling...")
            X_train_resampled, y_train_resampled = self.sampler.fit_resample(X_train, y_train)
            print(f"Original dataset shape: {X_train.shape}")
            print(f"Resampled dataset shape: {X_train_resampled.shape}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # If validation set not provided, create one
        if X_val is None or y_val is None:
            X_train_resampled, X_val, y_train_resampled, y_val = train_test_split(
                X_train_resampled, y_train_resampled,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train_resampled
            )
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            
            # Calculate multiple metrics
            f1 = f1_score(y_val, y_pred, average='weighted')
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            # Use balanced accuracy as primary metric for imbalanced data
            score = balanced_acc
            self.model_scores[name] = {
                'balanced_accuracy': balanced_acc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'score': score
            }
            
            print(f"{name} - Balanced Accuracy: {balanced_acc:.4f}, F1: {f1:.4f}")
        
        # Select best 2 models based on balanced accuracy
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        self.best_models = [(name, self.models[name]) for name, _ in sorted_models[:2]]
        
        print("\n" + "="*60)
        print("BEST 2 MODELS SELECTED FOR ENSEMBLE:")
        print("="*60)
        for i, (name, _) in enumerate(self.best_models, 1):
            scores = self.model_scores[name]
            print(f"\n{i}. {name}")
            print(f"   Balanced Accuracy: {scores['balanced_accuracy']:.4f}")
            print(f"   F1 Score: {scores['f1_score']:.4f}")
            print(f"   Precision: {scores['precision']:.4f}")
            print(f"   Recall: {scores['recall']:.4f}")
        
        # Create ensemble using voting classifier
        print("\n" + "="*60)
        print("Creating Ensemble Model (Soft Voting)...")
        print("="*60)
        
        self.ensemble = VotingClassifier(
            estimators=self.best_models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble on resampled training data only (not validation)
        self.ensemble.fit(X_train_resampled, y_train_resampled)
        print("Ensemble model trained successfully!")
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        if self.ensemble is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities using the ensemble model."""
        if self.ensemble is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.ensemble.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the ensemble model on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        
        Returns:
        --------
        dict : Dictionary containing evaluation metrics
        """
        if self.ensemble is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("\n" + "="*60)
        print("ENSEMBLE MODEL EVALUATION")
        print("="*60)
        
        # Calculate metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Try to calculate ROC AUC if binary classification
        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                print(f"ROC AUC Score: {roc_auc:.4f}")
        except (ValueError, IndexError) as e:
            print(f"ROC AUC could not be calculated: {str(e)}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def get_model_rankings(self):
        """Get rankings of all trained models."""
        if not self.model_scores:
            raise ValueError("No models trained yet. Call fit() first.")
        
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        print("\n" + "="*60)
        print("ALL MODELS RANKING (by Balanced Accuracy)")
        print("="*60)
        
        for i, (name, scores) in enumerate(sorted_models, 1):
            print(f"\n{i}. {name}")
            print(f"   Balanced Accuracy: {scores['balanced_accuracy']:.4f}")
            print(f"   F1 Score: {scores['f1_score']:.4f}")
            print(f"   Precision: {scores['precision']:.4f}")
            print(f"   Recall: {scores['recall']:.4f}")
        
        return sorted_models
