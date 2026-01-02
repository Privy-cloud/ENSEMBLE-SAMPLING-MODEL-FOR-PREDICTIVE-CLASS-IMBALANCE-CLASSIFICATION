"""
Ensemble Sampling Model for Predictive Class Imbalance Classification
Using Bank Marketing Dataset from UCI Repository

This package contains modules for loading data, applying sampling techniques,
and training machine learning models for imbalanced classification.
"""

__version__ = "1.0.0"
__author__ = "Capstone Project Team"

from .data_loader import BankMarketingDataLoader
from .ensemble_sampler import EnsembleSampler, prepare_train_test_split
from .model_trainer import ModelTrainer

__all__ = [
    'BankMarketingDataLoader',
    'EnsembleSampler',
    'prepare_train_test_split',
    'ModelTrainer'
]
