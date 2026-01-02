"""
Main script to demonstrate the Ensemble Sampling Model for Class Imbalance Classification.
This script trains multiple ML models and creates an ensemble of the best 2.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ensemble_model import EnsembleSamplingModel


def create_imbalanced_dataset(n_samples=10000, n_features=20, weights=(0.9, 0.1)):
    """
    Create an imbalanced binary classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    weights : tuple
        Class distribution weights
    
    Returns:
    --------
    X : array
        Features
    y : array
        Labels
    """
    print("Creating imbalanced dataset...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=list(weights),
        flip_y=0.01,
        random_state=42
    )
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"Dataset created with {n_samples} samples and {n_features} features")
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class imbalance ratio: {counts[0]/counts[1]:.2f}:1")
    
    return X, y


def load_or_create_dataset():
    """
    Load dataset from file or create a synthetic one.
    
    Returns:
    --------
    X : array
        Features
    y : array
        Labels
    """
    try:
        # Try to load from CSV if available
        print("Attempting to load dataset from 'data.csv'...")
        data = pd.read_csv('data.csv')
        
        # Assume last column is the target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
    except FileNotFoundError:
        print("No 'data.csv' found. Creating synthetic imbalanced dataset...\n")
        X, y = create_imbalanced_dataset(
            n_samples=10000,
            n_features=20,
            weights=(0.9, 0.1)  # 90% class 0, 10% class 1
        )
    
    return X, y


def main():
    """Main function to run the ensemble model training and evaluation."""
    
    # Configurable sampling strategies to test
    SAMPLING_STRATEGIES = ['smote', 'smotetomek', 'none']
    
    print("="*60)
    print("ENSEMBLE SAMPLING MODEL FOR CLASS IMBALANCE CLASSIFICATION")
    print("="*60)
    print()
    
    # Load or create dataset
    X, y = load_or_create_dataset()
    
    # Split data into train and test sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different sampling strategies
    best_strategy = None
    best_score = 0
    best_model = None
    
    for strategy in SAMPLING_STRATEGIES:
        print("\n" + "="*60)
        print(f"TESTING SAMPLING STRATEGY: {strategy.upper()}")
        print("="*60)
        
        # Initialize and train the ensemble model
        ensemble_model = EnsembleSamplingModel(
            sampling_strategy=strategy,
            random_state=42
        )
        
        # Train the model (it will automatically select best 2 models)
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Get model rankings
        ensemble_model.get_model_rankings()
        
        # Evaluate the ensemble on test set
        results = ensemble_model.evaluate(X_test_scaled, y_test)
        
        # Track best strategy
        if results['balanced_accuracy'] > best_score:
            best_score = results['balanced_accuracy']
            best_strategy = strategy
            best_model = ensemble_model
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nBest Sampling Strategy: {best_strategy.upper()}")
    print(f"Best Balanced Accuracy: {best_score:.4f}")
    print(f"\nBest 2 Models in Ensemble:")
    for i, (name, _) in enumerate(best_model.best_models, 1):
        print(f"{i}. {name}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nThe ensemble model has been successfully trained using the")
    print("BEST 2 performing models and is ready for predictions.")
    print("="*60)


if __name__ == "__main__":
    main()
