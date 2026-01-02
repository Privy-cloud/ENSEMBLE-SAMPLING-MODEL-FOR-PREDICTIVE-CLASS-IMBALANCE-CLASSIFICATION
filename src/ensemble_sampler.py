"""
Ensemble Sampling Models for Class Imbalance Classification
Implements various sampling techniques to handle class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import (
    SMOTE, 
    RandomOverSampler, 
    ADASYN,
    BorderlineSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    NearMiss
)
from imblearn.combine import SMOTEENN, SMOTETomek


class EnsembleSampler:
    """
    Ensemble sampling techniques for handling class imbalance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ensemble sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.sampling_strategies = {}
        self._initialize_samplers()
    
    def _initialize_samplers(self):
        """Initialize all sampling strategies."""
        self.sampling_strategies = {
            'original': None,
            'random_oversampling': RandomOverSampler(random_state=self.random_state),
            'random_undersampling': RandomUnderSampler(random_state=self.random_state),
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'tomek_links': TomekLinks(),
            'nearmiss': NearMiss(),
            'smote_enn': SMOTEENN(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state)
        }
    
    def apply_sampling(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        strategy: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply the specified sampling strategy.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            strategy: Sampling strategy name
            
        Returns:
            Tuple of (resampled features, resampled target)
        """
        if strategy not in self.sampling_strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available strategies: {list(self.sampling_strategies.keys())}"
            )
        
        print(f"\nApplying {strategy} sampling...")
        print(f"Original class distribution: {dict(y.value_counts())}")
        
        if strategy == 'original':
            return X, y
        
        sampler = self.sampling_strategies[strategy]
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        print(f"Resampled class distribution: {dict(y_resampled.value_counts())}")
        
        return X_resampled, y_resampled
    
    def get_available_strategies(self) -> list:
        """
        Get list of available sampling strategies.
        
        Returns:
            List of strategy names
        """
        return list(self.sampling_strategies.keys())
    
    def compare_strategies(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different sampling strategies.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for strategy in self.sampling_strategies.keys():
            try:
                X_resampled, y_resampled = self.apply_sampling(X, y, strategy)
                
                results[strategy] = {
                    'original_size': len(y),
                    'resampled_size': len(y_resampled),
                    'original_distribution': dict(y.value_counts()),
                    'resampled_distribution': dict(y_resampled.value_counts()),
                    'features_shape': X_resampled.shape
                }
            except Exception as e:
                results[strategy] = {'error': str(e)}
        
        return results


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\nTrain set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Train class distribution: {dict(y_train.value_counts())}")
    print(f"Test class distribution: {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    from data_loader import BankMarketingDataLoader
    
    # Load data
    loader = BankMarketingDataLoader()
    df = loader.load_data(sample_size=5000)  # Use sample for quick testing
    X, y = loader.preprocess_data(df)
    
    # Initialize sampler
    sampler = EnsembleSampler()
    
    # Compare different sampling strategies
    print("\nComparing sampling strategies...")
    results = sampler.compare_strategies(X, y)
    
    for strategy, result in results.items():
        print(f"\n{strategy.upper()}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Original size: {result['original_size']}")
            print(f"  Resampled size: {result['resampled_size']}")
            print(f"  Resampled distribution: {result['resampled_distribution']}")
