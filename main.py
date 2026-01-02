"""
Main Pipeline for Ensemble Sampling Model for Predictive Class Imbalance Classification
Uses Bank Marketing Dataset from UCI Repository
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import BankMarketingDataLoader
from src.ensemble_sampler import EnsembleSampler, prepare_train_test_split
from src.model_trainer import ModelTrainer


def run_experiment(
    sampling_strategy: str = 'smote',
    test_size: float = 0.2,
    sample_size: int = None,
    random_state: int = 42
):
    """
    Run a complete experiment with the specified sampling strategy.
    
    Args:
        sampling_strategy: Sampling strategy to use
        test_size: Proportion of test set
        sample_size: Sample size for quick testing (None for full dataset)
        random_state: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("ENSEMBLE SAMPLING MODEL FOR PREDICTIVE CLASS IMBALANCE CLASSIFICATION")
    print("Bank Marketing Dataset - UCI Repository")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading Bank Marketing Dataset...")
    loader = BankMarketingDataLoader()
    df = loader.load_data(sample_size=sample_size)
    loader.get_dataset_info(df)
    
    # Step 2: Preprocess Data
    print("\n[STEP 2] Preprocessing Data...")
    X, y = loader.preprocess_data(df)
    
    # Step 3: Split Data
    print("\n[STEP 3] Splitting Data into Train and Test Sets...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 4: Apply Sampling Strategy
    print(f"\n[STEP 4] Applying Sampling Strategy: {sampling_strategy}")
    sampler = EnsembleSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = sampler.apply_sampling(
        X_train, y_train, strategy=sampling_strategy
    )
    
    # Step 5: Train and Evaluate Models
    print("\n[STEP 5] Training and Evaluating Models...")
    trainer = ModelTrainer(random_state=random_state)
    results = trainer.train_all_models(
        X_train_resampled, y_train_resampled, X_test, y_test
    )
    
    # Step 6: Display Results Summary
    print("\n[STEP 6] Results Summary")
    print("="*80)
    summary = trainer.get_results_summary()
    print(summary.to_string(index=False))
    print("="*80)
    
    # Save results to CSV
    results_file = f'results_{sampling_strategy}.csv'
    summary.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return results, summary


def compare_all_sampling_strategies(sample_size: int = None):
    """
    Compare all available sampling strategies.
    
    Args:
        sample_size: Sample size for quick testing (None for full dataset)
    """
    print("\n" + "="*80)
    print("COMPARING ALL SAMPLING STRATEGIES")
    print("="*80)
    
    # Initialize sampler to get available strategies
    sampler = EnsembleSampler()
    strategies = sampler.get_available_strategies()
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n\n{'#'*80}")
        print(f"RUNNING EXPERIMENT WITH: {strategy.upper()}")
        print(f"{'#'*80}")
        
        try:
            results, summary = run_experiment(
                sampling_strategy=strategy,
                sample_size=sample_size
            )
            all_results[strategy] = summary
        except Exception as e:
            print(f"Error with {strategy}: {e}")
            all_results[strategy] = None
    
    # Combine all results
    print("\n\n" + "="*80)
    print("FINAL COMPARISON - ALL SAMPLING STRATEGIES")
    print("="*80)
    
    for strategy, summary in all_results.items():
        if summary is not None and not summary.empty:
            print(f"\n{strategy.upper()}:")
            print(summary.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ensemble Sampling Model for Predictive Class Imbalance Classification'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='smote',
        help='Sampling strategy (smote, random_oversampling, etc.)'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all sampling strategies'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for quick testing (default: use full dataset)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.compare_all:
        compare_all_sampling_strategies(sample_size=args.sample_size)
    else:
        run_experiment(
            sampling_strategy=args.strategy,
            test_size=args.test_size,
            sample_size=args.sample_size,
            random_state=args.random_state
        )
