"""
Quick Start Example for Bank Marketing Dataset Analysis
Demonstrates basic usage of the ensemble sampling model implementation.
"""

import sys
sys.path.insert(0, 'src')

from src.data_loader import BankMarketingDataLoader
from src.ensemble_sampler import EnsembleSampler, prepare_train_test_split
from src.model_trainer import ModelTrainer

print("=" * 80)
print("QUICK START EXAMPLE - BANK MARKETING DATASET")
print("=" * 80)

# Step 1: Load a sample of the data
print("\n[1] Loading Bank Marketing Dataset (sample)...")
loader = BankMarketingDataLoader()
df = loader.load_data(sample_size=2000)  # Use 2000 samples for quick demo

# Step 2: Preprocess the data
print("\n[2] Preprocessing data...")
X, y = loader.preprocess_data(df)

# Step 3: Split into train and test sets
print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)

# Step 4: Apply SMOTE sampling
print("\n[4] Applying SMOTE sampling...")
sampler = EnsembleSampler()
X_train_smote, y_train_smote = sampler.apply_sampling(X_train, y_train, strategy='smote')

# Step 5: Train a Random Forest model
print("\n[5] Training Random Forest model...")
trainer = ModelTrainer()
rf_model = trainer.train_model('random_forest', X_train_smote, y_train_smote)

# Step 6: Evaluate the model
print("\n[6] Evaluating model...")
metrics = trainer.evaluate_model(rf_model, X_test, y_test, 'Random Forest with SMOTE')

# Display key metrics
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
if 'roc_auc' in metrics:
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
else:
    print("ROC-AUC:   N/A")
print("=" * 80)

print("\n✓ Quick start example completed successfully!")
print("  To run the full pipeline, use: python main.py")
print("  To compare all strategies, use: python main.py --compare-all")
