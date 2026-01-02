# Usage Guide

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Quick Start Example

```bash
python quick_start.py
```

This will:
- Load a sample of the Bank Marketing dataset
- Apply SMOTE sampling
- Train a Random Forest model
- Display evaluation metrics

### 3. Run Full Pipeline

```bash
# Run with default settings (SMOTE sampling, full dataset)
python main.py

# Run with a specific sampling strategy
python main.py --strategy random_oversampling

# Run with a sample for quick testing
python main.py --sample-size 5000 --strategy smote

# Compare all sampling strategies
python main.py --compare-all
```

## Available Sampling Strategies

1. **original** - No sampling (baseline)
2. **smote** - Synthetic Minority Over-sampling Technique (recommended)
3. **random_oversampling** - Randomly duplicate minority class
4. **random_undersampling** - Randomly remove majority class
5. **adasyn** - Adaptive Synthetic Sampling
6. **borderline_smote** - SMOTE for borderline samples
7. **tomek_links** - Remove Tomek links
8. **nearmiss** - Undersample based on nearest neighbors
9. **smote_enn** - SMOTE + Edited Nearest Neighbors
10. **smote_tomek** - SMOTE + Tomek Links

## Command Line Options

```
python main.py [OPTIONS]

Options:
  --strategy TEXT         Sampling strategy to use (default: smote)
  --compare-all          Compare all sampling strategies
  --sample-size INTEGER  Number of samples to use (default: full dataset)
  --test-size FLOAT      Test set proportion (default: 0.2)
  --random-state INTEGER Random seed (default: 42)
```

## Examples

### Example 1: Train with SMOTE
```bash
python main.py --strategy smote
```

### Example 2: Compare All Strategies
```bash
python main.py --compare-all
```

### Example 3: Quick Test with Sample
```bash
python main.py --sample-size 3000 --strategy smote
```

### Example 4: Custom Test Size
```bash
python main.py --test-size 0.3 --strategy random_oversampling
```

## Using the Python API

### Basic Usage

```python
from src.data_loader import BankMarketingDataLoader
from src.ensemble_sampler import EnsembleSampler, prepare_train_test_split
from src.model_trainer import ModelTrainer

# Load data
loader = BankMarketingDataLoader()
df = loader.load_data()
X, y = loader.preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

# Apply sampling
sampler = EnsembleSampler()
X_train_resampled, y_train_resampled = sampler.apply_sampling(
    X_train, y_train, strategy='smote'
)

# Train and evaluate
trainer = ModelTrainer()
results = trainer.train_all_models(
    X_train_resampled, y_train_resampled, X_test, y_test
)

# View results
summary = trainer.get_results_summary()
print(summary)
```

### Train a Single Model

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Train a specific model
model = trainer.train_model('random_forest', X_train, y_train)

# Evaluate
metrics = trainer.evaluate_model(model, X_test, y_test, 'Random Forest')

print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Compare Sampling Strategies

```python
from src.ensemble_sampler import EnsembleSampler

sampler = EnsembleSampler()

# Get all available strategies
strategies = sampler.get_available_strategies()
print(f"Available strategies: {strategies}")

# Compare strategies
comparison = sampler.compare_strategies(X, y)
for strategy, result in comparison.items():
    print(f"{strategy}: {result}")
```

### Save and Load Models

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Train and save
model = trainer.train_model('xgboost', X_train, y_train)
trainer.save_model(model, 'xgboost_custom')

# Load later
loaded_model = trainer.load_model('xgboost_custom')
```

## Exploratory Data Analysis

To explore the dataset interactively:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

The notebook includes:
- Dataset overview
- Class imbalance analysis
- Feature distributions
- Correlation analysis
- Missing values check
- Statistical summaries

## Output Files

After running the pipeline, you'll find:

- **models/*.joblib** - Trained models (excluded from git)
- **data/*.csv** - Downloaded dataset (excluded from git)
- **results_*.csv** - Performance metrics (excluded from git)

## Understanding the Results

### Key Metrics for Imbalanced Classification

1. **F1-Score** - Best single metric for imbalanced data (harmonic mean of precision and recall)
2. **Recall** - Ability to identify positive cases (minimize false negatives)
3. **Precision** - Accuracy of positive predictions (minimize false positives)
4. **ROC-AUC** - Overall discrimination ability
5. **MCC** - Matthews Correlation Coefficient (good for imbalanced data)

### Interpreting the Confusion Matrix

```
[[TN  FP]
 [FN  TP]]
```

- **TN (True Negative)**: Correctly predicted "no"
- **FP (False Positive)**: Incorrectly predicted "yes"
- **FN (False Negative)**: Incorrectly predicted "no"
- **TP (True Positive)**: Correctly predicted "yes"

## Troubleshooting

### Dataset Download Issues

If the automatic download fails:

1. Manually download from: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
2. Extract `bank-additional-full.csv` to the `data/` directory
3. Or use the demo data generator:
   ```bash
   python generate_demo_data.py
   ```

### Memory Issues

For large datasets, use sampling:
```bash
python main.py --sample-size 10000
```

### Slow Training

To speed up training:
- Use a smaller sample: `--sample-size 5000`
- Train specific models instead of all models
- Use fewer estimators in ensemble methods

## Advanced Usage

### Custom Sampling Strategy

```python
from imblearn.over_sampling import SMOTE

# Create custom SMOTE with different parameters
custom_smote = SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=42)
X_resampled, y_resampled = custom_smote.fit_resample(X_train, y_train)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

## Getting Help

For issues or questions:
1. Check this usage guide
2. Review the main README.md
3. Examine the code examples in quick_start.py
4. Open an issue on GitHub

## References

- [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
