"""
Example usage of the Ensemble Sampling Model.
This script demonstrates how to use the ensemble model with custom data.
"""

from ensemble_model import EnsembleSamplingModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# Create a highly imbalanced dataset
print("Creating highly imbalanced dataset (95:5 ratio)...")
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # Highly imbalanced
    flip_y=0.01,
    random_state=42
)

unique, counts = np.unique(y, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")
print(f"Imbalance ratio: {counts[0]/counts[1]:.2f}:1\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the ensemble model with SMOTE sampling
print("Initializing Ensemble Sampling Model with SMOTE...\n")
model = EnsembleSamplingModel(
    sampling_strategy='smote',
    random_state=42
)

# Train the model - it will automatically select and ensemble the best 2 models
print("Training multiple models and selecting best 2 for ensemble...\n")
model.fit(X_train, y_train)

# View all model rankings
model.get_model_rankings()

# Evaluate the ensemble on test set
print("\nEvaluating ensemble on test set...")
results = model.evaluate(X_test, y_test)

# Make predictions on new data
print("\n" + "="*60)
print("Making predictions on sample data...")
sample_predictions = model.predict(X_test[:5])
sample_probabilities = model.predict_proba(X_test[:5])

print(f"Predictions: {sample_predictions}")
print(f"Probabilities:\n{sample_probabilities}")
print("="*60)
