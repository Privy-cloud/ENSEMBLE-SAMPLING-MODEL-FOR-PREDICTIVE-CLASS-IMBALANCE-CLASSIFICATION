# ENSEMBLE SAMPLING MODEL FOR PREDICTIVE CLASS IMBALANCE CLASSIFICATION

**CAPSTONE PROJECT FINAL YEAR**

A comprehensive machine learning solution that trains multiple models and creates an ensemble of the **BEST 2** performing models to handle imbalanced classification problems.

## 🎯 Features

- **Multiple ML Models Training**: Trains 6 different models:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Decision Tree

- **Automatic Model Selection**: Automatically evaluates and selects the **BEST 2** models based on balanced accuracy

- **Ensemble Learning**: Creates a soft voting ensemble combining the top 2 models

- **Class Imbalance Handling**: Supports multiple sampling strategies:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Random Under-sampling
  - SMOTEENN (SMOTE + Edited Nearest Neighbors)
  - SMOTETomek (SMOTE + Tomek Links)

- **Comprehensive Evaluation**: Uses metrics designed for imbalanced data:
  - Balanced Accuracy
  - F1 Score
  - Precision & Recall
  - ROC AUC
  - Confusion Matrix

## 📋 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from ensemble_model import EnsembleSamplingModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data
X, y = load_your_data()

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ensemble model (automatically selects BEST 2 models)
model = EnsembleSamplingModel(sampling_strategy='smote', random_state=42)
model.fit(X_train, y_train)

# Evaluate
results = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Run Complete Pipeline

Run the main script to see the full pipeline in action:

```bash
python main.py
```

This will:
1. Create or load an imbalanced dataset
2. Train multiple ML models
3. Select the BEST 2 models
4. Create an ensemble
5. Evaluate performance with different sampling strategies
6. Display comprehensive results

### Run Example

```bash
python example.py
```

## 📊 How It Works

1. **Data Preparation**: Apply sampling techniques to handle class imbalance
2. **Model Training**: Train 6 different ML models on the resampled data
3. **Model Evaluation**: Evaluate each model using balanced accuracy and other metrics
4. **Best 2 Selection**: Automatically select the top 2 performing models
5. **Ensemble Creation**: Combine the best 2 models using soft voting
6. **Final Evaluation**: Assess ensemble performance on test data

## 🎓 Sampling Strategies

- **SMOTE**: Creates synthetic samples for minority class
- **ADASYN**: Adaptive synthetic sampling with density distribution
- **Under-sampling**: Randomly removes majority class samples
- **SMOTEENN**: SMOTE followed by cleaning with Edited Nearest Neighbors
- **SMOTETomek**: SMOTE followed by Tomek links removal
- **None**: No sampling (use original distribution)

## 📈 Output Example

```
BEST 2 MODELS SELECTED FOR ENSEMBLE:
============================================================

1. XGBoost
   Balanced Accuracy: 0.8542
   F1 Score: 0.8321
   Precision: 0.8456
   Recall: 0.8198

2. Random Forest
   Balanced Accuracy: 0.8498
   F1 Score: 0.8287
   Precision: 0.8401
   Recall: 0.8175

============================================================
Creating Ensemble Model (Soft Voting)...
============================================================
Ensemble model trained successfully!
```

## 🔧 Customization

You can customize the models and parameters in `ensemble_model.py`:

```python
# Initialize with different sampling strategy
model = EnsembleSamplingModel(
    sampling_strategy='smotetomek',  # Change strategy
    random_state=42
)
```

## 📝 API Reference

### EnsembleSamplingModel

**Methods:**

- `fit(X_train, y_train, X_val=None, y_val=None)`: Train models and create ensemble
- `predict(X)`: Make predictions using ensemble
- `predict_proba(X)`: Get probability predictions
- `evaluate(X_test, y_test)`: Evaluate ensemble performance
- `get_model_rankings()`: Display rankings of all trained models

## 🎯 Use Cases

- Fraud Detection
- Medical Diagnosis
- Anomaly Detection
- Credit Risk Assessment
- Rare Event Prediction
- Any classification problem with imbalanced classes

## 📄 License

This is a capstone project for educational purposes.

## 👥 Contributors

Capstone Project Team - Final Year
