# Ensemble Sampling Model for Predictive Class Imbalance Classification

**CAPSTONE PROJECT - FINAL YEAR**

## Overview

This project implements ensemble sampling techniques for handling class imbalance in machine learning classification tasks. The implementation uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository to train and evaluate various machine learning models.

The Bank Marketing dataset is particularly suitable for this project as it exhibits significant class imbalance, making it ideal for demonstrating the effectiveness of different sampling strategies.

## Dataset

**Bank Marketing Dataset** - UCI Machine Learning Repository
- **Source**: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- **Description**: Data related to direct marketing campaigns (phone calls) of a Portuguese banking institution
- **Goal**: Predict if a client will subscribe to a term deposit (binary classification)
- **Features**: 20 input variables (demographic, social, economic indicators, and campaign-related)
- **Class Imbalance**: The dataset exhibits significant imbalance between positive and negative classes

## Features

### Sampling Techniques
- **Original** - No sampling (baseline)
- **Random Oversampling** - Randomly duplicate minority class samples
- **Random Undersampling** - Randomly remove majority class samples
- **SMOTE** - Synthetic Minority Over-sampling Technique
- **ADASYN** - Adaptive Synthetic Sampling
- **Borderline SMOTE** - SMOTE focused on borderline samples
- **Tomek Links** - Remove Tomek links (overlapping samples)
- **NearMiss** - Undersample based on nearest neighbors
- **SMOTE + ENN** - SMOTE followed by Edited Nearest Neighbors
- **SMOTE + Tomek** - SMOTE followed by Tomek Links removal

### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost
- Naive Bayes
- K-Nearest Neighbors

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

## Project Structure

```
ENSEMBLE-SAMPLING-MODEL-FOR-PREDICTIVE-CLASS-IMBALANCE-CLASSIFICATION/
│
├── data/                          # Dataset directory (created on first run)
│   └── bank-additional-full.csv   # Bank Marketing dataset (auto-downloaded)
│
├── models/                        # Saved trained models (created during training)
│   └── *.joblib                   # Serialized model files
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb  # EDA notebook
│
├── src/                          # Source code
│   ├── data_loader.py            # Dataset downloading and preprocessing
│   ├── ensemble_sampler.py       # Sampling techniques implementation
│   └── model_trainer.py          # Model training and evaluation
│
├── main.py                       # Main pipeline script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Privy-cloud/ENSEMBLE-SAMPLING-MODEL-FOR-PREDICTIVE-CLASS-IMBALANCE-CLASSIFICATION.git
cd ENSEMBLE-SAMPLING-MODEL-FOR-PREDICTIVE-CLASS-IMBALANCE-CLASSIFICATION
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main pipeline with default settings (SMOTE sampling):
```bash
python main.py
```

### Command Line Options

```bash
# Use a specific sampling strategy
python main.py --strategy smote

# Compare all sampling strategies
python main.py --compare-all

# Use a sample of the data for quick testing
python main.py --sample-size 5000

# Customize test set size
python main.py --test-size 0.3

# Set random seed
python main.py --random-state 42
```

### Available Sampling Strategies
- `original` - No sampling
- `random_oversampling` - Random oversampling
- `random_undersampling` - Random undersampling
- `smote` - SMOTE (default)
- `adasyn` - ADASYN
- `borderline_smote` - Borderline SMOTE
- `tomek_links` - Tomek Links
- `nearmiss` - NearMiss
- `smote_enn` - SMOTE + ENN
- `smote_tomek` - SMOTE + Tomek

### Examples

1. **Run with SMOTE sampling (recommended):**
```bash
python main.py --strategy smote
```

2. **Compare all sampling strategies:**
```bash
python main.py --compare-all
```

3. **Quick test with sample data:**
```bash
python main.py --sample-size 5000 --strategy smote
```

### Using Python API

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

# Apply SMOTE
sampler = EnsembleSampler()
X_train_resampled, y_train_resampled = sampler.apply_sampling(
    X_train, y_train, strategy='smote'
)

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(
    X_train_resampled, y_train_resampled, X_test, y_test
)

# Get results summary
summary = trainer.get_results_summary()
print(summary)
```

### Exploratory Data Analysis

Launch Jupyter Notebook to explore the dataset:
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## Results

The pipeline automatically generates:
- Trained models saved in `models/` directory
- Results CSV files with performance metrics
- Detailed console output with:
  - Dataset statistics
  - Class distribution before/after sampling
  - Model training progress
  - Evaluation metrics for each model
  - Comparative summary of all models

## Key Performance Metrics

For imbalanced classification, we focus on:
- **F1-Score**: Harmonic mean of precision and recall
- **Recall**: Ability to identify positive cases (important for imbalanced data)
- **Precision**: Accuracy of positive predictions
- **ROC-AUC**: Overall model discrimination ability
- **MCC**: Correlation between predicted and actual values (good for imbalanced data)

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **imbalanced-learn** - Sampling techniques for imbalanced datasets
- **XGBoost** - Gradient boosting framework
- **matplotlib & seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

## Dataset Citation

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

## Contributing

This is a capstone project. For suggestions or improvements, please open an issue or submit a pull request.

## License

This project is for educational purposes as part of a final year capstone project.

## Acknowledgments

- UCI Machine Learning Repository for providing the Bank Marketing dataset
- The imbalanced-learn library developers for sampling techniques
- scikit-learn community for machine learning tools

## Authors

Capstone Project Team - Final Year

---

**Note**: The dataset will be automatically downloaded from the UCI repository on first run. An internet connection is required for the initial setup.
