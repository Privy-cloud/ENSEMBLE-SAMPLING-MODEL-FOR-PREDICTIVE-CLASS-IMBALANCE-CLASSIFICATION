"""
Demo Data Generator for Bank Marketing Dataset
Creates a synthetic dataset similar to the Bank Marketing dataset for testing purposes.
"""

import pandas as pd
import numpy as np
import os


def generate_demo_bank_marketing_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Bank Marketing dataset for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic Bank Marketing data
    """
    np.random.seed(random_state)
    
    # Define feature distributions based on Bank Marketing dataset characteristics
    data = {
        # Numerical features
        'age': np.random.randint(18, 95, n_samples),
        'duration': np.random.exponential(scale=250, size=n_samples).astype(int),
        'campaign': np.random.poisson(lam=2.5, size=n_samples) + 1,
        'pdays': np.random.choice([999] + list(range(0, 30)), size=n_samples, p=[0.8] + [0.2/30]*30),
        'previous': np.random.poisson(lam=0.5, size=n_samples),
        'emp.var.rate': np.random.normal(0.1, 1.5, n_samples),
        'cons.price.idx': np.random.normal(93.5, 0.5, n_samples),
        'cons.conf.idx': np.random.normal(-40, 5, n_samples),
        'euribor3m': np.random.normal(3.5, 1.5, n_samples),
        'nr.employed': np.random.normal(5190, 70, n_samples),
        
        # Categorical features
        'job': np.random.choice(
            ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
             'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
            n_samples
        ),
        'marital': np.random.choice(['divorced', 'married', 'single', 'unknown'], n_samples, p=[0.1, 0.6, 0.3, 0.0]),
        'education': np.random.choice(
            ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
             'professional.course', 'university.degree', 'unknown'],
            n_samples,
            p=[0.05, 0.05, 0.1, 0.2, 0.01, 0.15, 0.4, 0.04]
        ),
        'default': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.9, 0.02, 0.08]),
        'housing': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.45, 0.5, 0.05]),
        'loan': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.8, 0.15, 0.05]),
        'contact': np.random.choice(['cellular', 'telephone'], n_samples, p=[0.6, 0.4]),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], n_samples),
        'poutcome': np.random.choice(['failure', 'nonexistent', 'success'], n_samples, p=[0.1, 0.85, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with class imbalance (approximately 11% positive class)
    # Create some correlation with features
    prob = 0.05 + (df['duration'] > 300) * 0.15 + (df['poutcome'] == 'success') * 0.3
    prob = np.clip(prob, 0, 1)
    df['y'] = np.random.binomial(1, prob, n_samples)
    df['y'] = df['y'].map({0: 'no', 1: 'yes'})
    
    return df


def save_demo_data(output_dir: str = 'data', filename: str = 'bank-additional-full.csv'):
    """
    Generate and save demo Bank Marketing dataset.
    
    Args:
        output_dir: Directory to save the data
        filename: Filename for the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate demo data (41188 samples - similar to actual dataset size)
    print("Generating demo Bank Marketing dataset...")
    df = generate_demo_bank_marketing_data(n_samples=41188)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"Demo dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    
    return output_path


if __name__ == "__main__":
    # Generate and save demo data
    save_demo_data()
