"""
Data Loader for Bank Marketing Dataset from UCI Repository
Downloads and preprocesses the Bank Marketing dataset for machine learning models.
"""

import os
import pandas as pd
import requests
from typing import Tuple
import zipfile
from io import BytesIO


class BankMarketingDataLoader:
    """
    Loads and preprocesses the Bank Marketing dataset from UCI Machine Learning Repository.
    
    Dataset Information:
    - The data is related with direct marketing campaigns of a Portuguese banking institution.
    - The marketing campaigns were based on phone calls.
    - Classification goal is to predict if the client will subscribe a term deposit (variable y).
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store the downloaded data
        """
        self.data_dir = data_dir
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        self.csv_file = 'bank-additional-full.csv'
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_data(self) -> str:
        """
        Download the Bank Marketing dataset from UCI repository.
        
        Returns:
            Path to the downloaded CSV file
        """
        zip_path = os.path.join(self.data_dir, 'bank-additional.zip')
        csv_path = os.path.join(self.data_dir, self.csv_file)
        
        # Check if data already exists
        if os.path.exists(csv_path):
            print(f"Dataset already exists at {csv_path}")
            return csv_path
        
        print(f"Downloading Bank Marketing dataset from UCI repository...")
        try:
            response = requests.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Extract the zip file
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                # Extract all files to data directory
                zip_ref.extractall(self.data_dir)
            
            print(f"Dataset downloaded and extracted to {self.data_dir}")
            return csv_path
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """
        Load the Bank Marketing dataset.
        
        Args:
            sample_size: Optional sample size for testing (None for full dataset)
            
        Returns:
            DataFrame containing the dataset
        """
        csv_path = self.download_data()
        
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, sep=';')
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"Loaded sample of {len(df)} records")
        else:
            print(f"Loaded {len(df)} records")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the Bank Marketing dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Separate features and target
        X = df.drop('y', axis=1)
        y = df['y'].map({'yes': 1, 'no': 0})
        
        # Handle categorical variables using one-hot encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print(f"Preprocessed data shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        print(f"Class imbalance ratio: {(y.value_counts()[0] / y.value_counts()[1]):.2f}:1")
        
        return X, y
    
    def get_dataset_info(self, df: pd.DataFrame) -> None:
        """
        Display information about the dataset.
        
        Args:
            df: DataFrame to analyze
        """
        print("\n" + "="*80)
        print("BANK MARKETING DATASET INFORMATION")
        print("="*80)
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumn names and types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nTarget variable distribution:")
        print(df['y'].value_counts())
        print(f"\nFirst few rows:")
        print(df.head())
        print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    loader = BankMarketingDataLoader()
    
    # Load the data
    df = loader.load_data()
    
    # Display dataset information
    loader.get_dataset_info(df)
    
    # Preprocess the data
    X, y = loader.preprocess_data(df)
    
    print(f"\nFinal preprocessed dataset ready for modeling:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
