import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder as oe, StandardScaler
from src.config import FEATURE_WEIGHTS, LOG_FEATURES, CATEGORICAL_FEATURES, DATA_PATHS

def load_raw_data():
    """Load and validate raw data with proper type conversion"""
    # Read CSV with explicit dtype handling
    df = pd.read_csv(DATA_PATHS['raw_data'])
    
    # Force conversion of all numeric features
    numeric_features = [f for f in FEATURE_WEIGHTS.keys() 
                       if f not in CATEGORICAL_FEATURES and f != 'account_worth']
    
    for feature in numeric_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
    
    # Ensure target is numeric and drop invalid rows
    df['account_worth'] = pd.to_numeric(df['account_worth'], errors='coerce')
    df = df.dropna(subset=['account_worth'])
    
    return df

def log_scale(df):
    """Apply log scaling to specified features with zero handling"""
    for feature in LOG_FEATURES:
        if feature in df.columns:
            # Replace zeros with small value to avoid -inf
            df[feature] = np.log1p(df[feature].clip(lower=1e-10))
    return df

def preprocess_data(df):
    """Full preprocessing pipeline with error handling"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Apply log scaling first
    df = log_scale(df)
    
    # 2. Apply feature weights to numeric columns only
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature in df.columns and feature not in CATEGORICAL_FEATURES:
            df[feature] = df[feature].astype(float) * weight
    
    # 3. Encode categorical features
    encoder = oe(categories=[CATEGORICAL_FEATURES['max_rank'], 
                            CATEGORICAL_FEATURES['fame_level']])
    
    # Ensure categorical columns exist before encoding
    cat_cols = ['max_rank', 'fame_level']
    if all(col in df.columns for col in cat_cols):
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        with open(DATA_PATHS['encoder'], 'wb') as f:
            pickle.dump(encoder, f)
    
    # 4. Separate features and target
    X = df.drop('account_worth', axis=1)
    y = df['account_worth'].values  # Convert to numpy array
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    num_cols = [col for col in X.columns if col not in CATEGORICAL_FEATURES]
    
    if num_cols:  # Only scale if we have numeric columns
        X[num_cols] = scaler.fit_transform(X[num_cols].astype(float))
        with open(DATA_PATHS['scaler'], 'wb') as f:
            pickle.dump(scaler, f)
    
    # 6. Save processed data
    df_processed = pd.concat([X, pd.Series(y, name='account_worth')], axis=1)
    df_processed.to_csv(DATA_PATHS['processed_data'], index=False)
    
    return X, y

def prepare_training_data():
    """Main data preparation function"""
    try:
        df = load_raw_data()
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
        return preprocess_data(df)
    except Exception as e:
        print(f"Error in data preparation: {str(e)}")
        raise