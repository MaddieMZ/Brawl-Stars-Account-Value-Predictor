import pandas as pd
import numpy as np
import pickle
from src.config import DATA_PATHS, FEATURE_WEIGHTS, LOG_FEATURES

class ValuePredictor:
    def __init__(self):
        """Initialize predictor with proper feature name handling"""
        try:
            # Load models with feature name validation
            with open(DATA_PATHS['model'], 'rb') as f:
                self.model = pickle.load(f)
            
            with open(DATA_PATHS['encoder'], 'rb') as f:
                self.encoder = pickle.load(f)
                # Store encoder feature names
                self.encoder_features = getattr(self.encoder, 'feature_names_in_', ['max_rank', 'fame_level'])
                
            with open(DATA_PATHS['scaler'], 'rb') as f:
                self.scaler = pickle.load(f)
                # Store scaler feature names
                self.scaler_features = getattr(self.scaler, 'feature_names_in_', [])
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model files: {str(e)}")

    def _validate_features(self, input_df):
        """Ensure all required features are present"""
        # Features needed by the model
        required_features = set(self.scaler_features) | {'max_rank', 'fame_level'}
        
        # Remove target variable if present
        if 'account_worth' in required_features:
            required_features.remove('account_worth')
            
        missing = required_features - set(input_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def _preprocess_input(self, acc_data):
        """Preprocess input with proper feature handling"""
        try:
            # Create DataFrame first to validate features
            input_df = pd.DataFrame([{k: v for k, v in acc_data.items() 
                                    if k != 'account_worth'}])
            
            self._validate_features(input_df)
            
            # Convert numeric features
            numeric_features = [f for f in FEATURE_WEIGHTS.keys() 
                              if f not in ['max_rank', 'fame_level']]
            
            for feature in numeric_features:
                if feature in input_df.columns:
                    input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0)
            
            # Apply log scaling
            for feature in LOG_FEATURES:
                if feature in input_df.columns:
                    input_df[feature] = np.log1p(input_df[feature].clip(lower=0))
            
            # Apply weights
            for feature, weight in FEATURE_WEIGHTS.items():
                if feature in input_df.columns and feature not in ['max_rank', 'fame_level']:
                    input_df[feature] = input_df[feature] * weight
            
            # Encode categoricals
            if all(col in input_df.columns for col in ['max_rank', 'fame_level']):
                input_df[['max_rank', 'fame_level']] = self.encoder.transform(
                    input_df[['max_rank', 'fame_level']]
                )
            
            # Scale numeric features
            num_cols = [col for col in self.scaler_features 
                       if col in input_df.columns and col not in ['max_rank', 'fame_level']]
            if num_cols:
                input_df[num_cols] = self.scaler.transform(input_df[num_cols])
            
            return input_df
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise

    def predict(self, acc_data):
        """Make prediction with full error handling"""
        try:
            input_df = self._preprocess_input(acc_data)
            prediction = self.model.predict(input_df)[0]
            return float(prediction)
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return 0.0  # Return default value instead of None