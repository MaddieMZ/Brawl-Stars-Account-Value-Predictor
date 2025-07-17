from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import pickle
from src.data_processing import prepare_training_data
from src.config import DATA_PATHS

def train_model(test_size=0.2, random_state=42):
    # Load and preprocess data
    X, Y = prepare_training_data()
    
    # Split data - ensuring consistent shapes
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state
    )
    model.fit(X_train, Y_train)
    
    # Make predictions on TEST SET only
    predictions = model.predict(X_test)
    
    # Calculate metrics using TEST SET values
    mae = mean_absolute_error(Y_test, predictions)  # Changed from Y to Y_test
    r2 = r2_score(Y_test, predictions)
    
    print("\nModel Training Results:")
    print(f"- MAE: ${mae:.2f}")
    print(f"- R² Score: {r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance.to_string(index=False))
    
    # Save model
    with open(DATA_PATHS['model'], 'wb') as f:
        pickle.dump(model, f)
    
    return model, mae, r2