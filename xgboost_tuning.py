import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from itertools import product
import pickle
import json

def prepare_tuning_data(df, feature_columns):
    """
    Prepare data for parameter tuning using walk-forward optimization
    """
    X = df[feature_columns]
    y = df['Target_Returns']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def grid_search_cv(X, y, param_grid, cv=5):
    """
    Perform grid search with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in product(*param_grid.values())]
    
    results = []
    best_score = float('inf')
    best_params = None
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for params in param_combinations:
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model with current parameters
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                eval_metric='rmse',
                verbose=False
            )
            
            # Get validation score
            pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - pred) ** 2))
            scores.append(rmse)
        
        # Calculate average score across folds
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'params': params,
            'mean_rmse': mean_score,
            'std_rmse': std_score
        })
        
        print(f"RMSE: {mean_score:.4f} (±{std_score:.4f}) with parameters: {params}")
        
        # Update best parameters if needed
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, results

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    # Load existing model to get feature columns
    with open('models/xgboost_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    feature_columns = model_dict['feature_columns']
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Prepare data
    X_scaled, y = prepare_tuning_data(df, feature_columns)
    
    # Perform grid search
    print("\nPerforming grid search...")
    best_params, results = grid_search_cv(X_scaled, y, param_grid)
    
    # Save results
    print("\nSaving results...")
    with open('models/xgboost_tuning_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'all_results': results
        }, f, indent=2)
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_scaled, y)
    
    # Save tuned model
    with open('models/xgboost_model_tuned.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler_X': model_dict['scaler_X'],
            'scaler_y': model_dict['scaler_y'],
            'feature_columns': feature_columns,
            'best_params': best_params
        }, f)
    
    print("Tuned model saved to models/xgboost_model_tuned.pkl")
