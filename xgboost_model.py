import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta

def prepare_features(df, target_col='Close', prediction_days=1):
    """
    Prepare features for ML model, including lag features
    """
    df = df.copy()
    
    # Create target variable (future price)
    df['Target'] = df[target_col].shift(-prediction_days)
    
    # Create lag features
    for lag in [1, 3, 7, 14, 30]:
        df[f'Price_Lag_{lag}'] = df[target_col].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
    # Create target returns
    df['Target_Returns'] = df['Target'] / df[target_col] - 1
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def split_data(df, train_ratio=0.8):
    """
    Split data into training and testing sets, respecting time order
    """
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    return train_data, test_data

def prepare_ml_features(df):
    """
    Select and prepare features for the ML model
    """
    feature_columns = [
        # Price features
        'Returns', 'Log_Returns', 
        # Technical indicators
        'MA7', 'MA30', 'MA90', 'Volatility', 'RSI', 
        'MACD', 'Signal_Line', 'Price_Momentum',
        # Lag features
        'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7', 
        'Price_Lag_14', 'Price_Lag_30',
        'Returns_Lag_1', 'Returns_Lag_3', 'Returns_Lag_7',
        # Date features
        'Month', 'DayOfWeek'
    ]
    
    # Add volume features if available
    if 'Volume_MA7' in df.columns:
        feature_columns.extend(['Volume_MA7', 'Volume_MA30', 'Volume_Price_Ratio'])
    
    return feature_columns

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost model with early stopping if validation data is provided
    """
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 1000
    }
    
    model = xgb.XGBRegressor(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=100
        )
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X, y, scaler_y=None):
    """
    Evaluate model performance using multiple metrics
    """
    predictions = model.predict(X)
    
    if scaler_y:
        y = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    metrics = {
        'MAE': mean_absolute_error(y, predictions),
        'RMSE': np.sqrt(mean_squared_error(y, predictions)),
        'R2': r2_score(y, predictions)
    }
    
    return metrics, predictions

def plot_predictions(df, actual, predictions, title='Bitcoin Price Predictions'):
    """
    Create interactive plot comparing actual vs predicted values
    """
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=(title, 'Prediction Error'),
                       vertical_spacing=0.2)
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=df.index, y=actual,
                  name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=predictions,
                  name='Predicted', line=dict(color='red')),
        row=1, col=1
    )
    
    # Prediction Error
    error = actual - predictions
    fig.add_trace(
        go.Scatter(x=df.index, y=error,
                  name='Error', line=dict(color='gray')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    # Prepare features and target
    print("Preparing features...")
    df_ml = prepare_features(df, prediction_days=1)
    feature_columns = prepare_ml_features(df_ml)
    
    # Split data
    print("Splitting data...")
    train_data, test_data = split_data(df_ml)
    
    # Prepare training and testing sets
    X_train = train_data[feature_columns]
    y_train = train_data['Target_Returns']
    X_test = test_data[feature_columns]
    y_test = test_data['Target_Returns']
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target (if using price instead of returns)
    scaler_y = StandardScaler()
    y_train_scaled = y_train  # Not scaling returns
    y_test_scaled = y_test
    
    # Train model
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train_scaled, y_train_scaled)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_metrics, train_pred = evaluate_model(model, X_train_scaled, y_train_scaled)
    test_metrics, test_pred = evaluate_model(model, X_test_scaled, y_test_scaled)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTesting Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    print("\nSaving model...")
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_columns': feature_columns
        }, f)
    
    # Create and save visualizations
    print("Creating visualizations...")
    # Training predictions
    fig_train = plot_predictions(
        train_data, y_train, train_pred,
        'Training Set: Actual vs Predicted Returns'
    )
    fig_train.write_html('data/xgboost_training_results.html')
    
    # Testing predictions
    fig_test = plot_predictions(
        test_data, y_test, test_pred,
        'Test Set: Actual vs Predicted Returns'
    )
    fig_test.write_html('data/xgboost_testing_results.html')
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = go.Figure(go.Bar(
        x=feature_importance['feature'],
        y=feature_importance['importance'],
        text=feature_importance['importance'].round(4),
        textposition='auto'
    ))
    
    fig_importance.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        template='plotly_dark'
    )
    
    fig_importance.write_html('data/xgboost_feature_importance.html')
    print("\nVisualization files have been saved to the data directory.")
