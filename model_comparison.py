import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_models():
    """
    Load both Prophet and XGBoost models
    """
    # Load Prophet model
    with open('models/prophet_model_tuned.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    
    # Load XGBoost model and associated objects
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_dict = pickle.load(f)
    
    return prophet_model, xgb_dict

def make_prophet_predictions(model, df):
    """
    Generate predictions using Prophet model
    """
    future = model.make_future_dataframe(periods=0)  # Only historical dates
    forecast = model.predict(future)
    return forecast['yhat']

def make_xgboost_predictions(model_dict, df):
    """
    Generate predictions using XGBoost model
    """
    # Prepare features
    X = df[model_dict['feature_columns']]
    X_scaled = model_dict['scaler_X'].transform(X)
    
    # Make predictions
    predictions = model_dict['model'].predict(X_scaled)
    return predictions

def calculate_metrics(actual, predicted):
    """
    Calculate various performance metrics
    """
    metrics = {
        'MAE': mean_absolute_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'R2': r2_score(actual, predicted)
    }
    return metrics

def plot_model_comparison(df, prophet_pred, xgb_pred, prophet_metrics, xgb_metrics):
    """
    Create comparison plots for both models
    """
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Price Predictions Comparison',
                                     'Prediction Errors'),
                       vertical_spacing=0.2)
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'],
                  name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=prophet_pred,
                  name='Prophet', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=xgb_pred,
                  name='XGBoost', line=dict(color='green')),
        row=1, col=1
    )
    
    # Prediction Errors
    prophet_error = df['Close'] - prophet_pred
    xgb_error = df['Close'] - xgb_pred
    
    fig.add_trace(
        go.Scatter(x=df.index, y=prophet_error,
                  name='Prophet Error', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=xgb_error,
                  name='XGBoost Error', line=dict(color='green')),
        row=2, col=1
    )
    
    # Add metrics as annotations
    metrics_text = (
        f"Prophet Metrics:<br>"
        f"MAE: ${prophet_metrics['MAE']:,.2f}<br>"
        f"RMSE: ${prophet_metrics['RMSE']:,.2f}<br>"
        f"R²: {prophet_metrics['R2']:.4f}<br><br>"
        f"XGBoost Metrics:<br>"
        f"MAE: ${xgb_metrics['MAE']:,.2f}<br>"
        f"RMSE: ${xgb_metrics['RMSE']:,.2f}<br>"
        f"R²: {xgb_metrics['R2']:.4f}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=metrics_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark',
        title='Model Comparison: Prophet vs XGBoost'
    )
    
    return fig

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    # Load models
    print("Loading models...")
    prophet_model, xgb_dict = load_models()
    
    # Generate predictions
    print("Generating predictions...")
    prophet_predictions = make_prophet_predictions(prophet_model, df)
    xgb_predictions = make_xgboost_predictions(xgb_dict, df)
    
    # Calculate metrics
    print("Calculating metrics...")
    prophet_metrics = calculate_metrics(df['Close'], prophet_predictions)
    xgb_metrics = calculate_metrics(df['Close'], xgb_predictions)
    
    # Print metrics
    print("\nProphet Model Metrics:")
    for metric, value in prophet_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nXGBoost Model Metrics:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create and save comparison visualization
    print("\nCreating comparison visualization...")
    fig = plot_model_comparison(
        df, prophet_predictions, xgb_predictions,
        prophet_metrics, xgb_metrics
    )
    fig.write_html('data/model_comparison.html')
    print("Comparison plot saved to data/model_comparison.html")
