import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

def prepare_data_for_prophet(df):
    """
    Prepare the data in the format required by Prophet (ds and y columns)
    """
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df.index
    prophet_df['y'] = df['Close']
    return prophet_df

def train_prophet_model(df, seasonality_mode='multiplicative', 
                       changepoint_prior_scale=0.05,
                       seasonality_prior_scale=10,
                       daily_seasonality=True,
                       weekly_seasonality=True,
                       yearly_seasonality=True):
    """
    Train a Prophet model with custom parameters
    """
    # Create and configure the model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
    
    # Add additional regressors (you might want to add more based on your analysis)
    if 'Returns' in df.columns:
        model.add_regressor('Returns')
    
    # Fit the model
    prophet_df = prepare_data_for_prophet(df)
    if 'Returns' in df.columns:
        prophet_df['Returns'] = df['Returns']
    
    model.fit(prophet_df)
    return model

def make_future_predictions(model, days=30):
    """
    Make predictions for the next specified number of days
    """
    future_dates = model.make_future_dataframe(periods=days)
    forecast = model.predict(future_dates)
    return forecast

def evaluate_model(model, actual_df):
    """
    Evaluate the model using various metrics
    """
    # Prepare actual data
    prophet_df = prepare_data_for_prophet(actual_df)
    
    # Make predictions for the historical dates
    forecast = model.predict(prophet_df)
    
    # Calculate metrics
    metrics = {
        'mae': np.mean(np.abs(forecast['yhat'] - prophet_df['y'])),
        'rmse': np.sqrt(np.mean((forecast['yhat'] - prophet_df['y'])**2)),
        'mape': np.mean(np.abs((prophet_df['y'] - forecast['yhat']) / prophet_df['y'])) * 100
    }
    
    return metrics, forecast

def plot_forecast(actual_df, forecast_df, model_metrics=None):
    """
    Create an interactive plot showing actual vs predicted values
    """
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Bitcoin Price Forecast', 'Forecast Components'),
                       vertical_spacing=0.2)
    
    # Actual vs Predicted plot
    fig.add_trace(
        go.Scatter(x=actual_df.index, y=actual_df['Close'],
                  name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                  name='Predicted', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                  name='Upper Bound', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                  name='Lower Bound', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    # Trend component
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['trend'],
                  name='Trend', line=dict(color='green')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price Forecast with Prophet',
        template='plotly_dark',
        showlegend=True,
        height=800
    )
    
    if model_metrics:
        metrics_text = (f"MAE: ${model_metrics['mae']:,.2f}<br>"
                       f"RMSE: ${model_metrics['rmse']:,.2f}<br>"
                       f"MAPE: {model_metrics['mape']:.2f}%")
        
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
    
    return fig

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    print("Training Prophet model...")
    model = train_prophet_model(df)
    
    # Save the model
    with open('models/prophet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/prophet_model.pkl")
    
    # Evaluate the model
    metrics, historical_forecast = evaluate_model(model, df)
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")
    
    # Make future predictions
    future_forecast = make_future_predictions(model, days=30)
    
    # Create and save visualization
    fig = plot_forecast(df, future_forecast, metrics)
    fig.write_html('data/forecast_results.html')
    print("\nForecast plot saved to data/forecast_results.html")
    
    # Print next 7 days forecast
    print("\nNext 7 days forecast:")
    next_week = future_forecast[future_forecast['ds'] > df.index[-1]].head(7)
    print(next_week[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
