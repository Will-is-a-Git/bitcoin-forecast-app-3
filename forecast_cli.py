import pandas as pd
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import argparse

def load_model(model_path='models/prophet_model_tuned.pkl'):
    """
    Load the trained Prophet model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(model, days=30):
    """
    Make predictions for the specified number of days
    """
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

def format_predictions(forecast, days=30):
    """
    Format the predictions into a readable format
    """
    # Get only future predictions
    last_date = forecast['ds'].max() - timedelta(days=days)
    future_forecast = forecast[forecast['ds'] > last_date]
    
    # Format the results
    results = pd.DataFrame({
        'Date': future_forecast['ds'],
        'Predicted Price': future_forecast['yhat'].round(2),
        'Lower Bound': future_forecast['yhat_lower'].round(2),
        'Upper Bound': future_forecast['yhat_upper'].round(2)
    })
    
    return results

def plot_predictions(forecast, days=30):
    """
    Create an interactive plot of the predictions
    """
    # Get only future predictions for highlighting
    last_date = forecast['ds'].max() - timedelta(days=days)
    
    fig = go.Figure()
    
    # Historical fitted values
    fig.add_trace(go.Scatter(
        x=forecast[forecast['ds'] <= last_date]['ds'],
        y=forecast[forecast['ds'] <= last_date]['yhat'],
        name='Historical Fit',
        line=dict(color='blue')
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=forecast[forecast['ds'] > last_date]['ds'],
        y=forecast[forecast['ds'] > last_date]['yhat'],
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast[forecast['ds'] > last_date]['ds'],
        y=forecast[forecast['ds'] > last_date]['yhat_upper'],
        name='Upper Bound',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast[forecast['ds'] > last_date]['ds'],
        y=forecast[forecast['ds'] > last_date]['yhat_lower'],
        name='Lower Bound',
        line=dict(color='gray', dash='dash'),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title='Bitcoin Price Forecast',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitcoin Price Forecasting CLI')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to forecast (default: 30)')
    parser.add_argument('--model', type=str, default='models/prophet_model_tuned.pkl',
                        help='Path to the trained model (default: models/prophet_model_tuned.pkl)')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        
        # Make predictions
        print(f"\nGenerating {args.days}-day forecast...")
        forecast = make_prediction(model, args.days)
        
        # Format and display results
        results = format_predictions(forecast, args.days)
        print("\nForecast Results:")
        print(results.to_string(index=False))
        
        # Create and save visualization
        print("\nCreating forecast visualization...")
        fig = plot_predictions(forecast, args.days)
        output_file = f'data/forecast_{args.days}days.html'
        fig.write_html(output_file)
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
