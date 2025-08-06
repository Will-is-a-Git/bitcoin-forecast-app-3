import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

def prepare_data_for_prophet(df):
    """
    Prepare the data in the format required by Prophet
    """
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df.index
    prophet_df['y'] = df['Close']
    return prophet_df

def grid_search_parameters(df):
    """
    Perform grid search to find optimal Prophet parameters
    """
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'daily_seasonality': [True, False]
    }
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) 
                 for v in itertools.product(*param_grid.values())]
    
    # Initialize storage for results
    results = []
    best_rmse = float('inf')
    best_params = None
    prophet_df = prepare_data_for_prophet(df)
    
    print(f"Testing {len(all_params)} parameter combinations...")
    
    # Try all parameter combinations
    for params in all_params:
        model = Prophet(**params)
        
        try:
            # Fit model and perform cross validation
            model.fit(prophet_df)
            df_cv = cross_validation(model, initial='730 days', 
                                   period='180 days', horizon='30 days')
            df_p = performance_metrics(df_cv)
            
            # Store results
            rmse = df_p['rmse'].mean()
            results.append(dict(params=params, rmse=rmse))
            
            # Update best parameters if needed
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                
            print(f"RMSE: {rmse:.2f} with parameters: {params}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    return best_params, results

def train_best_model(df, params):
    """
    Train Prophet model with the best parameters
    """
    model = Prophet(**params)
    prophet_df = prepare_data_for_prophet(df)
    model.fit(prophet_df)
    return model

def plot_parameter_comparison(results):
    """
    Create visualization of parameter search results
    """
    fig = go.Figure()
    
    # Extract RMSE values and parameter combinations
    rmse_values = [r['rmse'] for r in results]
    param_combinations = [str(r['params']) for r in results]
    
    # Create bar plot
    fig.add_trace(go.Bar(
        x=list(range(len(results))),
        y=rmse_values,
        text=[f"RMSE: {rmse:.2f}" for rmse in rmse_values],
        hovertext=param_combinations,
        name='RMSE by Parameter Set'
    ))
    
    fig.update_layout(
        title='Prophet Parameter Tuning Results',
        xaxis_title='Parameter Combination',
        yaxis_title='RMSE',
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

if __name__ == "__main__":
    # Load the preprocessed data
    print("Loading data...")
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    
    # Perform parameter search
    print("\nPerforming parameter search...")
    best_params, results = grid_search_parameters(df)
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train model with best parameters
    print("\nTraining model with best parameters...")
    best_model = train_best_model(df, best_params)
    
    # Save the tuned model
    with open('models/prophet_model_tuned.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Tuned model saved to models/prophet_model_tuned.pkl")
    
    # Create and save parameter comparison visualization
    fig = plot_parameter_comparison(results)
    fig.write_html('data/parameter_tuning_results.html')
    print("Parameter tuning results visualization saved to data/parameter_tuning_results.html")
