import streamli@st.cache_data
def load_data():
    """Load the historical Bitcoin data"""
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    return df

@st.cache_resource
def load_models():
    """Load both Prophet and XGBoost models"""
    # Load Prophet model
    with open('models/prophet_model_tuned.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    
    # Load XGBoost model
    with open('models/xgboost_model_tuned.pkl', 'rb') as f:
        xgb_dict = pickle.load(f)
    
    return prophet_model, xgb_dict

def make_forecast(prophet_model, xgb_dict, days):
    """Generate forecasts using both models"""
    # Prophet forecast
    future = prophet_model.make_future_dataframe(periods=days)
    prophet_forecast = prophet_model.predict(future)
    
    return prophet_forecast, None  # We'll focus on Prophet for simplicity

def plot_forecast(df, prophet_forecast, days):
    """Create a simple forecast plot"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Forecast
    future_dates = pd.date_range(
        start=df.index[-1],
        periods=days + 1,
        freq='D'
    )[1:]
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=prophet_forecast['yhat'].tail(days),
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=prophet_forecast['yhat_upper'].tail(days),
            fill=None,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Upper Bound'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=prophet_forecast['yhat_lower'].tail(days),
            fill='tonexty',
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Lower Bound'
        )
    )
    
    fig.update_layout(
        title='Bitcoin Price Forecast',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        showlegend=True
    )
    
    return figndas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Page config
st.set_page_config(
    page_title="Bitcoin Price Forecast",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the historical Bitcoin data"""
    df = pd.read_csv('data/btc_data.csv', index_col=0, parse_dates=True)
    return df

@st.cache_resource
def load_models():
    """Load both Prophet and XGBoost models"""
    # Load Prophet model
    with open('models/prophet_model_tuned.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    
    # Load XGBoost model
    with open('models/xgboost_model_tuned.pkl', 'rb') as f:
        xgb_dict = pickle.load(f)
    
    return prophet_model, xgb_dict

def get_latest_data():
    """Fetch the most recent Bitcoin data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Get last 90 days
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    return btc

def make_prophet_forecast(model, days):
    """Generate forecast using Prophet model"""
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

def prepare_features(df, xgb_dict):
    """Prepare features for XGBoost prediction"""
    # Get the required features
    feature_columns = xgb_dict['feature_columns']
    missing_cols = set(feature_columns) - set(df.columns)
    
    if missing_cols:
        st.warning(f"Missing features: {missing_cols}")
        return None
    
    X = df[feature_columns]
    X_scaled = xgb_dict['scaler_X'].transform(X)
    return X_scaled

def make_xgboost_forecast(model_dict, X_scaled, days):
    """Generate forecast using XGBoost model"""
    predictions = model_dict['model'].predict(X_scaled)
    return predictions

def plot_forecasts(historical_df, prophet_forecast, xgb_forecast, days):
    """Create an interactive plot with both models' forecasts"""
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Bitcoin Price Forecast',
                                     'Model Comparison'),
                       vertical_spacing=0.2)
    
    # Historical data
    fig.add_trace(
        go.Scatter(x=historical_df.index, y=historical_df['Close'],
                  name='Historical', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Prophet forecast
    future_dates = pd.date_range(
        start=historical_df.index[-1],
        periods=days + 1,
        freq='D'
    )[1:]
    
    fig.add_trace(
        go.Scatter(x=future_dates, 
                  y=prophet_forecast['yhat'].tail(days),
                  name='Prophet Forecast',
                  line=dict(color='red')),
        row=1, col=1
    )
    
    # XGBoost forecast
    fig.add_trace(
        go.Scatter(x=future_dates,
                  y=xgb_forecast[-days:],
                  name='XGBoost Forecast',
                  line=dict(color='green')),
        row=1, col=1
    )
    
    # Confidence intervals for Prophet
    fig.add_trace(
        go.Scatter(x=future_dates,
                  y=prophet_forecast['yhat_upper'].tail(days),
                  name='Prophet Upper Bound',
                  line=dict(color='rgba(255,0,0,0.2)')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=future_dates,
                  y=prophet_forecast['yhat_lower'].tail(days),
                  name='Prophet Lower Bound',
                  line=dict(color='rgba(255,0,0,0.2)'),
                  fill='tonexty'),
        row=1, col=1
    )
    
    # Model comparison (differences)
    diff = prophet_forecast['yhat'].tail(days).values - xgb_forecast[-days:]
    fig.add_trace(
        go.Scatter(x=future_dates,
                  y=diff,
                  name='Forecast Difference',
                  line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark',
        title_text="Bitcoin Price Forecast Comparison"
    )
    
    return fig

# Main app
def main():
    st.title("📈 Bitcoin Price Forecasting")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_data()
        prophet_model, xgb_dict = load_models()
    
    # Simple interface
    st.sidebar.header("⚙️ Settings")
    days = st.sidebar.slider("Forecast Days", 1, 30, 7)
    
    if st.sidebar.button("🔄 Update Data"):
        with st.spinner("Fetching latest Bitcoin data..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            latest = yf.download('BTC-USD', start=start_date, end=end_date)
            st.success("Data updated!")
            st.experimental_rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Forecast")
        prophet_forecast, xgb_forecast = make_forecast(prophet_model, xgb_dict, days)
        fig = plot_forecast(df, prophet_forecast, days)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Forecast Summary")
        
        # Current price
        current_price = df['Close'].iloc[-1]
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{df['Returns'].iloc[-1]*100:.2f}%"
        )
        
        # Next day forecast
        tomorrow_forecast = prophet_forecast['yhat'].iloc[-days]
        forecast_change = (tomorrow_forecast - current_price) / current_price * 100
        st.metric(
            f"{days}-Day Forecast",
            f"${tomorrow_forecast:,.2f}",
            f"{forecast_change:.2f}%"
        )
        
        # Confidence interval
        st.write("Forecast Range:")
        st.write(f"Upper: ${prophet_forecast['yhat_upper'].iloc[-days]:,.2f}")
        st.write(f"Lower: ${prophet_forecast['yhat_lower'].iloc[-days]:,.2f}")
        
        # Recent trends
        st.subheader("Recent Trends")
        st.write("7-Day Average:", f"${df['MA7'].iloc[-1]:,.2f}")
        st.write("RSI (14):", f"{df['RSI'].iloc[-1]:.2f}")
        st.write("Volatility:", f"{df['Volatility'].iloc[-1]*100:.2f}%")

if __name__ == "__main__":
    main()
