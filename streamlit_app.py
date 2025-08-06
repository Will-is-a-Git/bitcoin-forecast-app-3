import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet

# Page config
st.set_page_config(
    page_title="Bitcoin Price Forecast",
    page_icon="📈",
    layout="wide"
)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_bitcoin_data(days=365):
    """Fetch Bitcoin historical data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    return btc

@st.cache_data
def preprocess_data(df):
    """Preprocess the Bitcoin data"""
    df = df.copy()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    
    # Technical indicators
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=30).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

@st.cache_resource
def train_prophet_model(df):
    """Train a Prophet model"""
    # Prepare data for Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df.index
    prophet_df['y'] = df['Close']
    
    # Create and train model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model.fit(prophet_df)
    
    return model

def plot_forecast(df, forecast, days):
    """Create forecast plot"""
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
            y=forecast['yhat'].tail(days),
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast['yhat_upper'].tail(days),
            fill=None,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Upper Bound'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast['yhat_lower'].tail(days),
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
    
    return fig

def main():
    st.title("📈 Bitcoin Price Forecasting")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    days = st.sidebar.slider("Forecast Days", 1, 30, 7)
    history_days = st.sidebar.selectbox(
        "Historical Data Period",
        options=[30, 90, 180, 365],
        index=2
    )
    
    # Fetch and process data
    with st.spinner("Fetching Bitcoin data..."):
        df = fetch_bitcoin_data(days=history_days)
        df = preprocess_data(df)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Forecast")
        
        # Train model and make forecast
        with st.spinner("Generating forecast..."):
            model = train_prophet_model(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Create and display plot
            fig = plot_forecast(df, forecast, days)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Summary")
        
        # Current price
        current_price = df['Close'].iloc[-1]
        daily_return = df['Returns'].iloc[-1] * 100
        
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{daily_return:.2f}%"
        )
        
        # Forecast
        forecast_price = forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_price - current_price) / current_price) * 100
        
        st.metric(
            f"{days}-Day Forecast",
            f"${forecast_price:,.2f}",
            f"{forecast_change:.2f}%"
        )
        
        # Technical Indicators
        st.subheader("Technical Indicators")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric(
                "RSI (14)",
                f"{df['RSI'].iloc[-1]:.1f}",
                None
            )
            
            st.metric(
                "7-Day MA",
                f"${df['MA7'].iloc[-1]:,.2f}",
                None
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{df['Volatility'].iloc[-1]*100:.1f}%",
                None
            )
            
            st.metric(
                "30-Day MA",
                f"${df['MA30'].iloc[-1]:,.2f}",
                None
            )
        
        # Forecast Range
        st.subheader("Forecast Range")
        st.write(f"Upper: ${forecast['yhat_upper'].iloc[-1]:,.2f}")
        st.write(f"Lower: ${forecast['yhat_lower'].iloc[-1]:,.2f}")
        
        # Download data
        if st.button("📥 Download Historical Data"):
            csv = df.to_csv()
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="bitcoin_historical_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
