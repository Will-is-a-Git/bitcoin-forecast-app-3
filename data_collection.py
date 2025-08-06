import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

def fetch_bitcoin_data(start_date='2015-01-01', end_date=None):
    """
    Fetch Bitcoin historical data from Yahoo Finance
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching Bitcoin data from {start_date} to {end_date}...")
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    return btc

def preprocess_data(df):
    """
    Clean and preprocess the Bitcoin data with advanced features
    """
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill
    df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
    
    # Add basic features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Add date features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    
    # Add technical indicators
    # Moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA90'] = df['Close'].rolling(window=90).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=30).std()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Price momentum
    df['Price_Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Volume features
    if 'Volume' in df.columns:
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_MA30'] = df['Volume'].rolling(window=30).mean()
        df['Volume_Price_Ratio'] = df['Volume'] / df['Close']
    
    return df

def visualize_price_history(df):
    """
    Create an interactive plot of Bitcoin price history
    """
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    fig.update_layout(
        title='Bitcoin Price History',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

if __name__ == "__main__":
    # Fetch data
    btc_data = fetch_bitcoin_data()
    
    # Preprocess data
    btc_processed = preprocess_data(btc_data)
    
    # Save data
    btc_processed.to_csv('data/btc_data.csv')
    print(f"Data saved to data/btc_data.csv")
    
    # Create visualization
    fig = visualize_price_history(btc_processed)
    fig.write_html('data/price_history.html')
    print(f"Interactive plot saved to data/price_history.html")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(btc_processed['Close'].describe())
