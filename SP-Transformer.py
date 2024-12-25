import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import nest_asyncio
import psutil
import random
import time
from ib_insync import IB, Stock

# Apply nest_asyncio to handle event loops in Spyder/Jupyter environments
nest_asyncio.apply()

# Helper function to close any old TWS/IB Gateway processes
def close_old_ibkr_processes():
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            # Check if the process is related to TWS or IB Gateway
            if any(keyword in proc.info['cmdline'] for keyword in ['java', 'ibgateway']):
                print(f"Terminating old IBKR process: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()  # or proc.kill() to forcefully terminate
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# Ensure old IBKR processes are closed
close_old_ibkr_processes()

# Function to establish a connection with retry and unique clientId
def connect_ibkr(max_retries=5):
    # Disconnect any existing connection
    global ib
    if 'ib' in globals() and ib.isConnected():
        print("Disconnecting existing IBKR connection.")
        ib.disconnect()
    
    # Initialize new connection
    ib = IB()
    for attempt in range(max_retries):
        try:
            client_id = random.randint(1, 9999)  # Randomize client ID
            print(f"Attempting to connect with clientId={client_id}")
            ib.connect('127.0.0.1', 7497, clientId=client_id)
            print("Connected to IBKR")
            return ib
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                print("Max retries reached. Unable to connect to IBKR.")
                return None

# Initialize the IBKR connection with retries
ib = connect_ibkr()

# Check if the connection was successful
if ib is None:
    print("Failed to connect to IBKR. Please ensure TWS or IB Gateway is running and API access is enabled.")
else:
    # Proceed with the rest of the script
    print("IBKR connection established. Proceeding with data fetching and processing.")

    # Fetch leveraged ETF data from IBKR
    def fetch_ibkr_data(symbol, endDateTime, durationStr='10 Y', barSizeSetting='1 day'):
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(contract, endDateTime=endDateTime, durationStr=durationStr, 
                                    barSizeSetting=barSizeSetting, whatToShow='ADJUSTED_LAST', useRTH=True)
        df = pd.DataFrame(bars)
        print(f"Columns in IBKR data for {symbol}: {df.columns}")
        if 'close' in df.columns:
            df.set_index("date", inplace=True)
            df.rename(columns={"close": symbol}, inplace=True)
            return df[[symbol]]
        else:
            raise KeyError(f"'close' column not found in data for {symbol}. Columns available: {df.columns}")

    # Fetch SPXL and SPXS data
    try:
        spxl_df = fetch_ibkr_data('SPXL', endDateTime='', durationStr='10 Y')
        spxs_df = fetch_ibkr_data('SPXS', endDateTime='', durationStr='10 Y')
    except KeyError as e:
        print(e)
    
    # Load data for model training from Yahoo Finance
    tickers = {
        'Crude_Oil': 'CL=F',       # Crude Oil Futures
        'Gold': 'GC=F',             # Gold Futures
        'USD_Index': 'DX-Y.NYB',    # U.S. Dollar Index
        'SPY': 'SPY',               # S&P 500 ETF
        'TNX_Yield': '^TNX',        # 10-Year Treasury Yield
        'VIX': '^VIX'               # Volatility Index
    }
    start_date = "2000-01-01"
    end_date = "2023-12-31"
    data_frames = {name: yf.download(ticker, start=start_date, end=end_date)[['Adj Close']].rename(columns={'Adj Close': name}) for name, ticker in tickers.items()}
    df = pd.concat(data_frames.values(), axis=1, join='outer')

    # Resample to ensure daily frequency and fill missing data
    df = df.resample('D').ffill()

    # Integrate SPXL and SPXS data into the main DataFrame
    df = pd.concat([df, spxl_df, spxs_df], axis=1)
    df.rename(columns={'SPXL': 'SPXL', 'SPXS': 'SPXS'}, inplace=True)

    # Check for any remaining NaN values
    df.fillna(method='bfill', inplace=True)

    # Ensure the DataFrame index is timezone-naive
    df.index = df.index.tz_localize(None)

    # Compute the second derivative for SPY
    df['SPY_First_Derivative'] = df['SPY'].diff()
    df['SPY_Second_Derivative'] = df['SPY_First_Derivative'].diff()
    df.dropna(inplace=True)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    # Dataset and Model Definition (as in the original code)
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, sequence_length, prediction_length, target_col_index):
            self.data = data
            self.sequence_length = sequence_length
            self.prediction_length = prediction_length
            self.target_col_index = target_col_index
    
        def __len__(self):
            return len(self.data) - self.sequence_length - self.prediction_length + 1
    
        def __getitem__(self, idx):
            x = self.data[idx:idx+self.sequence_length, :]
            y = self.data[idx+self.sequence_length:idx+self.sequence_length+self.prediction_length, self.target_col_index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    class TransformerTimeSeriesModel(nn.Module):
        def __init__(self, feature_size, num_layers=2, nhead=2, hidden_dim=64):
            super(TransformerTimeSeriesModel, self).__init__()
            self.embedding = nn.Linear(feature_size, hidden_dim)
            self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
            self.transformer = nn.Transformer(
                d_model=hidden_dim,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=128,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)
    
        def forward(self, x):
            x = self.embedding(x) + self.positional_encoding
            output = self.transformer(x, x)
            output = self.fc(output[:, -1, :])  # Use the last output for prediction
            return output
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to collect results
    all_forecast_dates, all_actuals, all_predictions = [], [], []
    mae_list, rmse_list = [], []
    target_col_index = list(df.columns).index('SPY_Second_Derivative')
    
    # Training and Prediction Loop (one day ahead, walk-forward validation, as per previous script)
    
    # Implement the Trading Strategy Logic
    def backtest_leveraged_etf(df, predictions):
        initial_balance = 10000  # Starting capital
        balance = initial_balance
        position = 0  # Current position: +1 for SPXL, -1 for SPXS, 0 for cash
        daily_returns = []
    
        for i in range(1, len(predictions)):
            prediction = predictions[i]
            actual_spy = df['SPY'].iloc[i]
            
            # Determine position based on prediction
            if prediction > 0:  # Positive acceleration, go long on SPXL
                if position != 1:
                    balance += position * df['SPXL'].iloc[i-1]  # Close existing position
                    position = balance / df['SPXL'].iloc[i]
                    balance = 0
            elif prediction < 0:  # Negative acceleration, go long on SPXS
                if position != -1:
                    balance += position * df['SPXS'].iloc[i-1]
                    position = -balance / df['SPXS'].iloc[i]
                    balance = 0
            else:
                balance += position * (df['SPXL'].iloc[i] if position > 0 else df['SPXS'].iloc[i])
                position = 0
    
            # Calculate daily return
            daily_balance = balance + position * (df['SPXL'].iloc[i] if position > 0 else df['SPXS'].iloc[i])
            daily_returns.append((daily_balance - initial_balance) / initial_balance)
    
        # Final balance after closing all positions
        balance += position * (df['SPXL'].iloc[-1] if position > 0 else df['SPXS'].iloc[-1])
    
        total_return = (balance - initial_balance) / initial_balance
        print(f"Final Balance: ${balance:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        return daily_returns
    
    # Run the backtest
    daily_returns = backtest_leveraged_etf(df, all_predictions)
    
    # Plot performance
    plt.figure(figsize=(14, 7))
    plt.plot(daily_returns, label='Cumulative Strategy Returns', color='green')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Returns')
    plt.title('Leveraged ETF Strategy Backtest')
    plt.legend()
    plt.show()
