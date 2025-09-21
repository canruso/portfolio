from __future__ import annotations
from typing import Optional
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# TODO: add payload validation for the methods inside the CryptoAccessor class
class CryptoAccessor:
    
    def __init__(self):
        # Initialize Alpaca crypto client (no API keys needed for crypto data)
        self.client = CryptoHistoricalDataClient()
    
    @st.cache_data(show_spinner=False)
    def get_spot_price(_self, symbols: str) -> list[float]:
        """
        Return the latest available close price for multiple crypto symbols.
        
        Parameters
        ----------
        symbols : str
            Space-separated crypto symbols (e.g. "BTC ETH SOL")

        Returns
        -------
        list[float]
            List of current prices, one for each symbol
        """
        try:
            # Convert symbols to Alpaca format (BTC -> BTC/USD)
            symbol_list = symbols.split()
            alpaca_symbols = [f'{s.strip()}/USD' for s in symbol_list]
            
            # Get latest bars by requesting data from the last 15 minutes
            request_params = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbols,
                timeframe=TimeFrame.Minute,
                start=datetime.utcnow() - timedelta(minutes=15)
            )
            
            bars = _self.client.get_crypto_bars(request_params)
            
            ret_val = []
            for symbol in alpaca_symbols:
                try:
                    if symbol in bars.data:
                        # Get the latest close price from the returned bars
                        symbol_bars = bars.data[symbol]
                        if symbol_bars:
                            latest_price = float(symbol_bars[-1].close)
                            ret_val.append(latest_price)
                        else:
                            # This can happen if there was no trade in the last 15 mins
                            print(f"No recent bar data for {symbol}")
                            ret_val.append(0.0)
                    else:
                        print(f"Symbol {symbol} not found in response")
                        ret_val.append(0.0)
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
                    ret_val.append(0.0)
            
            return ret_val
            
        except Exception as e:
            print(f'Error getting spot price for crypto: {symbols}, error message: {str(e)}')
            return []

    @st.cache_data(show_spinner=False)
    def get_crypto_momentum(_self, symbols: list[str]) -> pd.DataFrame:
        """
        Fetches historical data to calculate various momentum metrics.
        """
        if not symbols:
            return pd.DataFrame()
            
        try:
            alpaca_symbols = [f'{s}/USD' for s in symbols]
            
            # Fetch daily data for the last 60 days to cover all periods
            request_params = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbols,
                timeframe=TimeFrame.Day,
                start=datetime.utcnow() - timedelta(days=60) # A little buffer
            )
            
            bars = _self.client.get_crypto_bars(request_params).df
            if bars.empty:
                return pd.DataFrame()

            # Calculate momentum for each symbol from the multi-index DataFrame
            all_momentum = []
            unique_symbols = bars.index.get_level_values('symbol').unique()

            for symbol in unique_symbols:
                symbol_df = bars.loc[symbol].copy()
                if symbol_df.empty:
                    continue

                symbol_df['1D_change'] = symbol_df['close'].pct_change(periods=1)
                symbol_df['12D_change'] = symbol_df['close'].pct_change(periods=12)
                symbol_df['26D_change'] = symbol_df['close'].pct_change(periods=26)
                symbol_df['52D_change'] = symbol_df['close'].pct_change(periods=52)
                
                # Get the latest momentum values
                latest_momentum_series = symbol_df.iloc[-1]
                
                momentum_data = {
                    'symbol': symbol.replace('/USD', ''),
                    '1D_change': latest_momentum_series['1D_change'],
                    '12D_change': latest_momentum_series['12D_change'],
                    '26D_change': latest_momentum_series['26D_change'],
                    '52D_change': latest_momentum_series['52D_change'],
                }
                all_momentum.append(momentum_data)

            if not all_momentum:
                return pd.DataFrame()

            return pd.DataFrame(all_momentum)

        except Exception as e:
            print(f'Error getting momentum for crypto: {symbols}, error message: {str(e)}')
            return pd.DataFrame()

    @st.cache_data(show_spinner=False)
    def get_history(_self, symbols: list[str], start: str = "2024-01-01", end: str = "2024-09-21") -> pd.DataFrame:
        """
        Download historical OHLCV data for one or more crypto symbols within a date range.
        
        Parameters
        ----------
        symbols : list[str]
            One or more ticker symbols (e.g., ["BTC", "ETH"]).
        start : str
            Inclusive start date (YYYY-MM-DD).
        end : str
            Exclusive end date (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            A DataFrame with OHLCV data. May be empty if no data is available.
        """
        try:
            # Convert to Alpaca format
            alpaca_symbols = [f'{s}/USD' for s in symbols]
            
            # Create request for daily bars
            request_params = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbols,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start, '%Y-%m-%d'),
                end=datetime.strptime(end, '%Y-%m-%d')
            )
            
            bars = _self.client.get_crypto_bars(request_params)
            
            if bars.df is None or bars.df.empty:
                return pd.DataFrame()
                
            return bars.df
            
        except Exception as e:
            print(f'Error getting history for crypto: {symbols}, error message: {str(e)}')
            return pd.DataFrame()