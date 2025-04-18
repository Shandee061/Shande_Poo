import pandas as pd
import numpy as np
import logging
import time
import socket
import json
from datetime import datetime, timedelta
import threading
import random

logger = logging.getLogger(__name__)

class ProfitProDataCollector:
    """
    Data collector for Profit Pro trading platform.
    Responsible for fetching market data, both historical and real-time.
    """
    
    def __init__(self, host='localhost', port=8080, user='', password=''):
        """
        Initialize the data collector.
        
        Args:
            host (str): Profit Pro API host
            port (int): Profit Pro API port
            user (str): Username for authentication
            password (str): Password for authentication
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connected = False
        self.socket = None
        self.lock = threading.Lock()
        
        # Cache for data
        self.price_cache = {}
        self.book_cache = {}
        
        # Connection attempt
        self.connect()
    
    def connect(self):
        """
        Connect to Profit Pro API.
        
        In a real implementation, this would establish a connection to the Profit Pro API.
        For this demo, we'll simulate the connection.
        """
        try:
            logger.info(f"Connecting to Profit Pro at {self.host}:{self.port}")
            
            # In a real implementation, we would create a socket connection here
            # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.socket.connect((self.host, self.port))
            # self.socket.settimeout(10)
            
            # Simulate connection
            time.sleep(0.5)
            self.connected = True
            
            logger.info("Connected to Profit Pro successfully")
            
            # Start a background thread for real-time data
            threading.Thread(target=self._realtime_data_thread, daemon=True).start()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Profit Pro: {str(e)}")
            self.connected = False
            return False
    
    def update_connection(self, host, port, user, password):
        """
        Update connection parameters and reconnect.
        
        Args:
            host (str): New host
            port (int): New port
            user (str): New username
            password (str): New password
        """
        # Close existing connection if any
        self.disconnect()
        
        # Update parameters
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        
        # Reconnect
        return self.connect()
    
    def disconnect(self):
        """Disconnect from Profit Pro API."""
        if self.connected and self.socket:
            try:
                # self.socket.close()
                pass
            except:
                pass
        
        self.connected = False
        self.socket = None
        logger.info("Disconnected from Profit Pro")
    
    def is_connected(self):
        """Check if connected to Profit Pro API."""
        return self.connected
    
    def get_latest_data(self, symbol, timeframe):
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol (str): The trading symbol (e.g., "WINFUT")
            timeframe (str): The timeframe (e.g., "1min", "5min", "daily")
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and indicators
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Profit Pro, attempting to reconnect...")
                if not self.connect():
                    logger.error("Failed to reconnect to Profit Pro")
                    return None
            
            # In a real implementation, we would send a request to the Profit Pro API
            # and parse the response
            
            # For this demo, we'll generate simulated data
            with self.lock:
                if symbol in self.price_cache:
                    # Return cached data
                    return self.price_cache[symbol].copy()
                else:
                    # Generate new data
                    data = self._generate_sample_data(symbol, timeframe, 100)
                    self.price_cache[symbol] = data
                    return data.copy()
        
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date):
        """
        Get historical market data for a symbol.
        
        Args:
            symbol (str): The trading symbol (e.g., "WINFUT")
            timeframe (str): The timeframe (e.g., "1min", "5min", "daily")
            start_date (datetime or str): Start date for historical data
            end_date (datetime or str): End date for historical data
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and indicators
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Profit Pro, attempting to reconnect...")
                if not self.connect():
                    logger.error("Failed to reconnect to Profit Pro")
                    return None
            
            # Convert dates to datetime if they are strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # In a real implementation, we would send a request to the Profit Pro API
            # and parse the response
            
            # For this demo, we'll generate simulated data
            days_diff = (end_date - start_date).days + 1
            
            if timeframe == "daily":
                num_bars = days_diff
            elif timeframe == "60min":
                num_bars = days_diff * 8  # Assuming 8 trading hours per day
            elif timeframe == "30min":
                num_bars = days_diff * 16
            elif timeframe == "15min":
                num_bars = days_diff * 32
            elif timeframe == "5min":
                num_bars = days_diff * 96
            else:  # 1min
                num_bars = days_diff * 480
            
            data = self._generate_sample_data(symbol, timeframe, num_bars, start_date)
            
            return data
        
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def get_order_book(self, symbol):
        """
        Get the current order book for a symbol.
        
        Args:
            symbol (str): The trading symbol (e.g., "WINFUT")
            
        Returns:
            dict: Order book data with bids and asks
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Profit Pro, attempting to reconnect...")
                if not self.connect():
                    logger.error("Failed to reconnect to Profit Pro")
                    return None
            
            # In a real implementation, we would send a request to the Profit Pro API
            # and parse the response
            
            # For this demo, we'll generate simulated order book data
            with self.lock:
                if symbol in self.book_cache:
                    # Return cached data
                    return self.book_cache[symbol].copy()
                else:
                    # Generate new data
                    if symbol not in self.price_cache:
                        # Generate price data first
                        self.get_latest_data(symbol, "1min")
                    
                    if symbol in self.price_cache:
                        last_price = self.price_cache[symbol].iloc[-1]['close']
                        book = self._generate_sample_order_book(symbol, last_price)
                        self.book_cache[symbol] = book
                        return book.copy()
                    else:
                        return None
        
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            return None
    
    def get_times_and_trades(self, symbol, limit=100):
        """
        Get recent trades for a symbol (Times & Trades).
        
        Args:
            symbol (str): The trading symbol (e.g., "WINFUT")
            limit (int): Maximum number of trades to return
            
        Returns:
            pd.DataFrame: DataFrame with trade data
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Profit Pro, attempting to reconnect...")
                if not self.connect():
                    logger.error("Failed to reconnect to Profit Pro")
                    return None
            
            # In a real implementation, we would send a request to the Profit Pro API
            # and parse the response
            
            # For this demo, we'll generate simulated times & trades data
            if symbol not in self.price_cache:
                # Generate price data first
                self.get_latest_data(symbol, "1min")
            
            if symbol in self.price_cache:
                last_price = self.price_cache[symbol].iloc[-1]['close']
                
                # Generate trades
                now = datetime.now()
                times = [now - timedelta(seconds=i) for i in range(limit)]
                prices = [last_price * (1 + np.random.normal(0, 0.0005)) for _ in range(limit)]
                volumes = [np.random.randint(1, 10) for _ in range(limit)]
                sides = [np.random.choice(['BUY', 'SELL']) for _ in range(limit)]
                
                trades = pd.DataFrame({
                    'timestamp': times,
                    'price': prices,
                    'volume': volumes,
                    'side': sides
                })
                
                return trades.sort_values('timestamp', ascending=False)
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error getting times and trades: {str(e)}")
            return None
    
    def _realtime_data_thread(self):
        """Background thread for simulating real-time data updates."""
        while self.connected:
            try:
                # Update cached data with new ticks
                with self.lock:
                    for symbol in list(self.price_cache.keys()):
                        df = self.price_cache[symbol]
                        
                        # Get last row
                        last_row = df.iloc[-1]
                        
                        # Create new row with updated values
                        new_time = last_row.name + pd.Timedelta(minutes=1)
                        last_close = last_row['close']
                        
                        # Random price movement (more volatile for demonstration)
                        price_change = np.random.normal(0, last_close * 0.002)
                        new_close = max(0.01, last_close + price_change)
                        
                        new_high = max(new_close * (1 + abs(np.random.normal(0, 0.001))), new_close)
                        new_low = min(new_close * (1 - abs(np.random.normal(0, 0.001))), new_close)
                        new_open = last_close
                        
                        # Random volume
                        new_volume = int(last_row['volume'] * (0.8 + 0.4 * np.random.random()))
                        
                        # Create new row
                        new_row = pd.DataFrame({
                            'open': [new_open],
                            'high': [new_high],
                            'low': [new_low],
                            'close': [new_close],
                            'volume': [new_volume]
                        }, index=[new_time])
                        
                        # Append to dataframe
                        df = pd.concat([df, new_row])
                        
                        # Recalculate indicators
                        df = self._calculate_indicators(df)
                        
                        # Keep only the last 300 rows to avoid memory issues
                        if len(df) > 300:
                            df = df.iloc[-300:]
                        
                        # Update cache
                        self.price_cache[symbol] = df
                        
                        # Update order book
                        if symbol in self.book_cache:
                            self.book_cache[symbol] = self._generate_sample_order_book(symbol, new_close)
            
            except Exception as e:
                logger.error(f"Error in real-time data thread: {str(e)}")
            
            # Update every 5 seconds
            time.sleep(5)
    
    def _generate_sample_data(self, symbol, timeframe, num_bars, start_date=None):
        """
        Generate sample OHLCV data for testing.
        
        Args:
            symbol (str): The trading symbol
            timeframe (str): The timeframe
            num_bars (int): Number of bars to generate
            start_date (datetime, optional): Start date for the data
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and indicators
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=num_bars // 96 + 1)  # Assuming 5min bars
        
        # Create date range based on timeframe
        if timeframe == "1min":
            dates = pd.date_range(start=start_date, periods=num_bars, freq='1min')
        elif timeframe == "5min":
            dates = pd.date_range(start=start_date, periods=num_bars, freq='5min')
        elif timeframe == "15min":
            dates = pd.date_range(start=start_date, periods=num_bars, freq='15min')
        elif timeframe == "30min":
            dates = pd.date_range(start=start_date, periods=num_bars, freq='30min')
        elif timeframe == "60min":
            dates = pd.date_range(start=start_date, periods=num_bars, freq='60min')
        else:  # daily
            dates = pd.date_range(start=start_date, periods=num_bars, freq='D')
        
        # Filter for trading hours (9:00 to 17:00) on weekdays
        trading_dates = []
        for date in dates:
            if date.weekday() < 5 and 9 <= date.hour < 17:  # Monday to Friday, 9 AM to 5 PM
                trading_dates.append(date)
        
        # Generate prices using random walk with drift
        np.random.seed(42)  # For reproducibility
        
        # Starting price depends on the symbol
        if symbol == "WINFUT":
            price = 120000.0  # Approximate value for WINFUT
        else:
            price = 100.0
        
        prices = [price]
        
        # Random walk with drift
        for _ in range(1, len(trading_dates)):
            # Random price change with drift (slight upward bias)
            price_change = np.random.normal(0.0001, 0.002) * prices[-1]
            new_price = max(0.01, prices[-1] + price_change)
            prices.append(new_price)
        
        # Generate OHLC data
        opens = prices.copy()
        closes = []
        highs = []
        lows = []
        volumes = []
        
        for i in range(len(trading_dates)):
            # Close is the open of the next bar, with some random variation
            if i < len(prices) - 1:
                close = prices[i+1] * (1 + np.random.normal(0, 0.0005))
            else:
                close = prices[i] * (1 + np.random.normal(0, 0.001))
            
            # High and low are derived from open and close
            high = max(opens[i], close) * (1 + abs(np.random.normal(0, 0.001)))
            low = min(opens[i], close) * (1 - abs(np.random.normal(0, 0.001)))
            
            # Volume is random
            volume = int(np.random.normal(1000, 300))
            
            closes.append(close)
            highs.append(high)
            lows.append(low)
            volumes.append(max(1, volume))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=trading_dates)
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        return df
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for the data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Simple Moving Averages
        df['sma_9'] = df['close'].rolling(window=9).mean()
        df['sma_21'] = df['close'].rolling(window=21).mean()
        
        # Exponential Moving Averages
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        return df
    
    def _generate_sample_order_book(self, symbol, last_price):
        """
        Generate a sample order book for testing.
        
        Args:
            symbol (str): The trading symbol
            last_price (float): The last traded price
            
        Returns:
            dict: Order book with bids and asks
        """
        # Generate 10 levels of bids and asks
        bids = []
        asks = []
        
        for i in range(1, 11):
            # Bids decrease in price from the last price
            bid_price = last_price * (1 - i * 0.0005)
            bid_volume = np.random.randint(1, 20)
            bids.append({'price': bid_price, 'volume': bid_volume})
            
            # Asks increase in price from the last price
            ask_price = last_price * (1 + i * 0.0005)
            ask_volume = np.random.randint(1, 20)
            asks.append({'price': ask_price, 'volume': ask_volume})
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'bids': bids,
            'asks': asks
        }
